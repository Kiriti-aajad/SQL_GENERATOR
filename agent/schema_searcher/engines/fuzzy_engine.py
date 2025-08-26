"""
Fuzzy string matching engine for typo-tolerant schema retrieval.
FIXED: All HTML entity encoding, validation issues, and actual fuzzy matching implementation
"""

from fuzzywuzzy import fuzz, process
from typing import List, Dict, Any, Optional, Tuple, Set
import re
import logging
from datetime import datetime

from agent.schema_searcher.engines.base_engine import BaseSearchEngine
from agent.schema_searcher.core.data_models import RetrievedColumn, ColumnType, SearchMethod
from agent.schema_searcher.loaders.schema_loader import SchemaLoader
from agent.schema_searcher.loaders.xml_loader import XMLLoader
from agent.schema_searcher.utils.performance import track_execution_time

logger = logging.getLogger(__name__)

# FIXED: Simplified exceptions
class FuzzyEngineError(Exception):
    """Base exception for fuzzy engine errors"""
    pass

class FuzzyInputValidationError(FuzzyEngineError):
    """Raised when input validation fails"""
    pass

class FuzzyInitializationError(FuzzyEngineError):
    """Raised when fuzzy engine initialization fails"""
    pass

class FuzzySearchError(FuzzyEngineError):
    """Raised when fuzzy search fails"""
    pass

class FuzzySearchEngine(BaseSearchEngine):
    """
    Fuzzy string matching engine for typo-tolerant schema retrieval.
    FIXED: Now actually performs fuzzy matching with proper scoring
    """

    def __init__(self, threshold: int = 60, max_results: int = 200):
        super().__init__(SearchMethod.FUZZY, logger)
        
        # FIXED: Better parameter validation
        if not isinstance(threshold, int) or threshold < 20 or threshold > 95:
            raise FuzzyEngineError(f"Invalid threshold {threshold}, must be between 20 and 95")
        if not isinstance(max_results, int) or max_results <= 0:
            raise FuzzyEngineError(f"Invalid max_results {max_results}, must be positive")
        
        self.threshold = threshold
        self.max_results = max_results
        self.schema_data: List[Dict[str, Any]] = []
        self.xml_data: List[Dict[str, Any]] = []
        self.search_candidates: List[str] = []
        self.candidate_mapping: Dict[str, Dict[str, Any]] = {}
        self.enhanced_candidates: Dict[str, List[str]] = {}
        self.pattern_mappings: Dict[str, List[str]] = {}
        
        # Reasoning enhancement parameters
        self._fuzzy_threshold = threshold
        self._reasoning_mode_enabled = False
        self._entity_type_weights: Dict[str, float] = {}
        self._adaptive_matching = False
        
        # Statistics
        self.total_searches = 0
        self.successful_searches = 0
        self.failed_searches = 0

    def initialize(self) -> None:
        """Initialize fuzzy search engine with proper validation"""
        self.logger.info("Initializing fuzzy search engine")
        try:
            schema_loader = SchemaLoader()
            xml_loader = XMLLoader()
            self.schema_data = schema_loader.load()
            raw_xml_data = xml_loader.load()
            
            # Better validation - not fail-fast
            if not self.schema_data:
                self.logger.warning("No schema data loaded")
                self.schema_data = []
            
            self.xml_data = self._flatten_xml_data(raw_xml_data)
            self._build_candidates()
            self._build_pattern_mappings()
            
            # Allow empty candidates - system can still work
            if not self.search_candidates:
                self.logger.warning("No search candidates built - fuzzy search may be limited")
            
            self.logger.info(f"Fuzzy search initialized with {len(self.search_candidates)} candidates")
        except Exception as e:
            self.logger.error(f"Failed to initialize fuzzy engine: {e}")
            raise FuzzyInitializationError(f"Fuzzy engine initialization failed: {e}")

    def _flatten_xml_data(self, raw_xml_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten XML data with validation"""
        flattened = []
        if not raw_xml_data:
            self.logger.debug("No XML data provided for flattening")
            return flattened
            
        for xml_table in raw_xml_data:
            try:
                table_name = xml_table.get('table', '')
                xml_column = xml_table.get('xml_column', '')
                fields = xml_table.get('fields', [])
                
                if not table_name:
                    continue
                
                for field in fields:
                    field_name = field.get('name', '')
                    if not field_name:
                        continue
                        
                    xpath = field.get('xpath', '')
                    description_parts = [
                        f"XML field {field_name}",
                        f"in table {table_name}",
                        f"column {xml_column}",
                        f"xpath {xpath}" if xpath else "",
                        "xml structured data field"
                    ]
                    flattened.append({
                        'table': table_name,
                        'column': field_name,
                        'xml_column': xml_column,
                        'xpath': xpath,
                        'sql_expression': field.get('sql_expression', ''),
                        'type': 'xml',
                        'datatype': 'xml_field',
                        'description': " | ".join(filter(None, description_parts)),
                        'searchable_terms': self._extract_xml_searchable_terms(field_name, xpath)
                    })
            except Exception as e:
                self.logger.debug(f"Error processing XML table data: {e}")
                continue
                
        return flattened

    def _extract_xml_searchable_terms(self, field_name: str, xpath: str) -> List[str]:
        """Extract searchable terms from XML field name and xpath"""
        terms = []
        try:
            if field_name:
                terms.append(field_name)
                camel_parts = re.findall(r'[A-Z][a-z]*|[a-z]+', field_name)
                terms.extend(camel_parts)
            if xpath:
                xpath_elements = re.findall(r'/([^/\[\]]+)', xpath)
                terms.extend(xpath_elements)
                for element in xpath_elements:
                    camel_parts = re.findall(r'[A-Z][a-z]*|[a-z]+', element)
                    terms.extend(camel_parts)
            return [term.lower() for term in terms if term and len(term) > 1]
        except Exception as e:
            self.logger.debug(f"Error extracting searchable terms: {e}")
            return []

    def _build_pattern_mappings(self) -> None:
        """Build pattern mappings for term expansion"""
        self.pattern_mappings = {
            'id': ['identifier', 'identity', 'key'],
            'nm': ['name'],
            'cd': ['code'],
            'dt': ['date', 'datetime'],
            'qty': ['quantity'],
            'amt': ['amount'],
            'addr': ['address'],
            'desc': ['description'],
            'ref': ['reference'],
            'num': ['number'],
            'cnt': ['count'],
            'val': ['value'],
            'stat': ['status'],
            'ind': ['indicator'],
            'flg': ['flag'],
            'tbl': ['table'],
            'vw': ['view'],
            'sp': ['stored procedure', 'procedure'],
            'fn': ['function'],
            'cust': ['customer', 'client'],
            'acct': ['account'],
            'ord': ['order'],
            'prod': ['product'],
            'cat': ['category'],
            'usr': ['user'],
            'grp': ['group'],
            'org': ['organization'],
            'dept': ['department'],
            'mgr': ['manager'],
            'emp': ['employee'],
            'sup': ['supplier'],
            'inv': ['inventory', 'invoice'],
            'pay': ['payment'],
            'trans': ['transaction'],
            'bal': ['balance'],
            'curr': ['current', 'currency'],
            'ctpt': ['counterparty', 'customer', 'client'],
            'app': ['application'],
            'wf': ['workflow'],
            'xml': ['xmlout', 'xml_out']
        }

    def _build_candidates(self) -> None:
        """Build search candidates with validation"""
        self.search_candidates = []
        self.candidate_mapping = {}
        self.enhanced_candidates = {}
        
        all_data = self.schema_data + self.xml_data
        for item in all_data:
            try:
                table = item.get('table', '').strip()
                column = item.get('column', '') or item.get('name', '')
                column = column.strip()
                
                if not table or not column:
                    continue
                    
                # Create searchable candidates
                candidates = [
                    f"{table}.{column}",
                    f"{table} {column}",
                    table,
                    column,
                    f"{table}_{column}",
                    column.replace('_', ' '),
                    table.replace('_', ' ')
                ]
                
                item_key = f"{table}.{column}"
                self.enhanced_candidates[item_key] = candidates
                
                for candidate in candidates:
                    if candidate and candidate.strip():
                        candidate_clean = candidate.strip()
                        if candidate_clean not in self.candidate_mapping:
                            self.search_candidates.append(candidate_clean)
                            self.candidate_mapping[candidate_clean] = item
                        
            except Exception as e:
                self.logger.debug(f"Error building candidates for item: {e}")
                continue

    @track_execution_time
    def search(self, query: str, top_k: int = 20) -> List[RetrievedColumn]:
        """
        FIXED: Actual fuzzy search with proper scoring and ranking
        """
        self.total_searches += 1
        
        # FIXED: Sensible input validation
        if not query or not isinstance(query, str):
            self.failed_searches += 1
            raise FuzzyInputValidationError("Query must be a non-empty string")
        
        processed_query = query.strip()
        if not processed_query:
            self.failed_searches += 1
            raise FuzzyInputValidationError("Query cannot be empty or only whitespace")
        
        if top_k <= 0:
            self.failed_searches += 1
            raise FuzzyInputValidationError("top_k must be positive")
        
        # Allow search even with no candidates (graceful degradation)
        if not self.search_candidates:
            self.logger.warning("No search candidates available - returning empty results")
            self.successful_searches += 1
            return []

        try:
            # FIXED: Actually perform fuzzy matching using fuzzywuzzy
            query_lower = processed_query.lower()
            
            # Get fuzzy matches with scores
            fuzzy_matches = process.extract(
                query_lower,
                self.search_candidates,
                scorer=fuzz.partial_ratio,  # Good for partial matches
                limit=min(100, len(self.search_candidates))  # Get more candidates for processing
            )
            
            # Filter by threshold and sort by score
            filtered_matches = [
                (candidate, score) for candidate, score in fuzzy_matches # pyright: ignore[reportAssignmentType]
                if score >= self.threshold
            ]
            
            # Sort by score descending
            filtered_matches.sort(key=lambda x: x[1], reverse=True)
            
            # Create results with proper confidence scoring
            results = []
            seen_columns = set()
            
            for candidate, fuzzy_score in filtered_matches:
                if candidate not in self.candidate_mapping:
                    continue
                    
                try:
                    item = self.candidate_mapping[candidate]
                    table = item.get('table', '').strip()
                    column = (item.get('column', '') or item.get('name', '')).strip()
                    
                    if not table or not column:
                        continue
                    
                    # Deduplication
                    column_key = f"{table.lower()}.{column.lower()}"
                    if column_key in seen_columns:
                        continue
                    seen_columns.add(column_key)
                    
                    # Calculate confidence score from fuzzy score
                    confidence_score = min(1.0, max(0.1, fuzzy_score / 100.0))
                    
                    result = self._create_retrieved_column(item, confidence_score)
                    if result:
                        results.append(result)
                        
                    if len(results) >= top_k:
                        break
                        
                except Exception as e:
                    self.logger.debug(f"Error creating result from candidate '{candidate}': {e}")
                    continue
            
            self.successful_searches += 1
            self.logger.debug(f"Fuzzy search returned {len(results)} results for query: '{query}' (threshold: {self.threshold})")
            return results
            
        except Exception as e:
            self.failed_searches += 1
            self.logger.error(f"Fuzzy search failed for query '{query}': {e}")
            raise FuzzySearchError(f"Fuzzy search failed: {e}")

    # FIXED: Add missing async method for compatibility
    async def search_async(self, query: str = None, keywords: List[str] = None, top_k: int = 20) -> List[RetrievedColumn]: # pyright: ignore[reportArgumentType]
        """Async search method for compatibility with retrieval agent"""
        # Handle both query and keywords parameters
        if query:
            search_query = query
        elif keywords:
            search_query = " ".join(keywords)
        else:
            raise FuzzyInputValidationError("Either query or keywords must be provided")
        
        # Fuzzy search is synchronous, so just call the regular search method
        return self.search(search_query, top_k)

    def search_with_strategy(self, strategy: Dict[str, Any], target_gaps: List[str]) -> List[RetrievedColumn]:
        """
        Strategy-based fuzzy search with enhanced fuzzy matching
        """
        if not target_gaps:
            raise FuzzyInputValidationError("No target gaps provided for strategy-based search")
        
        self.logger.debug(f"Executing strategy-based fuzzy search for {len(target_gaps)} target gaps")
        
        try:
            max_results = strategy.get('max_results', 20)
            if not isinstance(max_results, int) or max_results <= 0:
                max_results = 20
            
            keywords = strategy.get('keywords', target_gaps)
            focus_areas = strategy.get('focus_areas', target_gaps)
            
            # Enable reasoning mode temporarily
            old_reasoning_state = self._reasoning_mode_enabled
            self._reasoning_mode_enabled = True
            
            try:
                all_results = []
                seen_columns = set()
                
                # Search for each keyword/gap with enhanced fuzzy matching
                for search_term in keywords:
                    if not search_term or not isinstance(search_term, str):
                        continue
                    
                    # Get fuzzy matches for this specific term
                    term_matches = self._fuzzy_search_for_term(search_term, focus_areas)
                    
                    # Add unique results
                    for candidate, score in term_matches:
                        if candidate not in self.candidate_mapping:
                            continue
                        
                        item = self.candidate_mapping[candidate]
                        table = item.get('table', '').strip()
                        column = (item.get('column', '') or item.get('name', '')).strip()
                        
                        if not table or not column:
                            continue
                        
                        column_key = f"{table.lower()}.{column.lower()}"
                        if column_key in seen_columns:
                            continue
                        
                        seen_columns.add(column_key)
                        result = self._create_retrieved_column(item, score)
                        if result:
                            all_results.append(result)
                            
                            if len(all_results) >= max_results:
                                break
                    
                    if len(all_results) >= max_results:
                        break
                
                # Fallback if insufficient results
                if len(all_results) < max_results // 4:
                    self.logger.debug(f"Strategy search returned only {len(all_results)} results, trying fallback")
                    fallback_query = " ".join(str(kw) for kw in keywords[:3] if kw)
                    if fallback_query:
                        try:
                            fallback_results = self.search(fallback_query, max_results)
                            
                            # Add unique fallback results
                            for result in fallback_results:
                                column_key = f"{result.table.lower()}.{result.column.lower()}"
                                if column_key not in seen_columns:
                                    seen_columns.add(column_key)
                                    all_results.append(result)
                                    
                                    if len(all_results) >= max_results:
                                        break
                        except Exception as e:
                            self.logger.debug(f"Fallback search failed: {e}")
                
                final_results = all_results[:max_results]
                self.logger.debug(f"Strategy-based fuzzy search returned {len(final_results)} results")
                return final_results
                
            finally:
                # Restore previous reasoning state
                self._reasoning_mode_enabled = old_reasoning_state
                
        except Exception as e:
            self.logger.error(f"Strategy-based fuzzy search failed: {e}")
            raise FuzzySearchError(f"Strategy-based fuzzy search failed: {e}")

    def _fuzzy_search_for_term(self, search_term: str, focus_areas: List[str]) -> List[Tuple[str, float]]:
        """Perform fuzzy matching for a specific term with dynamic threshold"""
        if not search_term or not search_term.strip():
            return []
            
        try:
            # Expand search term with pattern mappings
            expanded_terms = self._expand_search_term(search_term)
            
            all_matches = []
            
            # Search with each expanded term
            for term in expanded_terms:
                if not term or not term.strip():
                    continue
                    
                # Use fuzzywuzzy to find best matches
                matches = process.extract(
                    term, 
                    self.search_candidates, 
                    scorer=fuzz.partial_ratio,
                    limit=50
                )
                
                # Filter by dynamic threshold
                threshold = self._get_dynamic_threshold(term, focus_areas)
                filtered_matches = [
                    (candidate, score/100.0) for candidate, score in matches # pyright: ignore[reportAssignmentType]
                    if score >= threshold
                ]
                
                all_matches.extend(filtered_matches)
            
            # Remove duplicates and sort by score
            unique_matches = {}
            for candidate, score in all_matches:
                if candidate not in unique_matches or score > unique_matches[candidate]:
                    unique_matches[candidate] = score
            
            # Sort by score descending
            sorted_matches = sorted(
                unique_matches.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            return sorted_matches[:20]  # Limit for performance
            
        except Exception as e:
            self.logger.debug(f"Fuzzy search failed for term '{search_term}': {e}")
            return []

    def _expand_search_term(self, term: str) -> List[str]:
        """Expand search term using pattern mappings and variations"""
        if not term or not term.strip():
            return [term] if term else []
            
        expanded = [term]
        try:
            term_lower = term.lower().strip()
            
            # Add pattern mapping expansions
            for abbrev, full_forms in self.pattern_mappings.items():
                if abbrev in term_lower:
                    for full_form in full_forms:
                        expanded.append(term_lower.replace(abbrev, full_form))
                elif any(full_form in term_lower for full_form in full_forms):
                    if full_forms:  # Ensure list is not empty
                        expanded.append(term_lower.replace(full_forms[0], abbrev))
            
            # Add camelCase variations
            if '_' in term:
                camel_case = ''.join(word.capitalize() for word in term.split('_'))
                expanded.append(camel_case)
            elif any(c.isupper() for c in term[1:]):  # Has camelCase
                snake_case = re.sub(r'([A-Z])', r'_\1', term).lower().lstrip('_')
                expanded.append(snake_case)
            
            # Add common variations
            expanded.extend([
                term.replace('_', ''),
                term.replace(' ', ''),
                term.replace('-', '_'),
                term.replace('_', ' ')
            ])
            
            # Remove duplicates and empty strings
            return list(set(t for t in expanded if t and t.strip()))
        except Exception as e:
            self.logger.debug(f"Term expansion failed for '{term}': {e}")
            return [term]

    def _get_dynamic_threshold(self, term: str, focus_areas: List[str]) -> int:
        """Calculate dynamic fuzzy threshold based on term and context"""
        try:
            base_threshold = self._fuzzy_threshold
            
            # Lower threshold for shorter terms (more tolerance)
            if len(term) <= 3:
                base_threshold = max(40, base_threshold - 20)
            elif len(term) <= 5:
                base_threshold = max(50, base_threshold - 10)
            
            # Adjust based on entity type weights
            for focus_area in focus_areas:
                if isinstance(focus_area, str) and focus_area.lower() in term.lower():
                    weight = self._entity_type_weights.get(focus_area, 1.0)
                    base_threshold = int(base_threshold * max(0.5, min(2.0, weight)))
                    break
            
            # Ensure reasonable bounds
            return max(30, min(90, base_threshold))
        except Exception as e:
            self.logger.debug(f"Dynamic threshold calculation failed: {e}")
            return self._fuzzy_threshold

    def _create_retrieved_column(self, item: Dict[str, Any], confidence: float = 1.0) -> Optional[RetrievedColumn]:
        """Create RetrievedColumn with proper confidence scoring"""
        try:
            table = item.get('table', '').strip()
            column = (item.get('column', '') or item.get('name', '')).strip()
            if not table or not column:
                return None
            
            # Validate confidence
            confidence = max(0.0, min(1.0, confidence))
            
            column_type = ColumnType.XML if item.get('type') == 'xml' else ColumnType.RELATIONAL
            if column_type == ColumnType.XML:
                xml_column = item.get('xml_column', '').strip()
                if not xml_column:
                    column_type = ColumnType.RELATIONAL
            
            if column_type == ColumnType.XML and item.get('xml_column'):
                return RetrievedColumn(
                    table=table,
                    column=column,
                    datatype='xml_field',
                    type=column_type,
                    description=item.get('description', ''),
                    confidence_score=confidence,
                    retrieval_method=SearchMethod.FUZZY,
                    xml_column=item.get('xml_column', ''),
                    xpath=item.get('xpath', ''),
                    sql_expression=item.get('sql_expression', ''),
                    retrieved_at=datetime.now()
                )
            else:
                return RetrievedColumn(
                    table=table,
                    column=column,
                    datatype=item.get('datatype', 'unknown'),
                    type=ColumnType.RELATIONAL,
                    description=item.get('description', ''),
                    confidence_score=confidence,
                    retrieval_method=SearchMethod.FUZZY,
                    retrieved_at=datetime.now()
                )
        except Exception as e:
            self.logger.debug(f"Error creating retrieved column: {e}")
            return None

    def ensure_initialized(self) -> None:
        """Ensure the engine is initialized"""
        if not self.search_candidates:
            self.initialize()

    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            self.search_candidates.clear()
            self.candidate_mapping.clear()
            self.enhanced_candidates.clear()
            self.pattern_mappings.clear()
            self.schema_data.clear()
            self.xml_data.clear()
            self._entity_type_weights.clear()
            self.logger.debug("Fuzzy search engine cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def health_check(self) -> Dict[str, Any]:
        """Enhanced health check"""
        try:
            base_health = super().health_check()
            
            # Calculate success rate
            success_rate = 0.0
            if self.total_searches > 0:
                success_rate = (self.successful_searches / self.total_searches) * 100
            
            base_health.update({
                "threshold": self.threshold,
                "max_results": self.max_results,
                "candidates_count": len(self.search_candidates),
                "enhanced_candidates": len(self.enhanced_candidates),
                "pattern_mappings": len(self.pattern_mappings),
                "schema_records": len(self.schema_data),
                "xml_records": len(self.xml_data),
                "search_statistics": {
                    "total_searches": self.total_searches,
                    "successful_searches": self.successful_searches,
                    "failed_searches": self.failed_searches,
                    "success_rate": round(success_rate, 2)
                },
                "fuzzy_parameters": {
                    "fuzzy_threshold": self._fuzzy_threshold,
                    "adaptive_matching": self._adaptive_matching,
                    "entity_type_weights_count": len(self._entity_type_weights)
                }
            })
            
            # Determine overall status
            if not self.search_candidates:
                base_health["status"] = "degraded"  # Not critical - can still work
            elif success_rate < 50 and self.total_searches > 5:
                base_health["status"] = "degraded"
            else:
                base_health["status"] = "healthy"
            
            return base_health
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "engine_type": "FUZZY"
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive fuzzy engine statistics"""
        try:
            success_rate = 0.0
            if self.total_searches > 0:
                success_rate = (self.successful_searches / self.total_searches) * 100
            
            return {
                "engine_type": "FUZZY",
                "search_statistics": {
                    "total_searches": self.total_searches,
                    "successful_searches": self.successful_searches,
                    "failed_searches": self.failed_searches,
                    "success_rate": round(success_rate, 2)
                },
                "engine_statistics": {
                    "candidates_count": len(self.search_candidates),
                    "enhanced_candidates": len(self.enhanced_candidates),
                    "pattern_mappings": len(self.pattern_mappings),
                    "schema_records": len(self.schema_data),
                    "xml_records": len(self.xml_data)
                },
                "parameters": {
                    "threshold": self.threshold,
                    "max_results": self.max_results,
                    "fuzzy_threshold": self._fuzzy_threshold,
                    "adaptive_matching": self._adaptive_matching
                }
            }
        except Exception as e:
            return {
                "engine_type": "FUZZY",
                "error": str(e)
            }
