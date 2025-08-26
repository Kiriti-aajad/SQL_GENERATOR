"""
Robust NLP search engine for schema search with ACTUAL NLP processing.
FIXED: All character encoding, implements real NLP search, and proper integration
"""

from typing import List, Dict, Any, Optional
import re
import logging
from datetime import datetime
from functools import lru_cache

try:
    import spacy
    from spacy.matcher import Matcher, PhraseMatcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
    Matcher = None
    PhraseMatcher = None

from agent.schema_searcher.engines.base_engine import BaseSearchEngine
from agent.schema_searcher.core.data_models import RetrievedColumn, ColumnType, SearchMethod
from agent.schema_searcher.loaders.schema_loader import SchemaLoader
from agent.schema_searcher.loaders.xml_loader import XMLLoader
from agent.schema_searcher.utils.performance import track_execution_time

logger = logging.getLogger(__name__)

class NLPSearchEngine(BaseSearchEngine):
    """
    NLP search engine with ACTUAL NLP processing using spaCy.
    FIXED: Now performs real NLP analysis instead of returning all data
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        super().__init__(SearchMethod.NLP, logger)
        self.model_name = model_name
        self.nlp = None
        self.matcher = None
        self.phrase_matcher = None
        self.schema_data: List[Dict[str, Any]] = []
        self.xml_data: List[Dict[str, Any]] = []
        self.discovered_patterns: Dict[str, List[str]] = {}

        # Enhanced business patterns for banking domain
        self.business_patterns = self._init_enhanced_business_patterns()
        self.operation_patterns = self._init_enhanced_operation_patterns()

        # Reasoning support
        self._strategy_parameters: Dict[str, Any] = {}
        self._reasoning_mode_enabled: bool = False
        
        # Statistics
        self.total_searches = 0
        self.successful_searches = 0
        self.failed_searches = 0

    def initialize(self) -> None:
        """Initialize NLP engine with proper error handling"""
        self.logger.info("Initializing NLP search engine")
        try:
            self._init_enhanced_spacy_model()
            
            # Load schema data
            loader = SchemaLoader()
            self.schema_data = loader.load()
            
            # Load XML data
            raw_xml = XMLLoader().load()
            self.xml_data = self._flatten_xml_data_enhanced(raw_xml)
            
            # Discover patterns from actual data
            self._discover_schema_patterns()
            self._build_enhanced_custom_patterns()
            
            self.logger.info(
                f"NLP engine initialized: "
                f"{len(self.schema_data)} relational, {len(self.xml_data)} XML fields, "
                f"spaCy available: {SPACY_AVAILABLE}, model loaded: {self.nlp is not None}"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP engine: {e}")
            # Don't raise - allow graceful degradation
            self.schema_data = []
            self.xml_data = []

    def _init_enhanced_spacy_model(self) -> None:
        """Initialize spaCy model with fallback handling"""
        if not SPACY_AVAILABLE:
            self.logger.warning("spaCy not available, using pattern matching only")
            return
            
        try:
            self.nlp = spacy.load(self.model_name) # pyright: ignore[reportOptionalMemberAccess]
            self.matcher = Matcher(self.nlp.vocab) # pyright: ignore[reportOptionalCall]
            self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER") # pyright: ignore[reportOptionalCall]
            self.logger.info(f"spaCy loaded: {self.model_name}")
        except OSError:
            self.logger.warning(f"spaCy model '{self.model_name}' not found, trying alternative")
            # Try alternative models
            for alt_model in ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]:
                if alt_model != self.model_name:
                    try:
                        self.nlp = spacy.load(alt_model) # type: ignore
                        self.matcher = Matcher(self.nlp.vocab) # pyright: ignore[reportOptionalCall]
                        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER") # pyright: ignore[reportOptionalCall]
                        self.logger.info(f"Loaded alternative spaCy model: {alt_model}")
                        break
                    except OSError:
                        continue
            else:
                self.logger.warning("No spaCy model available, using pattern matching only")
                self.nlp = self.matcher = self.phrase_matcher = None

    def _flatten_xml_data_enhanced(self, raw_xml_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten XML data with enhanced descriptions"""
        flattened = []
        if not raw_xml_data:
            return flattened
            
        for xml_table in raw_xml_data:
            table = xml_table.get('table', '')
            col = xml_table.get('xml_column', '')
            
            for field in xml_table.get('fields', []):
                name = field.get('name', '')
                xpath = field.get('xpath', '')
                
                if not name:
                    continue
                    
                # Create rich description for NLP matching
                desc_parts = [
                    f"XML field {name}",
                    f"in table {table}",
                    f"column {col}",
                ]
                if xpath:
                    desc_parts.append(f"xpath {xpath}")
                
                # Add semantic keywords based on field name
                semantic_keywords = self._extract_semantic_keywords(name)
                if semantic_keywords:
                    desc_parts.append(f"related to: {', '.join(semantic_keywords)}")
                
                flattened.append({
                    'table': table,
                    'column': name,
                    'xml_column': col,
                    'xpath': xpath,
                    'sql_expression': field.get('sql_expression', ''),
                    'type': 'xml',
                    'datatype': 'xml_field',
                    'description': " | ".join(desc_parts),
                    'semantic_keywords': semantic_keywords
                })
        return flattened

    def _extract_semantic_keywords(self, field_name: str) -> List[str]:
        """Extract semantic keywords from field names for better matching"""
        keywords = []
        name_lower = field_name.lower()
        
        # Check against business patterns
        for category, patterns in self.business_patterns.items():
            for pattern in patterns:
                if pattern.lower() in name_lower:
                    keywords.append(category)
                    keywords.append(pattern)
        
        # Add camelCase/snake_case variations
        if '_' in field_name:
            keywords.extend(field_name.split('_'))
        elif any(c.isupper() for c in field_name[1:]):
            # CamelCase splitting
            parts = re.findall(r'[A-Z][a-z]*|[a-z]+', field_name)
            keywords.extend(parts)
        
        return list(set(keywords))

    def _init_enhanced_business_patterns(self) -> Dict[str, List[str]]:
        """Enhanced business patterns for banking domain"""
        return {
            "financial": ["amount", "payment", "price", "cost", "value", "total", "sum", "account", "balance", "money", "currency", "dollar", "fee", "charge"],
            "customer": ["customer", "client", "user", "person", "company", "individual", "holder", "owner", "party"],
            "temporal": ["date", "time", "created", "modified", "year", "month", "period", "duration", "timestamp", "when", "schedule", "expiry", "maturity"],
            "location": ["address", "city", "country", "location", "zipcode", "zip", "postal", "region", "state", "branch"],
            "status": ["status", "active", "inactive", "pending", "approved", "rejected", "closed", "open", "valid", "expired"],
            "identifier": ["id", "code", "number", "reference", "key", "identifier", "ref", "num", "uuid", "guid"],
            "workflow": ["workflow", "process", "step", "task", "queue", "stage", "phase", "action"],
            "application": ["application", "app", "request", "form", "submission", "loan", "credit", "mortgage"],
            "tracking": ["tracking", "monitoring", "history", "progress", "audit", "log", "trace"],
            "transaction": ["transaction", "transfer", "deposit", "withdrawal", "payment", "debit", "credit", "txn"],
            "banking": ["bank", "banking", "financial", "institution", "branch", "atm", "card", "loan", "mortgage", "savings", "checking"]
        }

    def _init_enhanced_operation_patterns(self) -> Dict[str, List[str]]:
        """Enhanced operation patterns"""
        return {
            "select": ["show", "get", "find", "list", "display", "retrieve", "fetch"],
            "count": ["count", "number", "how many", "total number", "quantity"],
            "sum": ["sum", "total", "add up", "aggregate", "combine"],
            "average": ["average", "avg", "mean", "typical"],
            "max": ["max", "highest", "largest", "top", "maximum", "greatest"],
            "min": ["min", "lowest", "smallest", "least", "minimum"],
            "filter": ["where", "filter", "with", "having", "matching"],
            "group": ["group", "categorize", "organize", "classify"]
        }

    def _discover_schema_patterns(self) -> None:
        """Discover patterns from actual schema data"""
        self.discovered_patterns = {}
        
        all_items = self.schema_data + self.xml_data
        if not all_items:
            return
            
        # Collect column names and descriptions
        columns = []
        descriptions = []
        
        for item in all_items:
            col_name = item.get('column') or item.get('name', '')
            if col_name:
                columns.append(col_name.lower())
            
            desc = item.get('description', '')
            if desc:
                descriptions.append(desc.lower())
        
        # Find common patterns in column names
        common_prefixes = self._find_common_patterns(columns, 'prefix')
        common_suffixes = self._find_common_patterns(columns, 'suffix')
        
        self.discovered_patterns = {
            'common_prefixes': common_prefixes,
            'common_suffixes': common_suffixes,
            'total_columns': len(columns) # pyright: ignore[reportAttributeAccessIssue]
        }

    def _find_common_patterns(self, items: List[str], pattern_type: str) -> List[str]:
        """Find common prefixes or suffixes in column names"""
        patterns = {}
        min_length = 3
        
        for item in items:
            if len(item) < min_length:
                continue
                
            if pattern_type == 'prefix':
                for i in range(min_length, min(len(item) + 1, 8)):
                    pattern = item[:i]
                    patterns[pattern] = patterns.get(pattern, 0) + 1
            elif pattern_type == 'suffix':
                for i in range(min_length, min(len(item) + 1, 8)):
                    pattern = item[-i:]
                    patterns[pattern] = patterns.get(pattern, 0) + 1
        
        # Return patterns that appear in at least 2 items
        return [pattern for pattern, count in patterns.items() if count >= 2]

    def _build_enhanced_custom_patterns(self) -> None:
        """Build custom patterns for spaCy matcher if available"""
        if not self.matcher:
            return
            
        try:
            # Add patterns for common banking terms
            banking_patterns = [
                [{"LOWER": "customer"}, {"LOWER": {"IN": ["id", "number", "code", "identifier"]}}],
                [{"LOWER": "account"}, {"LOWER": {"IN": ["number", "balance", "type", "status"]}}],
                [{"LOWER": {"IN": ["loan", "credit", "mortgage"]}}, {"LOWER": {"IN": ["amount", "balance", "payment"]}}],
                [{"LOWER": "transaction"}, {"LOWER": {"IN": ["id", "amount", "date", "type"]}}],
            ]
            
            for i, pattern in enumerate(banking_patterns):
                self.matcher.add(f"BANKING_PATTERN_{i}", [pattern])
                
            self.logger.debug(f"Added {len(banking_patterns)} custom patterns to spaCy matcher")
            
        except Exception as e:
            self.logger.warning(f"Failed to build custom patterns: {e}")

    @track_execution_time
    def search(self, query: str, top_k: int = 25) -> List[RetrievedColumn]:
        """
        FIXED: Actually perform NLP search instead of returning all data
        """
        self.total_searches += 1
        
        # Input validation
        if not query or not isinstance(query, str):
            self.failed_searches += 1
            return []
            
        query = query.strip()
        if not query:
            self.failed_searches += 1
            return []
            
        if not (self.schema_data or self.xml_data):
            self.logger.warning("No schema data available for NLP search")
            self.failed_searches += 1
            return []

        try:
            # FIXED: Actually use the query for searching
            all_items = self.schema_data + self.xml_data
            scored_items = []
            
            # Process query with spaCy if available
            query_tokens = self._extract_query_tokens(query)
            query_concepts = self._extract_concepts(query)
            
            # Score each item based on relevance to query
            for item in all_items:
                score = self._calculate_relevance_score(item, query, query_tokens, query_concepts)
                if score > 0:  # Only include relevant items
                    scored_items.append((item, score))
            
            # Sort by relevance score (descending)
            scored_items.sort(key=lambda x: x[1], reverse=True)
            
            # Convert to RetrievedColumn objects
            results = []
            seen = set()
            
            for item, score in scored_items:
                if len(results) >= top_k:
                    break
                    
                table = item.get('table', '').strip()
                column = (item.get('column') or item.get('name', '')).strip()
                
                if not table or not column:
                    continue
                
                # Deduplication
                key = f"{table.lower()}.{column.lower()}"
                if key in seen:
                    continue
                seen.add(key)
                
                # Determine column type
                column_type = ColumnType.XML if item.get('type') == 'xml' else ColumnType.RELATIONAL
                if column_type == ColumnType.XML and not item.get('xml_column', '').strip():
                    column_type = ColumnType.RELATIONAL
                
                # Calculate confidence based on relevance score
                confidence_score = min(1.0, max(0.1, score / 10.0))  # Normalize to 0.1-1.0
                
                result = RetrievedColumn(
                    table=table,
                    column=column,
                    datatype=('xml_field' if column_type == ColumnType.XML else item.get('datatype', 'unknown')),
                    type=column_type,
                    description=item.get('description', ''),
                    confidence_score=confidence_score,
                    retrieval_method=SearchMethod.NLP,
                    xml_column=item.get('xml_column', '') if column_type == ColumnType.XML else None,
                    xpath=item.get('xpath', '') if column_type == ColumnType.XML else None,
                    sql_expression=item.get('sql_expression', '') if column_type == ColumnType.XML else None,
                    retrieved_at=datetime.now()
                )
                results.append(result)

            self.successful_searches += 1
            self.logger.debug(f"NLP search returned {len(results)} results for '{query}' (from {len(scored_items)} relevant items)")
            return results

        except Exception as e:
            self.failed_searches += 1
            self.logger.error(f"NLP search error: {e}")
            return []

    def _extract_query_tokens(self, query: str) -> List[str]:
        """Extract meaningful tokens from query"""
        if self.nlp:
            try:
                doc = self.nlp(query.lower())
                # Extract meaningful tokens (skip stop words, punctuation)
                tokens = [token.lemma_ for token in doc 
                         if not token.is_stop and not token.is_punct and len(token.text) > 1]
                return tokens
            except Exception as e:
                self.logger.debug(f"spaCy token extraction failed: {e}")
        
        # Fallback: simple tokenization
        tokens = re.findall(r'\b\w+\b', query.lower())
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        return [token for token in tokens if token not in stop_words and len(token) > 1]

    def _extract_concepts(self, query: str) -> List[str]:
        """Extract business concepts from query"""
        concepts = []
        query_lower = query.lower()
        
        # Check against business patterns
        for category, patterns in self.business_patterns.items():
            for pattern in patterns:
                if pattern.lower() in query_lower:
                    concepts.append(category)
                    concepts.append(pattern)
        
        return list(set(concepts))

    def _calculate_relevance_score(self, item: Dict[str, Any], query: str, query_tokens: List[str], query_concepts: List[str]) -> float:
        """Calculate relevance score for an item based on query"""
        score = 0.0
        
        # Get searchable text from item
        table = item.get('table', '').lower()
        column = (item.get('column') or item.get('name', '')).lower()
        description = item.get('description', '').lower()
        datatype = item.get('datatype', '').lower()
        
        # Combine all searchable text
        searchable_text = f"{table} {column} {description} {datatype}"
        
        # Add semantic keywords if available
        semantic_keywords = item.get('semantic_keywords', [])
        if semantic_keywords:
            searchable_text += " " + " ".join(str(kw).lower() for kw in semantic_keywords)
        
        # Score based on exact matches
        query_lower = query.lower()
        if query_lower in searchable_text:
            score += 10.0
        
        # Score based on token matches
        for token in query_tokens:
            if token in searchable_text:
                score += 3.0
            # Partial matches
            if any(token in word for word in searchable_text.split()):
                score += 1.0
        
        # Score based on concept matches
        for concept in query_concepts:
            if concept.lower() in searchable_text:
                score += 2.0
        
        # Boost for column name matches
        if any(token in column for token in query_tokens):
            score += 5.0
        
        # Boost for table name matches
        if any(token in table for token in query_tokens):
            score += 3.0
        
        return score

    # FIXED: Add missing async method for compatibility
    async def search_async(
        self, 
        query: Optional[str] = None, 
        keywords: Optional[List[str]] = None, 
        top_k: Optional[int] = None
    ) -> List[RetrievedColumn]:
        """Async search method for compatibility with retrieval agent"""
        # Handle both query and keywords parameters
        if query:
            search_query = query
        elif keywords:
            search_query = " ".join(str(kw) for kw in keywords if kw)
        else:
            return []
        
        # NLP search is synchronous, so just call the regular search method
        return self.search(search_query, top_k or 25)

    def ensure_initialized(self) -> None:
        """Ensure the engine is initialized"""
        if not (self.schema_data or self.xml_data):
            self.initialize()

    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            self.nlp = None
            self.matcher = None
            self.phrase_matcher = None
            self.schema_data.clear()
            self.xml_data.clear()
            self.discovered_patterns.clear()
            self.logger.info("NLP search engine cleaned up")
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
                "model_name": self.model_name,
                "spacy_available": SPACY_AVAILABLE,
                "nlp_loaded": self.nlp is not None,
                "matcher_loaded": self.matcher is not None,
                "phrase_matcher_loaded": self.phrase_matcher is not None,
                "schema_records": len(self.schema_data),
                "xml_records": len(self.xml_data),
                "discovered_patterns": len(self.discovered_patterns),
                "business_patterns": len(self.business_patterns),
                "search_statistics": {
                    "total_searches": self.total_searches,
                    "successful_searches": self.successful_searches,
                    "failed_searches": self.failed_searches,
                    "success_rate": round(success_rate, 2)
                }
            })
            
            if self._reasoning_mode_enabled:
                base_health.update({
                    "reasoning_enabled": True,
                    "strategy_params": list(self._strategy_parameters.keys())
                })
            
            # Determine overall status
            if len(self.schema_data) == 0 and len(self.xml_data) == 0:
                base_health["status"] = "degraded"
            elif success_rate < 50 and self.total_searches > 5:
                base_health["status"] = "degraded"
            else:
                base_health["status"] = "healthy"
                
            return base_health
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "engine_type": "NLP"
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive NLP engine statistics"""
        try:
            success_rate = 0.0
            if self.total_searches > 0:
                success_rate = (self.successful_searches / self.total_searches) * 100
            
            return {
                "engine_type": "NLP",
                "search_statistics": {
                    "total_searches": self.total_searches,
                    "successful_searches": self.successful_searches,
                    "failed_searches": self.failed_searches,
                    "success_rate": round(success_rate, 2)
                },
                "nlp_capabilities": {
                    "spacy_available": SPACY_AVAILABLE,
                    "model_loaded": self.nlp is not None,
                    "matcher_available": self.matcher is not None,
                    "model_name": self.model_name
                },
                "data_statistics": {
                    "schema_records": len(self.schema_data),
                    "xml_records": len(self.xml_data),
                    "business_patterns": len(self.business_patterns),
                    "discovered_patterns": len(self.discovered_patterns)
                }
            }
        except Exception as e:
            return {
                "engine_type": "NLP",
                "error": str(e)
            }

    # ─── ENHANCED METHODS FOR MATHSTRAL REASONING ───

    def search_with_strategy(self, strategy: Dict[str, Any], target_gaps: List[str]) -> List[RetrievedColumn]:
        """Execute a gap-aware, strategy-driven search"""
        self._reasoning_mode_enabled = True
        keywords = strategy.get("keywords", target_gaps)
        max_results = strategy.get("max_results", 25)
        
        combined_results = []
        seen = set()

        # Search for each keyword/gap
        for term in keywords:
            if not term or not isinstance(term, str):
                continue
                
            subset = self.search(term, max_results)
            for result in subset:
                key = f"{result.table.lower()}.{result.column.lower()}"
                if key not in seen:
                    seen.add(key)
                    combined_results.append(result)
                    if len(combined_results) >= max_results:
                        break
                        
            if len(combined_results) >= max_results:
                break

        # If too few results, try broader search
        if len(combined_results) < max_results // 2:
            fallback_query = " ".join(str(kw) for kw in keywords[:3] if kw)
            if fallback_query:
                fallback_results = self.search(fallback_query, max_results)
                for result in fallback_results:
                    key = f"{result.table.lower()}.{result.column.lower()}"
                    if key not in seen:
                        combined_results.append(result)
                        if len(combined_results) >= max_results:
                            break

        return combined_results

    def set_search_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set search parameters for strategy-based search"""
        super().set_search_parameters(parameters)
        self._strategy_parameters.update(parameters)
        self._reasoning_mode_enabled = True
        self.logger.debug(f"[NLP] strategy params set: {parameters}")

    def adjust_search_parameters(self, current_results: List[RetrievedColumn], target_criteria: Dict[str, Any]) -> None:
        """Adjust search parameters based on current results"""
        super().adjust_search_parameters(current_results, target_criteria)
        
        min_results = target_criteria.get("min_results", 5)
        if len(current_results) < min_results:
            # Relax matching criteria
            self._strategy_parameters["relaxed_matching"] = True
            self.business_patterns.setdefault("relaxed", []).extend([".*", "general", "common"])
            self.logger.debug("[NLP] Relaxed matching criteria for better recall")

    def get_search_suggestions(self) -> List[str]:
        """Provide search optimization suggestions"""
        suggestions = super().get_search_suggestions()
        
        if self.total_searches > 0:
            success_rate = (self.successful_searches / self.total_searches)
            if success_rate < 0.8:
                suggestions.append("Low NLP success rate: consider updating business patterns")
        
        if not SPACY_AVAILABLE:
            suggestions.append("spaCy not available: install spaCy for better NLP processing")
        elif not self.nlp:
            suggestions.append(f"spaCy model '{self.model_name}' not loaded: check model installation")
        
        if len(self.discovered_patterns) < 5:
            suggestions.append("Few discovered patterns: consider enriching schema descriptions")
        
        return suggestions

    def get_strategy_parameters(self) -> Dict[str, Any]:
        """Return current strategy parameters"""
        return self._strategy_parameters.copy()

    def reset_strategy_parameters(self) -> None:
        """Clear any strategy overrides"""
        super().reset_strategy_parameters()
        self._strategy_parameters.clear()
        self._reasoning_mode_enabled = False
        self.logger.debug("[NLP] strategy parameters reset")

    def is_reasoning_enabled(self) -> bool:
        """Whether this engine is currently in reasoning mode"""
        return self._reasoning_mode_enabled

    def success_rate(self) -> float:
        """Calculate current success rate"""
        if self.total_searches == 0:
            return 1.0
        return self.successful_searches / self.total_searches
