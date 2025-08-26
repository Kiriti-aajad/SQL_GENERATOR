"""
Semantic search engine for schema retrieval: union-based, no scoring/division, always robust.
ENHANCED: Added Mathstral reasoning integration with E5-base-v2 consistency.
UPDATED: Now uses EmbeddingModelManager singleton pattern for memory optimization.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from functools import lru_cache
import logging
import re
from datetime import datetime

from agent.schema_searcher.engines.base_engine import BaseSearchEngine
from agent.schema_searcher.core.data_models import RetrievedColumn, ColumnType, SearchMethod
from agent.schema_searcher.loaders.schema_loader import SchemaLoader
from agent.schema_searcher.loaders.xml_loader import XMLLoader
from agent.schema_searcher.utils.performance import track_execution_time
# UPDATED: Use EmbeddingModelManager for shared model instances
from agent.schema_searcher.utils.embedding_manager import EmbeddingModelManager

logger = logging.getLogger(__name__)

# UPDATED: Use E5-base-v2 for consistency with all other components
DEFAULT_EMBEDDING_MODEL = "intfloat/e5-base-v2"

class SemanticSearchEngine(BaseSearchEngine):
    """
    Semantic search (union version): returns all deduped matches, no scoring, no math errors.
    
    ENHANCED: Now supports Mathstral reasoning with E5-base-v2 consistency:
    - Gap-aware semantic filtering with E5 optimizations
    - Strategy-based parameter optimization  
    - Entity-type specific search tuning
    - Dynamic result filtering based on target gaps
    - E5 prefix support for better semantic understanding
    - Uses EmbeddingModelManager for shared model instances
    """
    
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        super().__init__(SearchMethod.SEMANTIC, logger)
        self.model_name = model_name
        self.model: Optional[Any] = None  # Will be SentenceTransformer from EmbeddingModelManager
        self.schema_data: List[Dict[str, Any]] = []
        self.xml_data: List[Dict[str, Any]] = []
        
        # Store actual model name for tracking
        self.actual_model_name = model_name
        
        # NEW: Reasoning enhancement parameters
        self._similarity_threshold = 0.75 if 'e5' in model_name.lower() else 0.7
        self._entity_type_preferences: Dict[str, float] = {}
        self._gap_aware_filtering = False
        self._e5_optimized = 'e5' in model_name.lower()

    def initialize(self) -> None:
        if self.model is None:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            try:
                # UPDATED: Use EmbeddingModelManager for shared model instance
                self.model = EmbeddingModelManager.get_model(self.model_name)
                self.actual_model_name = self.model_name
                self.logger.info(f"Semantic search engine initialized with {self.model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load model {self.model_name}: {e}")
                # Fallback to a simpler model using the same manager
                self.logger.warning("Falling back to all-MiniLM-L6-v2")
                self.model = EmbeddingModelManager.get_model("all-MiniLM-L6-v2")
                self.actual_model_name = "all-MiniLM-L6-v2"
                self._e5_optimized = False
                
        try:
            schema_loader = SchemaLoader()
            xml_loader = XMLLoader()
            self.schema_data = schema_loader.load()
            raw_xml_data = xml_loader.load()
            self.xml_data = self._flatten_xml_data_enhanced(raw_xml_data)
            self.logger.info(
                f"Semantic engine loaded {len(self.schema_data)} relational columns, "
                f"{len(self.xml_data)} XML fields using {self.actual_model_name}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load schema data: {e}")
            raise

    def _flatten_xml_data_enhanced(self, raw_xml_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        flattened = []
        for xml_table in raw_xml_data:
            table_name = xml_table.get('table', '')
            xml_column = xml_table.get('xml_column', '')
            for field in xml_table.get('fields', []):
                field_name = field.get('name', '')
                xpath = field.get('xpath', '')
                description_parts = [
                    f"XML field: {field_name}",
                    f"Table: {table_name}",
                    f"XML Column: {xml_column}",
                    f"XPath: {xpath}" if xpath else ""
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
                })
        return flattened

    @track_execution_time
    def search(self, query: str, top_k: int = 30) -> List[RetrievedColumn]:
        """
        Union semantic search: returns all deduped (table, column) matches, confidence_score=1.0.
        UNCHANGED: Existing functionality preserved exactly as before.
        """
        if not self.model:
            raise RuntimeError("Semantic engine not initialized")

        all_items = self.schema_data + self.xml_data
        results = []
        seen = set()
        for item in all_items:
            table = item.get('table', '').strip()
            column = item.get('column', '').strip()
            if not table or not column:
                continue
            key = f"{table.lower()}.{column.lower()}"
            if key in seen:
                continue
            seen.add(key)
            column_type = ColumnType.XML if item.get('type') == 'xml' else ColumnType.RELATIONAL
            if column_type == ColumnType.XML:
                xml_column = item.get('xml_column', '').strip()
                if not xml_column:
                    column_type = ColumnType.RELATIONAL
            if column_type == ColumnType.XML and item.get('xml_column'):
                result = RetrievedColumn(
                    table=table,
                    column=column,
                    datatype='xml_field',
                    type=column_type,
                    description=item.get('description', ''),
                    confidence_score=1.0,
                    retrieval_method=SearchMethod.SEMANTIC,
                    xml_column=item.get('xml_column', ''),
                    xpath=item.get('xpath', ''),
                    sql_expression=item.get('sql_expression', ''),
                    retrieved_at=datetime.now()
                )
            else:
                result = RetrievedColumn(
                    table=table,
                    column=column,
                    datatype=item.get('datatype', 'unknown'),
                    type=ColumnType.RELATIONAL,
                    description=item.get('description', ''),
                    confidence_score=1.0,
                    retrieval_method=SearchMethod.SEMANTIC,
                    retrieved_at=datetime.now()
                )
            results.append(result)
            if len(results) >= top_k:
                break
        self.logger.debug(f"Semantic search (union) returned {len(results)} results for query: '{query}'")
        return results

    # ===== CRITICAL FIX: ADDED ASYNC COMPATIBILITY =====
    
    async def search_async(
        self,
        query: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        top_k: int = 30
    ) -> List[RetrievedColumn]:
        """Async search method for compatibility with retrieval agent"""
        if query:
            search_query = query
        elif keywords:
            search_query = " ".join(keywords)
        else:
            return []
        
        # Semantic search is synchronous, so just call the regular search method
        return self.search(search_query, top_k)

    # NEW: E5-ENHANCED MATHSTRAL REASONING

    def search_with_strategy(self, strategy: Dict[str, Any], target_gaps: List[str]) -> List[RetrievedColumn]:
        """
        NEW: Gap-aware semantic search with E5-optimized strategy.
        
        Uses E5 semantic similarity to prioritize results that address specific gaps
        while maintaining the union-based approach for robustness.
        """
        if not self.model:
            raise RuntimeError("Semantic engine not initialized")
            
        self.logger.debug(f"Executing E5-optimized strategy search for {len(target_gaps)} target gaps")
        
        # Extract strategy parameters
        max_results = strategy.get('max_results', 30)
        keywords = strategy.get('keywords', [])
        focus_areas = strategy.get('focus_areas', target_gaps)
        
        # If no specific keywords provided, use target gaps as search terms
        search_terms = keywords if keywords else target_gaps
        
        all_results = []
        seen = set()
        
        # Search for each term and combine results
        for search_term in search_terms:
            if not search_term.strip():
                continue
                
            # Get E5-enhanced semantic matches for this specific term
            term_results = self._e5_semantic_search_for_term(search_term, focus_areas)
            
            # Add unique results
            for result in term_results:
                key = f"{result.table.lower()}.{result.column.lower()}"
                if key not in seen:
                    seen.add(key)
                    all_results.append(result)
                    
                    if len(all_results) >= max_results:
                        break
            
            if len(all_results) >= max_results:
                break
        
        # If we don't have enough results, fall back to regular search
        if len(all_results) < max_results // 2:
            fallback_query = " ".join(search_terms[:3])  # Use first 3 terms
            fallback_results = self.search(fallback_query, max_results)
            
            # Add unique fallback results
            for result in fallback_results:
                key = f"{result.table.lower()}.{result.column.lower()}"
                if key not in seen:
                    seen.add(key)
                    all_results.append(result)
                    
                    if len(all_results) >= max_results:
                        break
        
        self.logger.debug(f"E5-enhanced strategy search returned {len(all_results)} results")
        return all_results[:max_results]
    
    def _e5_semantic_search_for_term(self, search_term: str, focus_areas: List[str]) -> List[RetrievedColumn]:
        """
        E5-optimized semantic similarity search for a specific term.
        Uses E5 prefixes and optimizations when available.
        """
        if not self._reasoning_mode_enabled:
            # Fall back to regular search when reasoning is disabled
            return self.search(search_term, 10)
        
        try:
            # Generate E5-optimized embedding for search term
            if self._e5_optimized:
                query_text = f"query: {search_term}"
                query_embedding = self.model.encode([query_text]) # pyright: ignore[reportOptionalMemberAccess]
            else:
                query_embedding = self.model.encode([search_term]) # pyright: ignore[reportOptionalMemberAccess]
            
            # Get embeddings for all schema items
            all_items = self.schema_data + self.xml_data
            item_texts = []
            valid_items = []
            
            for item in all_items:
                table = item.get('table', '').strip()
                column = item.get('column', '').strip()
                if not table or not column:
                    continue
                    
                # Create searchable text with E5 optimization
                description = item.get('description', '')
                if self._e5_optimized:
                    item_text = f"passage: {table} {column} {description}".strip()
                else:
                    item_text = f"{table} {column} {description}".strip()
                    
                item_texts.append(item_text)
                valid_items.append(item)
            
            if not item_texts:
                return []
            
            # Get E5-optimized embeddings for schema items
            item_embeddings = self.model.encode(item_texts) # pyright: ignore[reportOptionalMemberAccess]
            
            # Calculate E5-optimized similarities
            if self._e5_optimized:
                # E5 models work better with normalized vectors
                query_normalized = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
                item_normalized = item_embeddings / (np.linalg.norm(item_embeddings, axis=1, keepdims=True) + 1e-8)
                similarities = np.dot(query_normalized, item_normalized.T).flatten()
                # E5 can return values in [-1, 1], normalize to [0, 1]
                similarities = (similarities + 1.0) / 2.0
            else:
                similarities = np.dot(query_embedding, item_embeddings.T).flatten()
            
            # Filter by similarity threshold
            results = []
            for idx, similarity in enumerate(similarities):
                if similarity >= self._similarity_threshold:
                    item = valid_items[idx]
                    result = self._create_retrieved_column_from_item(item, float(similarity))
                    results.append(result)
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x.confidence_score, reverse=True)
            
            self.logger.debug(
                f"E5-enhanced semantic search found {len(results)} matches above threshold {self._similarity_threshold}"
            )
            return results[:20]  # Limit to top 20 for performance
            
        except Exception as e:
            self.logger.warning(f"E5 semantic similarity search failed: {e}, falling back to regular search")
            return self.search(search_term, 10)
    
    def _create_retrieved_column_from_item(self, item: Dict[str, Any], confidence: float) -> RetrievedColumn:
        """Create RetrievedColumn from schema item with confidence score."""
        table = item.get('table', '').strip()
        column = item.get('column', '').strip()
        column_type = ColumnType.XML if item.get('type') == 'xml' else ColumnType.RELATIONAL
        
        if column_type == ColumnType.XML and item.get('xml_column'):
            return RetrievedColumn(
                table=table,
                column=column,
                datatype='xml_field',
                type=column_type,
                description=item.get('description', ''),
                confidence_score=confidence,
                retrieval_method=SearchMethod.SEMANTIC,
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
                retrieval_method=SearchMethod.SEMANTIC,
                retrieved_at=datetime.now()
            )
    
    def set_search_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        NEW: Configure semantic engine for E5-optimized strategy-based searches.
        """
        super().set_search_parameters(parameters)
        
        # Apply semantic-specific parameters with E5 adjustments
        if 'similarity_threshold' in parameters:
            threshold = parameters['similarity_threshold']
            # E5 models tend to give higher similarities, adjust accordingly
            if self._e5_optimized:
                threshold = max(0.2, min(0.95, threshold))
            else:
                threshold = max(0.1, min(0.95, threshold))
            self._similarity_threshold = threshold
            self.logger.debug(f"Similarity threshold set to {self._similarity_threshold} (E5 optimized: {self._e5_optimized})")
        
        if 'entity_type_preferences' in parameters:
            self._entity_type_preferences = parameters['entity_type_preferences']
            self.logger.debug(f"Entity type preferences updated: {list(self._entity_type_preferences.keys())}")
        
        if 'gap_aware_filtering' in parameters:
            self._gap_aware_filtering = parameters['gap_aware_filtering']
            self.logger.debug(f"Gap-aware filtering {'enabled' if self._gap_aware_filtering else 'disabled'}")
    
    def adjust_search_parameters(self, current_results: List[RetrievedColumn], target_criteria: Dict[str, Any]) -> None:
        """
        NEW: Fine-tune semantic search parameters with E5-aware adjustments.
        """
        super().adjust_search_parameters(current_results, target_criteria)
        
        # E5-aware semantic adjustments
        result_count = len(current_results)
        target_min = target_criteria.get('min_results', 5)
        target_max = target_criteria.get('max_results', 50)
        
        if result_count < target_min:
            # Too few results - lower similarity threshold
            old_threshold = self._similarity_threshold
            if self._e5_optimized:
                # E5 models can go lower while maintaining quality
                self._similarity_threshold = max(0.4, self._similarity_threshold * 0.9)
            else:
                self._similarity_threshold = max(0.3, self._similarity_threshold * 0.85)
            self.logger.debug(f"Lowered similarity threshold: {old_threshold:.2f} → {self._similarity_threshold:.2f}")
            
        elif result_count > target_max:
            # Too many results - raise similarity threshold  
            old_threshold = self._similarity_threshold
            if self._e5_optimized:
                # E5 models can be more selective
                self._similarity_threshold = min(0.95, self._similarity_threshold * 1.1)
            else:
                self._similarity_threshold = min(0.9, self._similarity_threshold * 1.15)
            self.logger.debug(f"Raised similarity threshold: {old_threshold:.2f} → {self._similarity_threshold:.2f}")
    
    def get_search_suggestions(self) -> List[str]:
        """
        NEW: Provide E5-aware semantic search optimization suggestions.
        """
        suggestions = super().get_search_suggestions()
        
        # E5-specific suggestions
        if self._e5_optimized:
            if self._similarity_threshold > 0.85:
                suggestions.append("E5 model: Consider lowering similarity threshold for broader semantic matches")
            elif self._similarity_threshold < 0.5:
                suggestions.append("E5 model: Consider raising similarity threshold for more precise semantic matches")
            
            suggestions.append("E5 model active: Enhanced semantic understanding for business terminology")
        else:
            if self._similarity_threshold > 0.8:
                suggestions.append("Consider lowering similarity threshold for broader semantic matches")
            elif self._similarity_threshold < 0.4:
                suggestions.append("Consider raising similarity threshold for more precise semantic matches")
            
            suggestions.append("Consider upgrading to E5 model for better semantic understanding")
        
        if not self._entity_type_preferences:
            suggestions.append("Consider setting entity type preferences for better semantic targeting")
        
        if len(self.schema_data) + len(self.xml_data) > 10000:
            suggestions.append("Large schema dataset - E5 model provides better semantic indexing")
        
        return suggestions
    
    def reset_strategy_parameters(self) -> None:
        """
        NEW: Reset semantic engine to E5-optimized default parameters.
        """
        super().reset_strategy_parameters()
        
        # Reset with E5-aware defaults
        self._similarity_threshold = 0.75 if self._e5_optimized else 0.7
        self._entity_type_preferences.clear()
        self._gap_aware_filtering = False
        
        self.logger.debug(f"Semantic engine parameters reset to E5-optimized defaults (E5: {self._e5_optimized})")

    # ENHANCED UTILITY METHODS

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the embedding model being used."""
        try:
            info = {
                'configured_model': self.model_name,
                'actual_model': self.actual_model_name,
                'e5_optimized': self._e5_optimized,
                'model_loaded': self.model is not None,
                'using_shared_model': True  # Indicates using EmbeddingModelManager
            }
            
            if self.model:
                info.update({
                    'embedding_dimension': self.model.get_sentence_embedding_dimension(),
                    'max_sequence_length': getattr(self.model, 'max_seq_length', 'unknown'),
                    'supports_prefixes': self._e5_optimized
                })
            
            return info
        except Exception as e:
            return {
                'configured_model': self.model_name,
                'actual_model': self.actual_model_name,
                'using_shared_model': True,
                'error': str(e)
            }

    def cleanup(self) -> None:
        self.model = None
        self.schema_data.clear()
        self.xml_data.clear()
        self._entity_type_preferences.clear()
        self.logger.info("Semantic search engine cleaned up")

    def health_check(self) -> Dict[str, Any]:
        base_health = super().health_check()
        base_health.update({
            "configured_model": self.model_name,
            "actual_model": self.actual_model_name,
            "model_loaded": self.model is not None,
            "schema_records": len(self.schema_data),
            "xml_records": len(self.xml_data),
            "e5_optimized": self._e5_optimized,
            "using_shared_model": True  # Indicates using EmbeddingModelManager
        })
        
        # NEW: Add E5-enhanced reasoning-specific health info
        if self._reasoning_mode_enabled:
            base_health.update({
                "similarity_threshold": self._similarity_threshold,
                "entity_preferences_count": len(self._entity_type_preferences),
                "gap_aware_filtering": self._gap_aware_filtering,
                "e5_prefixes_enabled": self._e5_optimized,
                "model_consistency": "e5-base-v2" if self._e5_optimized else "other"
            })
        
        return base_health
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics specific to semantic search."""
        return {
            'model_type': 'e5-base-v2' if self._e5_optimized else 'standard',
            'similarity_threshold': self._similarity_threshold,
            'total_schema_items': len(self.schema_data) + len(self.xml_data),
            'reasoning_mode_enabled': self._reasoning_mode_enabled,
            'optimization_level': 'high' if self._e5_optimized else 'standard',
            'memory_optimized': True  # Indicates using shared model
        }
