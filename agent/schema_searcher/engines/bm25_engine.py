"""
BM25 keyword-based search engine for schema retrieval.
FIXED: All HTML entity encoding, validation issues, and search logic problems
"""

from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Optional
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
class BM25EngineError(Exception):
    """Base exception for BM25 engine errors"""
    pass

class BM25InputValidationError(BM25EngineError):
    """Raised when input validation fails"""
    pass

class BM25IndexError(BM25EngineError):
    """Raised when BM25 index is not properly initialized"""
    pass

class BM25SearchError(BM25EngineError):
    """Raised when BM25 search fails"""
    pass

class BM25SearchEngine(BaseSearchEngine):
    """
    BM25 keyword-based search engine for schema retrieval.
    FIXED: All HTML encoding issues, validation problems, and search logic
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        super().__init__(SearchMethod.BM25, logger)
        
        # FIXED: Better parameter validation
        if k1 <= 0 or k1 > 5:
            raise BM25EngineError(f"Invalid k1 parameter {k1}, must be between 0 and 5")
        if b < 0 or b > 1:
            raise BM25EngineError(f"Invalid b parameter {b}, must be between 0 and 1")
        
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.table_index: Dict[str, List[int]] = {}
        self.column_index: Dict[str, List[int]] = {}
        
        # Statistics
        self.total_searches = 0
        self.successful_searches = 0
        self.failed_searches = 0

    def initialize(self) -> None:
        """Initialize BM25 index from schema and XML metadata"""
        self.logger.info("Building BM25 index from schema metadata")
        try:
            schema_loader = SchemaLoader()
            xml_loader = XMLLoader()
            relational_data = schema_loader.load()
            raw_xml_data = xml_loader.load()
            
            # FIXED: Better validation
            if not relational_data:
                self.logger.warning("No relational schema data loaded")
                relational_data = []
            
            xml_data = self._flatten_xml_data(raw_xml_data)
            self.metadata = relational_data + xml_data
            
            if not self.metadata:
                raise BM25IndexError("No metadata available after loading schema and XML data")
            
            self._build_lookup_indices()
            self._build_corpus()
            self._build_index()
            
            if not self.bm25:
                raise BM25IndexError("BM25 index creation failed")
            
            self.logger.info(f"BM25 index built with {len(self.metadata)} documents, "
                             f"{len(self.table_index)} unique tables, {len(self.column_index)} unique columns")
        except (BM25IndexError, BM25EngineError):
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize BM25 engine: {e}")
            raise BM25IndexError(f"BM25 engine initialization failed: {e}")

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
                    self.logger.debug("Skipping XML table with missing table name")
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
            except Exception as e:
                self.logger.debug(f"Error processing XML table data: {e}")
                continue
                
        return flattened

    def _build_lookup_indices(self) -> None:
        """Build lookup indices with validation"""
        self.table_index.clear()
        self.column_index.clear()
        
        for idx, doc in enumerate(self.metadata):
            try:
                table_name = doc.get('table', '').lower()
                if table_name:
                    if table_name not in self.table_index:
                        self.table_index[table_name] = []
                    self.table_index[table_name].append(idx)
                
                column_name = (doc.get('column', '') or doc.get('name', '')).lower()
                if column_name:
                    if column_name not in self.column_index:
                        self.column_index[column_name] = []
                    self.column_index[column_name].append(idx)
            except Exception as e:
                self.logger.debug(f"Error building index for document {idx}: {e}")
                continue

    def _build_corpus(self) -> None:
        """Build corpus with validation"""
        self.corpus = []
        for doc in self.metadata:
            try:
                text_parts = [
                    doc.get('table', ''),
                    doc.get('column', ''),
                    doc.get('datatype', ''),
                    doc.get('description', ''),
                ]
                if doc.get('type') == 'xml':
                    text_parts.extend([
                        doc.get('xml_column', ''),
                        doc.get('xpath', ''),
                        "xml xml_field",
                    ])
                text = ' '.join(filter(None, text_parts)).lower()
                if text.strip():
                    self.corpus.append(text)
                else:
                    self.corpus.append("empty_document")  # Placeholder for empty docs
            except Exception as e:
                self.logger.debug(f"Error building corpus entry: {e}")
                self.corpus.append("error_document")  # Placeholder for error docs
        
        if not self.corpus:
            raise BM25IndexError("Cannot build corpus - no valid text content found")

    def _build_index(self) -> None:
        """Build BM25 index with validation"""
        if not self.corpus:
            raise BM25IndexError("Cannot build BM25 index - corpus is empty")
        
        try:
            tokenized_corpus = [doc.split() for doc in self.corpus]
            
            # Ensure we have valid tokens
            if not tokenized_corpus or all(not tokens for tokens in tokenized_corpus):
                raise BM25IndexError("Cannot build BM25 index - no valid tokens found")
            
            self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
            self.logger.debug(f"BM25 index built with {len(tokenized_corpus)} tokenized docs")
        except Exception as e:
            raise BM25IndexError(f"BM25 index creation failed: {e}")

    @track_execution_time
    def search(self, query: str, top_k: int = 25) -> List[RetrievedColumn]:
        """
        BM25 search with FIXED validation and logic
        """
        self.total_searches += 1
        
        # FIXED: Better input validation
        if not self.bm25:
            self.failed_searches += 1
            raise BM25IndexError("BM25 engine not initialized - call initialize() first")
        
        if not query or not isinstance(query, str):
            self.failed_searches += 1
            raise BM25InputValidationError("Query must be a non-empty string")
        
        if top_k <= 0:
            self.failed_searches += 1
            raise BM25InputValidationError("top_k must be positive")
        
        # FIXED: More lenient query processing
        processed_query = query.strip()
        if not processed_query:
            self.failed_searches += 1
            raise BM25InputValidationError("Query cannot be empty or only whitespace")

        try:
            # FIXED: Better tokenization
            query_tokens = processed_query.lower().split()
            
            if not query_tokens:
                self.failed_searches += 1
                raise BM25InputValidationError("No valid tokens found in query")
            
            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)
            
            # FIXED: Better result processing
            scored_indices = [(i, score) for i, score in enumerate(scores) if score > 0]
            scored_indices.sort(key=lambda x: x[1], reverse=True)  # Sort by score
            
            results = []
            seen_combinations = set()

            for idx, score in scored_indices:
                if idx >= len(self.metadata):
                    continue
                    
                try:
                    doc = self.metadata[idx]
                    table_name = doc.get('table', '')
                    column_name = doc.get('column', '') or doc.get('name', '')
                    
                    if not table_name or not column_name:
                        continue
                    
                    # FIXED: Proper deduplication
                    combination_key = f"{table_name.lower()}.{column_name.lower()}"
                    if combination_key in seen_combinations:
                        continue
                    seen_combinations.add(combination_key)
                    
                    # Create result with proper confidence score
                    confidence_score = min(1.0, max(0.1, score / 10.0))  # Normalize to 0.1-1.0
                    result = self._create_retrieved_column_from_doc(doc, confidence_score)
                    
                    if result:
                        results.append(result)
                        if len(results) >= top_k:
                            break
                        
                except Exception as e:
                    self.logger.debug(f"Error processing result at index {idx}: {e}")
                    continue

            # FIXED: Always return what we have, don't fail if no results
            self.successful_searches += 1
            self.logger.debug(f"BM25 search returned {len(results)} results for query: '{query}'")
            return results
            
        except (BM25InputValidationError, BM25IndexError):
            raise
        except Exception as e:
            self.failed_searches += 1
            raise BM25SearchError(f"BM25 search failed for query '{query}': {e}")

    def _create_retrieved_column_from_doc(self, doc: Dict[str, Any], confidence: float = 1.0) -> Optional[RetrievedColumn]:
        """Create RetrievedColumn from document with proper validation"""
        try:
            table_name = doc.get('table', '')
            column_name = doc.get('column', '') or doc.get('name', '')
            
            if not table_name or not column_name:
                return None
            
            # Validate and normalize confidence
            confidence = max(0.0, min(1.0, confidence))
            
            # Determine column type
            column_type = ColumnType.XML if doc.get('type') == 'xml' else ColumnType.RELATIONAL
            
            # For XML columns, ensure we have required data
            if column_type == ColumnType.XML:
                xml_column = doc.get('xml_column', '').strip()
                if not xml_column:
                    column_type = ColumnType.RELATIONAL
            
            # Create appropriate RetrievedColumn
            if column_type == ColumnType.XML and doc.get('xml_column'):
                return RetrievedColumn(
                    table=table_name,
                    column=column_name,
                    datatype='xml_field',
                    type=column_type,
                    description=doc.get('description', ''),
                    confidence_score=confidence,
                    retrieval_method=SearchMethod.BM25,
                    xml_column=doc.get('xml_column', ''),
                    xpath=doc.get('xpath', ''),
                    sql_expression=doc.get('sql_expression', ''),
                    retrieved_at=datetime.now()
                )
            else:
                return RetrievedColumn(
                    table=table_name,
                    column=column_name,
                    datatype=doc.get('datatype', 'unknown'),
                    type=ColumnType.RELATIONAL,
                    description=doc.get('description', ''),
                    confidence_score=confidence,
                    retrieval_method=SearchMethod.BM25,
                    retrieved_at=datetime.now()
                )
        except Exception as e:
            self.logger.debug(f"Error creating retrieved column: {e}")
            return None

    # FIXED: Add missing search_async method for compatibility
    async def search_async(self, query: str = None, keywords: List[str] = None, top_k: int = 25) -> List[RetrievedColumn]: # pyright: ignore[reportArgumentType]
        """Async search method for compatibility with retrieval agent"""
        # Handle both query and keywords parameters
        if query:
            search_query = query
        elif keywords:
            search_query = " ".join(keywords)
        else:
            raise BM25InputValidationError("Either query or keywords must be provided")
        
        # BM25 search is synchronous, so just call the regular search method
        return self.search(search_query, top_k)

    def ensure_initialized(self) -> None:
        """Ensure the engine is initialized"""
        if not self.bm25:
            self.initialize()

    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            self.bm25 = None
            self.corpus.clear()
            self.metadata.clear()
            self.table_index.clear()
            self.column_index.clear()
            self.logger.debug("BM25 search engine cleaned up")
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
                "index_built": self.bm25 is not None,
                "corpus_size": len(self.corpus),
                "metadata_count": len(self.metadata),
                "unique_tables": len(self.table_index),
                "unique_columns": len(self.column_index),
                "search_statistics": {
                    "total_searches": self.total_searches,
                    "successful_searches": self.successful_searches,
                    "failed_searches": self.failed_searches,
                    "success_rate": round(success_rate, 2)
                },
                "parameters": {
                    "k1": self.k1,
                    "b": self.b
                }
            })
            
            # Determine overall status
            if not self.bm25 or not self.corpus:
                base_health["status"] = "critical"
            elif success_rate < 50 and self.total_searches > 5:
                base_health["status"] = "degraded"
            else:
                base_health["status"] = "healthy"
            
            return base_health
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "engine_type": "BM25"
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive BM25 engine statistics"""
        try:
            success_rate = 0.0
            if self.total_searches > 0:
                success_rate = (self.successful_searches / self.total_searches) * 100
            
            return {
                "engine_type": "BM25",
                "search_statistics": {
                    "total_searches": self.total_searches,
                    "successful_searches": self.successful_searches,
                    "failed_searches": self.failed_searches,
                    "success_rate": round(success_rate, 2)
                },
                "index_statistics": {
                    "corpus_size": len(self.corpus),
                    "metadata_count": len(self.metadata),
                    "unique_tables": len(self.table_index),
                    "unique_columns": len(self.column_index),
                    "index_initialized": self.bm25 is not None
                },
                "parameters": {
                    "k1": self.k1,
                    "b": self.b
                }
            }
        except Exception as e:
            return {
                "engine_type": "BM25",
                "error": str(e)
            }
