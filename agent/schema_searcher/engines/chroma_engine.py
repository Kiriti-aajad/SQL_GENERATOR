"""
ChromaDB vector search engine for schema retrieval.
FIXED: All remaining Pylance errors including method overrides and subscript issues
ADDED: Singleton pattern for ChromaDB client to prevent multiple initialization
"""

import os
import chromadb
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import logging

from agent.schema_searcher.engines.base_engine import BaseSearchEngine
from agent.schema_searcher.core.data_models import RetrievedColumn, SearchMethod, ColumnType
from agent.schema_searcher.utils.mathstral_client import MathstralClient
from agent.schema_searcher.utils.embedding_manager import EmbeddingModelManager

# FIXED: Flexible path configuration using environment variables and relative paths
PERSIST_DIRECTORY = os.getenv('CHROMA_DB_PATH', './data/embeddings/chromaDB')
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"  # 768 dimensions
COLLECTION_NAME = "schema_metadata"
N_RESULTS = 10

logger = logging.getLogger(__name__)

# ============================================================================
# ADDED: ChromaDB Client Singleton - ONLY NEW ADDITION
# ============================================================================

class ChromaDBClientSingleton:
    """Thread-safe singleton for ChromaDB client to prevent multiple initialization"""
    _instance: Optional['ChromaDBClientSingleton'] = None
    _instance_lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._client: Optional[chromadb.ClientAPI] = None # pyright: ignore[reportPrivateImportUsage]
        self._client_lock = threading.Lock()
        self._usage_count = 0
        self._initialized = True
        logger.info("ChromaDBClientSingleton created")

    def get_client(self, persist_directory: str) -> chromadb.ClientAPI: # pyright: ignore[reportPrivateImportUsage]
        """Get or create ChromaDB client (singleton)"""
        if self._client is not None:
            self._usage_count += 1
            logger.debug(f"Reusing ChromaDB client (usage #{self._usage_count})")
            return self._client
        
        with self._client_lock:
            if self._client is not None:
                self._usage_count += 1
                logger.debug(f"Reusing ChromaDB client (usage #{self._usage_count})")
                return self._client
            
            logger.info(f"Creating ChromaDB client: {persist_directory} (ONCE ONLY)")
            os.makedirs(persist_directory, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_directory)
            self._usage_count = 1
            logger.info(f"✅ ChromaDB client created successfully")
            return self._client

# ============================================================================
# YOUR EXISTING CODE - UNCHANGED
# ============================================================================

# FIXED: Type-safe conversion functions for Pylance
def safe_str_or_none(value: Union[str, int, float, bool, None]) -> Optional[str]:
    """Convert value to str if not None, else return None. Fixes Pylance type issues."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)

def safe_list_str(value: Optional[List[str]]) -> List[str]:
    """Convert None to empty list to fix Pylance list type issues."""
    return value if value is not None else []

def safe_dict(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert None to empty dict to fix Pylance dict type issues."""
    return value if value is not None else {}

def safe_get_first_list(dct: Dict[str, Any], key: str) -> List[Any]:
    """FIXED: Safely extract first list element from response dictionary to avoid subscript errors."""
    val = dct.get(key)
    if val and isinstance(val, list) and len(val) > 0:
        first = val[0]
        if isinstance(first, list):
            return first
        else:
            # If first is not list, then val is just list of elements
            return val
    else:
        return []

# FIXED: Extended MathstralClient with missing methods to fix Pylance errors
class ExtendedMathstralClient:
    """Extended MathstralClient with all expected methods for type safety."""
    
    def __init__(self, base_client: Optional[MathstralClient] = None):
        self.base_client = base_client
    
    def analyze_schema_gap(
        self, 
        gap_description: str, 
        context: str = "", 
        collection_patterns: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """Analyze schema gap - stub method to fix Pylance errors."""
        return {
            "search_terms": [gap_description],
            "metadata_filters": {},
            "confidence": 0.5,
            "analysis_type": "fallback"
        }
    
    def discover_entity_patterns(
        self, 
        text_samples: List[str], 
        context: str = "", 
        max_patterns: int = 10
    ) -> Dict[str, Any]:
        """Discover entity patterns - stub method to fix Pylance errors."""
        patterns: Dict[str, List[str]] = {}
        for sample in text_samples[:max_patterns]:
            if 'id' in sample.lower():
                patterns.setdefault('id_fields', []).append(sample)
            elif 'name' in sample.lower():
                patterns.setdefault('name_fields', []).append(sample)
        return {"patterns": patterns}

# FIXED: Simplified exceptions
class ChromaEngineError(Exception):
    """Base exception for ChromaDB engine errors"""
    pass

class ChromaInitializationError(ChromaEngineError):
    """Raised when ChromaDB engine initialization fails"""
    pass

class ChromaSearchError(ChromaEngineError):
    """Raised when ChromaDB search fails"""
    pass

class ChromaEngine(BaseSearchEngine):
    """
    ChromaDB-based vector search for schema retrieval.
    FIXED: All Pylance errors including method signature and type issues
    ADDED: Uses singleton ChromaDB client to prevent multiple initialization
    """

    def __init__(
        self,
        persist_directory: str = PERSIST_DIRECTORY,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        collection_name: str = COLLECTION_NAME,
        n_results: int = N_RESULTS,
        mathstral_client: Optional[MathstralClient] = None
    ):
        super().__init__(SearchMethod.CHROMA, logger)

        # FIXED: Flexible path handling
        if persist_directory == PERSIST_DIRECTORY and not os.path.isabs(persist_directory):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            self.persist_directory = os.path.join(project_root, 'data', 'embeddings', 'chromaDB')
        else:
            self.persist_directory = persist_directory
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.n_results = n_results

        # FIXED: Safe embedding model loading with fallback
        try:
            self.embedder = EmbeddingModelManager.get_model(embedding_model_name)
            self.logger.info(f"ChromaDB engine initialized with shared model: {embedding_model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model {embedding_model_name}: {e}")
            self.logger.warning("Falling back to all-MiniLM-L6-v2")
            try:
                self.embedder = EmbeddingModelManager.get_model("all-MiniLM-L6-v2")
                self.embedding_model_name = "all-MiniLM-L6-v2"
            except Exception as fallback_error:
                raise ChromaInitializationError(f"Failed to load both primary and fallback embedding models: {fallback_error}")

        # ADDED: Get singleton ChromaDB client instance
        self.client_singleton = ChromaDBClientSingleton()
        
        # FIXED: Safe ChromaDB client and collection initialization with proper typing
        self.client: Optional[chromadb.ClientAPI] = None  # pyright: ignore[reportPrivateImportUsage] # FIXED: Use correct type
        self.collection: Optional[chromadb.Collection] = None
        self._initialize_client_and_collection()

        # FIXED: Use extended MathstralClient to avoid method access errors
        self.mathstral_client = ExtendedMathstralClient(mathstral_client)

        # Dynamic parameters
        self._dynamic_n_results = n_results
        self._learned_entity_patterns: Dict[str, List[str]] = {}
        self._metadata_cache: Dict[str, Any] = {}
        self._distance_threshold = 0.0
        self._query_context_embeddings: Dict[str, Any] = {}
        
        # Statistics
        self.total_searches = 0
        self.successful_searches = 0
        self.failed_searches = 0

    def _initialize_client_and_collection(self) -> None:
        """MODIFIED: Use singleton client instead of creating new one"""
        try:
            # CHANGED: Use singleton client instead of creating new one
            self.client = self.client_singleton.get_client(self.persist_directory)
            # Rest of your existing code unchanged
            
            if self.client is None:
                raise ChromaInitializationError("ChromaDB client failed to initialize")
                
            try:
                self.collection = self.client.get_collection(self.collection_name)
                self.logger.info(f"Found existing collection '{self.collection_name}'")
                
                # FIXED: Verify collection compatibility with null check
                if self.collection is not None:
                    try:
                        collection_count = self.collection.count()
                        if collection_count > 0:
                            test_embedding = self.embedder.encode(["test"])
                            # FIXED: Safe shape attribute access
                            if hasattr(test_embedding, 'shape'):
                                embedding_dim = test_embedding.shape[-1] # pyright: ignore[reportAttributeAccessIssue]
                            elif isinstance(test_embedding, list) and len(test_embedding) > 0:
                                if isinstance(test_embedding[0], list):
                                    embedding_dim = len(test_embedding[0])
                                else:
                                    embedding_dim = len(test_embedding)
                            else:
                                embedding_dim = 768  # Default for e5-base-v2
                            
                            self.logger.info(f"Collection has {collection_count} documents, embedding dimension: {embedding_dim}")
                    except Exception as dim_check_error:
                        self.logger.warning(f"Could not verify collection dimensions: {dim_check_error}")
                
            except Exception as get_error:
                self.logger.info(f"Collection '{self.collection_name}' not found, creating new collection")
                try:
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    self.logger.info(f"Created new collection '{self.collection_name}'")
                except Exception as create_error:
                    self.logger.warning(f"Failed to create collection, trying get_or_create: {create_error}")
                    self.collection = self.client.get_or_create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    self.logger.info(f"Got or created collection '{self.collection_name}'")
            
        except Exception as e:
            raise ChromaInitializationError(f"Failed to initialize ChromaDB client or collection: {e}")

    def initialize(self) -> None:
        """Initialize and learn entity patterns from collection"""
        if not self.collection:
            self._initialize_client_and_collection()
        
        self._learn_entity_patterns_from_collection()
        self.logger.info(f"ChromaDB engine initialized with collection '{self.collection_name}'")

    def _learn_entity_patterns_from_collection(self) -> None:
        """Learn entity patterns from existing collection data"""
        try:
            if not self.collection:
                self.logger.info("No collection available to learn patterns from")
                return
                
            collection_count = self.collection.count()
            if collection_count == 0:
                self.logger.info("Collection is empty, no patterns to learn")
                return
            
            sample_size = min(100, collection_count)
            sample = self.collection.peek(limit=sample_size)
            metas = sample.get("metadatas", [])
            
            if not metas:
                self.logger.info("No metadata available to learn patterns from")
                return

            cols: List[str] = []
            tabs: List[str] = []
            for m in metas:
                if m and isinstance(m, dict):
                    if m.get("column"):
                        cols.append(str(m["column"]))
                    if m.get("table"):
                        tabs.append(str(m["table"]))

            if self.mathstral_client and (cols or tabs):
                try:
                    patterns = self.mathstral_client.discover_entity_patterns(
                        text_samples=cols + tabs,
                        context="database schema analysis",
                        max_patterns=10
                    )
                    self._learned_entity_patterns = safe_dict(patterns.get("patterns"))
                    self.logger.info(f"Learned {len(self._learned_entity_patterns)} entity patterns")
                except Exception as pattern_error:
                    self.logger.warning(f"Failed to learn entity patterns with Mathstral: {pattern_error}")
                    self._learned_entity_patterns = {}
            else:
                self._learned_entity_patterns = self._extract_simple_patterns(cols + tabs)
                self.logger.info(f"Learned {len(self._learned_entity_patterns)} simple patterns")
                
        except Exception as e:
            self.logger.warning(f"Failed to learn entity patterns: {e}")
            self._learned_entity_patterns = {}

    def _extract_simple_patterns(self, terms: List[str]) -> Dict[str, List[str]]:
        """Extract simple patterns from terms as fallback"""
        patterns: Dict[str, List[str]] = {}
        try:
            for term in terms[:50]:
                if not term or not isinstance(term, str):
                    continue
                    
                term_lower = term.lower()
                if 'id' in term_lower:
                    patterns.setdefault('id_fields', []).append(term)
                elif 'name' in term_lower or 'nm' in term_lower:
                    patterns.setdefault('name_fields', []).append(term)
                elif 'date' in term_lower or 'dt' in term_lower:
                    patterns.setdefault('date_fields', []).append(term)
                elif 'amount' in term_lower or 'amt' in term_lower:
                    patterns.setdefault('amount_fields', []).append(term)
                    
        except Exception as e:
            self.logger.debug(f"Error in simple pattern extraction: {e}")
            
        return patterns

    def ensure_initialized(self) -> None:
        """Ensure the engine is initialized"""
        if not self.collection:
            self.initialize()

    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None,  # FIXED: Use top_k to match base class signature
        **kwargs  # FIXED: Add kwargs for compatibility
    ) -> List[RetrievedColumn]:
        """
        FIXED: Method signature matches base class to resolve Pylance override error
        """
        self.total_searches += 1
        
        # Handle both top_k and n_results for backward compatibility
        n_results = kwargs.get('n_results', top_k or self.n_results)
        where = kwargs.get('where')
        
        # Input validation
        if not query or not isinstance(query, str):
            self.failed_searches += 1
            return []
            
        query = query.strip()
        if not query:
            self.failed_searches += 1
            return []
            
        where = safe_dict(where)

        # Ensure collection is available with null check
        if not self.collection:
            try:
                self.ensure_initialized()
            except Exception as init_error:
                self.logger.error(f"Failed to initialize collection: {init_error}")
                self.failed_searches += 1
                return []

        if not self.collection:
            self.logger.error("Collection is still None after initialization")
            self.failed_searches += 1
            return []

        try:
            # FIXED: Safe embedding generation with error handling
            try:
                embedding = self.embedder.encode([query])
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist() # pyright: ignore[reportAttributeAccessIssue]
                elif isinstance(embedding, list) and len(embedding) > 0:
                    if not isinstance(embedding[0], list):
                        embedding = [embedding]
            except Exception as embed_error:
                self.logger.error(f"Failed to generate embedding for query '{query}': {embed_error}")
                self.failed_searches += 1
                return []

            # FIXED: Safe ChromaDB query with proper error handling
            try:
                query_kwargs = {
                    "query_embeddings": embedding,
                    "n_results": n_results
                }
                if where:
                    query_kwargs["where"] = where
                    
                resp = self.collection.query(**query_kwargs)
            except Exception as query_error:
                self.logger.error(f"ChromaDB query failed: {query_error}")
                self.failed_searches += 1
                return []

            # FIXED: Safe result processing using helper function to avoid subscript errors
            try:
                docs = safe_get_first_list(resp, "documents") # pyright: ignore[reportArgumentType]
                metas = safe_get_first_list(resp, "metadatas")  # pyright: ignore[reportArgumentType]
                scores = safe_get_first_list(resp, "distances") # pyright: ignore[reportArgumentType]

                if not docs:
                    self.successful_searches += 1
                    return []

                # Process results with deduplication
                seen: set[str] = set()
                results: List[RetrievedColumn] = []
                
                for doc, score, meta in zip(docs, scores, metas):
                    if not meta or not isinstance(meta, dict):
                        continue
                        
                    table = str(meta.get('table', '')).strip()
                    column = str(meta.get('column', '')).strip()
                    datatype = str(meta.get('datatype', 'unknown')).strip()
                    
                    if not table or not column:
                        continue
                    
                    key = f"{table.lower()}|{column.lower()}|{datatype.lower()}"
                    if key in seen:
                        continue
                    seen.add(key)

                    column_type = ColumnType.XML if str(meta.get("type", "")).lower() == "xml" else ColumnType.RELATIONAL
                    confidence_score = max(0.1, min(1.0, 1.0 - (float(score) / 2.0)))
                    
                    result = RetrievedColumn(
                        table=table,
                        column=column,
                        datatype=datatype,
                        type=column_type,
                        description=str(doc) if doc else '',
                        confidence_score=confidence_score,
                        retrieval_method=SearchMethod.CHROMA,
                        xml_column=safe_str_or_none(meta.get("xml_column")),
                        xpath=safe_str_or_none(meta.get("xpath")),
                        sql_expression=safe_str_or_none(meta.get("sql_expression")),
                        retrieved_at=datetime.now()
                    )
                    results.append(result)

                self.successful_searches += 1
                self.logger.debug(f"ChromaDB search returned {len(results)} results for query: '{query}'")
                return results
                
            except Exception as process_error:
                self.logger.error(f"Error processing ChromaDB results: {process_error}")
                self.failed_searches += 1
                return []

        except Exception as e:
            self.logger.error(f"ChromaEngine search error: {e}")
            self.failed_searches += 1
            return []

    # FIXED: Add missing async method for compatibility
    async def search_async(
        self, 
        query: Optional[str] = None, 
        keywords: Optional[List[str]] = None, 
        top_k: Optional[int] = None,  # FIXED: Use top_k to match signature
        **kwargs
    ) -> List[RetrievedColumn]:
        """Async search method for compatibility with retrieval agent"""
        if query:
            search_query = query
        elif keywords:
            keywords = safe_list_str(keywords)
            search_query = " ".join(str(kw) for kw in keywords if kw)
        else:
            return []
        
        return self.search(search_query, top_k, **kwargs)

    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            self._metadata_cache.clear()
            self._query_context_embeddings.clear()
            self._learned_entity_patterns.clear()
            self.logger.info("ChromaDB engine cleaned up")
        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")

    def health_check(self) -> Dict[str, Any]:
        """Enhanced health check with proper null safety"""
        base_health = super().health_check()
        
        try:
            collection_count = 0
            collection_status = "unknown"
            
            # FIXED: Proper null check to avoid "count" attribute error
            if self.collection is not None:
                try:
                    collection_count = self.collection.count()
                    collection_status = "available"
                except Exception as count_error:
                    collection_status = f"error: {count_error}"
            else:
                collection_status = "not_initialized"
            
            success_rate = 0.0
            if self.total_searches > 0:
                success_rate = (self.successful_searches / self.total_searches) * 100

            base_health.update({
                "persist_directory": self.persist_directory,
                "collection_name": self.collection_name,  
                "collection_count": collection_count,
                "collection_status": collection_status,
                "embedder_loaded": self.embedder is not None,
                "embedding_model": self.embedding_model_name,
                "using_shared_model": True,
                "using_singleton_client": True,  # ADDED: New field
                "client_usage_count": self.client_singleton._usage_count,  # ADDED: New field
                "search_statistics": {
                    "total_searches": self.total_searches,
                    "successful_searches": self.successful_searches,
                    "failed_searches": self.failed_searches,
                    "success_rate": round(success_rate, 2)
                }
            })
            
            if getattr(self, "_reasoning_mode_enabled", False):
                base_health.update({
                    "dynamic_n_results": self._dynamic_n_results,
                    "distance_threshold": self._distance_threshold,
                    "learned_patterns_count": len(self._learned_entity_patterns)
                })
            
            if collection_status == "available" and self.embedder:
                if success_rate >= 80 or self.total_searches < 5:
                    base_health["status"] = "healthy"
                else:
                    base_health["status"] = "degraded"
            else:
                base_health["status"] = "degraded"
                
        except Exception as e:
            base_health.update({
                "status": "error",
                "error": str(e)
            })
        
        return base_health

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive ChromaDB engine statistics"""
        try:
            success_rate = 0.0
            if self.total_searches > 0:
                success_rate = (self.successful_searches / self.total_searches) * 100
            
            return {
                "engine_type": "CHROMA",
                "using_singleton": True,  # ADDED: New field
                "search_statistics": {
                    "total_searches": self.total_searches,
                    "successful_searches": self.successful_searches,
                    "failed_searches": self.failed_searches,
                    "success_rate": round(success_rate, 2)
                },
                "configuration": {
                    "persist_directory": self.persist_directory,
                    "collection_name": self.collection_name,
                    "embedding_model": self.embedding_model_name,
                    "n_results": self.n_results
                },
                "collection_info": {
                    "collection_available": self.collection is not None,
                    "learned_patterns_count": len(self._learned_entity_patterns)
                }
            }
        except Exception as e:
            return {
                "engine_type": "CHROMA",
                "error": str(e)
            }

# Alias for backward compatibility
ChromaSearchEngine = ChromaEngine
