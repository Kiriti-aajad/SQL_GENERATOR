"""
Central configuration manager for Schema Searcher Agent.
This module loads file paths, environment variables, and constants
used across the schema retrieval system.

FIXED: Added XMLSchemaManager instantiation to resolve "XML schema manager not available" warning
UPDATED: Changed default embedding model to intfloat/e5-base-v2 for system-wide consistency.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)

# Load .env if it exists (optional)
env_path = Path(__file__).parent.parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Root project directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Define environment keys with fixed chromaDB path
_ENV_KEYS = {
    "SCHEMA_PATH": "data/metadata/schema.json",
    "XML_SCHEMA_PATH": "data/metadata/xml_schema.json",
    "JOINS_PATH": "data/metadata/joins_verified.json",
    "FAISS_INDEX_PATH": "data/embeddings/faiss/index.faiss",
    "FAISS_METADATA_PATH": "data/embeddings/faiss/metadata.pkl",
    "CHROMA_PERSIST_DIR": "data/embeddings/chromaDB",
    "LOG_DIR": "logs",
    "LOG_FILE": "schema_search.log"
}

# Paths
SCHEMA_PATH: Path = PROJECT_ROOT / os.getenv("SCHEMA_PATH", _ENV_KEYS["SCHEMA_PATH"])
XML_SCHEMA_PATH: Path = PROJECT_ROOT / os.getenv("XML_SCHEMA_PATH", _ENV_KEYS["XML_SCHEMA_PATH"])
JOINS_PATH: Path = PROJECT_ROOT / os.getenv("JOINS_PATH", _ENV_KEYS["JOINS_PATH"])
FAISS_INDEX_PATH: Path = PROJECT_ROOT / os.getenv("FAISS_INDEX_PATH", _ENV_KEYS["FAISS_INDEX_PATH"])
FAISS_METADATA_PATH: Path = PROJECT_ROOT / os.getenv("FAISS_METADATA_PATH", _ENV_KEYS["FAISS_METADATA_PATH"])
CHROMA_PERSIST_DIR: Path = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", _ENV_KEYS["CHROMA_PERSIST_DIR"])
LOG_DIR: Path = PROJECT_ROOT / os.getenv("LOG_DIR", _ENV_KEYS["LOG_DIR"])
LOG_FILE: Path = LOG_DIR / os.getenv("LOG_FILE", _ENV_KEYS["LOG_FILE"])

# Model/settings - UPDATED: Changed default to E5-base-v2 for system consistency
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/e5-base-v2")
LLM_SQL_MODEL: str = os.getenv("LLM_SQL_MODEL_NAME", "sqlcoder-7b")
LLM_ENDPOINT: str = os.getenv("LLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")

QUERY_TIMEOUT_SEC: int = int(os.getenv("QUERY_TIMEOUT_SEC", "45"))
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "25"))
MIN_CONFIDENCE_SCORE: float = float(os.getenv("MIN_CONFIDENCE", "0.15"))
MAX_RESULTS_TOP_K: int = int(os.getenv("MAX_RESULTS_TOP_K", "50"))
COMPREHENSIVE_TIMEOUT_SEC: int = int(os.getenv("COMPREHENSIVE_TIMEOUT_SEC", "60"))
AGGRESSIVE_CONFIDENCE: float = float(os.getenv("AGGRESSIVE_CONFIDENCE", "0.05"))
ENABLE_PARALLEL_EXECUTION: bool = os.getenv("ENABLE_PARALLEL_EXECUTION", "true").lower() == "true"
CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "7200"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))
ENGINE_TIMEOUT_SEC: int = int(os.getenv("ENGINE_TIMEOUT_SEC", "15"))
ENGINE_MAX_WORKERS: int = int(os.getenv("ENGINE_MAX_WORKERS", "5"))

RETRIEVAL_METHODS: List[str] = list(os.getenv(
    "RETRIEVAL_METHODS", "semantic,bm25,faiss,fuzzy,nlp"
).split(","))

# ENHANCED: Updated configuration presets with E5-aware settings
CONFIGURATION_PRESETS = {
    "fast": {
        "methods": ["bm25", "fuzzy"],
        "top_k": 15,
        "confidence_threshold": 0.3,
        "timeout": 20,
        "parallel_execution": True,
        "embedding_optimized": False
    },
    "balanced": {
        "methods": ["semantic", "bm25", "faiss"],
        "top_k": 25,
        "confidence_threshold": 0.2,
        "timeout": 35,
        "parallel_execution": True,
        "embedding_optimized": True
    },
    "comprehensive": {
        "methods": ["semantic", "bm25", "faiss", "fuzzy", "nlp"],
        "top_k": 40,
        "confidence_threshold": 0.1,
        "timeout": 60,
        "parallel_execution": True,
        "embedding_optimized": True
    },
    "maximum": {
        "methods": ["semantic", "bm25", "faiss", "fuzzy", "nlp"],
        "top_k": 50,
        "confidence_threshold": 0.05,
        "timeout": 90,
        "parallel_execution": True,
        "embedding_optimized": True
    }
}

# ENHANCED: Updated method settings with E5-specific optimizations
METHOD_SETTINGS = {
    "semantic": {
        "default_top_k": 20,
        "confidence_boost": 1.1,
        "timeout_multiplier": 1.2,
        "supports_e5_prefixes": True
    },
    "bm25": {
        "default_top_k": 30,
        "confidence_boost": 0.9,
        "timeout_multiplier": 0.8,
        "supports_e5_prefixes": False
    },
    "faiss": {
        "default_top_k": 25,
        "confidence_boost": 1.0,
        "timeout_multiplier": 1.0,
        "supports_e5_prefixes": False
    },
    "fuzzy": {
        "default_top_k": 20,
        "confidence_boost": 0.7,
        "timeout_multiplier": 0.9,
        "supports_e5_prefixes": False
    },
    "nlp": {
        "default_top_k": 15,
        "confidence_boost": 0.95,
        "timeout_multiplier": 1.1,
        "supports_e5_prefixes": False
    }
}

# ENHANCED: Updated quality settings for E5 model
QUALITY_SETTINGS = {
    "min_description_length": 10,
    "prefer_complete_xml": True,
    "boost_recent_results": True,
    "deduplicate_aggressively": False,
    "merge_similar_descriptions": True,
    "e5_similarity_threshold": 0.75,
    "enable_e5_prefixes": True,
    "semantic_boost_factor": 1.1
}

# CRITICAL FIX: Initialize XMLSchemaManager instance
_xml_schema_manager_instance = None

def _initialize_xml_schema_manager():
    """CRITICAL FIX: Initialize XMLSchemaManager for config access"""
    global _xml_schema_manager_instance
    
    if _xml_schema_manager_instance is None:
        try:
            # Import XMLSchemaManager with fallback paths
            try:
                from agent.schema_searcher.managers.xml_schema_manager import XMLSchemaManager
            except ImportError:
                try:
                    from ..managers.xml_schema_manager import XMLSchemaManager
                except ImportError:
                    logger.warning("XMLSchemaManager class not found")
                    return None
            
            # Create instance with XML schema path
            _xml_schema_manager_instance = XMLSchemaManager(str(XML_SCHEMA_PATH))
            
            if _xml_schema_manager_instance.is_available():
                logger.info("XMLSchemaManager initialized successfully in config")
            else:
                logger.warning("XMLSchemaManager initialized but not fully available")
            
        except Exception as e:
            logger.error(f"Failed to initialize XMLSchemaManager in config: {e}")
            _xml_schema_manager_instance = None
    
    return _xml_schema_manager_instance

# CRITICAL FIX: Config class with xml_schema_manager attribute
class SchemaSearcherConfig:
    """CRITICAL FIX: Configuration class with xml_schema_manager attribute"""
    
    def __init__(self):
        # All existing config attributes
        self.schema_path = SCHEMA_PATH
        self.xml_schema_path = XML_SCHEMA_PATH
        self.joins_path = JOINS_PATH
        self.faiss_index_path = FAISS_INDEX_PATH
        self.faiss_metadata_path = FAISS_METADATA_PATH
        self.chroma_persist_dir = CHROMA_PERSIST_DIR
        self.log_dir = LOG_DIR
        self.log_file = LOG_FILE
        
        # Model settings
        self.embedding_model_name = EMBEDDING_MODEL_NAME
        self.llm_sql_model = LLM_SQL_MODEL
        self.llm_endpoint = LLM_ENDPOINT
        
        # Performance settings
        self.query_timeout_sec = QUERY_TIMEOUT_SEC
        self.default_top_k = DEFAULT_TOP_K
        self.min_confidence_score = MIN_CONFIDENCE_SCORE
        self.max_results_top_k = MAX_RESULTS_TOP_K
        
        # Configuration presets
        self.configuration_presets = CONFIGURATION_PRESETS
        self.method_settings = METHOD_SETTINGS
        self.quality_settings = QUALITY_SETTINGS
        
        # CRITICAL FIX: Add xml_schema_manager attribute
        self.xml_schema_manager = _initialize_xml_schema_manager()
    
    def get_xml_schema_manager(self):
        """Get XMLSchemaManager instance"""
        return self.xml_schema_manager
    
    def is_xml_schema_manager_available(self) -> bool:
        """Check if XMLSchemaManager is available"""
        return (self.xml_schema_manager is not None and 
                hasattr(self.xml_schema_manager, 'is_available') and
                self.xml_schema_manager.is_available())

# CRITICAL FIX: Global config instance
_config_instance = None

def get_config() -> SchemaSearcherConfig:
    """CRITICAL FIX: Get singleton config instance with XMLSchemaManager"""
    global _config_instance
    
    if _config_instance is None:
        _config_instance = SchemaSearcherConfig()
        logger.info("Configuration instance created with XMLSchemaManager")
    
    return _config_instance

# Existing helper functions (unchanged)
def get_configuration_preset(preset_name: str) -> Dict[str, Any]:
    """Get configuration preset with E5 optimizations."""
    if preset_name not in CONFIGURATION_PRESETS:
        preset_name = "balanced"
    preset = CONFIGURATION_PRESETS[preset_name].copy()
    
    preset["embedding_model"] = EMBEDDING_MODEL_NAME
    preset["e5_optimized"] = "e5" in EMBEDDING_MODEL_NAME.lower()
    
    return preset

def get_method_settings(method: str) -> Dict[str, Any]:
    """Get method settings with E5 awareness."""
    settings = METHOD_SETTINGS.get(method.lower(), {
        "default_top_k": DEFAULT_TOP_K,
        "confidence_boost": 1.0,
        "timeout_multiplier": 1.0,
        "supports_e5_prefixes": False
    })
    
    settings["embedding_model"] = EMBEDDING_MODEL_NAME
    settings["e5_optimized"] = "e5" in EMBEDDING_MODEL_NAME.lower()
    
    return settings

def get_optimized_settings_for_engines(engine_count: int) -> Dict[str, Any]:
    """Get optimized settings based on engine count with E5 optimizations."""
    if engine_count >= 5:
        preset = get_configuration_preset("comprehensive")
    elif engine_count >= 3:
        preset = get_configuration_preset("balanced")
    elif engine_count >= 2:
        preset = get_configuration_preset("fast")
    else:
        preset = {
            "methods": RETRIEVAL_METHODS[:1],
            "top_k": MAX_RESULTS_TOP_K,
            "confidence_threshold": AGGRESSIVE_CONFIDENCE,
            "timeout": COMPREHENSIVE_TIMEOUT_SEC,
            "parallel_execution": False,
            "embedding_optimized": False
        }
    
    preset["embedding_model"] = EMBEDDING_MODEL_NAME
    preset["e5_optimized"] = "e5" in EMBEDDING_MODEL_NAME.lower()
    
    return preset

# NEW: E5-specific helper functions
def get_embedding_model_info() -> Dict[str, Any]:
    """Get comprehensive embedding model information."""
    return {
        "model_name": EMBEDDING_MODEL_NAME,
        "is_e5_model": "e5" in EMBEDDING_MODEL_NAME.lower(),
        "supports_prefixes": "e5" in EMBEDDING_MODEL_NAME.lower(),
        "default_similarity_threshold": QUALITY_SETTINGS.get("e5_similarity_threshold", 0.7),
        "recommended_confidence_boost": 1.1 if "e5" in EMBEDDING_MODEL_NAME.lower() else 1.0
    }

def is_e5_model_configured() -> bool:
    """Check if E5 model is configured."""
    return "e5" in EMBEDDING_MODEL_NAME.lower()

def get_e5_optimized_settings() -> Dict[str, Any]:
    """Get E5-optimized settings if E5 model is configured."""
    if is_e5_model_configured():
        return {
            "use_prefixes": True,
            "similarity_threshold": 0.75,
            "confidence_boost": 1.1,
            "enable_semantic_boost": True,
            "normalized_similarities": True
        }
    else:
        return {
            "use_prefixes": False,
            "similarity_threshold": 0.7,
            "confidence_boost": 1.0,
            "enable_semantic_boost": False,
            "normalized_similarities": False
        }

# CRITICAL FIX: Updated exports including new functions
__all__ = [
    "SCHEMA_PATH", "XML_SCHEMA_PATH", "JOINS_PATH",
    "FAISS_INDEX_PATH", "FAISS_METADATA_PATH", "CHROMA_PERSIST_DIR",
    "LOG_FILE", "PROJECT_ROOT",
    "EMBEDDING_MODEL_NAME", "LLM_SQL_MODEL", "LLM_ENDPOINT",
    "QUERY_TIMEOUT_SEC", "DEFAULT_TOP_K", "MIN_CONFIDENCE_SCORE", "RETRIEVAL_METHODS",
    "MAX_RESULTS_TOP_K", "COMPREHENSIVE_TIMEOUT_SEC", "AGGRESSIVE_CONFIDENCE",
    "ENABLE_PARALLEL_EXECUTION", "CACHE_TTL_SECONDS", "MAX_RETRIES",
    "ENGINE_TIMEOUT_SEC", "ENGINE_MAX_WORKERS",
    "CONFIGURATION_PRESETS", "METHOD_SETTINGS", "QUALITY_SETTINGS",
    "get_configuration_preset", "get_method_settings", "get_optimized_settings_for_engines",
    # E5-specific exports
    "get_embedding_model_info", "is_e5_model_configured", "get_e5_optimized_settings",
    # CRITICAL FIX: New exports for XML schema manager
    "SchemaSearcherConfig", "get_config"
]
