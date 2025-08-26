"""
SQL Generator Package - Multi-model SQL generation with NGROK support
Main package initialization for the SQL generator system
FIXED: Added missing SQLGenerator import and simplified imports
"""

# FIXED: Import SQLGenerator for get_default_generator()
from agent.sql_generator.generator import SQLGenerator

# FIXED: Simplified config imports (we've already corrected this file)
from .config.model_configs import ModelConfigs, ModelType, GenerationStrategy

# FIXED: Simplified models imports - combine related imports
from .models.request_models import (
    SQLGenerationRequest, BatchSQLRequest, 
    create_simple_request, create_analytical_request, 
    create_xml_request, create_ngrok_optimized_request
)
from .models.response_models import (
    SQLGenerationResponse, ResponseStatus, 
    create_success_response, create_error_response
)

# Package metadata
__version__ = "1.0.0"
__author__ = "SQL Generator Team"
__description__ = "Multi-model SQL generation system with NGROK support"

# FIXED: Export main classes and functions (added SQLGenerator)
__all__ = [
    # Core generator
    "SQLGenerator",  # ADDED: Export the main generator class
    
    # Configuration
    "ModelConfigs",
    "ModelType", 
    "GenerationStrategy",
    
    # Request models
    "SQLGenerationRequest",
    "BatchSQLRequest",
    "create_simple_request",
    "create_analytical_request", 
    "create_xml_request",
    "create_ngrok_optimized_request",
    
    # Response models
    "SQLGenerationResponse",
    "ResponseStatus",
    "create_success_response",
    "create_error_response",
    
    # Helper functions
    "get_default_generator",
    "get_version_info"
]

# Package-level configuration
DEFAULT_CONFIG = ModelConfigs()

def get_default_generator():
    """Get a default SQL generator instance"""
    return SQLGenerator(DEFAULT_CONFIG)  # ✅ Now works with import

# Version info
def get_version_info():
    """Get detailed version information"""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "ngrok_support": True,
        "xml_support": True,
        "ensemble_support": True
    }
