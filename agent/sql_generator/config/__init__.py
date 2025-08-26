"""
Configuration Package - Model configurations and settings for NGROK-enabled SQL generator
"""

from .model_configs import (
    ModelConfigs,
    ModelEndpoint, 
    ModelCapabilities,
    ModelType,
    QueryComplexity,
    GenerationStrategy,
    orchestrator_integration,
    model_configs
)

# Export all configuration classes
__all__ = [
    "ModelConfigs",
    "ModelEndpoint",
    "ModelCapabilities", 
    "ModelType",
    "QueryComplexity",
    "GenerationStrategy",
    "orchestrator_integration",
    "model_configs",
    "get_config",
    "get_default_config",
    "validate_config"
]

# Default configuration instance
default_config = model_configs

def get_default_config():
    """Get default model configuration"""
    return default_config

def get_config():
    """Get configuration (for backwards compatibility)"""
    return default_config

def validate_config():
    """Validate the default configuration"""
    return default_config.validate_configuration()

# Default configuration dictionary (for components that expect dict format)
DEFAULT_CONFIG = {
    "models": {
        "deepseek": {
            "name": "DeepSeek-R1-Distill-Qwen-7B",
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "endpoint": "http://localhost:8000"
        },
        "mathstral": {
            "name": "Mathstral-7B-v0.1",
            "model_id": "mistralai/Mathstral-7B-v0.1", 
            "endpoint": "http://localhost:8000"
        }
    },
    "timeouts": {
        "request": 30,
        "connection": 10
    },
    "ngrok": {
        "enabled": True,
        "base_url": "https://8f15221a7ceb.ngrok-free.app"
    }
}
