"""
Enhanced Server Configuration with Environment Variable Support
Compatible with Complete Integrated Hybrid AI Agent Orchestrator
Production-ready with backwards compatibility
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class ServerConfig(BaseSettings):
    # Server settings
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    WORKERS: int = Field(default=1, description="Number of worker processes")
    RELOAD: bool = Field(default=False, description="Enable auto-reload for development")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # Security settings
    ALLOWED_ORIGINS: List[str] = Field(default=["*"], description="CORS allowed origins")
    
    # Enhanced Hybrid Orchestrator Settings
    MAX_CONCURRENT_REQUESTS: int = Field(default=10, description="Max concurrent requests")
    DEFAULT_TIMEOUT: int = Field(default=180, description="Default request timeout in seconds")
    
    # Component-specific timeouts
    NLP_TIMEOUT: float = Field(default=30.0, description="NLP processing timeout")
    SCHEMA_TIMEOUT: float = Field(default=25.0, description="Schema search timeout") 
    PROMPT_TIMEOUT: float = Field(default=20.0, description="Prompt building timeout")
    SQL_TIMEOUT: float = Field(default=35.0, description="SQL generation timeout")
    
    # Advanced Caching Configuration
    ENABLE_CACHING: bool = Field(default=True, description="Enable result caching")
    CACHE_TTL_SECONDS: int = Field(default=3600, description="Cache TTL")
    MAX_CACHE_SIZE: int = Field(default=1000, description="Maximum cache size")
    
    # Performance Monitoring
    ENABLE_PERFORMANCE_MONITORING: bool = Field(default=True, description="Enable performance monitoring")
    ENABLE_BACKGROUND_MONITORING: bool = Field(default=False, description="Enable background health checks")
    
    # Retry and Fallback Settings
    ENABLE_GRACEFUL_FALLBACKS: bool = Field(default=True, description="Enable component fallbacks")
    RETRY_ATTEMPTS: int = Field(default=2, description="Number of retry attempts")
    RETRY_DELAY: float = Field(default=1.0, description="Delay between retries")
    
    # API Keys (from environment) - These are the actual API keys used by the tools
    MISTRAL_API_KEY: Optional[str] = Field(default=None, description="Mistral API key")
    DEEPSEEK_API_KEY: Optional[str] = Field(default=None, description="DeepSeek API key") 
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    MATHSTRAL_API_KEY: Optional[str] = Field(default=None, description="Mathstral API key")
    
    # Database/Metadata Settings
    METADATA_SOURCE: str = Field(default="local", description="Metadata source")
    DATABASE_URL: Optional[str] = Field(default=None, description="Database connection URL")
    METADATA_CACHE_TTL: int = Field(default=7200, description="Metadata cache TTL")
    
    # Logging Configuration
    LOG_DIR: str = Field(default="logs", description="Log directory path")
    LOG_FILE: str = Field(default="server.log", description="Log file name")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    ENABLE_FILE_LOGGING: bool = Field(default=True, description="Enable file logging")
    ENABLE_CONSOLE_LOGGING: bool = Field(default=True, description="Enable console logging")
    
    # File Watcher Settings
    RELOAD_EXCLUDES: List[str] = Field(
        default=["*.log", "logs/*", "__pycache__/*", "*.pyc", ".git/*"],
        description="Files/patterns to exclude from file watcher"
    )
    
    # Processing Mode Settings
    DEFAULT_PROCESSING_MODE: str = Field(
        default="comprehensive",
        description="Default processing mode"
    )
    ENABLE_ADAPTIVE_MODE: bool = Field(default=True, description="Enable adaptive processing mode")
    
    # Component Discovery Settings
    COMPONENT_DISCOVERY_PATHS: List[str] = Field(
        default=[
            "orchestrator",
            "agent.nlp_processor.orchestration",
            "agent.schema_searcher.core", 
            "agent.prompt_builder",
            "agent.sql_generator"
        ],
        description="Component discovery paths"
    )
    
    # Health Check Settings
    HEALTH_CHECK_TIMEOUT: float = Field(default=5.0, description="Health check timeout")
    COMPONENT_HEALTH_CACHE_TTL: int = Field(default=60, description="Component health cache TTL")
    
    # Memory Management
    MAX_MEMORY_MB: int = Field(default=2048, description="Maximum memory usage in MB")
    ENABLE_MEMORY_MONITORING: bool = Field(default=True, description="Enable memory monitoring")
    
    # Request Processing
    MAX_QUERY_LENGTH: int = Field(default=2000, description="Maximum query length")
    MAX_BATCH_SIZE: int = Field(default=10, description="Maximum batch processing size")
    ENABLE_REQUEST_VALIDATION: bool = Field(default=True, description="Enable request validation")
    
    # Development/Debug Settings
    DEBUG_MODE: bool = Field(default=False, description="Enable debug mode")
    ENABLE_COMPONENT_DEBUGGING: bool = Field(default=False, description="Enable component debugging")
    MOCK_UNAVAILABLE_COMPONENTS: bool = Field(default=True, description="Mock unavailable components")
    
    # Field validators for Pydantic v2 compatibility
    @field_validator("ALLOWED_ORIGINS", mode="before")  
    @classmethod
    def split_allowed_origins(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return ["*"]
        if isinstance(v, str):
            origins = [origin.strip() for origin in v.split(",") if origin.strip()]
            return origins if origins else ["*"]
        if isinstance(v, list):
            return v
        return ["*"]

    @field_validator("RELOAD_EXCLUDES", mode="before")
    @classmethod  
    def split_reload_excludes(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return ["*.log", "logs/*", "__pycache__/*", "*.pyc", ".git/*"]
        if isinstance(v, str):
            excludes = [pattern.strip() for pattern in v.split(",") if pattern.strip()]
            return excludes if excludes else ["*.log", "logs/*", "__pycache__/*", "*.pyc", ".git/*"]
        if isinstance(v, list):
            return v
        return ["*.log", "logs/*", "__pycache__/*", "*.pyc", ".git/*"]

    @field_validator("COMPONENT_DISCOVERY_PATHS", mode="before")
    @classmethod
    def split_component_paths(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return ["orchestrator", "agent.nlp_processor.orchestration", "agent.schema_searcher.core", "agent.prompt_builder", "agent.sql_generator"]
        if isinstance(v, str):
            paths = [path.strip() for path in v.split(",") if path.strip()]
            return paths if paths else ["orchestrator", "agent.nlp_processor.orchestration", "agent.schema_searcher.core", "agent.prompt_builder", "agent.sql_generator"]
        if isinstance(v, list):
            return v
        return ["orchestrator", "agent.nlp_processor.orchestration", "agent.schema_searcher.core", "agent.prompt_builder", "agent.sql_generator"]
    
    @field_validator('LOG_LEVEL')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f'LOG_LEVEL must be one of {valid_levels}')
        return v_upper
    
    @field_validator('DEFAULT_PROCESSING_MODE')
    @classmethod
    def validate_processing_mode(cls, v: str) -> str:
        valid_modes = ['basic', 'standard', 'comprehensive', 'adaptive']
        v_lower = v.lower()
        if v_lower not in valid_modes:
            raise ValueError(f'DEFAULT_PROCESSING_MODE must be one of {valid_modes}')
        return v_lower
    
    @field_validator('MAX_CONCURRENT_REQUESTS')
    @classmethod
    def validate_concurrent_requests(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError('MAX_CONCURRENT_REQUESTS must be between 1 and 100')
        return v
    
    @field_validator('PORT')
    @classmethod
    def validate_port(cls, v: int) -> int:
        if v < 1 or v > 65535:
            raise ValueError('PORT must be between 1 and 65535')
        return v
    
    def get_hybrid_config(self) -> Dict[str, Any]:
        """Get configuration for HybridAIAgentOrchestrator"""
        return {
            # Component timeouts
            "nlp_timeout": self.NLP_TIMEOUT,
            "schema_timeout": self.SCHEMA_TIMEOUT, 
            "prompt_timeout": self.PROMPT_TIMEOUT,
            "sql_timeout": self.SQL_TIMEOUT,
            
            # Caching configuration
            "enable_caching": self.ENABLE_CACHING,
            "cache_ttl": self.CACHE_TTL_SECONDS,
            "max_cache_size": self.MAX_CACHE_SIZE,
            
            # Performance settings
            "max_concurrent_requests": self.MAX_CONCURRENT_REQUESTS,
            "enable_performance_monitoring": self.ENABLE_PERFORMANCE_MONITORING,
            "enable_background_monitoring": self.ENABLE_BACKGROUND_MONITORING,
            
            # Fallback settings
            "enable_graceful_fallbacks": self.ENABLE_GRACEFUL_FALLBACKS,
            "retry_attempts": self.RETRY_ATTEMPTS,
            "retry_delay": self.RETRY_DELAY
        }
    
    def get_api_keys(self) -> Dict[str, Optional[str]]:
        """Get available API keys for LLM tools"""
        return {
            "mistral": self.MISTRAL_API_KEY,
            "deepseek": self.DEEPSEEK_API_KEY, 
            "openai": self.OPENAI_API_KEY,
            "mathstral": self.MATHSTRAL_API_KEY
        }
    
    def get_uvicorn_config(self) -> Dict[str, Any]:
        """Get uvicorn-specific configuration"""
        config = {
            "host": self.HOST,
            "port": self.PORT,
            "workers": self.WORKERS,
            "reload": self.RELOAD,
            "log_level": self.LOG_LEVEL.lower(),
            "access_log": True
        }
        
        # Only add reload_excludes if reload is enabled and excludes exist
        if self.RELOAD and self.RELOAD_EXCLUDES:
            config["reload_excludes"] = self.RELOAD_EXCLUDES
            
        return config
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return not self.DEBUG_MODE and not self.RELOAD and self.WORKERS > 1
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            "level": self.LOG_LEVEL,
            "format": self.LOG_FORMAT,
            "log_dir": self.LOG_DIR,
            "log_file": self.LOG_FILE,
            "enable_file": self.ENABLE_FILE_LOGGING,
            "enable_console": self.ENABLE_CONSOLE_LOGGING
        }
    
    # Backwards compatibility properties for main.py
    @property
    def logging(self):
        """Backwards compatible logging object"""
        class LoggingCompat:
            def __init__(self, config):
                self.log_level = config.LOG_LEVEL
                self.log_file = config.LOG_FILE
                self.console_logging = config.ENABLE_CONSOLE_LOGGING
        return LoggingCompat(self)
    
    @property 
    def fastapi_server(self):
        """Backwards compatible FastAPI server object"""
        class FastAPICompat:
            def __init__(self, config):
                self.host = config.HOST
                self.port = config.PORT
                self.reload = config.RELOAD
                self.allowed_origins = config.ALLOWED_ORIGINS
                self.cors_enabled = True
                self.docs_enabled = not config.is_production()
        return FastAPICompat(self)
    
    @property
    def database(self):
        """Backwards compatible database object"""
        class DatabaseCompat:
            def __init__(self, config):
                self.connection_string = config.DATABASE_URL or ""
                self.timeout = config.DEFAULT_TIMEOUT
        return DatabaseCompat(self)
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "env_prefix": "",
        "extra": "ignore"
    }


# Environment-specific configurations
class DevelopmentConfig(ServerConfig):
    DEBUG_MODE: bool = True
    RELOAD: bool = False
    LOG_LEVEL: str = "INFO"
    ENABLE_COMPONENT_DEBUGGING: bool = True
    MAX_CONCURRENT_REQUESTS: int = 3
    ENABLE_BACKGROUND_MONITORING: bool = False


class ProductionConfig(ServerConfig):
    DEBUG_MODE: bool = False
    RELOAD: bool = False
    WORKERS: int = 4
    MAX_CONCURRENT_REQUESTS: int = 20
    ENABLE_BACKGROUND_MONITORING: bool = True
    LOG_LEVEL: str = "INFO"
    ENABLE_COMPONENT_DEBUGGING: bool = False
    RELOAD_EXCLUDES: List[str] = []


class TestingConfig(ServerConfig):
    DEBUG_MODE: bool = True
    ENABLE_CACHING: bool = False
    MOCK_UNAVAILABLE_COMPONENTS: bool = True
    MAX_CONCURRENT_REQUESTS: int = 1
    LOG_LEVEL: str = "WARNING"
    ENABLE_BACKGROUND_MONITORING: bool = False


# Factory function to get appropriate config
def get_config() -> ServerConfig:
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    config_map = {
        "production": ProductionConfig,
        "prod": ProductionConfig,
        "testing": TestingConfig,
        "test": TestingConfig,
        "development": DevelopmentConfig,
        "dev": DevelopmentConfig
    }
    
    config_class = config_map.get(env, DevelopmentConfig)
    return config_class()


# Backwards compatibility aliases and functions
OrchestratorConfig = ServerConfig

# Global config instance for singleton pattern
_global_config_instance: Optional[ServerConfig] = None

def get_global_config() -> ServerConfig:
    """Get singleton config instance for backwards compatibility"""
    global _global_config_instance
    if _global_config_instance is None:
        _global_config_instance = get_config()
    return _global_config_instance

def get_orchestrator_config() -> ServerConfig:
    """Alias for orchestrator compatibility"""
    return get_global_config()

# Convenience functions
def get_hybrid_config() -> Dict[str, Any]:
    """Get hybrid orchestrator configuration"""
    return get_config().get_hybrid_config()

def get_uvicorn_config() -> Dict[str, Any]:
    """Get uvicorn server configuration"""
    return get_config().get_uvicorn_config()

def is_development() -> bool:
    """Check if running in development mode"""
    return os.getenv("ENVIRONMENT", "development").lower() in ["development", "dev"]

def is_production() -> bool:
    """Check if running in production mode"""
    return os.getenv("ENVIRONMENT", "development").lower() in ["production", "prod"]

# Project paths
PROJECT_ROOT = Path(__file__).parent
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Usage example for the server startup
if __name__ == "__main__":
    # Load environment-specific config
    app_config = get_config()
    
    print("Server Configuration:")
    print(f"  Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"  Host: {app_config.HOST}:{app_config.PORT}")
    print(f"  Workers: {app_config.WORKERS}")
    print(f"  Debug Mode: {app_config.DEBUG_MODE}")
    print(f"  Log Level: {app_config.LOG_LEVEL}")
    print(f"  Max Concurrent Requests: {app_config.MAX_CONCURRENT_REQUESTS}")
    print(f"  Caching Enabled: {app_config.ENABLE_CACHING}")
    print(f"  Reload Excludes: {app_config.RELOAD_EXCLUDES}")
    
    # Show hybrid config
    hybrid_config = app_config.get_hybrid_config()
    print("\nHybrid Orchestrator Config:")
    for key, value in hybrid_config.items():
        print(f"  {key}: {value}")
    
    # Show available API keys (masked)
    api_keys = app_config.get_api_keys()
    print("\nLLM API Keys Status:")
    for service, key in api_keys.items():
        status = "Available" if key else "Missing"
        print(f"  {service}: {status}")
    
    # Show uvicorn config
    uvicorn_config = app_config.get_uvicorn_config()
    print("\nUvicorn Configuration:")
    for key, value in uvicorn_config.items():
        if value is not None:
            print(f"  {key}: {value}")

    # Test backwards compatibility
    print("\nBackwards Compatibility Test:")
    print(f"  config.logging.log_level: {app_config.logging.log_level}")
    print(f"  config.fastapi_server.allowed_origins: {app_config.fastapi_server.allowed_origins}")
    print(f"  config.database.connection_string: {app_config.database.connection_string}")

    # Test aliases
    orchestrator_config = get_orchestrator_config()
    global_config = get_global_config()
    print(f"  Aliases working: {orchestrator_config is global_config}")
