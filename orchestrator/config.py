"""
Enhanced Server Configuration with Environment Variable Support + PromptBuilder Integration
Compatible with Complete Integrated Hybrid AI Agent Orchestrator
UPDATED: Added PromptBuilder configuration for centralized architecture
Production-ready with backwards compatibility
FIXED: Nested configuration structure for orchestrator compatibility + PromptBuilder
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from dataclasses import dataclass
from enum import Enum

# NEW: PromptBuilder enums
class PromptBuilderMode(Enum):
    """PromptBuilder processing modes"""
    SOPHISTICATED = "sophisticated"
    BASIC = "basic"
    FALLBACK = "fallback"
    ADAPTIVE = "adaptive"

class PromptQuality(Enum):
    """PromptBuilder quality levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ADAPTIVE = "adaptive"

# NEW: PromptBuilder Configuration
@dataclass
class PromptBuilderConfig:
    """CENTRALIZED PromptBuilder Configuration"""
    # Core settings
    enable_prompt_builder: bool = True
    prompt_builder_timeout: float = 20.0
    enable_template_optimization: bool = True
    max_prompt_length: int = 8000
    prompt_builder_fallback: bool = True
    
    # Processing settings
    processing_mode: PromptBuilderMode = PromptBuilderMode.SOPHISTICATED
    quality_level: PromptQuality = PromptQuality.HIGH
    enable_schema_awareness: bool = True
    enable_context_enhancement: bool = True
    enable_domain_specialization: bool = True
    
    # Performance settings
    max_concurrent_builds: int = 10
    enable_prompt_caching: bool = True
    cache_ttl_seconds: int = 300
    enable_performance_monitoring: bool = True
    
    # Quality settings
    min_prompt_quality_score: float = 0.7
    enable_prompt_validation: bool = True
    enable_advanced_templates: bool = True
    fallback_to_basic: bool = True
    
    # Centralized architecture settings
    centralized_service: bool = True
    eliminate_duplication: bool = True
    manager_integration: bool = True
    bridge_integration: bool = True
    
    # Template settings
    enable_mathstral_templates: bool = True
    enable_traditional_templates: bool = True
    enable_banking_domain_templates: bool = True
    enable_nlp_enhanced_templates: bool = True

@dataclass
class NLPSchemaIntegrationConfig:
    """NLP Schema Integration Configuration"""
    enable_nlp_schema_integration: bool = True
    nlp_timeout: float = 30.0
    schema_timeout: float = 25.0
    integration_timeout: float = 60.0
    banking_domain_enabled: bool = True
    max_retry_attempts: int = 2
    # NEW: PromptBuilder integration
    enable_prompt_builder_integration: bool = True
    use_centralized_prompt_service: bool = True

@dataclass
class MathstralManagerConfig:
    """Mathstral Manager Configuration"""
    enable_mathstral_manager: bool = True
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    timeout: float = 35.0
    max_retries: int = 3
    fallback_enabled: bool = True
    # NEW: PromptBuilder integration
    use_centralized_prompt_builder: bool = True
    enable_sophisticated_prompts: bool = True

@dataclass
class TraditionalManagerConfig:
    """Traditional Manager Configuration"""
    enable_traditional_manager: bool = True
    deepseek_enabled: bool = True
    openai_enabled: bool = False
    timeout: float = 30.0
    max_retries: int = 2
    # NEW: PromptBuilder integration
    use_centralized_prompt_builder: bool = True
    enable_optimized_prompts: bool = True

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

    # Component-specific timeouts (for internal use)
    NLP_TIMEOUT: float = Field(default=30.0, description="NLP processing timeout")
    SCHEMA_TIMEOUT: float = Field(default=25.0, description="Schema search timeout")
    PROMPT_TIMEOUT: float = Field(default=20.0, description="Prompt building timeout")
    SQL_TIMEOUT: float = Field(default=35.0, description="SQL generation timeout")

    # NEW: PromptBuilder-specific settings
    ENABLE_PROMPT_BUILDER: bool = Field(default=True, description="Enable PromptBuilder")
    PROMPT_BUILDER_TIMEOUT: float = Field(default=20.0, description="PromptBuilder timeout")
    PROMPT_BUILDER_MODE: str = Field(default="sophisticated", description="PromptBuilder mode")
    ENABLE_PROMPT_CACHING: bool = Field(default=True, description="Enable prompt caching")
    PROMPT_CACHE_TTL: int = Field(default=300, description="Prompt cache TTL")
    MAX_PROMPT_LENGTH: int = Field(default=8000, description="Maximum prompt length")
    ENABLE_CENTRALIZED_PROMPTS: bool = Field(default=True, description="Enable centralized prompt service")

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

    # Component Enable/Disable Flags
    ENABLE_NLP_SCHEMA_INTEGRATION: bool = Field(default=True, description="Enable NLP schema integration")
    ENABLE_MATHSTRAL_MANAGER: bool = Field(default=True, description="Enable Mathstral manager")
    ENABLE_TRADITIONAL_MANAGER: bool = Field(default=True, description="Enable traditional manager")
    BANKING_DOMAIN_ENABLED: bool = Field(default=True, description="Enable banking domain features")

    # Field validators (keeping existing ones and adding new ones)
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

    # NEW: PromptBuilder validation
    @field_validator('PROMPT_BUILDER_MODE')
    @classmethod
    def validate_prompt_builder_mode(cls, v: str) -> str:
        valid_modes = ['sophisticated', 'basic', 'fallback', 'adaptive']
        v_lower = v.lower()
        if v_lower not in valid_modes:
            raise ValueError(f'PROMPT_BUILDER_MODE must be one of {valid_modes}')
        return v_lower

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

    # NEW: PromptBuilder configuration property
    @property
    def prompt_builder(self) -> PromptBuilderConfig:
        """Get PromptBuilder configuration"""
        # Map string mode to enum
        mode_mapping = {
            'sophisticated': PromptBuilderMode.SOPHISTICATED,
            'basic': PromptBuilderMode.BASIC,
            'fallback': PromptBuilderMode.FALLBACK,
            'adaptive': PromptBuilderMode.ADAPTIVE
        }
        
        return PromptBuilderConfig(
            enable_prompt_builder=self.ENABLE_PROMPT_BUILDER,
            prompt_builder_timeout=self.PROMPT_BUILDER_TIMEOUT,
            enable_template_optimization=True,
            max_prompt_length=self.MAX_PROMPT_LENGTH,
            prompt_builder_fallback=True,
            processing_mode=mode_mapping.get(self.PROMPT_BUILDER_MODE.lower(), PromptBuilderMode.SOPHISTICATED),
            quality_level=PromptQuality.HIGH,
            enable_schema_awareness=True,
            enable_context_enhancement=True,
            enable_domain_specialization=self.BANKING_DOMAIN_ENABLED,
            max_concurrent_builds=min(self.MAX_CONCURRENT_REQUESTS, 10),
            enable_prompt_caching=self.ENABLE_PROMPT_CACHING,
            cache_ttl_seconds=self.PROMPT_CACHE_TTL,
            enable_performance_monitoring=self.ENABLE_PERFORMANCE_MONITORING,
            centralized_service=self.ENABLE_CENTRALIZED_PROMPTS,
            eliminate_duplication=True,
            manager_integration=True,
            bridge_integration=True,
            enable_mathstral_templates=self.ENABLE_MATHSTRAL_MANAGER,
            enable_traditional_templates=self.ENABLE_TRADITIONAL_MANAGER,
            enable_banking_domain_templates=self.BANKING_DOMAIN_ENABLED,
            enable_nlp_enhanced_templates=self.ENABLE_NLP_SCHEMA_INTEGRATION
        )

    # UPDATED: Enhanced nested configuration properties
    @property
    def nlp_schema_integration(self) -> NLPSchemaIntegrationConfig:
        """Get NLP schema integration configuration"""
        return NLPSchemaIntegrationConfig(
            enable_nlp_schema_integration=self.ENABLE_NLP_SCHEMA_INTEGRATION,
            nlp_timeout=self.NLP_TIMEOUT,
            schema_timeout=self.SCHEMA_TIMEOUT,
            integration_timeout=self.DEFAULT_TIMEOUT,
            banking_domain_enabled=self.BANKING_DOMAIN_ENABLED,
            max_retry_attempts=self.RETRY_ATTEMPTS,
            enable_prompt_builder_integration=self.ENABLE_PROMPT_BUILDER,
            use_centralized_prompt_service=self.ENABLE_CENTRALIZED_PROMPTS
        )

    @property
    def mathstral_manager(self) -> MathstralManagerConfig:
        """Get Mathstral manager configuration"""
        return MathstralManagerConfig(
            enable_mathstral_manager=self.ENABLE_MATHSTRAL_MANAGER,
            api_key=self.MATHSTRAL_API_KEY,
            timeout=self.SQL_TIMEOUT,
            max_retries=self.RETRY_ATTEMPTS,
            fallback_enabled=self.ENABLE_GRACEFUL_FALLBACKS,
            use_centralized_prompt_builder=self.ENABLE_CENTRALIZED_PROMPTS,
            enable_sophisticated_prompts=self.ENABLE_PROMPT_BUILDER
        )

    @property
    def traditional_manager(self) -> TraditionalManagerConfig:
        """Get traditional manager configuration"""
        return TraditionalManagerConfig(
            enable_traditional_manager=self.ENABLE_TRADITIONAL_MANAGER,
            deepseek_enabled=bool(self.DEEPSEEK_API_KEY),
            openai_enabled=bool(self.OPENAI_API_KEY),
            timeout=self.SQL_TIMEOUT,
            max_retries=self.RETRY_ATTEMPTS,
            use_centralized_prompt_builder=self.ENABLE_CENTRALIZED_PROMPTS,
            enable_optimized_prompts=self.ENABLE_PROMPT_BUILDER
        )

    # UPDATED: Enhanced get_hybrid_config with PromptBuilder settings
    def get_hybrid_config(self) -> Dict[str, Any]:
        """
        ENHANCED: Get configuration for HybridAIAgentOrchestrator with PromptBuilder support
        """
        return {
            # Existing settings
            "enable_caching": self.ENABLE_CACHING,
            "cache_ttl": self.CACHE_TTL_SECONDS,
            "max_cache_size": self.MAX_CACHE_SIZE,
            "max_concurrent_requests": self.MAX_CONCURRENT_REQUESTS,
            "enable_performance_monitoring": self.ENABLE_PERFORMANCE_MONITORING,
            "enable_background_monitoring": self.ENABLE_BACKGROUND_MONITORING,
            "enable_graceful_fallbacks": self.ENABLE_GRACEFUL_FALLBACKS,
            "retry_attempts": self.RETRY_ATTEMPTS,
            "retry_delay": self.RETRY_DELAY,
            "debug_mode": self.DEBUG_MODE,
            "enable_component_debugging": self.ENABLE_COMPONENT_DEBUGGING,
            
            # NEW: PromptBuilder settings
            "enable_prompt_builder": self.ENABLE_PROMPT_BUILDER,
            "prompt_builder_timeout": self.PROMPT_BUILDER_TIMEOUT,
            "prompt_builder_mode": self.PROMPT_BUILDER_MODE,
            "enable_prompt_caching": self.ENABLE_PROMPT_CACHING,
            "prompt_cache_ttl": self.PROMPT_CACHE_TTL,
            "max_prompt_length": self.MAX_PROMPT_LENGTH,
            "enable_centralized_prompts": self.ENABLE_CENTRALIZED_PROMPTS,
            
            # Centralized architecture flags
            "centralized_architecture": True,
            "eliminate_prompt_duplication": True,
            "manager_prompt_integration": True
        }

    # Keep all existing methods (get_api_keys, get_uvicorn_config, etc.) unchanged
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

    def get_component_timeouts(self) -> Dict[str, float]:
        """Get component timeouts for internal use"""
        return {
            "nlp_timeout": self.NLP_TIMEOUT,
            "schema_timeout": self.SCHEMA_TIMEOUT,
            "prompt_timeout": self.PROMPT_TIMEOUT,
            "sql_timeout": self.SQL_TIMEOUT,
            "default_timeout": self.DEFAULT_TIMEOUT,
            "prompt_builder_timeout": self.PROMPT_BUILDER_TIMEOUT  # NEW
        }

    # Keep all existing backwards compatibility properties unchanged
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

# Keep all existing environment-specific configurations unchanged
class DevelopmentConfig(ServerConfig):
    DEBUG_MODE: bool = True
    RELOAD: bool = False
    LOG_LEVEL: str = "INFO"
    ENABLE_COMPONENT_DEBUGGING: bool = True
    MAX_CONCURRENT_REQUESTS: int = 3
    ENABLE_BACKGROUND_MONITORING: bool = False
    # NEW: Development PromptBuilder settings
    ENABLE_PROMPT_BUILDER: bool = True
    PROMPT_BUILDER_MODE: str = "sophisticated"
    ENABLE_CENTRALIZED_PROMPTS: bool = True

class ProductionConfig(ServerConfig):
    DEBUG_MODE: bool = False
    RELOAD: bool = False
    WORKERS: int = 4
    MAX_CONCURRENT_REQUESTS: int = 20
    ENABLE_BACKGROUND_MONITORING: bool = True
    LOG_LEVEL: str = "INFO"
    ENABLE_COMPONENT_DEBUGGING: bool = False
    RELOAD_EXCLUDES: List[str] = []
    # NEW: Production PromptBuilder settings
    ENABLE_PROMPT_BUILDER: bool = True
    PROMPT_BUILDER_MODE: str = "sophisticated"
    ENABLE_CENTRALIZED_PROMPTS: bool = True
    ENABLE_PROMPT_CACHING: bool = True

class TestingConfig(ServerConfig):
    DEBUG_MODE: bool = True
    ENABLE_CACHING: bool = False
    MOCK_UNAVAILABLE_COMPONENTS: bool = True
    MAX_CONCURRENT_REQUESTS: int = 1
    LOG_LEVEL: str = "WARNING"
    ENABLE_BACKGROUND_MONITORING: bool = False
    # NEW: Testing PromptBuilder settings
    ENABLE_PROMPT_BUILDER: bool = False
    PROMPT_BUILDER_MODE: str = "basic"
    ENABLE_CENTRALIZED_PROMPTS: bool = False

# Keep all existing factory functions and aliases unchanged
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

class OrchestratorConfig(ServerConfig):
    """Orchestrator configuration alias for backwards compatibility"""
    pass

# Keep all existing global functions unchanged
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

# Keep existing project paths
PROJECT_ROOT = Path(__file__).parent
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"

LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ENHANCED: Updated usage example with PromptBuilder info
if __name__ == "__main__":
    app_config = get_config()
    
    print("Server Configuration:")
    print(f"  Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"  Host: {app_config.HOST}:{app_config.PORT}")
    print(f"  Workers: {app_config.WORKERS}")
    print(f"  Debug Mode: {app_config.DEBUG_MODE}")
    print(f"  Log Level: {app_config.LOG_LEVEL}")
    print(f"  Max Concurrent Requests: {app_config.MAX_CONCURRENT_REQUESTS}")
    print(f"  Caching Enabled: {app_config.ENABLE_CACHING}")

    # NEW: PromptBuilder configuration
    print("\nPromptBuilder Configuration:")
    pb_config = app_config.prompt_builder
    print(f"  Enabled: {pb_config.enable_prompt_builder}")
    print(f"  Mode: {pb_config.processing_mode}")
    print(f"  Quality Level: {pb_config.quality_level}")
    print(f"  Centralized Service: {pb_config.centralized_service}")
    print(f"  Manager Integration: {pb_config.manager_integration}")
    print(f"  Eliminate Duplication: {pb_config.eliminate_duplication}")
    print(f"  Timeout: {pb_config.prompt_builder_timeout}")
    print(f"  Caching: {pb_config.enable_prompt_caching}")

    hybrid_config = app_config.get_hybrid_config()
    print("\nEnhanced Hybrid Orchestrator Config:")
    for key, value in hybrid_config.items():
        print(f"  {key}: {value}")

    print("\nNested Configuration Objects:")
    print(f"  NLP Schema Integration: {app_config.nlp_schema_integration.enable_nlp_schema_integration}")
    print(f"  NLP PromptBuilder Integration: {app_config.nlp_schema_integration.enable_prompt_builder_integration}")
    print(f"  Mathstral Manager: {app_config.mathstral_manager.enable_mathstral_manager}")
    print(f"  Mathstral Centralized Prompts: {app_config.mathstral_manager.use_centralized_prompt_builder}")
    print(f"  Traditional Manager: {app_config.traditional_manager.enable_traditional_manager}")
    print(f"  Traditional Centralized Prompts: {app_config.traditional_manager.use_centralized_prompt_builder}")

    timeouts = app_config.get_component_timeouts()
    print("\nComponent Timeouts:")
    for key, value in timeouts.items():
        print(f"  {key}: {value}")

    api_keys = app_config.get_api_keys()
    print("\nLLM API Keys Status:")
    for service, key in api_keys.items():
        status = "Available" if key else "Missing"
        print(f"  {service}: {status}")

    print(f"\nENTRALIZED PROMPT BUILDER CONFIGURATION READY!")
