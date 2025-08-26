"""
SQL Executor Module
Simplified SQL execution engine for pre-validated queries from SQL Generator
"""

from .executor import SQLExecutor
from .models import (
    ExecutionRequest, ExecutionResult, QueryMetadata, 
    OutputFormat, ExecutionMode, QueryType, ErrorType, ExecutionStatus,
    ColumnInfo, PerformanceStats,
    create_success_result, create_empty_result, create_error_result, create_blocked_result
)
from .connection_manager import ConnectionManager, connection_manager
from .formatters import ResultFormatter, format_as_json, format_as_csv, format_as_html, format_as_excel
from .error_handler import ErrorHandler, validate_readonly_query, handle_error, handle_empty_result
from .config import (
    DatabaseConfig, ExecutorConfig, SecurityConfig,
    get_database_config, get_executor_config, get_security_config
)

__version__ = "1.0.0"
__author__ = "SQL AI Agent Team"
__description__ = "SQL Executor module for safe, monitored query execution"

# Package-level constants
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_ROWS = 10000
DEFAULT_OUTPUT_FORMAT = "json"

# Main exports for easy import
__all__ = [
    # Main classes
    'SQLExecutor',
    'ConnectionManager',
    'ResultFormatter',
    'ErrorHandler',
    
    # Data models
    'ExecutionRequest', 
    'ExecutionResult',
    'QueryMetadata',
    'ColumnInfo',
    'PerformanceStats',
    
    # Enums
    'OutputFormat',
    'ExecutionMode', 
    'QueryType',
    'ErrorType',
    'ExecutionStatus',
    
    # Configuration
    'DatabaseConfig',
    'ExecutorConfig', 
    'SecurityConfig',
    
    # Factory functions
    'create_success_result',
    'create_empty_result', 
    'create_error_result',
    'create_blocked_result',
    
    # Convenience functions
    'format_as_json',
    'format_as_csv', 
    'format_as_html',
    'format_as_excel',
    'validate_readonly_query',
    'handle_error',
    'handle_empty_result',
    
    # Configuration functions
    'get_database_config',
    'get_executor_config',
    'get_security_config',
    
    # Global instances
    'connection_manager',
    
    # Package metadata
    '__version__',
    '__author__',
    '__description__'
]

# Initialize package-level configuration
try:
    # Test configuration loading
    _config_test = get_executor_config()
    _db_config_test = get_database_config()
    print(f" SQL Executor module v{__version__} initialized successfully")
except Exception as e:
    print(f" SQL Executor module loaded with configuration warning: {e}")
