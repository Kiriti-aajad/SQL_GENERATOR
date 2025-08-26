"""
Configuration management for SQL Executor
Handles MS SQL Server connection settings and execution parameters
"""


import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import logging


# Load environment variables from .env file
load_dotenv()


logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """MS SQL Server database configuration"""
    
    # Database connection credentials from .env
    user: str
    password: str
    server: str
    database: str
    port: int = 1433
    
    # Connection settings (defaults updated to bypass SSL trust issues for testing)
    connection_timeout: int = 30
    command_timeout: int = 300  # 5 minutes max query execution
    encrypt: bool = False  # Default to False to avoid trust validation failures
    trust_server_certificate: bool = True  # Default to True for self-signed certs during testing
    
    # Connection pool settings
    min_pool_size: int = 2
    max_pool_size: int = 20  # Increased from 10 to avoid exhaustion during retries
    pool_timeout: int = 30  # ← Added this parameter to fix the error
    
    def get_connection_string(self) -> str:
        """Build MS SQL Server connection string"""
        return (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={self.server},{self.port};"
            f"DATABASE={self.database};"
            f"UID={self.user};"
            f"PWD={self.password};"
            f"Encrypt={'yes' if self.encrypt else 'no'};"
            f"TrustServerCertificate={'yes' if self.trust_server_certificate else 'no'};"
            f"Connection Timeout={self.connection_timeout};"
            f"Command Timeout={self.command_timeout};"
        )
    
    def get_async_connection_string(self) -> str:
        """Build async connection string for aioodbc"""
        return self.get_connection_string()


@dataclass
class ExecutorConfig:
    """SQL Executor configuration and limits"""
    
    # Query execution limits (READ-ONLY environment)
    max_execution_time: int = 300  # 5 minutes maximum
    max_result_rows: int = 10000   # Maximum rows to return
    default_timeout: int = 30      # Default query timeout in seconds
    
    # Result formatting options
    default_output_format: str = "json"
    supported_formats: tuple = ("json", "csv", "html", "excel")
    
    # Safety settings (READ-ONLY constraints)
    enforce_read_only: bool = True  # Block non-SELECT queries
    enable_query_logging: bool = True
    log_execution_time: bool = True
    
    # Performance settings
    enable_result_caching: bool = False  # Disabled by default
    cache_ttl_seconds: int = 300  # 5 minutes cache TTL
    stream_large_results: bool = True  # Stream results > 1000 rows
    
    # Error handling
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 1
    enable_detailed_errors: bool = True


@dataclass
class SecurityConfig:
    """Security and validation configuration"""
    
    # READ-ONLY enforcement patterns
    blocked_keywords: tuple = (
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'TRUNCATE', 'ALTER',
        'CREATE', 'MERGE', 'BULK', 'EXEC', 'EXECUTE', 'SP_'
    )
    
    # Allowed query patterns (READ-ONLY)
    allowed_patterns: tuple = (
        'SELECT', 'WITH', 'FROM', 'JOIN', 'WHERE', 'GROUP BY',
        'ORDER BY', 'HAVING', 'UNION', 'EXCEPT', 'INTERSECT'
    )
    
    # Validation settings
    validate_query_syntax: bool = True
    check_table_access: bool = True
    log_security_events: bool = True


class ConfigManager:
    """Central configuration manager"""
    
    def __init__(self):
        self._database_config: Optional[DatabaseConfig] = None
        self._executor_config: Optional[ExecutorConfig] = None
        self._security_config: Optional[SecurityConfig] = None
        
    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration"""
        if self._database_config is None:
            self._database_config = self._load_database_config()
        return self._database_config
    
    @property
    def executor(self) -> ExecutorConfig:
        """Get executor configuration"""
        if self._executor_config is None:
            self._executor_config = self._load_executor_config()
        return self._executor_config
    
    @property
    def security(self) -> SecurityConfig:
        """Get security configuration"""
        if self._security_config is None:
            self._security_config = self._load_security_config()
        return self._security_config
    
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration from environment variables"""
        
        # Get required database credentials with logging for missing vars
        db_user = os.getenv('db_user')
        db_password = os.getenv('db_password')
        db_server = os.getenv('db_server')
        db_name = os.getenv('db_name')
        
        # Validate required credentials with logging
        missing_vars = []
        if not db_user: missing_vars.append('db_user')
        if not db_password: missing_vars.append('db_password')
        if not db_server: missing_vars.append('db_server')
        if not db_name: missing_vars.append('db_name')
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        logger.info("Database credentials loaded successfully")
        
        return DatabaseConfig(
            user=db_user,  # type: ignore
            password=db_password,  # type: ignore
            server=db_server,  # type: ignore
            database=db_name,  # type: ignore
            port=int(os.getenv('db_port', 1433)),
            connection_timeout=int(os.getenv('DB_CONNECTION_TIMEOUT', 30)),
            command_timeout=int(os.getenv('DB_COMMAND_TIMEOUT', 300)),
            encrypt=os.getenv('DB_ENCRYPT', 'false').lower() == 'true',  # Default to False for testing
            trust_server_certificate=os.getenv('DB_TRUST_CERT', 'true').lower() == 'true',  # Default to True for bypass
            min_pool_size=int(os.getenv('DB_MIN_POOL_SIZE', 2)),
            max_pool_size=int(os.getenv('DB_MAX_POOL_SIZE', 10)),
            pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', 30))  # ← This now works correctly
        )
    
    def _load_executor_config(self) -> ExecutorConfig:
        """Load executor configuration from environment variables"""
        
        return ExecutorConfig(
            max_execution_time=int(os.getenv('EXECUTOR_MAX_EXECUTION_TIME', 300)),
            max_result_rows=int(os.getenv('EXECUTOR_MAX_ROWS', 10000)),
            default_timeout=int(os.getenv('EXECUTOR_DEFAULT_TIMEOUT', 30)),
            default_output_format=os.getenv('EXECUTOR_DEFAULT_FORMAT', 'json'),
            enforce_read_only=os.getenv('EXECUTOR_READ_ONLY', 'true').lower() == 'true',
            enable_query_logging=os.getenv('EXECUTOR_ENABLE_LOGGING', 'true').lower() == 'true',
            log_execution_time=os.getenv('EXECUTOR_LOG_TIME', 'true').lower() == 'true',
            enable_result_caching=os.getenv('EXECUTOR_ENABLE_CACHE', 'false').lower() == 'true',
            cache_ttl_seconds=int(os.getenv('EXECUTOR_CACHE_TTL', 300)),
            stream_large_results=os.getenv('EXECUTOR_STREAM_RESULTS', 'true').lower() == 'true',
            max_retry_attempts=int(os.getenv('EXECUTOR_MAX_RETRIES', 3)),
            retry_delay_seconds=int(os.getenv('EXECUTOR_RETRY_DELAY', 1)),
            enable_detailed_errors=os.getenv('EXECUTOR_DETAILED_ERRORS', 'true').lower() == 'true'
        )
    
    def _load_security_config(self) -> SecurityConfig:
        """Load security configuration"""
        
        # Use default security settings for READ-ONLY environment
        return SecurityConfig()
    
    def validate_configuration(self) -> bool:
        """Validate all configuration settings"""
        
        try:
            # Test database configuration
            db_config = self.database
            connection_string = db_config.get_connection_string()
            
            if not connection_string:
                return False
            
            # Validate executor settings
            exec_config = self.executor
            
            if exec_config.max_execution_time <= 0:
                return False
            
            if exec_config.max_result_rows <= 0:
                return False
            
            if exec_config.default_output_format not in exec_config.supported_formats:
                return False
            
            # Log SSL settings for debugging connection issues
            logger.info(f"SSL settings: Encrypt={db_config.encrypt}, TrustServerCertificate={db_config.trust_server_certificate}")
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def get_environment_info(self) -> dict:
        """Get current environment configuration summary"""
        
        return {
            'database_server': self.database.server,
            'database_name': self.database.database,
            'read_only_mode': self.executor.enforce_read_only,
            'max_execution_time': self.executor.max_execution_time,
            'max_result_rows': self.executor.max_result_rows,
            'default_format': self.executor.default_output_format,
            'connection_pool_size': self.database.max_pool_size,
            'query_logging_enabled': self.executor.enable_query_logging,
            'caching_enabled': self.executor.enable_result_caching,
            'ssl_encrypt': self.database.encrypt,
            'ssl_trust_cert': self.database.trust_server_certificate,
            'pool_timeout': self.database.pool_timeout  # ← Added pool_timeout to environment info
        }


# Global configuration instance
config = ConfigManager()


# Convenience functions for easy access
def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return config.database


def get_executor_config() -> ExecutorConfig:
    """Get executor configuration"""
    return config.executor


def get_security_config() -> SecurityConfig:
    """Get security configuration"""
    return config.security


def validate_environment() -> bool:
    """Validate environment configuration"""
    return config.validate_configuration()


# Export main configuration objects
__all__ = [
    'DatabaseConfig',
    'ExecutorConfig', 
    'SecurityConfig',
    'ConfigManager',
    'config',
    'get_database_config',
    'get_executor_config',
    'get_security_config',
    'validate_environment'
]
