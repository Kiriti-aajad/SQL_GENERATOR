"""
Data models for SQL Executor requests and responses
Defines request/response structures for MS SQL Server execution
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime
import json

class ExecutionMode(Enum):
    """Execution modes for queries"""
    DRY_RUN = "dry_run"          # Validate without executing
    EXECUTE = "execute"          # Execute and return results
    EXPLAIN_PLAN = "explain_plan" # Return execution plan only

class QueryType(Enum):
    """Types of SQL queries (READ-ONLY focus)"""
    SELECT = "SELECT"
    WITH_CTE = "WITH"            # Common Table Expressions
    UNION = "UNION"              # Union queries
    UNKNOWN = "UNKNOWN"          # Unclassified queries

class OutputFormat(Enum):
    """Supported output formats"""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    EXCEL = "excel"

class ExecutionStatus(Enum):
    """Execution status indicators"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    EMPTY_RESULT = "empty_result"
    BLOCKED = "blocked"          # Blocked due to READ-ONLY restrictions

class ErrorType(Enum):
    """Error classification for better handling"""
    CONNECTION_ERROR = "connection_error"
    SYNTAX_ERROR = "syntax_error"
    PERMISSION_ERROR = "permission_error"
    TIMEOUT_ERROR = "timeout_error"
    READ_ONLY_VIOLATION = "read_only_violation"
    RESOURCE_LIMIT = "resource_limit"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class QueryMetadata:
    """Metadata about the SQL query from Generator"""
    
    query_type: QueryType
    tables_used: List[str] = field(default_factory=list)
    columns_selected: List[str] = field(default_factory=list)
    has_joins: bool = False
    has_xml_operations: bool = False
    has_aggregations: bool = False
    complexity_score: int = 0      # 1-100 complexity rating
    estimated_rows: Optional[int] = None
    confidence_score: float = 1.0   # Generator's confidence

@dataclass
class ExecutionRequest:
    """Complete execution request from SQL Generator"""
    
    # Required fields
    sql_query: str
    user_id: str
    
    # Query metadata from Generator
    query_metadata: Optional[QueryMetadata] = None
    
    # Execution options
    execution_mode: ExecutionMode = ExecutionMode.EXECUTE
    max_rows: int = 10000
    timeout_seconds: int = 30
    
    # Output preferences
    output_format: OutputFormat = OutputFormat.JSON
    include_execution_time: bool = True
    include_row_count: bool = True
    
    # Request tracking
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary"""
        return {
            'sql_query': self.sql_query,
            'user_id': self.user_id,
            'query_metadata': self.query_metadata.__dict__ if self.query_metadata else None,
            'execution_mode': self.execution_mode.value,
            'max_rows': self.max_rows,
            'timeout_seconds': self.timeout_seconds,
            'output_format': self.output_format.value,
            'request_id': self.request_id,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class ColumnInfo:
    """Information about result set columns"""
    
    name: str
    data_type: str
    max_length: Optional[int] = None
    is_nullable: bool = True
    ordinal_position: int = 0

@dataclass
class PerformanceStats:
    """Query execution performance metrics"""
    
    execution_time_ms: int
    rows_returned: int
    connection_time_ms: int = 0
    query_parse_time_ms: int = 0
    result_format_time_ms: int = 0
    
    # MS SQL Server specific metrics
    logical_reads: Optional[int] = None
    physical_reads: Optional[int] = None
    cpu_time_ms: Optional[int] = None

@dataclass
class ExecutionResult:
    """Comprehensive execution result"""
    
    # Execution status
    status: ExecutionStatus
    success: bool
    message: str
    
    # Query results (empty list if no data)
    data: List[Dict[str, Any]] = field(default_factory=list)
    columns: List[ColumnInfo] = field(default_factory=list)
    row_count: int = 0
    
    # Performance information
    performance: Optional[PerformanceStats] = None
    
    # Error information (if applicable)
    error_type: Optional[ErrorType] = None
    error_details: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    
    # Request tracking
    request_id: Optional[str] = None
    execution_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Export options
    export_available: bool = False
    supported_formats: List[str] = field(default_factory=list)
    
    def is_empty_result(self) -> bool:
        """Check if query returned no data"""
        return self.success and self.row_count == 0
    
    def has_error(self) -> bool:
        """Check if execution had errors"""
        return not self.success
    
    def is_timeout(self) -> bool:
        """Check if execution timed out"""
        return self.status == ExecutionStatus.TIMEOUT
    
    def is_blocked(self) -> bool:
        """Check if query was blocked (READ-ONLY violation)"""
        return self.status == ExecutionStatus.BLOCKED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization"""
        return {
            'status': self.status.value,
            'success': self.success,
            'message': self.message,
            'data': self.data,
            'columns': [col.__dict__ for col in self.columns],
            'row_count': self.row_count,
            'performance': self.performance.__dict__ if self.performance else None,
            'error_type': self.error_type.value if self.error_type else None,
            'error_details': self.error_details,
            'suggestions': self.suggestions,
            'request_id': self.request_id,
            'execution_timestamp': self.execution_timestamp.isoformat(),
            'export_available': self.export_available,
            'supported_formats': self.supported_formats
        }
    
    def to_json(self) -> str:
        """Convert result to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)

@dataclass
class ConnectionInfo:
    """Database connection information"""
    
    server: str
    database: str
    connected_at: datetime
    connection_id: Optional[str] = None
    is_healthy: bool = True
    last_activity: Optional[datetime] = None

@dataclass  
class ValidationResult:
    """Result of query validation"""
    
    is_valid: bool
    is_read_only: bool
    blocked_keywords: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    estimated_complexity: int = 0

# Factory functions for common scenarios

def create_success_result(
    data: List[Dict[str, Any]], 
    columns: List[ColumnInfo],
    execution_time_ms: int,
    request_id: Optional[str] = None
) -> ExecutionResult:
    """Create successful execution result"""
    
    performance = PerformanceStats(
        execution_time_ms=execution_time_ms,
        rows_returned=len(data)
    )
    
    return ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        success=True,
        message="Query executed successfully",
        data=data,
        columns=columns,
        row_count=len(data),
        performance=performance,
        request_id=request_id,
        export_available=True,
        supported_formats=['json', 'csv', 'excel']
    )

def create_empty_result(
    execution_time_ms: int,
    request_id: Optional[str] = None
) -> ExecutionResult:
    """Create result for queries that return no data"""
    
    performance = PerformanceStats(
        execution_time_ms=execution_time_ms,
        rows_returned=0
    )
    
    return ExecutionResult(
        status=ExecutionStatus.EMPTY_RESULT,
        success=True,
        message="Query executed successfully but returned no data",
        data=[],
        columns=[],
        row_count=0,
        performance=performance,
        request_id=request_id,
        suggestions=[
            "Check if the queried tables contain data",
            "Consider broadening your WHERE conditions",
            "Verify that date ranges are not too restrictive"
        ]
    )

def create_error_result(
    error_type: ErrorType,
    error_message: str,
    suggestions: Optional[List[str]] = None,
    request_id: Optional[str] = None
) -> ExecutionResult:
    """Create error execution result"""
    
    return ExecutionResult(
        status=ExecutionStatus.FAILED,
        success=False,
        message=f"Query execution failed: {error_message}",
        error_type=error_type,
        error_details=error_message,
        suggestions=suggestions or [],
        request_id=request_id
    )

def create_blocked_result(
    blocked_operation: str,
    request_id: Optional[str] = None
) -> ExecutionResult:
    """Create result for blocked operations (READ-ONLY violation)"""
    
    return ExecutionResult(
        status=ExecutionStatus.BLOCKED,
        success=False,
        message=f"Operation blocked: {blocked_operation} operations are not allowed in READ-ONLY mode",
        error_type=ErrorType.READ_ONLY_VIOLATION,
        error_details=f"Attempted {blocked_operation} operation in READ-ONLY database",
        suggestions=[
            "This database connection only allows SELECT queries",
            "Use SELECT statements to retrieve data",
            "Contact administrator for write access if needed"
        ],
        request_id=request_id
    )

# Export all model classes and functions
__all__ = [
    # Enums
    'ExecutionMode', 'QueryType', 'OutputFormat', 'ExecutionStatus', 'ErrorType',
    
    # Data classes
    'QueryMetadata', 'ExecutionRequest', 'ColumnInfo', 'PerformanceStats',
    'ExecutionResult', 'ConnectionInfo', 'ValidationResult',
    
    # Factory functions
    'create_success_result', 'create_empty_result', 
    'create_error_result', 'create_blocked_result'
]
