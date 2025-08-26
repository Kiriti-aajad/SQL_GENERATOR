"""
Unified API Models for SQL Generator
Compatible with Enhanced Configuration and Orchestrator System
Eliminates duplicate definitions and adds comprehensive validation
FIXED: Confidence type consistency, field shadowing, and validator issues
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union, Literal
from enum import Enum
from datetime import datetime
import uuid

class QueryType(str, Enum):
    """Types of queries supported"""
    SQL_GENERATION = "sql_generation"
    SCHEMA_SEARCH = "schema_search"
    DATA_ANALYSIS = "data_analysis"
    HEALTH_CHECK = "health_check"
    COMPONENT_STATUS = "component_status"

class ConfidenceLevel(str, Enum):
    """Confidence levels for responses"""
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"
    UNKNOWN = "unknown"

class ProcessingMode(str, Enum):
    """Processing modes for orchestrator"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    ADAPTIVE = "adaptive"

class ComponentStatus(str, Enum):
    """Component health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNAVAILABLE = "unavailable"
    ERROR = "error"

# Base Request Model
class BaseRequest(BaseModel):
    """Base request model with common fields"""
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Session identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

# Main Query Request Model (Unified)
class QueryRequest(BaseRequest):
    """Unified query request model - replaces all duplicate definitions"""
    query: str = Field(..., min_length=1, max_length=2000, description="User's natural language query")
    query_type: QueryType = Field(default=QueryType.SQL_GENERATION, description="Type of query being made")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Database context information")
    processing_mode: ProcessingMode = Field(default=ProcessingMode.COMPREHENSIVE, description="Processing mode")
    
    # Processing options
    include_explanation: bool = Field(default=True, description="Include explanation of generated SQL")
    max_results: Optional[int] = Field(default=100, ge=1, le=10000, description="Maximum number of results")
    timeout: Optional[int] = Field(default=30, ge=5, le=300, description="Query timeout in seconds")
    
    # Advanced options
    enable_caching: bool = Field(default=True, description="Enable result caching")
    enable_fallbacks: bool = Field(default=True, description="Enable graceful fallbacks")
    debug_mode: bool = Field(default=False, description="Enable debug information")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or v.isspace():
            raise ValueError('Query cannot be empty or whitespace only')
        # Remove excessive whitespace
        cleaned = ' '.join(v.split())
        if len(cleaned) < 3:
            raise ValueError('Query must be at least 3 characters long')
        return cleaned
    
    @validator('context')
    def validate_context(cls, v):
        if v is not None and not isinstance(v, dict):
            raise ValueError('Database context must be a dictionary')
        return v or {}
    
    @validator('timeout')
    def validate_timeout(cls, v, values):
        if v is not None:
            query_type = values.get('query_type')
            if query_type == QueryType.HEALTH_CHECK and v > 10:
                raise ValueError('Health check timeout cannot exceed 10 seconds')
            elif query_type == QueryType.SCHEMA_SEARCH and v > 60:
                raise ValueError('Schema search timeout cannot exceed 60 seconds')
        return v

# Specialized Request Models
class SQLGenerationRequest(QueryRequest):
    """Specific request model for SQL generation"""
    query_type: Literal[QueryType.SQL_GENERATION] = Field(default=QueryType.SQL_GENERATION) # pyright: ignore[reportIncompatibleVariableOverride]
    
    # SQL-specific options
    target_database: Optional[str] = Field(default=None, description="Target database system")
    schema_filter: Optional[List[str]] = Field(default=None, description="Filter by specific schemas")
    table_filter: Optional[List[str]] = Field(default=None, description="Filter by specific tables")
    
    # Query optimization
    optimize_query: bool = Field(default=True, description="Apply query optimization")
    include_indexes: bool = Field(default=False, description="Include index suggestions")
    validate_syntax: bool = Field(default=True, description="Validate SQL syntax")

class SchemaSearchRequest(BaseRequest):
    """Schema search request model"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query for schema elements")
    search_type: str = Field(default="semantic", description="Type of search (semantic, keyword, hybrid)")
    
    # Filters
    table_filter: Optional[List[str]] = Field(default=None, description="Filter by specific tables")
    column_filter: Optional[List[str]] = Field(default=None, description="Filter by specific columns")
    schema_filter: Optional[List[str]] = Field(default=None, description="Filter by specific schemas")
    
    # Options
    include_columns: bool = Field(default=True, description="Include column information")
    include_relationships: bool = Field(default=True, description="Include relationship information")
    include_metadata: bool = Field(default=False, description="Include metadata information")
    max_results: int = Field(default=50, ge=1, le=1000, description="Maximum number of results")
    
    @validator('query')
    def validate_search_query(cls, v):
        if not v or v.isspace():
            raise ValueError('Search query cannot be empty')
        return v.strip()

# Base Response Model
class BaseResponse(BaseModel):
    """Base response model with common fields"""
    success: bool = Field(..., description="Whether the request was successful")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    
    # Error handling
    error: Optional[str] = Field(default=None, description="Error message if any")
    error_code: Optional[str] = Field(default=None, description="Error code for programmatic handling")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    
    # Debug information
    debug_info: Optional[Dict[str, Any]] = Field(default=None, description="Debug information")

# FIXED: Main Query Response Model with ConfidenceLevel enum instead of float
class QueryResponse(BaseResponse):
    """Unified query response model - FIXED confidence type to use enum"""
    query: Optional[str] = Field(default=None, description="Original query")
    message: Optional[str] = Field(default=None, description="Response message")
    
    # SQL Generation results
    sql: Optional[str] = Field(default=None, description="Generated SQL query")
    explanation: Optional[str] = Field(default=None, description="Explanation of the query/result")
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.UNKNOWN, description="Confidence level")
    
    # Processing metadata
    generated_by: Optional[str] = Field(default=None, description="Component that generated the response")
    processing_mode: Optional[ProcessingMode] = Field(default=None, description="Processing mode used")
    fallback_applied: bool = Field(default=False, description="Whether fallback processing was used")
    cached_result: bool = Field(default=False, description="Whether result was retrieved from cache")
    
    # Schema context
    schema_context: Optional[Dict[str, Any]] = Field(default=None, description="Schema context used")
    tables_used: List[str] = Field(default_factory=list, description="Tables referenced in query")
    columns_used: List[str] = Field(default_factory=list, description="Columns referenced in query")
    relationships_used: List[str] = Field(default_factory=list, description="Relationships used in query")
    
    # Performance metrics
    component_timings: Optional[Dict[str, float]] = Field(default=None, description="Individual component execution times")
    cache_hit_ratio: Optional[float] = Field(default=None, description="Cache hit ratio if applicable")
    
    @validator('confidence')
    def validate_confidence(cls, v, values):
        # Auto-adjust confidence based on success and fallback
        if values.get('success'):
            if values.get('fallback_applied') and v == ConfidenceLevel.HIGH:
                return ConfidenceLevel.LOW  # Cap confidence for fallback responses
        return v
    
    @validator('sql')
    def validate_sql(cls, v):
        if v is not None:
            # Basic SQL validation
            v = v.strip()
            if v and not v.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH', '--')):
                # Allow comments at the start
                pass
        return v

# Specialized Response Models
class SchemaSearchResponse(BaseResponse):
    """Schema search response model"""
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Search results")
    total_count: int = Field(default=0, description="Total number of matching results")
    search_type: Optional[str] = Field(default=None, description="Type of search performed")
    
    # Search metadata
    query_analyzed: Optional[str] = Field(default=None, description="Analyzed/processed query")
    search_terms: List[str] = Field(default_factory=list, description="Extracted search terms")
    filters_applied: Optional[Dict[str, Any]] = Field(default=None, description="Filters that were applied")

class HealthCheckResponse(BaseResponse):
    """Health check response model"""
    status: ComponentStatus = Field(..., description="Overall system health status")
    components: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Individual component health details")
    version: Optional[str] = Field(default=None, description="Application version")
    uptime: Optional[float] = Field(default=None, description="System uptime in seconds")
    
    # System metrics
    memory_usage: Optional[Dict[str, Any]] = Field(default=None, description="Memory usage statistics")
    performance_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Performance metrics")
    configuration: Optional[Dict[str, Any]] = Field(default=None, description="System configuration summary")

# Database and Schema Models
class DatabaseContext(BaseModel):
    """Database context information"""
    tables: List[str] = Field(default_factory=list, description="Available tables")
    schemas: List[str] = Field(default_factory=list, description="Available schemas")
    database_type: Optional[str] = Field(default=None, description="Type of database (mssql, postgres, etc.)")
    version: Optional[str] = Field(default=None, description="Database version")
    connection_info: Optional[Dict[str, Any]] = Field(default=None, description="Connection metadata")
    capabilities: List[str] = Field(default_factory=list, description="Database capabilities")

# FIXED: Renamed 'schema' field to avoid shadowing BaseModel.schema
class TableInfo(BaseModel):
    """Table information model"""
    name: str = Field(..., description="Table name")
    schema_name: Optional[str] = Field(default=None, description="Schema name")  # FIXED: Renamed from 'schema'
    columns: List[Dict[str, Any]] = Field(default_factory=list, description="Column information")
    indexes: List[Dict[str, Any]] = Field(default_factory=list, description="Index information")
    relationships: List[Dict[str, Any]] = Field(default_factory=list, description="Relationship information")
    row_count: Optional[int] = Field(default=None, description="Approximate row count")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class ColumnInfo(BaseModel):
    """Column information model"""
    name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="Data type")
    is_nullable: bool = Field(default=True, description="Whether column allows NULL values")
    is_primary_key: bool = Field(default=False, description="Whether column is primary key")
    is_foreign_key: bool = Field(default=False, description="Whether column is foreign key")
    default_value: Optional[str] = Field(default=None, description="Default value")
    max_length: Optional[int] = Field(default=None, description="Maximum length for string types")
    description: Optional[str] = Field(default=None, description="Column description")

# Error Models
class ValidationError(BaseModel):
    """Validation error details"""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    invalid_value: Optional[Any] = Field(default=None, description="The invalid value")

class APIError(BaseResponse):
    """API error response model"""
    success: Literal[False] = Field(default=False) # pyright: ignore[reportIncompatibleVariableOverride]
    error_type: str = Field(..., description="Type of error")
    validation_errors: List[ValidationError] = Field(default_factory=list, description="Validation errors")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for fixing the error")

# FIXED: Legacy support with proper field migration
class LegacyQueryRequest(BaseModel):
    """Legacy query request with old field names - FIXED validators"""
    # Include all base fields
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Session identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    # Legacy field names
    user_query: str = Field(..., min_length=1, max_length=2000, description="Legacy field name for query")
    database_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Legacy field name for context")
    timeout_seconds: Optional[int] = Field(default=30, ge=5, le=300, description="Legacy field name for timeout")
    
    # Standard fields for compatibility
    query_type: QueryType = Field(default=QueryType.SQL_GENERATION, description="Type of query being made")
    processing_mode: ProcessingMode = Field(default=ProcessingMode.COMPREHENSIVE, description="Processing mode")
    include_explanation: bool = Field(default=True, description="Include explanation of generated SQL")
    max_results: Optional[int] = Field(default=100, ge=1, le=10000, description="Maximum number of results")
    enable_caching: bool = Field(default=True, description="Enable result caching")
    enable_fallbacks: bool = Field(default=True, description="Enable graceful fallbacks")
    debug_mode: bool = Field(default=False, description="Enable debug information")
    
    # Properties to provide modern field access
    @property
    def query(self) -> str:
        """Provide access to query via modern field name"""
        return self.user_query
    
    @property
    def context(self) -> Dict[str, Any]:
        """Provide access to context via modern field name"""
        return self.database_context or {}
    
    @property
    def timeout(self) -> Optional[int]:
        """Provide access to timeout via modern field name"""
        return self.timeout_seconds
    
    @validator('user_query')
    def validate_user_query(cls, v):
        if not v or v.isspace():
            raise ValueError('Query cannot be empty or whitespace only')
        cleaned = ' '.join(v.split())
        if len(cleaned) < 3:
            raise ValueError('Query must be at least 3 characters long')
        return cleaned
    
    @validator('database_context')
    def validate_database_context(cls, v):
        if v is not None and not isinstance(v, dict):
            raise ValueError('Database context must be a dictionary')
        return v or {}

# Backwards Compatibility Aliases
SQLRequest = SQLGenerationRequest
SQLResponse = QueryResponse

# Additional backward compatibility for TargetLLM if needed
class TargetLLM(str, Enum):
    """Target LLM options for backwards compatibility"""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    CLAUDE = "claude"
    GEMINI = "gemini"
    LOCAL = "local"

# Request/Response Union Types for FastAPI
QueryRequestUnion = Union[QueryRequest, SQLGenerationRequest, SchemaSearchRequest, LegacyQueryRequest]
QueryResponseUnion = Union[QueryResponse, SchemaSearchResponse, HealthCheckResponse, APIError]

# FIXED: Helper functions with proper ConfidenceLevel support
def confidence_to_level(confidence: Union[float, ConfidenceLevel]) -> ConfidenceLevel:
    """Convert float confidence to ConfidenceLevel enum"""
    if isinstance(confidence, ConfidenceLevel):
        return confidence
    
    if isinstance(confidence, (int, float)):
        if confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence > 0.0:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNKNOWN
    
    return ConfidenceLevel.UNKNOWN

def level_to_confidence(level: ConfidenceLevel) -> float:
    """Convert ConfidenceLevel enum to float"""
    mapping = {
        ConfidenceLevel.HIGH: 0.9,
        ConfidenceLevel.MEDIUM: 0.7,
        ConfidenceLevel.LOW: 0.3,
        ConfidenceLevel.UNKNOWN: 0.0
    }
    return mapping.get(level, 0.0)

# Model validation functions
def validate_request_model(data: Dict[str, Any], model_class: BaseModel) -> BaseModel:
    """Validate and create request model instance"""
    try:
        return model_class(**data) # pyright: ignore[reportCallIssue]
    except Exception as e:
        raise ValueError(f"Invalid request format: {str(e)}")

# FIXED: Helper functions with ConfidenceLevel support
def create_error_response(
    error_message: str, 
    error_code: Optional[str] = None, 
    session_id: Optional[str] = None, 
    query: Optional[str] = None
) -> QueryResponse:
    """Create standardized error response"""
    return QueryResponse(
        success=False,
        error=error_message,
        error_code=error_code,
        session_id=session_id,
        query=query,
        confidence=ConfidenceLevel.UNKNOWN,  # FIXED: Use enum instead of float
        generated_by="error_handler"
    )

def create_success_response(
    sql: Optional[str] = None, 
    explanation: Optional[str] = None,
    confidence: Union[ConfidenceLevel, float] = ConfidenceLevel.MEDIUM,  # FIXED: Support both types
    query: Optional[str] = None,
    **kwargs
) -> QueryResponse:
    """Create standardized success response"""
    # Convert confidence to proper enum if needed
    confidence_level = confidence_to_level(confidence)
    
    return QueryResponse(
        success=True,
        query=query,
        sql=sql,
        explanation=explanation,
        confidence=confidence_level,  # FIXED: Always use enum
        **kwargs
    )

# Export all models for easy importing
__all__ = [
    # Enums
    'QueryType', 'ConfidenceLevel', 'ProcessingMode', 'ComponentStatus', 'TargetLLM',
    
    # Request Models
    'BaseRequest', 'QueryRequest', 'SQLGenerationRequest', 'SchemaSearchRequest', 'LegacyQueryRequest',
    
    # Response Models  
    'BaseResponse', 'QueryResponse', 'SchemaSearchResponse', 'HealthCheckResponse',
    
    # Database Models
    'DatabaseContext', 'TableInfo', 'ColumnInfo',
    
    # Error Models
    'ValidationError', 'APIError',
    
    # Backwards Compatibility
    'SQLRequest', 'SQLResponse',
    
    # Union Types
    'QueryRequestUnion', 'QueryResponseUnion',
    
    # Utility Functions
    'validate_request_model', 'create_error_response', 'create_success_response',
    'confidence_to_level', 'level_to_confidence'
]
