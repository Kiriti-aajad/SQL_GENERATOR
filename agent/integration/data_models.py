"""
Integration Data Models for NLP-Schema-Orchestrator Pipeline
Standardized data types and contracts for seamless component integration
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

# =====================================================
# ENUMS AND CONSTANTS
# =====================================================

class QueryIntent(Enum):
    """Standardized query intent classifications"""
    SIMPLE_LOOKUP = "simple_lookup"
    AGGREGATION = "aggregation"
    COMPLEX_ANALYSIS = "complex_analysis"
    TEMPORAL_FILTER = "temporal_filter"
    JOIN_ANALYSIS = "join_analysis"
    REPORTING = "reporting"
    UNKNOWN = "unknown"

class ComponentStatus(Enum):
    """Component health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ERROR = "error"
    UNAVAILABLE = "unavailable"

class ProcessingMethod(Enum):
    """Processing method types"""
    NLP_ENHANCED = "nlp_enhanced"
    SCHEMA_DIRECT = "schema_direct"
    INTELLIGENT_AGENT = "intelligent_agent"
    FALLBACK = "fallback"
    ERROR_RECOVERY = "error_recovery"

class RequestType(Enum):
    """Request type classifications"""
    COMPLETE_SCHEMA = "complete_schema"
    NLP_ENHANCED_SCHEMA = "nlp_enhanced_schema"
    BASIC_SEARCH = "basic_search"
    TARGETED_SEARCH = "targeted_search"
    HEALTH_CHECK = "health_check"

# =====================================================
# CORE DATA MODELS
# =====================================================

@dataclass
class NLPEnhancedQuery:
    """
    Standardized NLP processing output for schema integration
    Contains all NLP insights and enhancements
    """
    original_query: str
    enhanced_query: str
    primary_intent: QueryIntent
    confidence_score: float
    target_tables: List[str] = field(default_factory=list)
    target_columns: List[str] = field(default_factory=list)
    semantic_entities: List[str] = field(default_factory=list)
    temporal_context: Optional[Dict[str, Any]] = None
    requires_joins: bool = False
    requires_xml: bool = True
    complexity_score: float = 0.5
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'original_query': self.original_query,
            'enhanced_query': self.enhanced_query,
            'primary_intent': self.primary_intent.value,
            'confidence_score': self.confidence_score,
            'target_tables': self.target_tables,
            'target_columns': self.target_columns,
            'semantic_entities': self.semantic_entities,
            'temporal_context': self.temporal_context,
            'requires_joins': self.requires_joins,
            'requires_xml': self.requires_xml,
            'complexity_score': self.complexity_score,
            'processing_metadata': self.processing_metadata
        }

@dataclass
class SchemaSearchRequest:
    """
    Standardized schema search request format
    Accepts both basic queries and NLP-enhanced requests
    """
    request_id: str
    query: str
    enhanced_query: Optional[str] = None
    request_type: RequestType = RequestType.COMPLETE_SCHEMA
    target_tables: List[str] = field(default_factory=list)
    target_columns: List[str] = field(default_factory=list)
    intent_classification: Dict[str, Any] = field(default_factory=dict)
    include_xml: bool = True
    include_joins: bool = True
    max_results: int = 30
    min_confidence: float = 0.0
    nlp_context: Optional[Dict[str, Any]] = None
    processing_hints: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API consumption"""
        return {
            'request_id': self.request_id,
            'query': self.query,
            'enhanced_query': self.enhanced_query,
            'request_type': self.request_type.value,
            'target_tables': self.target_tables,
            'target_columns': self.target_columns,
            'intent_classification': self.intent_classification,
            'include_xml': self.include_xml,
            'include_joins': self.include_joins,
            'max_results': self.max_results,
            'min_confidence': self.min_confidence,
            'nlp_context': self.nlp_context,
            'processing_hints': self.processing_hints,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemaSearchRequest':
        """Create from dictionary data"""
        return cls(
            request_id=data.get('request_id', str(uuid.uuid4())),
            query=data['query'],
            enhanced_query=data.get('enhanced_query'),
            request_type=RequestType(data.get('request_type', 'complete_schema')),
            target_tables=data.get('target_tables', []),
            target_columns=data.get('target_columns', []),
            intent_classification=data.get('intent_classification', {}),
            include_xml=data.get('include_xml', True),
            include_joins=data.get('include_joins', True),
            max_results=data.get('max_results', 30),
            min_confidence=data.get('min_confidence', 0.0),
            nlp_context=data.get('nlp_context'),
            processing_hints=data.get('processing_hints', {}),
            timestamp=data.get('timestamp', datetime.now().isoformat())
        )

@dataclass
class SchemaSearchResponse:
    """
    Standardized schema search response format
    Contains enriched results with NLP insights
    """
    request_id: str
    query: str
    status: str  # "success", "partial", "error"
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    nlp_insights: Optional[Dict[str, Any]] = None
    processing_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    response_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response"""
        return {
            'request_id': self.request_id,
            'query': self.query,
            'status': self.status,
            'data': self.data,
            'error': self.error,
            'execution_time_ms': self.execution_time_ms,
            'nlp_insights': self.nlp_insights,
            'processing_chain': self.processing_chain,
            'metadata': self.metadata,
            'response_timestamp': self.response_timestamp
        }
    
    def is_successful(self) -> bool:
        """Check if response is successful"""
        return self.status == "success"
    
    def has_data(self) -> bool:
        """Check if response contains meaningful data"""
        if not self.data:
            return False
        
        tables = self.data.get('tables', [])
        columns = self.data.get('columns_by_table', {})
        total_columns = self.data.get('total_columns', 0)
        
        return len(tables) > 0 or len(columns) > 0 or total_columns > 0

@dataclass
class ComponentHealth:
    """
    Standardized component health information
    Used for monitoring and system diagnostics
    """
    component_name: str
    status: ComponentStatus
    capabilities: Dict[str, Any] = field(default_factory=dict)
    supported_methods: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    last_check: str = field(default_factory=lambda: datetime.now().isoformat())
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for health reporting"""
        return {
            'component_name': self.component_name,
            'status': self.status.value,
            'capabilities': self.capabilities,
            'supported_methods': self.supported_methods,
            'error_message': self.error_message,
            'last_check': self.last_check,
            'performance_metrics': self.performance_metrics
        }
    
    def is_healthy(self) -> bool:
        """Check if component is healthy"""
        return self.status == ComponentStatus.HEALTHY

@dataclass
class IntegrationPipelineResult:
    """
    Complete pipeline processing result
    Contains all processing steps and their outcomes
    """
    request_id: str
    original_query: str
    final_response: SchemaSearchResponse
    nlp_processing: Optional[NLPEnhancedQuery] = None
    component_chain: List[str] = field(default_factory=list)
    total_execution_time_ms: float = 0.0
    success: bool = True
    error_details: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for comprehensive reporting"""
        return {
            'request_id': self.request_id,
            'original_query': self.original_query,
            'final_response': self.final_response.to_dict(),
            'nlp_processing': self.nlp_processing.to_dict() if self.nlp_processing else None,
            'component_chain': self.component_chain,
            'total_execution_time_ms': self.total_execution_time_ms,
            'success': self.success,
            'error_details': self.error_details,
            'performance_metrics': self.performance_metrics
        }

# =====================================================
# SPECIALIZED DATA MODELS
# =====================================================

@dataclass
class NLPInsights:
    """
    NLP processing insights for orchestrator consumption
    Extracted from NLPEnhancedQuery for easier access
    """
    detected_intent: Dict[str, Any] = field(default_factory=dict)
    target_tables_predicted: List[str] = field(default_factory=list)
    target_columns_predicted: List[str] = field(default_factory=list)
    semantic_entities: List[str] = field(default_factory=list)
    processing_hints: Dict[str, Any] = field(default_factory=dict)
    temporal_context: Optional[Dict[str, Any]] = None
    prediction_accuracy: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for response inclusion"""
        return {
            'detected_intent': self.detected_intent,
            'target_tables_predicted': self.target_tables_predicted,
            'target_columns_predicted': self.target_columns_predicted,
            'semantic_entities': self.semantic_entities,
            'processing_hints': self.processing_hints,
            'temporal_context': self.temporal_context,
            'prediction_accuracy': self.prediction_accuracy
        }

@dataclass
class ProcessingStatistics:
    """
    Processing statistics for performance monitoring
    """
    component_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_processing_time_ms: float = 0.0
    last_request_time: Optional[str] = None
    
    def calculate_success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    def update_request_stats(self, success: bool, processing_time_ms: float):
        """Update statistics with new request"""
        self.total_requests += 1
        self.last_request_time = datetime.now().isoformat()
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Update running average
        if self.total_requests == 1:
            self.average_processing_time_ms = processing_time_ms
        else:
            self.average_processing_time_ms = (
                (self.average_processing_time_ms * (self.total_requests - 1) + processing_time_ms) 
                / self.total_requests
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            'component_name': self.component_name,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate_percent': round(self.calculate_success_rate(), 2),
            'average_processing_time_ms': round(self.average_processing_time_ms, 2),
            'last_request_time': self.last_request_time
        }

# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def create_error_response(
    request_id: str, 
    query: str, 
    error_message: str, 
    execution_time_ms: float = 0.0
) -> SchemaSearchResponse:
    """
    Create standardized error response
    """
    return SchemaSearchResponse(
        request_id=request_id,
        query=query,
        status='error',
        error=error_message,
        execution_time_ms=execution_time_ms,
        processing_chain=['error_handler'],
        metadata={
            'error_type': 'processing_error',
            'timestamp': datetime.now().isoformat()
        }
    )

def create_success_response(
    request_id: str,
    query: str,
    data: Dict[str, Any],
    execution_time_ms: float = 0.0,
    nlp_insights: Optional[Dict[str, Any]] = None,
    processing_chain: List[str] = None # type: ignore
) -> SchemaSearchResponse:
    """
    Create standardized success response
    """
    return SchemaSearchResponse(
        request_id=request_id,
        query=query,
        status='success',
        data=data,
        execution_time_ms=execution_time_ms,
        nlp_insights=nlp_insights,
        processing_chain=processing_chain or ['integration_pipeline'],
        metadata={
            'response_type': 'successful_processing',
            'timestamp': datetime.now().isoformat()
        }
    )

def validate_schema_search_request(request: SchemaSearchRequest) -> Optional[str]:
    """
    Validate schema search request
    Returns error message if invalid, None if valid
    """
    if not request.query or not request.query.strip():
        return "Query cannot be empty"
    
    if request.max_results <= 0:
        return "max_results must be greater than 0"
    
    if not (0.0 <= request.min_confidence <= 1.0):
        return "min_confidence must be between 0.0 and 1.0"
    
    if not request.request_id:
        return "request_id is required"
    
    return None

# =====================================================
# TYPE ALIASES FOR CLARITY
# =====================================================

# Type aliases for common data structures
ComponentHealthMap = Dict[str, ComponentHealth]
ProcessingStatisticsMap = Dict[str, ProcessingStatistics]
IntentClassificationResult = Dict[str, Any]
SchemaContextData = Dict[str, Any]
NLPContextData = Dict[str, Any]

# =====================================================
# VERSION INFORMATION
# =====================================================

DATA_MODELS_VERSION = "1.0.0"
INTEGRATION_API_VERSION = "v1"
SUPPORTED_REQUEST_TYPES = [rt.value for rt in RequestType]
SUPPORTED_QUERY_INTENTS = [qi.value for qi in QueryIntent]
