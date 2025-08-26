"""
Core Exception Handling for NLP Processor
Custom exception hierarchy for banking domain NLP processing
Provides analyst-friendly error messages and comprehensive error handling
"""

import logging
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from datetime import datetime


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for analyst reporting"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for system monitoring"""
    NLP_PROCESSING = "nlp_processing"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    DATA_ACCESS = "data_access"


class NLPProcessorBaseException(Exception):
    """
    Base exception for all NLP Processor errors
    Provides consistent error handling and analyst-friendly messages
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.NLP_PROCESSING,
        details: Optional[Dict[str, Any]] = None,
        analyst_message: Optional[str] = None,
        suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.category = category
        self.details = details or {}
        self.analyst_message = analyst_message or self._generate_analyst_message()
        self.suggestions = suggestions or []
        self.timestamp = datetime.now().isoformat()
        
        # Log the exception
        self._log_exception()
    
    def _generate_analyst_message(self) -> str:
        """Generate user-friendly message for analysts"""
        return f"An error occurred while processing your query. Please contact support if the issue persists."
    
    def _log_exception(self) -> None:
        """Log exception with appropriate level based on severity"""
        log_message = f"{self.error_code}: {self.message}"
        if self.details:
            log_message += f" | Details: {self.details}"
            
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif self.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "analyst_message": self.analyst_message,
            "severity": self.severity.value,
            "category": self.category.value,
            "details": self.details,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp
        }


# =============================================================================
# NLP-SPECIFIC EXCEPTIONS
# =============================================================================

class IntentClassificationError(NLPProcessorBaseException):
    """Exception raised when intent classification fails"""
    
    def __init__(self, query_text: str, confidence_score: Optional[float] = None, **kwargs):
        self.query_text = query_text
        self.confidence_score = confidence_score
        
        message = f"Failed to classify intent for query: '{query_text[:100]}...'"
        if confidence_score is not None:
            message += f" | Confidence: {confidence_score:.2f}"
            
        details = {
            "query_text": query_text,
            "confidence_score": confidence_score,
            "component": "intent_classifier"
        }
        
        analyst_message = "Unable to understand the type of analysis you're requesting. Please rephrase your query with more specific business terms."
        
        suggestions = [
            "Try using specific banking terms like 'customers', 'loans', 'deposits'",
            "Be more specific about what data you want to analyze",
            "Use phrases like 'show me', 'calculate', or 'find all'"
        ]
        
        super().__init__(
            message=message,
            category=ErrorCategory.NLP_PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            analyst_message=analyst_message,
            suggestions=suggestions,
            **kwargs
        )


class EntityExtractionError(NLPProcessorBaseException):
    """Exception raised when entity extraction fails"""
    
    def __init__(self, query_text: str, entity_type: Optional[str] = None, **kwargs):
        self.query_text = query_text
        self.entity_type = entity_type
        
        message = f"Failed to extract entities from query: '{query_text[:100]}...'"
        if entity_type:
            message += f" | Expected entity type: {entity_type}"
            
        details = {
            "query_text": query_text,
            "entity_type": entity_type,
            "component": "entity_extractor"
        }
        
        analyst_message = "Unable to identify key business entities in your query. Please specify more clearly what you're looking for."
        
        suggestions = [
            "Include specific names like customer names, product types, or regions",
            "Use clear identifiers like account numbers or transaction IDs",
            "Specify time periods like 'last 30 days' or 'Q1 2024'"
        ]
        
        super().__init__(
            message=message,
            category=ErrorCategory.NLP_PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            analyst_message=analyst_message,
            suggestions=suggestions,
            **kwargs
        )


class QueryAmbiguityError(NLPProcessorBaseException):
    """Exception raised when query is too ambiguous to process"""
    
    def __init__(self, query_text: str, ambiguous_terms: Optional[List[str]] = None, **kwargs):
        self.query_text = query_text
        self.ambiguous_terms = ambiguous_terms or []
        
        message = f"Query is too ambiguous to process: '{query_text[:100]}...'"
        if self.ambiguous_terms:
            message += f" | Ambiguous terms: {', '.join(self.ambiguous_terms)}"
            
        details = {
            "query_text": query_text,
            "ambiguous_terms": self.ambiguous_terms,
            "component": "query_analyzer"
        }
        
        analyst_message = "Your query has multiple possible interpretations. Please be more specific about what you want to analyze."
        
        suggestions = [
            "Specify the exact time period you're interested in",
            "Clarify which specific metrics or data you need",
            "Use more specific business terminology"
        ]
        
        super().__init__(
            message=message,
            category=ErrorCategory.NLP_PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            analyst_message=analyst_message,
            suggestions=suggestions,
            **kwargs
        )


class TemporalProcessingError(NLPProcessorBaseException):
    """Exception raised when temporal expressions cannot be processed"""
    
    def __init__(self, temporal_expression: str, **kwargs):
        self.temporal_expression = temporal_expression
        
        message = f"Failed to process temporal expression: '{temporal_expression}'"
        
        details = {
            "temporal_expression": temporal_expression,
            "component": "temporal_processor"
        }
        
        analyst_message = "Unable to understand the time period in your query. Please use standard date formats."
        
        suggestions = [
            "Use formats like 'last 30 days', 'Q1 2024', or 'January 2024'",
            "Specify exact dates like '2024-01-01 to 2024-01-31'",
            "Use relative terms like 'yesterday', 'last week', 'this month'"
        ]
        
        super().__init__(
            message=message,
            category=ErrorCategory.NLP_PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            analyst_message=analyst_message,
            suggestions=suggestions,
            **kwargs
        )


# =============================================================================
# INTEGRATION EXCEPTIONS
# =============================================================================

class SchemaSearcherError(NLPProcessorBaseException):
    """Exception raised when schema searcher integration fails"""
    
    def __init__(self, operation: str, schema_element: Optional[str] = None, **kwargs):
        self.operation = operation
        self.schema_element = schema_element
        
        message = f"Schema searcher operation failed: {operation}"
        if schema_element:
            message += f" | Schema element: {schema_element}"
            
        details = {
            "operation": operation,
            "schema_element": schema_element,
            "component": "schema_searcher_bridge"
        }
        
        analyst_message = "Unable to find the requested data in the database schema. The information might not be available."
        
        suggestions = [
            "Check if the data you're looking for exists in the system",
            "Try using different table or column names",
            "Contact your database administrator for schema information"
        ]
        
        super().__init__(
            message=message,
            category=ErrorCategory.INTEGRATION,
            severity=ErrorSeverity.HIGH,
            details=details,
            analyst_message=analyst_message,
            suggestions=suggestions,
            **kwargs
        )


class XMLManagerError(NLPProcessorBaseException):
    """Exception raised when XML Manager integration fails"""
    
    def __init__(self, xml_field: Optional[str] = None, operation: Optional[str] = None, **kwargs):
        self.xml_field = xml_field
        self.operation = operation
        
        message = f"XML Manager operation failed"
        if operation:
            message += f": {operation}"
        if xml_field:
            message += f" | XML field: {xml_field}"
            
        details = {
            "xml_field": xml_field,
            "operation": operation,
            "component": "xml_manager_bridge"
        }
        
        analyst_message = "Unable to access XML data fields. Some requested information might not be available."
        
        suggestions = [
            "Verify that the XML data source is accessible",
            "Try requesting different data fields",
            "Contact system administrator if XML data is critical"
        ]
        
        super().__init__(
            message=message,
            category=ErrorCategory.INTEGRATION,
            severity=ErrorSeverity.HIGH,
            details=details,
            analyst_message=analyst_message,
            suggestions=suggestions,
            **kwargs
        )


class PromptBuilderError(NLPProcessorBaseException):
    """Exception raised when prompt builder integration fails"""
    
    def __init__(self, prompt_type: Optional[str] = None, **kwargs):
        self.prompt_type = prompt_type
        
        message = f"Prompt builder operation failed"
        if prompt_type:
            message += f" for prompt type: {prompt_type}"
            
        details = {
            "prompt_type": prompt_type,
            "component": "prompt_builder_bridge"
        }
        
        analyst_message = "Unable to generate the SQL query for your request. Please try rephrasing your question."
        
        suggestions = [
            "Simplify your query and try again",
            "Break down complex requests into smaller parts",
            "Use more standard business terminology"
        ]
        
        super().__init__(
            message=message,
            category=ErrorCategory.INTEGRATION,
            severity=ErrorSeverity.HIGH,
            details=details,
            analyst_message=analyst_message,
            suggestions=suggestions,
            **kwargs
        )


# =============================================================================
# PERFORMANCE EXCEPTIONS
# =============================================================================

class TimeoutError(NLPProcessorBaseException):
    """Exception raised when processing exceeds time limits"""
    
    def __init__(self, operation: str, timeout_seconds: float, **kwargs):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        
        message = f"Operation '{operation}' timed out after {timeout_seconds} seconds"
        
        details = {
            "operation": operation,
            "timeout_seconds": timeout_seconds,
            "component": "performance_monitor"
        }
        
        analyst_message = f"Your query is taking too long to process. Please try a simpler query or contact support."
        
        suggestions = [
            "Try breaking down complex queries into smaller parts",
            "Reduce the date range for your analysis",
            "Be more specific to reduce the amount of data processed"
        ]
        
        super().__init__(
            message=message,
            category=ErrorCategory.PERFORMANCE,
            severity=ErrorSeverity.HIGH,
            details=details,
            analyst_message=analyst_message,
            suggestions=suggestions,
            **kwargs
        )


class PerformanceLimitError(NLPProcessorBaseException):
    """Exception raised when performance limits are exceeded"""
    
    def __init__(self, limit_type: str, current_value: float, limit_value: float, **kwargs):
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value
        
        message = f"Performance limit exceeded for {limit_type}: {current_value} > {limit_value}"
        
        details = {
            "limit_type": limit_type,
            "current_value": current_value,
            "limit_value": limit_value,
            "component": "performance_monitor"
        }
        
        analyst_message = f"Your query requires more resources than currently available. Please try a simpler analysis."
        
        suggestions = [
            "Reduce the scope of your analysis",
            "Try processing data in smaller chunks",
            "Contact system administrator to increase resource limits"
        ]
        
        super().__init__(
            message=message,
            category=ErrorCategory.PERFORMANCE,
            severity=ErrorSeverity.HIGH,
            details=details,
            analyst_message=analyst_message,
            suggestions=suggestions,
            **kwargs
        )


class MemoryError(NLPProcessorBaseException):
    """Exception raised when memory usage exceeds limits"""
    
    def __init__(self, current_memory_mb: float, limit_memory_mb: float, **kwargs):
        self.current_memory_mb = current_memory_mb
        self.limit_memory_mb = limit_memory_mb
        
        message = f"Memory usage exceeded: {current_memory_mb:.2f}MB > {limit_memory_mb:.2f}MB"
        
        details = {
            "current_memory_mb": current_memory_mb,
            "limit_memory_mb": limit_memory_mb,
            "component": "performance_monitor"
        }
        
        analyst_message = "Your query requires too much memory to process. Please try a smaller analysis."
        
        suggestions = [
            "Reduce the date range for your query",
            "Limit the number of columns or tables in your analysis",
            "Try processing data in smaller batches"
        ]
        
        super().__init__(
            message=message,
            category=ErrorCategory.PERFORMANCE,
            severity=ErrorSeverity.HIGH,
            details=details,
            analyst_message=analyst_message,
            suggestions=suggestions,
            **kwargs
        )


# =============================================================================
# BUSINESS LOGIC EXCEPTIONS
# =============================================================================

class DomainMappingError(NLPProcessorBaseException):
    """Exception raised when business domain mapping fails"""
    
    def __init__(self, business_term: str, domain: str = "banking", **kwargs):
        self.business_term = business_term
        self.domain = domain
        
        message = f"Failed to map business term '{business_term}' in {domain} domain"
        
        details = {
            "business_term": business_term,
            "domain": domain,
            "component": "domain_mapper"
        }
        
        analyst_message = f"The term '{business_term}' is not recognized in our banking terminology. Please use standard banking terms."
        
        suggestions = [
            "Use standard banking terms like 'customer', 'account', 'loan', 'deposit'",
            "Check the business glossary for accepted terminology",
            "Try synonyms or alternative phrasing"
        ]
        
        super().__init__(
            message=message,
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            analyst_message=analyst_message,
            suggestions=suggestions,
            **kwargs
        )


class ValidationError(NLPProcessorBaseException):
    """Exception raised when validation fails"""
    
    def __init__(self, validation_type: str, failed_rules: Optional[List[str]] = None, **kwargs):
        self.validation_type = validation_type
        self.failed_rules = failed_rules or []
        
        message = f"Validation failed for {validation_type}"
        if self.failed_rules:
            message += f" | Failed rules: {', '.join(self.failed_rules)}"
            
        details = {
            "validation_type": validation_type,
            "failed_rules": self.failed_rules,
            "component": "validation_engine"
        }
        
        analyst_message = "Your query doesn't meet the required validation criteria. Please check your input."
        
        suggestions = [
            "Ensure all required parameters are provided",
            "Check that date ranges are valid",
            "Verify that numerical values are within acceptable limits"
        ]
        
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            analyst_message=analyst_message,
            suggestions=suggestions,
            **kwargs
        )


# =============================================================================
# DATA ACCESS EXCEPTIONS
# =============================================================================

class DataAccessError(NLPProcessorBaseException):
    """Exception raised when data access fails"""
    
    def __init__(self, data_source: str, operation: Optional[str] = None, **kwargs):
        self.data_source = data_source
        self.operation = operation
        
        message = f"Data access failed for source: {data_source}"
        if operation:
            message += f" | Operation: {operation}"
            
        details = {
            "data_source": data_source,
            "operation": operation,
            "component": "data_access"
        }
        
        analyst_message = f"Unable to access the requested data source. The information might not be available right now."
        
        suggestions = [
            "Check if the data source is available",
            "Try accessing different data or time periods",
            "Contact your database administrator if the issue persists"
        ]
        
        super().__init__(
            message=message,
            category=ErrorCategory.DATA_ACCESS,
            severity=ErrorSeverity.HIGH,
            details=details,
            analyst_message=analyst_message,
            suggestions=suggestions,
            **kwargs
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def handle_exception(e: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Global exception handler that converts any exception to standardized format
    
    Args:
        e: The exception to handle
        context: Additional context information
        
    Returns:
        Standardized error dictionary
    """
    if isinstance(e, NLPProcessorBaseException):
        error_dict = e.to_dict()
    else:
        # Handle unexpected exceptions
        error_dict = {
            "error_code": "UnexpectedError",
            "message": str(e),
            "analyst_message": "An unexpected error occurred. Please contact support.",
            "severity": ErrorSeverity.HIGH.value,
            "category": ErrorCategory.NLP_PROCESSING.value,
            "details": {"original_exception": type(e).__name__, "context": context or {}},
            "suggestions": ["Contact technical support", "Try again later"],
            "timestamp": datetime.now().isoformat()
        }
    
    if context:
        error_dict["details"].update(context)
    
    return error_dict


def get_exception_for_component(component_name: str, error_message: str, **kwargs) -> NLPProcessorBaseException:
    """
    Factory function to create appropriate exception based on component
    
    Args:
        component_name: Name of the component where error occurred
        error_message: Error message
        **kwargs: Additional parameters for specific exception types
        
    Returns:
        Appropriate exception instance
    """
    component_exceptions = {
        "intent_classifier": IntentClassificationError,
        "entity_extractor": EntityExtractionError,
        "temporal_processor": TemporalProcessingError,
        "schema_searcher": SchemaSearcherError,
        "xml_manager": XMLManagerError,
        "prompt_builder": PromptBuilderError,
        "domain_mapper": DomainMappingError,
        "validation_engine": ValidationError,
    }
    
    exception_class = component_exceptions.get(component_name, NLPProcessorBaseException)
    return exception_class(message=error_message, **kwargs)


# =============================================================================
# EXCEPTION REGISTRY
# =============================================================================

class ExceptionRegistry:
    """Registry to track and analyze exception patterns"""
    
    def __init__(self):
        self.exception_counts: Dict[str, int] = {}
        self.recent_exceptions: List[Dict[str, Any]] = []
        self.max_recent_exceptions = 100
    
    def register_exception(self, exception: NLPProcessorBaseException) -> None:
        """Register an exception occurrence"""
        error_code = exception.error_code
        
        # Update counts
        self.exception_counts[error_code] = self.exception_counts.get(error_code, 0) + 1
        
        # Add to recent exceptions
        self.recent_exceptions.append({
            "error_code": error_code,
            "message": exception.message,
            "severity": exception.severity.value,
            "category": exception.category.value,
            "timestamp": exception.timestamp
        })
        
        # Keep only recent exceptions
        if len(self.recent_exceptions) > self.max_recent_exceptions:
            self.recent_exceptions.pop(0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get exception statistics"""
        most_common = None
        if self.exception_counts:
            most_common = max(self.exception_counts.items(), key=lambda x: x[1])
        
        return {
            "total_exceptions": sum(self.exception_counts.values()),
            "exception_counts": self.exception_counts,
            "most_common": most_common,
            "recent_count": len(self.recent_exceptions)
        }


# Global exception registry instance
exception_registry = ExceptionRegistry()
