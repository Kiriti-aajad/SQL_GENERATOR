"""
Error handling and recovery for SQL Executor
Handles READ-ONLY violations, connection issues, timeouts, and user guidance
"""

import re
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from enum import Enum

from .models import (
    ExecutionResult, ErrorType, ExecutionStatus, QueryType,
    create_error_result, create_blocked_result, create_empty_result
)
from .config import get_security_config, get_executor_config

# Set up logging
logger = logging.getLogger(__name__) # type: ignore

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryAction(Enum):
    """Possible recovery actions"""
    RETRY = "retry"
    MODIFY_QUERY = "modify_query"
    CONTACT_ADMIN = "contact_admin"
    USE_ALTERNATIVE = "use_alternative"
    NO_ACTION = "no_action"

class ErrorPattern:
    """Error pattern definition for intelligent detection"""
    
    def __init__(self, pattern: str, error_type: ErrorType, severity: ErrorSeverity, 
                 suggestions: List[str], recovery_action: RecoveryAction):
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.error_type = error_type
        self.severity = severity
        self.suggestions = suggestions
        self.recovery_action = recovery_action

class ReadOnlyValidator:
    """Validator for READ-ONLY database restrictions"""
    
    def __init__(self):
        self.security_config = get_security_config()
        self.blocked_patterns = self._compile_blocked_patterns()
        self.allowed_patterns = self._compile_allowed_patterns()
    
    def _compile_blocked_patterns(self) -> List[re.Pattern]:
        """Compile blocked keyword patterns"""
        patterns = []
        for keyword in self.security_config.blocked_keywords:
            # Create pattern that matches keyword at word boundaries
            pattern = rf'\b{re.escape(keyword)}\b'
            patterns.append(re.compile(pattern, re.IGNORECASE))
        return patterns
    
    def _compile_allowed_patterns(self) -> List[re.Pattern]:
        """Compile allowed keyword patterns"""
        patterns = []
        for keyword in self.security_config.allowed_patterns:
            pattern = rf'\b{re.escape(keyword)}\b'
            patterns.append(re.compile(pattern, re.IGNORECASE))
        return patterns
    
    def validate_query(self, sql_query: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate query against READ-ONLY restrictions
        Returns: (is_valid, blocked_keywords, warnings)
        """
        blocked_keywords = []
        warnings = []
        
        # Check for blocked keywords
        for pattern in self.blocked_patterns:
            matches = pattern.findall(sql_query)
            if matches:
                blocked_keywords.extend(matches)
        
        # Additional validation for common dangerous patterns
        dangerous_patterns = [
            (r'DELETE\s+(?!.*WHERE)', "DELETE without WHERE clause detected"),
            (r'UPDATE\s+.*(?!WHERE)', "UPDATE without WHERE clause detected"),
            (r'TRUNCATE\s+TABLE', "TRUNCATE TABLE operation detected"),
            (r'DROP\s+(TABLE|DATABASE|INDEX)', "DROP operation detected"),
            (r'ALTER\s+(TABLE|DATABASE)', "ALTER operation detected"),
            (r'EXEC\s*\(', "Dynamic SQL execution detected"),
            (r'SP_\w+', "Stored procedure execution detected")
        ]
        
        for pattern, warning in dangerous_patterns:
            if re.search(pattern, sql_query, re.IGNORECASE):
                warnings.append(warning)
                if "DELETE" in warning or "UPDATE" in warning:
                    blocked_keywords.append(pattern.split('\\')[0])
        
        is_valid = len(blocked_keywords) == 0
        return is_valid, blocked_keywords, warnings

class ConnectionErrorHandler:
    """Handler for database connection errors"""
    
    def __init__(self):
        self.config = get_executor_config()
        self.error_patterns = self._initialize_connection_patterns()
    
    def _initialize_connection_patterns(self) -> List[ErrorPattern]:
        """Initialize connection error patterns"""
        return [
            ErrorPattern(
                pattern=r"timeout|timed out",
                error_type=ErrorType.TIMEOUT_ERROR,
                severity=ErrorSeverity.MEDIUM,
                suggestions=[
                    "The query took too long to execute",
                    "Consider adding more specific WHERE conditions",
                    "Try breaking complex queries into smaller parts",
                    "Contact administrator if this persists"
                ],
                recovery_action=RecoveryAction.MODIFY_QUERY
            ),
            ErrorPattern(
                pattern=r"connection.*failed|cannot connect|server.*unavailable",
                error_type=ErrorType.CONNECTION_ERROR,
                severity=ErrorSeverity.HIGH,
                suggestions=[
                    "Database server is currently unavailable",
                    "Check your network connection",
                    "Wait a moment and try again",
                    "Contact database administrator if problem persists"
                ],
                recovery_action=RecoveryAction.RETRY
            ),
            ErrorPattern(
                pattern=r"login failed|authentication|access denied|permission",
                error_type=ErrorType.PERMISSION_ERROR,
                severity=ErrorSeverity.HIGH,
                suggestions=[
                    "Database credentials may be incorrect",
                    "Your account may not have necessary permissions",
                    "Contact database administrator for access",
                    "Verify your connection credentials"
                ],
                recovery_action=RecoveryAction.CONTACT_ADMIN
            ),
            ErrorPattern(
                pattern=r"invalid object name|table.*not found|doesn't exist",
                error_type=ErrorType.SYNTAX_ERROR,
                severity=ErrorSeverity.MEDIUM,
                suggestions=[
                    "The specified table or column doesn't exist",
                    "Check spelling of table and column names",
                    "Verify the table exists in the current database",
                    "Use the schema browser to find available tables"
                ],
                recovery_action=RecoveryAction.MODIFY_QUERY
            ),
            ErrorPattern(
                pattern=r"syntax error|incorrect syntax",
                error_type=ErrorType.SYNTAX_ERROR,
                severity=ErrorSeverity.MEDIUM,
                suggestions=[
                    "SQL query contains syntax errors",
                    "Check for missing commas, quotes, or parentheses",
                    "Verify SQL keyword spelling",
                    "Review query structure and try again"
                ],
                recovery_action=RecoveryAction.MODIFY_QUERY
            )
        ]
    
    def analyze_error(self, error_message: str) -> Tuple[ErrorType, ErrorSeverity, List[str], RecoveryAction]:
        """Analyze error message and provide recommendations"""
        
        # Try to match error patterns
        for pattern in self.error_patterns:
            if pattern.pattern.search(error_message):
                return pattern.error_type, pattern.severity, pattern.suggestions, pattern.recovery_action
        
        # Default unknown error handling
        return (
            ErrorType.UNKNOWN_ERROR,
            ErrorSeverity.MEDIUM,
            [
                "An unexpected error occurred during query execution",
                "Please review your query syntax",
                "Contact support if the problem persists"
            ],
            RecoveryAction.NO_ACTION
        )

class RetryHandler:
    """Handler for automatic retry logic"""
    
    def __init__(self):
        self.config = get_executor_config()
        self.max_retries = self.config.max_retry_attempts
        self.base_delay = self.config.retry_delay_seconds
    
    async def retry_with_backoff(self, operation, *args, **kwargs):
        """Execute operation with exponential backoff retry"""
        
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return await operation(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                # Don't retry on certain error types
                if self._is_non_retryable_error(e):
                    break
                
                # Calculate delay with exponential backoff
                delay = self.base_delay * (2 ** attempt)
                
                if attempt < self.max_retries - 1:  # Don't wait after last attempt
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
        
        # If we get here, all retries failed
        raise last_exception # type: ignore
    
    def _is_non_retryable_error(self, error: Exception) -> bool:
        """Check if error should not be retried"""
        error_message = str(error).lower()
        
        non_retryable_patterns = [
            'syntax error',
            'permission denied',
            'invalid object name',
            'login failed',
            'access denied'
        ]
        
        return any(pattern in error_message for pattern in non_retryable_patterns)

class ErrorHandler:
    """Main error handler orchestrator"""
    
    def __init__(self):
        self.readonly_validator = ReadOnlyValidator()
        self.connection_handler = ConnectionErrorHandler()
        self.retry_handler = RetryHandler()
        self.config = get_executor_config()
    
    def validate_readonly_query(self, sql_query: str, request_id: Optional[str] = None) -> Optional[ExecutionResult]:
        """
        Validate query for READ-ONLY restrictions
        Returns ExecutionResult if blocked, None if allowed
        """
        is_valid, blocked_keywords, warnings = self.readonly_validator.validate_query(sql_query)
        
        if not is_valid:
            blocked_operations = ', '.join(set(blocked_keywords))
            logger.warning(f"Blocked READ-ONLY violation: {blocked_operations} in query")
            
            return create_blocked_result(
                blocked_operation=blocked_operations,
                request_id=request_id
            )
        
        # Log warnings but allow execution
        if warnings and self.config.enable_detailed_errors:
            logger.warning(f"Query warnings: {'; '.join(warnings)}")
        
        return None  # Query is allowed
    
    def handle_execution_error(self, error: Exception, sql_query: str, 
                              request_id: Optional[str] = None) -> ExecutionResult:
        """Handle execution errors with intelligent analysis"""
        
        error_message = str(error)
        logger.error(f"Query execution error: {error_message}")
        
        # Analyze error for type and suggestions
        error_type, severity, suggestions, recovery_action = self.connection_handler.analyze_error(error_message)
        
        # Add query-specific suggestions
        enhanced_suggestions = self._enhance_suggestions(suggestions, sql_query, error_type)
        
        # Create error result
        result = create_error_result(
            error_type=error_type,
            error_message=error_message,
            suggestions=enhanced_suggestions,
            request_id=request_id
        )
        
        # Add severity and recovery information
        result.error_details = f"[{severity.value.upper()}] {error_message}"
        
        return result
    
    def handle_empty_result(self, sql_query: str, execution_time_ms: int,
                           request_id: Optional[str] = None) -> ExecutionResult:
        """Handle empty result sets with intelligent suggestions"""
        
        # Create base empty result
        result = create_empty_result(
            execution_time_ms=execution_time_ms,
            request_id=request_id
        )
        
        # Add query-specific suggestions
        enhanced_suggestions = self._analyze_empty_result_causes(sql_query)
        result.suggestions.extend(enhanced_suggestions)
        
        return result
    
    def handle_timeout_error(self, sql_query: str, timeout_seconds: int,
                            request_id: Optional[str] = None) -> ExecutionResult:
        """Handle query timeout with optimization suggestions"""
        
        suggestions = [
            f"Query exceeded {timeout_seconds} second timeout limit",
            "Consider adding more specific WHERE conditions to limit results",
            "Break complex queries into smaller, simpler parts",
            "Add indexes on frequently queried columns (contact administrator)"
        ]
        
        # Analyze query for specific timeout causes
        timeout_suggestions = self._analyze_timeout_causes(sql_query)
        suggestions.extend(timeout_suggestions)
        
        return ExecutionResult(
            status=ExecutionStatus.TIMEOUT,
            success=False,
            message=f"Query execution timed out after {timeout_seconds} seconds",
            error_type=ErrorType.TIMEOUT_ERROR,
            error_details=f"Query exceeded maximum execution time of {timeout_seconds} seconds",
            suggestions=suggestions,
            request_id=request_id # type: ignore
        )
    
    def _enhance_suggestions(self, base_suggestions: List[str], sql_query: str, 
                            error_type: ErrorType) -> List[str]:
        """Enhance suggestions based on query analysis"""
        
        enhanced = base_suggestions.copy()
        
        # Add query-specific suggestions based on error type
        if error_type == ErrorType.TIMEOUT_ERROR:
            enhanced.extend(self._analyze_timeout_causes(sql_query))
        
        elif error_type == ErrorType.SYNTAX_ERROR:
            enhanced.extend(self._analyze_syntax_issues(sql_query))
        
        elif error_type == ErrorType.CONNECTION_ERROR:
            enhanced.append("Verify database server is running and accessible")
        
        return enhanced
    
    def _analyze_empty_result_causes(self, sql_query: str) -> List[str]:
        """Analyze potential causes of empty results"""
        
        suggestions = []
        query_upper = sql_query.upper()
        
        # Check for restrictive WHERE conditions
        if 'WHERE' in query_upper:
            suggestions.append("Your WHERE conditions might be too restrictive")
            
            # Check for date filters
            if any(keyword in query_upper for keyword in ['DATE', 'DATETIME', 'TIMESTAMP']):
                suggestions.append("Check if your date range filters are appropriate")
            
            # Check for equality conditions
            if '=' in sql_query and 'LIKE' not in query_upper:
                suggestions.append("Consider using LIKE instead of exact matches for text fields")
        
        # Check for JOINs that might eliminate results
        if 'INNER JOIN' in query_upper:
            suggestions.append("INNER JOINs might be eliminating all results - consider LEFT JOIN")
        
        # Check for specific table patterns
        if 'tblCounterparty' in sql_query:
            suggestions.append("Verify that counterparty data exists for your criteria")
        
        if 'XML' in query_upper:
            suggestions.append("Check if XML data exists and is properly formatted")
        
        return suggestions
    
    def _analyze_timeout_causes(self, sql_query: str) -> List[str]:
        """Analyze potential causes of query timeouts"""
        
        suggestions = []
        query_upper = sql_query.upper()
        
        # Check for missing WHERE clause
        if 'WHERE' not in query_upper:
            suggestions.append("Add WHERE clause to limit the number of rows processed")
        
        # Check for complex JOINs
        join_count = query_upper.count('JOIN')
        if join_count > 3:
            suggestions.append(f"Query has {join_count} JOINs which may impact performance")
        
        # Check for XML operations
        if '.value(' in sql_query or '.exist(' in sql_query:
            suggestions.append("XML operations can be slow - consider pre-processing XML data")
        
        # Check for wildcards
        if 'SELECT *' in query_upper:
            suggestions.append("Avoid SELECT * - specify only needed columns")
        
        # Check for LIKE with leading wildcards
        if "'%%" in sql_query or '"%' in sql_query:
            suggestions.append("LIKE patterns starting with % are slow - avoid if possible")
        
        return suggestions
    
    def _analyze_syntax_issues(self, sql_query: str) -> List[str]:
        """Analyze potential syntax issues"""
        
        suggestions = []
        
        # Check for common syntax issues
        if sql_query.count('(') != sql_query.count(')'):
            suggestions.append("Check for unmatched parentheses")
        
        if sql_query.count("'") % 2 != 0:
            suggestions.append("Check for unmatched single quotes")
        
        if sql_query.count('"') % 2 != 0:
            suggestions.append("Check for unmatched double quotes")
        
        return suggestions
    
    async def execute_with_error_handling(self, operation, sql_query: str, 
                                         request_id: Optional[str] = None):
        """Execute operation with comprehensive error handling"""
        
        try:
            # First validate for READ-ONLY restrictions
            readonly_result = self.validate_readonly_query(sql_query, request_id)
            if readonly_result:
                return readonly_result
            
            # Execute with retry logic for retryable errors
            return await self.retry_handler.retry_with_backoff(operation)
            
        except asyncio.TimeoutError:
            return self.handle_timeout_error(sql_query, self.config.default_timeout, request_id)
            
        except Exception as e:
            return self.handle_execution_error(e, sql_query, request_id)

# Convenience functions
def validate_readonly_query(sql_query: str) -> Optional[ExecutionResult]:
    """Quick READ-ONLY validation"""
    handler = ErrorHandler()
    return handler.validate_readonly_query(sql_query)

def handle_error(error: Exception, sql_query: str) -> ExecutionResult:
    """Quick error handling"""
    handler = ErrorHandler()
    return handler.handle_execution_error(error, sql_query)

def handle_empty_result(sql_query: str, execution_time: int) -> ExecutionResult:
    """Quick empty result handling"""
    handler = ErrorHandler()
    return handler.handle_empty_result(sql_query, execution_time)

# Export main classes and functions
__all__ = [
    'ErrorHandler', 'ReadOnlyValidator', 'ConnectionErrorHandler', 'RetryHandler',
    'ErrorSeverity', 'RecoveryAction', 'ErrorPattern',
    'validate_readonly_query', 'handle_error', 'handle_empty_result'
]
