"""
Validation models for SQL syntax validation
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationIssue:
    """Individual validation issue"""
    message: str
    severity: ValidationSeverity
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    issue_type: str = "syntax"
    suggestion: Optional[str] = None

@dataclass
class ValidationResult:
    """SQL validation result"""
    is_valid: bool
    issues: List[ValidationIssue]
    corrected_sql: Optional[str] = None
    confidence_score: float = 1.0
    validation_time_ms: float = 0.0
    
    def has_errors(self) -> bool:
        """Check if there are any error-level issues"""
        return any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                  for issue in self.issues)
    
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues"""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)
    
    def get_error_messages(self) -> List[str]:
        """Get all error messages"""
        return [issue.message for issue in self.issues 
                if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
    
    def get_warning_messages(self) -> List[str]:
        """Get all warning messages"""
        return [issue.message for issue in self.issues 
                if issue.severity == ValidationSeverity.WARNING]
