"""
SQL Corrector - Attempts to fix common SQL syntax and semantic issues
Provides intelligent correction suggestions and automated fixes
"""

import re
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import the correct classes from syntax_validator
from ..validation.syntax_validator import EnhancedSQLSyntaxValidator, ValidationLevel, SQLDialect


class CorrectionLevel(Enum):
    """Correction aggressiveness levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class CorrectionType(Enum):
    """Types of corrections that can be applied"""
    SYNTAX_FIX = "syntax_fix"
    SECURITY_FIX = "security_fix"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SEMANTIC_CORRECTION = "semantic_correction"
    XML_CORRECTION = "xml_correction"


@dataclass
class CorrectionAction:
    """Individual correction action"""
    correction_type: CorrectionType
    description: str
    original_pattern: str
    corrected_pattern: str
    confidence: float
    position: Optional[int] = None
    reasoning: str = ""


@dataclass
class CorrectionResult:
    """Result of SQL correction attempt"""
    was_corrected: bool
    original_sql: str
    corrected_sql: Optional[str] = None
    corrections_applied: List[str] = field(default_factory=list)
    correction_confidence: float = 0.0
    
    # Enhanced fields for NGROK compatibility
    correction_attempts: int = 1
    ngrok_retry_corrections: bool = False
    
    correction_details: Dict[str, Any] = field(default_factory=dict)
    correction_actions: List[CorrectionAction] = field(default_factory=list)


class SQLCorrector:
    """SQL corrector with intelligent fixing capabilities"""
    
    def __init__(self, 
                 correction_level: CorrectionLevel = CorrectionLevel.MODERATE,
                 dialect: SQLDialect = SQLDialect.MSSQL):
        self.correction_level = correction_level
        self.dialect = dialect
        self.logger = logging.getLogger("SQLCorrector")
        self.validator = EnhancedSQLSyntaxValidator(dialect=dialect)
        
        # Correction patterns (pattern, replacement, confidence, reasoning)
        self.correction_patterns = [
            # Common syntax fixes
            (r'\bSELCT\b', 'SELECT', 0.95, "Fix SELECT spelling"),
            (r'\bFROM\s+WHERE\b', 'FROM table_name WHERE', 0.8, "Add missing table name"),
            (r'\bWHERE\s+AND\b', 'WHERE condition AND', 0.7, "Fix WHERE clause syntax"),
            (r'\bGROUP\s+BYy\b', 'GROUP BY', 0.9, "Fix GROUP BY spelling"),
            (r'\bORDER\s+BYY\b', 'ORDER BY', 0.9, "Fix ORDER BY spelling"),
            
            # Quote fixes
            (r"([^'])''([^'])", r"\1'\2", 0.8, "Fix double single quotes"),
            (r'([^"])"([^"])', r'\1"\2', 0.8, "Fix quote escaping"),
            
            # Common XML corrections
            (r'\.value\s*\(\s*([^,)]+)\s*\)', r'.value(\1, \'varchar(max)\')', 0.7, "Add data type to XML value method"),
            (r'\.query\s*\(\s*([^)]+)\s*\)', r'.query(\1)', 0.8, "Fix XML query method syntax"),
            
            # Performance optimizations
            (r'\bSELECT\s+\*\s+FROM\s+(\w+)\s+WHERE\b', r'SELECT column1, column2 FROM \1 WHERE', 0.6, "Replace SELECT * with specific columns"),
            
            # Security fixes (convert dangerous operations to safe SELECT)
            (r'\b(DROP|DELETE|INSERT|UPDATE|CREATE|ALTER)\s+.*', 'SELECT 1 -- Converted from unsafe operation', 0.9, "Convert unsafe operation to SELECT"),
        ]
        
        # XML-specific correction patterns
        self.xml_correction_patterns = [
            (r'\.value\s*\(\s*(["\'][^"\']*["\'])\s*\)', r'.value(\1, \'varchar(max)\')', 0.8, "Add missing data type to XML value()"),
            (r'\.query\s*\(\s*(["\'][^"\']*["\'])\s*\)', r'.query(\1)', 0.9, "Fix XML query() syntax"),
            (r'\.exist\s*\(\s*(["\'][^"\']*["\'])\s*\)', r'.exist(\1)', 0.9, "Fix XML exist() syntax"),
            (r'\.nodes\s*\(\s*(["\'][^"\']*["\'])\s*\)', r'.nodes(\1)', 0.9, "Fix XML nodes() syntax"),
        ]
        
        # Parentheses balancing
        self.bracket_patterns = [
            (r'\(\s*SELECT', '(SELECT', 0.9, "Fix subquery parentheses"),
            (r'SELECT\s*\)', 'SELECT)', 0.9, "Fix closing parentheses"),
        ]
    
    def correct_sql(self, sql_query: str, max_attempts: int = 3) -> CorrectionResult:
        """Attempt to correct SQL query issues"""
        
        if not sql_query or not sql_query.strip():
            return CorrectionResult(
                was_corrected=False,
                original_sql=sql_query,
                correction_details={"error": "Empty SQL query"}
            )
        
        original_sql = sql_query
        current_sql = sql_query
        corrections_applied = []
        correction_actions = []
        attempt = 0
        
        self.logger.info(f"Starting SQL correction for query: {sql_query[:100]}...")
        
        while attempt < max_attempts:
            attempt += 1
            
            # Validate current SQL
            validation_result = self.validator.validate_sql(current_sql)
            
            if validation_result.is_valid and validation_result.overall_score > 0.8:
                # SQL is valid and good quality
                break
            
            # Apply corrections based on validation issues
            corrected_sql, applied_corrections, actions = self._apply_corrections(
                current_sql, validation_result
            )
            
            if corrected_sql == current_sql:
                # No more corrections possible
                break
            
            current_sql = corrected_sql
            corrections_applied.extend(applied_corrections)
            correction_actions.extend(actions)
        
        # Calculate overall correction confidence
        confidence = self._calculate_correction_confidence(
            original_sql, current_sql, correction_actions
        )
        
        was_corrected = current_sql != original_sql
        
        return CorrectionResult(
            was_corrected=was_corrected,
            original_sql=original_sql,
            corrected_sql=current_sql if was_corrected else None,
            corrections_applied=corrections_applied,
            correction_confidence=confidence,
            correction_attempts=attempt,
            correction_details={
                "validation_attempts": attempt,
                "final_validation_score": self.validator.validate_sql(current_sql).overall_score,
                "correction_level": self.correction_level.value,
                "total_corrections": len(corrections_applied)
            },
            correction_actions=correction_actions
        )
    
    def _apply_corrections(self, sql: str, validation_result) -> Tuple[str, List[str], List[CorrectionAction]]:
        """Apply corrections based on validation results"""
        
        corrected_sql = sql
        applied_corrections = []
        correction_actions = []
        
        # Apply syntax corrections
        corrected_sql, syntax_corrections, syntax_actions = self._apply_syntax_corrections(corrected_sql)
        applied_corrections.extend(syntax_corrections)
        correction_actions.extend(syntax_actions)
        
        # Apply XML corrections if XML operations detected
        if validation_result.xml_validation and validation_result.xml_validation.has_xml_operations:
            corrected_sql, xml_corrections, xml_actions = self._apply_xml_corrections(corrected_sql)
            applied_corrections.extend(xml_corrections)
            correction_actions.extend(xml_actions)
        
        # Apply security corrections
        corrected_sql, security_corrections, security_actions = self._apply_security_corrections(corrected_sql)
        applied_corrections.extend(security_corrections)
        correction_actions.extend(security_actions)
        
        # Apply performance optimizations (if aggressive mode)
        if self.correction_level == CorrectionLevel.AGGRESSIVE:
            corrected_sql, perf_corrections, perf_actions = self._apply_performance_corrections(corrected_sql)
            applied_corrections.extend(perf_corrections)
            correction_actions.extend(perf_actions)
        
        return corrected_sql, applied_corrections, correction_actions
    
    def _apply_syntax_corrections(self, sql: str) -> Tuple[str, List[str], List[CorrectionAction]]:
        """Apply syntax corrections"""
        
        corrected_sql = sql
        corrections = []
        actions = []
        
        # Apply pattern-based corrections
        for pattern, replacement, confidence, reasoning in self.correction_patterns:
            if re.search(pattern, corrected_sql, re.IGNORECASE):
                old_sql = corrected_sql
                corrected_sql = re.sub(pattern, replacement, corrected_sql, flags=re.IGNORECASE)
                
                if corrected_sql != old_sql:
                    correction_desc = f"Applied pattern correction: {reasoning}"
                    corrections.append(correction_desc)
                    
                    action = CorrectionAction(
                        correction_type=CorrectionType.SYNTAX_FIX,
                        description=correction_desc,
                        original_pattern=pattern,
                        corrected_pattern=replacement,
                        confidence=confidence,
                        reasoning=reasoning
                    )
                    actions.append(action)
        
        # Fix parentheses balance
        corrected_sql, paren_corrections, paren_actions = self._fix_parentheses_balance(corrected_sql)
        corrections.extend(paren_corrections)
        actions.extend(paren_actions)
        
        return corrected_sql, corrections, actions
    
    def _apply_xml_corrections(self, sql: str) -> Tuple[str, List[str], List[CorrectionAction]]:
        """Apply XML-specific corrections"""
        
        corrected_sql = sql
        corrections = []
        actions = []
        
        for pattern, replacement, confidence, reasoning in self.xml_correction_patterns:
            if re.search(pattern, corrected_sql, re.IGNORECASE):
                old_sql = corrected_sql
                corrected_sql = re.sub(pattern, replacement, corrected_sql, flags=re.IGNORECASE)
                
                if corrected_sql != old_sql:
                    correction_desc = f"XML correction: {reasoning}"
                    corrections.append(correction_desc)
                    
                    action = CorrectionAction(
                        correction_type=CorrectionType.XML_CORRECTION,
                        description=correction_desc,
                        original_pattern=pattern,
                        corrected_pattern=replacement,
                        confidence=confidence,
                        reasoning=reasoning
                    )
                    actions.append(action)
        
        return corrected_sql, corrections, actions
    
    def _apply_security_corrections(self, sql: str) -> Tuple[str, List[str], List[CorrectionAction]]:
        """Apply security corrections"""
        
        corrected_sql = sql
        corrections = []
        actions = []
        
        # Check for dangerous operations and convert them
        dangerous_patterns = [
            r'\b(DROP|DELETE|INSERT|UPDATE|CREATE|ALTER|TRUNCATE)\s+.*',
            r'\b(EXEC|EXECUTE)\s+.*',
            r'\b(GRANT|REVOKE|DENY)\s+.*'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, corrected_sql, re.IGNORECASE):
                old_sql = corrected_sql
                corrected_sql = re.sub(pattern, 'SELECT 1 -- Converted from unsafe operation', corrected_sql, flags=re.IGNORECASE)
                
                if corrected_sql != old_sql:
                    correction_desc = "Converted dangerous operation to safe SELECT"
                    corrections.append(correction_desc)
                    
                    action = CorrectionAction(
                        correction_type=CorrectionType.SECURITY_FIX,
                        description=correction_desc,
                        original_pattern=pattern,
                        corrected_pattern='SELECT 1 -- Converted from unsafe operation',
                        confidence=0.95,
                        reasoning="Security: Converted potentially dangerous operation"
                    )
                    actions.append(action)
        
        return corrected_sql, corrections, actions
    
    def _apply_performance_corrections(self, sql: str) -> Tuple[str, List[str], List[CorrectionAction]]:
        """Apply performance optimizations"""
        
        corrected_sql = sql
        corrections = []
        actions = []
        
        # Only apply if aggressive correction level
        if self.correction_level != CorrectionLevel.AGGRESSIVE:
            return corrected_sql, corrections, actions
        
        # Replace SELECT * with specific columns (example)
        if re.search(r'\bSELECT\s+\*\s+FROM\b', corrected_sql, re.IGNORECASE):
            old_sql = corrected_sql
            corrected_sql = re.sub(
                r'\bSELECT\s+\*\s+FROM\b', 
                'SELECT column1, column2 FROM', 
                corrected_sql, 
                flags=re.IGNORECASE
            )
            
            if corrected_sql != old_sql:
                correction_desc = "Replaced SELECT * with specific columns"
                corrections.append(correction_desc)
                
                action = CorrectionAction(
                    correction_type=CorrectionType.PERFORMANCE_OPTIMIZATION,
                    description=correction_desc,
                    original_pattern=r'\bSELECT\s+\*\s+FROM\b',
                    corrected_pattern='SELECT column1, column2 FROM',
                    confidence=0.6,
                    reasoning="Performance: Avoid SELECT * in production queries"
                )
                actions.append(action)
        
        return corrected_sql, corrections, actions
    
    def _fix_parentheses_balance(self, sql: str) -> Tuple[str, List[str], List[CorrectionAction]]:
        """Fix unbalanced parentheses"""
        
        corrections = []
        actions = []
        
        open_count = sql.count('(')
        close_count = sql.count(')')
        
        if open_count == close_count:
            return sql, corrections, actions
        
        corrected_sql = sql
        
        if open_count > close_count:
            # Add missing closing parentheses
            missing = open_count - close_count
            corrected_sql += ')' * missing
            
            correction_desc = f"Added {missing} missing closing parentheses"
            corrections.append(correction_desc)
            
            action = CorrectionAction(
                correction_type=CorrectionType.SYNTAX_FIX,
                description=correction_desc,
                original_pattern="unbalanced_parentheses",
                corrected_pattern=f"added_{missing}_closing",
                confidence=0.8,
                reasoning="Fixed unbalanced parentheses by adding closing parentheses"
            )
            actions.append(action)
            
        elif close_count > open_count:
            # Remove extra closing parentheses
            extra = close_count - open_count
            for _ in range(extra):
                corrected_sql = corrected_sql.rsplit(')', 1)[0]
            
            correction_desc = f"Removed {extra} extra closing parentheses"
            corrections.append(correction_desc)
            
            action = CorrectionAction(
                correction_type=CorrectionType.SYNTAX_FIX,
                description=correction_desc,
                original_pattern="unbalanced_parentheses",
                corrected_pattern=f"removed_{extra}_closing",
                confidence=0.7,
                reasoning="Fixed unbalanced parentheses by removing extra closing parentheses"
            )
            actions.append(action)
        
        return corrected_sql, corrections, actions
    
    def _calculate_correction_confidence(self, 
                                       original_sql: str, 
                                       corrected_sql: str, 
                                       actions: List[CorrectionAction]) -> float:
        """Calculate overall confidence in corrections"""
        
        if not actions:
            return 1.0  # No corrections needed
        
        # Weight corrections by their individual confidence
        total_weight = 0.0
        weighted_confidence = 0.0
        
        for action in actions:
            weight = 1.0
            # Higher weight for security fixes
            if action.correction_type == CorrectionType.SECURITY_FIX:
                weight = 2.0
            # Lower weight for aggressive performance optimizations
            elif action.correction_type == CorrectionType.PERFORMANCE_OPTIMIZATION:
                weight = 0.5
            
            total_weight += weight
            weighted_confidence += action.confidence * weight
        
        if total_weight == 0:
            return 0.5
        
        base_confidence = weighted_confidence / total_weight
        
        # Reduce confidence if too many corrections were needed
        correction_penalty = min(len(actions) * 0.1, 0.3)
        final_confidence = max(base_confidence - correction_penalty, 0.1)
        
        return min(final_confidence, 1.0)
    
    def suggest_corrections(self, sql: str) -> List[str]:
        """Suggest corrections without applying them"""
        
        suggestions = []
        validation_result = self.validator.validate_sql(sql)
        
        # Add suggestions based on validation issues
        if validation_result.issues:
            suggestions.extend([f"Fix issue: {issue}" for issue in validation_result.issues])
        
        if validation_result.warnings:
            suggestions.extend([f"Consider: {warning}" for warning in validation_result.warnings])
        
        # Add pattern-based suggestions
        for pattern, replacement, confidence, reasoning in self.correction_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                suggestions.append(f"Suggestion: {reasoning}")
        
        return suggestions
    
    def can_correct(self, sql: str) -> bool:
        """Check if SQL can potentially be corrected"""
        
        if not sql or not sql.strip():
            return False
        
        # Check if any correction patterns match
        for pattern, _, confidence, _ in self.correction_patterns:
            if re.search(pattern, sql, re.IGNORECASE) and confidence > 0.7:
                return True
        
        # Check for correctable XML issues
        for pattern, _, confidence, _ in self.xml_correction_patterns:
            if re.search(pattern, sql, re.IGNORECASE) and confidence > 0.7:
                return True
        
        # Check for parentheses imbalance
        if sql.count('(') != sql.count(')'):
            return True
        
        return False


# Utility functions
def correct_sql_simple(sql: str, correction_level: CorrectionLevel = CorrectionLevel.MODERATE) -> str:
    """Simple SQL correction utility function"""
    corrector = SQLCorrector(correction_level=correction_level)
    result = corrector.correct_sql(sql)
    return result.corrected_sql if result.was_corrected else sql # type: ignore


def get_correction_suggestions(sql: str) -> List[str]:
    """Get correction suggestions for SQL"""
    corrector = SQLCorrector()
    return corrector.suggest_corrections(sql)


# Example usage and testing
if __name__ == "__main__":
    # Test the corrector
    corrector = SQLCorrector()
    
    test_queries = [
        "SELCT * FROM customers WHERE id = 1",  # Typo
        "SELECT * FROM WHERE id = 1",  # Missing table
        "SELECT name FROM customers WHERE (",  # Unbalanced parentheses
        "DROP TABLE customers",  # Security issue
        "SELECT c.name, c.xml_column.value('(/Customer/CIF)[1]') FROM customers c",  # Missing XML data type
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        result = corrector.correct_sql(query)
        print(f"Was Corrected: {result.was_corrected}")
        if result.was_corrected:
            print(f"Corrected SQL: {result.corrected_sql}")
            print(f"Corrections: {result.corrections_applied}")
            print(f"Confidence: {result.correction_confidence:.2f}")
