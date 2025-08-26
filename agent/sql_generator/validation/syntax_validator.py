"""
Enhanced SQL Syntax Validator - Validates SQL syntax with XML support
Provides comprehensive validation for SQL queries including XML operations
ENHANCED: Added detailed logging and relaxed XML validation options
"""

import re
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from agent.sql_generator.validation.models import ValidationResult

# SQL parsing libraries (optional, fallback to regex if not available)
try:
    import sqlparse
    SQLPARSE_AVAILABLE = True
except ImportError:
    SQLPARSE_AVAILABLE = False
    
try:
    import sqlalchemy
    from sqlalchemy import text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

class ValidationLevel(Enum):
    """Validation strictness levels"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"

class SQLDialect(Enum):
    """Supported SQL dialects"""
    MSSQL = "mssql"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    ORACLE = "oracle"
    GENERIC = "generic"

@dataclass
class XMLValidationResult:
    """XML-specific validation result"""
    has_xml_operations: bool = False
    xml_methods_used: List[str] = field(default_factory=list)
    xml_syntax_valid: bool = True
    xml_issues: List[str] = field(default_factory=list)
    xpath_expressions: List[str] = field(default_factory=list)
    xml_complexity_score: float = 0.0

@dataclass
class SyntaxValidationResult:
    """Detailed syntax validation result"""
    is_valid: bool
    syntax_score: float
    semantic_score: float
    performance_score: float
    overall_score: float
    
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # XML validation results
    xml_validation: Optional[XMLValidationResult] = None
    
    # Detailed analysis
    validation_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_response_validation_result(self):
        """Convert to response model ValidationResult"""
        from ..models.response_models import ValidationResult
        
        return ValidationResult(
            is_valid=self.is_valid,
            syntax_score=self.syntax_score,
            semantic_score=self.semantic_score,
            performance_score=self.performance_score,
            overall_score=self.overall_score,
            issues=self.issues,
            warnings=self.warnings,
            suggestions=self.suggestions,
            xml_validation_passed=self.xml_validation.xml_syntax_valid if self.xml_validation else True,
            xml_syntax_issues=self.xml_validation.xml_issues if self.xml_validation else [],
            xml_operations_detected=self.xml_validation.xml_methods_used if self.xml_validation else [],
            validation_details=self.validation_details
        )

class EnhancedSQLSyntaxValidator:
    """ENHANCED: SQL syntax validator with comprehensive logging and flexible validation"""
    
    def __init__(self, 
                 dialect: SQLDialect = SQLDialect.MSSQL, 
                 validation_level: ValidationLevel = ValidationLevel.STANDARD,
                 relax_xml_validation: bool = False,
                 enable_detailed_logging: bool = True):
        self.dialect = dialect
        self.validation_level = validation_level
        self.relax_xml_validation = relax_xml_validation
        self.enable_detailed_logging = enable_detailed_logging
        
        # ENHANCED: Setup detailed logging
        self.logger = logging.getLogger("SQLSyntaxValidator")
        if self.enable_detailed_logging and not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
        
        # Security patterns - these operations are blocked
        self.blocked_operations = [
            r'\b(DROP|DELETE|INSERT|UPDATE|CREATE|ALTER|TRUNCATE|EXEC|EXECUTE)\b',
            r'\b(GRANT|REVOKE|DENY)\b',
            r'\b(BACKUP|RESTORE)\b',
            r'\bxp_\w+\b',  # Extended stored procedures
            r'\bsp_\w+\b',  # System stored procedures (selective)
        ]
        
        # Dangerous functions
        self.dangerous_functions = [
            r'\bOPENROWSET\b',
            r'\bOPENDATASOURCE\b',
            r'\bBULK\s+INSERT\b',
            r'\bxp_cmdshell\b'
        ]
        
        # XML operation patterns
        self.xml_patterns = {
            'value': r'\.value\s*\(',
            'query': r'\.query\s*\(',
            'exist': r'\.exist\s*\(',
            'nodes': r'\.nodes\s*\(',
            'modify': r'\.modify\s*\('
        }
        
        # XPath pattern
        self.xpath_pattern = r'["\'][\(\)/\w\[\]@\s:.-]+["\']'
        
        # Common SQL patterns for validation
        self.sql_keywords = [
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER',
            'GROUP BY', 'ORDER BY', 'HAVING', 'UNION', 'INTERSECT', 'EXCEPT',
            'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'AS', 'DISTINCT', 'TOP', 'LIMIT'
        ]
        
        # Performance warning patterns
        self.performance_warnings = [
            (r'SELECT\s+\*', "Avoid SELECT * in production queries"),
            (r'(?i)WHERE\s+\w+\s+LIKE\s+["\']%\w+%["\']', "Leading wildcard in LIKE may be slow"),
            (r'(?i)WHERE\s+FUNCTION\s*\(', "Functions in WHERE clause prevent index usage"),
            (r'(?i)ORDER\s+BY\s+\w+\s+DESC\s+LIMIT\s+\d+', "Consider indexing for ORDER BY with LIMIT")
        ]
        
        self.logger.info(f"SQL Validator initialized: dialect={dialect.value}, level={validation_level.value}, relax_xml={relax_xml_validation}")
    
    def validate_sql(self, sql_query: str) -> SyntaxValidationResult:
        """ENHANCED: Comprehensive SQL validation with detailed logging"""
        
        start_time = time.time()
        self.logger.debug(f"Starting SQL validation for query: {sql_query[:100]}...")
        
        try:
            # Clean and normalize SQL
            cleaned_sql = self._clean_sql(sql_query)
            
            if not cleaned_sql.strip():
                self.logger.warning("Empty SQL query detected")
                return SyntaxValidationResult(
                    is_valid=False,
                    syntax_score=0.0,
                    semantic_score=0.0,
                    performance_score=0.0,
                    overall_score=0.0,
                    issues=["Empty SQL query"],
                    validation_details={"validation_time_ms": 0}
                )
            
            # Security validation (highest priority)
            self.logger.debug("Performing security validation...")
            security_issues = self._validate_security(cleaned_sql)
            if security_issues:
                self.logger.error(f"Security validation failed: {security_issues}")
                return SyntaxValidationResult(
                    is_valid=False,
                    syntax_score=0.0,
                    semantic_score=0.0,
                    performance_score=0.0,
                    overall_score=0.0,
                    issues=security_issues,
                    validation_details={"validation_time_ms": (time.time() - start_time) * 1000}
                )
            
            # Syntax validation
            self.logger.debug("Performing syntax validation...")
            syntax_score, syntax_issues, syntax_warnings = self._validate_syntax(cleaned_sql)
            
            # Semantic validation
            self.logger.debug("Performing semantic validation...")
            semantic_score, semantic_issues, semantic_warnings = self._validate_semantics(cleaned_sql)
            
            # Performance validation
            self.logger.debug("Performing performance validation...")
            performance_score, performance_warnings, performance_suggestions = self._validate_performance(cleaned_sql)
            
            # XML validation
            self.logger.debug("Performing XML validation...")
            xml_validation = self._validate_xml_operations(cleaned_sql)
            
            # Calculate overall scores
            overall_score = (syntax_score + semantic_score + performance_score) / 3
            is_valid = syntax_score >= 0.7 and len(syntax_issues) == 0
            
            # ENHANCED: More lenient validation for production use
            if not is_valid and self.validation_level == ValidationLevel.BASIC:
                # In basic mode, only fail on critical issues
                critical_issues = [issue for issue in syntax_issues 
                                 if 'blocked operation' in issue.lower() 
                                 or 'injection' in issue.lower()
                                 or 'unbalanced' in issue.lower()]
                
                if not critical_issues:
                    is_valid = True
                    self.logger.info("Validation passed in basic mode despite warnings")
            
            # Combine all issues and warnings
            all_issues = syntax_issues + semantic_issues
            all_warnings = syntax_warnings + semantic_warnings + performance_warnings
            all_suggestions = performance_suggestions
            
            validation_time = (time.time() - start_time) * 1000
            
            self.logger.info(f"SQL validation complete: valid={is_valid}, score={overall_score:.2f}, time={validation_time:.1f}ms")
            
            if all_issues:
                self.logger.warning(f"Validation issues found: {all_issues}")
            
            if all_warnings:
                self.logger.debug(f"Validation warnings: {all_warnings}")
            
            return SyntaxValidationResult(
                is_valid=is_valid,
                syntax_score=syntax_score,
                semantic_score=semantic_score,
                performance_score=performance_score,
                overall_score=overall_score,
                issues=all_issues,
                warnings=all_warnings,
                suggestions=all_suggestions,
                xml_validation=xml_validation,
                validation_details={
                    "validation_time_ms": validation_time,
                    "sql_length": len(cleaned_sql),
                    "dialect": self.dialect.value,
                    "validation_level": self.validation_level.value,
                    "sqlparse_used": SQLPARSE_AVAILABLE,
                    "sqlalchemy_used": SQLALCHEMY_AVAILABLE
                }
            )
            
        except Exception as e:
            self.logger.error(f"SQL validation failed with exception: {str(e)}")
            return SyntaxValidationResult(
                is_valid=False,
                syntax_score=0.0,
                semantic_score=0.0,
                performance_score=0.0,
                overall_score=0.0,
                issues=[f"Validation error: {str(e)}"],
                validation_details={"validation_time_ms": (time.time() - start_time) * 1000}
            )
    
    def validate_with_xml_support(self, sql_query: str) -> 'ValidationResult':
        """ENHANCED: Validate SQL with XML support and return response model ValidationResult"""
        self.logger.debug(f"XML-supported validation requested for: {sql_query[:50]}...")
        syntax_result = self.validate_sql(sql_query)
        return syntax_result.to_response_validation_result()
    
    def _clean_sql(self, sql: str) -> str:
        """Clean and normalize SQL query"""
        if not sql:
            return ""
        
        # Remove comments
        sql = re.sub(r'--.*?$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        
        # Normalize whitespace
        sql = re.sub(r'\s+', ' ', sql)
        sql = sql.strip()
        
        return sql
    
    def _validate_security(self, sql: str) -> List[str]:
        """ENHANCED: Validate SQL for security issues with detailed logging"""
        issues = []
        sql_upper = sql.upper()
        
        self.logger.debug("Checking for blocked operations...")
        # Check for blocked operations
        for pattern in self.blocked_operations:
            if re.search(pattern, sql_upper, re.IGNORECASE):
                issue = f"Blocked operation detected: {pattern}"
                issues.append(issue)
                self.logger.warning(f"Security violation: {issue}")
        
        self.logger.debug("Checking for dangerous functions...")
        # Check for dangerous functions
        for pattern in self.dangerous_functions:
            if re.search(pattern, sql_upper, re.IGNORECASE):
                issue = f"Dangerous function detected: {pattern}"
                issues.append(issue)
                self.logger.warning(f"Security violation: {issue}")
        
        self.logger.debug("Checking for SQL injection patterns...")
        # Check for SQL injection patterns
        injection_patterns = [
            r"'\s*;\s*--",
            r"'\s*OR\s+'\d+'\s*=\s*'\d+'",
            r"'\s*UNION\s+",
            r"'\s*AND\s+\d+=\d+\s*--"
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                issues.append("Potential SQL injection pattern detected")
                self.logger.error("SQL injection pattern detected")
                break
        
        if not issues:
            self.logger.debug("Security validation passed")
        
        return issues
    
    def _validate_syntax(self, sql: str) -> Tuple[float, List[str], List[str]]:
        """ENHANCED: Validate SQL syntax with detailed logging"""
        issues = []
        warnings = []
        score = 1.0
        
        self.logger.debug(f"Validating SQL syntax for: {sql[:50]}...")
        
        # Basic syntax checks
        if not sql.strip():
            issues.append("Empty SQL query")
            return 0.0, issues, warnings
        
        # Check for basic SQL structure
        sql_upper = sql.upper()
        
        # Must start with SELECT for read-only operations
        if not sql_upper.strip().startswith('SELECT'):
            issues.append("Only SELECT statements are allowed")
            score -= 0.5
            self.logger.warning("Non-SELECT statement detected")
        
        # Check parentheses balance
        open_parens = sql.count('(')
        close_parens = sql.count(')')
        if open_parens != close_parens:
            issues.append("Unbalanced parentheses")
            score -= 0.3
            self.logger.warning(f"Unbalanced parentheses: {open_parens} open, {close_parens} close")
        
        # Check quote balance
        single_quotes = sql.count("'") - sql.count("\\'")
        if single_quotes % 2 != 0:
            issues.append("Unbalanced single quotes")
            score -= 0.3
            self.logger.warning("Unbalanced single quotes detected")
        
        # Use sqlparse if available for detailed syntax checking
        if SQLPARSE_AVAILABLE:
            try:
                parsed = sqlparse.parse(sql)
                if not parsed:
                    issues.append("SQL parsing failed")
                    score -= 0.5
                    self.logger.warning("SQL parsing failed")
                else:
                    self.logger.debug("SQL parsing successful")
                    # Additional syntax validation using sqlparse
                    for statement in parsed:
                        if statement.get_type() != 'SELECT':
                            warnings.append(f"Non-SELECT statement detected: {statement.get_type()}")
            except Exception as e:
                warnings.append(f"SQL parsing warning: {str(e)}")
                score -= 0.1
                self.logger.debug(f"SQL parsing warning: {e}")
        
        # Check for required FROM clause in SELECT statements
        if 'SELECT' in sql_upper and 'FROM' not in sql_upper:
            # Allow scalar selects or functions
            if not re.search(r'SELECT\s+\d+|SELECT\s+[A-Z_]+\s*\(', sql_upper):
                warnings.append("SELECT statement without FROM clause")
        
        self.logger.debug(f"Syntax validation complete: score={score}, issues={len(issues)}, warnings={len(warnings)}")
        return max(score, 0.0), issues, warnings
    
    def _validate_semantics(self, sql: str) -> Tuple[float, List[str], List[str]]:
        """ENHANCED: Validate SQL semantics with detailed logging"""
        issues = []
        warnings = []
        score = 1.0
        
        sql_upper = sql.upper()
        self.logger.debug("Performing semantic validation...")
        
        # Check for common semantic issues
        
        # GROUP BY semantic checks
        if 'GROUP BY' in sql_upper:
            self.logger.debug("Validating GROUP BY semantics")
            # Simple check for SELECT columns in GROUP BY
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_upper)
            if select_match:
                select_columns = select_match.group(1)
                if '*' not in select_columns and 'COUNT(' not in select_columns and 'SUM(' not in select_columns:
                    # More sophisticated GROUP BY validation would go here
                    pass
        
        # JOIN semantic checks
        join_count = len(re.findall(r'\bJOIN\b', sql_upper))
        if join_count > 5:
            warnings.append(f"Many JOINs detected ({join_count}), consider query optimization")
            score -= 0.1
            self.logger.debug(f"Many JOINs detected: {join_count}")
        
        # Check for potential Cartesian products
        from_tables = len(re.findall(r'FROM\s+\w+', sql_upper))
        join_conditions = len(re.findall(r'ON\s+\w+\.\w+\s*=\s*\w+\.\w+', sql_upper))
        
        if from_tables > 1 and join_conditions == 0 and 'JOIN' not in sql_upper:
            warnings.append("Potential Cartesian product detected")
            score -= 0.2
            self.logger.warning("Potential Cartesian product detected")
        
        # Check for ORDER BY without LIMIT (performance concern)
        if 'ORDER BY' in sql_upper and 'LIMIT' not in sql_upper and 'TOP' not in sql_upper:
            warnings.append("ORDER BY without LIMIT may return large result sets")
            self.logger.debug("ORDER BY without LIMIT detected")
        
        self.logger.debug(f"Semantic validation complete: score={score}, warnings={len(warnings)}")
        return max(score, 0.0), issues, warnings
    
    def _validate_performance(self, sql: str) -> Tuple[float, List[str], List[str]]:
        """ENHANCED: Validate SQL for performance issues with detailed logging"""
        warnings = []
        suggestions = []
        score = 1.0
        
        self.logger.debug("Performing performance validation...")
        
        # Check performance warning patterns
        for pattern, message in self.performance_warnings:
            if re.search(pattern, sql, re.IGNORECASE):
                warnings.append(message)
                score -= 0.1
                self.logger.debug(f"Performance warning: {message}")
        
        # Additional performance checks
        sql_upper = sql.upper()
        
        # Check for SELECT *
        if 'SELECT *' in sql_upper:
            suggestions.append("Specify exact columns instead of SELECT * for better performance")
            self.logger.debug("SELECT * detected")
        
        # Check for functions in WHERE clause
        if re.search(r'WHERE\s+\w+\s*\(\s*\w+\s*\)', sql_upper):
            suggestions.append("Avoid functions on columns in WHERE clause to use indexes")
            self.logger.debug("Functions in WHERE clause detected")
        
        # Check for LIKE with leading wildcard
        if re.search(r"LIKE\s+['\"]%", sql, re.IGNORECASE):
            suggestions.append("Leading wildcards in LIKE prevent index usage")
            self.logger.debug("Leading wildcard in LIKE detected")
        
        # Check for nested subqueries
        subquery_count = sql.count('(SELECT')
        if subquery_count > 2:
            suggestions.append("Consider JOINs instead of nested subqueries for better performance")
            score -= 0.1
            self.logger.debug(f"Multiple nested subqueries detected: {subquery_count}")
        
        self.logger.debug(f"Performance validation complete: score={score}, suggestions={len(suggestions)}")
        return max(score, 0.0), warnings, suggestions
    
    def _validate_xml_operations(self, sql: str) -> XMLValidationResult:
        """ENHANCED: Validate XML operations with relaxed mode support"""
        
        xml_result = XMLValidationResult()
        self.logger.debug("Validating XML operations...")
        
        # Check for XML methods
        xml_methods_found = []
        for method, pattern in self.xml_patterns.items():
            if re.search(pattern, sql, re.IGNORECASE):
                xml_methods_found.append(method)
        
        if xml_methods_found:
            xml_result.has_xml_operations = True
            xml_result.xml_methods_used = xml_methods_found
            self.logger.debug(f"XML operations detected: {xml_methods_found}")
        
        # Extract XPath expressions
        xpath_matches = re.findall(self.xpath_pattern, sql)
        xml_result.xpath_expressions = xpath_matches
        if xpath_matches:
            self.logger.debug(f"XPath expressions found: {len(xpath_matches)}")
        
        # Calculate XML complexity score
        complexity_score = 0.0
        complexity_score += len(xml_methods_found) * 0.2  # Each XML method adds complexity
        complexity_score += len(xpath_matches) * 0.1     # Each XPath expression adds complexity
        
        # Check for complex XPath patterns
        for xpath in xpath_matches:
            if '[' in xpath and ']' in xpath:  # Predicates
                complexity_score += 0.1
            if '//' in xpath:  # Descendant axis
                complexity_score += 0.1
            if '@' in xpath:   # Attributes
                complexity_score += 0.05
        
        xml_result.xml_complexity_score = min(complexity_score, 1.0)
        
        # Validate XML syntax
        xml_issues = []
        
        if self.relax_xml_validation:
            # ENHANCED: Relaxed XML validation for production use
            self.logger.debug("Using relaxed XML validation")
            xml_result.xml_syntax_valid = True
            xml_issues = []  # Only critical XML issues in relaxed mode
        else:
            # Strict XML validation
            self.logger.debug("Using strict XML validation")
            
            # Check for proper XML method syntax
            for method in xml_methods_found:
                pattern = self.xml_patterns[method]
                matches = re.findall(pattern + r'[^)]*\)', sql, re.IGNORECASE)
                for match in matches:
                    # Basic validation - ensure closing parenthesis
                    if not match.endswith(')'):
                        xml_issues.append(f"Unclosed {method} method call")
            
            # Check XPath syntax (basic)
            for xpath in xpath_matches:
                if xpath.count('(') != xpath.count(')'):
                    xml_issues.append(f"Unbalanced parentheses in XPath: {xpath}")
                if xpath.count('[') != xpath.count(']'):
                    xml_issues.append(f"Unbalanced brackets in XPath: {xpath}")
            
            xml_result.xml_syntax_valid = len(xml_issues) == 0
        
        xml_result.xml_issues = xml_issues
        
        if xml_issues:
            self.logger.warning(f"XML validation issues: {xml_issues}")
        else:
            self.logger.debug("XML validation passed")
        
        return xml_result
    
    def quick_validate(self, sql: str) -> bool:
        """ENHANCED: Quick validation for basic syntax checking with logging"""
        try:
            self.logger.debug(f"Quick validation for: {sql[:50]}...")
            
            # Security check first
            security_issues = self._validate_security(sql)
            if security_issues:
                self.logger.debug("Quick validation failed: security issues")
                return False
            
            # Basic syntax check
            cleaned_sql = self._clean_sql(sql)
            if not cleaned_sql:
                self.logger.debug("Quick validation failed: empty SQL")
                return False
            
            # Check basic structure
            sql_upper = cleaned_sql.upper()
            if not sql_upper.startswith('SELECT'):
                self.logger.debug("Quick validation failed: not a SELECT statement")
                return False
            
            # Check parentheses balance
            if sql.count('(') != sql.count(')'):
                self.logger.debug("Quick validation failed: unbalanced parentheses")
                return False
            
            self.logger.debug("Quick validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Quick validation error: {e}")
            return False
    
    def get_validation_summary(self, sql: str) -> Dict[str, Any]:
        """ENHANCED: Get validation summary with key metrics and logging"""
        self.logger.debug("Generating validation summary...")
        result = self.validate_sql(sql)
        
        summary = {
            "is_valid": result.is_valid,
            "overall_score": result.overall_score,
            "syntax_score": result.syntax_score,
            "semantic_score": result.semantic_score,
            "performance_score": result.performance_score,
            "issue_count": len(result.issues),
            "warning_count": len(result.warnings),
            "suggestion_count": len(result.suggestions),
            "has_xml_operations": result.xml_validation.has_xml_operations if result.xml_validation else False,
            "xml_complexity": result.xml_validation.xml_complexity_score if result.xml_validation else 0.0,
            "validation_time_ms": result.validation_details.get("validation_time_ms", 0)
        }
        
        self.logger.debug(f"Validation summary generated: {summary}")
        return summary

# Utility functions
def validate_sql_quick(sql: str, dialect: SQLDialect = SQLDialect.MSSQL) -> bool:
    """Quick SQL validation utility function"""
    validator = EnhancedSQLSyntaxValidator(dialect=dialect)
    return validator.quick_validate(sql)

def get_sql_validation_report(sql: str, dialect: SQLDialect = SQLDialect.MSSQL) -> Dict[str, Any]:
    """Get comprehensive SQL validation report"""
    validator = EnhancedSQLSyntaxValidator(dialect=dialect)
    return validator.get_validation_summary(sql)

# Example usage and testing
if __name__ == "__main__":
    # Test the enhanced validator
    validator = EnhancedSQLSyntaxValidator(
        relax_xml_validation=True,
        enable_detailed_logging=True
    )
    
    test_queries = [
        "SELECT * FROM customers WHERE id = 1",
        "SELECT name, email FROM customers WHERE region = 'North'",
        "SELECT c.name, c.CTPT_XML.value('(/Customer/CIF)[1]', 'varchar(20)') as CIF FROM customers c",
        "DROP TABLE customers",  # Should fail security check
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        result = validator.validate_sql(query)
        print(f"Valid: {result.is_valid}")
        print(f"Score: {result.overall_score:.2f}")
        if result.issues:
            print(f"Issues: {result.issues}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        if result.xml_validation and result.xml_validation.has_xml_operations:
            print(f"XML Operations: {result.xml_validation.xml_methods_used}")
