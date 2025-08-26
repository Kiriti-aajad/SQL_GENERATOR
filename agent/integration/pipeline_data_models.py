"""
Pipeline Data Models for NLP-Schema Integration System
Standardized data contracts for seamless component compatibility
Addresses critical data format compatibility between pipeline stages

Author: Enhanced for Pipeline Compatibility
Version: 2.0.0 - PIPELINE DATA STANDARDIZATION  
Date: 2025-08-06
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataValidationLevel(Enum):
    STRICT = "strict"
    MODERATE = "moderate"
    RELAXED = "relaxed"

class PipelineStage(Enum):
    NLP_PROCESSING = "nlp_processing"
    SCHEMA_SEARCH = "schema_search"
    PROMPT_BUILDING = "prompt_building"
    SQL_GENERATION = "sql_generation"

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validation_level: DataValidationLevel = DataValidationLevel.MODERATE

@dataclass
class StandardizedSchemaContext:
    """
    Universal schema format for all pipeline stages
    Addresses schema_searcher → prompt_builder → sql_generator compatibility
    """
    # Core schema data
    tables: List[str]
    columns_by_table: Dict[str, List[Dict[str, Any]]]
    joins: List[Dict[str, Any]]
    
    # Quality metadata
    confidence_score: float = 0.8
    total_columns: int = 0
    has_xml_data: bool = False
    
    # NLP enhancement data
    nlp_insights: Optional[Dict[str, Any]] = field(default_factory=dict)
    predicted_tables: List[str] = field(default_factory=list)
    predicted_columns: List[str] = field(default_factory=list)
    
    # Processing metadata
    processing_time: float = 0.0
    search_method: str = "unknown"
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and calculations"""
        if self.total_columns == 0:
            self.total_columns = sum(len(cols) for cols in self.columns_by_table.values())
        
        # Validate data consistency
        self._validate_internal_consistency()
    
    def _validate_internal_consistency(self) -> ValidationResult:
        """Validate internal data consistency"""
        errors = []
        warnings = []
        
        # Check that all tables in columns_by_table are in tables list
        for table_name in self.columns_by_table.keys():
            if table_name not in self.tables:
                warnings.append(f"Table '{table_name}' in columns_by_table not in tables list")
        
        # Check confidence score range
        if not 0.0 <= self.confidence_score <= 1.0:
            errors.append(f"Confidence score {self.confidence_score} not in valid range [0.0, 1.0]")
        
        # Check predicted tables exist in tables
        for predicted_table in self.predicted_tables:
            if predicted_table not in self.tables:
                warnings.append(f"Predicted table '{predicted_table}' not found in available tables")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
    
    def to_prompt_builder_format(self) -> Dict[str, Any]:
        """
        Convert to prompt builder expected format
        CRITICAL: This fixes schema_searcher → prompt_builder compatibility
        """
        try:
            prompt_format = {
                # Core schema information
                "schema": {
                    "tables": self.tables,
                    "table_definitions": self.columns_by_table,
                    "relationships": self.joins,
                    "metadata": {
                        "total_columns": self.total_columns,
                        "has_xml": self.has_xml_data,
                        "confidence": self.confidence_score
                    }
                },
                
                # NLP enhancement context
                "nlp_context": self.nlp_insights,
                
                # Processing hints for prompt builder
                "processing_hints": {
                    "predicted_tables": self.predicted_tables,
                    "predicted_columns": self.predicted_columns,
                    "search_method": self.search_method,
                    "quality_level": "high" if self.confidence_score > 0.8 else "medium"
                },
                
                # Quality metadata
                "quality_indicators": {
                    "confidence_score": self.confidence_score,
                    "data_completeness": len(self.tables) > 0 and len(self.columns_by_table) > 0,
                    "processing_time": self.processing_time
                }
            }
            
            logger.debug(f"Schema context converted to prompt builder format: {len(self.tables)} tables")
            return prompt_format
            
        except Exception as e:
            logger.error(f"Failed to convert to prompt builder format: {e}")
            # Return minimal safe format
            return {
                "schema": {
                    "tables": self.tables,
                    "table_definitions": self.columns_by_table,
                    "relationships": self.joins
                },
                "nlp_context": self.nlp_insights or {},
                "error": f"Conversion error: {str(e)}"
            }
    
    def to_sql_generator_format(self) -> Dict[str, Any]:
        """
        Convert to SQL generator expected format
        CRITICAL: This enables SQL generator to receive proper schema context
        """
        try:
            sql_format = {
                # Schema context for SQL generation
                "schema_context": {
                    "available_tables": self.tables,
                    "table_columns": self.columns_by_table,
                    "table_joins": self.joins,
                    "xml_fields": self.has_xml_data,
                    "total_fields": self.total_columns
                },
                
                # Banking/NLP insights for context
                "banking_insights": self.nlp_insights,
                
                # Generation optimization hints
                "generation_hints": {
                    "focus_tables": self.predicted_tables,
                    "focus_columns": self.predicted_columns,
                    "confidence_level": self.confidence_score,
                    "complexity_indicator": "high" if len(self.joins) > 3 else "medium",
                    "has_xml_processing": self.has_xml_data
                },
                
                # Quality control information
                "quality_metadata": {
                    "schema_confidence": self.confidence_score,
                    "search_quality": self.search_method,
                    "data_freshness": self.processing_time,
                    "validation_passed": True
                }
            }
            
            logger.debug(f"Schema context converted to SQL generator format: {len(self.tables)} tables, {self.total_columns} columns")
            return sql_format
            
        except Exception as e:
            logger.error(f"Failed to convert to SQL generator format: {e}")
            # Return minimal safe format
            return {
                "schema_context": {
                    "available_tables": self.tables,
                    "table_columns": self.columns_by_table,
                    "table_joins": self.joins
                },
                "banking_insights": self.nlp_insights or {},
                "error": f"Conversion error: {str(e)}"
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for monitoring"""
        return {
            "tables_count": len(self.tables),
            "total_columns": self.total_columns,
            "joins_count": len(self.joins),
            "confidence_score": self.confidence_score,
            "has_nlp_insights": bool(self.nlp_insights),
            "predicted_elements": len(self.predicted_tables) + len(self.predicted_columns),
            "processing_time_ms": self.processing_time * 1000
        }

@dataclass
class StandardizedPromptResult:
    """
    Universal prompt format for SQL generator compatibility
    Addresses prompt_builder → sql_generator compatibility
    """
    # Core prompt data
    generated_prompt: str
    prompt_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Context preservation
    schema_context_used: Optional[StandardizedSchemaContext] = None
    nlp_context_applied: Dict[str, Any] = field(default_factory=dict)
    
    # Quality indicators
    prompt_confidence: float = 0.8
    generation_method: str = "standard"
    processing_time: float = 0.0
    
    # Optimization data
    prompt_optimizations: List[str] = field(default_factory=list)
    context_truncated: bool = False
    fallback_used: bool = False
    
    def __post_init__(self):
        """Validate prompt result"""
        self._validate_prompt_quality()
    
    def _validate_prompt_quality(self) -> ValidationResult:
        """Validate prompt quality and consistency"""
        errors = []
        warnings = []
        
        # Check prompt is not empty
        if not self.generated_prompt or not self.generated_prompt.strip():
            errors.append("Generated prompt is empty")
        
        # Check prompt length is reasonable
        if len(self.generated_prompt) > 10000:
            warnings.append(f"Prompt is very long ({len(self.generated_prompt)} chars)")
        elif len(self.generated_prompt) < 50:
            warnings.append(f"Prompt is very short ({len(self.generated_prompt)} chars)")
        
        # Check confidence score
        if not 0.0 <= self.prompt_confidence <= 1.0:
            errors.append(f"Prompt confidence {self.prompt_confidence} not in valid range")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
    
    def to_sql_generator_input(self) -> Dict[str, Any]:
        """
        Convert to SQL generator expected input format
        CRITICAL: This fixes prompt_builder → sql_generator compatibility
        """
        try:
            # Prepare banking insights from multiple sources
            banking_insights = {}
            
            # Merge NLP context and metadata
            if self.nlp_context_applied:
                banking_insights.update(self.nlp_context_applied)
            
            if self.prompt_metadata.get('nlp_insights'):
                banking_insights.update(self.prompt_metadata['nlp_insights'])
            
            # Extract schema context
            schema_context = {}
            if self.schema_context_used:
                schema_context = self.schema_context_used.to_sql_generator_format()
            elif self.prompt_metadata.get('schema_context'):
                schema_context = self.prompt_metadata['schema_context']
            
            # Prepare generation hints
            generation_hints = {
                "prompt_confidence": self.prompt_confidence,
                "generation_method": self.generation_method,
                "optimizations_applied": self.prompt_optimizations,
                "context_quality": "high" if not self.fallback_used else "medium",
                "processing_time": self.processing_time
            }
            
            sql_input = {
                # Primary prompt
                "prompt": self.generated_prompt,
                
                # Context information
                "banking_insights": banking_insights,
                "schema_context": schema_context.get("schema_context", {}),
                
                # Generation control
                "generation_hints": generation_hints,
                "quality_metadata": {
                    "prompt_confidence": self.prompt_confidence,
                    "context_truncated": self.context_truncated,
                    "fallback_used": self.fallback_used,
                    "validation_passed": True
                },
                
                # Additional metadata
                "prompt_metadata": self.prompt_metadata
            }
            
            logger.debug(f"Prompt result converted to SQL generator input: {len(self.generated_prompt)} chars")
            return sql_input
            
        except Exception as e:
            logger.error(f"Failed to convert prompt to SQL generator input: {e}")
            # Return minimal safe format
            return {
                "prompt": self.generated_prompt,
                "banking_insights": self.nlp_context_applied or {},
                "schema_context": {},
                "error": f"Conversion error: {str(e)}"
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get prompt summary for monitoring"""
        return {
            "prompt_length": len(self.generated_prompt),
            "confidence_score": self.prompt_confidence,
            "generation_method": self.generation_method,
            "optimizations_count": len(self.prompt_optimizations),
            "context_truncated": self.context_truncated,
            "fallback_used": self.fallback_used,
            "processing_time_ms": self.processing_time * 1000
        }

@dataclass
class StandardizedSQLRequest:
    """
    Universal SQL generation request format
    Ensures consistent input to SQL generators
    """
    # Core request data
    prompt: str
    banking_insights: Dict[str, Any] = field(default_factory=dict)
    schema_context: Dict[str, Any] = field(default_factory=dict)
    
    # Generation options
    target_llm: str = "mistral"
    generation_options: Dict[str, Any] = field(default_factory=dict)
    quality_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Context and metadata
    nlp_insights: Dict[str, Any] = field(default_factory=dict)
    processing_context: Dict[str, Any] = field(default_factory=dict)
    
    # Request tracking
    request_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Initialize request with defaults"""
        if not self.request_id:
            self.request_id = f"sql_req_{int(datetime.now().timestamp())}"
        
        # Set default generation options
        if not self.generation_options:
            self.generation_options = {
                "temperature": 0.1,
                "max_tokens": 2000,
                "timeout": 30.0
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for SQL generator"""
        return {
            "prompt": self.prompt,
            "banking_insights": self.banking_insights,
            "schema_context": self.schema_context,
            "target_llm": self.target_llm,
            "generation_options": self.generation_options,
            "quality_requirements": self.quality_requirements,
            "nlp_insights": self.nlp_insights,
            "processing_context": self.processing_context,
            "request_id": self.request_id,
            "timestamp": self.timestamp
        }
    
    def validate(self) -> ValidationResult:
        """Validate SQL request completeness"""
        errors = []
        warnings = []
        
        if not self.prompt or not self.prompt.strip():
            errors.append("SQL request prompt is empty")
        
        if not self.schema_context:
            warnings.append("No schema context provided")
        
        if not self.banking_insights:
            warnings.append("No banking insights provided")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

@dataclass
class PipelineValidationReport:
    """
    Comprehensive validation report for pipeline data flow
    """
    stage: PipelineStage
    validation_results: List[ValidationResult] = field(default_factory=list)
    data_quality_score: float = 0.0
    processing_time: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    
    def add_validation(self, result: ValidationResult):
        """Add validation result"""
        self.validation_results.append(result)
        self._update_quality_score()
    
    def _update_quality_score(self):
        """Update overall quality score"""
        if not self.validation_results:
            self.data_quality_score = 0.0
            return
        
        valid_count = sum(1 for r in self.validation_results if r.is_valid)
        self.data_quality_score = valid_count / len(self.validation_results)
    
    def is_healthy(self) -> bool:
        """Check if pipeline stage is healthy"""
        return self.data_quality_score >= 0.8 and all(r.is_valid for r in self.validation_results)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        total_errors = sum(len(r.errors) for r in self.validation_results)
        total_warnings = sum(len(r.warnings) for r in self.validation_results)
        
        return {
            "stage": self.stage.value,
            "quality_score": self.data_quality_score,
            "is_healthy": self.is_healthy(),
            "total_validations": len(self.validation_results),
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "processing_time_ms": self.processing_time * 1000,
            "recommendations_count": len(self.recommendations)
        }

# Factory functions for easy creation

def create_schema_context_from_raw(
    raw_schema_result: Dict[str, Any],
    nlp_insights: Optional[Dict[str, Any]] = None
) -> StandardizedSchemaContext:
    """
    Factory function to create StandardizedSchemaContext from raw schema searcher output
    CRITICAL: This enables schema_searcher → standardized format conversion
    """
    try:
        return StandardizedSchemaContext(
            tables=raw_schema_result.get('tables', []),
            columns_by_table=raw_schema_result.get('columns_by_table', {}),
            joins=raw_schema_result.get('joins', []),
            confidence_score=raw_schema_result.get('confidence_score', 0.8),
            total_columns=raw_schema_result.get('total_columns', 0),
            has_xml_data=raw_schema_result.get('has_xml_data', False),
            nlp_insights=nlp_insights or {},
            predicted_tables=raw_schema_result.get('predicted_tables', []),
            predicted_columns=raw_schema_result.get('predicted_columns', []),
            processing_time=raw_schema_result.get('processing_time', 0.0),
            search_method=raw_schema_result.get('method', 'unknown'),
            quality_metrics=raw_schema_result.get('quality_metrics', {})
        )
    except Exception as e:
        logger.error(f"Failed to create schema context from raw data: {e}")
        # Return minimal valid context
        return StandardizedSchemaContext(
            tables=[],
            columns_by_table={},
            joins=[],
            confidence_score=0.5
        )

def create_prompt_result_from_raw(
    raw_prompt_result: Union[str, Dict[str, Any]],
    schema_context: Optional[StandardizedSchemaContext] = None,
    nlp_context: Optional[Dict[str, Any]] = None
) -> StandardizedPromptResult:
    """
    Factory function to create StandardizedPromptResult from raw prompt builder output
    CRITICAL: This enables prompt_builder → standardized format conversion
    """
    try:
        if isinstance(raw_prompt_result, str):
            # Simple string prompt
            return StandardizedPromptResult(
                generated_prompt=raw_prompt_result,
                schema_context_used=schema_context,
                nlp_context_applied=nlp_context or {},
                prompt_confidence=0.8,
                generation_method="string_based"
            )
        elif isinstance(raw_prompt_result, dict):
            # Complex prompt result
            return StandardizedPromptResult(
                generated_prompt=raw_prompt_result.get('generated_prompt', raw_prompt_result.get('prompt', '')),
                prompt_metadata=raw_prompt_result.get('metadata', {}),
                schema_context_used=schema_context,
                nlp_context_applied=nlp_context or {},
                prompt_confidence=raw_prompt_result.get('confidence', 0.8),
                generation_method=raw_prompt_result.get('method', 'dict_based'),
                processing_time=raw_prompt_result.get('processing_time', 0.0),
                prompt_optimizations=raw_prompt_result.get('optimizations', []),
                context_truncated=raw_prompt_result.get('context_truncated', False),
                fallback_used=raw_prompt_result.get('fallback_used', False)
            )
        else:
            raise ValueError(f"Unsupported prompt result type: {type(raw_prompt_result)}")
    except Exception as e:
        logger.error(f"Failed to create prompt result from raw data: {e}")
        # Return minimal valid result
        return StandardizedPromptResult(
            generated_prompt=str(raw_prompt_result) if raw_prompt_result else "",
            prompt_confidence=0.3,
            generation_method="error_fallback",
            fallback_used=True
        )

# Utility functions for pipeline validation

def validate_pipeline_data_flow(
    schema_context: StandardizedSchemaContext,
    prompt_result: StandardizedPromptResult,
    sql_request: StandardizedSQLRequest
) -> PipelineValidationReport:
    """
    Validate complete pipeline data flow
    CRITICAL: This ensures end-to-end pipeline compatibility
    """
    report = PipelineValidationReport(stage=PipelineStage.SQL_GENERATION)
    
    # Validate schema context
    schema_validation = schema_context._validate_internal_consistency()
    report.add_validation(schema_validation)
    
    # Validate prompt result
    prompt_validation = prompt_result._validate_prompt_quality()
    report.add_validation(prompt_validation)
    
    # Validate SQL request
    sql_validation = sql_request.validate()
    report.add_validation(sql_validation)
    
    # Add pipeline-specific recommendations
    if report.data_quality_score < 0.8:
        report.recommendations.extend([
            "Consider improving data validation at earlier pipeline stages",
            "Review NLP insights quality and completeness",
            "Validate schema search results before prompt generation"
        ])
    
    return report

# Export main classes for external use
__all__ = [
    'StandardizedSchemaContext',
    'StandardizedPromptResult', 
    'StandardizedSQLRequest',
    'PipelineValidationReport',
    'ValidationResult',
    'DataValidationLevel',
    'PipelineStage',
    'create_schema_context_from_raw',
    'create_prompt_result_from_raw',
    'validate_pipeline_data_flow'
]

if __name__ == "__main__":
    # Example usage and validation - FIXED: Removed emojis
    print("Pipeline Data Models for NLP-Schema Integration")
    print("=" * 60)
    
    # Test schema context creation
    raw_schema = {
        'tables': ['customers', 'accounts', 'transactions'],
        'columns_by_table': {
            'customers': [{'column': 'id', 'type': 'int'}, {'column': 'name', 'type': 'varchar'}],
            'accounts': [{'column': 'account_id', 'type': 'int'}, {'column': 'balance', 'type': 'decimal'}]
        },
        'joins': [{'left_table': 'customers', 'right_table': 'accounts', 'condition': 'customers.id = accounts.customer_id'}],
        'confidence_score': 0.9,
        'total_columns': 4,
        'processing_time': 0.5
    }
    
    nlp_insights = {
        'detected_intent': 'customer_lookup',
        'entities': ['customer', 'account', 'balance'],
        'confidence': 0.85
    }
    
    # Create standardized schema context
    schema_context = create_schema_context_from_raw(raw_schema, nlp_insights)
    
    print(f"Schema Context Created:")
    print(f"   - Tables: {len(schema_context.tables)}")
    print(f"   - Total Columns: {schema_context.total_columns}")
    print(f"   - Confidence: {schema_context.confidence_score}")
    
    # Test format conversions
    prompt_format = schema_context.to_prompt_builder_format()
    sql_format = schema_context.to_sql_generator_format()
    
    print(f"Format Conversions Successful:")
    print(f"   - Prompt Builder Format: {len(prompt_format)} keys")
    print(f"   - SQL Generator Format: {len(sql_format)} keys")
    
    # Test prompt result
    prompt_result = StandardizedPromptResult(
        generated_prompt="Generate SQL for customer account lookup",
        schema_context_used=schema_context,
        prompt_confidence=0.9
    )
    
    sql_input = prompt_result.to_sql_generator_input()
    print(f"Prompt Result Conversion: {len(sql_input)} keys")
    
    print("\nPipeline Data Models Ready for Integration!")
    print("   This resolves the critical data format compatibility issues")
    print("   between schema_searcher → prompt_builder → sql_generator")
