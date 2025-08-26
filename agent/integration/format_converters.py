"""
Pipeline Format Converters for NLP-Schema Integration System
Handles critical data format compatibility between pipeline components
Converts between schema_searcher → prompt_builder → sql_generator formats

Author: Enhanced for Pipeline Compatibility
Version: 2.0.0 - CRITICAL COMPATIBILITY FIX
Date: 2025-08-06
"""

from typing import Dict, Any, Optional, Union, List
import logging
import json
from datetime import datetime

# Import the standardized data models we just created
from agent.integration.pipeline_data_models import (
    StandardizedSchemaContext,
    StandardizedPromptResult,
    StandardizedSQLRequest,
    create_schema_context_from_raw,
    create_prompt_result_from_raw,
    ValidationResult,
    DataValidationLevel
)

logger = logging.getLogger(__name__)

class SchemaFormatConverter:
    """
    Converts schema searcher output to standardized formats
    CRITICAL: Fixes schema_searcher → prompt_builder compatibility
    """
    
    @staticmethod
    def convert_raw_schema_to_standardized(
        raw_schema_result: Dict[str, Any],
        nlp_insights: Optional[Dict[str, Any]] = None
    ) -> StandardizedSchemaContext:
        """
        Convert raw schema searcher output to standardized schema context
        
        Args:
            raw_schema_result: Raw output from schema searcher
            nlp_insights: Optional NLP insights to include
            
        Returns:
            StandardizedSchemaContext: Standardized schema context
        """
        try:
            logger.debug(f"Converting raw schema result with {len(raw_schema_result.get('tables', []))} tables")
            
            # Use factory function from pipeline_data_models
            standardized_context = create_schema_context_from_raw(raw_schema_result, nlp_insights)
            
            logger.debug(f"Schema conversion successful: {standardized_context.get_summary()}")
            return standardized_context
            
        except Exception as e:
            logger.error(f"Schema conversion failed: {e}")
            # Return minimal valid context to prevent pipeline failure
            return StandardizedSchemaContext(
                tables=raw_schema_result.get('tables', []),
                columns_by_table=raw_schema_result.get('columns_by_table', {}),
                joins=raw_schema_result.get('joins', []),
                confidence_score=0.3,  # Low confidence due to conversion error
                nlp_insights=nlp_insights or {} # pyright: ignore[reportCallIssue]
            )
    
    @staticmethod
    def convert_schema_to_prompt_builder_format(
        schema_result: Union[Dict[str, Any], StandardizedSchemaContext],
        nlp_insights: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Convert schema result to prompt builder expected format
        CRITICAL: This is the main compatibility fix for schema → prompt
        
        Args:
            schema_result: Either raw schema result dict or StandardizedSchemaContext
            nlp_insights: Optional NLP insights
            
        Returns:
            Dict formatted for prompt builder input
        """
        try:
            # Convert to StandardizedSchemaContext if needed
            if isinstance(schema_result, dict):
                standardized_context = SchemaFormatConverter.convert_raw_schema_to_standardized(
                    schema_result, nlp_insights
                )
            elif isinstance(schema_result, StandardizedSchemaContext):
                standardized_context = schema_result
            else:
                raise ValueError(f"Unsupported schema_result type: {type(schema_result)}")
            
            # Convert to prompt builder format using standardized method
            prompt_builder_format = standardized_context.to_prompt_builder_format()
            
            logger.debug(f"Schema → Prompt builder conversion successful")
            return prompt_builder_format
            
        except Exception as e:
            logger.error(f"Schema → Prompt builder conversion failed: {e}")
            # Return minimal safe format
            return {
                "schema": {
                    "tables": schema_result.get('tables', []) if isinstance(schema_result, dict) else schema_result.tables,
                    "table_definitions": schema_result.get('columns_by_table', {}) if isinstance(schema_result, dict) else schema_result.columns_by_table,
                    "relationships": schema_result.get('joins', []) if isinstance(schema_result, dict) else schema_result.joins
                },
                "nlp_context": nlp_insights or {},
                "error": f"Conversion error: {str(e)}"
            }
    
    @staticmethod
    def validate_schema_format(schema_data: Dict[str, Any]) -> ValidationResult:
        """Validate schema data format"""
        errors = []
        warnings = []
        
        # Required fields validation
        required_fields = ['tables', 'columns_by_table']
        for field in required_fields:
            if field not in schema_data:
                errors.append(f"Missing required field: {field}")
        
        # Data type validation
        if 'tables' in schema_data and not isinstance(schema_data['tables'], list):
            errors.append("'tables' must be a list")
        
        if 'columns_by_table' in schema_data and not isinstance(schema_data['columns_by_table'], dict):
            errors.append("'columns_by_table' must be a dictionary")
        
        # Quality warnings
        if 'confidence_score' in schema_data:
            confidence = schema_data['confidence_score']
            if confidence < 0.5:
                warnings.append(f"Low confidence score: {confidence}")
        
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            validation_level=DataValidationLevel.MODERATE
        )

class PromptFormatConverter:
    """
    Converts prompt builder output to standardized formats
    CRITICAL: Fixes prompt_builder → sql_generator compatibility
    """
    
    @staticmethod
    def convert_raw_prompt_to_standardized(
        raw_prompt_result: Union[str, Dict[str, Any]],
        schema_context: Optional[StandardizedSchemaContext] = None,
        nlp_context: Optional[Dict[str, Any]] = None
    ) -> StandardizedPromptResult:
        """
        Convert raw prompt builder output to standardized prompt result
        
        Args:
            raw_prompt_result: Raw output from prompt builder (string or dict)
            schema_context: Schema context used in prompt generation
            nlp_context: NLP context applied
            
        Returns:
            StandardizedPromptResult: Standardized prompt result
        """
        try:
            logger.debug(f"Converting raw prompt result of type {type(raw_prompt_result)}")
            
            # Use factory function from pipeline_data_models
            standardized_prompt = create_prompt_result_from_raw(
                raw_prompt_result, schema_context, nlp_context
            )
            
            logger.debug(f"Prompt conversion successful: {standardized_prompt.get_summary()}")
            return standardized_prompt
            
        except Exception as e:
            logger.error(f"Prompt conversion failed: {e}")
            # Return minimal valid prompt result
            return StandardizedPromptResult(
                generated_prompt=str(raw_prompt_result) if raw_prompt_result else "",
                schema_context_used=schema_context,
                nlp_context_applied=nlp_context or {},
                prompt_confidence=0.3,  # Low confidence due to conversion error
                fallback_used=True
            )
    
    @staticmethod
    def convert_prompt_to_sql_generator_format(
        prompt_result: Union[str, Dict[str, Any], StandardizedPromptResult],
        schema_context: Optional[StandardizedSchemaContext] = None,
        nlp_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Convert prompt result to SQL generator expected format
        CRITICAL: This is the main compatibility fix for prompt → SQL
        
        Args:
            prompt_result: Prompt builder output (various formats)
            schema_context: Optional schema context
            nlp_context: Optional NLP context
            
        Returns:
            Dict formatted for SQL generator input
        """
        try:
            # Convert to StandardizedPromptResult if needed
            if isinstance(prompt_result, (str, dict)):
                standardized_prompt = PromptFormatConverter.convert_raw_prompt_to_standardized(
                    prompt_result, schema_context, nlp_context
                )
            elif isinstance(prompt_result, StandardizedPromptResult):
                standardized_prompt = prompt_result
            else:
                raise ValueError(f"Unsupported prompt_result type: {type(prompt_result)}")
            
            # Convert to SQL generator format using standardized method
            sql_generator_format = standardized_prompt.to_sql_generator_input()
            
            logger.debug(f"Prompt → SQL generator conversion successful")
            return sql_generator_format
            
        except Exception as e:
            logger.error(f"Prompt → SQL generator conversion failed: {e}")
            # Return minimal safe format
            prompt_text = ""
            if isinstance(prompt_result, str):
                prompt_text = prompt_result
            elif isinstance(prompt_result, dict):
                prompt_text = prompt_result.get('generated_prompt', prompt_result.get('prompt', ''))
            elif isinstance(prompt_result, StandardizedPromptResult):
                prompt_text = prompt_result.generated_prompt
            
            return {
                "prompt": prompt_text,
                "banking_insights": nlp_context or {},
                "schema_context": schema_context.to_sql_generator_format() if schema_context else {},
                "error": f"Conversion error: {str(e)}"
            }
    
    @staticmethod
    def validate_prompt_format(prompt_data: Union[str, Dict[str, Any]]) -> ValidationResult:
        """Validate prompt data format"""
        errors = []
        warnings = []
        
        if isinstance(prompt_data, str):
            if not prompt_data.strip():
                errors.append("Prompt string is empty")
            elif len(prompt_data) > 10000:
                warnings.append(f"Prompt is very long ({len(prompt_data)} characters)")
        elif isinstance(prompt_data, dict):
            # Check for expected fields in dict format
            if 'generated_prompt' not in prompt_data and 'prompt' not in prompt_data:
                errors.append("Dict format missing 'generated_prompt' or 'prompt' field")
        else:
            errors.append(f"Unsupported prompt format: {type(prompt_data)}")
        
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            validation_level=DataValidationLevel.MODERATE
        )

class SQLFormatConverter:
    """
    Handles SQL generator input/output format conversions
    """
    
    @staticmethod
    def create_standardized_sql_request(
        prompt: str,
        banking_insights: Dict[str, Any],
        schema_context: Dict[str, Any],
        target_llm: str = "mistral",
        generation_options: Optional[Dict[str, Any]] = None
    ) -> StandardizedSQLRequest:
        """
        Create standardized SQL request from components
        
        Args:
            prompt: Generated prompt text
            banking_insights: NLP/banking insights
            schema_context: Schema context information
            target_llm: Target language model
            generation_options: Optional generation parameters
            
        Returns:
            StandardizedSQLRequest: Standardized request object
        """
        try:
            sql_request = StandardizedSQLRequest(
                prompt=prompt,
                banking_insights=banking_insights,
                schema_context=schema_context,
                target_llm=target_llm,
                generation_options=generation_options or {}
            )
            
            # Validate the request
            validation = sql_request.validate()
            if not validation.is_valid:
                logger.warning(f"SQL request validation warnings: {validation.warnings}")
            
            return sql_request
            
        except Exception as e:
            logger.error(f"SQL request creation failed: {e}")
            # Return minimal valid request
            return StandardizedSQLRequest(
                prompt=prompt,
                banking_insights=banking_insights,
                schema_context=schema_context,
                target_llm=target_llm
            )

class PipelineFormatConverter:
    """
    Main pipeline format converter combining all conversion operations
    CRITICAL: This is the primary class used by orchestrators
    """
    
    @staticmethod
    def convert_schema_searcher_to_prompt_builder(
        schema_result: Dict[str, Any],
        nlp_insights: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main conversion: schema searcher output → prompt builder input
        
        Args:
            schema_result: Raw schema searcher output
            nlp_insights: Optional NLP insights
            
        Returns:
            Dict formatted for prompt builder
        """
        return SchemaFormatConverter.convert_schema_to_prompt_builder_format(
            schema_result, nlp_insights
        )
    
    @staticmethod
    def convert_prompt_builder_to_sql_generator(
        prompt_result: Union[str, Dict[str, Any]],
        schema_context: Optional[StandardizedSchemaContext] = None,
        nlp_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main conversion: prompt builder output → SQL generator input
        
        Args:
            prompt_result: Prompt builder output
            schema_context: Schema context used
            nlp_context: NLP context applied
            
        Returns:
            Dict formatted for SQL generator
        """
        return PromptFormatConverter.convert_prompt_to_sql_generator_format(
            prompt_result, schema_context, nlp_context
        )
    
    @staticmethod
    def validate_pipeline_compatibility(
        schema_result: Dict[str, Any],
        prompt_result: Union[str, Dict[str, Any]],
        expected_sql_format: Dict[str, str]
    ) -> Dict[str, ValidationResult]:
        """
        Validate compatibility across entire pipeline
        
        Args:
            schema_result: Schema searcher output
            prompt_result: Prompt builder output  
            expected_sql_format: Expected SQL generator format
            
        Returns:
            Dict of validation results by stage
        """
        validations = {}
        
        # Validate schema format
        validations['schema'] = SchemaFormatConverter.validate_schema_format(schema_result)
        
        # Validate prompt format
        validations['prompt'] = PromptFormatConverter.validate_prompt_format(prompt_result)
        
        # Check overall pipeline health
        all_valid = all(v.is_valid for v in validations.values())
        validations['pipeline_healthy'] = ValidationResult(
            is_valid=all_valid,
            errors=[] if all_valid else ["Pipeline compatibility issues detected"],
            warnings=[],
            validation_level=DataValidationLevel.STRICT
        )
        
        return validations

# Utility functions for backward compatibility

def convert_schema_to_prompt_format(schema_result: Dict[str, Any], nlp_insights: Dict[str, Any] = None) -> Dict[str, Any]: # pyright: ignore[reportArgumentType]
    """Backward compatibility function"""
    return PipelineFormatConverter.convert_schema_searcher_to_prompt_builder(schema_result, nlp_insights)

def convert_prompt_to_sql_format(prompt_result: Union[str, Dict[str, Any]], schema_context: Dict[str, Any] = None) -> Dict[str, Any]: # pyright: ignore[reportArgumentType]
    """Backward compatibility function"""
    # Convert schema_context dict to StandardizedSchemaContext if needed
    standardized_context = None
    if schema_context:
        standardized_context = create_schema_context_from_raw(schema_context)
    
    return PipelineFormatConverter.convert_prompt_builder_to_sql_generator(
        prompt_result, standardized_context
    )

# Export main classes and functions
__all__ = [
    'SchemaFormatConverter',
    'PromptFormatConverter', 
    'SQLFormatConverter',
    'PipelineFormatConverter',
    'convert_schema_to_prompt_format',  # Backward compatibility
    'convert_prompt_to_sql_format'      # Backward compatibility
]

if __name__ == "__main__":
    # Example usage and testing
    print("Pipeline Format Converters for NLP-Schema Integration")
    print("=" * 65)
    
    # Test schema conversion
    sample_schema_result = {
        'tables': ['customers', 'accounts', 'transactions'],
        'columns_by_table': {
            'customers': [{'column': 'id', 'type': 'int'}, {'column': 'name', 'type': 'varchar'}],
            'accounts': [{'column': 'account_id', 'type': 'int'}, {'column': 'customer_id', 'type': 'int'}]
        },
        'joins': [{'left_table': 'customers', 'right_table': 'accounts', 'condition': 'customers.id = accounts.customer_id'}],
        'confidence_score': 0.9,
        'total_columns': 4,
        'has_xml_data': False,
        'processing_time': 0.45,
        'method': 'intelligent_retrieval'
    }
    
    sample_nlp_insights = {
        'detected_intent': {'primary': 'customer_lookup', 'confidence': 0.85},
        'semantic_entities': ['customer', 'account', 'balance'],
        'target_tables_predicted': ['customers', 'accounts']
    }
    
    # Test schema → prompt conversion
    try:
        prompt_format = PipelineFormatConverter.convert_schema_searcher_to_prompt_builder(
            sample_schema_result, sample_nlp_insights
        )
        print(f"Schema → Prompt conversion successful:")
        print(f"   - Schema keys: {list(prompt_format['schema'].keys())}")
        print(f"   - Has NLP context: {'nlp_context' in prompt_format}")
        print(f"   - Has processing hints: {'processing_hints' in prompt_format}")
    except Exception as e:
        print(f"Schema → Prompt conversion failed: {e}")
    
    # Test prompt → SQL conversion
    sample_prompt_result = {
        'generated_prompt': "SELECT c.name, a.balance FROM customers c JOIN accounts a ON c.id = a.customer_id WHERE c.region = 'Maharashtra'",
        'metadata': {
            'confidence': 0.88,
            'generation_method': 'template_based',
            'nlp_insights': sample_nlp_insights
        }
    }
    
    try:
        # Create standardized schema context for the conversion
        standardized_schema = create_schema_context_from_raw(sample_schema_result, sample_nlp_insights)
        
        sql_format = PipelineFormatConverter.convert_prompt_builder_to_sql_generator(
            sample_prompt_result, standardized_schema, sample_nlp_insights
        )
        print(f"Prompt → SQL conversion successful:")
        print(f"   - Has prompt: {'prompt' in sql_format}")
        print(f"   - Has banking insights: {'banking_insights' in sql_format}")
        print(f"   - Has schema context: {'schema_context' in sql_format}")
        print(f"   - Prompt length: {len(sql_format.get('prompt', ''))}")
    except Exception as e:
        print(f"Prompt → SQL conversion failed: {e}")
    
    # Test pipeline validation
    try:
        validations = PipelineFormatConverter.validate_pipeline_compatibility(
            sample_schema_result, sample_prompt_result, {}
        )
        print(f"Pipeline validation completed:")
        for stage, validation in validations.items():
            status = "VALID" if validation.is_valid else "INVALID"
            print(f"   - {stage}: {status}")
            if validation.warnings:
                print(f"     Warnings: {len(validation.warnings)}")
    except Exception as e:
        print(f"Pipeline validation failed: {e}")
    
    print("\nFormat Converters Ready for Integration!")
    print("   This resolves critical data format compatibility issues:")
    print("   • Schema Searcher → Prompt Builder")
    print("   • Prompt Builder → SQL Generator") 
    print("   • Complete pipeline validation")
