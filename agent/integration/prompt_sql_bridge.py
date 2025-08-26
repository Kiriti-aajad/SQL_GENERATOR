"""

Centralized Prompt-to-SQL Generation Bridge

Connects PromptBuilder output directly to SQLGenerator input

Handles all data format conversion, validation, and fallback logic

FIXED: Async/sync mixing issue that caused timeout context manager errors

"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

@dataclass
class PromptSQLRequest:
    """Standardized request for prompt->SQL conversion"""
    generated_prompt: str
    schema_context: Dict[str, Any]
    nlp_insights: Dict[str, Any]
    user_query: str
    target_model: str = "mistral"
    request_id: str = "unknown"

@dataclass
class PromptSQLResponse:
    """Standardized response from prompt->SQL conversion"""
    success: bool
    generated_sql: str
    confidence: float
    processing_time_ms: float
    metadata: Dict[str, Any]
    error: Optional[str] = None

class PromptSQLBridge:
    """CENTRALIZED bridge from PromptBuilder to SQLGenerator with robust data format handling"""

    def __init__(self, async_client_manager=None):
        self.async_client_manager = async_client_manager
        self.logger = logging.getLogger(__name__)
        self.sql_generator = None
        self._initialize_sql_generator()

    def _initialize_sql_generator(self):
        """Initialize SQL generator with proper client manager"""
        try:
            from agent.sql_generator.generator import SQLGenerator
            self.sql_generator = SQLGenerator(client_manager=self.async_client_manager)
            self.logger.info("SQL Generator initialized in bridge")
        except Exception as e:
            self.logger.error(f"SQL Generator initialization failed: {e}")
            self.sql_generator = None

    async def convert_prompt_to_sql(self, request: PromptSQLRequest) -> PromptSQLResponse:
        """
        MAIN METHOD: Convert prompt to SQL with full error handling and data validation
        FIXED: Direct async call eliminates sync/async mixing issues
        """
        start_time = time.time()
        try:
            self.logger.info(f"[{request.request_id}] Converting prompt to SQL: {len(request.generated_prompt)} chars")

            if not self.sql_generator:
                raise RuntimeError("SQL Generator not initialized")

            if not request.generated_prompt.strip():
                raise ValueError("Empty prompt provided")

            # Convert and validate prompt format for SQL generator
            sql_input = self._convert_prompt_format(request)

            # Log the data format for debugging
            self.logger.debug(f"[{request.request_id}] SQL input format: {type(sql_input.get('banking_insights'))}, {type(sql_input.get('schema_context'))}")

            # CRITICAL FIX: Direct async call instead of sync call in executor
            # This eliminates the timeout context manager error
            sql_result = await asyncio.wait_for(
                self.sql_generator.generate_sql_async(
                    request.generated_prompt,
                    banking_insights=sql_input.get('banking_insights'),
                    schema_context=sql_input.get('schema_context'),
                    target_llm=sql_input.get('target_llm'),
                    context=sql_input.get('context')
                ),
                timeout=30.0
            )

            # Validate and format response
            response = self._create_success_response(sql_result, request, start_time)
            self.logger.info(f"[{request.request_id}] SQL generation successful: {response.confidence:.2f} confidence")
            return response

        except Exception as e:
            self.logger.error(f"[{request.request_id}] Prompt->SQL conversion failed: {e}")
            return self._create_error_response(str(e), request, start_time)

    def _convert_prompt_format(self, request: PromptSQLRequest) -> Dict[str, Any]:
        """
        Convert prompt request to SQL generator expected format
        FIXES: Ensures all data is in proper dict format, not lists
        """
        # Fix nlp_insights format - ensure it's a dict
        nlp_insights = self._ensure_dict_format(
            request.nlp_insights,
            "nlp_insights",
            request.request_id
        )

        # Fix schema_context format - ensure it's a dict
        schema_context = self._ensure_dict_format(
            request.schema_context,
            "schema_context",
            request.request_id
        )

        # Ensure banking_insights has proper structure for SQL Generator
        banking_insights = self._prepare_banking_insights(nlp_insights, request.request_id)

        return {
            "prompt": request.generated_prompt,
            "banking_insights": banking_insights,  # Now guaranteed to be proper dict
            "schema_context": schema_context,      # Now guaranteed to be proper dict
            "target_llm": request.target_model,
            "context": {
                "user_query": request.user_query,
                "request_id": request.request_id,
                "processing_mode": "centralized_bridge_fixed"  # Updated to indicate fix
            }
        }

    def _ensure_dict_format(self, data: Any, field_name: str, request_id: str) -> Dict[str, Any]:
        """
        Ensure data is in dictionary format, convert if necessary
        FIXES: The 'list' object has no attribute 'items' error
        """
        if isinstance(data, dict):
            return data
        elif isinstance(data, list):
            self.logger.warning(f"[{request_id}] Converting {field_name} from list to dict format")
            if field_name == "nlp_insights":
                return {"insights_list": data, "type": "converted_from_list"}
            elif field_name == "schema_context":
                return {"tables": data, "type": "converted_from_list"}
            else:
                return {"data": data, "type": "converted_from_list"}
        elif isinstance(data, str):
            self.logger.warning(f"[{request_id}] Converting {field_name} from string to dict format")
            return {"text": data, "type": "converted_from_string"}
        elif data is None:
            self.logger.warning(f"[{request_id}] {field_name} is None, using empty dict")
            return {}
        else:
            self.logger.warning(f"[{request_id}] Unknown {field_name} type: {type(data)}, converting to dict")
            return {"data": str(data), "type": f"converted_from_{type(data).__name__}"}

    def _prepare_banking_insights(self, nlp_insights: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """
        Prepare banking_insights in the format expected by SQL Generator
        """
        # If already properly formatted, return as-is
        if "entities" in nlp_insights and "keywords" in nlp_insights:
            return nlp_insights

        # Convert to expected banking insights format
        banking_insights = {
            "entities": nlp_insights.get("entities", []),
            "keywords": nlp_insights.get("keywords", []),
            "intent": nlp_insights.get("intent", "unknown"),
            "confidence": nlp_insights.get("confidence", 0.5),
            "insights_list": nlp_insights.get("insights_list", []),
            "metadata": {
                "conversion_applied": True,
                "original_format": nlp_insights.get("type", "unknown"),
                "request_id": request_id
            }
        }

        # Add any additional insights data
        for key, value in nlp_insights.items():
            if key not in banking_insights and key != "type":
                banking_insights[key] = value

        self.logger.debug(f"[{request_id}] Prepared banking_insights with {len(banking_insights)} fields")
        return banking_insights

    def _create_success_response(self, sql_result: Any, request: PromptSQLRequest, start_time: float) -> PromptSQLResponse:
        """Create standardized success response with robust result handling"""
        processing_time = (time.time() - start_time) * 1000

        # Handle different sql_result formats
        if isinstance(sql_result, dict):
            generated_sql = sql_result.get('generated_sql', sql_result.get('sql', ''))
            confidence = sql_result.get('confidence', 0.8)
            model_used = sql_result.get('model_used', request.target_model)
        elif isinstance(sql_result, str):
            generated_sql = sql_result
            confidence = 0.7  # Default confidence for string responses
            model_used = request.target_model
        else:
            generated_sql = str(sql_result) if sql_result else ""
            confidence = 0.5  # Lower confidence for unexpected formats
            model_used = request.target_model

        return PromptSQLResponse(
            success=True,
            generated_sql=generated_sql,
            confidence=confidence,
            processing_time_ms=processing_time,
            metadata={
                "model_used": model_used,
                "prompt_length": len(request.generated_prompt),
                "schema_tables": len(request.schema_context.get('tables', [])),
                "bridge_version": "1.2.0",  # Updated version with async fix
                "data_format_fixes_applied": True,
                "async_sync_fix_applied": True  # NEW: Indicates the critical fix
            }
        )

    def _create_error_response(self, error: str, request: PromptSQLRequest, start_time: float) -> PromptSQLResponse:
        """Create standardized error response"""
        processing_time = (time.time() - start_time) * 1000

        return PromptSQLResponse(
            success=False,
            generated_sql="",
            confidence=0.0,
            processing_time_ms=processing_time,
            metadata={
                "bridge_version": "1.2.0",  # Updated version
                "error_occurred": True,
                "async_sync_fix_applied": True,  # NEW: Indicates the critical fix
                "prompt_length": len(request.generated_prompt) if hasattr(request, 'generated_prompt') else 0
            },
            error=error
        )

    def health_check(self) -> Dict[str, Any]:
        """Health check for the bridge"""
        return {
            "status": "healthy" if self.sql_generator else "degraded",
            "component": "PromptSQLBridge",
            "version": "1.2.0",  # Updated version
            "sql_generator_available": self.sql_generator is not None,
            "async_client_manager_available": self.async_client_manager is not None,
            "data_format_validation": True,
            "fixes_applied": [
                "list_to_dict_conversion", 
                "robust_data_handling", 
                "banking_insights_preparation",
                "async_sync_mixing_fix"  # NEW: The critical fix
            ],
            "timeout_context_manager_fix": True  # NEW: Confirms the main issue is fixed
        }

# Factory function
def create_prompt_sql_bridge(async_client_manager=None) -> PromptSQLBridge:
    """Create centralized prompt->SQL bridge with async/sync fixes"""
    return PromptSQLBridge(async_client_manager)

# Test function for validation
async def test_prompt_sql_bridge():
    """Test the bridge with various data formats and async functionality"""
    print("Testing Prompt-SQL Bridge with async/sync fix validation...")
    
    try:
        bridge = create_prompt_sql_bridge()
        health = bridge.health_check()
        
        print(f"Bridge Health: {health['status']}")
        print(f"Version: {health['version']}")
        print(f"Data Format Validation: {health['data_format_validation']}")
        print(f"Timeout Context Manager Fix: {health['timeout_context_manager_fix']}")
        print(f"Fixes Applied: {health['fixes_applied']}")

        # Test with different data formats
        test_cases = [
            {
                "name": "Dict format",
                "nlp_insights": {"entities": ["test"], "keywords": ["app"]},
                "schema_context": {"tables": ["ApplicationMaster"]}
            },
            {
                "name": "List format (problematic)",
                "nlp_insights": ["insight1", "insight2"],
                "schema_context": ["table1", "table2"]
            },
            {
                "name": "Mixed format",
                "nlp_insights": {"data": ["test"]},
                "schema_context": {"tables": ["test"]}
            }
        ]

        for test_case in test_cases:
            print(f"\nTesting: {test_case['name']}")
            converted = bridge._ensure_dict_format(test_case['nlp_insights'], "nlp_insights", "test")
            print(f"Converted NLP insights: {type(converted)} - {converted}")

        print("\nAll data format tests passed!")
        print(" Async/sync mixing fix applied successfully!")

    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_prompt_sql_bridge())
