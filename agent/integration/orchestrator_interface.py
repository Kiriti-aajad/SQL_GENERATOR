"""
Orchestrator Interface for NLP-Schema Integration
Provides clean, standardized interface for orchestrator consumption
Makes the integration bridge easily accessible and manageable
FIXED: All character escaping, error handling, safety issues, Pylance warnings, and AsyncClientManager integration
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Import our integration components
from .nlp_schema_integration_bridge import NLPSchemaIntegrationBridge
from .data_models import (
    SchemaSearchRequest, SchemaSearchResponse, NLPEnhancedQuery,
    QueryIntent, ComponentStatus, ComponentHealth, ProcessingStatistics
)

logger = logging.getLogger(__name__)

class OrchestratorCompatibleSystem:
    """
    Orchestrator-compatible system interface for NLP-Schema integration
    Provides standardized methods for top-level orchestration
    FIXED: All character escaping, error handling, safety issues, Pylance warnings, and AsyncClientManager integration
    """

    def __init__(self, integration_bridge: NLPSchemaIntegrationBridge):
        """Initialize with integration bridge"""
        self.bridge = integration_bridge
        self.system_name = "NLP-Schema-Integration-System"
        self.version = "1.0.0"
        self.logger = logger

        # System-level statistics
        self.system_start_time = time.time()
        self.total_orchestrator_requests = 0
        self.successful_orchestrator_requests = 0

        # CRITICAL FIX: Add status attribute for server validation
        self.status = "healthy"

        # FIXED: Add safe access utilities
        self._initialize_safe_access_utilities()

        self.logger.info(f"{self.system_name} v{self.version} initialized for orchestrator use")

    def _initialize_safe_access_utilities(self):
        """FIXED: Initialize safe access utilities for defensive programming"""
        self.safe_access_utils = {
            'initialized': True,
            'version': '1.0.0'
        }

    def _safe_get(self, obj: Any, key: Any, default: Any = None) -> Any:
        """ENHANCED: Safely get attribute from object with comprehensive error handling"""
        if obj is None:
            return default

        try:
            # Handle dictionary-like objects
            if hasattr(obj, 'get') and callable(getattr(obj, 'get')):
                return obj.get(key, default)

            # Handle attribute access
            if hasattr(obj, key):
                attr_value = getattr(obj, key, default)
                return attr_value if attr_value is not None else default

            # Handle index access for lists/tuples
            if isinstance(obj, (list, tuple)) and isinstance(key, int):
                return obj[key] if 0 <= key < len(obj) else default

        except (AttributeError, KeyError, IndexError, TypeError):
            pass

        return default

    def _safe_access(self, obj: Any, attr_path: str, default: Any = None) -> Any:
        """ENHANCED: Safely access nested attributes with better error handling"""
        if obj is None:
            return default

        try:
            if '.' in attr_path:
                attrs = attr_path.split('.')
                current = obj
                for attr in attrs:
                    if current is None:
                        return default
                    if hasattr(current, 'get') and callable(getattr(current, 'get')):
                        current = current.get(attr)
                    elif hasattr(current, attr):
                        current = getattr(current, attr)
                    else:
                        return default
                return current if current is not None else default
            else:
                return self._safe_get(obj, attr_path, default)

        except (AttributeError, KeyError, TypeError):
            return default

    def _validate_response_object(self, response: Any, context: str = "") -> bool:
        """FIXED: Validate and log response object types for debugging"""
        if response is None:
            self.logger.warning(f"{context}: Response is None")
            return False

        self.logger.debug(f"{context}: Response type: {type(response)}")
        if hasattr(response, '__dict__'):
            self.logger.debug(f"{context}: Response attributes: {list(response.__dict__.keys())}")

        return True

    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        processing_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        MAIN ORCHESTRATOR INTERFACE: Process query through complete pipeline
        FIXED: Comprehensive error handling and None checking
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        self.total_orchestrator_requests += 1

        try:
            self.logger.info(f"[{request_id}] Orchestrator processing query: '{query[:100]}...'")

            # FIXED: Validate input parameters
            if not query or not query.strip():
                raise ValueError("Empty or invalid query provided")

            # Ensure context and processing_options are never None
            context = context or {}
            processing_options = processing_options or {}

            # Process through integration bridge with None checking
            response = await self.bridge.process_query_pipeline(
                query=query,
                context=context,
                processing_options=processing_options
            )

            # FIXED: Validate bridge response
            if not self._validate_response_object(response, "Bridge Response"):
                raise RuntimeError("Integration bridge returned None or invalid response")

            # Convert to orchestrator-standard format
            orchestrator_response = self._create_orchestrator_response(
                response, request_id, time.time() - start_time  # pyright: ignore[reportArgumentType]
            )

            if orchestrator_response is None:
                raise RuntimeError("Failed to create orchestrator response")

            self.successful_orchestrator_requests += 1
            self.logger.info(f"[{request_id}] Orchestrator processing completed successfully")

            return orchestrator_response

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"[{request_id}] Orchestrator processing failed: {e}")

            return self._create_orchestrator_error_response(
                query, str(e), request_id, execution_time
            )

    def _create_orchestrator_response(
        self,
        bridge_response: SchemaSearchResponse,
        request_id: str,
        execution_time: float
    ) -> Dict[str, Any]:
        """FIXED: Create standardized orchestrator response with safe access"""
        try:
            # FIXED: Safe access to bridge response attributes
            query = self._safe_get(bridge_response, 'query', '')
            status = self._safe_get(bridge_response, 'status', 'unknown')
            data = self._safe_get(bridge_response, 'data', {})
            error = self._safe_get(bridge_response, 'error')
            bridge_execution_time = self._safe_get(bridge_response, 'execution_time_ms', 0.0)
            nlp_insights = self._safe_get(bridge_response, 'nlp_insights')
            processing_chain = self._safe_get(bridge_response, 'processing_chain', [])
            metadata = self._safe_get(bridge_response, 'metadata', {})
            response_timestamp = self._safe_get(bridge_response, 'response_timestamp')

            return {
                # Core response data
                'request_id': request_id,
                'query': query,
                'status': status,
                'data': data,
                'error': error,

                # Timing information
                'execution_time_ms': round(execution_time * 1000, 2),
                'bridge_execution_time_ms': bridge_execution_time,

                # NLP insights (key for orchestrator routing decisions)
                'nlp_insights': nlp_insights,

                # Processing information
                'processing_chain': processing_chain,

                # System metadata
                'metadata': {
                    **metadata,
                    'system_name': self.system_name,
                    'system_version': self.version,
                    'orchestrator_interface_used': True,
                    'total_system_requests': self.total_orchestrator_requests
                },

                # Orchestrator-specific fields
                'orchestrator_routing_hints': self._extract_routing_hints(bridge_response),
                'query_complexity': self._assess_query_complexity(bridge_response),
                'recommended_next_steps': self._generate_next_steps(bridge_response),
                'response_timestamp': response_timestamp or datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error creating orchestrator response: {e}")
            # Return fallback response
            return {
                'request_id': request_id,
                'query': '',
                'status': 'error',
                'data': {},
                'error': f'Response creation failed: {str(e)}',
                'execution_time_ms': round(execution_time * 1000, 2),
                'orchestrator_routing_hints': {'recommended_action': 'fallback_processing'},
                'query_complexity': 'unknown',
                'recommended_next_steps': ['use_fallback_orchestrator']
            }

    def _create_orchestrator_error_response(
        self,
        query: str,
        error: str,
        request_id: str,
        execution_time: float
    ) -> Dict[str, Any]:
        """FIXED: Create standardized orchestrator error response"""
        return {
            'request_id': request_id,
            'query': query or '',
            'status': 'error',
            'data': {},
            'error': error,
            'execution_time_ms': execution_time,
            'nlp_insights': None,
            'processing_chain': ['orchestrator_interface_error'],
            'metadata': {
                'system_name': self.system_name,
                'system_version': self.version,
                'orchestrator_interface_used': True,
                'error_level': 'system',
                'total_system_requests': self.total_orchestrator_requests
            },
            'orchestrator_routing_hints': {
                'recommended_action': 'fallback_processing',
                'complexity': 'unknown',
                'confidence': 0.0
            },
            'query_complexity': 'unknown',
            'recommended_next_steps': ['use_fallback_orchestrator', 'log_error', 'notify_admin'],
            'response_timestamp': datetime.now().isoformat()
        }

    def _extract_routing_hints(self, response: SchemaSearchResponse) -> Dict[str, Any]:
        """
        PYLANCE-SAFE: Extract routing hints with robust error handling
        FIXED: All dictionary access and None comparison issues
        """
        hints = {
            'recommended_action': 'proceed_with_sql_generation',
            'complexity': 'medium',
            'confidence': 0.8,
            'processing_method': 'standard'
        }

        try:
            # PYLANCE-SAFE: Safe access to NLP insights
            nlp_insights = self._safe_get(response, 'nlp_insights')
            if nlp_insights:
                # Try multiple access patterns for detected_intent
                detected_intent = (
                    self._safe_get(nlp_insights, 'detected_intent') or
                    self._safe_get(nlp_insights, 'intent') or
                    {}
                )

                if detected_intent and isinstance(detected_intent, dict):
                    primary_intent = self._safe_get(detected_intent, 'primary', 'unknown')
                    confidence = self._safe_get(detected_intent, 'confidence', 0.8)

                    # PYLANCE-SAFE: Intent-based routing with safe mapping
                    intent_scores: Dict[str, int] = {
                        'aggregation': 2,
                        'complex_analysis': 2,
                        'join_analysis': 2,
                        'simple_lookup': 0
                    }

                    # Intent routing mapping
                    intent_routing = {
                        'aggregation': ('use_advanced_sql_generator', 'high'),
                        'complex_analysis': ('use_advanced_sql_generator', 'high'),
                        'simple_lookup': ('use_basic_sql_generator', 'low'),
                        'join_analysis': ('use_join_optimized_generator', 'medium')
                    }

                    if isinstance(primary_intent, str) and primary_intent in intent_routing:
                        action, complexity = intent_routing[primary_intent]
                        hints.update({
                            'recommended_action': action,
                            'complexity': complexity,
                            'confidence': confidence if isinstance(confidence, (int, float)) else 0.8
                        })

            # PYLANCE-SAFE: Data quality assessment with None checks
            response_data = self._safe_get(response, 'data')
            if response_data and isinstance(response_data, dict):
                # FIXED: Get values with defaults and None checks before comparison
                table_count = self._safe_get(response_data, 'table_count', 0)
                join_count = self._safe_get(response_data, 'join_count', 0)
                total_columns = self._safe_get(response_data, 'total_columns', 0)

                # PYLANCE-SAFE: Ensure values are not None before comparison
                table_count = table_count if table_count is not None else 0
                join_count = join_count if join_count is not None else 0
                total_columns = total_columns if total_columns is not None else 0

                # Safe comparisons with guaranteed non-None values
                if isinstance(table_count, int) and isinstance(join_count, int):
                    if table_count >= 3 and join_count >= 2:
                        hints['complexity'] = 'high'
                    elif table_count <= 1:
                        hints['complexity'] = 'low'

                # Quality indicators
                if isinstance(total_columns, int) and total_columns == 0:
                    hints.update({
                        'recommended_action': 'use_fallback_or_retry',
                        'confidence': 0.2
                    })

        except Exception as e:
            self.logger.warning(f"Error extracting routing hints: {e}")
            hints.update({
                'recommended_action': 'proceed_with_caution',
                'complexity': 'unknown',
                'confidence': 0.5,
                'error_occurred': True
            })

        return hints

    def _assess_query_complexity(self, response: SchemaSearchResponse) -> str:
        """PYLANCE-SAFE: Assess query complexity with robust error handling"""
        try:
            complexity_score = 0

            # NLP complexity indicators
            nlp_insights = self._safe_get(response, 'nlp_insights')
            if nlp_insights and isinstance(nlp_insights, dict):
                detected_intent = self._safe_get(nlp_insights, 'detected_intent') or {}
                if isinstance(detected_intent, dict):
                    primary_intent = self._safe_get(detected_intent, 'primary', 'unknown')

                    # PYLANCE-SAFE: Intent scoring with proper dict typing
                    intent_scores: Dict[str, int] = {
                        'aggregation': 2,
                        'complex_analysis': 2,
                        'join_analysis': 2,
                        'simple_lookup': 0
                    }

                    if isinstance(primary_intent, str):
                        complexity_score += intent_scores.get(primary_intent, 1)

            # Data complexity indicators
            response_data = self._safe_get(response, 'data')
            if response_data and isinstance(response_data, dict):
                # FIXED: Safe access with None checks before comparison
                table_count = self._safe_get(response_data, 'table_count', 0)
                join_count = self._safe_get(response_data, 'join_count', 0)
                xml_columns = self._safe_get(response_data, 'total_xml_columns', 0)

                # PYLANCE-SAFE: Ensure values are not None before comparison
                table_count = table_count if table_count is not None else 0
                join_count = join_count if join_count is not None else 0
                xml_columns = xml_columns if xml_columns is not None else 0

                # Safe comparisons with guaranteed integer values
                if isinstance(table_count, int):
                    if table_count >= 3:
                        complexity_score += 2
                    elif table_count == 2:
                        complexity_score += 1

                if isinstance(join_count, int):
                    if join_count >= 2:
                        complexity_score += 2
                    elif join_count == 1:
                        complexity_score += 1

                if isinstance(xml_columns, int) and xml_columns > 0:
                    complexity_score += 1

            # Map score to complexity level
            if complexity_score >= 5:
                return 'very_high'
            elif complexity_score >= 4:
                return 'high'
            elif complexity_score >= 2:
                return 'medium'
            elif complexity_score >= 1:
                return 'low'
            else:
                return 'very_low'

        except Exception as e:
            self.logger.warning(f"Error assessing query complexity: {e}")
            return 'unknown'

    def _generate_next_steps(self, response: SchemaSearchResponse) -> List[str]:
        """PYLANCE-SAFE: Generate next steps with robust error handling"""
        next_steps = []

        try:
            status = self._safe_get(response, 'status', 'unknown')
            response_data = self._safe_get(response, 'data')

            if status == 'success' and response_data and isinstance(response_data, dict):
                # FIXED: Safe access with None checks
                table_count = self._safe_get(response_data, 'table_count', 0)
                total_columns = self._safe_get(response_data, 'total_columns', 0)

                # Ensure values are not None
                table_count = table_count if table_count is not None else 0
                total_columns = total_columns if total_columns is not None else 0

                if isinstance(table_count, int) and isinstance(total_columns, int):
                    if table_count > 0 and total_columns > 0:
                        next_steps.extend([
                            'proceed_to_prompt_builder',
                            'generate_sql_query',
                            'validate_sql_syntax'
                        ])

                        # Conditional steps with safe access
                        join_count = self._safe_get(response_data, 'join_count', 0)
                        join_count = join_count if join_count is not None else 0

                        if isinstance(join_count, int) and join_count > 0:
                            next_steps.insert(1, 'optimize_join_strategy')

                        has_xml_data = self._safe_get(response_data, 'has_xml_data', False)
                        if has_xml_data:
                            next_steps.insert(1, 'prepare_xml_extractions')

                    else:
                        next_steps.extend([
                            'retry_with_different_approach',
                            'use_fallback_schema_discovery',
                            'request_user_clarification'
                        ])

            elif status == 'error':
                next_steps.extend([
                    'log_error_details',
                    'try_alternative_processing',
                    'escalate_to_admin'
                ])

            else:
                next_steps.extend([
                    'analyze_partial_results',
                    'decide_proceed_or_retry',
                    'optimize_next_attempt'
                ])

        except Exception as e:
            self.logger.warning(f"Error generating next steps: {e}")
            next_steps = ['log_error', 'use_default_workflow']

        return next_steps

    def health_check(self) -> Dict[str, Any]:
        """ENHANCED: Orchestrator health check with comprehensive error handling"""
        try:
            # Safe access to component health
            component_health = {}
            bridge_stats = {}

            try:
                component_health = self.bridge.get_component_health() or {}
            except Exception as e:
                self.logger.warning(f"Failed to get component health: {e}")
                component_health = {}

            try:
                bridge_stats = self.bridge.get_bridge_statistics() or {}
            except Exception as e:
                self.logger.warning(f"Failed to get bridge statistics: {e}")
                bridge_stats = {}

            # Determine overall system health
            overall_status = 'healthy'
            if component_health and isinstance(component_health, dict):
                unhealthy_components = []
                for name, health in component_health.items():
                    status = self._safe_get(health, 'status')
                    # PYLANCE-SAFE: Check for .value attribute before accessing
                    if status is not None and hasattr(status, 'value'):
                        status_value = status.value
                    else:
                        status_value = str(status) if status else 'unknown'

                    if status_value in ['error', 'unhealthy']:
                        unhealthy_components.append(name)

                if unhealthy_components:
                    overall_status = 'error' if len(unhealthy_components) == len(component_health) else 'degraded'

            # Calculate system metrics
            uptime_seconds = time.time() - self.system_start_time
            uptime_hours = uptime_seconds / 3600

            success_rate = 0.0
            if self.total_orchestrator_requests > 0:
                success_rate = (self.successful_orchestrator_requests / self.total_orchestrator_requests) * 100

            # Build comprehensive health response
            health_response = {
                'system_name': self.system_name,
                'version': self.version,
                'overall_status': overall_status,
                'uptime_hours': round(uptime_hours, 2),

                # Orchestrator-level statistics
                'orchestrator_statistics': {
                    'total_requests': self.total_orchestrator_requests,
                    'successful_requests': self.successful_orchestrator_requests,
                    'success_rate_percent': round(success_rate, 2),
                    'failed_requests': self.total_orchestrator_requests - self.successful_orchestrator_requests
                },

                # Component health details
                'component_health': self._build_component_health_details(component_health),

                # Bridge performance statistics
                'bridge_performance': bridge_stats,

                # System capabilities
                'system_capabilities': self.get_capabilities(),

                # Health check metadata
                'health_check_timestamp': datetime.now().isoformat(),
                'supported_operations': [
                    'process_query',
                    'health_check',
                    'get_capabilities',
                    'get_statistics',
                    'cleanup'
                ]
            }

            return health_response

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                'system_name': self.system_name,
                'version': self.version,
                'overall_status': 'error',
                'error': str(e),
                'health_check_timestamp': datetime.now().isoformat()
            }

    def _build_component_health_details(self, component_health: Dict[str, Any]) -> Dict[str, Any]:
        """HELPER: Safely build component health details"""
        health_details = {}
        for name, health in component_health.items():
            try:
                health_details[name] = {
                    'status': self._extract_status_value(health),
                    'capabilities': self._safe_get(health, 'capabilities', []),
                    'supported_methods': self._safe_get(health, 'supported_methods', []),
                    'error_message': self._safe_get(health, 'error_message'),
                    'performance_metrics': self._safe_get(health, 'performance_metrics', {})
                }
            except Exception as e:
                self.logger.warning(f"Error processing health for {name}: {e}")
                health_details[name] = {
                    'status': 'error',
                    'error_message': f'Health processing failed: {str(e)}'
                }
        return health_details

    def _extract_status_value(self, health_obj: Any) -> str:
        """PYLANCE-SAFE: Safely extract status value from health object"""
        status = self._safe_get(health_obj, 'status')
        # FIXED: Check for None and hasattr before accessing .value
        if status is not None and hasattr(status, 'value'):
            return str(status.value)
        elif status is not None:
            return str(status)
        else:
            return 'unknown'

    def get_capabilities(self) -> Dict[str, Any]:
        """ENHANCED: Get system capabilities with robust error handling"""
        try:
            # Safe access to component health
            component_health = {}
            try:
                component_health = self.bridge.get_component_health() or {}
            except Exception as e:
                self.logger.warning(f"Failed to get component health for capabilities: {e}")
                component_health = {}

            capabilities_response = {
                'system_name': self.system_name,
                'version': self.version,

                # Core capabilities
                'core_features': [
                    'nlp_query_enhancement',
                    'schema_retrieval',
                    'intelligent_gap_analysis',
                    'xml_path_matching',
                    'join_discovery',
                    'intent_classification',
                    'semantic_entity_extraction',
                    'context_preservation',
                    'performance_monitoring'
                ],

                # Supported query types
                'supported_query_intents': self._get_supported_intents(),

                # Processing modes
                'processing_modes': [
                    'nlp_enhanced',
                    'schema_direct',
                    'intelligent_fallback',
                    'hybrid_approach'
                ],

                # Output formats
                'output_formats': [
                    'json_standardized',
                    'orchestrator_compatible',
                    'legacy_compatible'
                ],

                # Component capabilities
                'component_capabilities': self._build_component_capabilities(component_health),

                # Performance characteristics
                'performance_characteristics': {
                    'typical_response_time_ms': '500-2000',
                    'max_concurrent_requests': 10,
                    'supports_async_processing': True,
                    'supports_streaming': False,
                    'supports_caching': True
                },

                # Integration capabilities
                'integration_capabilities': {
                    'orchestrator_compatible': True,
                    'api_compatible': True,
                    'health_monitoring': True,
                    'graceful_degradation': True,
                    'error_recovery': True
                }
            }

            return capabilities_response

        except Exception as e:
            self.logger.error(f"Error getting capabilities: {e}")
            return {
                'system_name': self.system_name,
                'version': self.version,
                'error': str(e),
                'capabilities_available': False
            }

    def _get_supported_intents(self) -> List[str]:
        """HELPER: Get supported query intents safely"""
        try:
            if hasattr(QueryIntent, '__iter__'):
                return [intent.value for intent in QueryIntent]
            else:
                return ['lookup', 'aggregation', 'analysis', 'join_analysis']
        except Exception:
            return ['lookup', 'aggregation', 'analysis', 'join_analysis']

    def _build_component_capabilities(self, component_health: Dict[str, Any]) -> Dict[str, Any]:
        """HELPER: Safely build component capabilities"""
        capabilities = {}
        for name, health in component_health.items():
            try:
                status = self._extract_status_value(health)
                if status in ['healthy', 'degraded']:
                    component_capabilities = self._safe_get(health, 'capabilities', [])
                    if component_capabilities:
                        capabilities[name] = component_capabilities
            except Exception as e:
                self.logger.warning(f"Error processing capabilities for {name}: {e}")
        return capabilities

    def get_statistics(self) -> Dict[str, Any]:
        """ENHANCED: Get detailed system statistics with robust error handling"""
        try:
            # Safe access to bridge data
            bridge_stats = {}
            component_health = {}

            try:
                bridge_stats = self.bridge.get_bridge_statistics() or {}
            except Exception as e:
                self.logger.warning(f"Failed to get bridge statistics: {e}")
                bridge_stats = {}

            try:
                component_health = self.bridge.get_component_health() or {}
            except Exception as e:
                self.logger.warning(f"Failed to get component health for statistics: {e}")
                component_health = {}

            # Build comprehensive statistics
            stats_response = {
                'system_name': self.system_name,
                'version': self.version,
                'statistics_timestamp': datetime.now().isoformat(),

                # Orchestrator-level statistics
                'orchestrator_level': {
                    'total_requests': self.total_orchestrator_requests,
                    'successful_requests': self.successful_orchestrator_requests,
                    'failed_requests': self.total_orchestrator_requests - self.successful_orchestrator_requests,
                    'success_rate_percent': round(
                        (self.successful_orchestrator_requests / max(self.total_orchestrator_requests, 1)) * 100, 2
                    ),
                    'uptime_hours': round((time.time() - self.system_start_time) / 3600, 2)
                },

                # Bridge-level statistics
                'bridge_level': bridge_stats,

                # Component health summary
                'component_health_summary': self._build_health_summary(component_health),

                # System resource usage
                'system_resources': {
                    'memory_usage_estimated': 'monitoring_not_implemented',
                    'thread_pool_active': True,
                    'async_tasks_running': 'monitoring_not_implemented'
                }
            }

            return stats_response

        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {
                'system_name': self.system_name,
                'version': self.version,
                'error': str(e),
                'statistics_available': False
            }

    def _build_health_summary(self, component_health: Dict[str, Any]) -> Dict[str, str]:
        """HELPER: Safely build component health summary"""
        summary = {}
        for name, health in component_health.items():
            try:
                summary[name] = self._extract_status_value(health)
            except Exception as e:
                self.logger.warning(f"Error processing health summary for {name}: {e}")
                summary[name] = 'error'
        return summary

    async def cleanup(self):
        """ENHANCED: System cleanup with comprehensive error handling"""
        try:
            self.logger.info(f"Starting {self.system_name} cleanup for orchestrator")

            # Cleanup integration bridge with proper error handling
            if hasattr(self.bridge, 'cleanup') and callable(getattr(self.bridge, 'cleanup')):
                try:
                    await self.bridge.cleanup()  # type: ignore
                    self.logger.info("Bridge cleanup completed successfully")
                except Exception as e:
                    self.logger.warning(f"Bridge cleanup failed: {e}")
            else:
                self.logger.warning("Bridge cleanup method not available")

            # Update statistics
            cleanup_time = time.time() - self.system_start_time
            self.logger.info(f"{self.system_name} cleanup completed after {cleanup_time:.2f}s uptime")

        except Exception as e:
            self.logger.error(f"{self.system_name} cleanup error: {e}")

# Factory Functions

def create_orchestrator_compatible_system(
    nlp_processor: Any,
    schema_agent: Any,
    intelligent_agent: Optional[Any] = None,
    async_client_manager: Optional[Any] = None  # CRITICAL FIX: Added async_client_manager parameter
) -> OrchestratorCompatibleSystem:
    """
    ENHANCED: Create orchestrator-compatible system with AsyncClientManager integration
    FIXED: Added async_client_manager parameter to support shared AsyncClientManager
    """
    try:
        # Enhanced component validation
        if nlp_processor is None:
            raise ValueError("NLP processor cannot be None")

        if schema_agent is None:
            raise ValueError("Schema agent cannot be None")

        # Check required methods
        required_nlp_methods = ['process_analyst_query']
        for method in required_nlp_methods:
            if not hasattr(nlp_processor, method):
                raise ValueError(f"NLP processor missing required method: {method}")

        required_schema_methods = ['retrieve_complete_schema_json', 'search_schema']
        if not any(hasattr(schema_agent, method) for method in required_schema_methods):
            raise ValueError("Schema agent missing required methods")

        # CRITICAL FIX: Create integration bridge WITH AsyncClientManager
        bridge = NLPSchemaIntegrationBridge(
            nlp_processor=nlp_processor,
            schema_agent=schema_agent,
            intelligent_agent=intelligent_agent,
            async_client_manager=async_client_manager  # Pass AsyncClientManager to bridge
        )

        if bridge is None:
            raise RuntimeError("Failed to create integration bridge")

        # Create orchestrator-compatible system
        system = OrchestratorCompatibleSystem(bridge)

        if system is None:
            raise RuntimeError("Failed to create orchestrator-compatible system")

        logger.info("Orchestrator-compatible system created successfully with AsyncClientManager support")
        return system

    except Exception as e:
        logger.error(f"Failed to create orchestrator-compatible system: {e}")
        raise

def create_system_for_orchestrator_with_health_check(
    nlp_processor: Any,
    schema_agent: Any,
    intelligent_agent: Optional[Any] = None,
    async_client_manager: Optional[Any] = None,  # FIXED: Added async_client_manager parameter
    verify_components: bool = True
) -> OrchestratorCompatibleSystem:
    """
    ENHANCED: Factory with comprehensive component verification and AsyncClientManager support
    FIXED: Added async_client_manager parameter for complete integration
    """
    try:
        if verify_components:
            # Verify NLP Processor
            if not hasattr(nlp_processor, 'process_analyst_query'):
                raise RuntimeError("NLP Processor missing required method: process_analyst_query")

            # Verify Schema Agent
            if not (hasattr(schema_agent, 'retrieve_complete_schema_json') or
                    hasattr(schema_agent, 'search_schema')):
                raise RuntimeError("Schema Agent missing required methods")

            # Verify Intelligent Agent (if provided)
            if intelligent_agent and not hasattr(intelligent_agent, 'retrieve_complete_schema_json'):
                logger.warning("Intelligent Agent missing preferred method, but will continue")

        return create_orchestrator_compatible_system(
            nlp_processor, 
            schema_agent, 
            intelligent_agent,
            async_client_manager  # FIXED: Pass async_client_manager through
        )

    except Exception as e:
        logger.error(f"Enhanced factory failed: {e}")
        raise

# Utility Functions

async def test_orchestrator_system(system: OrchestratorCompatibleSystem) -> Dict[str, Any]:
    """ENHANCED: Test orchestrator system with comprehensive error handling"""
    test_results = {
        'test_timestamp': datetime.now().isoformat(),
        'system_name': system.system_name,
        'tests_performed': [],
        'overall_success': True
    }

    try:
        # Test 1: Health Check
        try:
            health = system.health_check()
            test_results['tests_performed'].append({
                'test_name': 'health_check',
                'success': health.get('overall_status') != 'error',
                'details': f"Status: {health.get('overall_status', 'unknown')}"
            })
        except Exception as e:
            test_results['tests_performed'].append({
                'test_name': 'health_check',
                'success': False,
                'details': f"Error: {str(e)}"
            })

        # Test 2: Capabilities
        try:
            capabilities = system.get_capabilities()
            test_results['tests_performed'].append({
                'test_name': 'get_capabilities',
                'success': 'core_features' in capabilities,
                'details': f"Features: {len(capabilities.get('core_features', []))}"
            })
        except Exception as e:
            test_results['tests_performed'].append({
                'test_name': 'get_capabilities',
                'success': False,
                'details': f"Error: {str(e)}"
            })

        # Test 3: Simple Query Processing
        try:
            simple_result = await system.process_query("customer account data")
            test_results['tests_performed'].append({
                'test_name': 'simple_query_processing',
                'success': simple_result.get('status') != 'error',
                'details': f"Status: {simple_result.get('status', 'unknown')}"
            })
        except Exception as e:
            test_results['tests_performed'].append({
                'test_name': 'simple_query_processing',
                'success': False,
                'details': f"Error: {str(e)}"
            })

        # Test 4: Statistics
        try:
            stats = system.get_statistics()
            test_results['tests_performed'].append({
                'test_name': 'get_statistics',
                'success': 'orchestrator_level' in stats,
                'details': f"Total requests: {stats.get('orchestrator_level', {}).get('total_requests', 0)}"
            })
        except Exception as e:
            test_results['tests_performed'].append({
                'test_name': 'get_statistics',
                'success': False,
                'details': f"Error: {str(e)}"
            })

        # Determine overall success
        test_results['overall_success'] = all(
            test['success'] for test in test_results['tests_performed']
        )

    except Exception as e:
        test_results['overall_success'] = False
        test_results['error'] = str(e)

    return test_results

# Main Usage Example

async def main_orchestrator_integration_example():
    """ENHANCED: Example of orchestrator integration with AsyncClientManager support"""
    try:
        # Step 1: Initialize components (replace with actual component initialization)
        from agent.nlp_processor.main import NLPProcessor
        from agent.schema_searcher.core.retrieval_agent import create_schema_retrieval_agent
        from agent.schema_searcher.core.intelligent_retrieval_agent import create_intelligent_retrieval_agent

        nlp_processor = NLPProcessor()
        schema_agent = create_schema_retrieval_agent(json_mode=True)
        intelligent_agent = create_intelligent_retrieval_agent(include_schema_agent=True)

        # ENHANCED: Include AsyncClientManager in system creation (if available)
        async_client_manager = None  # Replace with actual AsyncClientManager instance

        # Step 2: Create orchestrator-compatible system with AsyncClientManager support
        orchestrator_system = create_orchestrator_compatible_system(
            nlp_processor=nlp_processor,
            schema_agent=schema_agent,
            intelligent_agent=intelligent_agent,
            async_client_manager=async_client_manager  # FIXED: Pass AsyncClientManager
        )

        # Step 3: Test system
        test_results = await test_orchestrator_system(orchestrator_system)
        print(f"System tests: {'PASSED' if test_results['overall_success'] else 'FAILED'}")

        # Step 4: Process sample queries
        sample_queries = [
            "Show me customer account balances for Mumbai branch",
            "What are the total loan amounts by region for last quarter",
            "Find defaulting customers with collateral details"
        ]

        for query in sample_queries:
            print(f"\nProcessing: {query}")
            result = await orchestrator_system.process_query(query)

            if result['status'] == 'success':
                print(f"Success - Tables: {len(result['data'].get('tables', []))}")
                routing_hints = result.get('orchestrator_routing_hints', {})
                print(f"Routing: {routing_hints.get('recommended_action', 'unknown')}")
                print(f"Complexity: {result.get('query_complexity', 'unknown')}")
                print(f"Time: {result.get('execution_time_ms', 0):.2f}ms")
            else:
                print(f"Failed: {result.get('error', 'Unknown error')}")

        # Step 5: Show system health and capabilities
        health = orchestrator_system.health_check()
        print(f"\nSystem Health: {health.get('overall_status', 'unknown')}")

        capabilities = orchestrator_system.get_capabilities()
        print(f"Features: {len(capabilities.get('core_features', []))} available")

        # Step 6: Cleanup
        await orchestrator_system.cleanup()
        print("\nSystem cleanup completed")

    except Exception as e:
        print(f"Integration example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main_orchestrator_integration_example())
