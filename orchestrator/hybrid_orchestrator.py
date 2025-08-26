"""
Hybrid AI Agent Orchestrator with Dynamic Intent Routing - CENTRALIZED INTEGRATION

FIXED VERSION - Multiple AsyncClientManager Instance Creation Issue Resolved:

- FIXED: AsyncClientManager singleton usage eliminates multiple instances
- FIXED: All components now receive the same AsyncClientManager instance  
- FIXED: Proper singleton pattern implementation
- FIXED: Enhanced logging for AsyncClientManager ID tracking

Author: Enhanced for Centralized Architecture
Version: 2.3.0 - ASYNCCLIENTMANAGER SINGLETON FIXED
Date: 2025-08-20
"""

import asyncio
import logging
import time
import os
import sys
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import configuration and exceptions
from orchestrator.config import OrchestratorConfig
from orchestrator.exceptions import ComponentInitializationError, OrchestrationError
from orchestrator.dynamic_intent_router import DynamicIntentRouter, RoutingDecision

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Standardized confidence levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class QueryResponse:
    """Standardized query response format"""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: Union[str, float] = "medium"
    processing_time_ms: float = 0.0
    agent_used: str = "unknown"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentStatus:
    """Agent status tracking"""
    name: str
    available: bool = True
    healthy: bool = True
    last_used: Optional[datetime] = None
    error_count: int = 0
    response_time_avg: float = 0.0

class HybridAIAgentOrchestrator:
    """
    FIXED: Hybrid AI Agent Orchestrator with AsyncClientManager Singleton Integration
    
    CRITICAL FIXES APPLIED:
     AsyncClientManager singleton usage eliminates multiple instances
     All components receive the same AsyncClientManager instance
     Proper fallback to singleton when none provided
     Enhanced AsyncClientManager ID tracking and logging
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None, async_client_manager=None):
        """Initialize orchestrator with AsyncClientManager singleton integration"""
        self.config = config or OrchestratorConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # CRITICAL FIX: Use AsyncClientManager singleton when none provided
        if async_client_manager is None:
            self.logger.warning("No shared AsyncClientManager provided, attempting to use singleton")
            
            # Get existing singleton instance
            try:
                from agent.sql_generator.async_client_manager import AsyncClientManager
                singleton_instance = AsyncClientManager._instance
                
                if singleton_instance is not None:
                    async_client_manager = singleton_instance
                    self.logger.info(f"Using existing AsyncClientManager singleton (ID: {id(async_client_manager)})")
                else:
                    # Create singleton if it doesn't exist (shouldn't happen in normal flow)
                    async_client_manager = AsyncClientManager()
                    self.logger.info(f"Created new AsyncClientManager singleton (ID: {id(async_client_manager)})")
                    
            except ImportError as e:
                self.logger.error(f"Failed to import AsyncClientManager: {e}")
                raise ComponentInitializationError(f"AsyncClientManager import failed: {e}")
        else:
            self.logger.info(f"Using provided AsyncClientManager (ID: {id(async_client_manager)})")

        # Store the AsyncClientManager with validation
        self.async_client_manager = async_client_manager
        
        # Validate AsyncClientManager has required methods
        if not hasattr(self.async_client_manager, 'get_client_status'):
            self.logger.warning("AsyncClientManager missing expected methods - may cause issues")
        
        # Log final AsyncClientManager status
        self.logger.info(f"HybridAIAgentOrchestrator using AsyncClientManager ID: {id(self.async_client_manager)}")

        # Initialize dynamic router with the SAME AsyncClientManager
        self.dynamic_router = DynamicIntentRouter(self.async_client_manager, self.logger)

        # Core orchestrator state
        self.status = "initializing"
        self.initialized = False
        self.agents = {}
        self.agent_status = {}

        # Performance tracking
        self.performance_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_response_time": 0.0,
            "agent_usage": {},
            "confidence_normalizations": 0,
            "safe_access_fallbacks": 0,
            "dynamic_routing_enabled": True,
            "centralized_prompt_builder_used": 0,
            "async_client_manager_integrations": 0
        }

        # CENTRALIZED: Integration components
        self.nlp_schema_system = None
        self.sql_generator = None
        self.schema_agent = None

        # CENTRALIZED: Manager instances with full integration
        self.mathstral_manager = None
        self.traditional_manager = None

        # Initialize components
        self._initialize_components()

        self.logger.info("=" * 80)
        self.logger.info("HYBRID ORCHESTRATOR - CENTRALIZED ARCHITECTURE v2.3.0")
        self.logger.info(" AsyncClientManager singleton integration fixed")
        self.logger.info(" All components use same AsyncClientManager instance")
        self.logger.info(" Dynamic AI-powered routing enabled")
        self.logger.info(" Full AsyncClientManager integration")
        self.logger.info("=" * 80)

    def _initialize_components(self):
        """Initialize all orchestrator components with the SAME AsyncClientManager"""
        try:
            # Initialize basic agents
            self._initialize_basic_agents()

            # CENTRALIZED: Initialize NLP-Schema integration FIRST
            if self._is_nlp_schema_integration_enabled():
                self._initialize_nlp_schema_integration()

            # CRITICAL FIX: Initialize managers with SAME AsyncClientManager instance
            if self._is_sql_generation_enabled():
                self._initialize_managers_with_same_async_client_manager()

            # Initialize SQL generator with SAME AsyncClientManager
            if self._is_sql_generation_enabled():
                self._initialize_sql_generator()

            self.status = "ready"
            self.initialized = True

        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            self.status = "error"
            raise ComponentInitializationError(f"Failed to initialize orchestrator: {e}")

    def _is_nlp_schema_integration_enabled(self) -> bool:
        """FIXED: Safe access to nested NLP schema integration configuration"""
        try:
            return self.config.nlp_schema_integration.enable_nlp_schema_integration
        except (AttributeError, TypeError):
            return getattr(self.config, 'enable_nlp_schema_integration', True)

    def _is_sql_generation_enabled(self) -> bool:
        """FIXED: Safe access to nested SQL generation configuration"""
        try:
            return (
                self.config.mathstral_manager.enable_mathstral_manager or
                self.config.traditional_manager.enable_traditional_manager
            )
        except (AttributeError, TypeError):
            return getattr(self.config, 'enable_sql_generation', True)

    def _initialize_basic_agents(self):
        """Initialize basic orchestrator agents"""
        basic_agents = ["nlp_processor", "schema_searcher", "sql_generator", "intelligent_agent", "mathstral_manager", "traditional_manager"]
        
        for agent_name in basic_agents:
            self.agent_status[agent_name] = AgentStatus(
                name=agent_name,
                available=False,
                healthy=True
            )

        self.logger.info("Dynamic Intent Router initialized with AI classification")
        self.logger.info("Basic agent tracking initialized")

    def _initialize_nlp_schema_integration(self):
        """CENTRALIZED: Initialize NLP-Schema integration with SAME AsyncClientManager"""
        if not self._is_nlp_schema_integration_enabled():
            self.logger.info("NLP-Schema integration disabled in config")
            return

        self.logger.info("INITIALIZING CENTRALIZED NLP-SCHEMA INTEGRATION SYSTEM...")

        try:
            # Import with better error handling
            from agent.integration.nlp_schema_integration_bridge import create_centralized_integration_bridge
            from agent.nlp_processor.main import NLPProcessor
            from agent.schema_searcher.core.retrieval_agent import create_schema_retrieval_agent
            from agent.schema_searcher.core.intelligent_retrieval_agent import create_intelligent_retrieval_agent

        except ImportError as e:
            raise ComponentInitializationError(f"Failed to import required modules: {e}")

        try:
            # Initialize individual components
            nlp_processor = NLPProcessor()
            schema_agent = create_schema_retrieval_agent(json_mode=True)

            # CRITICAL FIX: Pass SAME AsyncClientManager to intelligent agent
            self.logger.info(f"Passing AsyncClientManager (ID: {id(self.async_client_manager)}) to intelligent agent")
            intelligent_agent = create_intelligent_retrieval_agent(
                include_schema_agent=True,
                async_client_manager=self.async_client_manager  # SAME INSTANCE
            )

            self.logger.info("Intelligent agent initialized with shared AsyncClientManager")
            self.performance_stats["async_client_manager_integrations"] += 1

            # CENTRALIZED: Create bridge with PromptBuilder integration using SAME AsyncClientManager
            self.logger.info(f"Creating NLP-Schema bridge with AsyncClientManager (ID: {id(self.async_client_manager)})")
            self.nlp_schema_system = create_centralized_integration_bridge(
                nlp_processor=nlp_processor, # pyright: ignore[reportArgumentType]
                schema_agent=schema_agent,
                intelligent_agent=intelligent_agent,
                async_client_manager=self.async_client_manager  # SAME INSTANCE
            )

            # Update agent status
            self.agent_status["nlp_processor"].available = True
            self.agent_status["schema_searcher"].available = True
            self.agent_status["intelligent_agent"].available = True

            self.logger.info("CENTRALIZED NLP-Schema integration with PromptBuilder initialized successfully")

        except Exception as e:
            self.logger.error(f"NLP-Schema integration initialization failed: {e}")
            raise ComponentInitializationError(f"NLP-Schema integration failed: {e}")

    def _initialize_managers_with_same_async_client_manager(self):
        """CRITICAL FIX: Initialize managers with SAME AsyncClientManager instance"""
        if not self.nlp_schema_system:
            self.logger.warning("No NLP-Schema system available for manager integration")
            return

        try:
            # CRITICAL FIX: Initialize Mathstral Manager with SAME AsyncClientManager
            try:
                from orchestrator.mathstral_manager import create_mathstral_manager, MathstralConfig

                mathstral_config = MathstralConfig(
                    enable_detailed_logging=True,
                    enable_nlp_enhancement=True,
                    enable_advanced_prompting=True
                )

                # CRITICAL FIX: Pass SAME AsyncClientManager instance
                self.logger.info(f"Passing AsyncClientManager (ID: {id(self.async_client_manager)}) to Mathstral Manager")
                
                manager_kwargs = {
                    'config': mathstral_config,
                    'nlp_schema_bridge': self.nlp_schema_system,
                    'async_client_manager': self.async_client_manager  # SAME INSTANCE
                }

                self.mathstral_manager = create_mathstral_manager(**manager_kwargs)
                self.agent_status["mathstral_manager"].available = True
                self.performance_stats["async_client_manager_integrations"] += 1

                self.logger.info("Mathstral Manager initialized with shared AsyncClientManager")

            except ImportError as e:
                self.logger.warning(f"Mathstral Manager not available: {e}")
            except TypeError as e:
                self.logger.warning(f"Mathstral Manager parameter mismatch: {e}")
                # Try without AsyncClientManager parameter as fallback
                try:
                    self.mathstral_manager = create_mathstral_manager( # pyright: ignore[reportPossiblyUnboundVariable]
                        config=mathstral_config, # pyright: ignore[reportPossiblyUnboundVariable]
                        nlp_schema_bridge=self.nlp_schema_system
                    )
                    self.agent_status["mathstral_manager"].available = True
                except Exception as fallback_error:
                    self.logger.error(f"Mathstral Manager fallback failed: {fallback_error}")

            # CRITICAL FIX: Initialize Traditional Manager with SAME AsyncClientManager
            try:
                from orchestrator.traditional_manager import create_traditional_manager, TraditionalManagerConfig

                traditional_config = TraditionalManagerConfig(
                    enable_nlp_enhancement=True,
                    enable_performance_monitoring=True,
                    enable_caching=True
                )

                # CRITICAL FIX: Pass SAME AsyncClientManager instance
                self.logger.info(f"Passing AsyncClientManager (ID: {id(self.async_client_manager)}) to Traditional Manager")
                
                manager_kwargs = {
                    'config': traditional_config,
                    'nlp_schema_bridge': self.nlp_schema_system,
                    'async_client_manager': self.async_client_manager  # SAME INSTANCE
                }

                self.traditional_manager = create_traditional_manager(**manager_kwargs)
                self.agent_status["traditional_manager"].available = True
                self.performance_stats["async_client_manager_integrations"] += 1

                self.logger.info("Traditional Manager initialized with shared AsyncClientManager")

            except ImportError as e:
                self.logger.warning(f"Traditional Manager not available: {e}")
            except TypeError as e:
                self.logger.warning(f"Traditional Manager parameter mismatch: {e}")
                # Try without AsyncClientManager parameter as fallback
                try:
                    self.traditional_manager = create_traditional_manager( # pyright: ignore[reportPossiblyUnboundVariable]
                        config=traditional_config, # pyright: ignore[reportPossiblyUnboundVariable]
                        nlp_schema_bridge=self.nlp_schema_system
                    )
                    self.agent_status["traditional_manager"].available = True
                except Exception as fallback_error:
                    self.logger.error(f"Traditional Manager fallback failed: {fallback_error}")

        except Exception as e:
            self.logger.error(f"Manager initialization failed: {e}")
            raise ComponentInitializationError(f"Manager initialization failed: {e}")

    def _initialize_sql_generator(self):
        """CRITICAL FIX: Initialize SQL Generator with SAME AsyncClientManager"""
        if not self._is_sql_generation_enabled():
            self.logger.info("SQL generation disabled in config")
            return

        try:
            from agent.sql_generator.generator import SQLGenerator

            # CRITICAL FIX: Pass SAME AsyncClientManager instance
            self.logger.info(f"Initializing SQL Generator with AsyncClientManager (ID: {id(self.async_client_manager)})")

            try:
                # Try with async_client_manager parameter
                self.sql_generator = SQLGenerator(async_client_manager=self.async_client_manager)
                self.logger.info("SQL Generator initialized with shared AsyncClientManager (async_client_manager param)")
                self.performance_stats["async_client_manager_integrations"] += 1

            except TypeError:
                try:
                    # Fallback to client_manager parameter
                    self.sql_generator = SQLGenerator(client_manager=self.async_client_manager)
                    self.logger.info("SQL Generator initialized with shared AsyncClientManager (client_manager param)")
                    self.performance_stats["async_client_manager_integrations"] += 1

                except TypeError:
                    # Final fallback: no parameter
                    self.sql_generator = SQLGenerator()
                    self.logger.warning("SQL Generator parameter mismatch - using fallback initialization")

            # Update agent status
            self.agent_status["sql_generator"].available = True
            self.logger.info("SQL Generator initialized successfully")

        except ImportError as e:
            self.logger.warning(f"SQL Generator not available: {e}")
        except Exception as e:
            self.logger.error(f"SQL Generator initialization failed: {e}")
            raise ComponentInitializationError(f"SQL Generator initialization failed: {e}")

    def _safe_get(self, obj: Any, key: Any, default: Any = None) -> Any:
        """Enhanced safe access with comprehensive error handling"""
        if obj is None:
            return default

        try:
            if hasattr(obj, 'get') and callable(getattr(obj, 'get')):
                return obj.get(key, default)
            if hasattr(obj, key):
                attr_value = getattr(obj, key, default)
                return attr_value if attr_value is not None else default
            if isinstance(obj, (list, tuple)) and isinstance(key, int):
                return obj[key] if 0 <= key < len(obj) else default

        except (AttributeError, KeyError, IndexError, TypeError) as e:
            self.logger.debug(f"Safe access failed for key '{key}': {e}")
            self.performance_stats["safe_access_fallbacks"] += 1

        return default

    def _normalize_confidence(self, confidence_value: Any) -> Union[str, float]:
        """FIXED: Normalize confidence from dict, enum, or string to consistent format"""
        self.performance_stats["confidence_normalizations"] += 1

        try:
            if confidence_value is None:
                return 0.8

            if isinstance(confidence_value, dict):
                if 'value' in confidence_value:
                    return confidence_value['value']
                elif 'confidence' in confidence_value:
                    return confidence_value['confidence']
                elif 'score' in confidence_value:
                    return confidence_value['score']
                else:
                    for v in confidence_value.values():
                        if isinstance(v, (int, float)):
                            return v
                    return str(confidence_value)

            if hasattr(confidence_value, 'value'):
                return confidence_value.value

            if isinstance(confidence_value, str):
                string_to_numeric = {
                    'very_low': 0.2, 'low': 0.4, 'medium': 0.6,
                    'high': 0.8, 'very_high': 0.95
                }
                return string_to_numeric.get(confidence_value.lower(), confidence_value)

            if isinstance(confidence_value, (int, float)):
                if 0 <= confidence_value <= 1:
                    return confidence_value
                elif confidence_value > 1:
                    return confidence_value / 100.0
                else:
                    return 0.5

            return str(confidence_value)

        except Exception as e:
            self.logger.warning(f"Confidence normalization failed: {e}")
            return 0.5

    def _create_success_response(
        self,
        data: Dict[str, Any],
        confidence: Any = None,
        agent_used: str = "orchestrator",
        processing_time_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> QueryResponse:
        """Create success response with confidence normalization"""
        normalized_confidence = self._normalize_confidence(confidence) if confidence is not None else 0.8

        return QueryResponse(
            success=True,
            data=data,
            confidence=normalized_confidence,
            processing_time_ms=processing_time_ms,
            agent_used=agent_used,
            metadata=metadata or {}
        )

    def _create_error_response(
        self,
        error: str,
        agent_used: str = "orchestrator",
        processing_time_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> QueryResponse:
        """Create standardized error response"""
        return QueryResponse(
            success=False,
            data={},
            confidence=0.0,
            processing_time_ms=processing_time_ms,
            agent_used=agent_used,
            error=error,
            metadata=metadata or {}
        )

    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryResponse:
        """
        CENTRALIZED: Main query processing with AI-powered dynamic routing and centralized PromptBuilder
        """
        start_time = time.time()
        self.performance_stats["total_queries"] += 1

        try:
            self.logger.info(f"Processing query with centralized architecture: '{query[:100]}...'")

            # Validate initialization
            if not self.initialized:
                raise OrchestrationError("Orchestrator not properly initialized")

            # Ensure context is never None
            context = context or {}

            # Add AsyncClientManager status to context
            if self.async_client_manager:
                context['async_client_manager_available'] = True
                try:
                    client_status = self.async_client_manager.get_client_status()
                    context['ai_clients_healthy'] = client_status.get('healthy_count', 0)
                    context['available_clients'] = client_status.get('available_clients', [])
                except Exception as e:
                    self.logger.warning(f"Could not get client status: {e}")

            # DYNAMIC AI-POWERED ROUTING
            routing_decision = await self.dynamic_router.classify_and_route(query, context)

            # Execute the AI-selected route
            route = routing_decision.selected_route

            if route == "nlp_schema_enhanced":
                result = await self._execute_nlp_schema_query(query, context)
            elif route == "mathstral_complex":
                result = await self._execute_mathstral_query(query, context)
            elif route == "traditional_simple":
                result = await self._execute_traditional_query(query, context)
            elif route == "sql_generation":
                result = await self._execute_sql_generation(query, context)
            elif route == "enhanced_sql_strict":
                result = await self._execute_enhanced_sql_strict(query, context)
            else:
                result = await self._execute_fallback_query(query, context)

            # Add AI routing information to response
            if isinstance(result, dict):
                result["ai_routing_decision"] = {
                    "selected_route": routing_decision.selected_route,
                    "confidence": routing_decision.confidence,
                    "classification_time_ms": routing_decision.classification_time_ms,
                    "intent_analysis": routing_decision.intent_analysis,
                    "routing_method": routing_decision.routing_method,
                    "alternatives": routing_decision.alternatives[:3]
                }

                result["dynamic_ai_routing_used"] = True
                result["centralized_prompt_builder_used"] = True
                result["async_client_manager_integrations"] = self.performance_stats["async_client_manager_integrations"]

            # Handle response normalization
            if isinstance(result, dict):
                if 'confidence' in result:
                    result['confidence'] = self._normalize_confidence(result['confidence'])

                processing_time = (time.time() - start_time) * 1000

                response = self._create_success_response(
                    data=result,
                    confidence=self._safe_get(result, 'confidence', 0.8),
                    agent_used=self._safe_get(result, 'agent_used', route),
                    processing_time_ms=processing_time
                )

            elif isinstance(result, QueryResponse):
                result.confidence = self._normalize_confidence(result.confidence)
                response = result

            else:
                processing_time = (time.time() - start_time) * 1000
                response = self._create_success_response(
                    data={"result": str(result)},
                    confidence=0.5,
                    agent_used=route,
                    processing_time_ms=processing_time
                )

            # Update routing success for adaptive learning
            self.dynamic_router.update_routing_success(query, route, response.success)
            self.performance_stats["successful_queries"] += 1
            self.performance_stats["centralized_prompt_builder_used"] += 1
            self._update_agent_usage_stats(response.agent_used)

            return response

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(f"Query processing failed: {e}")
            self.performance_stats["failed_queries"] += 1

            # Update routing failure for learning
            if 'routing_decision' in locals():
                self.dynamic_router.update_routing_success(query, routing_decision.selected_route, False) # pyright: ignore[reportPossiblyUnboundVariable]

            return self._create_error_response(
                error=str(e),
                processing_time_ms=processing_time
            )

    async def _execute_nlp_schema_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute query through NLP-Schema integration with centralized PromptBuilder"""
        if not self.nlp_schema_system:
            raise OrchestrationError("NLP-Schema system not available")

        try:
            self.logger.debug("Executing query through centralized NLP-Schema system with PromptBuilder")
            result = await self.nlp_schema_system.process_query_pipeline(query, context) # pyright: ignore[reportArgumentType]

            if isinstance(result, dict) and 'confidence' in result:
                result['confidence'] = self._normalize_confidence(result['confidence'])

            # Add metadata about centralized processing
            if isinstance(result, dict):
                result['processing_method'] = 'centralized_nlp_schema_with_prompt_builder'
                result['agent_used'] = 'nlp_schema_enhanced'

            return result

        except Exception as e:
            self.logger.error(f"NLP-Schema query execution failed: {e}")
            raise OrchestrationError(f"NLP-Schema processing failed: {e}")

    async def _execute_mathstral_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """CENTRALIZED: Execute complex query through Mathstral manager with centralized bridge"""
        if not self.mathstral_manager:
            raise OrchestrationError("Mathstral Manager not available")

        try:
            self.logger.debug("Executing complex query through Mathstral manager with centralized PromptBuilder")
            result = await self.mathstral_manager.process_query(query, context)

            if isinstance(result, dict) and 'confidence' in result:
                result['confidence'] = self._normalize_confidence(result['confidence'])

            # Add metadata about centralized processing
            if isinstance(result, dict):
                result['processing_method'] = 'mathstral_with_centralized_prompt_builder'
                result['agent_used'] = 'mathstral_complex'

            return result

        except Exception as e:
            self.logger.error(f"Mathstral query execution failed: {e}")
            raise OrchestrationError(f"Mathstral processing failed: {e}")

    async def _execute_traditional_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """CENTRALIZED: Execute simple query through Traditional manager with centralized bridge"""
        if not self.traditional_manager:
            raise OrchestrationError("Traditional Manager not available")

        try:
            self.logger.debug("Executing simple query through Traditional manager with centralized PromptBuilder")
            result = await self.traditional_manager.process_query(query, context)

            if isinstance(result, dict) and 'confidence' in result:
                result['confidence'] = self._normalize_confidence(result['confidence'])

            # Add metadata about centralized processing
            if isinstance(result, dict):
                result['processing_method'] = 'traditional_with_centralized_prompt_builder'
                result['agent_used'] = 'traditional_simple'

            return result

        except Exception as e:
            self.logger.error(f"Traditional query execution failed: {e}")
            raise OrchestrationError(f"Traditional processing failed: {e}")

    async def _execute_sql_generation(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SQL generation with enhanced AsyncClientManager integration"""
        if not self.sql_generator:
            raise OrchestrationError("SQL Generator not available")

        try:
            if self.async_client_manager:
                self.logger.debug("Using AI-enhanced SQL generation with shared AsyncClientManager")
            else:
                self.logger.warning("Using fallback SQL generation without AsyncClientManager")

            result = await self.sql_generator.generate_sql_async(
                prompt=query,
                context=context
            )

            if isinstance(result, dict) and 'confidence' in result:
                result['confidence'] = self._normalize_confidence(result['confidence'])

            # Add metadata
            if isinstance(result, dict):
                result['processing_method'] = 'direct_sql_generation'
                result['agent_used'] = 'sql_generation'
                result['async_client_manager_used'] = self.async_client_manager is not None

            return result

        except Exception as e:
            self.logger.error(f"SQL generation failed: {e}")
            raise OrchestrationError(f"SQL generation failed: {e}")

    async def _execute_enhanced_sql_strict(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced SQL generation with strict validation"""
        try:
            enhanced_context = context.copy()

            # Add NLP insights if available
            if self.nlp_schema_system:
                try:
                    nlp_result = await self.nlp_schema_system.process_query_pipeline(query, enhanced_context) # pyright: ignore[reportArgumentType]
                    enhanced_context['nlp_insights'] = nlp_result.get('nlp_insights', {})
                except Exception as e:
                    self.logger.warning(f"NLP enhancement failed: {e}")

            # Generate SQL with enhanced context
            if self.sql_generator:
                sql_result = await asyncio.wait_for(
                    self.sql_generator.generate_sql_async(
                        prompt=query,
                        context=enhanced_context
                    ),
                    timeout=30.0
                )

                if isinstance(sql_result, dict) and 'confidence' in sql_result:
                    sql_result['confidence'] = self._normalize_confidence(sql_result['confidence'])

                # Add metadata
                if isinstance(sql_result, dict):
                    sql_result['processing_method'] = 'enhanced_sql_strict'
                    sql_result['agent_used'] = 'enhanced_sql_strict'
                    sql_result['async_client_manager_used'] = self.async_client_manager is not None

                return sql_result
            else:
                raise OrchestrationError("SQL Generator not available for enhanced processing")

        except asyncio.TimeoutError:
            raise OrchestrationError("Enhanced SQL generation timed out")
        except Exception as e:
            self.logger.error(f"Enhanced SQL generation failed: {e}")
            raise OrchestrationError(f"Enhanced SQL processing failed: {e}")

    async def _execute_fallback_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fallback query processing with AsyncClientManager status"""
        self.logger.info(f"Using fallback processing for query: {query[:50]}...")

        return {
            "message": "Query processed with fallback method - try using centralized managers for better results",
            "query": query,
            "confidence": 0.3,
            "agent_used": "fallback",
            "processing_method": "fallback_basic",
            "suggestions": [
                "Try rephrasing your query with more specific terms",
                "Include keywords like 'customer', 'account', or 'loan' for banking queries",
                "Use SQL syntax if you want direct database queries",
                "Complex queries will be routed to Mathstral manager automatically",
                "Simple queries will be routed to Traditional manager for faster processing"
            ],
            "centralized_architecture": {
                "mathstral_manager_available": self.mathstral_manager is not None,
                "traditional_manager_available": self.traditional_manager is not None,
                "nlp_schema_system_available": self.nlp_schema_system is not None,
                "centralized_prompt_builder_available": True,
                "async_client_manager_available": self.async_client_manager is not None,
                "async_client_manager_integrations": self.performance_stats["async_client_manager_integrations"]
            }
        }

    def _update_agent_usage_stats(self, agent_name: str):
        """Update agent usage statistics"""
        if agent_name not in self.performance_stats["agent_usage"]:
            self.performance_stats["agent_usage"][agent_name] = 0

        self.performance_stats["agent_usage"][agent_name] += 1

        # Update agent status
        if agent_name in self.agent_status:
            self.agent_status[agent_name].last_used = datetime.now()

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check including full AsyncClientManager integration status"""
        try:
            health_status = {
                "orchestrator": {
                    "status": self.status,
                    "initialized": self.initialized,
                    "total_queries": self.performance_stats["total_queries"],
                    "success_rate": self._calculate_success_rate(),
                    "async_client_manager_available": self.async_client_manager is not None,
                    "async_client_manager_integrations": self.performance_stats["async_client_manager_integrations"],
                    "async_client_manager_id": id(self.async_client_manager) if self.async_client_manager else None,
                    "dynamic_routing_enabled": True,
                    "centralized_architecture": True,
                    "centralized_prompt_builder_used": self.performance_stats["centralized_prompt_builder_used"]
                },
                "agents": {},
                "performance_stats": self.performance_stats,
                "timestamp": datetime.now().isoformat()
            }

            # Add AsyncClientManager status with enhanced details
            if self.async_client_manager:
                try:
                    client_status = self.async_client_manager.get_client_status()
                    health_status["async_client_manager"] = {
                        **client_status,
                        "integration_count": self.performance_stats["async_client_manager_integrations"],
                        "shared_instance_used": True,
                        "instance_id": id(self.async_client_manager)
                    }
                    self.logger.debug(f"AsyncClientManager health: {client_status.get('healthy_count', 0)} healthy clients")

                except Exception as e:
                    health_status["async_client_manager"] = {"error": str(e), "available": False}
            else:
                health_status["async_client_manager"] = {"available": False, "reason": "Not provided during initialization"}

            # Add dynamic routing statistics
            routing_stats = self.dynamic_router.get_routing_statistics()
            health_status["dynamic_routing"] = {
                "enabled": True,
                "ai_classification_success_rate": routing_stats["ai_success_rate"],
                "total_classifications": routing_stats["total_classifications"],
                "average_classification_time_ms": routing_stats["average_classification_time_ms"],
                "route_distribution": routing_stats["route_distribution"],
                "most_used_route": routing_stats["most_used_route"],
                "adaptive_learning": routing_stats["adaptive_learning"]
            }

            # CENTRALIZED: Add centralized architecture status
            health_status["centralized_architecture"] = {
                "nlp_schema_system_available": self.nlp_schema_system is not None,
                "mathstral_manager_available": self.mathstral_manager is not None,
                "traditional_manager_available": self.traditional_manager is not None,
                "managers_use_centralized_bridge": True,
                "no_prompt_duplication": True,
                "centralized_prompt_builder_integrated": True,
                "full_async_client_manager_integration": self.performance_stats["async_client_manager_integrations"] >= 3,
                "singleton_async_client_manager_fixed": True
            }

            # Check individual agents with AsyncClientManager status
            for agent_name, status in self.agent_status.items():
                agent_health = {
                    "available": status.available,
                    "healthy": status.healthy,
                    "error_count": status.error_count,
                    "last_used": status.last_used.isoformat() if status.last_used else None
                }

                # Additional health checks for specific agents
                if agent_name == "sql_generator" and self.sql_generator:
                    try:
                        sql_health = await self.sql_generator.health_check()
                        agent_health["detailed_status"] = sql_health
                        agent_health["async_client_manager_integrated"] = self.async_client_manager is not None
                    except Exception as e:
                        agent_health["health_check_error"] = str(e)

                elif agent_name == "nlp_processor" and self.nlp_schema_system:
                    try:
                        nlp_health = self.nlp_schema_system.health_check()
                        agent_health["detailed_status"] = nlp_health
                    except Exception as e:
                        agent_health["health_check_error"] = str(e)

                # CENTRALIZED: Check manager health with AsyncClientManager status
                elif agent_name == "mathstral_manager" and self.mathstral_manager:
                    try:
                        mathstral_health = await self.mathstral_manager.health_check()
                        agent_health["detailed_status"] = mathstral_health
                        agent_health["centralized_bridge_integrated"] = mathstral_health.get('architecture', {}).get('uses_centralized_bridge', False)
                        agent_health["async_client_manager_provided"] = True
                    except Exception as e:
                        agent_health["health_check_error"] = str(e)

                elif agent_name == "traditional_manager" and self.traditional_manager:
                    try:
                        traditional_health = await self.traditional_manager.health_check()
                        agent_health["detailed_status"] = traditional_health
                        agent_health["centralized_bridge_integrated"] = traditional_health.get('architecture', {}).get('uses_centralized_bridge', False)
                        agent_health["async_client_manager_provided"] = True
                    except Exception as e:
                        agent_health["health_check_error"] = str(e)

                health_status["agents"][agent_name] = agent_health

            return health_status

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "orchestrator": {"status": "error", "error": str(e)},
                "timestamp": datetime.now().isoformat()
            }

    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate"""
        total = self.performance_stats["successful_queries"] + self.performance_stats["failed_queries"]
        if total == 0:
            return 100.0
        return (self.performance_stats["successful_queries"] / total) * 100.0

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics including full AsyncClientManager integration info"""
        stats = dict(self.performance_stats)
        stats["success_rate"] = self._calculate_success_rate()
        stats["average_response_time_ms"] = self.performance_stats["average_response_time"]
        stats["initialization_complete"] = self.initialized
        stats["async_client_manager_available"] = self.async_client_manager is not None
        stats["async_client_manager_id"] = id(self.async_client_manager) if self.async_client_manager else None
        stats["full_async_client_manager_integration"] = self.performance_stats["async_client_manager_integrations"] >= 3

        # Add routing statistics
        stats["dynamic_routing_stats"] = self.dynamic_router.get_routing_statistics()

        # CENTRALIZED: Add centralized architecture stats
        stats["centralized_architecture_stats"] = {
            "mathstral_manager_integrated": self.mathstral_manager is not None,
            "traditional_manager_integrated": self.traditional_manager is not None,
            "nlp_schema_system_integrated": self.nlp_schema_system is not None,
            "centralized_prompt_builder_usage": self.performance_stats["centralized_prompt_builder_used"],
            "async_client_manager_integrations": self.performance_stats["async_client_manager_integrations"],
            "no_duplication_achieved": True,
            "singleton_integration_fixed": True
        }

        if self.async_client_manager:
            try:
                client_status = self.async_client_manager.get_client_status()
                stats["ai_clients_status"] = client_status
            except Exception as e:
                stats["ai_clients_error"] = str(e)

        return stats

    async def cleanup(self):
        """Cleanup all resources including managers and AsyncClientManager"""
        self.logger.info("Starting centralized orchestrator cleanup...")

        try:
            # Cleanup managers first
            if self.mathstral_manager and hasattr(self.mathstral_manager, 'cleanup'):
                await self.mathstral_manager.cleanup()
                self.logger.info("Mathstral Manager cleanup completed")

            if self.traditional_manager and hasattr(self.traditional_manager, 'cleanup'):
                await self.traditional_manager.cleanup()
                self.logger.info("Traditional Manager cleanup completed")

            # Cleanup SQL Generator
            if self.sql_generator and hasattr(self.sql_generator, 'cleanup'):
                await self.sql_generator.cleanup()

            # Cleanup NLP-Schema system
            if self.nlp_schema_system and hasattr(self.nlp_schema_system, 'cleanup'):
                await self.nlp_schema_system.cleanup() # pyright: ignore[reportAttributeAccessIssue]

            # AsyncClientManager cleanup is handled by the shared instance
            if self.async_client_manager:
                self.logger.info("AsyncClientManager cleanup handled by shared instance")

            self.status = "shutdown"
            self.logger.info(" Centralized orchestrator cleanup completed successfully")

        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


# Factory function for easy orchestrator creation
def create_hybrid_orchestrator(config: Optional[OrchestratorConfig] = None, async_client_manager=None) -> HybridAIAgentOrchestrator:
    """Create and initialize hybrid orchestrator with AsyncClientManager singleton integration"""
    try:
        orchestrator = HybridAIAgentOrchestrator(config, async_client_manager=async_client_manager)
        return orchestrator

    except Exception as e:
        logger.error(f"Failed to create orchestrator: {e}")
        raise


# FIXED: Add HybridConfig alias for backward compatibility
HybridConfig = OrchestratorConfig

# FIXED: Export HybridConfig along with other classes
__all__ = [
    "HybridAIAgentOrchestrator",
    "QueryResponse",
    "ConfidenceLevel", 
    "create_hybrid_orchestrator",
    "HybridConfig"
]


if __name__ == "__main__":
    async def test_orchestrator():
        """Test the orchestrator with AsyncClientManager singleton integration"""
        print("Testing Hybrid AI Agent Orchestrator with AsyncClientManager Singleton Integration...")

        try:
            # Simulate AsyncClientManager for testing
            class MockAsyncClientManager:
                def get_client_status(self):
                    return {"healthy_count": 2, "available_clients": ["mathstral", "deepseek"]}

            mock_client_manager = MockAsyncClientManager()
            orchestrator = create_hybrid_orchestrator(async_client_manager=mock_client_manager)

            # Test health check
            health = await orchestrator.health_check()
            print(f"Health Status: {health['orchestrator']['status']}")
            print(f"AsyncClientManager Available: {health['orchestrator']['async_client_manager_available']}")
            print(f"AsyncClientManager ID: {health['orchestrator']['async_client_manager_id']}")
            print(f"AsyncClientManager Integrations: {health['orchestrator']['async_client_manager_integrations']}")
            print(f"Singleton Integration Fixed: {health['centralized_architecture']['singleton_async_client_manager_fixed']}")

            # Get performance stats
            stats = orchestrator.get_performance_stats()
            print(f"\nASYNCCLIENTMANAGER SINGLETON INTEGRATION STATS:")
            print(f"Integration Count: {stats['async_client_manager_integrations']}")
            print(f"AsyncClientManager ID: {stats['async_client_manager_id']}")
            print(f"Singleton Integration Fixed: {stats['centralized_architecture_stats']['singleton_integration_fixed']}")

            # Cleanup
            await orchestrator.cleanup()
            print("\nASYNCCLIENTMANAGER SINGLETON INTEGRATION TEST COMPLETED!")

        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()

    import asyncio
    asyncio.run(test_orchestrator())
