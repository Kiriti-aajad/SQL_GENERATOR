"""
Intelligent Orchestrator for Advanced Query Routing and Processing
Provides intelligent routing capabilities for the NLP-Schema integration system
Renamed from IntelligentAgent to IntelligentOrchestrator for server compatibility
Author: KIRITI AAJAD (Enhanced for Orchestrator Integration)
Version: 2.0.0 - ORCHESTRATOR COMPATIBILITY FIX
Date: 2025-08-06
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union
import json
import sys
import os
from dataclasses import dataclass
from enum import Enum

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agent.nlp_processor.utils.metadata_loader import get_metadata_loader
    METADATA_LOADER_AVAILABLE = True
except ImportError:
    METADATA_LOADER_AVAILABLE = False
    get_metadata_loader = None

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    """Routing strategies for intelligent orchestrator"""
    COMPLEXITY_BASED = "complexity_based"
    INTENT_BASED = "intent_based"
    HYBRID_ROUTING = "hybrid_routing"
    PERFORMANCE_OPTIMIZED = "performance_optimized"

@dataclass
class IntelligentConfig:
    """Configuration for Intelligent Orchestrator"""
    # Routing configuration
    enable_intelligent_routing: bool = True
    routing_strategy: RoutingStrategy = RoutingStrategy.HYBRID_ROUTING
    routing_timeout: float = 5.0
    
    # Complexity thresholds
    complexity_threshold_mathstral: float = 0.6
    complexity_threshold_traditional: float = 0.4
    
    # Intent classification
    enable_intent_classification: bool = True
    intent_confidence_threshold: float = 0.7
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 300
    max_concurrent_routes: int = 10
    
    # Quality thresholds
    min_confidence_threshold: float = 0.5
    enable_fallback_routing: bool = True
    
    # Logging and monitoring
    enable_detailed_logging: bool = True
    enable_performance_monitoring: bool = True
    log_level: str = "INFO"

class IntelligentOrchestrator:
    """Intelligent Orchestrator for advanced query routing and processing"""

    def __init__(self, config: Optional[IntelligentConfig] = None):
        """Initialize the intelligent orchestrator with enhanced routing capabilities"""
        try:
            # Add status attribute that validation system expects
            self.status = "initializing"
            
            self.config = config or IntelligentConfig()
            self.metadata_loader = get_metadata_loader() if METADATA_LOADER_AVAILABLE else None # pyright: ignore[reportOptionalCall]
            self.schema_cache = {}
            self.routing_cache = {}
            self.performance_metrics = {
                "total_routes": 0,
                "successful_routes": 0,
                "routing_times": [],
                "complexity_assessments": []
            }
            self.initialization_time = time.time()
            
            # Configure logging
            if self.config.log_level == "DEBUG":
                logger.setLevel(logging.DEBUG)
            elif self.config.log_level == "ERROR":
                logger.setLevel(logging.ERROR)
            else:
                logger.setLevel(logging.INFO)
            
            # Set status to healthy after successful initialization
            self.status = "healthy"
            logger.info("IntelligentOrchestrator initialized successfully")
            
        except Exception as e:
            # Set error status if initialization fails
            self.status = "error"
            logger.error(f"IntelligentOrchestrator initialization failed: {e}")
            raise

    # Status management methods
    def set_status(self, status: str):
        """Set the orchestrator status"""
        self.status = status
        logger.debug(f"Orchestrator status updated to: {status}")

    def get_status(self) -> str:
        """Get the current orchestrator status"""
        return self.status

    def is_healthy(self) -> bool:
        """Check if orchestrator is in a healthy state"""
        return self.status in ["healthy", "ready"]

    async def route_query(
        self,
        user_query: str,
        context: Dict[str, Any],
        request_id: str
    ) -> Dict[str, Any]:
        """Route query based on intelligent analysis of complexity and intent"""
        start_time = time.time()
        
        try:
            logger.info(f"[{request_id}] Starting intelligent query routing")
            logger.debug(f"[{request_id}] Query: {user_query[:100]}...")
            
            # Check cache first
            if self.config.enable_caching:
                cache_key = self._generate_cache_key(user_query, context)
                cached_result = self.routing_cache.get(cache_key)
                if cached_result:
                    logger.info(f"[{request_id}] Using cached routing decision")
                    return cached_result
            
            # Perform intelligent routing analysis
            routing_analysis = await self._analyze_query_for_routing(user_query, context, request_id)
            
            # Make routing decision
            routing_decision = self._make_routing_decision(routing_analysis, request_id)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.performance_metrics["total_routes"] += 1
            self.performance_metrics["routing_times"].append(processing_time)
            
            if routing_decision["success"]:
                self.performance_metrics["successful_routes"] += 1
            
            # Cache the result
            if self.config.enable_caching and routing_decision["success"]:
                self.routing_cache[cache_key] = routing_decision # pyright: ignore[reportPossiblyUnboundVariable]
            
            logger.info(f"[{request_id}] Routing completed: {routing_decision['manager_used']} "
                       f"(confidence: {routing_decision['confidence']:.2f})")
            
            return routing_decision
            
        except Exception as e:
            logger.error(f"[{request_id}] Routing failed: {e}")
            # Return fallback routing decision
            return self._get_fallback_routing_decision(user_query, str(e), request_id)

    async def _analyze_query_for_routing(
        self,
        user_query: str,
        context: Dict[str, Any],
        request_id: str
    ) -> Dict[str, Any]:
        """Analyze query to determine optimal routing strategy"""
        
        analysis = {
            "complexity_score": 0.0,
            "intent_classification": {},
            "entity_analysis": {},
            "schema_requirements": {},
            "performance_prediction": {}
        }
        
        try:
            # Complexity analysis
            analysis["complexity_score"] = self._assess_query_complexity(user_query)
            
            # Intent classification
            if self.config.enable_intent_classification:
                analysis["intent_classification"] = self._classify_query_intent(user_query, context)
            
            # Entity analysis
            analysis["entity_analysis"] = self._analyze_entities(user_query, context)
            
            # Schema requirements assessment
            analysis["schema_requirements"] = await self._assess_schema_requirements(user_query, context)
            
            # Performance prediction
            analysis["performance_prediction"] = self._predict_performance(analysis)
            
            logger.debug(f"[{request_id}] Analysis complete: complexity={analysis['complexity_score']:.2f}")
            
        except Exception as e:
            logger.warning(f"[{request_id}] Analysis failed: {e}, using basic assessment")
            analysis["complexity_score"] = 0.5  # Default medium complexity
        
        return analysis

    def _assess_query_complexity(self, query: str) -> float:
        """Assess query complexity based on linguistic and structural features"""
        complexity_score = 0.0
        query_lower = query.lower()
        
        # Length-based complexity
        word_count = len(query.split())
        if word_count > 20:
            complexity_score += 0.3
        elif word_count > 10:
            complexity_score += 0.1
        
        # SQL complexity indicators
        complex_keywords = [
            'join', 'union', 'group by', 'having', 'window', 'partition',
            'case when', 'subquery', 'exists', 'in (select', 'with'
        ]
        for keyword in complex_keywords:
            if keyword in query_lower:
                complexity_score += 0.2
        
        # Aggregation complexity
        aggregation_keywords = [
            'sum', 'count', 'avg', 'max', 'min', 'group', 'aggregate'
        ]
        agg_count = sum(1 for keyword in aggregation_keywords if keyword in query_lower)
        complexity_score += min(agg_count * 0.15, 0.3)
        
        # Temporal complexity
        temporal_keywords = [
            'between', 'date', 'time', 'year', 'month', 'day',
            'last', 'previous', 'current', 'range'
        ]
        temporal_count = sum(1 for keyword in temporal_keywords if keyword in query_lower)
        complexity_score += min(temporal_count * 0.1, 0.2)
        
        # Multiple table indicators
        table_indicators = ['from', 'join', 'table', 'customer', 'account', 'transaction']
        table_count = sum(1 for indicator in table_indicators if indicator in query_lower)
        if table_count > 3:
            complexity_score += 0.2
        
        # Normalize score to [0, 1]
        return min(complexity_score, 1.0)

    def _classify_query_intent(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Classify query intent for routing decisions"""
        query_lower = query.lower()
        
        # Define intent patterns
        intent_patterns = {
            'complex_analysis': [
                'analysis', 'compare', 'trend', 'pattern', 'correlation',
                'statistical', 'aggregate', 'summarize', 'breakdown'
            ],
            'simple_lookup': [
                'show', 'find', 'get', 'list', 'display', 'what is',
                'who is', 'where is', 'details', 'information'
            ],
            'aggregation': [
                'total', 'sum', 'count', 'average', 'maximum', 'minimum',
                'group by', 'sum of', 'number of'
            ],
            'reporting': [
                'report', 'dashboard', 'summary', 'overview', 'statistics',
                'metrics', 'kpi', 'performance'
            ],
            'temporal_analysis': [
                'history', 'historical', 'over time', 'trend', 'change',
                'growth', 'decline', 'period', 'timeline'
            ]
        }
        
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            intent_scores[intent] = score
        
        # Determine primary intent
        primary_intent = max(intent_scores, key=intent_scores.get) if intent_scores else 'simple_lookup' # pyright: ignore[reportCallIssue, reportArgumentType]
        confidence = intent_scores[primary_intent] / max(sum(intent_scores.values()), 1)
        
        return {
            'primary': primary_intent,
            'confidence': confidence,
            'all_scores': intent_scores
        }

    def _analyze_entities(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze entities in the query for routing context"""
        query_lower = query.lower()
        
        # Entity patterns
        entity_patterns = {
            'financial_amounts': ['crore', 'lakh', 'million', 'billion', 'amount', 'value'],
            'locations': ['mumbai', 'delhi', 'bangalore', 'chennai', 'region', 'state'],
            'customers': ['customer', 'client', 'counterparty', 'ctpt', 'account holder'],
            'time_periods': ['year', 'month', 'quarter', 'week', 'day', 'period'],
            'business_metrics': ['revenue', 'profit', 'loss', 'margin', 'roi', 'turnover']
        }
        
        detected_entities = {}
        for entity_type, patterns in entity_patterns.items():
            matches = [pattern for pattern in patterns if pattern in query_lower]
            if matches:
                detected_entities[entity_type] = {
                    'matches': matches,
                    'count': len(matches)
                }
        
        return {
            'detected_entities': detected_entities,
            'entity_complexity': len(detected_entities),
            'total_matches': sum(entity['count'] for entity in detected_entities.values())
        }

    async def _assess_schema_requirements(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess schema requirements for the query"""
        try:
            if self.metadata_loader:
                # Get basic schema info
                all_metadata = self.metadata_loader.load_all_metadata()
                tables = list(all_metadata.get("tables", {}).keys())
                
                # Estimate required tables
                query_lower = query.lower()
                relevant_tables = []
                for table in tables:
                    if any(keyword in table.lower() for keyword in query_lower.split()):
                        relevant_tables.append(table)
                
                return {
                    'estimated_tables': len(relevant_tables),
                    'relevant_tables': relevant_tables[:10],  # Top 10
                    'total_available_tables': len(tables),
                    'schema_complexity': 'high' if len(relevant_tables) > 5 else 'medium' if len(relevant_tables) > 2 else 'low'
                }
            else:
                return {
                    'estimated_tables': 0,
                    'relevant_tables': [],
                    'total_available_tables': 0,
                    'schema_complexity': 'unknown',
                    'metadata_unavailable': True
                }
                
        except Exception as e:
            logger.warning(f"Schema requirements assessment failed: {e}")
            return {'error': str(e), 'schema_complexity': 'unknown'}

    def _predict_performance(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Predict performance characteristics for routing decision"""
        complexity = analysis.get('complexity_score', 0.5)
        schema_reqs = analysis.get('schema_requirements', {})
        entity_analysis = analysis.get('entity_analysis', {})
        
        # Estimate processing time
        estimated_time = 5.0  # Base time
        if complexity > 0.7:
            estimated_time += 10.0
        elif complexity > 0.4:
            estimated_time += 5.0
        
        # Adjust for schema complexity
        table_count = schema_reqs.get('estimated_tables', 0)
        estimated_time += table_count * 0.5
        
        # Adjust for entity complexity
        entity_complexity = entity_analysis.get('entity_complexity', 0)
        estimated_time += entity_complexity * 1.0
        
        return {
            'estimated_processing_time': estimated_time,
            'memory_requirements': 'high' if complexity > 0.7 else 'medium',
            'cpu_intensity': 'high' if complexity > 0.6 else 'low',
            'io_requirements': 'high' if table_count > 5 else 'medium'
        }

    def _make_routing_decision(self, analysis: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Make final routing decision based on analysis"""
        complexity_score = analysis.get('complexity_score', 0.5)
        intent = analysis.get('intent_classification', {}).get('primary', 'simple_lookup')
        
        # Determine manager based on complexity and intent
        if complexity_score >= self.config.complexity_threshold_mathstral:
            manager_used = "mathstral"
            reasoning = f"High complexity score ({complexity_score:.2f}) indicates complex query requiring advanced processing"
        elif intent in ['complex_analysis', 'aggregation', 'reporting', 'temporal_analysis']:
            manager_used = "mathstral"
            reasoning = f"Intent '{intent}' requires advanced analytical capabilities"
        else:
            manager_used = "traditional"
            reasoning = f"Low complexity ({complexity_score:.2f}) and simple intent '{intent}' suitable for traditional processing"
        
        # Calculate confidence
        confidence = 0.8  # Base confidence
        if complexity_score > 0.8 or complexity_score < 0.2:
            confidence = 0.9  # High confidence for very clear cases
        
        # Check for fallback conditions
        performance_pred = analysis.get('performance_prediction', {})
        estimated_time = performance_pred.get('estimated_processing_time', 5.0)
        
        if estimated_time > 30.0 and manager_used == "mathstral":
            # Consider fallback for very long queries
            if self.config.enable_fallback_routing:
                manager_used = "hybrid"
                reasoning += " (with hybrid fallback due to high processing time estimate)"
        
        routing_decision = {
            "success": True,
            "manager_used": manager_used,
            "confidence": confidence,
            "reasoning": reasoning,
            "routing_metadata": {
                "complexity_score": complexity_score,
                "intent": intent,
                "estimated_processing_time": estimated_time,
                "analysis_summary": {
                    "complexity": complexity_score,
                    "intent_confidence": analysis.get('intent_classification', {}).get('confidence', 0.0),
                    "entity_complexity": analysis.get('entity_analysis', {}).get('entity_complexity', 0),
                    "schema_complexity": analysis.get('schema_requirements', {}).get('schema_complexity', 'unknown')
                }
            },
            "direct_result": False,  # Indicates this is a routing decision, not a direct result
            "request_id": request_id
        }
        
        return routing_decision

    def _get_fallback_routing_decision(self, query: str, error: str, request_id: str) -> Dict[str, Any]:
        """Get fallback routing decision when analysis fails"""
        # Simple fallback: route to traditional for short queries, mathstral for long ones
        word_count = len(query.split())
        manager_used = "mathstral" if word_count > 15 else "traditional"
        
        return {
            "success": False,
            "manager_used": manager_used,
            "confidence": 0.3,  # Low confidence for fallback
            "reasoning": f"Fallback routing due to analysis error: {error}",
            "routing_metadata": {
                "fallback_used": True,
                "error": error,
                "word_count": word_count
            },
            "direct_result": False,
            "request_id": request_id,
            "warnings": [f"Routing analysis failed: {error}"]
        }

    def _generate_cache_key(self, query: str, context: Dict[str, Any]) -> str:
        """Generate cache key for routing decisions"""
        import hashlib
        content = f"{query}_{json.dumps(context, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check method required by main.py"""
        try:
            uptime = time.time() - self.initialization_time
            
            # Check metadata loader availability
            metadata_status = "available" if self.metadata_loader else "unavailable"
            
            # Calculate success rate
            total_routes = self.performance_metrics["total_routes"]
            successful_routes = self.performance_metrics["successful_routes"]
            success_rate = (successful_routes / total_routes) if total_routes > 0 else 1.0
            
            # Calculate average routing time
            routing_times = self.performance_metrics["routing_times"]
            avg_routing_time = sum(routing_times) / len(routing_times) if routing_times else 0.0
            
            # Determine overall health
            is_healthy = (
                success_rate >= self.config.min_confidence_threshold and
                avg_routing_time < self.config.routing_timeout and
                uptime > 0
            )
            
            # Update internal status based on health check results
            if is_healthy:
                self.status = "healthy"
            else:
                self.status = "degraded"
            
            health_response = {
                "status": self.status,
                "component": "IntelligentOrchestrator",
                "uptime_seconds": uptime,
                "routing_enabled": self.config.enable_intelligent_routing,
                "intent_classification_enabled": self.config.enable_intent_classification,
                
                # Performance metrics
                "performance_metrics": {
                    "total_routes": total_routes,
                    "successful_routes": successful_routes,
                    "success_rate": success_rate,
                    "average_routing_time": avg_routing_time,
                    "cache_size": len(self.routing_cache)
                },
                
                # Component status
                "component_status": {
                    "metadata_loader": metadata_status,
                    "routing_cache": "enabled" if self.config.enable_caching else "disabled",
                    "fallback_routing": "enabled" if self.config.enable_fallback_routing else "disabled"
                },
                
                # Configuration info
                "configuration": {
                    "routing_strategy": self.config.routing_strategy.value,
                    "complexity_threshold_mathstral": self.config.complexity_threshold_mathstral,
                    "complexity_threshold_traditional": self.config.complexity_threshold_traditional,
                    "routing_timeout": self.config.routing_timeout
                }
            }
            
            return health_response
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            # Update status on error
            self.status = "error"
            return {
                "status": self.status,
                "component": "IntelligentOrchestrator",
                "error": str(e),
                "routing_enabled": False
            }

    # Additional methods for compatibility and functionality...
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        routing_times = self.performance_metrics["routing_times"]
        
        return {
            "total_routes": self.performance_metrics["total_routes"],
            "successful_routes": self.performance_metrics["successful_routes"],
            "success_rate": (
                self.performance_metrics["successful_routes"] / 
                max(self.performance_metrics["total_routes"], 1)
            ),
            "average_routing_time": sum(routing_times) / max(len(routing_times), 1),
            "min_routing_time": min(routing_times) if routing_times else 0,
            "max_routing_time": max(routing_times) if routing_times else 0,
            "cache_hit_ratio": len(self.routing_cache) / max(self.performance_metrics["total_routes"], 1),
            "uptime_seconds": time.time() - self.initialization_time
        }

# Export the correct classes for main.py
__all__ = ['IntelligentOrchestrator', 'IntelligentConfig']

# Factory function for creating instances
def create_intelligent_orchestrator(config: Optional[IntelligentConfig] = None) -> IntelligentOrchestrator:
    """Factory function to create IntelligentOrchestrator instances"""
    try:
        orchestrator = IntelligentOrchestrator(config)
        if orchestrator.status != "healthy":
            logger.warning(f"Created orchestrator with status: {orchestrator.status}")
        return orchestrator
    except Exception as e:
        logger.error(f"Failed to create intelligent orchestrator: {e}")
        raise
