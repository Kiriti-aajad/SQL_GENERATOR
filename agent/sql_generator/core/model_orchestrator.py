"""
Model Orchestrator - CORRECTED VERSION
Intelligent routing logic for multi-model SQL generation
FIXED: Updated for simplified MathstralClient and DeepSeekClient compatibility
FIXED: Removed any potential BaseModelClient dependencies
"""

import re
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# FIXED: Import only what we need, avoid potential BaseModelClient dependencies
try:
    from ..config.model_configs import QueryComplexity, GenerationStrategy
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # Define fallback enums if config not available
    class QueryComplexity(Enum):
        SIMPLE = "simple"
        MODERATE = "moderate" 
        COMPLEX = "complex"
        ANALYTICAL = "analytical"
    
    class GenerationStrategy(Enum):
        SINGLE_MODEL = "single_model"
        ENSEMBLE = "ensemble"

# FIXED: Define our own ModelType enum to avoid config dependencies
class ModelType(Enum):
    """Simplified model types matching our actual clients"""
    MATHSTRAL = "mathstral"
    DEEPSEEK = "deepseek"

class RoutingDecision(Enum):
    """Model routing decision types"""
    SINGLE_MATHSTRAL = "single_mathstral"
    SINGLE_DEEPSEEK = "single_deepseek"
    ENSEMBLE_BOTH = "ensemble_both"
    ENSEMBLE_MATHSTRAL_PRIMARY = "ensemble_mathstral_primary"
    ENSEMBLE_DEEPSEEK_PRIMARY = "ensemble_deepseek_primary"

@dataclass
class QueryAnalysis:
    """Results of query analysis"""
    query_text: str
    complexity: QueryComplexity
    query_type: str
    analytical_score: float
    code_generation_score: float
    mathematical_keywords: List[str]
    sql_operations: List[str]
    performance_requirements: str
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class ModelRoutingResult:
    """Model routing decision result"""
    primary_models: List[ModelType]
    generation_strategy: GenerationStrategy
    routing_decision: RoutingDecision
    confidence_threshold: float
    reasoning: str
    expected_performance: Dict[str, Any]
    metadata: Dict[str, Any]

class ModelOrchestrator:
    """Intelligent model routing orchestrator - SIMPLIFIED VERSION"""
    
    def __init__(self, model_configs=None):
        self.logger = logging.getLogger("ModelOrchestrator")
        
        # FIXED: Simplified configuration without complex dependencies
        self.ngrok_enabled = True  # Assume NGROK is available
        
        # Query analysis patterns (unchanged - these are good)
        self.analytical_patterns = [
            r'\b(correlation|statistical?|variance|std(dev)?|percentile)\b',
            r'\b(regression|trend|calculate|mathematical|financial)\b',
            r'\b(scientific|analytical|mean|median|deviation)\b',
            r'\b(coefficient|distribution|probability|ratio)\b'
        ]
        
        self.code_generation_patterns = [
            r'\b(generate|create|build|write|construct|develop)\b',
            r'\b(optimize|improve|refactor|performance|efficient)\b',
            r'\b(clean|readable|maintainable|scalable)\b'
        ]
        
        # XML-specific patterns for routing
        self.xml_patterns = [
            r'\b(xml|\.value\(|\.query\(|\.exist\(|\.nodes\()\b',
            r'\b(xpath|xml\s+data|xml\s+column)\b'
        ]
        
        # FIXED: Simplified complexity patterns with corrected regex
        self.complexity_patterns = {
            QueryComplexity.SIMPLE: [
                r'^\s*select\s+\*\s+from\s+\w+(\s+where\s+.+)?\s*;?\s*$',
                r'\b(insert\s+into|update\s+.+\s+set|delete\s+from)\b',
                r'\b(show|list|display)\s+'
            ],
            QueryComplexity.MODERATE: [
                r'\b(join|group\s+by|order\s+by|having)\b',
                r'\b(union|intersect|except)\b',
                r'\bsubquery\b|\(\s*select\b',
                r'\.value\s*\('  # XML operations
            ],
            QueryComplexity.COMPLEX: [
                r'\b(window\s+function|over\s*\(|row_number|rank|dense_rank)\b',
                r'\bwith\s+.+\s+as\s*\(|recursive\b',
                r'\b(pivot|unpivot|case\s+when)\b',
                r'multiple.*\.value\s*\('  # Multiple XML extractions
            ],
            QueryComplexity.ANALYTICAL: [
                r'\b(corr|covar|regr_|percentile_|stddev|variance)\b',
                r'\b(lag|lead|first_value|last_value|nth_value)\b',
                r'\b(sum|avg|count|min|max)\s*\(\s*distinct\b'
            ]
        }
        
        # Performance tracking
        self.routing_history: List[Dict[str, Any]] = []
        self.model_performance: Dict[ModelType, Dict[str, float]] = {}
        
        self.logger.info("ModelOrchestrator initialized with simplified configuration")

    def analyze_query(self, query_text: str, rich_context: str = "") -> QueryAnalysis:
        """Comprehensive query analysis for routing decisions with XML support"""
        
        # Basic preprocessing
        query_lower = query_text.lower().strip()
        
        # Determine complexity (including XML complexity)
        complexity = self._determine_complexity(query_text)
        
        # Analyze query type (including XML queries)
        query_type = self._classify_query_type(query_text)
        
        # Calculate analytical score
        analytical_score = self._calculate_analytical_score(query_text, rich_context)
        
        # Calculate code generation score (enhanced for XML)
        code_generation_score = self._calculate_code_generation_score(query_text)
        
        # Extract keywords and operations
        mathematical_keywords = self._extract_mathematical_keywords(query_text)
        sql_operations = self._extract_sql_operations(query_text)
        
        # Determine performance requirements
        performance_requirements = self._assess_performance_requirements(query_text)
        
        # Calculate overall analysis confidence
        confidence = self._calculate_analysis_confidence(
            query_text, complexity, analytical_score, code_generation_score
        )
        
        # Gather additional metadata (including XML detection)
        metadata = {
            "query_length": len(query_text),
            "word_count": len(query_text.split()),
            "has_rich_context": len(rich_context) > 100,
            "context_length": len(rich_context),
            "sql_keyword_count": len(sql_operations),
            "mathematical_keyword_count": len(mathematical_keywords),
            "has_xml_operations": self._has_xml_operations(query_text),
            "xml_complexity_score": self._calculate_xml_complexity(query_text),
            "ngrok_enabled": self.ngrok_enabled
        }
        
        return QueryAnalysis(
            query_text=query_text,
            complexity=complexity,
            query_type=query_type,
            analytical_score=analytical_score,
            code_generation_score=code_generation_score,
            mathematical_keywords=mathematical_keywords,
            sql_operations=sql_operations,
            performance_requirements=performance_requirements,
            confidence=confidence,
            metadata=metadata
        )

    def route_query(self, query_analysis: QueryAnalysis) -> ModelRoutingResult:
        """Make intelligent routing decision based on query analysis"""
        
        # Determine routing strategy
        routing_decision = self._determine_routing_strategy(query_analysis)
        
        # Get models and strategy based on routing decision
        primary_models, generation_strategy = self._get_models_for_routing(routing_decision)
        
        # Determine confidence threshold
        confidence_threshold = self._get_confidence_threshold(query_analysis, routing_decision)
        
        # Generate reasoning
        reasoning = self._generate_routing_reasoning(query_analysis, routing_decision)
        
        # Estimate expected performance
        expected_performance = self._estimate_performance(primary_models, query_analysis)
        
        # Track routing decision
        self._track_routing_decision(query_analysis, routing_decision)
        
        return ModelRoutingResult(
            primary_models=primary_models,
            generation_strategy=generation_strategy,
            routing_decision=routing_decision,
            confidence_threshold=confidence_threshold,
            reasoning=reasoning,
            expected_performance=expected_performance,
            metadata={
                "analysis_confidence": query_analysis.confidence,
                "routing_strategy": routing_decision.value,
                "analytical_weight": query_analysis.analytical_score,
                "code_generation_weight": query_analysis.code_generation_score,
                "complexity_factor": query_analysis.complexity.value,
                "ngrok_enabled": self.ngrok_enabled,
                "xml_operations_detected": query_analysis.metadata.get("has_xml_operations", False)
            }
        )

    # FIXED: All the helper methods remain the same but simplified
    def _has_xml_operations(self, query_text: str) -> bool:
        """Check if query contains XML operations"""
        query_lower = query_text.lower()
        return any(re.search(pattern, query_lower, re.IGNORECASE) for pattern in self.xml_patterns)

    def _calculate_xml_complexity(self, query_text: str) -> float:
        """Calculate XML operation complexity score"""
        if not self._has_xml_operations(query_text):
            return 0.0
        
        query_lower = query_text.lower()
        xml_score = 0.0
        
        # Count XML method calls
        xml_methods = ['.value(', '.query(', '.exist(', '.nodes(', '.modify(']
        for method in xml_methods:
            xml_score += query_lower.count(method) * 0.2
        
        # Count XPath expressions (simplified pattern)
        xpath_count = len(re.findall(r'["\'][\(\)/\w\[\]@\s:.-]+["\']', query_text))
        xml_score += min(xpath_count * 0.1, 0.3)
        
        return min(xml_score, 1.0)

    def _determine_complexity(self, query_text: str) -> QueryComplexity:
        """Determine query complexity using pattern matching with XML support"""
        query_lower = query_text.lower()
        
        # Check patterns in order of complexity (most complex first)
        for complexity in [QueryComplexity.ANALYTICAL, QueryComplexity.COMPLEX, 
                          QueryComplexity.MODERATE, QueryComplexity.SIMPLE]:
            
            patterns = self.complexity_patterns.get(complexity, [])
            for pattern in patterns:
                try:
                    if re.search(pattern, query_lower, re.IGNORECASE):
                        return complexity
                except re.error:
                    # Skip invalid regex patterns
                    continue
        
        return QueryComplexity.MODERATE

    def _classify_query_type(self, query_text: str) -> str:
        """Classify the type of SQL query including XML queries"""
        query_lower = query_text.lower().strip()
        
        # Check for XML operations first
        if self._has_xml_operations(query_text):
            if 'select' in query_lower:
                return "xml_select"
            else:
                return "xml_operation"
        
        # Standard SQL classification
        if query_lower.startswith('select'):
            if any(keyword in query_lower for keyword in ['join', 'group by', 'having']):
                return "complex_select"
            else:
                return "simple_select"
        elif query_lower.startswith(('insert', 'update', 'delete')):
            return query_lower.split()[0]
        elif query_lower.startswith(('create', 'alter', 'drop')):
            return query_lower.split()[0]
        elif any(keyword in query_lower for keyword in ['correlation', 'statistical', 'calculate']):
            return "analytical"
        elif not any(sql_keyword in query_lower for sql_keyword in ['select', 'insert', 'update', 'delete']):
            return "natural_language"
        
        return "unknown"

    def _calculate_analytical_score(self, query_text: str, rich_context: str = "") -> float:
        """Calculate how analytical/mathematical the query is (0.0 to 1.0)"""
        combined_text = f"{query_text} {rich_context}".lower()
        score = 0.0
        
        # Check analytical patterns
        for pattern in self.analytical_patterns:
            try:
                matches = len(re.findall(pattern, combined_text, re.IGNORECASE))
                score += matches * 0.2
            except re.error:
                continue
        
        # Check for mathematical functions
        mathematical_functions = [
            'sum', 'avg', 'count', 'min', 'max', 'stddev', 'variance',
            'corr', 'covar', 'percentile', 'median', 'mode'
        ]
        
        for func in mathematical_functions:
            if func in combined_text:
                score += 0.15
        
        return min(score, 1.0)

    def _calculate_code_generation_score(self, query_text: str) -> float:
        """Calculate how much the query focuses on code generation (0.0 to 1.0)"""
        query_lower = query_text.lower()
        score = 0.0
        
        # Check code generation patterns
        for pattern in self.code_generation_patterns:
            try:
                matches = len(re.findall(pattern, query_lower, re.IGNORECASE))
                score += matches * 0.25
            except re.error:
                continue
        
        # Boost for XML operations (DeepSeek handles these well)
        if self._has_xml_operations(query_text):
            score += 0.3
        
        return min(score, 1.0)

    def _extract_mathematical_keywords(self, query_text: str) -> List[str]:
        """Extract mathematical keywords from query"""
        mathematical_keywords = [
            'correlation', 'statistical', 'variance', 'stddev', 'percentile',
            'regression', 'trend', 'calculate', 'mathematical', 'financial',
            'scientific', 'analytical', 'mean', 'median', 'deviation'
        ]
        
        query_lower = query_text.lower()
        return [keyword for keyword in mathematical_keywords if keyword in query_lower]

    def _extract_sql_operations(self, query_text: str) -> List[str]:
        """Extract SQL operations from query including XML operations"""
        sql_operations = [
            'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP',
            'JOIN', 'UNION', 'GROUP BY', 'ORDER BY', 'HAVING', 'WHERE'
        ]
        
        query_upper = query_text.upper()
        found_operations = []
        
        for operation in sql_operations:
            if operation in query_upper:
                found_operations.append(operation)
        
        # Check XML operations
        if '.value(' in query_text:
            found_operations.append('XML_VALUE')
        if '.query(' in query_text:
            found_operations.append('XML_QUERY')
        
        return found_operations

    def _assess_performance_requirements(self, query_text: str) -> str:
        """Assess performance requirements of the query"""
        query_lower = query_text.lower()
        
        if any(indicator in query_lower for indicator in ['fast', 'quick', 'optimize', 'performance']):
            return "high"
        elif any(indicator in query_lower for indicator in ['complex', 'detailed', 'comprehensive']) or \
             self._calculate_xml_complexity(query_text) > 0.5:
            return "low"
        
        return "medium"

    def _calculate_analysis_confidence(self, query_text: str, complexity: QueryComplexity,
                                     analytical_score: float, code_generation_score: float) -> float:
        """Calculate confidence in the query analysis"""
        base_confidence = 0.7
        
        # Boost confidence for clear patterns
        if analytical_score > 0.6 or code_generation_score > 0.6:
            base_confidence += 0.2
        
        # Boost confidence for well-structured queries
        if any(keyword in query_text.upper() for keyword in ['SELECT', 'FROM', 'WHERE']):
            base_confidence += 0.1
        
        # Reduce confidence for very short queries
        if len(query_text.strip()) < 10:
            base_confidence -= 0.3
        
        return min(max(base_confidence, 0.0), 1.0)

    def _determine_routing_strategy(self, analysis: QueryAnalysis) -> RoutingDecision:
        """Determine the routing strategy based on query analysis"""
        has_xml = analysis.metadata.get("has_xml_operations", False)
        
        # XML queries generally favor DeepSeek (better code generation)
        if analysis.analytical_score > 0.7 and not has_xml:
            return RoutingDecision.SINGLE_MATHSTRAL if analysis.complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE] else RoutingDecision.ENSEMBLE_MATHSTRAL_PRIMARY
        
        # High code generation score OR XML operations -> Favor DeepSeek  
        elif analysis.code_generation_score > 0.7 or has_xml:
            return RoutingDecision.SINGLE_DEEPSEEK if analysis.complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE] else RoutingDecision.ENSEMBLE_DEEPSEEK_PRIMARY
        
        # Complex queries -> Use ensemble
        elif analysis.complexity in [QueryComplexity.COMPLEX, QueryComplexity.ANALYTICAL]:
            return RoutingDecision.ENSEMBLE_BOTH
        
        # Default to DeepSeek for general queries
        else:
            return RoutingDecision.SINGLE_DEEPSEEK

    def _get_models_for_routing(self, routing_decision: RoutingDecision) -> Tuple[List[ModelType], GenerationStrategy]:
        """Get models and strategy based on routing decision"""
        routing_map = {
            RoutingDecision.SINGLE_MATHSTRAL: (
                [ModelType.MATHSTRAL], 
                GenerationStrategy.SINGLE_MODEL
            ),
            RoutingDecision.SINGLE_DEEPSEEK: (
                [ModelType.DEEPSEEK], 
                GenerationStrategy.SINGLE_MODEL
            ),
            RoutingDecision.ENSEMBLE_BOTH: (
                [ModelType.MATHSTRAL, ModelType.DEEPSEEK], 
                GenerationStrategy.ENSEMBLE
            ),
            RoutingDecision.ENSEMBLE_MATHSTRAL_PRIMARY: (
                [ModelType.MATHSTRAL, ModelType.DEEPSEEK], 
                GenerationStrategy.ENSEMBLE
            ),
            RoutingDecision.ENSEMBLE_DEEPSEEK_PRIMARY: (
                [ModelType.DEEPSEEK, ModelType.MATHSTRAL], 
                GenerationStrategy.ENSEMBLE
            )
        }
        
        return routing_map.get(routing_decision, ([ModelType.DEEPSEEK], GenerationStrategy.SINGLE_MODEL))

    def _get_confidence_threshold(self, analysis: QueryAnalysis, routing_decision: RoutingDecision) -> float:
        """Determine confidence threshold based on query and routing"""
        base_threshold = 0.7
        
        # Lower threshold for ensemble approaches
        if routing_decision in [RoutingDecision.ENSEMBLE_BOTH, 
                              RoutingDecision.ENSEMBLE_MATHSTRAL_PRIMARY,
                              RoutingDecision.ENSEMBLE_DEEPSEEK_PRIMARY]:
            base_threshold = 0.6
        
        # Adjust based on query confidence
        if analysis.confidence < 0.5:
            base_threshold = max(base_threshold - 0.1, 0.4)
        elif analysis.confidence > 0.9:
            base_threshold = min(base_threshold + 0.1, 0.9)
        
        return base_threshold

    def _generate_routing_reasoning(self, analysis: QueryAnalysis, routing_decision: RoutingDecision) -> str:
        """Generate human-readable reasoning for the routing decision"""
        has_xml = analysis.metadata.get("has_xml_operations", False)
        
        reasoning_templates = {
            RoutingDecision.SINGLE_MATHSTRAL: f"High analytical score ({analysis.analytical_score:.2f}) favors Mathstral",
            RoutingDecision.SINGLE_DEEPSEEK: f"General SQL query suits DeepSeek",
            RoutingDecision.ENSEMBLE_BOTH: f"Complex {analysis.complexity.value} query benefits from ensemble",
            RoutingDecision.ENSEMBLE_MATHSTRAL_PRIMARY: f"Analytical query with ensemble validation",
            RoutingDecision.ENSEMBLE_DEEPSEEK_PRIMARY: f"Code-focused query with ensemble quality check"
        }
        
        base_reason = reasoning_templates.get(routing_decision, "Standard routing based on analysis")
        
        if has_xml:
            base_reason += " (XML operations detected - DeepSeek specialization)"
        
        return base_reason

    def _estimate_performance(self, models: List[ModelType], analysis: QueryAnalysis) -> Dict[str, Any]:
        """Estimate expected performance for the routing decision"""
        # Base performance estimates (in milliseconds) - adjusted for NGROK
        model_speeds = {
            ModelType.MATHSTRAL: 2500,
            ModelType.DEEPSEEK: 2000
        }
        
        # Estimate response time
        if len(models) == 1:
            estimated_time = model_speeds.get(models[0], 2200)
        else:
            max_time = max(model_speeds.get(model, 2200) for model in models)
            estimated_time = max_time + 500  # Ensemble overhead
        
        # Add XML processing overhead
        if analysis.metadata.get("has_xml_operations", False):
            xml_complexity = analysis.metadata.get("xml_complexity_score", 0)
            estimated_time += int(xml_complexity * 300)
        
        return {
            "estimated_response_time_ms": estimated_time,
            "accuracy_estimate": 0.85,
            "model_count": len(models),
            "complexity_factor": analysis.complexity.value,
            "confidence_in_routing": analysis.confidence
        }

    def _track_routing_decision(self, analysis: QueryAnalysis, routing_decision: RoutingDecision):
        """Track routing decision for performance analysis"""
        routing_record = {
            "timestamp": time.time(),
            "query_complexity": analysis.complexity.value,
            "query_type": analysis.query_type,
            "routing_decision": routing_decision.value,
            "confidence": analysis.confidence,
            "has_xml_operations": analysis.metadata.get("has_xml_operations", False)
        }
        
        self.routing_history.append(routing_record)
        
        # Keep only last 1000 records
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]

    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get analytics on routing decisions"""
        if not self.routing_history:
            return {"message": "No routing history available"}
        
        total_routes = len(self.routing_history)
        
        # Basic analytics
        decision_stats = {}
        for record in self.routing_history:
            decision = record["routing_decision"]
            decision_stats[decision] = decision_stats.get(decision, 0) + 1
        
        return {
            "total_routing_decisions": total_routes,
            "routing_distribution": {
                decision: (count / total_routes) * 100 
                for decision, count in decision_stats.items()
            }
        }

# Export the main class
__all__ = ["ModelOrchestrator", "QueryAnalysis", "ModelRoutingResult", "RoutingDecision"]
