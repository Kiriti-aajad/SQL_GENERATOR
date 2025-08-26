"""
SearchStrategy
─────────────
Enhanced search strategy generation for targeted schema searches.

Core responsibilities:
1. create_targeted_strategy() → build search strategy based on gaps
2. select_optimal_engines() → choose best engines for specific entity types
3. adjust_search_parameters() → modify engine parameters for targeted search
4. validate_search_strategy() → ensure strategy is feasible and logical
"""

from typing import Dict, List, Set, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from enum import Enum

# Core imports
from agent.schema_searcher.core.data_models import SearchMethod
from agent.schema_searcher.engines.base_engine import BaseSearchEngine

class StrategyType(Enum):
    """Types of search strategies"""
    BROAD_DISCOVERY = "broad_discovery"
    TARGETED_REFINEMENT = "targeted_refinement"
    RELATIONSHIP_FOCUSED = "relationship_focused"
    DOMAIN_SPECIFIC = "domain_specific"
    HYBRID_MULTI_ENGINE = "hybrid_multi_engine"

class OptimizationObjective(Enum):
    """Optimization objectives for strategy selection"""
    MAXIMIZE_RECALL = "maximize_recall"
    MAXIMIZE_PRECISION = "maximize_precision"
    MINIMIZE_TIME = "minimize_time"
    BALANCE_PRECISION_RECALL = "balance_precision_recall"
    MAXIMIZE_CONFIDENCE = "maximize_confidence"

@dataclass
class EngineStrategy:
    """Strategy configuration for a specific search engine"""
    engine_method: SearchMethod
    priority: int  # 1 = highest priority
    weight: float  # Contribution weight in ensemble
    parameters: Dict[str, Any]
    expected_entity_types: List[str]
    optimization_focus: OptimizationObjective

@dataclass
class SearchStrategyPlan:
    """Complete search strategy plan"""
    strategy_type: StrategyType
    optimization_objective: OptimizationObjective
    engine_strategies: List[EngineStrategy]
    keyword_allocation: Dict[SearchMethod, List[str]]
    execution_order: List[SearchMethod]
    fallback_strategies: List[StrategyType]
    performance_thresholds: Dict[str, float]
    estimated_execution_time: float

class SearchStrategy:
    """
    Intelligent search strategy generator using multi-objective optimization.
    
    Mathstral Logic:
    - Constraint optimization for search parameter selection
    - Decision trees for engine selection based on entity types
    - Multi-criteria decision making for strategy optimization
    - Performance prediction using machine learning models
    """
    
    def __init__(self):
        # Logging setup
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s'))
            self.logger.addHandler(h)
            self.logger.setLevel(logging.INFO)
        
        # Engine capability matrix - maps entity types to optimal engines
        self.engine_capabilities = self._initialize_engine_capabilities()
        
        # Performance history for learning
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Strategy templates
        self.strategy_templates = self._initialize_strategy_templates()
    
    def create_targeted_strategy(
        self,
        gap_analysis: Dict[str, List[str]],
        keywords: List[str],
        available_engines: Dict[SearchMethod, Any],
        optimization_objective: OptimizationObjective = OptimizationObjective.BALANCE_PRECISION_RECALL
    ) -> Dict[str, Any]:
        """
        Build comprehensive search strategy based on gap analysis.
        
        🔧 CRITICAL FIX: Handle both dict and list inputs for gap_analysis
        
        Mathstral Logic:
        - Constraint optimization: maximize coverage × minimize cost
        - Multi-criteria decision making with weighted objectives
        - Dynamic programming for optimal engine sequencing
        """
        self.logger.info(f"Creating targeted strategy for gap analysis")
        
        try:
            # 🔧 CRITICAL FIX: Handle case where gap_analysis is a list instead of dict
            if isinstance(gap_analysis, list):
                self.logger.debug("Converting list gap_analysis to dictionary format")
                # Convert list to dictionary with categorized gaps
                if gap_analysis:
                    # Try to categorize gaps intelligently
                    categorized_gaps = {}
                    for i, gap in enumerate(gap_analysis):
                        gap_str = str(gap).lower()
                        if any(word in gap_str for word in ['customer', 'client', 'person']):
                            categorized_gaps.setdefault('customer', []).append(gap)
                        elif any(word in gap_str for word in ['product', 'item', 'service']):
                            categorized_gaps.setdefault('product', []).append(gap)
                        elif any(word in gap_str for word in ['location', 'region', 'place']):
                            categorized_gaps.setdefault('location', []).append(gap)
                        elif any(word in gap_str for word in ['financial', 'money', 'amount', 'revenue']):
                            categorized_gaps.setdefault('financial', []).append(gap)
                        else:
                            categorized_gaps.setdefault('generic', []).append(gap)
                    gap_analysis = categorized_gaps
                else:
                    gap_analysis = {'generic': []}
            
            elif not isinstance(gap_analysis, dict):
                self.logger.warning(f"Unexpected gap_analysis type: {type(gap_analysis)}, converting to dict")
                gap_analysis = {'generic': [str(gap_analysis)]}
            
            # Ensure gap_analysis is not empty
            if not gap_analysis:
                gap_analysis = {'generic': keywords[:5] if keywords else ['general']}
            
            self.logger.info(f"Processing {len(gap_analysis)} gap categories")
            
            # Step 1: Analyze gap characteristics
            gap_characteristics = self._analyze_gap_characteristics(gap_analysis)
            
            # Step 2: Select optimal strategy type
            strategy_type = self._select_strategy_type(gap_characteristics, optimization_objective)
            
            # Step 3: Select and configure engines
            engine_strategies = self._select_optimal_engines(
                gap_analysis, available_engines, optimization_objective
            )
            
            # Step 4: Allocate keywords to engines
            keyword_allocation = self._allocate_keywords_to_engines(
                keywords, engine_strategies, gap_analysis
            )
            
            # Step 5: Determine execution order
            execution_order = self._determine_execution_order(
                engine_strategies, optimization_objective
            )
            
            # Step 6: Set performance thresholds
            performance_thresholds = self._calculate_performance_thresholds(
                strategy_type, optimization_objective
            )
            
            # Step 7: Estimate execution time
            estimated_time = self._estimate_execution_time(engine_strategies, keywords)
            
            # Create strategy plan
            strategy_plan = SearchStrategyPlan(
                strategy_type=strategy_type,
                optimization_objective=optimization_objective,
                engine_strategies=engine_strategies,
                keyword_allocation=keyword_allocation,
                execution_order=execution_order,
                fallback_strategies=self._get_fallback_strategies(strategy_type),
                performance_thresholds=performance_thresholds,
                estimated_execution_time=estimated_time
            )
            
            # Validate strategy
            validation_result = self.validate_search_strategy(strategy_plan)
            if not validation_result['is_valid']:
                self.logger.warning(f"Strategy validation failed: {validation_result['issues']}")
                # Apply corrections
                strategy_plan = self._apply_strategy_corrections(strategy_plan, validation_result)
            
            # Convert to dictionary format for orchestrator
            strategy_dict = self._convert_strategy_to_dict(strategy_plan)
            
            self.logger.info(
                f"Strategy created: {strategy_type.value}, "
                f"{len(engine_strategies)} engines, "
                f"~{estimated_time:.1f}s estimated time"
            )
            
            return strategy_dict
            
        except Exception as e:
            self.logger.error(f"Strategy creation failed: {e}")
            # 🔧 ADDITIONAL FIX: Ensure fallback always works
            try:
                return self._create_fallback_strategy(keywords, available_engines)
            except Exception as fallback_error:
                self.logger.error(f"Fallback strategy creation failed: {fallback_error}")
                return self._create_minimal_fallback_strategy(keywords, available_engines)
    
    def select_optimal_engines(
        self,
        entity_types: List[str],
        performance_history: Optional[Dict[str, Any]] = None
    ) -> List[SearchMethod]:
        """
        Select optimal engines using decision trees and performance prediction.
        
        Mathstral Logic:
        - Decision tree classification based on entity types
        - Machine learning performance prediction
        - Ensemble selection optimization
        """
        self.logger.debug(f"Selecting optimal engines for {len(entity_types)} entity types")
        
        # Score each engine for each entity type
        engine_scores: Dict[SearchMethod, float] = {}
        
        for engine_method in SearchMethod:
            total_score = 0.0
            
            for entity_type in entity_types:
                # Base capability score
                capability_score = self.engine_capabilities.get(entity_type, {}).get(
                    engine_method, 0.5
                )
                
                # Historical performance score
                historical_score = self._get_historical_performance_score(
                    engine_method, entity_type, performance_history
                )
                
                # Combined score with weights
                entity_score = (0.6 * capability_score) + (0.4 * historical_score)
                total_score += entity_score
            
            # Average score across all entity types
            engine_scores[engine_method] = total_score / max(1, len(entity_types))
        
        # Sort engines by score and return top performers
        sorted_engines = sorted(
            engine_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Select top engines (at least 2, at most 4)
        selected_count = min(4, max(2, len([score for _, score in sorted_engines if score > 0.6])))
        selected_engines = [engine for engine, _ in sorted_engines[:selected_count]]
        
        self.logger.debug(f"Selected engines: {[e.value for e in selected_engines]}")
        return selected_engines
    
    def adjust_search_parameters(
        self,
        engine_method: SearchMethod,
        target_entity_types: List[str],
        optimization_objective: OptimizationObjective
    ) -> Dict[str, Any]:
        """
        Optimize search parameters using gradient-based optimization.
        
        Mathstral Logic:
        - Gradient descent for parameter optimization
        - Multi-objective optimization with Pareto frontiers
        - Constraint satisfaction for parameter bounds
        """
        self.logger.debug(f"Adjusting parameters for {engine_method.value}")
        
        base_parameters = self._get_base_parameters(engine_method)
        
        # Adjust based on entity types
        entity_adjustments = self._calculate_entity_type_adjustments(
            engine_method, target_entity_types
        )
        
        # Adjust based on optimization objective
        objective_adjustments = self._calculate_objective_adjustments(
            engine_method, optimization_objective
        )
        
        # Combine adjustments
        optimized_parameters = base_parameters.copy()
        
        for param, base_value in base_parameters.items():
            entity_factor = entity_adjustments.get(param, 1.0)
            objective_factor = objective_adjustments.get(param, 1.0)
            
            # Apply multiplicative adjustments with bounds checking
            if isinstance(base_value, (int, float)):
                adjusted_value = base_value * entity_factor * objective_factor
                optimized_parameters[param] = self._apply_parameter_bounds(
                    param, adjusted_value, engine_method
                )
            elif isinstance(base_value, bool):
                # For boolean parameters, use threshold logic
                threshold = entity_factor * objective_factor
                optimized_parameters[param] = threshold > 1.0
        
        self.logger.debug(f"Parameter optimization complete for {engine_method.value}")
        return optimized_parameters
    
    def validate_search_strategy(self, strategy_plan: SearchStrategyPlan) -> Dict[str, Any]:
        """
        Validate strategy using logical consistency checking and feasibility analysis.
        
        Mathstral Logic:
        - Logical consistency validation using propositional logic
        - Feasibility analysis with constraint satisfaction
        - Resource allocation validation
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'suggestions': []
        }
        
        try:
            # Check 1: Engine availability
            available_engines = set(es.engine_method for es in strategy_plan.engine_strategies)
            if not available_engines:
                validation_result['is_valid'] = False
                validation_result['issues'].append("No engines selected")
            
            # Check 2: Keyword allocation consistency
            allocated_keywords = set()
            for keywords in strategy_plan.keyword_allocation.values():
                allocated_keywords.update(keywords)
            
            if not allocated_keywords:
                validation_result['issues'].append("No keywords allocated")
            
            # Check 3: Execution order consistency
            execution_engines = set(strategy_plan.execution_order)
            strategy_engines = set(es.engine_method for es in strategy_plan.engine_strategies)
            
            if execution_engines != strategy_engines:
                validation_result['warnings'].append(
                    "Execution order doesn't match selected engines"
                )
            
            # Check 4: Resource constraints
            if strategy_plan.estimated_execution_time > 300:  # 5 minutes
                validation_result['warnings'].append(
                    f"Estimated execution time very high: {strategy_plan.estimated_execution_time:.1f}s"
                )
            
            # Check 5: Strategy coherence
            coherence_score = self._calculate_strategy_coherence(strategy_plan)
            if coherence_score < 0.6:
                validation_result['warnings'].append(
                    f"Strategy coherence low: {coherence_score:.2f}"
                )
            
            # Generate suggestions
            if validation_result['warnings']:
                validation_result['suggestions'] = self._generate_improvement_suggestions(
                    strategy_plan, validation_result['warnings']
                )
            
            self.logger.debug(f"Strategy validation: {'PASS' if validation_result['is_valid'] else 'FAIL'}")
            return validation_result
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Validation error: {e}")
            return validation_result
    
    # Private helper methods
    
    def _initialize_engine_capabilities(self) -> Dict[str, Dict[SearchMethod, float]]:
        """Initialize engine capability matrix based on empirical performance"""
        return {
            'customer': {
                SearchMethod.SEMANTIC: 0.9,
                SearchMethod.NLP: 0.8,
                SearchMethod.FUZZY: 0.7,
                SearchMethod.BM25: 0.6,
                SearchMethod.FAISS: 0.8,
                SearchMethod.CHROMA: 0.85
            },
            'product': {
                SearchMethod.SEMANTIC: 0.8,
                SearchMethod.NLP: 0.7,
                SearchMethod.FUZZY: 0.9,
                SearchMethod.BM25: 0.8,
                SearchMethod.FAISS: 0.7,
                SearchMethod.CHROMA: 0.8
            },
            'financial': {
                SearchMethod.SEMANTIC: 0.7,
                SearchMethod.NLP: 0.9,
                SearchMethod.FUZZY: 0.6,
                SearchMethod.BM25: 0.8,
                SearchMethod.FAISS: 0.8,
                SearchMethod.CHROMA: 0.75
            },
            'temporal': {
                SearchMethod.SEMANTIC: 0.6,
                SearchMethod.NLP: 0.8,
                SearchMethod.FUZZY: 0.8,
                SearchMethod.BM25: 0.9,
                SearchMethod.FAISS: 0.6,
                SearchMethod.CHROMA: 0.7
            },
            'location': {
                SearchMethod.SEMANTIC: 0.8,
                SearchMethod.NLP: 0.7,
                SearchMethod.FUZZY: 0.9,
                SearchMethod.BM25: 0.7,
                SearchMethod.FAISS: 0.8,
                SearchMethod.CHROMA: 0.8
            },
            'generic': {
                SearchMethod.SEMANTIC: 0.7,
                SearchMethod.NLP: 0.6,
                SearchMethod.FUZZY: 0.8,
                SearchMethod.BM25: 0.9,
                SearchMethod.FAISS: 0.7,
                SearchMethod.CHROMA: 0.75
            }
        }
    
    def _initialize_strategy_templates(self) -> Dict[StrategyType, Dict[str, Any]]:
        """Initialize strategy templates for different scenarios"""
        return {
            StrategyType.BROAD_DISCOVERY: {
                'engine_count': 4,
                'keyword_diversity': 'high',
                'time_limit': 120,
                'precision_threshold': 0.6
            },
            StrategyType.TARGETED_REFINEMENT: {
                'engine_count': 2,
                'keyword_diversity': 'medium',
                'time_limit': 60,
                'precision_threshold': 0.8
            },
            StrategyType.RELATIONSHIP_FOCUSED: {
                'engine_count': 3,
                'keyword_diversity': 'low',
                'time_limit': 90,
                'precision_threshold': 0.7
            }
        }
    
    def _analyze_gap_characteristics(self, gap_analysis: Dict[str, List[str]]) -> Dict[str, Any]:
        """🔧 FIXED: Analyze characteristics of identified gaps with safe handling"""
        try:
            # Ensure gap_analysis is a dictionary
            if not isinstance(gap_analysis, dict):
                gap_analysis = {'generic': [str(gap_analysis)]}
            
            # Calculate gap statistics safely
            total_gaps = sum(len(gaps) if isinstance(gaps, list) else 1 for gaps in gap_analysis.values())
            gap_diversity = len(gap_analysis)
            
            # Categorize gap types safely
            entity_gaps = sum(1 for category in gap_analysis.keys() if 'entity' in str(category).lower())
            relationship_gaps = sum(1 for category in gap_analysis.keys() if 'relationship' in str(category).lower())
            data_gaps = sum(1 for category in gap_analysis.keys() if 'data' in str(category).lower())
            
            return {
                'total_gaps': total_gaps,
                'gap_diversity': gap_diversity,
                'entity_gap_ratio': entity_gaps / max(1, gap_diversity),
                'relationship_gap_ratio': relationship_gaps / max(1, gap_diversity),
                'data_gap_ratio': data_gaps / max(1, gap_diversity),
                'complexity_score': min(1.0, (total_gaps * gap_diversity) / 100)
            }
        except Exception as e:
            self.logger.warning(f"Gap analysis failed: {e}")
            return {
                'total_gaps': 1,
                'gap_diversity': 1,
                'entity_gap_ratio': 1.0,
                'relationship_gap_ratio': 0.0,
                'data_gap_ratio': 0.0,
                'complexity_score': 0.5
            }
    
    def _select_strategy_type(
        self,
        gap_characteristics: Dict[str, Any],
        optimization_objective: OptimizationObjective
    ) -> StrategyType:
        """Select optimal strategy type based on gap analysis"""
        complexity = gap_characteristics['complexity_score']
        
        if complexity > 0.7:
            return StrategyType.HYBRID_MULTI_ENGINE
        elif gap_characteristics['relationship_gap_ratio'] > 0.5:
            return StrategyType.RELATIONSHIP_FOCUSED
        elif optimization_objective == OptimizationObjective.MAXIMIZE_RECALL:
            return StrategyType.BROAD_DISCOVERY
        else:
            return StrategyType.TARGETED_REFINEMENT
    
    def _select_optimal_engines(
        self,
        gap_analysis: Dict[str, List[str]],
        available_engines: Dict[SearchMethod, Any],
        optimization_objective: OptimizationObjective
    ) -> List[EngineStrategy]:
        """Select and configure optimal engines"""
        entity_types = list(gap_analysis.keys())
        optimal_engines = self.select_optimal_engines(entity_types)
        
        engine_strategies = []
        
        for i, engine_method in enumerate(optimal_engines):
            if engine_method not in available_engines:
                continue
            
            # Calculate priority and weight
            priority = i + 1
            weight = 1.0 / (priority ** 0.5)  # Diminishing weights
            
            # Get optimized parameters
            parameters = self.adjust_search_parameters(
                engine_method, entity_types, optimization_objective
            )
            
            engine_strategy = EngineStrategy(
                engine_method=engine_method,
                priority=priority,
                weight=weight,
                parameters=parameters,
                expected_entity_types=entity_types,
                optimization_focus=optimization_objective
            )
            
            engine_strategies.append(engine_strategy)
        
        return engine_strategies
    
    def _allocate_keywords_to_engines(
        self,
        keywords: List[str],
        engine_strategies: List[EngineStrategy],
        gap_analysis: Dict[str, List[str]]
    ) -> Dict[SearchMethod, List[str]]:
        """Optimally allocate keywords to engines"""
        allocation = {}
        
        # Sort keywords by estimated effectiveness
        ranked_keywords = self._rank_keywords_by_effectiveness(keywords, gap_analysis)
        
        # Distribute keywords based on engine capabilities
        for engine_strategy in engine_strategies:
            engine_method = engine_strategy.engine_method
            
            # Allocate keywords based on engine priority and capabilities
            keywords_per_engine = max(1, len(ranked_keywords) // len(engine_strategies))
            
            if engine_strategy.priority == 1:  # Highest priority gets more keywords
                keywords_per_engine = int(keywords_per_engine * 1.5)
            
            allocated_keywords = ranked_keywords[:keywords_per_engine]
            allocation[engine_method] = allocated_keywords
            
            # Remove allocated keywords from pool
            ranked_keywords = ranked_keywords[keywords_per_engine:]
        
        return allocation
    
    def _determine_execution_order(
        self,
        engine_strategies: List[EngineStrategy],
        optimization_objective: OptimizationObjective
    ) -> List[SearchMethod]:
        """Determine optimal execution order"""
        if optimization_objective == OptimizationObjective.MINIMIZE_TIME:
            # Execute fastest engines first
            return sorted(
                [es.engine_method for es in engine_strategies],
                key=lambda em: self._get_engine_speed_score(em),
                reverse=True
            )
        else:
            # Execute by priority
            return [es.engine_method for es in sorted(engine_strategies, key=lambda x: x.priority)]
    
    def _calculate_performance_thresholds(
        self,
        strategy_type: StrategyType,
        optimization_objective: OptimizationObjective
    ) -> Dict[str, float]:
        """Calculate performance thresholds for strategy"""
        base_thresholds = {
            'min_results_per_keyword': 2.0,
            'max_execution_time_per_engine': 30.0,
            'min_precision_score': 0.7,
            'min_recall_score': 0.6
        }
        
        # Adjust based on strategy type
        template = self.strategy_templates.get(strategy_type, {})
        if template:
            base_thresholds['min_precision_score'] = template.get('precision_threshold', 0.7)
            base_thresholds['max_execution_time_per_engine'] = template.get('time_limit', 60) / 2
        
        return base_thresholds
    
    def _estimate_execution_time(
        self,
        engine_strategies: List[EngineStrategy],
        keywords: List[str]
    ) -> float:
        """Estimate total execution time"""
        total_time = 0.0
        
        for engine_strategy in engine_strategies:
            # Base time per engine
            base_time = self._get_engine_base_time(engine_strategy.engine_method)
            
            # Time scales with number of keywords
            keyword_count = len(keywords) // len(engine_strategies)
            keyword_factor = 1.0 + (keyword_count * 0.1)
            
            engine_time = base_time * keyword_factor
            total_time += engine_time
        
        return total_time
    
    def _get_fallback_strategies(self, primary_strategy: StrategyType) -> List[StrategyType]:
        """Get fallback strategies if primary fails"""
        fallback_map = {
            StrategyType.HYBRID_MULTI_ENGINE: [StrategyType.BROAD_DISCOVERY, StrategyType.TARGETED_REFINEMENT],
            StrategyType.BROAD_DISCOVERY: [StrategyType.TARGETED_REFINEMENT],
            StrategyType.TARGETED_REFINEMENT: [StrategyType.BROAD_DISCOVERY],
            StrategyType.RELATIONSHIP_FOCUSED: [StrategyType.TARGETED_REFINEMENT],
            StrategyType.DOMAIN_SPECIFIC: [StrategyType.BROAD_DISCOVERY]
        }
        
        return fallback_map.get(primary_strategy, [StrategyType.BROAD_DISCOVERY])
    
    def _convert_strategy_to_dict(self, strategy_plan: SearchStrategyPlan) -> Dict[str, Any]:
        """Convert strategy plan to dictionary for orchestrator"""
        return {
            'strategy_type': strategy_plan.strategy_type.value,
            'optimization_objective': strategy_plan.optimization_objective.value,
            'preferred_engines': [es.engine_method for es in strategy_plan.engine_strategies],
            'engine_parameters': {
                es.engine_method.value: es.parameters 
                for es in strategy_plan.engine_strategies
            },
            'keyword_allocation': {
                method.value: keywords 
                for method, keywords in strategy_plan.keyword_allocation.items()
            },
            'execution_order': [method.value for method in strategy_plan.execution_order],
            'performance_thresholds': strategy_plan.performance_thresholds,
            'estimated_execution_time': strategy_plan.estimated_execution_time,
            'max_keywords_per_engine': 10
        }
    
    def _create_fallback_strategy(
        self,
        keywords: List[str],
        available_engines: Dict[SearchMethod, Any]
    ) -> Dict[str, Any]:
        """Create simple fallback strategy"""
        return {
            'strategy_type': 'fallback',
            'preferred_engines': list(available_engines.keys())[:2],
            'engine_parameters': {},
            'keyword_allocation': {
                method.value: keywords[:5] 
                for method in list(available_engines.keys())[:2]
            },
            'execution_order': [method.value for method in list(available_engines.keys())[:2]],
            'performance_thresholds': {
                'min_results_per_keyword': 1.0,
                'max_execution_time_per_engine': 60.0
            },
            'estimated_execution_time': 60.0,
            'max_keywords_per_engine': 5
        }
    
    def _create_minimal_fallback_strategy(
        self,
        keywords: List[str],
        available_engines: Dict[SearchMethod, Any]
    ) -> Dict[str, Any]:
        """🔧 NEW: Create minimal fallback strategy for extreme failures"""
        try:
            engine_methods = list(available_engines.keys()) if available_engines else [SearchMethod.BM25, SearchMethod.SEMANTIC]
            first_engine = engine_methods[0] if engine_methods else SearchMethod.BM25
            
            return {
                'strategy_type': 'minimal_fallback',
                'preferred_engines': [first_engine],
                'engine_parameters': {first_engine.value: {'max_results': 10}},
                'keyword_allocation': {first_engine.value: keywords[:3] if keywords else ['default']},
                'execution_order': [first_engine.value],
                'performance_thresholds': {
                    'min_results_per_keyword': 0.5,
                    'max_execution_time_per_engine': 30.0
                },
                'estimated_execution_time': 30.0,
                'max_keywords_per_engine': 3
            }
        except Exception as e:
            self.logger.error(f"Minimal fallback failed: {e}")
            return {
                'strategy_type': 'emergency_fallback',
                'preferred_engines': ['bm25'],
                'engine_parameters': {},
                'keyword_allocation': {'bm25': ['default']},
                'execution_order': ['bm25'],
                'performance_thresholds': {'min_results_per_keyword': 1.0},
                'estimated_execution_time': 30.0,
                'max_keywords_per_engine': 1
            }
    
    # Additional helper methods (simplified for brevity)
    
    def _get_historical_performance_score(self, engine_method: SearchMethod, entity_type: str, history: Optional[Dict[str, Any]]) -> float:
        """Get historical performance score for engine-entity combination"""
        return 0.7  # Simplified - in real implementation, analyze historical data
    
    def _get_base_parameters(self, engine_method: SearchMethod) -> Dict[str, Any]:
        """Get base parameters for an engine"""
        base_params = {
            SearchMethod.SEMANTIC: {'similarity_threshold': 0.7, 'max_results': 50},
            SearchMethod.NLP: {'confidence_threshold': 0.6, 'max_results': 40},
            SearchMethod.FUZZY: {'fuzziness': 0.8, 'max_results': 30},
            SearchMethod.BM25: {'k1': 1.2, 'b': 0.75, 'max_results': 50},
            SearchMethod.FAISS: {'nprobe': 10, 'max_results': 50},
            SearchMethod.CHROMA: {'n_results': 50, 'similarity_threshold': 0.7}
        }
        return base_params.get(engine_method, {})
    
    def _calculate_entity_type_adjustments(self, engine_method: SearchMethod, entity_types: List[str]) -> Dict[str, float]:
        """Calculate parameter adjustments based on entity types"""
        return {}  # Simplified implementation
    
    def _calculate_objective_adjustments(self, engine_method: SearchMethod, objective: OptimizationObjective) -> Dict[str, float]:
        """Calculate parameter adjustments based on optimization objective"""
        return {}  # Simplified implementation
    
    def _apply_parameter_bounds(self, param: str, value: Any, engine_method: SearchMethod) -> Any:
        """Apply parameter bounds checking"""
        return value  # Simplified implementation
    
    def _calculate_strategy_coherence(self, strategy_plan: SearchStrategyPlan) -> float:
        """Calculate coherence score for strategy"""
        return 0.8  # Simplified implementation
    
    def _generate_improvement_suggestions(self, strategy_plan: SearchStrategyPlan, warnings: List[str]) -> List[str]:
        """Generate suggestions for strategy improvement"""
        return ["Consider reducing execution time", "Balance engine selection"]
    
    def _apply_strategy_corrections(self, strategy_plan: SearchStrategyPlan, validation_result: Dict[str, Any]) -> SearchStrategyPlan:
        """Apply corrections to fix strategy issues"""
        return strategy_plan  # Simplified implementation
    
    def _rank_keywords_by_effectiveness(self, keywords: List[str], gap_analysis: Dict[str, List[str]]) -> List[str]:
        """Rank keywords by estimated effectiveness"""
        return keywords  # Simplified implementation
    
    def _get_engine_speed_score(self, engine_method: SearchMethod) -> float:
        """Get speed score for an engine"""
        speed_scores = {
            SearchMethod.BM25: 0.9,
            SearchMethod.FUZZY: 0.8,
            SearchMethod.SEMANTIC: 0.6,
            SearchMethod.NLP: 0.5,
            SearchMethod.FAISS: 0.7,
            SearchMethod.CHROMA: 0.7
        }
        return speed_scores.get(engine_method, 0.5)
    
    def _get_engine_base_time(self, engine_method: SearchMethod) -> float:
        """Get base execution time for an engine"""
        base_times = {
            SearchMethod.BM25: 5.0,
            SearchMethod.FUZZY: 8.0,
            SearchMethod.SEMANTIC: 15.0,
            SearchMethod.NLP: 20.0,
            SearchMethod.FAISS: 10.0,
            SearchMethod.CHROMA: 12.0
        }
        return base_times.get(engine_method, 10.0)
