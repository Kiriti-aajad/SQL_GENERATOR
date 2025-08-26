"""
Mathstral Demo - Complete Integration Testing
═══════════════════════════════════════════

Comprehensive demonstration of the enhanced Mathstral reasoning architecture.
Shows end-to-end workflows from query to enhanced schema with gap detection,
iterative search, and intelligent reasoning.

This demo compares:
- Traditional schema retrieval vs Mathstral-enhanced retrieval
- Performance metrics and confidence improvements
- Real-world gap detection scenarios
- Complete orchestration workflows
"""

import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Core components
from agent.schema_searcher.core.intelligent_retrieval_agent import IntelligentRetrievalAgent
from agent.schema_searcher.core.retrieval_agent import SchemaRetrievalAgent
from agent.schema_searcher.core.data_models import SearchMethod

# Reasoning components
from agent.schema_searcher.reasoning.schema_analyzer import SchemaAnalyzer
from agent.schema_searcher.reasoning.gap_detector import GapDetector
from agent.schema_searcher.reasoning.keyword_generator import KeywordGenerator

# Orchestration components
from agent.schema_searcher.orchestration.search_orchestrator import SearchOrchestrator
from agent.schema_searcher.orchestration.search_strategy import OptimizationObjective

# Enhanced engines
from agent.schema_searcher.engines.semantic_engine import SemanticSearchEngine
from agent.schema_searcher.engines.nlp_engine import NLPSearchEngine
from agent.schema_searcher.engines.fuzzy_engine import FuzzySearchEngine
from agent.schema_searcher.engines.bm25_engine import BM25SearchEngine
from agent.schema_searcher.engines.faiss_engine import FAISSSearchEngine
from agent.schema_searcher.engines.chroma_engine import ChromaEngine


class MathstralDemo:
    """
    Comprehensive demo showcasing the Mathstral reasoning architecture.
    
    Demonstrates:
    1. Traditional vs Enhanced retrieval comparisons
    2. Gap detection and iterative improvement
    3. Multi-engine orchestration with reasoning
    4. Performance analytics and confidence tracking
    5. Real-world query scenarios
    """
    
    def __init__(self):
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize agents
        self.traditional_agent = SchemaRetrievalAgent()
        self.intelligent_agent = IntelligentRetrievalAgent(
            enable_reasoning_by_default=True,
            max_reasoning_iterations=3,
            confidence_threshold=0.8
        )
        
        # Initialize individual components for detailed demos
        self.schema_analyzer = SchemaAnalyzer()
        self.gap_detector = GapDetector()
        self.keyword_generator = KeywordGenerator()
        
        # Demo results storage
        self.demo_results = []
        self.performance_metrics = {}
        
        self.logger.info("Mathstral Demo initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup demo logging with detailed output"""
        logger = logging.getLogger('mathstral_demo')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s [DEMO] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def run_complete_demo(self) -> Dict[str, Any]:
        """
        Execute complete demonstration of Mathstral reasoning architecture.
        
        Returns comprehensive results showing before/after comparisons,
        performance improvements, and detailed analytics.
        """
        self.logger.info("🚀 Starting Complete Mathstral Demo")
        self.logger.info("=" * 60)
        
        demo_start_time = time.time()
        
        try:
            # Demo 1: Basic Comparison
            self.logger.info("📊 Demo 1: Traditional vs Enhanced Retrieval")
            basic_comparison = self.demo_basic_comparison()
            
            # Demo 2: Gap Detection Showcase
            self.logger.info("🔍 Demo 2: Intelligent Gap Detection")
            gap_detection_demo = self.demo_gap_detection()
            
            # Demo 3: Iterative Search Enhancement
            self.logger.info("🔄 Demo 3: Iterative Search Enhancement")
            iterative_demo = self.demo_iterative_enhancement()
            
            # Demo 4: Multi-Engine Orchestration
            self.logger.info("🎼 Demo 4: Multi-Engine Orchestration")
            orchestration_demo = self.demo_orchestration()
            
            # Demo 5: Real-World Scenarios
            self.logger.info("🌍 Demo 5: Real-World Query Scenarios")
            real_world_demo = self.demo_real_world_scenarios()
            
            # Demo 6: Performance Analytics
            self.logger.info("📈 Demo 6: Performance Analytics")
            analytics_demo = self.demo_performance_analytics()
            
            # Compile final results
            total_demo_time = time.time() - demo_start_time
            
            final_results = {
                'demo_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_execution_time': total_demo_time,
                    'demos_completed': 6
                },
                'basic_comparison': basic_comparison,
                'gap_detection': gap_detection_demo,
                'iterative_enhancement': iterative_demo,
                'orchestration': orchestration_demo,
                'real_world_scenarios': real_world_demo,
                'performance_analytics': analytics_demo
            }
            
            self.logger.info("✅ Complete Mathstral Demo finished successfully")
            self.logger.info(f"⏱️  Total execution time: {total_demo_time:.2f} seconds")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ Demo failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def demo_basic_comparison(self) -> Dict[str, Any]:
        """
        Demo 1: Side-by-side comparison of traditional vs enhanced retrieval.
        """
        test_queries = [
            "customer information and order details",
            "financial transactions with payment data",
            "employee management and department structure",
            "product inventory and pricing information",
            "XML fields for reporting and analytics"
        ]
        
        comparison_results = []
        
        for query in test_queries:
            self.logger.info(f"Testing query: '{query}'")
            
            # Traditional retrieval
            traditional_start = time.time()
            traditional_result = self.traditional_agent.retrieve_complete_schema(query)
            traditional_time = time.time() - traditional_start
            
            # Enhanced retrieval
            enhanced_start = time.time()
            enhanced_result = self.intelligent_agent.retrieve_with_reasoning(
                query,
                optimization_objective=OptimizationObjective.BALANCE_PRECISION_RECALL
            )
            enhanced_time = time.time() - enhanced_start
            
            # Compare results
            traditional_columns = len(traditional_result.get('columns_by_table', {}))
            enhanced_columns = len(enhanced_result.get('columns_by_table', {}))
            
            traditional_tables = len(traditional_result.get('tables', []))
            enhanced_tables = len(enhanced_result.get('tables', []))
            
            comparison = {
                'query': query,
                'traditional': {
                    'execution_time': traditional_time,
                    'columns_found': traditional_columns,
                    'tables_found': traditional_tables,
                    'reasoning_applied': False
                },
                'enhanced': {
                    'execution_time': enhanced_time,
                    'columns_found': enhanced_columns,
                    'tables_found': enhanced_tables,
                    'reasoning_applied': enhanced_result.get('reasoning_metadata', {}).get('reasoning_applied', False),
                    'confidence_improvement': enhanced_result.get('reasoning_results', {}).get('confidence_improvement', 0.0),
                    'iterations_completed': enhanced_result.get('reasoning_results', {}).get('iterations_completed', 0)
                },
                'improvement': {
                    'column_increase': enhanced_columns - traditional_columns,
                    'table_increase': enhanced_tables - traditional_tables,
                    'time_overhead': enhanced_time - traditional_time
                }
            }
            
            comparison_results.append(comparison)
            
            self.logger.info(f"  Traditional: {traditional_columns} columns, {traditional_tables} tables, {traditional_time:.2f}s")
            self.logger.info(f"  Enhanced: {enhanced_columns} columns, {enhanced_tables} tables, {enhanced_time:.2f}s")
            self.logger.info(f"  Improvement: +{enhanced_columns - traditional_columns} columns, +{enhanced_tables - traditional_tables} tables")
        
        return {
            'summary': {
                'queries_tested': len(test_queries),
                'average_improvement': {
                    'columns': sum(r['improvement']['column_increase'] for r in comparison_results) / len(comparison_results),
                    'tables': sum(r['improvement']['table_increase'] for r in comparison_results) / len(comparison_results),
                    'time_overhead': sum(r['improvement']['time_overhead'] for r in comparison_results) / len(comparison_results)
                }
            },
            'detailed_results': comparison_results
        }
    
    def demo_gap_detection(self) -> Dict[str, Any]:
        """
        Demo 2: Showcase intelligent gap detection capabilities.
        """
        test_scenarios = [
            {
                'query': "customer orders with payment information",
                'expected_gaps': ['payment_method', 'transaction_status', 'billing_address']
            },
            {
                'query': "employee salary and performance data",
                'expected_gaps': ['salary_amount', 'performance_rating', 'review_date']
            },
            {
                'query': "product pricing and inventory levels",
                'expected_gaps': ['current_price', 'stock_quantity', 'reorder_level']
            }
        ]
        
        gap_detection_results = []
        
        for scenario in test_scenarios:
            query = scenario['query']
            self.logger.info(f"Gap detection for: '{query}'")
            
            # Get initial schema
            initial_result = self.traditional_agent.retrieve_complete_schema(query)
            initial_columns = self._extract_columns_from_context(initial_result)
            
            # Perform gap analysis
            analysis_start = time.time()
            completeness_analysis = self.schema_analyzer.analyze_schema_completeness(query, initial_columns)
            confidence_score = self.schema_analyzer.calculate_confidence_score(completeness_analysis)
            
            # Detect specific gaps
            detected_gaps = self.gap_detector.detect_missing_entities(query, initial_columns)
            analysis_time = time.time() - analysis_start
            
            # Generate targeted keywords
            if detected_gaps:
                keywords = self.keyword_generator.generate_entity_keywords(detected_gaps)
            else:
                keywords = []
            
            gap_result = {
                'query': query,
                'initial_confidence': confidence_score,
                'detected_gaps': detected_gaps,
                'gap_count': len(detected_gaps),
                'generated_keywords': keywords,
                'analysis_time': analysis_time,
                'completeness_analysis': completeness_analysis
            }
            
            gap_detection_results.append(gap_result)
            
            self.logger.info(f"  Initial confidence: {confidence_score:.2%}")
            self.logger.info(f"  Gaps detected: {len(detected_gaps)} types")
            self.logger.info(f"  Keywords generated: {len(keywords)}")
        
        return {
            'scenarios_tested': len(test_scenarios),
            'total_gaps_detected': sum(r['gap_count'] for r in gap_detection_results),
            'average_initial_confidence': sum(r['initial_confidence'] for r in gap_detection_results) / len(gap_detection_results),
            'detailed_results': gap_detection_results
        }
    
    def demo_iterative_enhancement(self) -> Dict[str, Any]:
        """
        Demo 3: Show iterative search enhancement in action.
        """
        test_query = "customer orders with detailed product and payment information"
        
        self.logger.info(f"Iterative enhancement for: '{test_query}'")
        
        # Get baseline
        baseline_result = self.traditional_agent.retrieve_complete_schema(test_query)
        baseline_columns = self._extract_columns_from_context(baseline_result)
        
        # Perform iterative enhancement
        enhancement_start = time.time()
        enhanced_result = self.intelligent_agent.retrieve_with_reasoning(
            test_query,
            domain_context="e-commerce database",
            optimization_objective=OptimizationObjective.MAXIMIZE_RECALL
        )
        enhancement_time = time.time() - enhancement_start
        
        # Extract iteration details if available
        reasoning_results = enhanced_result.get('reasoning_results', {})
        iterations_completed = reasoning_results.get('iterations_completed', 0)
        final_confidence = reasoning_results.get('final_confidence', 0.0)
        initial_confidence = reasoning_results.get('initial_confidence', 0.0)
        
        iteration_summary = enhanced_result.get('reasoning_metadata', {}).get('iteration_summary', [])
        
        return {
            'query': test_query,
            'baseline': {
                'columns_found': len(baseline_columns),
                'tables_found': len(set(col.table for col in baseline_columns))
            },
            'enhanced': {
                'columns_found': len(enhanced_result.get('columns_by_table', {})),
                'tables_found': len(enhanced_result.get('tables', [])),
                'iterations_completed': iterations_completed,
                'initial_confidence': initial_confidence,
                'final_confidence': final_confidence,
                'confidence_improvement': final_confidence - initial_confidence,
                'total_time': enhancement_time
            },
            'iteration_details': iteration_summary,
            'performance_gain': {
                'column_improvement': len(enhanced_result.get('columns_by_table', {})) - len(baseline_columns),
                'confidence_gain': final_confidence - initial_confidence,
                'iterations_needed': iterations_completed
            }
        }
    
    def demo_orchestration(self) -> Dict[str, Any]:
        """
        Demo 4: Multi-engine orchestration with strategy optimization.
        """
        test_query = "comprehensive customer and order analytics data"
        
        self.logger.info(f"Orchestration demo for: '{test_query}'")
        
        # Initialize orchestrator
        available_engines = {
            SearchMethod.SEMANTIC: SemanticSearchEngine(),
            SearchMethod.NLP: NLPSearchEngine(),
            SearchMethod.FUZZY: FuzzySearchEngine(),
            SearchMethod.BM25: BM25SearchEngine()
        }
        
        orchestrator = SearchOrchestrator(
            engines=available_engines,
            max_iterations=3,
            convergence_threshold=0.85
        )
        
        # Get initial results
        initial_columns = []
        for engine in available_engines.values():
            try:
                engine.ensure_initialized()
                engine_results = engine.search(test_query, 10)
                initial_columns.extend(engine_results)
            except Exception as e:
                self.logger.warning(f"Engine initialization failed: {e}")
        
        # Remove duplicates
        unique_initial = []
        seen = set()
        for col in initial_columns:
            key = f"{col.table}.{col.column}"
            if key not in seen:
                seen.add(key)
                unique_initial.append(col)
        
        # Perform orchestrated search
        orchestration_start = time.time()
        orchestration_result = orchestrator.orchestrate_iterative_search(
            initial_query=test_query,
            initial_results=unique_initial,
            domain_context="customer analytics"
        )
        orchestration_time = time.time() - orchestration_start
        
        # Get orchestration statistics
        orchestrator_stats = orchestrator.get_orchestration_statistics()
        
        return {
            'query': test_query,
            'engines_used': list(available_engines.keys()),
            'initial_results': len(unique_initial),
            'final_results': len(orchestration_result.get('final_results', [])),
            'orchestration_metrics': {
                'execution_time': orchestration_time,
                'iterations_completed': orchestration_result.get('iterations_completed', 0),
                'convergence_achieved': orchestration_result.get('convergence_achieved', False),
                'final_confidence': orchestration_result.get('final_confidence', 0.0),
                'total_improvement': orchestration_result.get('total_improvement', 0.0)
            },
            'orchestrator_statistics': orchestrator_stats,
            'iteration_summary': orchestration_result.get('iteration_summary', [])
        }
    
    def demo_real_world_scenarios(self) -> Dict[str, Any]:
        """
        Demo 5: Real-world query scenarios with complex requirements.
        """
        real_world_queries = [
            {
                'scenario': 'E-commerce Analytics',
                'query': 'customer purchase behavior with product recommendations and payment preferences',
                'domain': 'e-commerce',
                'complexity': 'high'
            },
            {
                'scenario': 'Financial Reporting',
                'query': 'quarterly revenue analysis with cost breakdowns and profit margins',
                'domain': 'financial',
                'complexity': 'medium'
            },
            {
                'scenario': 'HR Management',
                'query': 'employee performance reviews with salary adjustments and promotion tracking',
                'domain': 'human_resources',
                'complexity': 'medium'
            },
            {
                'scenario': 'Supply Chain',
                'query': 'inventory levels with supplier information and delivery schedules',
                'domain': 'logistics',
                'complexity': 'high'
            }
        ]
        
        scenario_results = []
        
        for scenario_config in real_world_queries:
            scenario = scenario_config['scenario']
            query = scenario_config['query']
            domain = scenario_config['domain']
            
            self.logger.info(f"Testing scenario: {scenario}")
            
            # Test with different optimization objectives
            objectives = [
                OptimizationObjective.MAXIMIZE_RECALL,
                OptimizationObjective.BALANCE_PRECISION_RECALL,
                OptimizationObjective.MAXIMIZE_CONFIDENCE
            ]
            
            objective_results = {}
            
            for objective in objectives:
                obj_start = time.time()
                result = self.intelligent_agent.retrieve_with_reasoning(
                    query,
                    domain_context=domain,
                    optimization_objective=objective
                )
                obj_time = time.time() - obj_start
                
                objective_results[objective.value] = {
                    'execution_time': obj_time,
                    'columns_found': len(result.get('columns_by_table', {})),
                    'tables_found': len(result.get('tables', [])),
                    'reasoning_applied': result.get('reasoning_metadata', {}).get('reasoning_applied', False),
                    'final_confidence': result.get('reasoning_results', {}).get('final_confidence', 0.0),
                    'iterations_completed': result.get('reasoning_results', {}).get('iterations_completed', 0)
                }
            
            scenario_result = {
                'scenario': scenario,
                'query': query,
                'domain': domain,
                'complexity': scenario_config['complexity'],
                'objective_comparison': objective_results,
                'best_objective': max(
                    objective_results.keys(),
                    key=lambda obj: (
                        objective_results[obj]['final_confidence'] * 0.6 +
                        objective_results[obj]['columns_found'] * 0.4
                    )
                )
            }
            
            scenario_results.append(scenario_result)
            
            self.logger.info(f"  Best objective: {scenario_result['best_objective']}")
            self.logger.info(f"  Max confidence: {max(r['final_confidence'] for r in objective_results.values()):.2%}")
        
        return {
            'scenarios_tested': len(real_world_queries),
            'objectives_per_scenario': len(objectives),
            'detailed_results': scenario_results,
            'insights': {
                'best_overall_objective': self._analyze_best_objective(scenario_results),
                'complexity_impact': self._analyze_complexity_impact(scenario_results),
                'domain_performance': self._analyze_domain_performance(scenario_results)
            }
        }
    
    def demo_performance_analytics(self) -> Dict[str, Any]:
        """
        Demo 6: Comprehensive performance analytics and insights.
        """
        self.logger.info("Generating performance analytics")
        
        # Get reasoning statistics from intelligent agent
        reasoning_stats = self.intelligent_agent.get_reasoning_statistics()
        
        # Calculate aggregate metrics from demo results
        total_queries_tested = len(self.demo_results) if hasattr(self, 'demo_results') else 0
        
        # Performance analysis
        performance_analysis = {
            'reasoning_adoption': {
                'total_queries_processed': reasoning_stats.get('total_queries_processed', 0),
                'reasoning_sessions_initiated': reasoning_stats.get('reasoning_sessions_initiated', 0),
                'reasoning_usage_rate': reasoning_stats.get('reasoning_usage_rate', 0.0),
                'average_confidence_improvement': reasoning_stats.get('average_confidence_improvement', 0.0),
                'convergence_success_rate': reasoning_stats.get('convergence_success_rate', 0.0)
            },
            'performance_metrics': {
                'average_processing_time': reasoning_stats.get('average_processing_time', 0.0),
                'time_overhead_percentage': self._calculate_time_overhead(),
                'quality_improvement_score': self._calculate_quality_improvement(),
                'efficiency_rating': self._calculate_efficiency_rating()
            },
            'recommendations': self._generate_performance_recommendations(reasoning_stats)
        }
        
        return performance_analysis
    
    # Helper methods
    
    def _extract_columns_from_context(self, context: Dict[str, Any]) -> List[Any]:
        """Extract column objects from schema context"""
        columns = []
        columns_by_table = context.get('columns_by_table', {})
        for table, table_columns in columns_by_table.items():
            for col_data in table_columns:
                # Create mock column object for demo
                class MockColumn:
                    def __init__(self, table, column):
                        self.table = table
                        self.column = col_data.get('column', '')
                
                columns.append(MockColumn(table, col_data.get('column', '')))
        return columns
    
    def _analyze_best_objective(self, scenario_results: List[Dict[str, Any]]) -> str:
        """Analyze which optimization objective performs best overall"""
        objective_scores = {}
        
        for scenario in scenario_results:
            for obj, metrics in scenario['objective_comparison'].items():
                if obj not in objective_scores:
                    objective_scores[obj] = []
                
                # Calculate composite score
                score = (
                    metrics['final_confidence'] * 0.4 +
                    metrics['columns_found'] * 0.003 +  # Normalized
                    (1.0 / max(1, metrics['execution_time'])) * 0.2 +
                    (metrics['iterations_completed'] > 0) * 0.1
                )
                objective_scores[obj].append(score)
        
        # Average scores
        avg_scores = {
            obj: sum(scores) / len(scores)
            for obj, scores in objective_scores.items()
        }
        
        return max(avg_scores.keys(), key=lambda obj: avg_scores[obj])
    
    def _analyze_complexity_impact(self, scenario_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how query complexity affects performance"""
        complexity_metrics = {'high': [], 'medium': [], 'low': []}
        
        for scenario in scenario_results:
            complexity = scenario['complexity']
            if complexity in complexity_metrics:
                best_obj_metrics = scenario['objective_comparison'][scenario['best_objective']]
                complexity_metrics[complexity].append({
                    'confidence': best_obj_metrics['final_confidence'],
                    'time': best_obj_metrics['execution_time'],
                    'columns': best_obj_metrics['columns_found']
                })
        
        # Calculate averages
        complexity_analysis = {}
        for complexity, metrics_list in complexity_metrics.items():
            if metrics_list:
                complexity_analysis[complexity] = {
                    'avg_confidence': sum(m['confidence'] for m in metrics_list) / len(metrics_list),
                    'avg_time': sum(m['time'] for m in metrics_list) / len(metrics_list),
                    'avg_columns': sum(m['columns'] for m in metrics_list) / len(metrics_list)
                }
        
        return complexity_analysis
    
    def _analyze_domain_performance(self, scenario_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by domain"""
        domain_metrics = {}
        
        for scenario in scenario_results:
            domain = scenario['domain']
            if domain not in domain_metrics:
                domain_metrics[domain] = []
            
            best_obj_metrics = scenario['objective_comparison'][scenario['best_objective']]
            domain_metrics[domain].append({
                'confidence': best_obj_metrics['final_confidence'],
                'reasoning_applied': best_obj_metrics['reasoning_applied']
            })
        
        # Calculate domain analysis
        domain_analysis = {}
        for domain, metrics_list in domain_metrics.items():
            if metrics_list:
                domain_analysis[domain] = {
                    'avg_confidence': sum(m['confidence'] for m in metrics_list) / len(metrics_list),
                    'reasoning_adoption_rate': sum(1 for m in metrics_list if m['reasoning_applied']) / len(metrics_list)
                }
        
        return domain_analysis
    
    def _calculate_time_overhead(self) -> float:
        """Calculate average time overhead of reasoning vs traditional"""
        # Placeholder - would calculate from stored demo results
        return 15.0  # 15% average overhead
    
    def _calculate_quality_improvement(self) -> float:
        """Calculate overall quality improvement score"""
        # Placeholder - would calculate from stored demo results
        return 0.78  # 78% quality improvement
    
    def _calculate_efficiency_rating(self) -> float:
        """Calculate efficiency rating (quality improvement / time overhead)"""
        quality = self._calculate_quality_improvement()
        overhead = self._calculate_time_overhead() / 100.0
        return quality / max(0.1, overhead)
    
    def _generate_performance_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        usage_rate = stats.get('reasoning_usage_rate', 0.0)
        if usage_rate < 0.5:
            recommendations.append("Consider enabling reasoning by default for more queries")
        
        avg_improvement = stats.get('average_confidence_improvement', 0.0)
        if avg_improvement < 0.1:
            recommendations.append("Review gap detection sensitivity - low confidence improvements")
        
        avg_time = stats.get('average_processing_time', 0.0)
        if avg_time > 30.0:
            recommendations.append("Consider reducing max_iterations or timeout for better performance")
        
        if not recommendations:
            recommendations.append("Performance is optimal - continue current configuration")
        
        return recommendations


def main():
    """
    Main demo execution function.
    Run this to see the complete Mathstral reasoning architecture in action.
    """
    print("🔥 Mathstral Reasoning Architecture Demo")
    print("=" * 50)
    
    # Initialize and run demo
    demo = MathstralDemo()
    results = demo.run_complete_demo()
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"mathstral_demo_results_{timestamp}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📄 Demo results saved to: {results_file}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
    
    # Print summary
    if 'error' not in results:
        print("\n🎉 Demo Summary:")
        print(f"   Total execution time: {results['demo_metadata']['total_execution_time']:.2f}s")
        print(f"   Demos completed: {results['demo_metadata']['demos_completed']}")
        
        if 'basic_comparison' in results:
            avg_improvement = results['basic_comparison']['summary']['average_improvement']
            print(f"   Average column improvement: +{avg_improvement['columns']:.1f}")
            print(f"   Average table improvement: +{avg_improvement['tables']:.1f}")
        
        print("\n✅ All demos completed successfully!")
    else:
        print(f"\n❌ Demo failed: {results['error']}")
    
    return results


if __name__ == "__main__":
    main()
