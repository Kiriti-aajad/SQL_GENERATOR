"""
Ensemble Manager - CORRECTED VERSION
Simplified to work with dictionary responses from MathstralClient and DeepSeekClient
FIXED: No complex data structures, works with simple HTTP responses
FIXED: Compatible with simplified SQLGenerator architecture
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from enum import Enum

# FIXED: Only import what we actually need from our corrected configs
from agent.sql_generator.config.model_configs import ModelType

class EnsembleCombinationStrategy(Enum):
    """Strategies for combining ensemble results"""
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    MATHSTRAL_PRIMARY = "mathstral_primary"  
    DEEPSEEK_PRIMARY = "deepseek_primary"
    CONSENSUS_BASED = "consensus_based"
    BEST_RESPONSE = "best_response"

class EnsembleManager:
    """SIMPLIFIED: Manages ensemble operations for multi-model SQL generation"""
    
    def __init__(self):
        self.logger = logging.getLogger("EnsembleManager")
        
        # SIMPLIFIED: Basic configuration
        self.min_confidence_threshold = 0.6
        self.consensus_threshold = 0.8
        
        # SIMPLIFIED: Model specialization preferences
        self.model_preferences = {
            "mathstral": {
                "analytical": 0.9,
                "mathematical": 0.9, 
                "statistical": 0.9,
                "financial": 0.8
            },
            "deepseek": {
                "general": 0.9,
                "crud": 0.9,
                "xml": 0.9,
                "optimization": 0.8
            }
        }
        
        self.logger.info("EnsembleManager initialized with simplified configuration")

    async def combine_results(self, 
                             results: Dict[str, Dict[str, Any]], 
                             query_context: Dict[str, Any] = None, # pyright: ignore[reportArgumentType]
                             strategy: EnsembleCombinationStrategy = EnsembleCombinationStrategy.CONFIDENCE_WEIGHTED) -> Dict[str, Any]:
        """
        SIMPLIFIED: Combine results from multiple models
        
        Args:
            results: Dict of model_name -> response_dict from simplified clients
            query_context: Optional context about the query
            strategy: Combination strategy to use
            
        Returns:
            Combined result dictionary
        """
        
        start_time = time.time()
        query_context = query_context or {}
        
        self.logger.info(f"Combining results from {len(results)} models using {strategy.value}")
        
        # FIXED: Filter successful results (simple check)
        successful_results = {
            model_name: result for model_name, result in results.items()
            if result.get("success", False) and result.get("sql", "").strip()
        }
        
        if not successful_results:
            return self._create_error_result("No successful results available")
        
        if len(successful_results) == 1:
            # Only one successful result - return it with metadata
            model_name, result = next(iter(successful_results.items()))
            return self._create_single_result(model_name, result, start_time)
        
        # SIMPLIFIED: Apply combination strategy
        if strategy == EnsembleCombinationStrategy.CONFIDENCE_WEIGHTED:
            combined_result = self._confidence_weighted_combination(successful_results, query_context)
        elif strategy == EnsembleCombinationStrategy.MATHSTRAL_PRIMARY:
            combined_result = self._mathstral_primary_combination(successful_results)
        elif strategy == EnsembleCombinationStrategy.DEEPSEEK_PRIMARY:
            combined_result = self._deepseek_primary_combination(successful_results)
        elif strategy == EnsembleCombinationStrategy.CONSENSUS_BASED:
            combined_result = self._consensus_based_combination(successful_results)
        else:
            combined_result = self._best_response_combination(successful_results)
        
        # Add timing and metadata
        processing_time = (time.time() - start_time) * 1000
        combined_result["ensemble_processing_time_ms"] = processing_time
        combined_result["models_used"] = list(successful_results.keys())
        combined_result["strategy_used"] = strategy.value
        
        self.logger.info(f"Ensemble combination completed in {processing_time:.1f}ms")
        return combined_result

    def _confidence_weighted_combination(self, results: Dict[str, Dict[str, Any]], query_context: Dict[str, Any]) -> Dict[str, Any]:
        """SIMPLIFIED: Combine based on confidence scores"""
        
        weighted_scores = {}
        query_type = query_context.get("query_type", "general")
        
        for model_name, result in results.items():
            # Base confidence from model
            base_confidence = result.get("confidence", result.get("confidence_score", 0.5))
            
            # Specialization boost
            specialization_boost = 0.0
            if model_name in self.model_preferences:
                model_prefs = self.model_preferences[model_name]
                specialization_boost = model_prefs.get(query_type, 0.7) * 0.1
            
            # SQL quality boost (simple heuristics)
            sql = result.get("sql", "")
            quality_boost = self._calculate_sql_quality_score(sql) * 0.1
            
            # Final weighted score
            weighted_score = base_confidence + specialization_boost + quality_boost
            weighted_scores[model_name] = weighted_score
            
            self.logger.debug(f"{model_name} weighted score: {weighted_score:.3f}")
        
        # Select best model
        best_model = max(weighted_scores.keys(), key=lambda k: weighted_scores[k])
        best_result = results[best_model].copy()
        
        # Add ensemble metadata
        best_result.update({
            "selected_model": best_model,
            "selection_reason": "highest_weighted_confidence",
            "weighted_score": weighted_scores[best_model],
            "all_confidences": {model: results[model].get("confidence", 0.5) for model in results},
            "ensemble_strategy": "confidence_weighted"
        })
        
        return best_result

    def _mathstral_primary_combination(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """SIMPLIFIED: Prefer Mathstral if available and good enough"""
        
        # Check if Mathstral result is available
        if "mathstral" in results:
            mathstral_result = results["mathstral"]
            mathstral_confidence = mathstral_result.get("confidence", mathstral_result.get("confidence_score", 0))
            
            # Use Mathstral if it meets minimum threshold
            if mathstral_confidence >= self.min_confidence_threshold:
                result = mathstral_result.copy()
                result.update({
                    "selected_model": "mathstral",
                    "selection_reason": "mathstral_primary_preference",
                    "ensemble_strategy": "mathstral_primary"
                })
                return result
        
        # Fallback to confidence weighted
        return self._confidence_weighted_combination(results, {"query_type": "analytical"})

    def _deepseek_primary_combination(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """SIMPLIFIED: Prefer DeepSeek if available and good enough"""
        
        # Check if DeepSeek result is available
        if "deepseek" in results:
            deepseek_result = results["deepseek"]
            deepseek_confidence = deepseek_result.get("confidence", deepseek_result.get("confidence_score", 0))
            
            # Use DeepSeek if it meets minimum threshold
            if deepseek_confidence >= self.min_confidence_threshold:
                result = deepseek_result.copy()
                result.update({
                    "selected_model": "deepseek",
                    "selection_reason": "deepseek_primary_preference", 
                    "ensemble_strategy": "deepseek_primary"
                })
                return result
        
        # Fallback to confidence weighted
        return self._confidence_weighted_combination(results, {"query_type": "general"})

    def _consensus_based_combination(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """SIMPLIFIED: Check for consensus between models"""
        
        if len(results) < 2:
            return self._confidence_weighted_combination(results, {})
        
        # Calculate SQL similarity
        sql_queries = [result.get("sql", "").strip() for result in results.values()]
        consensus_score = self._calculate_consensus_score(sql_queries)
        
        if consensus_score >= self.consensus_threshold:
            # High consensus - pick the one with better confidence
            best_model = max(results.keys(), 
                           key=lambda k: results[k].get("confidence", results[k].get("confidence_score", 0)))
            
            result = results[best_model].copy()
            result.update({
                "selected_model": best_model,
                "selection_reason": f"high_consensus_{consensus_score:.2f}",
                "consensus_score": consensus_score,
                "ensemble_strategy": "consensus_based"
            })
            return result
        else:
            # Low consensus - fall back to confidence weighted
            result = self._confidence_weighted_combination(results, {})
            result["selection_reason"] = f"low_consensus_{consensus_score:.2f}_fallback"
            return result

    def _best_response_combination(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """SIMPLIFIED: Select the objectively best response"""
        
        scored_results = {}
        
        for model_name, result in results.items():
            # Calculate composite score
            confidence = result.get("confidence", result.get("confidence_score", 0.5))
            sql_quality = self._calculate_sql_quality_score(result.get("sql", ""))
            
            composite_score = (confidence * 0.7) + (sql_quality * 0.3)
            scored_results[model_name] = composite_score
        
        # Select best model
        best_model = max(scored_results.keys(), key=lambda k: scored_results[k])
        result = results[best_model].copy()
        
        result.update({
            "selected_model": best_model,
            "selection_reason": "best_composite_score",
            "composite_score": scored_results[best_model],
            "ensemble_strategy": "best_response"
        })
        
        return result

    def _calculate_consensus_score(self, sql_queries: List[str]) -> float:
        """SIMPLIFIED: Calculate consensus between SQL queries"""
        
        if len(sql_queries) < 2:
            return 1.0
        
        # Normalize queries for comparison
        normalized = [self._normalize_sql(sql) for sql in sql_queries]
        
        # Check for exact matches
        if len(set(normalized)) == 1:
            return 1.0
        
        # Calculate similarity (simple token-based)
        total_comparisons = 0
        similar_comparisons = 0
        
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                total_comparisons += 1
                similarity = self._calculate_sql_similarity(normalized[i], normalized[j])
                if similarity > 0.8:
                    similar_comparisons += 1
        
        return similar_comparisons / total_comparisons if total_comparisons > 0 else 0.0

    def _calculate_sql_similarity(self, sql1: str, sql2: str) -> float:
        """SIMPLIFIED: Calculate similarity between two SQL queries"""
        
        if sql1 == sql2:
            return 1.0
        
        # Token-based similarity
        tokens1 = set(sql1.split())
        tokens2 = set(sql2.split())
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)

    def _calculate_sql_quality_score(self, sql: str) -> float:
        """SIMPLIFIED: Basic SQL quality assessment"""
        
        if not sql.strip():
            return 0.0
        
        quality_score = 0.5  # Base score
        sql_upper = sql.upper()
        
        # Must have basic SQL structure
        if any(keyword in sql_upper for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
            quality_score += 0.2
        
        # Check for proper FROM clause in SELECT
        if 'SELECT' in sql_upper and 'FROM' in sql_upper:
            quality_score += 0.1
        
        # Check for proper semicolon ending
        if sql.strip().endswith(';'):
            quality_score += 0.1
        
        # Check for balanced parentheses
        if sql.count('(') == sql.count(')'):
            quality_score += 0.1
        
        return min(quality_score, 1.0)

    def _normalize_sql(self, sql: str) -> str:
        """SIMPLIFIED: Normalize SQL for comparison"""
        import re
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', sql.strip())
        
        # Remove comments
        normalized = re.sub(r'--.*', '', normalized)
        
        # Normalize case and remove trailing semicolon
        normalized = normalized.upper().rstrip(';')
        
        return normalized

    def _create_single_result(self, model_name: str, result: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Create result for single successful model"""
        
        single_result = result.copy()
        single_result.update({
            "selected_model": model_name,
            "selection_reason": "only_successful_model",
            "ensemble_strategy": "single_model",
            "ensemble_processing_time_ms": (time.time() - start_time) * 1000
        })
        
        return single_result

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result when no models succeed"""
        
        return {
            "sql": "-- Error: No valid SQL could be generated",
            "generated_sql": "-- Error: No valid SQL could be generated", 
            "success": False,
            "confidence": 0.0,
            "confidence_score": 0.0,
            "selected_model": "none",
            "selection_reason": "all_models_failed",
            "error": error_message,
            "ensemble_strategy": "error_fallback"
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """SIMPLIFIED: Get basic performance statistics"""
        return {
            "ensemble_manager_status": "active",
            "min_confidence_threshold": self.min_confidence_threshold,
            "consensus_threshold": self.consensus_threshold,
            "supported_strategies": [strategy.value for strategy in EnsembleCombinationStrategy]
        }

# Export the main class
__all__ = ["EnsembleManager", "EnsembleCombinationStrategy"]

# SIMPLIFIED: Test function
async def test_ensemble_manager():
    """Test the ensemble manager with sample results"""
    manager = EnsembleManager()
    
    # Sample results from simplified clients
    results = {
        "mathstral": {
            "sql": "SELECT AVG(amount), STDDEV(amount) FROM transactions WHERE date > '2023-01-01';",
            "success": True,
            "confidence": 0.85,
            "model_used": "mathstral-7b-v0.1"
        },
        "deepseek": {
            "sql": "SELECT AVG(amount), STDDEV(amount) FROM transactions WHERE date > '2023-01-01';", 
            "success": True,
            "confidence": 0.80,
            "model_used": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        }
    }
    
    query_context = {"query_type": "analytical"}
    
    # Test ensemble combination
    ensemble_result = await manager.combine_results(
        results, query_context, EnsembleCombinationStrategy.CONFIDENCE_WEIGHTED
    )
    
    print("✅ Ensemble Result:")
    print(f"   Selected Model: {ensemble_result['selected_model']}")
    print(f"   Final SQL: {ensemble_result['sql'][:50]}...")
    print(f"   Final Confidence: {ensemble_result.get('confidence', 0):.2f}")
    print(f"   Strategy: {ensemble_result['ensemble_strategy']}")
    print(f"   Reason: {ensemble_result['selection_reason']}")
    
    return ensemble_result

if __name__ == "__main__":
    asyncio.run(test_ensemble_manager())
