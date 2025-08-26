"""
Abstract base class for all schema retrieval engines.

Defines the common interface that every search engine (semantic, bm25, faiss, etc.)
must implement. Provides standardized init, search, cleanup, and health check methods.

ENHANCED: Added optional methods for Mathstral reasoning integration.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from logging import Logger # type: ignore
import time
from datetime import datetime

from agent.schema_searcher.core.data_models import RetrievedColumn, SearchMethod
from agent.schema_searcher.utils.validators import validate_query
from agent.schema_searcher.utils.performance import track_execution_time


class BaseSearchEngine(ABC):
    """
    Abstract base class for schema search engines.
    
    Subclasses must implement:
    - initialize()
    - search(query: str, top_k: int) -> List[RetrievedColumn]
    - cleanup()
    
    Optional methods for reasoning enhancement:
    - search_with_strategy() (for orchestrated searches)
    - set_search_parameters() (for dynamic parameter adjustment)
    - adjust_search_parameters() (for real-time optimization)
    - get_search_suggestions() (for feedback to orchestrator)
    """

    def __init__(self, method: SearchMethod, logger: Logger):
        self.method: SearchMethod = method
        self.logger: Logger = logger
        self._initialized: bool = False
        self._total_searches: int = 0
        self._successful_searches: int = 0
        self._last_error: Optional[str] = None
        self._last_execution_time: float = 0.0
        
        # NEW: Strategy and reasoning support
        self._strategy_parameters: Dict[str, Any] = {}
        self._reasoning_mode_enabled: bool = False

    @abstractmethod
    def initialize(self) -> None:
        """Setup engine resources like models, indexes, etc."""
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[RetrievedColumn]:
        """
        Execute search against engine with query.
        Must return a list of RetrievedColumn objects.
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Free resources, close clients/files."""
        ...

    # ===== NEW METHODS FOR MATHSTRAL REASONING INTEGRATION =====
    # These are OPTIONAL - engines can override them for enhanced functionality
    
    def search_with_strategy(self, strategy: Dict[str, Any], target_gaps: List[str]) -> List[RetrievedColumn]:
        """
        NEW: Execute searches with dynamically adjusted parameters based on strategy.
        
        Default implementation falls back to regular search, but engines can override
        for gap-aware, strategy-optimized searching.
        
        Parameters:
        -----------
        strategy : Dict[str, Any]
            Search strategy from orchestration layer
        target_gaps : List[str]
            Specific gaps this search should address
            
        Returns:
        --------
        List[RetrievedColumn]
            Search results optimized for target gaps
        """
        # Default fallback: use regular search
        query = strategy.get('primary_query', strategy.get('keywords', [''])[0] if strategy.get('keywords') else '')
        top_k = strategy.get('max_results', 10)
        
        self.logger.debug(f"[{self.method.value}] Using fallback search_with_strategy")
        return self.run_search(query, top_k)
    
    def set_search_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        NEW: Set engine-specific parameters for targeted searches.
        
        Called by orchestration layer to configure engine behavior
        for specific search strategies.
        
        Parameters:
        -----------
        parameters : Dict[str, Any]
            Engine-specific parameters from strategy
        """
        self._strategy_parameters.update(parameters)
        self._reasoning_mode_enabled = True
        self.logger.debug(f"[{self.method.value}] Strategy parameters updated: {list(parameters.keys())}")
    
    def adjust_search_parameters(self, current_results: List[RetrievedColumn], target_criteria: Dict[str, Any]) -> None:
        """
        NEW: Fine-tune search parameters based on result quality and target criteria.
        
        Called during iterative search to optimize parameters based on
        intermediate results.
        
        Parameters:
        -----------
        current_results : List[RetrievedColumn]
            Results from previous search iteration
        target_criteria : Dict[str, Any]
            Quality criteria and optimization targets
        """
        # Default implementation: basic parameter adjustment
        if len(current_results) < target_criteria.get('min_results', 5):
            # Too few results - relax constraints
            if 'similarity_threshold' in self._strategy_parameters:
                current_threshold = self._strategy_parameters['similarity_threshold']
                self._strategy_parameters['similarity_threshold'] = max(0.5, current_threshold * 0.9)
                self.logger.debug(f"[{self.method.value}] Relaxed similarity threshold to {self._strategy_parameters['similarity_threshold']}")
        
        elif len(current_results) > target_criteria.get('max_results', 50):
            # Too many results - tighten constraints  
            if 'similarity_threshold' in self._strategy_parameters:
                current_threshold = self._strategy_parameters['similarity_threshold']
                self._strategy_parameters['similarity_threshold'] = min(0.95, current_threshold * 1.1)
                self.logger.debug(f"[{self.method.value}] Tightened similarity threshold to {self._strategy_parameters['similarity_threshold']}")
    
    def get_search_suggestions(self) -> List[str]:
        """
        NEW: Provide feedback for search optimization.
        
        Returns suggestions for improving search effectiveness,
        used by orchestration layer for strategy refinement.
        
        Returns:
        --------
        List[str]
            Optimization suggestions
        """
        suggestions = []
        
        # Analyze performance and suggest improvements
        if self.success_rate() < 0.8:
            suggestions.append("Consider relaxing search constraints - low success rate detected")
        
        if self._last_execution_time > 10.0:
            suggestions.append("Consider reducing search scope - execution time high")
        
        if self._total_searches > 100 and self._successful_searches < 20:
            suggestions.append("Engine may not be suitable for current query types")
        
        return suggestions
    
    def get_strategy_parameters(self) -> Dict[str, Any]:
        """
        NEW: Get current strategy parameters.
        
        Returns:
        --------
        Dict[str, Any]
            Current strategy parameters
        """
        return self._strategy_parameters.copy()
    
    def reset_strategy_parameters(self) -> None:
        """
        NEW: Reset strategy parameters to defaults.
        
        Called when switching between reasoning and non-reasoning modes.
        """
        self._strategy_parameters.clear()
        self._reasoning_mode_enabled = False
        self.logger.debug(f"[{self.method.value}] Strategy parameters reset")
    
    def is_reasoning_enabled(self) -> bool:
        """
        NEW: Check if reasoning mode is enabled for this engine.
        
        Returns:
        --------
        bool
            True if reasoning enhancements are active
        """
        return self._reasoning_mode_enabled

    # ===== EXISTING METHODS (UNCHANGED) =====

    def ensure_initialized(self) -> None:
        """Ensure engine is ready to use"""
        if not self._initialized:
            try:
                self.initialize()
                self._initialized = True
                self.logger.info(f"[{self.method.value}] Engine initialized")
            except Exception as ex:
                self._last_error = str(ex)
                self.logger.error(f"[{self.method.value}] Initialization failed: {ex}")
                raise

    @track_execution_time
    def run_search(self, query: str, top_k: int = 10) -> List[RetrievedColumn]:
        """
        Public method that wraps around `search()` to:
        - ensure the engine is initialized
        - validate the query
        - track execution stats
        - handle and log exceptions
        """
        validate_query(query)
        self.ensure_initialized()

        self._total_searches += 1
        try:
            results = self.search(query, top_k=top_k)
            self._successful_searches += 1

            # FIX: Create new objects with correct retrieval_method instead of modifying frozen objects
            corrected_results = []
            for r in results:
                # Create new RetrievedColumn with correct method
                corrected_result = RetrievedColumn(
                    table=r.table,
                    column=r.column,
                    datatype=r.datatype,
                    type=r.type,
                    description=r.description,
                    confidence_score=r.confidence_score,
                    retrieval_method=self.method,  # Set correct method here
                    xml_column=r.xml_column,
                    xpath=r.xpath,
                    sql_expression=r.sql_expression,
                    nullable=r.nullable,
                    primary_key=r.primary_key,
                    foreign_key=r.foreign_key,
                    retrieved_at=r.retrieved_at if hasattr(r, 'retrieved_at') else datetime.now()
                )
                corrected_results.append(corrected_result)

            return corrected_results

        except Exception as ex:
            self._last_error = str(ex)
            self.logger.error(f"[{self.method.value}] Search failed: {ex}")
            return []

    def health_check(self) -> Dict[str, Any]:
        """Report status and internal stats"""
        base_health = {
            "engine": self.method.value,
            "initialized": self._initialized,
            "available": self._initialized and self._last_error is None,
            "total_searches": self._total_searches,
            "success_rate": self.success_rate(),
            "last_error": self._last_error,
            "last_execution_time_secs": self._last_execution_time,
        }
        
        # NEW: Add reasoning-related health info
        if self._reasoning_mode_enabled:
            base_health.update({
                "reasoning_enabled": True,
                "strategy_parameters_count": len(self._strategy_parameters),
                "strategy_parameters": list(self._strategy_parameters.keys())
            })
        
        return base_health

    def success_rate(self) -> float:
        if self._total_searches == 0:
            return 0.0
        return round(self._successful_searches / self._total_searches, 4)

    def __enter__(self):
        self.ensure_initialized()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()
