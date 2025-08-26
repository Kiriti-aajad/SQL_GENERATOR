# agent/schema_searcher/utils/performance.py
"""
Performance tracking and diagnostics for the schema retrieval system.

Provides decorators and classes to record query durations, engine performance,
and system usage stats.
"""

import time
import threading
from typing import Callable, Dict, Optional
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Tracks performance metrics for query executions and engine searches.
    Thread-safe for concurrent access in API environments.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.total_queries: int = 0
        self.total_errors: int = 0
        self.cumulative_time: float = 0.0
        self.query_durations: Dict[str, float] = {}
        self.last_error: Optional[str] = None

    def record_query(self, query: str, duration_s: float, result_count: int = 0) -> None:
        """Stores performance metrics for a completed query."""
        with self._lock:
            self.total_queries += 1
            self.cumulative_time += duration_s
            self.query_durations[query] = duration_s
            logger.debug(f"[Perf] Query recorded: {query} took {duration_s:.3f}s ({result_count} results)")

    def record_error(self, query: str, error: str) -> None:
        """Logs a failed query attempt."""
        with self._lock:
            self.total_errors += 1
            self.last_error = error
            logger.warning(f"[Perf] Query error for '{query}': {error}")

    def get_stats(self) -> Dict[str, float]:
        """Returns summary performance stats."""
        with self._lock:
            avg_time = self.cumulative_time / self.total_queries if self.total_queries else 0.0
            return {
                "total_queries": self.total_queries,
                "total_errors": self.total_errors,
                "average_query_time_s": round(avg_time, 3),
                "last_error": self.last_error or "",
            }

    def reset_stats(self) -> None:
        """Clears metrics for a fresh session."""
        with self._lock:
            self.total_queries = 0
            self.total_errors = 0
            self.cumulative_time = 0.0
            self.query_durations.clear()
            self.last_error = None


def track_execution_time(func: Callable) -> Callable:
    """
    Decorator to time a function’s execution and log it.

    Usage:
    @track_execution_time
    def my_function(...):
        ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = None
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end = time.perf_counter()
            duration = round(end - start, 4)
            logger.debug(f"[Perf] {func.__name__} executed in {duration:.4f}s")
    return wrapper
