"""
Core Metrics Tracking for NLP Processor
Performance and accuracy metrics for banking domain NLP processing
Provides comprehensive tracking for all system components
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import statistics
import threading
from contextlib import contextmanager

from .exceptions import NLPProcessorBaseException, ErrorCategory


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked by the system"""
    PROCESSING_TIME = "processing_time"
    ACCURACY = "accuracy"
    SUCCESS_RATE = "success_rate"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    ERROR_RATE = "error_rate"
    CONFIDENCE_SCORE = "confidence_score"


class ComponentType(Enum):
    """System components being tracked"""
    INTENT_CLASSIFIER = "intent_classifier"
    ENTITY_EXTRACTOR = "entity_extractor"
    QUERY_ANALYZER = "query_analyzer"
    TEMPORAL_PROCESSOR = "temporal_processor"
    DOMAIN_MAPPER = "domain_mapper"
    SCHEMA_SEARCHER = "schema_searcher"
    XML_MANAGER = "xml_manager"
    PROMPT_BUILDER = "prompt_builder"
    NLP_ORCHESTRATOR = "nlp_orchestrator"
    OVERALL_SYSTEM = "overall_system"


@dataclass
class MetricEntry:
    """Individual metric entry"""
    component: ComponentType
    metric_type: MetricType
    value: Union[float, int, bool]
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None


@dataclass
class ProcessingMetrics:
    """Metrics for a single processing operation"""
    start_time: datetime
    end_time: Optional[datetime] = None
    processing_time_ms: Optional[float] = None
    success: bool = True
    error_code: Optional[str] = None
    component_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for NLP components"""
    total_predictions: int = 0
    correct_predictions: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    confidence_scores: List[float] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Performance metrics aggregation"""
    avg_processing_time_ms: float = 0.0
    min_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0
    p95_processing_time_ms: float = 0.0
    p99_processing_time_ms: float = 0.0
    throughput_per_minute: float = 0.0
    success_rate: float = 0.0
    error_rate: float = 0.0


class MetricsCollector:
    """
    Main metrics collection and aggregation system
    Thread-safe metrics tracking for all NLP components
    """
    
    def __init__(self, retention_hours: int = 24, max_entries: int = 10000):
        """Initialize metrics collector"""
        self.retention_hours = retention_hours
        self.max_entries = max_entries
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Metrics storage
        self.metrics: Dict[ComponentType, List[MetricEntry]] = defaultdict(list)
        self.processing_sessions: Dict[str, ProcessingMetrics] = {}
        
        # Performance tracking
        self.processing_times: Dict[ComponentType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.success_counts: Dict[ComponentType, int] = defaultdict(int)
        self.error_counts: Dict[ComponentType, int] = defaultdict(int)
        self.total_requests: Dict[ComponentType, int] = defaultdict(int)
        
        # Accuracy tracking
        self.accuracy_metrics: Dict[ComponentType, AccuracyMetrics] = defaultdict(AccuracyMetrics)
        
        # Business query metrics
        self.query_patterns: Dict[str, int] = defaultdict(int)
        self.analyst_usage: Dict[str, int] = defaultdict(int)
        
        # Schema integration metrics
        self.schema_hit_rate: float = 0.0
        self.xml_field_usage: Dict[str, int] = defaultdict(int)
        self.join_optimization_success: int = 0
        
        logger.info("MetricsCollector initialized")
    
    @contextmanager
    def track_processing_time(self, component: ComponentType, operation: str = "process", session_id: Optional[str] = None):
        """Context manager to track processing time"""
        start_time = time.time()
        start_datetime = datetime.now()
        
        # Create processing session
        if session_id:
            with self._lock:
                self.processing_sessions[session_id] = ProcessingMetrics(start_time=start_datetime)
        
        try:
            yield
            success = True
            error_code = None
        except Exception as e:
            success = False
            error_code = getattr(e, 'error_code', type(e).__name__)
            raise
        finally:
            end_time = time.time()
            end_datetime = datetime.now()
            processing_time_ms = (end_time - start_time) * 1000
            
            # Record metrics
            self.record_processing_time(component, processing_time_ms, success, error_code) # type: ignore
            
            # Update session
            if session_id and session_id in self.processing_sessions:
                with self._lock:
                    session = self.processing_sessions[session_id]
                    session.end_time = end_datetime
                    session.processing_time_ms = processing_time_ms
                    session.success = success # type: ignore
                    session.error_code = error_code # type: ignore
    
    def record_processing_time(self, component: ComponentType, time_ms: float, success: bool = True, error_code: Optional[str] = None) -> None:
        """Record processing time for a component"""
        with self._lock:
            # Add to metrics
            metric = MetricEntry(
                component=component,
                metric_type=MetricType.PROCESSING_TIME,
                value=time_ms,
                timestamp=datetime.now(),
                context={"success": success, "error_code": error_code}
            )
            self.metrics[component].append(metric)
            
            # Update performance tracking
            self.processing_times[component].append(time_ms)
            self.total_requests[component] += 1
            
            if success:
                self.success_counts[component] += 1
            else:
                self.error_counts[component] += 1
            
            # Cleanup old metrics
            self._cleanup_old_metrics()
    
    def record_accuracy(self, component: ComponentType, is_correct: bool, confidence_score: float, prediction_type: str = "classification") -> None:
        """Record accuracy metrics for NLP components"""
        with self._lock:
            accuracy = self.accuracy_metrics[component]
            accuracy.total_predictions += 1
            
            if is_correct:
                accuracy.correct_predictions += 1
            
            accuracy.confidence_scores.append(confidence_score)
            
            # Calculate derived metrics
            if accuracy.total_predictions > 0:
                accuracy.precision = accuracy.correct_predictions / accuracy.total_predictions
                accuracy.recall = accuracy.precision  # Simplified for classification tasks
                accuracy.f1_score = 2 * (accuracy.precision * accuracy.recall) / (accuracy.precision + accuracy.recall) if (accuracy.precision + accuracy.recall) > 0 else 0
            
            # Record metric entry
            metric = MetricEntry(
                component=component,
                metric_type=MetricType.ACCURACY,
                value=accuracy.precision,
                timestamp=datetime.now(),
                context={
                    "prediction_type": prediction_type,
                    "confidence_score": confidence_score,
                    "is_correct": is_correct
                }
            )
            self.metrics[component].append(metric)
    
    def record_intent_classification(self, intent: str, confidence: float, is_correct: bool, query_text: str) -> None:
        """Record intent classification metrics"""
        self.record_accuracy(ComponentType.INTENT_CLASSIFIER, is_correct, confidence, "intent_classification")
        
        with self._lock:
            # Track query patterns
            self.query_patterns[intent] += 1
            
            # Add specific context
            metric = MetricEntry(
                component=ComponentType.INTENT_CLASSIFIER,
                metric_type=MetricType.CONFIDENCE_SCORE,
                value=confidence,
                timestamp=datetime.now(),
                context={
                    "intent": intent,
                    "query_length": len(query_text),
                    "is_correct": is_correct
                }
            )
            self.metrics[ComponentType.INTENT_CLASSIFIER].append(metric)
    
    def record_entity_extraction(self, entities_found: int, entities_expected: int, confidence_scores: List[float], entity_types: List[str]) -> None:
        """Record entity extraction metrics"""
        precision = entities_found / max(entities_expected, 1) if entities_expected > 0 else 0
        avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
        
        self.record_accuracy(ComponentType.ENTITY_EXTRACTOR, precision == 1.0, avg_confidence, "entity_extraction")
        
        with self._lock:
            metric = MetricEntry(
                component=ComponentType.ENTITY_EXTRACTOR,
                metric_type=MetricType.ACCURACY,
                value=precision,
                timestamp=datetime.now(),
                context={
                    "entities_found": entities_found,
                    "entities_expected": entities_expected,
                    "entity_types": entity_types,
                    "confidence_scores": confidence_scores
                }
            )
            self.metrics[ComponentType.ENTITY_EXTRACTOR].append(metric)
    
    def record_schema_integration(self, tables_found: int, joins_used: int, xml_fields_accessed: int, search_success: bool) -> None:
        """Record schema searcher integration metrics"""
        with self._lock:
            # Update schema hit rate
            total_schema_requests = self.total_requests[ComponentType.SCHEMA_SEARCHER]
            if search_success:
                self.success_counts[ComponentType.SCHEMA_SEARCHER] += 1
                if joins_used > 0:
                    self.join_optimization_success += 1
            
            self.total_requests[ComponentType.SCHEMA_SEARCHER] += 1
            
            if total_schema_requests > 0:
                self.schema_hit_rate = self.success_counts[ComponentType.SCHEMA_SEARCHER] / total_schema_requests
            
            # Record detailed metrics
            metric = MetricEntry(
                component=ComponentType.SCHEMA_SEARCHER,
                metric_type=MetricType.SUCCESS_RATE,
                value=1.0 if search_success else 0.0,
                timestamp=datetime.now(),
                context={
                    "tables_found": tables_found,
                    "joins_used": joins_used,
                    "xml_fields_accessed": xml_fields_accessed,
                    "search_success": search_success
                }
            )
            self.metrics[ComponentType.SCHEMA_SEARCHER].append(metric)
    
    def record_business_query(self, query_type: str, analyst_id: str, complexity_score: float, success: bool) -> None:
        """Record business query metrics"""
        with self._lock:
            # Track analyst usage
            self.analyst_usage[analyst_id] += 1
            
            # Track query patterns
            self.query_patterns[query_type] += 1
            
            metric = MetricEntry(
                component=ComponentType.OVERALL_SYSTEM,
                metric_type=MetricType.SUCCESS_RATE,
                value=1.0 if success else 0.0,
                timestamp=datetime.now(),
                context={
                    "query_type": query_type,
                    "analyst_id": analyst_id,
                    "complexity_score": complexity_score,
                    "success": success
                }
            )
            self.metrics[ComponentType.OVERALL_SYSTEM].append(metric)
    
    def record_xml_field_usage(self, xml_field: str, access_time_ms: float, success: bool) -> None:
        """Record XML field usage metrics"""
        with self._lock:
            if success:
                self.xml_field_usage[xml_field] += 1
            
            metric = MetricEntry(
                component=ComponentType.XML_MANAGER,
                metric_type=MetricType.PROCESSING_TIME,
                value=access_time_ms,
                timestamp=datetime.now(),
                context={
                    "xml_field": xml_field,
                    "success": success
                }
            )
            self.metrics[ComponentType.XML_MANAGER].append(metric)
    
    def get_performance_metrics(self, component: ComponentType, time_window_hours: int = 1) -> PerformanceMetrics:
        """Get performance metrics for a component"""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            # Filter recent metrics
            recent_times = []
            recent_successes = 0
            recent_errors = 0
            
            for metric in self.metrics[component]:
                if metric.timestamp >= cutoff_time and metric.metric_type == MetricType.PROCESSING_TIME:
                    recent_times.append(metric.value)
                    if metric.context.get("success", True):
                        recent_successes += 1
                    else:
                        recent_errors += 1
            
            if not recent_times:
                return PerformanceMetrics()
            
            # Calculate performance metrics
            recent_times.sort()
            total_requests = recent_successes + recent_errors
            
            performance = PerformanceMetrics(
                avg_processing_time_ms=statistics.mean(recent_times),
                min_processing_time_ms=min(recent_times),
                max_processing_time_ms=max(recent_times),
                p95_processing_time_ms=recent_times[int(0.95 * len(recent_times))] if len(recent_times) > 0 else 0,
                p99_processing_time_ms=recent_times[int(0.99 * len(recent_times))] if len(recent_times) > 0 else 0,
                throughput_per_minute=total_requests * (60 / time_window_hours),
                success_rate=(recent_successes / total_requests * 100) if total_requests > 0 else 0,
                error_rate=(recent_errors / total_requests * 100) if total_requests > 0 else 0
            )
            
            return performance
    
    def get_accuracy_metrics(self, component: ComponentType) -> AccuracyMetrics:
        """Get accuracy metrics for a component"""
        with self._lock:
            return self.accuracy_metrics[component]
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system metrics overview"""
        with self._lock:
            overview = {
                "timestamp": datetime.now().isoformat(),
                "components": {},
                "business_metrics": {
                    "total_queries_processed": sum(self.total_requests.values()),
                    "overall_success_rate": self._calculate_overall_success_rate(),
                    "schema_hit_rate": self.schema_hit_rate,
                    "top_query_patterns": dict(sorted(self.query_patterns.items(), key=lambda x: x[1], reverse=True)[:10]),
                    "active_analysts": len(self.analyst_usage),
                    "join_optimization_success": self.join_optimization_success
                },
                "integration_metrics": {
                    "xml_fields_accessed": len(self.xml_field_usage),
                    "total_xml_requests": self.total_requests[ComponentType.XML_MANAGER],
                    "schema_integration_success": self.success_counts[ComponentType.SCHEMA_SEARCHER],
                    "prompt_builder_success": self.success_counts[ComponentType.PROMPT_BUILDER]
                }
            }
            
            # Add component-specific metrics
            for component in ComponentType:
                if component in self.metrics and self.metrics[component]:
                    performance = self.get_performance_metrics(component, 1)
                    accuracy = self.get_accuracy_metrics(component)
                    
                    overview["components"][component.value] = {
                        "performance": {
                            "avg_processing_time_ms": performance.avg_processing_time_ms,
                            "success_rate": performance.success_rate,
                            "throughput_per_minute": performance.throughput_per_minute
                        },
                        "accuracy": {
                            "precision": accuracy.precision,
                            "total_predictions": accuracy.total_predictions,
                            "avg_confidence": statistics.mean(accuracy.confidence_scores) if accuracy.confidence_scores else 0
                        },
                        "total_requests": self.total_requests[component]
                    }
            
            return overview
    
    def get_analyst_metrics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get analyst-specific usage metrics"""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            analyst_queries = defaultdict(int)
            analyst_success = defaultdict(int)
            query_complexity = defaultdict(list)
            
            for metric in self.metrics[ComponentType.OVERALL_SYSTEM]:
                if metric.timestamp >= cutoff_time and "analyst_id" in metric.context:
                    analyst_id = metric.context["analyst_id"]
                    analyst_queries[analyst_id] += 1
                    
                    if metric.context.get("success", False):
                        analyst_success[analyst_id] += 1
                    
                    if "complexity_score" in metric.context:
                        query_complexity[analyst_id].append(metric.context["complexity_score"])
            
            return {
                "time_window_hours": time_window_hours,
                "analysts": {
                    analyst_id: {
                        "total_queries": analyst_queries[analyst_id],
                        "successful_queries": analyst_success[analyst_id],
                        "success_rate": (analyst_success[analyst_id] / analyst_queries[analyst_id] * 100) if analyst_queries[analyst_id] > 0 else 0,
                        "avg_query_complexity": statistics.mean(query_complexity[analyst_id]) if query_complexity[analyst_id] else 0
                    }
                    for analyst_id in analyst_queries
                },
                "total_active_analysts": len(analyst_queries)
            }
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall system success rate"""
        total_success = sum(self.success_counts.values())
        total_requests = sum(self.total_requests.values())
        return (total_success / total_requests * 100) if total_requests > 0 else 0
    
    def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics based on retention policy"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        for component in self.metrics:
            self.metrics[component] = [
                metric for metric in self.metrics[component]
                if metric.timestamp >= cutoff_time
            ]
            
            # Limit total entries
            if len(self.metrics[component]) > self.max_entries:
                self.metrics[component] = self.metrics[component][-self.max_entries:]
    
    def export_metrics(self, format_type: str = "json") -> Dict[str, Any]:
        """Export metrics for external analysis"""
        with self._lock:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "retention_hours": self.retention_hours,
                "system_overview": self.get_system_overview(),
                "analyst_metrics": self.get_analyst_metrics(),
                "detailed_metrics": {}
            }
            
            # Add detailed component metrics
            for component in ComponentType:
                if component in self.metrics:
                    export_data["detailed_metrics"][component.value] = {
                        "performance": self.get_performance_metrics(component).__dict__,
                        "accuracy": self.get_accuracy_metrics(component).__dict__,
                        "recent_entries": [
                            {
                                "metric_type": metric.metric_type.value,
                                "value": metric.value,
                                "timestamp": metric.timestamp.isoformat(),
                                "context": metric.context
                            }
                            for metric in self.metrics[component][-50:]  # Last 50 entries
                        ]
                    }
            
            return export_data
    
    def reset_metrics(self, component: Optional[ComponentType] = None) -> None:
        """Reset metrics for a component or all components"""
        with self._lock:
            if component:
                self.metrics[component].clear()
                self.processing_times[component].clear()
                self.success_counts[component] = 0
                self.error_counts[component] = 0
                self.total_requests[component] = 0
                self.accuracy_metrics[component] = AccuracyMetrics()
            else:
                self.metrics.clear()
                self.processing_times.clear()
                self.success_counts.clear()
                self.error_counts.clear()
                self.total_requests.clear()
                self.accuracy_metrics.clear()
                self.query_patterns.clear()
                self.analyst_usage.clear()
                self.xml_field_usage.clear()
                self.schema_hit_rate = 0.0
                self.join_optimization_success = 0
        
        logger.info(f"Reset metrics for {'all components' if not component else component.value}")


# Global metrics collector instance
metrics_collector = MetricsCollector()


# Convenience functions for easy integration
def track_processing_time(component: ComponentType, operation: str = "process", session_id: Optional[str] = None):
    """Convenience function to track processing time"""
    return metrics_collector.track_processing_time(component, operation, session_id)


def record_intent_classification(intent: str, confidence: float, is_correct: bool, query_text: str) -> None:
    """Convenience function to record intent classification metrics"""
    metrics_collector.record_intent_classification(intent, confidence, is_correct, query_text)


def record_entity_extraction(entities_found: int, entities_expected: int, confidence_scores: List[float], entity_types: List[str]) -> None:
    """Convenience function to record entity extraction metrics"""
    metrics_collector.record_entity_extraction(entities_found, entities_expected, confidence_scores, entity_types)


def record_schema_integration(tables_found: int, joins_used: int, xml_fields_accessed: int, search_success: bool) -> None:
    """Convenience function to record schema integration metrics"""
    metrics_collector.record_schema_integration(tables_found, joins_used, xml_fields_accessed, search_success)


def get_system_metrics() -> Dict[str, Any]:
    """Convenience function to get system metrics overview"""
    return metrics_collector.get_system_overview()
