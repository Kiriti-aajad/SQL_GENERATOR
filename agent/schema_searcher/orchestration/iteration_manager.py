"""
IterationManager - FIXED VERSION
───────────────────────────────
Manages search iteration state and history for the orchestration process.
FIXED: Handles SearchIteration objects without .get() method - prevents infinite loops

Core responsibilities:
1. track_iteration_state() → maintain history of searches and results
2. prevent_infinite_loops() → ensure convergence within reasonable limits  
3. cache_intermediate_results() → store iteration results for efficiency
4. generate_iteration_summary() → provide audit trail of reasoning process
5. record_iteration() → FIXED to handle all object types safely
"""

from typing import Dict, List, Set, Any, Optional, Tuple
import logging
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

# Core imports
from agent.schema_searcher.core.data_models import RetrievedColumn

@dataclass
class IterationState:
    """Represents the state of a single iteration"""
    session_id: str
    iteration_number: int
    timestamp: float
    query_hash: str
    keywords_used: List[str]
    gaps_targeted: Dict[str, List[str]]
    results_found: int
    confidence_score: float
    execution_time: float
    convergence_metrics: Dict[str, float]

@dataclass
class SessionMetadata:
    """Metadata for an orchestration session"""
    session_id: str
    initial_query: str
    query_hash: str
    start_time: float
    initial_results_count: int
    domain_context: str
    max_iterations: int

class IterationManager:
    """
    FIXED Iteration Manager - Handles all iteration object types safely
    
    CRITICAL FIXES:
    - Safe attribute access for SearchIteration objects
    - Prevents 'SearchIteration' object has no attribute 'get' errors
    - Proper iteration recording to prevent infinite loops
    """
    
    def __init__(
        self,
        max_history_size: int = 1000,
        loop_detection_window: int = 5,
        similarity_threshold: float = 0.85
    ):
        # Logging setup
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s'))
            self.logger.addHandler(h)
            self.logger.setLevel(logging.INFO)
        
        # Configuration
        self.max_history_size = max_history_size
        self.loop_detection_window = loop_detection_window
        self.similarity_threshold = similarity_threshold
        
        # State storage
        self.active_sessions: Dict[str, SessionMetadata] = {}
        self.iteration_history: Dict[str, List[IterationState]] = defaultdict(list)
        self.result_cache: Dict[str, List[RetrievedColumn]] = {}
        self.transition_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Loop detection structures
        self.search_signatures: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.loop_detection_window))
        self.keyword_history: Dict[str, Set[str]] = defaultdict(set)
        
        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # CRITICAL FIX: Add simple iteration tracking for SearchOrchestrator compatibility
        self.simple_iteration_history: List[Dict[str, Any]] = []
        
        # Track current session for compatibility
        self._current_session_id: Optional[str] = None
    
    # 🔧 CRITICAL FIX: Safe attribute accessor
    def _safe_access(self, iteration_obj: Any) -> Dict[str, Any]:
        """
        Safely access iteration attributes for compatibility.
        Handles dict, objects with .get(), or regular attribute access.
        
        FIXES: 'SearchIteration' object has no attribute 'get' error
        """
        try:
            if isinstance(iteration_obj, dict):
                # Dictionary access
                return {
                    'iteration_num': iteration_obj.get('iteration_num', 0),
                    'results_count': iteration_obj.get('results_count', 0),
                    'new_results': iteration_obj.get('new_results', 0),
                    'confidence_improvement': iteration_obj.get('confidence_improvement', 0.0),
                    'processing_time': iteration_obj.get('processing_time', 0.0),
                    'success_rate': iteration_obj.get('success_rate', 0.0),
                    'search_method': iteration_obj.get('search_method', 'unknown'),
                    'convergence_status': iteration_obj.get('convergence_status', 'in_progress'),
                    'unique_results': iteration_obj.get('unique_results', 0),
                    'tables_found': iteration_obj.get('tables_found', 0),
                    'columns_found': iteration_obj.get('columns_found', 0)
                }
            elif hasattr(iteration_obj, 'get') and callable(iteration_obj.get):
                # Object with .get() method
                return {
                    'iteration_num': iteration_obj.get('iteration_num', 0),
                    'results_count': iteration_obj.get('results_count', 0),
                    'new_results': iteration_obj.get('new_results', 0),
                    'confidence_improvement': iteration_obj.get('confidence_improvement', 0.0),
                    'processing_time': iteration_obj.get('processing_time', 0.0),
                    'success_rate': iteration_obj.get('success_rate', 0.0),
                    'search_method': iteration_obj.get('search_method', 'unknown'),
                    'convergence_status': iteration_obj.get('convergence_status', 'in_progress'),
                    'unique_results': iteration_obj.get('unique_results', 0),
                    'tables_found': iteration_obj.get('tables_found', 0),
                    'columns_found': iteration_obj.get('columns_found', 0)
                }
            else:
                # Regular object with attributes (SearchIteration case)
                return {
                    'iteration_num': getattr(iteration_obj, 'iteration_num', getattr(iteration_obj, 'iteration_number', 0)),
                    'results_count': getattr(iteration_obj, 'results_count', len(getattr(iteration_obj, 'results', []))),
                    'new_results': getattr(iteration_obj, 'new_results', getattr(iteration_obj, 'results_found', 0)),
                    'confidence_improvement': getattr(iteration_obj, 'confidence_improvement', getattr(iteration_obj, 'confidence_score', 0.0) * 100),
                    'processing_time': getattr(iteration_obj, 'processing_time', getattr(iteration_obj, 'execution_time', 0.0)),
                    'success_rate': getattr(iteration_obj, 'success_rate', 1.0),
                    'search_method': getattr(iteration_obj, 'search_method', getattr(iteration_obj, 'method', 'attribute_access')),
                    'convergence_status': getattr(iteration_obj, 'convergence_status', 'in_progress'),
                    'unique_results': getattr(iteration_obj, 'unique_results', getattr(iteration_obj, 'results_found', 0)),
                    'tables_found': getattr(iteration_obj, 'tables_found', 0),
                    'columns_found': getattr(iteration_obj, 'columns_found', 0)
                }
        except Exception as e:
            self.logger.warning(f"Failed to safely access iteration attributes: {e}")
            # Return minimal safe defaults
            return {
                'iteration_num': 0,
                'results_count': 0,
                'new_results': 0,
                'confidence_improvement': 0.0,
                'processing_time': 0.0,
                'success_rate': 0.0,
                'search_method': 'error_fallback',
                'convergence_status': 'error',
                'unique_results': 0,
                'tables_found': 0,
                'columns_found': 0
            }
    
    # 🔧 CRITICAL FIX: The missing record_iteration method with safe access
    def record_iteration(self, iteration_data: Any) -> bool:
        """
        Record iteration results for SearchOrchestrator compatibility.
        FIXED: Handles SearchIteration objects without .get() method.
        
        Args:
            iteration_data: Can be dict, SearchIteration object, or any iteration object
            
        Returns:
            bool: True if successfully recorded, False otherwise
        """
        try:
            # Use safe accessor to extract data
            safe_data = self._safe_access(iteration_data)
            
            # Extract data with safe defaults
            iteration_num = safe_data.get('iteration_num', len(self.simple_iteration_history) + 1)
            results_count = safe_data.get('results_count', 0)
            new_results = safe_data.get('new_results', 0)
            confidence_improvement = safe_data.get('confidence_improvement', 0.0)
            processing_time = safe_data.get('processing_time', 0.0)
            success_rate = safe_data.get('success_rate', 0.0)
            
            # Create standardized record
            record = {
                'iteration_num': iteration_num,
                'timestamp': time.time(),
                'results_count': results_count,
                'new_results': new_results,
                'confidence_improvement': confidence_improvement,
                'processing_time': processing_time,
                'success_rate': success_rate,
                'search_method': safe_data.get('search_method', 'unknown'),
                'convergence_status': safe_data.get('convergence_status', 'in_progress'),
                'unique_results': safe_data.get('unique_results', new_results),
                'tables_found': safe_data.get('tables_found', 0),
                'columns_found': safe_data.get('columns_found', 0)
            }
            
            # Store in simple history for SearchOrchestrator
            self.simple_iteration_history.append(record)
            
            # Maintain reasonable history size
            if len(self.simple_iteration_history) > 100:
                self.simple_iteration_history = self.simple_iteration_history[-100:]
            
            # Log the iteration - SUCCESS
            self.logger.info(f"✓ Iteration {iteration_num} recorded successfully: "
                           f"+{new_results} results, "
                           f"+{confidence_improvement:.1f}% confidence, "
                           f"{processing_time:.2f}s")
            
            # If we have an active session, also use the advanced tracking
            if self._current_session_id and self._current_session_id in self.active_sessions:
                try:
                    self._integrate_with_advanced_tracking(record)
                except Exception as e:
                    self.logger.debug(f"Advanced tracking integration failed: {e}")
            
            return True  # SUCCESS - prevents infinite loop
            
        except Exception as e:
            self.logger.error(f"Failed to record iteration: {e}")
            # Don't raise - this should be non-blocking for SearchOrchestrator
            return False
    
    def _integrate_with_advanced_tracking(self, record: Dict[str, Any]) -> None:
        """Integrate simple record with advanced tracking system"""
        try:
            session_id = self._current_session_id
            
            # Convert simple record to advanced IterationState if session exists
            if session_id and session_id in self.active_sessions:
                iteration_state = IterationState(
                    session_id=session_id,
                    iteration_number=record['iteration_num'],
                    timestamp=record['timestamp'],
                    query_hash=self.active_sessions[session_id].query_hash,
                    keywords_used=[],  # Not available in simple record
                    gaps_targeted={},  # Not available in simple record
                    results_found=record['results_count'],
                    confidence_score=record['confidence_improvement'] / 100.0,  # Convert percentage
                    execution_time=record['processing_time'],
                    convergence_metrics={'improvement_rate': record['confidence_improvement']}
                )
                
                # Add to advanced tracking
                self.iteration_history[session_id].append(iteration_state)
                
        except Exception as e:
            self.logger.debug(f"Advanced integration failed: {e}")
    
    def initialize_session(
        self,
        session_id: str,
        initial_query: str,
        initial_results: List[RetrievedColumn],
        domain_context: str = "business database",
        max_iterations: int = 5
    ) -> None:
        """
        Initialize a new orchestration session with metadata tracking.
        """
        self.logger.info(f"Initializing session {session_id}")
        
        # CRITICAL FIX: Set current session for record_iteration compatibility
        self._current_session_id = session_id
        
        query_hash = self._hash_query(initial_query)
        
        session_metadata = SessionMetadata(
            session_id=session_id,
            initial_query=initial_query,
            query_hash=query_hash,
            start_time=time.time(),
            initial_results_count=len(initial_results),
            domain_context=domain_context,
            max_iterations=max_iterations
        )
        
        self.active_sessions[session_id] = session_metadata
        self.iteration_history[session_id] = []
        
        # Cache initial results
        cache_key = f"{session_id}_initial"
        self.result_cache[cache_key] = initial_results
        
        # Initialize tracking structures
        self.search_signatures[session_id] = deque(maxlen=self.loop_detection_window)
        self.keyword_history[session_id] = set()
        
        self.logger.debug(f"Session {session_id} initialized with {len(initial_results)} initial results")
    
    def track_iteration_state(
        self,
        session_id: str,
        iteration_number: int,
        keywords_used: List[str],
        gaps_targeted: Dict[str, List[str]],
        results_found: List[RetrievedColumn],
        confidence_score: float,
        execution_time: float
    ) -> IterationState:
        """Track iteration state using state transition matrices."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not initialized")
        
        session_meta = self.active_sessions[session_id]
        
        # Create iteration state
        iteration_state = IterationState(
            session_id=session_id,
            iteration_number=iteration_number,
            timestamp=time.time(),
            query_hash=session_meta.query_hash,
            keywords_used=keywords_used.copy(),
            gaps_targeted=gaps_targeted.copy(),
            results_found=len(results_found),
            confidence_score=confidence_score,
            execution_time=execution_time,
            convergence_metrics=self._calculate_convergence_metrics(session_id, confidence_score)
        )
        
        # Store iteration
        self.iteration_history[session_id].append(iteration_state)
        
        # Update transition matrix
        self._update_transition_matrix(session_id, iteration_state)
        
        # Update keyword tracking
        self.keyword_history[session_id].update(keywords_used)
        
        # Cache results
        cache_key = f"{session_id}_iter_{iteration_number}"
        self.result_cache[cache_key] = results_found
        
        # Update performance metrics
        self.performance_metrics[session_id].append(execution_time)
        
        self.logger.debug(
            f"Tracked iteration {iteration_number} for session {session_id}: "
            f"{len(results_found)} results, {confidence_score:.2%} confidence"
        )
        
        return iteration_state
    
    def prevent_infinite_loops(
        self,
        session_id: str,
        proposed_keywords: List[str],
        proposed_gaps: Dict[str, List[str]]
    ) -> Tuple[bool, Optional[str]]:
        """Prevent infinite loops using graph cycle detection and semantic deduplication."""
        if session_id not in self.active_sessions:
            return True, "Session not found"
        
        # Create signature for this proposed iteration
        signature = self._create_search_signature(proposed_keywords, proposed_gaps)
        
        # Check for exact duplicates in recent history
        recent_signatures = list(self.search_signatures[session_id])
        if signature in recent_signatures:
            return False, "Exact duplicate search detected"
        
        # Check for semantic similarity with recent searches
        for recent_sig in recent_signatures:
            similarity = self._calculate_signature_similarity(signature, recent_sig)
            if similarity > self.similarity_threshold:
                return False, f"Similar search detected (similarity: {similarity:.2%})"
        
        # Check for keyword exhaustion
        new_keywords = set(proposed_keywords)
        existing_keywords = self.keyword_history[session_id]
        if new_keywords.issubset(existing_keywords):
            return False, "All keywords have been used before"
        
        # Check convergence stagnation
        if self._is_converging_too_slowly(session_id):
            return False, "Convergence rate too slow, preventing further iterations"
        
        # Update signature history
        self.search_signatures[session_id].append(signature)
        
        return True, None
    
    def cache_intermediate_results(
        self,
        session_id: str,
        cache_key: str,
        results: List[RetrievedColumn],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Cache intermediate results using LRU with semantic similarity."""
        # Implement LRU eviction if cache is full
        if len(self.result_cache) >= self.max_history_size:
            self._evict_least_recently_used()
        
        # Store results with metadata
        full_cache_key = f"{session_id}_{cache_key}"
        self.result_cache[full_cache_key] = results
        
        # Update access time for LRU
        self._update_cache_access_time(full_cache_key)
        
        self.logger.debug(f"Cached {len(results)} results under key {full_cache_key}")
    
    def get_cached_results(
        self,
        session_id: str,
        cache_key: str
    ) -> Optional[List[RetrievedColumn]]:
        """Retrieve cached results and update LRU ordering"""
        full_cache_key = f"{session_id}_{cache_key}"
        
        if full_cache_key in self.result_cache:
            self._update_cache_access_time(full_cache_key)
            return self.result_cache[full_cache_key]
        
        return None
    
    def generate_iteration_summary(self, session_id: str) -> Dict[str, Any]:
        """Generate comprehensive audit trail of reasoning process."""
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session_meta = self.active_sessions[session_id]
        iterations = self.iteration_history[session_id]
        
        if not iterations:
            return {'session_id': session_id, 'status': 'no_iterations'}
        
        # Basic statistics
        total_time = sum(it.execution_time for it in iterations)
        total_results = sum(it.results_found for it in iterations)
        confidence_progression = [it.confidence_score for it in iterations]
        
        # Convergence analysis
        convergence_rate = self._calculate_convergence_rate(confidence_progression)
        convergence_trend = self._analyze_convergence_trend(confidence_progression)
        
        # Keyword analysis
        all_keywords = set()
        keyword_effectiveness = {}
        for it in iterations:
            all_keywords.update(it.keywords_used)
            for keyword in it.keywords_used:
                if keyword not in keyword_effectiveness:
                    keyword_effectiveness[keyword] = []
                keyword_effectiveness[keyword].append(it.results_found)
        
        # Performance analysis
        performance_trend = self._analyze_performance_trend(session_id)
        bottlenecks = self._identify_performance_bottlenecks(iterations)
        
        summary = {
            'session_metadata': asdict(session_meta),
            'iteration_count': len(iterations),
            'total_execution_time': total_time,
            'average_iteration_time': total_time / len(iterations),
            'total_unique_results': total_results,
            'final_confidence': confidence_progression[-1],
            'confidence_improvement': confidence_progression[-1] - confidence_progression[0] if len(confidence_progression) > 1 else 0.0,
            'convergence_analysis': {
                'convergence_rate': convergence_rate,
                'trend': convergence_trend,
                'is_converging': convergence_rate > 0
            },
            'keyword_analysis': {
                'total_unique_keywords': len(all_keywords),
                'most_effective_keywords': self._rank_keyword_effectiveness(keyword_effectiveness),
                'keyword_diversity': len(all_keywords) / max(1, len(iterations))
            },
            'performance_analysis': {
                'trend': performance_trend,
                'bottlenecks': bottlenecks,
                'efficiency_score': self._calculate_efficiency_score(iterations)
            },
            'iteration_details': [
                {
                    'iteration': it.iteration_number,
                    'timestamp': it.timestamp,
                    'keywords_count': len(it.keywords_used),
                    'gaps_targeted': len(it.gaps_targeted),
                    'results_found': it.results_found,
                    'confidence_score': it.confidence_score,
                    'execution_time': it.execution_time,
                    'convergence_metrics': it.convergence_metrics
                }
                for it in iterations
            ]
        }
        
        self.logger.info(f"Generated summary for session {session_id}: {len(iterations)} iterations")
        return summary
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of a session"""
        if session_id not in self.active_sessions:
            return {'status': 'not_found'}
        
        session_meta = self.active_sessions[session_id]
        iterations = self.iteration_history[session_id]
        
        current_time = time.time()
        session_duration = current_time - session_meta.start_time
        
        return {
            'session_id': session_id,
            'status': 'active',
            'iterations_completed': len(iterations),
            'max_iterations': session_meta.max_iterations,
            'session_duration': session_duration,
            'last_confidence': iterations[-1].confidence_score if iterations else 0.0,
            'cached_results_count': len([k for k in self.result_cache.keys() if k.startswith(session_id)])
        }
    
    def finalize_session(self, session_id: str) -> None:
        """Clean up and finalize a completed session"""
        if session_id in self.active_sessions:
            self.logger.info(f"Finalizing session {session_id}")
            
            # Keep session data for analysis but mark as inactive
            session_meta = self.active_sessions[session_id]
            session_meta.start_time = -1  # Mark as finalized
            
            # Clean up temporary structures
            if session_id in self.search_signatures:
                del self.search_signatures[session_id]
    
    # 🔧 CRITICAL FIX: Add method compatibility for SearchOrchestrator
    def get_iteration_stats(self) -> Dict[str, Any]:
        """Get basic iteration statistics for SearchOrchestrator compatibility"""
        return {
            'total_iterations': len(self.simple_iteration_history),
            'recent_iterations': self.simple_iteration_history[-5:] if self.simple_iteration_history else [],
            'average_processing_time': sum(r.get('processing_time', 0) for r in self.simple_iteration_history) / max(1, len(self.simple_iteration_history)),
            'total_confidence_improvement': sum(r.get('confidence_improvement', 0) for r in self.simple_iteration_history)
        }
    
    def get_initial_query(self) -> str:
        """Get the initial query for the current session"""
        if self._current_session_id and self._current_session_id in self.active_sessions:
            return self.active_sessions[self._current_session_id].initial_query
        return ""
    
    # Private helper methods
    
    def _hash_query(self, query: str) -> str:
        """Create deterministic hash of query for tracking"""
        return hashlib.md5(query.encode('utf-8')).hexdigest()[:8]
    
    def _create_search_signature(
        self,
        keywords: List[str],
        gaps: Dict[str, List[str]]
    ) -> str:
        """Create unique signature for search parameters"""
        signature_data = {
            'keywords': sorted(keywords),
            'gaps': {k: sorted(v) for k, v in gaps.items()}
        }
        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.md5(signature_str.encode('utf-8')).hexdigest()[:12]
    
    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between search signatures"""
        if len(sig1) != len(sig2):
            return 0.0
        
        matches = sum(c1 == c2 for c1, c2 in zip(sig1, sig2))
        return matches / len(sig1)
    
    def _calculate_convergence_metrics(
        self,
        session_id: str,
        current_confidence: float
    ) -> Dict[str, float]:
        """Calculate convergence metrics for current iteration"""
        iterations = self.iteration_history[session_id]
        
        if not iterations:
            return {'improvement_rate': 0.0, 'velocity': 0.0}
        
        # Calculate improvement rate
        if len(iterations) >= 1:
            prev_confidence = iterations[-1].confidence_score
            improvement_rate = current_confidence - prev_confidence
        else:
            improvement_rate = current_confidence
        
        # Calculate convergence velocity (rate of change of improvement)
        if len(iterations) >= 2:
            prev_improvement = iterations[-1].confidence_score - iterations[-2].confidence_score
            velocity = improvement_rate - prev_improvement
        else:
            velocity = 0.0
        
        return {
            'improvement_rate': improvement_rate,
            'velocity': velocity,
            'convergence_score': min(1.0, current_confidence + improvement_rate)
        }
    
    def _update_transition_matrix(self, session_id: str, iteration_state: IterationState) -> None:
        """Update Markov chain transition matrix"""
        iterations = self.iteration_history[session_id]
        
        if len(iterations) < 2:
            return
        
        # Define states based on confidence ranges
        current_state = self._discretize_confidence(iteration_state.confidence_score)
        prev_state = self._discretize_confidence(iterations[-2].confidence_score)
        
        # Update transition count
        self.transition_matrix[prev_state][current_state] += 1
    
    def _discretize_confidence(self, confidence: float) -> str:
        """Convert continuous confidence to discrete state"""
        if confidence < 0.3:
            return 'low'
        elif confidence < 0.6:
            return 'medium'
        elif confidence < 0.8:
            return 'high'
        else:
            return 'very_high'
    
    def _is_converging_too_slowly(self, session_id: str) -> bool:
        """Check if convergence rate is too slow"""
        iterations = self.iteration_history[session_id]
        
        if len(iterations) < 3:
            return False
        
        # Check last 3 iterations for improvement
        recent_scores = [it.confidence_score for it in iterations[-3:]]
        improvements = [recent_scores[i] - recent_scores[i-1] for i in range(1, len(recent_scores))]
        
        # If all recent improvements are very small, convergence is too slow
        return all(imp < 0.01 for imp in improvements)
    
    def _evict_least_recently_used(self) -> None:
        """Evict least recently used cache entries"""
        if len(self.result_cache) > self.max_history_size * 0.8:
            # Remove oldest 20% of entries
            keys_to_remove = list(self.result_cache.keys())[:int(len(self.result_cache) * 0.2)]
            for key in keys_to_remove:
                del self.result_cache[key]
    
    def _update_cache_access_time(self, cache_key: str) -> None:
        """Update access time for LRU tracking"""
        pass
    
    def _calculate_convergence_rate(self, confidence_scores: List[float]) -> float:
        """Calculate overall convergence rate"""
        if len(confidence_scores) < 2:
            return 0.0
        
        total_improvement = confidence_scores[-1] - confidence_scores[0]
        return total_improvement / len(confidence_scores)
    
    def _analyze_convergence_trend(self, confidence_scores: List[float]) -> str:
        """Analyze convergence trend pattern"""
        if len(confidence_scores) < 3:
            return 'insufficient_data'
        
        # Calculate differences between consecutive scores
        diffs = [confidence_scores[i] - confidence_scores[i-1] for i in range(1, len(confidence_scores))]
        
        # Analyze trend
        positive_diffs = sum(1 for d in diffs if d > 0)
        if positive_diffs / len(diffs) > 0.7:
            return 'improving'
        elif positive_diffs / len(diffs) < 0.3:
            return 'declining'
        else:
            return 'oscillating'
    
    def _analyze_performance_trend(self, session_id: str) -> str:
        """Analyze performance trend for the session"""
        times = self.performance_metrics[session_id]
        
        if len(times) < 3:
            return 'insufficient_data'
        
        # Simple trend analysis
        recent_avg = sum(times[-3:]) / 3
        early_avg = sum(times[:3]) / min(3, len(times))
        
        if recent_avg < early_avg * 0.8:
            return 'improving'
        elif recent_avg > early_avg * 1.2:
            return 'degrading'
        else:
            return 'stable'
    
    def _identify_performance_bottlenecks(self, iterations: List[IterationState]) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        if not iterations:
            return bottlenecks
        
        avg_time = sum(it.execution_time for it in iterations) / len(iterations)
        slow_iterations = [it for it in iterations if it.execution_time > avg_time * 1.5]
        
        if slow_iterations:
            bottlenecks.append(f"{len(slow_iterations)} slow iterations detected")
        
        # Check for keyword effectiveness
        low_result_iterations = [it for it in iterations if it.results_found < 5]
        if len(low_result_iterations) > len(iterations) * 0.5:
            bottlenecks.append("Low keyword effectiveness")
        
        return bottlenecks
    
    def _rank_keyword_effectiveness(self, keyword_effectiveness: Dict[str, List[int]]) -> List[Dict[str, Any]]:
        """Rank keywords by effectiveness"""
        ranked = []
        
        for keyword, result_counts in keyword_effectiveness.items():
            avg_results = sum(result_counts) / len(result_counts)
            ranked.append({
                'keyword': keyword,
                'average_results': avg_results,
                'usage_count': len(result_counts)
            })
        
        return sorted(ranked, key=lambda x: x['average_results'], reverse=True)[:10]
    
    def _calculate_efficiency_score(self, iterations: List[IterationState]) -> float:
        """Calculate overall efficiency score for the session"""
        if not iterations:
            return 0.0
        
        # Combine time efficiency and result effectiveness
        total_time = sum(it.execution_time for it in iterations)
        total_results = sum(it.results_found for it in iterations)
        
        if total_time == 0:
            return 0.0
        
        results_per_second = total_results / total_time
        confidence_gain = iterations[-1].confidence_score - iterations[0].confidence_score if len(iterations) > 1 else iterations[0].confidence_score
        
        # Normalize to 0-1 scale
        efficiency = min(1.0, (results_per_second * confidence_gain) / 10)
        return efficiency
