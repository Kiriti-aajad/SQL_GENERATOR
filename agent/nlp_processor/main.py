"""
Main NLP Processor Entry Point - Updated for Orchestrator Integration
Primary interface for the NLP processor that integrates all components
Serves as the main entry point for your AI SQL Generator integration
Enhanced with orchestrator compatibility methods
"""

import logging
import time
import uuid
from typing import Dict, Any, List, Optional

# Import integration data models for compatibility
try:
    from agent.integration.data_models import QueryIntent, RequestType
except ImportError:
    # Fallback if integration not available
    QueryIntent = None
    RequestType = None

from .core.pipeline import NLPPipeline
from .core.data_models import AnalystQuery, EnhancedQueryResult
from .integration.schema_searcher_bridge import SchemaSearcherBridge
from .config_module import get_config

logger = logging.getLogger(__name__)

class NLPProcessor:
    """
    Main NLP Processor for AI SQL Generator Integration
    Coordinates all NLP processing and provides clean interface for your existing system
    Enhanced with orchestrator compatibility methods
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize NLP Processor with all components"""
        self.config = get_config()
        
        # Initialize core pipeline
        self.pipeline = NLPPipeline(config_path)
        
        # Initialize schema searcher bridge for integration
        self.schema_bridge = SchemaSearcherBridge()
        
        # Statistics tracking for orchestrator monitoring
        self.total_queries_processed = 0
        self.successful_processes = 0
        self.failed_processes = 0
        self.last_process_time = None
        self.average_processing_time_ms = 0.0
        
        # Performance tracking
        self.processing_times = []
        
        logger.info("NLP Processor initialized successfully with orchestrator compatibility")
    
    def process_analyst_query(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> EnhancedQueryResult:
        """
        Process analyst query and return enhanced result for your schema searcher
        Enhanced with performance tracking for orchestrator monitoring
        
        Args:
            query_text: Natural language query from analyst
            context: Optional context dictionary
            
        Returns:
            Enhanced query result ready for your existing AI SQL Generator
        """
        start_time = time.time()
        
        try:
            self.total_queries_processed += 1
            self.last_process_time = time.time()
            
            # Create analyst query
            analyst_query = AnalystQuery(
                query_text=query_text,
                context=context or {}
            )
            
            # Process through NLP pipeline
            processed_query = self.pipeline.process(analyst_query)
            
            # Enhance for schema searcher integration
            enhanced_result = self.schema_bridge.enhance_for_schema_search(processed_query)
            
            # Update success statistics
            processing_time = (time.time() - start_time) * 1000
            self.successful_processes += 1
            self._update_processing_time(processing_time)
            
            return enhanced_result
            
        except Exception as e:
            self.total_queries_processed += 1
            self.failed_processes += 1
            processing_time = (time.time() - start_time) * 1000
            self._update_processing_time(processing_time)
            
            logger.error(f"Error processing analyst query: {e}")
            raise
    
    def prepare_for_schema_search(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        ORCHESTRATOR METHOD: Prepare NLP output for Schema Retrieval Agent
        Converts NLP processor output to schema agent compatible format
        
        Args:
            query_text: Natural language query
            context: Optional context dictionary
            
        Returns:
            Schema agent compatible request dictionary
        """
        try:
            # Process query through NLP pipeline
            nlp_result = self.process_analyst_query(query_text, context)
            
            # Extract structured query data
            structured_query = getattr(nlp_result, 'structured_query', {})
            if isinstance(nlp_result, dict):
                structured_query = nlp_result.get('structured_query', {})
            
            # Extract intent classification
            intent_data = structured_query.get('intent_classification', {})
            search_scope = structured_query.get('search_scope', {})
            
            # Map NLP intent to schema agent format
            schema_request = {
                'query': query_text,
                'enhanced_query': structured_query.get('refined_query', query_text),
                'request_type': 'nlp_enhanced_schema',
                'request_id': str(uuid.uuid4()),
                
                # Target tables and columns from NLP analysis
                'target_tables': search_scope.get('target_tables', []),
                'target_columns': search_scope.get('target_columns', []),
                
                # Intent classification data
                'intent_classification': {
                    'primary_intent': intent_data.get('primary_intent', 'unknown'),
                    'confidence': intent_data.get('confidence', 0.8),
                    'secondary_intents': intent_data.get('secondary_intents', []),
                    'requires_joins': intent_data.get('requires_joins', True),
                    'requires_xml': intent_data.get('requires_xml', True),
                    'complexity': intent_data.get('complexity', 0.5)
                },
                
                # Processing parameters
                'include_xml': intent_data.get('requires_xml', True),
                'include_joins': intent_data.get('requires_joins', True),
                'max_results': self._determine_max_results_from_intent(intent_data),
                'min_confidence': 0.0,
                
                # Complete NLP context for schema agent
                'nlp_context': {
                    'structured_query': structured_query,
                    'processing_metadata': getattr(nlp_result, 'processing_metadata', {}),
                    'field_mappings': self._extract_field_mappings(nlp_result),
                    'business_context': getattr(nlp_result, 'business_context', {}),
                    'confidence_scores': getattr(nlp_result, 'confidence_scores', {}),
                    'temporal_expressions': structured_query.get('temporal_expressions', []),
                    'geographic_entities': structured_query.get('geographic_entities', []),
                    'numerical_entities': structured_query.get('numerical_entities', [])
                },
                
                # Processing hints
                'processing_hints': {
                    'nlp_enhanced': True,
                    'complexity_score': intent_data.get('complexity', 0.5),
                    'confidence_level': 'high' if intent_data.get('confidence', 0.8) > 0.8 else 'medium',
                    'processing_priority': 'high' if intent_data.get('complexity', 0.5) > 0.7 else 'normal'
                }
            }
            
            logger.debug(f"Prepared schema search request for query: '{query_text[:50]}...'")
            return schema_request
            
        except Exception as e:
            logger.error(f"Failed to prepare schema search request: {e}")
            # Return fallback request
            return {
                'query': query_text,
                'enhanced_query': query_text,
                'request_type': 'basic_search',
                'request_id': str(uuid.uuid4()),
                'intent_classification': {'primary_intent': 'unknown'},
                'include_xml': True,
                'include_joins': True,
                'max_results': 30,
                'processing_hints': {'fallback': True, 'error': str(e)}
            }
    
    def orchestrator_health_check(self) -> Dict[str, Any]:
        """
        ORCHESTRATOR METHOD: Standardized health check for orchestrator
        
        Returns:
            Comprehensive health status and capabilities
        """
        try:
            # Calculate success rate
            success_rate = 0.0
            if self.total_queries_processed > 0:
                success_rate = (self.successful_processes / self.total_queries_processed) * 100
            
            # Get system capabilities
            capabilities = self.get_system_capabilities()
            
            return {
                'component': 'NLPProcessor',
                'status': 'healthy',
                'version': '1.0.0',
                
                # Performance statistics
                'performance_metrics': {
                    'total_queries_processed': self.total_queries_processed,
                    'successful_processes': self.successful_processes,
                    'failed_processes': self.failed_processes,
                    'success_rate_percent': round(success_rate, 2),
                    'average_processing_time_ms': round(self.average_processing_time_ms, 2),
                    'last_process_time': self.last_process_time
                },
                
                # Capabilities
                'capabilities': capabilities,
                
                # Supported methods for orchestrator
                'supported_methods': [
                    'process_analyst_query',
                    'prepare_for_schema_search',
                    'orchestrator_health_check',
                    'get_system_capabilities',
                    'get_statistics'
                ],
                
                # Integration status
                'integration_status': {
                    'schema_bridge_available': self.schema_bridge is not None,
                    'pipeline_ready': self.pipeline is not None,
                    'orchestrator_compatible': True
                },
                
                'health_check_timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'component': 'NLPProcessor',
                'status': 'error',
                'error_message': str(e),
                'health_check_timestamp': time.time(),
                'capabilities': {},
                'supported_methods': []
            }
    
    def _extract_field_mappings(self, nlp_result: Any) -> List[Dict[str, Any]]:
        """Extract field mappings from NLP result"""
        try:
            field_mappings = []
            
            # Check for field_mappings attribute
            if hasattr(nlp_result, 'field_mappings'):
                mappings = nlp_result.field_mappings
                for mapping in mappings:
                    if hasattr(mapping, 'to_dict'):
                        field_mappings.append(mapping.to_dict())
                    else:
                        field_mappings.append({
                            'table': getattr(mapping, 'table', 'unknown'),
                            'column': getattr(mapping, 'name', getattr(mapping, 'column', 'unknown')),
                            'field_type': getattr(mapping, 'field_type', 'unknown'),
                            'confidence': getattr(mapping, 'confidence', 0.8)
                        })
            
            return field_mappings
            
        except Exception as e:
            logger.warning(f"Error extracting field mappings: {e}")
            return []
    
    def _determine_max_results_from_intent(self, intent_data: Dict[str, Any]) -> int:
        """Determine optimal result count based on query intent"""
        primary_intent = intent_data.get('primary_intent', 'unknown').lower()
        confidence = intent_data.get('confidence', 0.8)
        
        # Intent-based result mapping
        if 'aggregation' in primary_intent or 'complex' in primary_intent:
            base_results = 40
        elif 'simple' in primary_intent or 'lookup' in primary_intent:
            base_results = 15
        elif 'temporal' in primary_intent:
            base_results = 25
        elif 'join' in primary_intent:
            base_results = 50
        else:
            base_results = 30
        
        # Adjust based on confidence
        if confidence > 0.9:
            return base_results
        elif confidence > 0.7:
            return min(base_results + 10, 70)
        else:
            return min(base_results + 20, 80)
    
    def _update_processing_time(self, processing_time_ms: float):
        """Update running average of processing time"""
        self.processing_times.append(processing_time_ms)
        
        # Keep only last 100 processing times for average calculation
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        # Calculate running average
        self.average_processing_time_ms = sum(self.processing_times) / len(self.processing_times)
    
    def get_system_capabilities(self) -> Dict[str, Any]:
        """Get comprehensive system capabilities"""
        try:
            return {
                "pipeline_capabilities": getattr(self.pipeline.schema_context, 'total_physical_columns', 0),
                "xml_fields": getattr(self.pipeline.schema_context, 'total_xml_fields', 0),
                "verified_joins": len(getattr(self.pipeline.schema_context, 'join_intelligence', {})),
                "bridge_statistics": self.schema_bridge.get_bridge_statistics() if self.schema_bridge else {},
                "pipeline_statistics": self.pipeline.get_statistics() if hasattr(self.pipeline, 'get_statistics') else {},
                
                # Enhanced capabilities for orchestrator
                'orchestrator_features': [
                    'intent_classification',
                    'entity_extraction',
                    'schema_targeting',
                    'complexity_assessment',
                    'temporal_processing',
                    'business_context_analysis'
                ],
                
                'supported_intents': [
                    'simple_lookup',
                    'aggregation',
                    'complex_analysis',
                    'temporal_filter',
                    'join_analysis',
                    'reporting'
                ],
                
                'output_formats': [
                    'enhanced_query_result',
                    'schema_search_request',
                    'orchestrator_compatible'
                ]
            }
        except Exception as e:
            logger.error(f"Error getting capabilities: {e}")
            return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "total_processed": self.total_queries_processed,
            "successful_processes": self.successful_processes,
            "failed_processes": self.failed_processes,
            "success_rate": (self.successful_processes / max(self.total_queries_processed, 1)) * 100,
            "average_processing_time_ms": self.average_processing_time_ms,
            "pipeline_stats": self.pipeline.get_statistics() if hasattr(self.pipeline, 'get_statistics') else {},
            "bridge_stats": self.schema_bridge.get_bridge_statistics() if self.schema_bridge else {}
        }

def main():
    """Test NLP Processor functionality with orchestrator compatibility"""
    try:
        processor = NLPProcessor()
        print("NLP Processor initialized successfully!")
        
        # Test orchestrator compatibility
        health = processor.orchestrator_health_check()
        print(f"Health Status: {health['status']}")
        print(f"Orchestrator Compatible: {health['integration_status']['orchestrator_compatible']}")
        
        # Test with sample queries
        test_queries = [
            "Give me last 10 days created customers",
            "Which regions have maximum defaulters",
            "What is the sum of collateral for ABC Corporation"
        ]
        
        for query in test_queries:
            print(f"\nProcessing: {query}")
            
            # Test schema search preparation
            schema_request = processor.prepare_for_schema_search(query)
            print(f"Intent: {schema_request.get('intent_classification', {}).get('primary_intent')}")
            print(f"Target tables: {schema_request.get('target_tables', [])}")
            print(f"Max results: {schema_request.get('max_results')}")
        
        print(f"\nSystem capabilities: {processor.get_system_capabilities()}")
        
    except Exception as e:
        print(f"Error testing NLP Processor: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
