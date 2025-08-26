"""
Generator-Executor Bridge
Seamlessly connects SQL Generator with SQL Executor for end-to-end functionality
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
from dataclasses import dataclass, field
import json
import uuid

# CORRECTED IMPORTS - Fix the import paths
from agent.sql_executor import (
    SQLExecutor, ExecutionRequest, ExecutionResult, 
    OutputFormat, ExecutionMode, QueryMetadata, 
    QueryType, ErrorType, ExecutionStatus
)

# Set up logging
logger = logging.getLogger(__name__)

# CRITICAL FIX: Add these missing classes locally to avoid circular imports
class IntegrationStatus:
    """Local integration status tracking"""
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
    
    def record_request(self, success: bool, execution_time: float = 0.0):
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
    
    def get_stats(self):
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.successful_requests / max(self.total_requests, 1)) * 100,
            'uptime_seconds': time.time() - self.start_time
        }

class IntegrationConfig:
    """Local integration configuration"""
    DEFAULT_MAX_ROWS = 10000
    DEFAULT_TIMEOUT = 30
    ENABLE_CACHING = True
    MAX_RETRIES = 3
    BATCH_SIZE = 1000
    CONNECTION_TIMEOUT = 15
    QUERY_TIMEOUT = 300
    
    @classmethod
    def get_default_config(cls):
        """Get default configuration dictionary"""
        return {
            'max_rows': cls.DEFAULT_MAX_ROWS,
            'timeout': cls.DEFAULT_TIMEOUT,
            'enable_caching': cls.ENABLE_CACHING,
            'max_retries': cls.MAX_RETRIES,
            'batch_size': cls.BATCH_SIZE,
            'connection_timeout': cls.CONNECTION_TIMEOUT,
            'query_timeout': cls.QUERY_TIMEOUT
        }

# Create instances
integration_status = IntegrationStatus()

@dataclass
class BridgeRequest:
    """Request format for Generator-Executor bridge"""
    
    # From SQL Generator
    natural_language_query: str
    generated_sql: str
    user_id: str
    
    # Generator metadata (optional)
    generator_confidence: float = 1.0
    tables_used: List[str] = field(default_factory=list)
    query_complexity: int = 0
    estimated_rows: Optional[int] = None
    
    # Execution preferences
    execution_mode: str = "execute"  # execute, dry_run, explain_plan
    output_format: str = "json"      # json, csv, html, excel
    max_rows: int = IntegrationConfig.DEFAULT_MAX_ROWS
    timeout_seconds: int = IntegrationConfig.DEFAULT_TIMEOUT
    
    # Request tracking
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Additional options
    include_execution_plan: bool = False
    stream_results: bool = False
    cache_results: bool = False

@dataclass
class BridgeResponse:
    """Response format from Generator-Executor bridge"""
    
    # Request information
    request_id: str
    natural_language_query: str
    generated_sql: str
    
    # Execution results
    success: bool
    status: str
    message: str
    
    # Data results
    data: List[Dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    columns: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    total_time_ms: int = 0
    generation_time_ms: int = 0
    execution_time_ms: int = 0
    
    # Error information
    error_type: Optional[str] = None
    error_details: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    
    # Export options
    formatted_results: Optional[Union[str, bytes]] = None
    export_available: bool = False
    supported_formats: List[str] = field(default_factory=list)
    
    # Metadata
    execution_timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for JSON serialization"""
        return {
            'request_id': self.request_id,
            'natural_language_query': self.natural_language_query,
            'generated_sql': self.generated_sql,
            'success': self.success,
            'status': self.status,
            'message': self.message,
            'data': self.data,
            'row_count': self.row_count,
            'columns': self.columns,
            'performance': {
                'total_time_ms': self.total_time_ms,
                'generation_time_ms': self.generation_time_ms,
                'execution_time_ms': self.execution_time_ms
            },
            'error': {
                'type': self.error_type,
                'details': self.error_details,
                'suggestions': self.suggestions
            } if not self.success else None,
            'export': {
                'available': self.export_available,
                'supported_formats': self.supported_formats
            },
            'metadata': {
                'execution_timestamp': self.execution_timestamp.isoformat(),
                'user_id': self.user_id
            }
        }
    
    def to_json(self) -> str:
        """Convert response to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)

class RequestTransformer:
    """Transforms requests between Generator and Executor formats"""
    
    def __init__(self):
        self.config = IntegrationConfig()
    
    def bridge_to_executor_request(self, bridge_request: BridgeRequest) -> ExecutionRequest:
        """Transform BridgeRequest to ExecutionRequest"""
        
        # Map execution mode
        execution_mode_map = {
            'execute': ExecutionMode.EXECUTE,
            'dry_run': ExecutionMode.DRY_RUN,
            'explain_plan': ExecutionMode.EXPLAIN_PLAN
        }
        
        # Map output format
        output_format_map = {
            'json': OutputFormat.JSON,
            'csv': OutputFormat.CSV,
            'html': OutputFormat.HTML,
            'excel': OutputFormat.EXCEL
        }
        
        # Create query metadata
        query_metadata = QueryMetadata(
            query_type=self._detect_query_type(bridge_request.generated_sql),
            tables_used=bridge_request.tables_used,
            complexity_score=bridge_request.query_complexity,
            estimated_rows=bridge_request.estimated_rows,
            confidence_score=bridge_request.generator_confidence,
            has_joins=self._has_joins(bridge_request.generated_sql),
            has_xml_operations=self._has_xml_operations(bridge_request.generated_sql),
            has_aggregations=self._has_aggregations(bridge_request.generated_sql)
        )
        
        # Generate request ID if not provided
        if not bridge_request.request_id:
            bridge_request.request_id = str(uuid.uuid4())
        
        return ExecutionRequest(
            sql_query=bridge_request.generated_sql,
            user_id=bridge_request.user_id,
            query_metadata=query_metadata,
            execution_mode=execution_mode_map.get(bridge_request.execution_mode, ExecutionMode.EXECUTE),
            max_rows=bridge_request.max_rows,
            timeout_seconds=bridge_request.timeout_seconds,
            output_format=output_format_map.get(bridge_request.output_format, OutputFormat.JSON),
            include_execution_time=True,
            include_row_count=True,
            request_id=bridge_request.request_id,
            session_id=bridge_request.session_id
        )
    
    def executor_to_bridge_response(
        self, 
        execution_result: ExecutionResult, 
        bridge_request: BridgeRequest,
        total_time_ms: int,
        generation_time_ms: int = 0
    ) -> BridgeResponse:
        """Transform ExecutionResult to BridgeResponse"""
        
        # Extract column information
        columns = []
        for col in execution_result.columns:
            columns.append({
                'name': col.name,
                'data_type': col.data_type,
                'max_length': col.max_length,
                'is_nullable': col.is_nullable,
                'ordinal_position': col.ordinal_position
            })
        
        # Create bridge response
        response = BridgeResponse(
            request_id=bridge_request.request_id or str(uuid.uuid4()),
            natural_language_query=bridge_request.natural_language_query,
            generated_sql=bridge_request.generated_sql,
            success=execution_result.success,
            status=execution_result.status.value,
            message=execution_result.message,
            data=execution_result.data,
            row_count=execution_result.row_count,
            columns=columns,
            total_time_ms=total_time_ms,
            generation_time_ms=generation_time_ms,
            execution_time_ms=execution_result.performance.execution_time_ms if execution_result.performance else 0,
            error_type=execution_result.error_type.value if execution_result.error_type else None,
            error_details=execution_result.error_details,
            suggestions=execution_result.suggestions,
            export_available=execution_result.export_available,
            supported_formats=execution_result.supported_formats,
            execution_timestamp=execution_result.execution_timestamp,
            user_id=bridge_request.user_id
        )
        
        return response
    
    def _detect_query_type(self, sql_query: str) -> QueryType:
        """Detect the type of SQL query"""
        sql_upper = sql_query.upper().strip()
        
        if sql_upper.startswith('WITH'):
            return QueryType.WITH_CTE
        elif sql_upper.startswith('SELECT'):
            return QueryType.SELECT
        elif 'UNION' in sql_upper:
            return QueryType.UNION
        else:
            return QueryType.UNKNOWN
    
    def _has_joins(self, sql_query: str) -> bool:
        """Check if query contains JOIN operations"""
        return 'JOIN' in sql_query.upper()
    
    def _has_xml_operations(self, sql_query: str) -> bool:
        """Check if query contains XML operations"""
        return any(pattern in sql_query for pattern in ['.value(', '.exist(', '.nodes(', 'XML_'])
    
    def _has_aggregations(self, sql_query: str) -> bool:
        """Check if query contains aggregate functions"""
        aggregates = ['COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX(', 'GROUP BY']
        return any(agg in sql_query.upper() for agg in aggregates)

class WorkflowTracker:
    """Tracks workflow performance and statistics"""
    
    def __init__(self):
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
    
    def start_workflow(self, request_id: str, bridge_request: BridgeRequest) -> Dict[str, Any]:
        """Start tracking a workflow"""
        
        workflow_context = {
            'request_id': request_id,
            'start_time': time.time(),
            'start_datetime': datetime.utcnow(),
            'user_id': bridge_request.user_id,
            'natural_language_query': bridge_request.natural_language_query,
            'generated_sql': bridge_request.generated_sql,
            'execution_mode': bridge_request.execution_mode,
            'output_format': bridge_request.output_format
        }
        
        self.active_workflows[request_id] = workflow_context
        return workflow_context
    
    def end_workflow(self, request_id: str, success: bool) -> Dict[str, Any]:
        """End workflow tracking and calculate metrics"""
        
        if request_id not in self.active_workflows:
            return {'total_time_ms': 0}
        
        context = self.active_workflows[request_id]
        end_time = time.time()
        
        metrics = {
            'total_time_ms': int((end_time - context['start_time']) * 1000),
            'success': success,
            'user_id': context['user_id'],
            'execution_mode': context['execution_mode'],
            'output_format': context['output_format']
        }
        
        # Record in integration status
        integration_status.record_request(success, metrics['total_time_ms'])
        
        # Clean up
        del self.active_workflows[request_id]
        
        return metrics

class GeneratorExecutorBridge:
    """Main bridge class connecting SQL Generator with SQL Executor"""
    
    def __init__(self, sql_executor: Optional[SQLExecutor] = None):
        self.sql_executor = sql_executor or SQLExecutor()
        self.transformer = RequestTransformer()
        self.tracker = WorkflowTracker()
        self.config = IntegrationConfig()
        
        # Bridge statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Initialize flag
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the bridge"""
        if self.is_initialized:
            logger.warning("Generator-Executor bridge already initialized")
            return
        
        logger.info("Initializing Generator-Executor bridge...")
        
        try:
            # Initialize SQL Executor
            await self.sql_executor.initialize()
            
            self.is_initialized = True
            logger.info("Generator-Executor bridge initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Generator-Executor bridge: {e}")
            raise
    
    async def execute_generation_request(self, bridge_request: BridgeRequest) -> BridgeResponse:
        """
        Main method: Execute complete generation-to-execution workflow
        """
        # Ensure bridge is initialized
        if not self.is_initialized:
            await self.initialize()
        
        # Generate request ID if not provided
        if not bridge_request.request_id:
            bridge_request.request_id = str(uuid.uuid4())
        
        logger.info(f"Processing bridge request for user {bridge_request.user_id} [Request: {bridge_request.request_id}]")
        
        # Update counters
        self.total_requests += 1
        
        # Start workflow tracking
        workflow_context = self.tracker.start_workflow(bridge_request.request_id, bridge_request)
        
        try:
            # Transform bridge request to executor request
            executor_request = self.transformer.bridge_to_executor_request(bridge_request)
            
            # Execute query using SQL Executor
            execution_result = await self.sql_executor.execute_query(executor_request)
            
            # Calculate total time
            workflow_metrics = self.tracker.end_workflow(bridge_request.request_id, execution_result.success)
            
            # Transform executor result to bridge response
            bridge_response = self.transformer.executor_to_bridge_response(
                execution_result=execution_result,
                bridge_request=bridge_request,
                total_time_ms=workflow_metrics['total_time_ms'],
                generation_time_ms=0  # Would be provided by SQL Generator
            )
            
            # Format results if requested and successful
            if execution_result.success and bridge_request.output_format != 'json':
                try:
                    formatted_result = await self.sql_executor.format_result(
                        execution_result, 
                        executor_request.output_format
                    )
                    bridge_response.formatted_results = formatted_result
                except Exception as e:
                    logger.warning(f"Failed to format results: {e}")
            
            # Update success counter
            if execution_result.success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            
            logger.info(f"Bridge request completed: {execution_result.status.value} in {workflow_metrics['total_time_ms']}ms")
            return bridge_response
            
        except Exception as e:
            # Handle bridge-level errors
            self.failed_requests += 1
            self.tracker.end_workflow(bridge_request.request_id, False)
            
            logger.error(f"Bridge request failed: {e}")
            
            return self._create_error_response(
                bridge_request=bridge_request,
                error=e,
                error_message=f"Bridge execution failed: {str(e)}"
            )
    
    def _create_error_response(self, bridge_request: BridgeRequest, error: Exception, 
                              error_message: str) -> BridgeResponse:
        """Create error response for bridge failures"""
        
        return BridgeResponse(
            request_id=bridge_request.request_id or str(uuid.uuid4()),
            natural_language_query=bridge_request.natural_language_query,
            generated_sql=bridge_request.generated_sql,
            success=False,
            status='failed',
            message=error_message,
            error_type='bridge_error',
            error_details=str(error),
            suggestions=[
                "Check SQL Generator output format",
                "Verify database connectivity",
                "Review request parameters",
                "Contact system administrator if issue persists"
            ],
            user_id=bridge_request.user_id
        )
    
    def get_bridge_stats(self) -> Dict[str, Any]:
        """Get bridge performance statistics"""
        
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            'bridge_stats': {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate_percent': round(success_rate, 2),
                'is_initialized': self.is_initialized
            },
            'integration_stats': integration_status.get_stats(),
            'executor_stats': self.sql_executor.get_execution_stats(),
            'active_workflows': len(self.tracker.active_workflows)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive bridge health check"""
        
        health_status = {
            'bridge_healthy': self.is_initialized,
            'bridge_stats': self.get_bridge_stats(),
            'executor_health': await self.sql_executor.health_check(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Overall health assessment
        health_status['overall_healthy'] = (
            self.is_initialized and 
            health_status['executor_health'].get('overall_healthy', False)
        )
        
        return health_status
    
    async def shutdown(self):
        """Graceful shutdown of bridge"""
        logger.info("Shutting down Generator-Executor bridge...")
        
        # Shutdown SQL Executor
        await self.sql_executor.shutdown()
        
        self.is_initialized = False
        logger.info("Generator-Executor bridge shutdown complete")

# Global bridge instance
generator_executor_bridge = GeneratorExecutorBridge()

# Convenience functions
async def execute_bridge_request(bridge_request: BridgeRequest) -> BridgeResponse:
    """Execute bridge request using global bridge instance"""
    return await generator_executor_bridge.execute_generation_request(bridge_request)

async def execute_sql_directly(sql_query: str, user_id: str, **kwargs) -> BridgeResponse:
    """Execute SQL directly using global bridge instance"""
    bridge_request = BridgeRequest(
        natural_language_query=kwargs.get('description', 'Direct SQL execution'),
        generated_sql=sql_query,
        user_id=user_id,
        execution_mode=kwargs.get('execution_mode', 'execute'),
        output_format=kwargs.get('output_format', 'json'),
        max_rows=kwargs.get('max_rows', IntegrationConfig.DEFAULT_MAX_ROWS),
        timeout_seconds=kwargs.get('timeout_seconds', IntegrationConfig.DEFAULT_TIMEOUT)
    )
    
    return await generator_executor_bridge.execute_generation_request(bridge_request)

async def validate_sql_query(sql_query: str, user_id: str) -> BridgeResponse:
    """Validate SQL query using global bridge instance"""
    bridge_request = BridgeRequest(
        natural_language_query=f"Validation request for: {sql_query[:50]}...",
        generated_sql=sql_query,
        user_id=user_id,
        execution_mode='dry_run',
        output_format='json'
    )
    
    return await generator_executor_bridge.execute_generation_request(bridge_request)

async def bridge_health_check() -> Dict[str, Any]:
    """Perform bridge health check using global instance"""
    return await generator_executor_bridge.health_check()

# Export all bridge components
__all__ = [
    'GeneratorExecutorBridge', 'BridgeRequest', 'BridgeResponse',
    'RequestTransformer', 'WorkflowTracker', 'IntegrationStatus', 'IntegrationConfig',
    'generator_executor_bridge', 'integration_status',
    'execute_bridge_request', 'execute_sql_directly', 
    'validate_sql_query', 'bridge_health_check'
]
