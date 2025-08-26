"""
CENTRALIZED NLP-SCHEMA INTEGRATION BRIDGE WITH DYNAMIC SCHEMA PROCESSING AND SQL GENERATION

This is the SINGLE SOURCE OF TRUTH for:
- PromptBuilder integration
- Dynamic schema context preservation
- NLP-Schema-SQL pipeline coordination
- COMPLETE PIPELINE: Schema → Prompt → SQL
- No hardcoded domain logic - completely generic
- No duplication with other files
- ENHANCED: XML Schema integration from xml_schema.json

Version: 2.9.2 - COMPLETE PIPELINE WITH XML SCHEMA INTEGRATION
Date: 2025-08-22
"""

import time
import logging
import asyncio
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Protocol, runtime_checkable, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid

# Safe imports with comprehensive fallbacks
try:
    from agent.nlp_processor.main import NLPProcessor
    from agent.schema_searcher.core.retrieval_agent import create_schema_retrieval_agent
    from agent.schema_searcher.core.intelligent_retrieval_agent import create_intelligent_retrieval_agent
    NLP_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"NLP components not available: {e}")
    NLP_COMPONENTS_AVAILABLE = False

# PromptBuilder import
try:
    from agent.prompt_builder.core import PromptBuilder
    PROMPT_BUILDER_AVAILABLE = True
    logging.info("PromptBuilder imported successfully - dynamic schema integration enabled")
except ImportError as e:
    logging.warning(f"PromptBuilder not available: {e}")
    PromptBuilder = None
    PROMPT_BUILDER_AVAILABLE = False

# SearchOrchestrator import
try:
    from agent.schema_searcher.orchestration.search_orchestrator import SearchOrchestrator, SearchMethod
    SEARCH_ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"SearchOrchestrator not available: {e}")
    SEARCH_ORCHESTRATOR_AVAILABLE = False

# NEW: Prompt-SQL Bridge import
try:
    from agent.integration.prompt_sql_bridge import create_prompt_sql_bridge, PromptSQLRequest, PromptSQLResponse
    PROMPT_SQL_BRIDGE_AVAILABLE = True
    logging.info("Prompt-SQL Bridge imported successfully - complete pipeline enabled")
except ImportError as e:
    logging.warning(f"Prompt-SQL Bridge not available: {e}")
    PROMPT_SQL_BRIDGE_AVAILABLE = False

# ENHANCED: XML Schema Manager import
try:
    from agent.schema_searcher.managers.xml_schema_manager import XMLSchemaManager
    XML_SCHEMA_MANAGER_AVAILABLE = True
    logging.info("XML Schema Manager imported successfully - XML field processing enabled")
except ImportError as e:
    logging.warning(f"XML Schema Manager not available: {e}")
    XMLSchemaManager = None
    XML_SCHEMA_MANAGER_AVAILABLE = False


@runtime_checkable
class NLPProcessorProtocol(Protocol):
    def health_check(self) -> Union[Dict[str, Any], bool]: ...
    def process_analyst_query(self, query: str) -> Dict[str, Any]: ...
    def process_query(self, query: str) -> Dict[str, Any]: ...


@runtime_checkable
class SchemaAgentProtocol(Protocol):
    def health_check(self) -> Union[Dict[str, Any], bool]: ...
    def retrieve_complete_schema(self, query: str) -> Dict[str, Any]: ...
    def search_schema(self, query: str) -> Dict[str, Any]: ...


@runtime_checkable
class IntelligentAgentProtocol(Protocol):
    def health_check(self) -> Union[Dict[str, Any], bool]: ...
    def retrieve_complete_schema_json(self, request: Dict[str, Any]) -> Any: ...


class ProcessingResult(Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    DEGRADED = "degraded"


@dataclass
class RequestStatistics:
    """Enhanced request statistics tracking"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    degraded_requests: int = 0
    start_time: float = field(default_factory=time.time)
    avg_response_time: float = 0.0
    last_request_time: Optional[float] = None

    def calculate_success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    def increment_request(self, result: ProcessingResult, response_time: float = 0.0) -> None:
        self.total_requests += 1
        self.last_request_time = time.time()
        
        if response_time > 0:
            self.avg_response_time = (
                (self.avg_response_time * (self.total_requests - 1) + response_time) /
                self.total_requests
            )
        
        if result == ProcessingResult.SUCCESS:
            self.successful_requests += 1
        elif result == ProcessingResult.DEGRADED:
            self.degraded_requests += 1
        else:
            self.failed_requests += 1


@dataclass
class IntegrationConfig:
    """DYNAMIC configuration - no hardcoded domain assumptions"""
    # Core settings
    nlp_timeout: float = 30.0
    schema_timeout: float = 30.0
    integration_timeout: float = 60.0
    sql_generation_timeout: float = 45.0
    
    # Feature flags
    enable_sql_generation: bool = True  # ENSURE THIS IS TRUE
    use_search_orchestrator: bool = True
    use_prompt_builder: bool = True
    enable_detailed_logging: bool = False
    enable_xml_schema_integration: bool = True  # NEW: Enable XML schema integration
    
    # Advanced settings
    search_orchestrator_convergence: float = 0.8
    max_retry_attempts: int = 2
    circuit_breaker_threshold: int = 5
    
    # Dynamic schema settings
    max_tables_in_prompt: int = 20
    max_columns_per_table: int = 15
    max_joins_in_prompt: int = 15
    max_xml_fields_in_prompt: int = 10
    
    # NEW: XML Schema settings
    xml_schema_path: Optional[str] = None  # Will use auto-detection if None
    xml_enrichment_enabled: bool = True


class CentralizedAsyncManager:
    """CENTRALIZED async management - eliminates duplication"""
    
    @staticmethod
    async def safe_async_call(
        func: Callable,
        *args,
        timeout: float = 30.0,
        fallback_result: Any = None,
        **kwargs
    ) -> Any:
        try:
            if asyncio.iscoroutinefunction(func):
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            else:
                return await asyncio.wait_for(
                    asyncio.to_thread(func, *args, **kwargs),
                    timeout=timeout
                )
        except asyncio.TimeoutError:
            logging.warning(f"Async operation timed out after {timeout}s")
            return fallback_result
        except Exception as e:
            logging.error(f"Async operation failed: {e}")
            return fallback_result


class CentralizedPromptBuilder:
    """CENTRALIZED PromptBuilder service - DYNAMIC SCHEMA ONLY"""
    
    def __init__(self, async_client_manager=None, config: Optional[IntegrationConfig] = None):
        self.async_client_manager = async_client_manager
        self.config = config or IntegrationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize PromptBuilder
        if PROMPT_BUILDER_AVAILABLE:
            try:
                self.prompt_builder = PromptBuilder(async_client_manager)  # pyright: ignore[reportOptionalCall]
                self.use_prompt_builder = True
                self.logger.info("CENTRALIZED PromptBuilder initialized - schema-based prompts enabled")
            except Exception as e:
                self.logger.error(f"PromptBuilder initialization failed: {e}")
                self.prompt_builder = None
                self.use_prompt_builder = False
        else:
            self.prompt_builder = None
            self.use_prompt_builder = False
            self.logger.warning("PromptBuilder not available - using dynamic basic prompt generation")

    async def generate_sophisticated_prompt(
        self,
        query: str,
        schema_context: Dict[str, Any],
        domain_context: Optional[Dict[str, Any]] = None,
        request_id: str = "unknown"
    ) -> Dict[str, Any]:
        """Generate prompt using ONLY discovered schema - NO hardcoded domain logic"""
        
        try:
            if self.use_prompt_builder and self.prompt_builder:
                self.logger.info(f"[{request_id}] Generating sophisticated prompt using PromptBuilder")
                
                # Use schema context AS-IS (now includes XML enrichment)
                sophisticated_result = await self.prompt_builder.build_sophisticated_prompt(
                    query,
                    schema_context,  # Now contains XML field mappings
                    request_id
                )
                
                return {
                    "success": True,
                    "prompt": sophisticated_result.get("prompt", ""),
                    "method": "sophisticated_prompt_builder",
                    "quality": sophisticated_result.get("quality", "sophisticated"),
                    "template_based": True,
                    "metadata": sophisticated_result.get("metadata", {}),
                    "schema_preservation": {
                        "tables_preserved": len(schema_context.get('tables', [])),
                        "columns_preserved": sum(len(cols) for cols in schema_context.get('columns_by_table', {}).values()),
                        "joins_preserved": len(schema_context.get('joins', [])),
                        "relationships_preserved": len(schema_context.get('relationships', [])),
                        "xml_fields_preserved": len(schema_context.get('xml_field_mappings', {})),  # UPDATED
                        "xml_handled": schema_context.get('xml_schema_loaded', False)  # UPDATED
                    }
                }
            else:
                # Dynamic basic fallback using discovered schema (with XML)
                basic_prompt = self._generate_basic_prompt(query, schema_context, domain_context)
                
                return {
                    "success": True,
                    "prompt": basic_prompt,
                    "method": "basic_fallback",
                    "quality": "basic",
                    "template_based": False,
                    "schema_preservation": {
                        "tables_preserved": len(schema_context.get('tables', [])),
                        "columns_preserved": sum(len(cols) for cols in schema_context.get('columns_by_table', {}).values()),
                        "joins_preserved": len(schema_context.get('joins', [])),
                        "relationships_preserved": len(schema_context.get('relationships', [])),
                        "xml_fields_preserved": len(schema_context.get('xml_field_mappings', {})),  # UPDATED
                        "xml_handled": schema_context.get('xml_schema_loaded', False)  # UPDATED
                    }
                }
                
        except Exception as e:
            self.logger.error(f"[{request_id}] Prompt generation failed: {e}")
            return {
                "success": False,
                "prompt": f"-- Error generating prompt: {e}\nSELECT 'prompt_error' as error;",
                "method": "error_fallback",
                "quality": "error",
                "error": str(e)
            }

    def _generate_basic_prompt(
        self,
        query: str,
        schema_context: Dict[str, Any],
        domain_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate basic prompt using ONLY discovered schema WITH XML INTEGRATION"""
        
        # Extract discovered schema information (now with XML)
        tables = schema_context.get('tables', [])
        columns_by_table = schema_context.get('columns_by_table', {})
        joins = schema_context.get('joins', [])
        relationships = schema_context.get('relationships', [])
        constraints = schema_context.get('constraints', [])
        
        # NEW: XML field information
        xml_enhanced_tables = schema_context.get('xml_enhanced_tables', {})
        xml_field_mappings = schema_context.get('xml_field_mappings', {})
        xml_schema_loaded = schema_context.get('xml_schema_loaded', False)
        
        prompt = f"""You are an expert SQL generator with XML field expertise. Generate accurate, efficient SQL based on the discovered database schema.

USER QUERY: {query}

DISCOVERED DATABASE SCHEMA:
"""
        
        # Add discovered tables and columns WITH XML INTEGRATION
        if tables:
            prompt += f"\nTABLES AND COLUMNS ({len(tables)} tables discovered):\n"
            
            for table in tables[:self.config.max_tables_in_prompt]:
                # Regular database columns
                if table in columns_by_table:
                    table_columns = columns_by_table[table]
                    if isinstance(table_columns, list):
                        columns = []
                        for col in table_columns[:self.config.max_columns_per_table]:
                            if isinstance(col, dict):
                                col_name = col.get('column', col.get('name', str(col)))
                                col_type = col.get('type', '')
                                if col_type:
                                    columns.append(f"{col_name} ({col_type})")
                                else:
                                    columns.append(col_name)
                            else:
                                columns.append(str(col))
                        
                        if len(table_columns) > self.config.max_columns_per_table:
                            columns.append(f"... and {len(table_columns) - self.config.max_columns_per_table} more columns")
                        
                        prompt += f"  {table} (Database Columns):\n    {', '.join(columns)}\n"
                    else:
                        prompt += f"  {table}: {table_columns}\n"
                
                # NEW: XML fields for this table
                if table in xml_enhanced_tables and xml_schema_loaded:
                    xml_data = xml_enhanced_tables[table]
                    xml_column = xml_data['xml_column']
                    xml_fields = xml_data['fields']
                    
                    prompt += f"  {table} (XML Fields in {xml_column}):\n"
                    for field in xml_fields[:self.config.max_xml_fields_in_prompt]:
                        field_name = field['name']
                        sql_expr = field['sql_expression']
                        prompt += f"    {field_name}: {sql_expr}\n"
                    
                    if len(xml_fields) > self.config.max_xml_fields_in_prompt:
                        prompt += f"    ... and {len(xml_fields) - self.config.max_xml_fields_in_prompt} more XML fields\n"

        # Add discovered joins (FIXED: using correct keys)
        if joins:
            prompt += f"\nAVAILABLE JOINS ({len(joins)} joins discovered):\n"
            for join in joins[:self.config.max_joins_in_prompt]:
                if isinstance(join, dict):
                    # FIXED: Use correct join keys
                    source_table = join.get('source_table', 'unknown')
                    source_column = join.get('source_column', 'unknown')
                    target_table = join.get('target_table', 'unknown')
                    target_column = join.get('target_column', 'unknown')
                    join_type = join.get('join_type', 'INNER')
                    
                    prompt += f"  {source_table}.{source_column} -> {target_table}.{target_column} ({join_type})\n"
                else:
                    prompt += f"  {join}\n"
            
            if len(joins) > self.config.max_joins_in_prompt:
                prompt += f"  ... and {len(joins) - self.config.max_joins_in_prompt} more joins available\n"

        # NEW: XML field usage instructions
        if xml_field_mappings and xml_schema_loaded:
            prompt += f"\nXML FIELD USAGE ({len(xml_field_mappings)} XML fields available):\n"
            prompt += "⚠️  XML fields require special SQL syntax:\n"
            
            # Show examples for first few XML fields
            example_count = 0
            for field_key, mapping in xml_field_mappings.items():
                if example_count >= 3:
                    break
                
                table = mapping['table']
                field_name = mapping['field_name']
                sql_expression = mapping['sql_expression']
                
                prompt += f"  {table}.{field_name}: {sql_expression}\n"
                example_count += 1
            
            if len(xml_field_mappings) > 3:
                prompt += f"  ... and {len(xml_field_mappings) - 3} more XML fields available\n"
            
            prompt += f"\nXML USAGE INSTRUCTIONS:\n"
            prompt += "- For XML fields, ALWAYS use the exact SQL expressions shown above\n"
            prompt += "- NEVER reference XML field names directly as columns\n"
            prompt += "- Use XML column with .value() method for XML field extraction\n"

        prompt += f"""

INSTRUCTIONS:
- Generate clean, efficient SQL using the discovered schema above
- Use EXACT table and column names as provided in the schema
- For XML fields, use the exact SQL expressions provided (DO NOT modify them)
- Include appropriate JOINs when querying multiple tables using the discovered relationships
- Optimize for performance while maintaining accuracy
- Return only the SQL query without explanations

SQL QUERY FOR: {query}
"""
        
        return prompt


class SchemaContextPreserver:
    """ENHANCED schema context preservation with XML integration"""
    
    def __init__(self, logger, xml_schema_manager: Optional['XMLSchemaManager'] = None): # pyright: ignore[reportInvalidTypeForm]
        self.logger = logger
        self.xml_schema_manager = xml_schema_manager  # NEW: XML Schema Manager

    def validate_and_preserve_schema(self, schema_data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """ENHANCED method for dynamic schema validation and preservation WITH XML INTEGRATION"""
        
        try:
            # Validate core fields (flexible validation)
            core_fields = ['tables']
            missing_core = [f for f in core_fields if f not in schema_data or not schema_data[f]]
            
            if missing_core:
                self.logger.warning(f"[{request_id}] Missing core schema fields: {missing_core}")
            
            # Still proceed if we have some schema data
            if not schema_data.get('tables'):
                return {"error": f"No tables found in schema data"}
            
            # NEW: Enrich with XML schema data if available
            if self.xml_schema_manager and self.xml_schema_manager.is_available():
                self.logger.info(f"[{request_id}] Enriching schema with XML field mappings...")
                try:
                    # Use XMLSchemaManager to enrich the schema
                    enriched_schema_data = self.xml_schema_manager.enrich_schema_context(schema_data)
                    self.logger.info(f"[{request_id}] XML enrichment completed: "
                                   f"{enriched_schema_data.get('xml_tables_count', 0)} XML tables, "
                                   f"{enriched_schema_data.get('xml_fields_total', 0)} XML fields")
                except Exception as e:
                    self.logger.warning(f"[{request_id}] XML enrichment failed: {e}")
                    enriched_schema_data = schema_data  # Fallback to original
            else:
                self.logger.debug(f"[{request_id}] XML Schema Manager not available - skipping XML enrichment")
                enriched_schema_data = schema_data
            
            # Preserve ALL discovered schema information dynamically (now with XML)
            preserved_schema = {
                # Core schema - always preserve
                'tables': enriched_schema_data.get('tables', []),
                'columns_by_table': enriched_schema_data.get('columns_by_table', {}),
                'total_columns': enriched_schema_data.get('total_columns', 0),
                
                # Relationships - preserve all types discovered
                'joins': enriched_schema_data.get('joins', []),
                'relationships': enriched_schema_data.get('relationships', []),
                'foreign_keys': enriched_schema_data.get('foreign_keys', []),
                'constraints': enriched_schema_data.get('constraints', []),
                
                # XML data - NEW: preserve XML mappings from XMLSchemaManager
                'xml_enhanced_tables': enriched_schema_data.get('xml_enhanced_tables', {}),
                'xml_field_mappings': enriched_schema_data.get('xml_field_mappings', {}),
                'xml_tables_count': enriched_schema_data.get('xml_tables_count', 0),
                'xml_fields_total': enriched_schema_data.get('xml_fields_total', 0),
                'xml_schema_loaded': enriched_schema_data.get('xml_schema_loaded', False),
                
                # Legacy XML fields (for backward compatibility)
                'xml_mappings': enriched_schema_data.get('xml_mappings', []),
                'xml_paths': enriched_schema_data.get('xml_paths', []),
                'has_xml_data': bool(enriched_schema_data.get('xml_enhanced_tables') or enriched_schema_data.get('xml_mappings')),
                
                # Quality and confidence metrics
                'confidence_scores': enriched_schema_data.get('confidence_scores', {}),
                'quality_metrics': enriched_schema_data.get('quality_metrics', {}),
                'schema_completeness': enriched_schema_data.get('schema_completeness', 0.0),
                
                # Source and method tracking
                'source_method': enriched_schema_data.get('source_method', 'unknown'),
                'discovery_method': enriched_schema_data.get('discovery_method', 'unknown'),
                'orchestrator_used': enriched_schema_data.get('search_orchestrator_used', False),
                'fallback_used': enriched_schema_data.get('fallback_used', False),
                
                # Additional discovered information
                'indexes': enriched_schema_data.get('indexes', []),
                'views': enriched_schema_data.get('views', []),
                'procedures': enriched_schema_data.get('procedures', []),
                'functions': enriched_schema_data.get('functions', []),
                
                # Metadata (ENHANCED with XML)
                'schema_size': {
                    'total_tables': len(enriched_schema_data.get('tables', [])),
                    'total_columns': sum(len(cols) for cols in enriched_schema_data.get('columns_by_table', {}).values()),
                    'total_joins': len(enriched_schema_data.get('joins', [])),
                    'total_xml_fields': enriched_schema_data.get('xml_fields_total', 0)  # NEW
                }
            }
            
            # Enhanced logging with XML information
            schema_size = preserved_schema['schema_size']
            self.logger.info(f"[{request_id}] Enhanced schema preserved: "
                           f"{schema_size['total_tables']} tables, "
                           f"{schema_size['total_columns']} columns, "
                           f"{schema_size['total_joins']} joins, "
                           f"{schema_size['total_xml_fields']} XML fields")
            
            return preserved_schema
            
        except Exception as e:
            self.logger.error(f"[{request_id}] Enhanced schema preservation failed: {e}")
            return {"error": f"Schema preservation failed: {e}"}


class NLPSchemaIntegrationBridge:
    """
    ENHANCED CENTRALIZED NLP-SCHEMA INTEGRATION BRIDGE - COMPLETE PIPELINE WITH XML
    
    SINGLE SOURCE OF TRUTH FOR:
    - Dynamic PromptBuilder integration
    - Discovered schema context preservation
    - Generic NLP-Schema-SQL COMPLETE pipeline coordination
    - XML field integration from xml_schema.json
    - NO hardcoded domain logic - works with ANY database
    - No duplication with other files
    - NOW INCLUDES SQL GENERATION WITH XML SUPPORT!
    """
    
    def __init__(
        self,
        nlp_processor: Optional[NLPProcessorProtocol] = None,
        schema_agent: Optional[SchemaAgentProtocol] = None,
        intelligent_agent: Optional[IntelligentAgentProtocol] = None,
        config: Optional[IntegrationConfig] = None,
        async_client_manager=None
    ):
        """Initialize centralized bridge with COMPLETE pipeline including XML integration"""
        
        self.logger = self._setup_logger()
        self.config = config or IntegrationConfig()
        self.async_client_manager = async_client_manager
        
        # CENTRALIZED components - single instances
        self.nlp_processor = nlp_processor
        self.schema_agent = schema_agent
        self.intelligent_agent = intelligent_agent
        
        # CENTRALIZED managers - eliminate duplication, NO domain processor
        self.async_manager = CentralizedAsyncManager()
        
        # NEW: Initialize XML Schema Manager
        self.xml_schema_manager = None
        if XML_SCHEMA_MANAGER_AVAILABLE and self.config.enable_xml_schema_integration:
            try:
                xml_path = self.config.xml_schema_path or "E:/Github/sql-ai-agent/data/metadata/xml_schema.json"
                self.xml_schema_manager = XMLSchemaManager(xml_path)  # pyright: ignore[reportOptionalCall]
                
                if self.xml_schema_manager.is_available():
                    stats = self.xml_schema_manager.get_statistics()
                    self.logger.info(f"XML Schema Manager initialized: {stats['tables_count']} tables, "
                                   f"{stats['xml_fields_count']} XML fields")
                else:
                    self.logger.warning("XML Schema Manager initialized but no schema data loaded")
                    
            except Exception as e:
                self.logger.error(f"XML Schema Manager initialization failed: {e}")
                self.xml_schema_manager = None
        else:
            self.logger.info("XML Schema Manager disabled or not available")
        
        # Enhanced schema preserver with XML integration
        self.schema_preserver = SchemaContextPreserver(self.logger, self.xml_schema_manager)
        
        # CENTRALIZED PromptBuilder service - SCHEMA-BASED ONLY (now with XML)
        self.prompt_builder_service = CentralizedPromptBuilder(async_client_manager, self.config)
        
        # NEW: CENTRALIZED Prompt-SQL Bridge service
        self.prompt_sql_bridge = None
        if PROMPT_SQL_BRIDGE_AVAILABLE and self.config.enable_sql_generation:
            try:
                self.prompt_sql_bridge = create_prompt_sql_bridge(async_client_manager)  # pyright: ignore[reportPossiblyUnboundVariable, reportOptionalCall]
                self.logger.info("Prompt-SQL Bridge initialized - complete pipeline enabled")
            except Exception as e:
                self.logger.error(f"Prompt-SQL Bridge initialization failed: {e}")
                self.prompt_sql_bridge = None
        else:
            self.logger.warning("Prompt-SQL Bridge not available - pipeline will stop at prompt generation")
        
        # SearchOrchestrator initialization
        self.search_orchestrator_manager = self._init_search_orchestrator()
        
        # State tracking
        self.intelligent_agent_failures: List[float] = []
        self.intelligent_agent_disabled = False
        self.circuit_breaker_open = False
        
        # CENTRALIZED statistics (enhanced with XML)
        self.statistics: Dict[str, RequestStatistics] = {
            'integration_bridge': RequestStatistics(),
            'nlp_processing': RequestStatistics(),
            'schema_discovery': RequestStatistics(),
            'prompt_building': RequestStatistics(),
            'sql_generation': RequestStatistics(),
            'xml_enrichment': RequestStatistics(),  # NEW
        }
        
        self.logger.info("=" * 80)
        self.logger.info("ENHANCED CENTRALIZED NLP-SCHEMA INTEGRATION BRIDGE v2.9.2 INITIALIZED")
        self.logger.info("COMPLETE PIPELINE: Schema → XML Enrichment → Prompt → SQL")
        self.logger.info("Dynamic schema-based processing - NO hardcoded domain logic")
        self.logger.info("Generic system works with ANY database schema")
        self.logger.info("XML field integration from xml_schema.json enabled")
        self.logger.info("Schema-driven prompt generation with SQL output")
        self.logger.info("FIXED: Double-nesting issue resolved")
        self.logger.info("=" * 80)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _init_search_orchestrator(self):
        """Initialize SearchOrchestrator with proper engines - WITH DEBUG LOGS"""
        
        # DEBUG LOGS ADDED
        self.logger.info(f"DEBUG: Initializing SearchOrchestrator with AsyncClientManager ID: {id(self.async_client_manager)}")
        self.logger.info(f"DEBUG: SEARCH_ORCHESTRATOR_AVAILABLE = {SEARCH_ORCHESTRATOR_AVAILABLE}")
        self.logger.info(f"DEBUG: use_search_orchestrator = {self.config.use_search_orchestrator}")
        self.logger.info(f"DEBUG: schema_agent available = {self.schema_agent is not None}")
        self.logger.info(f"DEBUG: intelligent_agent available = {self.intelligent_agent is not None}")
        
        if SEARCH_ORCHESTRATOR_AVAILABLE and self.config.use_search_orchestrator:
            try:
                search_engines = []
                if self.schema_agent:
                    search_engines.append(self.schema_agent)
                    self.logger.info(f"DEBUG: Added schema_agent to search_engines")
                
                if self.intelligent_agent:
                    search_engines.append(self.intelligent_agent)
                    self.logger.info(f"DEBUG: Added intelligent_agent to search_engines")
                
                # DEBUG LOG ADDED
                self.logger.info(f"DEBUG: search_engines count = {len(search_engines)}")
                
                if search_engines:
                    # DEBUG LOG ADDED
                    self.logger.info(f"DEBUG: Creating SearchOrchestrator with AsyncClientManager ID: {id(self.async_client_manager)}")
                    
                    orchestrator = SearchOrchestrator(  # pyright: ignore[reportPossiblyUnboundVariable]
                        engines=search_engines,  # pyright: ignore[reportArgumentType]
                        async_client_manager=self.async_client_manager,
                        convergence_threshold=self.config.search_orchestrator_convergence
                    )
                    
                    # DEBUG VALIDATION ADDED
                    self.logger.info(f"DEBUG: SearchOrchestrator created successfully")
                    
                    # ADDITIONAL VALIDATION
                    if hasattr(orchestrator, 'async_client_manager'):
                        if orchestrator.async_client_manager:
                            self.logger.info(f"DEBUG: SearchOrchestrator has AsyncClientManager (ID: {id(orchestrator.async_client_manager)})")
                        else:
                            self.logger.error(f"DEBUG: SearchOrchestrator has None AsyncClientManager")
                    else:
                        self.logger.error(f"DEBUG: SearchOrchestrator missing async_client_manager attribute")
                    
                    return orchestrator
                else:
                    self.logger.warning("DEBUG: No search engines available for SearchOrchestrator")
                    return None
                    
            except Exception as e:
                self.logger.error(f"DEBUG: SearchOrchestrator initialization failed: {e}")
                self.logger.error(f"DEBUG: Exception details: {traceback.format_exc()}")
                return None
        else:
            self.logger.warning(f"DEBUG: SearchOrchestrator creation skipped - Available: {SEARCH_ORCHESTRATOR_AVAILABLE}, Enabled: {self.config.use_search_orchestrator}")
            return None

    async def process_query_pipeline(
        self,
        query: str,
        request_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        ENHANCED COMPLETE dynamic query processing pipeline - SCHEMA → XML → PROMPT → SQL
        FIXED: Returns flattened structure to prevent double-nesting
        """
        
        try:
            self.logger.info(f"[{request_id}] Processing ENHANCED query pipeline with XML integration: {query[:50]}...")
            pipeline_start = time.time()
            
            # STEP 1: Discover ACTUAL schema dynamically (now with XML enrichment)
            self.logger.info(f"[{request_id}] Step 1: Dynamic schema discovery with XML enrichment starting...")
            schema_result = await self.discover_schema_with_preservation(query, request_id)
            
            if not schema_result.get('success'):
                self.statistics['integration_bridge'].increment_request(ProcessingResult.FAILURE)
                return {
                    "success": False,
                    "error": f"Schema discovery failed: {schema_result.get('error', 'unknown')}",
                    "request_id": request_id,
                    "stage_failed": "schema_discovery",
                    "pipeline_type": "complete_schema_xml_prompt_sql"  # UPDATED
                }
            
            schema_stats = schema_result['data'].get('schema_size', {})
            xml_stats = {
                'xml_tables': schema_result['data'].get('xml_tables_count', 0),
                'xml_fields': schema_result['data'].get('xml_fields_total', 0)
            }
            
            self.logger.info(f"[{request_id}] Step 1 completed: Schema discovered using {schema_result.get('method', 'unknown')} "
                           f"({schema_stats.get('total_tables', 0)} tables, {schema_stats.get('total_columns', 0)} columns, "
                           f"{xml_stats['xml_fields']} XML fields)")
            
            # Track XML enrichment
            if xml_stats['xml_fields'] > 0:
                self.statistics['xml_enrichment'].increment_request(ProcessingResult.SUCCESS)
            
            # STEP 2: Generate prompt using discovered schema WITH XML
            self.logger.info(f"[{request_id}] Step 2: Dynamic prompt generation with XML integration starting...")
            prompt_result = await self.generate_sophisticated_prompt_for_manager(
                query,
                schema_result['data'],
                "hybrid_orchestrator",
                request_id
            )
            
            if not prompt_result.get('success'):
                self.statistics['integration_bridge'].increment_request(ProcessingResult.FAILURE)
                return {
                    "success": False,
                    "error": f"Prompt generation failed: {prompt_result.get('error', 'unknown')}",
                    "request_id": request_id,
                    "stage_failed": "prompt_generation",
                    "pipeline_type": "complete_schema_xml_prompt_sql"  # UPDATED
                }
            
            self.logger.info(f"[{request_id}] Step 2 completed: Prompt generated using {prompt_result.get('method', 'unknown')}")
            
            # STEP 3: Generate SQL using centralized bridge (now XML-aware)
            sql_result = None
            if self.config.enable_sql_generation and self.prompt_sql_bridge:
                self.logger.info(f"[{request_id}] Step 3: SQL generation with XML support starting...")
                
                try:
                    # Create SQL generation request
                    sql_request = PromptSQLRequest(  # pyright: ignore[reportOptionalCall] # pyright: ignore[reportPossiblyUnboundVariable] # type: ignore
                        generated_prompt=prompt_result['prompt'],
                        schema_context=schema_result['data'],  # Now includes XML mappings
                        nlp_insights=schema_result['data'].get('nlp_insights', {}),
                        user_query=query,
                        target_model="mistral",
                        request_id=request_id
                    )
                    
                    # Generate SQL through bridge
                    sql_response = await self.prompt_sql_bridge.convert_prompt_to_sql(sql_request)
                    
                    if sql_response.success:
                        self.statistics['sql_generation'].increment_request(ProcessingResult.SUCCESS)
                        self.logger.info(f"[{request_id}] Step 3 completed: SQL generated successfully "
                                       f"(confidence: {sql_response.confidence:.2f})")
                        
                        sql_result = {
                            "success": True,
                            "generated_sql": sql_response.generated_sql,
                            "confidence": sql_response.confidence,
                            "processing_time_ms": sql_response.processing_time_ms,
                            "metadata": sql_response.metadata
                        }
                    else:
                        self.statistics['sql_generation'].increment_request(ProcessingResult.FAILURE)
                        self.logger.warning(f"[{request_id}] Step 3 failed: SQL generation error - {sql_response.error}")
                        
                        sql_result = {
                            "success": False,
                            "generated_sql": "",
                            "confidence": 0.0,
                            "error": sql_response.error
                        }
                        
                except Exception as e:
                    self.statistics['sql_generation'].increment_request(ProcessingResult.FAILURE)
                    self.logger.error(f"[{request_id}] Step 3 exception: SQL generation failed - {e}")
                    
                    sql_result = {
                        "success": False,
                        "generated_sql": "",
                        "confidence": 0.0,
                        "error": str(e)
                    }
            else:
                self.logger.warning(f"[{request_id}] Step 3 skipped: SQL generation disabled or bridge unavailable")
                sql_result = {
                    "success": False,
                    "generated_sql": "",
                    "confidence": 0.0,
                    "error": "SQL generation not enabled or bridge unavailable"
                }
            
            # STEP 4: Create COMPLETE result with schema, XML, prompt AND SQL
            total_processing_time = time.time() - pipeline_start
            self.logger.info(f"[{request_id}] ENHANCED pipeline finished in {total_processing_time:.2f}s")
            
            # Track overall success
            overall_success = (
                schema_result.get('success', False) and
                prompt_result.get('success', False) and
                (sql_result.get('success', False) if self.config.enable_sql_generation else True)
            )
            
            if overall_success:
                self.statistics['integration_bridge'].increment_request(ProcessingResult.SUCCESS, total_processing_time)
            else:
                self.statistics['integration_bridge'].increment_request(ProcessingResult.DEGRADED, total_processing_time)
            
            # CRITICAL FIX: Return FLATTENED structure to prevent double-nesting
            # This eliminates the nested 'data' key that was causing the extraction failure
            return {
                "success": overall_success,
                
                # DIRECT FIELDS - No nested 'data' key to prevent double-wrapping
                "query": query,
                "generated_sql": sql_result.get('generated_sql', ''),  # DIRECT ACCESS
                "sql_confidence": sql_result.get('confidence', 0.0),
                "schema_context": schema_result['data'],
                "prompt": prompt_result['prompt'],
                "prompt_quality": prompt_result.get('quality', 'unknown'),
                "schema_method": schema_result.get('method', 'unknown'),
                "prompt_method": prompt_result.get('method', 'unknown'),
                "sql_generation_enabled": self.config.enable_sql_generation,
                "complete_pipeline": True,
                "pipeline_type": "complete_schema_xml_prompt_sql",  # UPDATED
                "schema_preservation": prompt_result.get('schema_preservation', {}),
                "template_based": prompt_result.get('template_based', False),
                
                # NEW: XML integration status
                "xml_integration_enabled": self.config.enable_xml_schema_integration,
                "xml_schema_loaded": schema_result['data'].get('xml_schema_loaded', False),
                "xml_fields_available": schema_result['data'].get('xml_fields_total', 0),
                "xml_tables_count": schema_result['data'].get('xml_tables_count', 0),
                
                # Metadata as separate top-level fields
                "request_id": request_id,
                "total_processing_time": total_processing_time,
                "schema_processing_time": schema_result.get('processing_time', 0),
                "sql_processing_time_ms": sql_result.get('processing_time_ms', 0),
                "schema_discovered": schema_result['data'].get('schema_size', {}),
                "sql_metadata": sql_result.get('metadata', {}),
                "components_used": {
                    "schema_discovery": True,
                    "xml_enrichment": self.xml_schema_manager is not None and self.xml_schema_manager.is_available(),
                    "prompt_building": True,
                    "sql_generation": self.config.enable_sql_generation,
                    "dynamic_schema_processing": True,
                    "schema_preservation": True,
                    "hardcoded_domain_logic": False
                },
                "pipeline_stages": [
                    {"stage": "schema_discovery", "method": schema_result.get('method', 'unknown'), "success": True},
                    {"stage": "xml_enrichment", "method": "xml_schema_manager", "success": xml_stats['xml_fields'] > 0},
                    {"stage": "dynamic_prompt_generation", "method": prompt_result.get('method', 'unknown'), "success": True},
                    {"stage": "sql_generation", "method": "centralized_bridge", "success": sql_result.get('success', False)}
                ],
                
                # FIX INDICATOR
                "double_nesting_fixed": True,
                "extraction_path": "result.data['generated_sql']",  # Now accessible directly
                "xml_integration_version": "2.9.2"  # NEW
            }
            
        except Exception as e:
            self.logger.error(f"[{request_id}] ENHANCED pipeline processing failed: {e}")
            self.logger.error(f"[{request_id}] Exception details: {traceback.format_exc()}")
            
            # Track failure
            self.statistics['integration_bridge'].increment_request(ProcessingResult.FAILURE)
            
            return {
                "success": False,
                "error": str(e),
                "request_id": request_id,
                "stage_failed": "pipeline_exception",
                "pipeline_type": "complete_schema_xml_prompt_sql",  # UPDATED
                "fallback_available": True,
                "exception_type": type(e).__name__
            }

    async def generate_sophisticated_prompt_for_manager(
        self,
        query: str,
        schema_context: Dict[str, Any],
        manager_type: str = "unknown",
        request_id: str = "unknown"
    ) -> Dict[str, Any]:
        """ENHANCED service using discovered schema WITH XML - NO domain assumptions"""
        
        # Preserve schema context as-is from discovery (now includes XML enrichment)
        preserved_schema = self.schema_preserver.validate_and_preserve_schema(schema_context, request_id)
        
        if "error" in preserved_schema:
            return {
                "success": False,
                "error": preserved_schema["error"],
                "method": "schema_preservation_failed",
                "quality": "error"
            }
        
        # Generate prompt using actual discovered schema WITH XML
        prompt_result = await self.prompt_builder_service.generate_sophisticated_prompt(
            query, preserved_schema, None, request_id  # No domain context
        )
        
        # Track statistics
        if prompt_result.get('success'):
            self.statistics['prompt_building'].increment_request(ProcessingResult.SUCCESS)
        else:
            self.statistics['prompt_building'].increment_request(ProcessingResult.FAILURE)
        
        # Add manager context
        prompt_result['requesting_manager'] = manager_type
        prompt_result['centralized_service'] = True
        prompt_result['dynamic_schema_based'] = True
        prompt_result['xml_enhanced'] = preserved_schema.get('xml_schema_loaded', False)  # NEW
        
        self.logger.info(f"[{request_id}] Generated sophisticated prompt for {manager_type} manager: "
                        f"quality={prompt_result.get('quality', 'unknown')}, "
                        f"method={prompt_result.get('method', 'unknown')}, "
                        f"xml_enhanced={prompt_result.get('xml_enhanced', False)}")
        
        return prompt_result

    async def discover_schema_with_preservation(
        self,
        query: str,
        request_id: str = "unknown"
    ) -> Dict[str, Any]:
        """ENHANCED schema discovery with full preservation AND XML ENRICHMENT"""
        
        start_time = time.time()
        
        try:
            # Try SearchOrchestrator first
            if self.search_orchestrator_manager:
                try:
                    schema_result = await self.search_orchestrator_manager.search_schema(query)
                    
                    if schema_result and hasattr(schema_result, 'results') and schema_result.results:
                        # Schema preservation now includes XML enrichment
                        preserved_schema = self.schema_preserver.validate_and_preserve_schema(
                            schema_result.to_dict(), request_id  # pyright: ignore[reportAttributeAccessIssue]
                        )
                        
                        if "error" not in preserved_schema:
                            self.statistics['schema_discovery'].increment_request(ProcessingResult.SUCCESS)
                            return {
                                "success": True,
                                "data": preserved_schema,
                                "method": "search_orchestrator",
                                "processing_time": time.time() - start_time,
                                "schema_source": "orchestrated_search"
                            }
                            
                except Exception as e:
                    self.logger.warning(f"[{request_id}] SearchOrchestrator failed: {e}")
            
            # Fallback to intelligent agent
            if self.intelligent_agent and not self.intelligent_agent_disabled:
                try:
                    schema_result = await self.async_manager.safe_async_call(
                        self.intelligent_agent.retrieve_complete_schema_json,
                        {"query": query},
                        timeout=self.config.schema_timeout
                    )
                    
                    if schema_result:
                        schema_data = schema_result.to_dict() if hasattr(schema_result, 'to_dict') else schema_result
                        # Schema preservation now includes XML enrichment
                        preserved_schema = self.schema_preserver.validate_and_preserve_schema(
                            schema_data, request_id
                        )
                        
                        if "error" not in preserved_schema:
                            self.statistics['schema_discovery'].increment_request(ProcessingResult.SUCCESS)
                            return {
                                "success": True,
                                "data": preserved_schema,
                                "method": "intelligent_agent",
                                "processing_time": time.time() - start_time,
                                "schema_source": "intelligent_retrieval"
                            }
                            
                except Exception as e:
                    self.logger.warning(f"[{request_id}] Intelligent agent failed: {e}")
                    self.intelligent_agent_failures.append(time.time())
            
            # Final fallback to basic schema agent
            if self.schema_agent:
                try:
                    schema_result = await self.async_manager.safe_async_call(
                        self.schema_agent.retrieve_complete_schema,
                        query,
                        timeout=self.config.schema_timeout
                    )
                    
                    if schema_result:
                        # Schema preservation now includes XML enrichment
                        preserved_schema = self.schema_preserver.validate_and_preserve_schema(
                            schema_result, request_id
                        )
                        
                        if "error" not in preserved_schema:
                            self.statistics['schema_discovery'].increment_request(ProcessingResult.SUCCESS)
                            return {
                                "success": True,
                                "data": preserved_schema,
                                "method": "basic_schema_agent",
                                "processing_time": time.time() - start_time,
                                "schema_source": "basic_retrieval"
                            }
                            
                except Exception as e:
                    self.logger.error(f"[{request_id}] Schema agent failed: {e}")
            
            # No fallback - fail for accuracy
            raise ValueError("All schema discovery methods failed")
            
        except Exception as e:
            self.logger.error(f"[{request_id}] Schema discovery completely failed: {e}")
            self.statistics['schema_discovery'].increment_request(ProcessingResult.FAILURE)
            
            return {
                "success": False,
                "error": str(e),
                "method": "all_failed",
                "processing_time": time.time() - start_time,
                "schema_source": "none"
            }

    def health_check(self) -> Dict[str, Any]:
        """ENHANCED health check for COMPLETE pipeline system WITH XML"""
        
        xml_manager_status = None
        if self.xml_schema_manager:
            xml_manager_status = self.xml_schema_manager.get_statistics()
        
        return {
            "status": "healthy",
            "component": "EnhancedCentralizedNLPSchemaBridge",
            "version": "2.9.2",
            "architecture_type": "complete_pipeline_schema_xml_prompt_sql",  # UPDATED
            "fixes_applied": {
                "double_nesting_issue": "FIXED - flattened response structure",
                "extraction_path": "result.data['generated_sql'] now works correctly",
                "xml_integration": "ADDED - XML field processing from xml_schema.json"
            },
            "centralized_services": {
                "prompt_builder": self.prompt_builder_service.use_prompt_builder,
                "schema_preservation": True,
                "async_management": True,
                "sql_generation": self.prompt_sql_bridge is not None,
                "xml_schema_manager": self.xml_schema_manager is not None and self.xml_schema_manager.is_available(),
                "hardcoded_domain_logic": False
            },
            "components": {
                "nlp_processor": self.nlp_processor is not None,
                "schema_agent": self.schema_agent is not None,
                "intelligent_agent": self.intelligent_agent is not None and not self.intelligent_agent_disabled,
                "search_orchestrator": self.search_orchestrator_manager is not None,
                "prompt_sql_bridge": self.prompt_sql_bridge is not None,
                "xml_schema_manager": self.xml_schema_manager is not None
            },
            "pipeline_capabilities": {
                "schema_discovery": True,
                "xml_field_integration": self.xml_schema_manager is not None and self.xml_schema_manager.is_available(),
                "prompt_generation": True,
                "sql_generation": self.config.enable_sql_generation and self.prompt_sql_bridge is not None,
                "complete_pipeline": self.config.enable_sql_generation and self.prompt_sql_bridge is not None
            },
            "xml_integration": {
                "enabled": self.config.enable_xml_schema_integration,
                "manager_available": XML_SCHEMA_MANAGER_AVAILABLE,
                "manager_loaded": self.xml_schema_manager is not None,
                "schema_loaded": self.xml_schema_manager.is_available() if self.xml_schema_manager else False,
                "statistics": xml_manager_status
            },
            "statistics": {
                name: {
                    "total_requests": stats.total_requests,
                    "success_rate": stats.calculate_success_rate(),
                    "avg_response_time": stats.avg_response_time
                }
                for name, stats in self.statistics.items()
            },
            "configuration": {
                "enable_sql_generation": self.config.enable_sql_generation,
                "enable_xml_schema_integration": self.config.enable_xml_schema_integration,
                "xml_enrichment_enabled": self.config.xml_enrichment_enabled,
                "max_tables_in_prompt": self.config.max_tables_in_prompt,
                "max_columns_per_table": self.config.max_columns_per_table,
                "max_joins_in_prompt": self.config.max_joins_in_prompt,
                "max_xml_fields_in_prompt": self.config.max_xml_fields_in_prompt
            },
            "no_duplication": True,
            "generic_database_support": True,
            "xml_field_support": True,  # NEW
            "complete_pipeline_enabled": self.config.enable_sql_generation and self.prompt_sql_bridge is not None,
            "timestamp": datetime.now().isoformat()
        }


# BACKWARD COMPATIBILITY: Keep both function names
def create_centralized_integration_bridge(
    nlp_processor: Optional[NLPProcessorProtocol] = None,
    schema_agent: Optional[SchemaAgentProtocol] = None,
    intelligent_agent: Optional[IntelligentAgentProtocol] = None,
    config: Optional[IntegrationConfig] = None,
    async_client_manager=None
) -> NLPSchemaIntegrationBridge:
    """BACKWARD COMPATIBILITY: Create centralized integration bridge"""
    
    return NLPSchemaIntegrationBridge(
        nlp_processor=nlp_processor,
        schema_agent=schema_agent,
        intelligent_agent=intelligent_agent,
        config=config,
        async_client_manager=async_client_manager
    )


def create_dynamic_integration_bridge(
    nlp_processor: Optional[NLPProcessorProtocol] = None,
    schema_agent: Optional[SchemaAgentProtocol] = None,
    intelligent_agent: Optional[IntelligentAgentProtocol] = None,
    config: Optional[IntegrationConfig] = None,
    async_client_manager=None
) -> NLPSchemaIntegrationBridge:
    """NEW: Create dynamic integration bridge - COMPLETE PIPELINE with XML integration"""
    
    return NLPSchemaIntegrationBridge(
        nlp_processor=nlp_processor,
        schema_agent=schema_agent,
        intelligent_agent=intelligent_agent,
        config=config,
        async_client_manager=async_client_manager
    )


# Exports for other files - KEEP BACKWARD COMPATIBILITY
__all__ = [
    'NLPSchemaIntegrationBridge',
    'CentralizedPromptBuilder',
    'SchemaContextPreserver',
    'IntegrationConfig',
    'ProcessingResult',
    'create_centralized_integration_bridge',  # KEEP for backward compatibility
    'create_dynamic_integration_bridge'
]


# Enhanced test function
async def test_complete_pipeline():
    """Test the COMPLETE pipeline including XML integration"""
    
    print("Testing ENHANCED NLP-Schema Integration Bridge v2.9.2 with XML integration...")
    
    try:
        config = IntegrationConfig(
            enable_sql_generation=True,  # ENSURE SQL generation is enabled
            use_prompt_builder=True,
            use_search_orchestrator=True,
            enable_xml_schema_integration=True,  # NEW: Enable XML integration
            xml_enrichment_enabled=True,  # NEW
            max_tables_in_prompt=10,
            max_columns_per_table=8,
            xml_schema_path="E:/Github/sql-ai-agent/data/metadata/xml_schema.json"  # NEW
        )
        
        bridge = NLPSchemaIntegrationBridge(config=config)
        health = bridge.health_check()
        
        print(f"Health: {health['status']}")
        print(f"Architecture: {health['architecture_type']}")
        print(f"SQL Generation: {health['pipeline_capabilities']['sql_generation']}")
        print(f"XML Integration: {health['pipeline_capabilities']['xml_field_integration']}")
        print(f"Complete Pipeline: {health['pipeline_capabilities']['complete_pipeline']}")
        print(f"Double Nesting Fix: {health['fixes_applied']['double_nesting_issue']}")
        print(f"XML Integration Fix: {health['fixes_applied']['xml_integration']}")
        
        # Test complete pipeline with XML support
        result = await bridge.process_query_pipeline("show customer collateral details", "test_001")
        
        print(f"Pipeline Success: {result['success']}")
        if result['success']:
            print(f"Generated SQL: {result.get('generated_sql', 'NO SQL GENERATED')}")
            print(f"SQL Confidence: {result.get('sql_confidence', 0.0)}")
            print(f"XML Fields Available: {result.get('xml_fields_available', 0)}")
            print(f"XML Tables Count: {result.get('xml_tables_count', 0)}")
            print(f"Double Nesting Fixed: {result.get('double_nesting_fixed', False)}")
            print(f"XML Integration Version: {result.get('xml_integration_version', 'Unknown')}")
        
        print("ENHANCED COMPLETE PIPELINE with XML integration test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_complete_pipeline())
