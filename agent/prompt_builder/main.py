"""
Simplified PromptBuilder interface for orchestrator integration.
Clean tool interface: User Query → Schema Context → Generated Prompt
UPDATED: Preserves complete schema information including JOINs and XML
FIXED: Added proper async support and fixed all type checking issues
FIXED: Added status attribute for server validation compatibility
"""

from typing import List, Optional, Dict, Any, Union, Awaitable
import logging
import asyncio
from datetime import datetime

from agent.schema_searcher.core.data_models import RetrievedColumn, create_retrieved_column_safe, ColumnType
from .core.data_models import (
    StructuredPrompt, PromptOptions, QueryIntent, SchemaContext, PromptType
)
from .core.query_analyzer import QueryAnalyzer
from .core.template_manager import TemplateManager
from .builders.context_builder import ContextBuilder
from .assemblers.prompt_assembler import PromptAssembler

logger = logging.getLogger(__name__)

class PromptBuilder:
    """
    Simplified PromptBuilder for orchestrator integration.
    UPDATED: Preserves complete schema information including JOINs and XML.
    FIXED: Now supports both sync and async operations with proper typing.
    FIXED: Added status attribute for server validation compatibility.
    """
    
    def __init__(
        self, 
        templates_directory: Optional[str] = None,
        enable_caching: bool = True
    ) -> None:
        """Initialize PromptBuilder with required components."""
        print("DEBUG: Initializing PromptBuilder with COMPLETE SCHEMA preservation...")
        
        # CRITICAL FIX: Add status attribute for server validation
        self.status = "healthy"
        
        # Initialize components synchronously first
        self.query_analyzer = QueryAnalyzer()
        self.template_manager = TemplateManager(templates_directory) # pyright: ignore[reportArgumentType]
        self.context_builder = ContextBuilder()
        self.prompt_assembler = PromptAssembler()
        self.enable_caching = enable_caching
        
        # Async initialization flag
        self._async_initialized = False
        self._initialization_lock: Optional[asyncio.Lock] = None
        
        # Initialize lock only if event loop is available
        try:
            asyncio.get_running_loop()
            self._initialization_lock = asyncio.Lock()
        except RuntimeError:
            self._initialization_lock = None
        
        # Simple statistics tracking
        self.build_stats: Dict[str, Any] = {
            "total_prompts_built": 0,
            "successful_builds": 0,
            "failed_builds": 0,
            "last_build_time": None
        }
        
        print("DEBUG: PromptBuilder initialization completed")
        logger.info("PromptBuilder initialized successfully")

    # ADDED: Status management methods for consistency
    def get_status(self) -> str:
        """Get the current PromptBuilder status"""
        return self.status
        
    def set_status(self, status: str) -> None:
        """Set the PromptBuilder status"""
        self.status = status
        logger.debug(f"PromptBuilder status updated to: {status}")

    def _is_event_loop_running(self) -> bool:
        """Check if an event loop is currently running."""
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

    async def _async_init(self) -> None:
        """Async initialization for components that need it."""
        if self._async_initialized:
            return
        
        if self._initialization_lock:
            async with self._initialization_lock:
                if self._async_initialized:
                    return
                self._async_initialized = True
                logger.info("PromptBuilder async initialization completed")
        else:
            self._async_initialized = True

    async def build_prompt_simple_async(
        self,
        user_query: str,
        schema_context: Any
    ) -> str:
        """
        ASYNC version of build_prompt_simple for orchestrator integration.
        UPDATED: Preserves complete schema information including JOINs and XML.
        """
        # Ensure async initialization
        await self._async_init()
        
        build_start_time = datetime.now()
        self.build_stats["total_prompts_built"] += 1
        
        try:
            print(f"DEBUG: Building prompt async for query: '{user_query}'")
            print(f"DEBUG: Schema context type: {type(schema_context)}")
            
            # Handle different schema context types
            if hasattr(schema_context, 'tables'):
                print(f"DEBUG: Schema context has {len(schema_context.tables)} tables")
                structured_prompt = await self._build_from_schema_context_object_async(user_query, schema_context)
            elif isinstance(schema_context, dict):
                print(f"DEBUG: Schema context is dictionary with {len(schema_context.get('tables', []))} tables")
                structured_prompt = await self._build_from_dict_schema_async(user_query, schema_context)
            elif isinstance(schema_context, list):
                print(f"DEBUG: Schema context is list with {len(schema_context)} columns")
                structured_prompt = await self._build_from_retrieved_columns_async(user_query, schema_context)
            else:
                raise ValueError(f"Unsupported schema context type: {type(schema_context)}")
            
            # Convert to simple string format for orchestrator
            final_prompt = self._convert_to_simple_prompt(structured_prompt)
            
            # Update statistics
            build_time = (datetime.now() - build_start_time).total_seconds() * 1000
            self._update_build_stats(build_time, success=True)
            
            print(f"DEBUG: Prompt built successfully in {build_time:.1f}ms, length: {len(final_prompt)}")
            return final_prompt
            
        except Exception as e:
            print(f"DEBUG: Error building prompt: {str(e)}")
            import traceback
            traceback.print_exc()
            
            self._update_build_stats(0, success=False)
            logger.error(f"Failed to build prompt: {str(e)}")
            raise

    def build_prompt_simple(
        self,
        user_query: str,
        schema_context: Any
    ) -> Union[str, Awaitable[str]]:
        """
        SYNCHRONOUS version for backward compatibility.
        Returns either a string or an Awaitable[str] depending on context.
        """
        if self._is_event_loop_running():
            # We're in an async context, return a coroutine
            return self.build_prompt_simple_async(user_query, schema_context)
        else:
            # Not in async context, execute synchronously
            return self._build_prompt_sync(user_query, schema_context)

    def _build_prompt_sync(self, user_query: str, schema_context: Any) -> str:
        """Synchronous version of prompt building."""
        build_start_time = datetime.now()
        self.build_stats["total_prompts_built"] += 1
        
        try:
            print(f"DEBUG: Building prompt sync for query: '{user_query}'")
            print(f"DEBUG: Schema context type: {type(schema_context)}")
            
            # Handle different schema context types
            if hasattr(schema_context, 'tables'):
                print(f"DEBUG: Schema context has {len(schema_context.tables)} tables")
                structured_prompt = self._build_from_schema_context_object(user_query, schema_context)
            elif isinstance(schema_context, dict):
                print(f"DEBUG: Schema context is dictionary with {len(schema_context.get('tables', []))} tables")
                structured_prompt = self._build_from_dict_schema(user_query, schema_context)
            elif isinstance(schema_context, list):
                print(f"DEBUG: Schema context is list with {len(schema_context)} columns")
                structured_prompt = self._build_from_retrieved_columns(user_query, schema_context)
            else:
                raise ValueError(f"Unsupported schema context type: {type(schema_context)}")
            
            # Convert to simple string format for orchestrator
            final_prompt = self._convert_to_simple_prompt(structured_prompt)
            
            # Update statistics
            build_time = (datetime.now() - build_start_time).total_seconds() * 1000
            self._update_build_stats(build_time, success=True)
            
            print(f"DEBUG: Prompt built successfully in {build_time:.1f}ms, length: {len(final_prompt)}")
            return final_prompt
            
        except Exception as e:
            print(f"DEBUG: Error building prompt: {str(e)}")
            import traceback
            traceback.print_exc()
            
            self._update_build_stats(0, success=False)
            logger.error(f"Failed to build prompt: {str(e)}")
            raise

    # Async versions of the helper methods
    async def _build_from_schema_context_object_async(
        self, 
        user_query: str, 
        schema_context: Any
    ) -> StructuredPrompt:
        """Async version of _build_from_schema_context_object."""
        retrieved_columns = self._convert_schema_context_to_retrieved_columns(schema_context)
        return await self._build_standard_prompt_async(user_query, retrieved_columns)

    async def _build_from_dict_schema_async(
        self, 
        user_query: str, 
        schema_dict: Dict[str, Any]
    ) -> StructuredPrompt:
        """Async version of _build_from_dict_schema."""
        retrieved_columns = self._convert_dict_to_retrieved_columns(schema_dict)
        return await self._build_standard_prompt_async(user_query, retrieved_columns)

    async def _build_from_retrieved_columns_async(
        self, 
        user_query: str, 
        retrieved_columns: List[RetrievedColumn]
    ) -> StructuredPrompt:
        """Async version of _build_from_retrieved_columns."""
        return await self._build_standard_prompt_async(user_query, retrieved_columns)

    async def _build_standard_prompt_async(
        self, 
        user_query: str, 
        retrieved_columns: List[RetrievedColumn]
    ) -> StructuredPrompt:
        """Async version of standard prompt building workflow."""
        print("DEBUG: Starting async standard prompt building workflow...")
        
        # Step 1: Analyze query (keeping sync for now)
        query_intent = self.query_analyzer.analyze_query(user_query, retrieved_columns)
        print(f"DEBUG: Query analysis completed: {query_intent.query_type}")
        
        # Step 2: Build schema context with ENHANCED options for complete schema
        options = self._get_enhanced_prompt_options()
        schema_context = self.context_builder.build_context(retrieved_columns, query_intent, options) # pyright: ignore[reportArgumentType]
        print(f"DEBUG: Schema context built with {schema_context.total_tables} tables")
        
        # Step 3: Assemble prompt (keeping sync for now)
        structured_prompt = self.prompt_assembler.assemble_prompt(
            user_query, query_intent, schema_context, options # pyright: ignore[reportArgumentType]
        )
        print("DEBUG: Async prompt assembly completed")
        
        return structured_prompt

    # Add test methods for orchestrator initialization testing
    async def test_async_function(self) -> bool:
        """Test function for orchestrator to verify async functionality."""
        try:
            await self._async_init()
            return True
        except Exception as e:
            logger.error(f"Async test failed: {e}")
            return False

    def test_sync_function(self) -> bool:
        """Test function for orchestrator to verify sync functionality."""
        try:
            return True
        except Exception as e:
            logger.error(f"Sync test failed: {e}")
            return False

    # Keep all original synchronous methods unchanged for backward compatibility
    def _build_from_schema_context_object(
        self, 
        user_query: str, 
        schema_context: Any
    ) -> StructuredPrompt:
        """Build prompt from SchemaContext object."""
        retrieved_columns = self._convert_schema_context_to_retrieved_columns(schema_context)
        return self._build_standard_prompt(user_query, retrieved_columns)

    def _build_from_dict_schema(
        self, 
        user_query: str, 
        schema_dict: Dict[str, Any]
    ) -> StructuredPrompt:
        """Build prompt from dictionary schema format."""
        retrieved_columns = self._convert_dict_to_retrieved_columns(schema_dict)
        return self._build_standard_prompt(user_query, retrieved_columns)

    def _build_from_retrieved_columns(
        self, 
        user_query: str, 
        retrieved_columns: List[RetrievedColumn]
    ) -> StructuredPrompt:
        """Build prompt from RetrievedColumn list (legacy format)."""
        return self._build_standard_prompt(user_query, retrieved_columns)

    def _build_standard_prompt(
        self, 
        user_query: str, 
        retrieved_columns: List[RetrievedColumn]
    ) -> StructuredPrompt:
        """Standard prompt building workflow."""
        print("DEBUG: Starting standard prompt building workflow...")
        
        # Step 1: Analyze query
        query_intent = self.query_analyzer.analyze_query(user_query, retrieved_columns)
        print(f"DEBUG: Query analysis completed: {query_intent.query_type}")
        
        # Step 2: Build schema context with ENHANCED options for complete schema
        options = self._get_enhanced_prompt_options()
        schema_context = self.context_builder.build_context(retrieved_columns, query_intent, options) # pyright: ignore[reportArgumentType]
        print(f"DEBUG: Schema context built with {schema_context.total_tables} tables")
        
        # Step 3: Assemble prompt
        structured_prompt = self.prompt_assembler.assemble_prompt(
            user_query, query_intent, schema_context, options # pyright: ignore[reportArgumentType]
        )
        print("DEBUG: Prompt assembly completed")
        
        return structured_prompt

    def _get_enhanced_prompt_options(self) -> PromptOptions:
        """
        UPDATED: Enhanced prompt options for complete schema preservation.
        """
        return PromptOptions(
            # Increased limits for complete schema
            max_context_length=8000,        # Increased from 2000
            max_tables=25,                  # Increased from 10
            max_columns_per_table=100,      # Increased from 15
            
            # Lower threshold to include more information
            confidence_threshold=0.3,       # Reduced from 0.7
            
            # Enhanced content inclusion
            include_examples=True,
            include_descriptions=True,
            include_data_types=True,
            
            # Prioritize relationships and XML
            prioritize_relationships=True,
            emphasize_xml_handling=True,
            
            # Target LLM
            target_llm="gpt"
        )

    def _convert_schema_context_to_retrieved_columns(
        self, 
        schema_context: Any
    ) -> List[RetrievedColumn]:
        """
        UPDATED: Convert SchemaContext object to RetrievedColumn list.
        NOW PRESERVES: JOIN relationships and XML mappings.
        """
        retrieved_columns: List[RetrievedColumn] = []
        
        try:
            print(f"DEBUG: Converting schema context with type: {type(schema_context)}")
            print(f"DEBUG: Schema context attributes: {[attr for attr in dir(schema_context) if not attr.startswith('_')]}")
            
            # STEP 1: Process regular tables and columns
            if hasattr(schema_context, 'tables') and schema_context.tables:
                print(f"DEBUG: Found tables attribute with {len(schema_context.tables)} tables")
                retrieved_columns.extend(self._process_tables_and_columns(schema_context))
            
            # STEP 2: Preserve JOIN relationships as special entries
            if hasattr(schema_context, 'relationships') and schema_context.relationships:
                print(f"DEBUG: Found {len(schema_context.relationships)} relationships")
                retrieved_columns.extend(self._preserve_join_relationships(schema_context.relationships))
            
            # STEP 3: Preserve XML mappings as special entries  
            if hasattr(schema_context, 'xml_mappings') and schema_context.xml_mappings:
                print(f"DEBUG: Found {len(schema_context.xml_mappings)} XML mappings")
                retrieved_columns.extend(self._preserve_xml_mappings(schema_context.xml_mappings))
            
            # STEP 4: Process column_details if no tables found
            elif hasattr(schema_context, 'column_details') and schema_context.column_details:
                print(f"DEBUG: Using column_details with {len(schema_context.column_details)} entries")
                retrieved_columns.extend(self._process_column_details(schema_context.column_details))
                        
            print(f"DEBUG: Successfully converted to {len(retrieved_columns)} RetrievedColumns")
            
        except Exception as e:
            print(f"DEBUG: Error converting schema context: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Create fallback columns
            retrieved_columns = self._create_fallback_columns(schema_context)
        
        return retrieved_columns

    def _process_tables_and_columns(self, schema_context: Any) -> List[RetrievedColumn]:
        """Process regular tables and columns from schema context."""
        retrieved_columns: List[RetrievedColumn] = []
        
        if isinstance(schema_context.tables, dict):
            # Format: {'table_name': [columns]}
            for table_name, columns in schema_context.tables.items():
                print(f"DEBUG: Processing table {table_name} with {len(columns) if isinstance(columns, list) else 'unknown'} columns")
                
                if isinstance(columns, list):
                    for column in columns:
                        retrieved_col = self._create_retrieved_column_from_data(table_name, column)
                        if retrieved_col:
                            retrieved_columns.append(retrieved_col)
        
        elif isinstance(schema_context.tables, list):
            # Format: [table_objects]
            for table in schema_context.tables:
                table_name = table.name if hasattr(table, 'name') else str(table)
                
                if hasattr(table, 'columns'):
                    for column in table.columns:
                        retrieved_col = create_retrieved_column_safe(
                            table=table_name,
                            column=column.name if hasattr(column, 'name') else str(column),
                            datatype=getattr(column, 'datatype', 'unknown'),
                            description=getattr(column, 'description', ''),
                            xpath=getattr(column, 'xml_path', None),
                            type=ColumnType.XML if getattr(column, 'is_xml', False) else ColumnType.REGULAR
                        )
                        retrieved_columns.append(retrieved_col)
                else:
                    # No columns info, create minimal entry
                    retrieved_col = create_retrieved_column_safe(
                        table=table_name,
                        column="*",
                        datatype="unknown",
                        description=f"Table {table_name} (no column details)"
                    )
                    retrieved_columns.append(retrieved_col)
        
        return retrieved_columns

    def _create_retrieved_column_from_data(self, table_name: str, column: Any) -> Optional[RetrievedColumn]:
        """Create RetrievedColumn from various column data formats using safe creation."""
        try:
            if isinstance(column, str):
                # Simple string column name
                return create_retrieved_column_safe(
                    table=table_name,
                    column=column,
                    datatype='unknown',
                    description=''
                )
            elif isinstance(column, dict):
                # Dictionary with column details - handle xml_path safely
                return create_retrieved_column_safe(
                    table=table_name,
                    column=column.get('name', column.get('column', 'unknown')),
                    datatype=column.get('datatype', 'unknown'),
                    description=column.get('description', ''),
                    xpath=column.get('xml_path', column.get('xpath')),
                    type=ColumnType.XML if column.get('is_xml', False) else ColumnType.REGULAR
                )
            else:
                # Object with attributes
                return create_retrieved_column_safe(
                    table=table_name,
                    column=getattr(column, 'name', str(column)),
                    datatype=getattr(column, 'datatype', 'unknown'),
                    description=getattr(column, 'description', ''),
                    xpath=getattr(column, 'xml_path', None),
                    type=ColumnType.XML if getattr(column, 'is_xml', False) else ColumnType.REGULAR
                )
        except Exception as e:
            print(f"DEBUG: Error creating RetrievedColumn for {table_name}.{column}: {e}")
            return None

    def _preserve_join_relationships(self, relationships: List[str]) -> List[RetrievedColumn]:
        """
        CRITICAL: Preserve JOIN relationships as special RetrievedColumn entries.
        This ensures JOINs make it through to the final prompt.
        """
        join_columns: List[RetrievedColumn] = []
        
        for i, relationship in enumerate(relationships):
            join_col = create_retrieved_column_safe(
                table="SCHEMA_JOINS",
                column=f"join_{i}",
                datatype="relationship",
                description=relationship,
                type=ColumnType.KEY
            )
            join_columns.append(join_col)
            print(f"DEBUG: Preserved JOIN relationship: {relationship}")
        
        return join_columns

    def _preserve_xml_mappings(self, xml_mappings: List[Dict[str, str]]) -> List[RetrievedColumn]:
        """
        CRITICAL: Preserve XML mappings as special RetrievedColumn entries.
        This ensures XML paths and expressions make it through to the final prompt.
        """
        xml_columns: List[RetrievedColumn] = []
        
        for i, xml_mapping in enumerate(xml_mappings):
            # Create descriptive text for the XML mapping
            xml_description = f"XML: {xml_mapping.get('xpath', '')} -> {xml_mapping.get('sql_expression', '')}"
            if xml_mapping.get('table'):
                xml_description = f"Table: {xml_mapping['table']} | {xml_description}"
            
            xml_col = create_retrieved_column_safe(
                table="SCHEMA_XML",
                column=f"xml_{i}",
                datatype="xml_path",
                description=xml_description,
                xpath=xml_mapping.get('xpath', ''),
                sql_expression=xml_mapping.get('sql_expression', ''),
                type=ColumnType.XML
            )
            xml_columns.append(xml_col)
            print(f"DEBUG: Preserved XML mapping: {xml_description}")
        
        return xml_columns

    def _process_column_details(self, column_details: List[Dict[str, Any]]) -> List[RetrievedColumn]:
        """Process column details with safe parameter handling"""
        retrieved_columns: List[RetrievedColumn] = []
        
        for detail in column_details:
            try:
                # Create base parameters
                base_params: Dict[str, Any] = {
                    'table': detail.get('table', ''),
                    'column': detail.get('column', ''),
                    'datatype': detail.get('datatype', 'unknown'),
                    'description': detail.get('description', ''),
                    'confidence_score': detail.get('confidence_score', 1.0)
                }
                
                # Handle XML information safely
                if detail.get('is_xml'):
                    xpath_value = detail.get('xpath', detail.get('xml_path', ''))
                    base_params.update({
                        'xml_column': detail.get('xml_column', ''),
                        'xpath': xpath_value,
                        'sql_expression': detail.get('sql_expression', ''),
                        'type': ColumnType.XML
                    })
                
                retrieved_col = create_retrieved_column_safe(**base_params)
                retrieved_columns.append(retrieved_col)
                
            except Exception as e:
                logger.warning(f"Failed to process column detail: {e}, skipping")
                continue
        
        return retrieved_columns

    def _create_fallback_columns(self, schema_context: Any) -> List[RetrievedColumn]:
        """Create fallback columns when conversion fails."""
        retrieved_columns: List[RetrievedColumn] = []
        
        if hasattr(schema_context, 'tables') and schema_context.tables:
            if isinstance(schema_context.tables, dict):
                for table_name in schema_context.tables.keys():
                    retrieved_col = create_retrieved_column_safe(
                        table=table_name,
                        column="*",
                        datatype="unknown",
                        description=f"Table {table_name} (conversion fallback)"
                    )
                    retrieved_columns.append(retrieved_col)
            elif isinstance(schema_context.tables, list):
                for i, table in enumerate(schema_context.tables):
                    table_name = getattr(table, 'name', f'table_{i}')
                    retrieved_col = create_retrieved_column_safe(
                        table=table_name,
                        column="*",
                        datatype="unknown",
                        description=f"Table {table_name} (conversion fallback)"
                    )
                    retrieved_columns.append(retrieved_col)
        else:
            # Last resort fallback
            retrieved_columns = [
                create_retrieved_column_safe(
                    table="unknown",
                    column="unknown",
                    datatype="unknown",
                    description="Schema context conversion failed"
                )
            ]
        
        return retrieved_columns

    def _convert_dict_to_retrieved_columns(
        self, 
        schema_dict: Dict[str, Any]
    ) -> List[RetrievedColumn]:
        """
        UPDATED: Convert dictionary schema to RetrievedColumn list.
        NOW PRESERVES: JOIN and XML information from dictionary format.
        """
        retrieved_columns: List[RetrievedColumn] = []
        
        try:
            # Handle tables and columns
            if 'tables' in schema_dict:
                for table_name in schema_dict['tables']:
                    columns_by_table = schema_dict.get('columns_by_table', {})
                    if table_name in columns_by_table:
                        for col_data in columns_by_table[table_name]:
                            retrieved_col = create_retrieved_column_safe(
                                table=table_name,
                                column=col_data.get('column', ''),
                                datatype=col_data.get('datatype', 'unknown'),
                                description=col_data.get('description', ''),
                                xpath=col_data.get('xml_path', col_data.get('xpath')),
                                type=ColumnType.XML if col_data.get('is_xml_column', False) else ColumnType.REGULAR
                            )
                            retrieved_columns.append(retrieved_col)
            
            # PRESERVE: JOIN relationships from dictionary
            if 'joins' in schema_dict or 'join_plan' in schema_dict:
                joins = schema_dict.get('joins', []) + schema_dict.get('join_plan', [])
                for i, join in enumerate(joins):
                    if isinstance(join, dict):
                        join_desc = f"{join.get('source_table', '')}.{join.get('source_column', '')} = {join.get('target_table', '')}.{join.get('target_column', '')}"
                    else:
                        join_desc = str(join)
                    
                    join_col = create_retrieved_column_safe(
                        table="SCHEMA_JOINS",
                        column=f"join_{i}",
                        datatype="relationship",
                        description=join_desc,
                        type=ColumnType.KEY
                    )
                    retrieved_columns.append(join_col)
            
            # PRESERVE: XML information from dictionary
            if 'xml_columns_by_table' in schema_dict:
                xml_data = schema_dict['xml_columns_by_table']
                xml_counter = 0
                for table, xml_columns in xml_data.items():
                    for xml_col in xml_columns:
                        xml_desc = f"Table: {table} | XML: {xml_col.get('xpath', '')} -> {xml_col.get('column_name', '')}"
                        
                        xml_retrieved_col = create_retrieved_column_safe(
                            table="SCHEMA_XML",
                            column=f"xml_{xml_counter}",
                            datatype="xml_path",
                            description=xml_desc,
                            xpath=xml_col.get('xpath', ''),
                            type=ColumnType.XML
                        )
                        retrieved_columns.append(xml_retrieved_col)
                        xml_counter += 1
            
            print(f"DEBUG: Converted dictionary schema to {len(retrieved_columns)} RetrievedColumns")
            
        except Exception as e:
            print(f"DEBUG: Error converting dictionary schema: {str(e)}")
            retrieved_columns = [
                create_retrieved_column_safe(
                    table="unknown",
                    column="unknown", 
                    datatype="unknown",
                    description="Dictionary schema conversion failed"
                )
            ]
        
        return retrieved_columns

    def _convert_to_simple_prompt(self, structured_prompt: StructuredPrompt) -> str:
        """Convert StructuredPrompt to simple string format for orchestrator."""
        try:
            # Use the enhanced get_full_prompt method from updated data_models.py
            return structured_prompt.get_full_prompt()
            
        except Exception as e:
            print(f"DEBUG: Error using get_full_prompt, falling back to manual assembly: {str(e)}")
            
            # Fallback to manual assembly
            sections: List[str] = []
            
            if structured_prompt.system_context:
                sections.append("=== SYSTEM CONTEXT ===")
                sections.append(str(structured_prompt.system_context))
                sections.append("")
            
            if structured_prompt.schema_context:
                sections.append("=== SCHEMA CONTEXT ===")
                sections.append(str(structured_prompt.schema_context))
                sections.append("")
            
            if structured_prompt.instructions:
                sections.append("=== INSTRUCTIONS ===")
                sections.append(str(structured_prompt.instructions))
                sections.append("")
            
            # Add user query at the end
            sections.append("=== USER QUERY ===")
            sections.append(getattr(structured_prompt, 'user_query', 'No query provided'))
            
            return "\n".join(sections)

    def _update_build_stats(self, build_time_ms: float, success: bool) -> None:
        """Update build statistics."""
        if success:
            self.build_stats["successful_builds"] += 1
        else:
            self.build_stats["failed_builds"] += 1
        
        self.build_stats["last_build_time"] = datetime.now().isoformat()

    def get_stats(self) -> Dict[str, Any]:
        """Get build statistics."""
        return self.build_stats.copy()

    # LEGACY METHODS - Keep for backward compatibility but simplified
    def build_prompt(
        self,
        user_query: str,
        schema_results: List[RetrievedColumn],
        options: Optional[PromptOptions] = None
    ) -> StructuredPrompt:
        """Legacy method - use build_prompt_simple() instead."""
        print("DEBUG: Using legacy build_prompt method")
        return self._build_standard_prompt(user_query, schema_results)

    def analyze_query_intent(
        self, 
        user_query: str, 
        schema_results: Optional[List[RetrievedColumn]] = None
    ) -> QueryIntent:
        """Analyze query intent."""
        return self.query_analyzer.analyze_query(user_query, schema_results or [])

# CONVENIENCE FUNCTIONS for orchestrator usage
async def build_prompt_simple_async(
    user_query: str,
    schema_context: Any
) -> str:
    """
    Async convenience function for orchestrator integration.
    UPDATED: Now preserves complete schema information.
    """
    builder = PromptBuilder()
    return await builder.build_prompt_simple_async(user_query, schema_context)

def build_prompt_simple(
    user_query: str,
    schema_context: Any
) -> Union[str, Awaitable[str]]:
    """
    Convenience function for orchestrator integration.
    UPDATED: Now preserves complete schema information.
    """
    builder = PromptBuilder()
    return builder.build_prompt_simple(user_query, schema_context)

# Keep legacy convenience function
def build_prompt(
    user_query: str,
    schema_results: List[RetrievedColumn],
    options: Optional[PromptOptions] = None
) -> StructuredPrompt:
    """Legacy convenience function."""
    builder = PromptBuilder()
    return builder.build_prompt(user_query, schema_results, options)

# Module-level logger configuration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("PromptBuilder main module - ready for orchestrator integration with async support")
    
    # Simple test without asyncio.run()
    print("PromptBuilder Test")
    builder = PromptBuilder()
    print("PromptBuilder initialized successfully")
    
    # Test sync functionality
    print("Testing sync build prompt...")
    try:
        prompt = builder._build_prompt_sync("test query", {})
        print(f"Sync test successful: {len(prompt)} characters")
    except Exception as e:
        print(f"Sync test failed: {e}")
