"""
Main PromptBuilder Class - CENTRALIZED SYSTEM WITH XML INTEGRATION

ENHANCED VERSION - Added XML field support and example queries

Integrates with existing components while keeping it simple

DEFENSIVE PROGRAMMING: Handles list/dict schema format variations

FIXED: Join formatting to use correct keys (source_table, target_table)

ENHANCED: XML field formatting with usage examples and mixed query samples

Version: 1.1.0 - XML INTEGRATION WITH EXAMPLE QUERIES

Date: 2025-08-22
"""

import logging
from typing import Dict, Any, Optional, List, Union
import asyncio
import time

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    ENHANCED PromptBuilder Class - Single source of truth for prompt generation WITH XML SUPPORT
    
    ARCHITECTURE:
    - Simple, working implementation
    - Uses existing components when available
    - Graceful degradation without complex fallbacks
    - Clean integration with centralized bridge
    - XML field formatting and examples
    
    DEFENSIVE PROGRAMMING: Handles any schema format including XML-enhanced schemas
    FIXED: Proper join display using source_table/target_table keys
    ENHANCED: XML field examples and mixed query samples
    """

    def __init__(self, async_client_manager=None):
        """Initialize PromptBuilder with minimal dependencies"""
        
        self.async_client_manager = async_client_manager
        self.logger = logging.getLogger("PromptBuilder")
        
        # Initialize core components
        self._init_components()
        
        self.logger.info("PromptBuilder initialized successfully with XML support")

    def _init_components(self):
        """Initialize components with safe imports"""
        
        # Template Manager
        try:
            from .template_manager import TemplateManager
            self.template_manager = TemplateManager()
            self.template_manager_available = True
        except ImportError as e:
            self.logger.warning(f"TemplateManager not available: {e}")
            self.template_manager = None
            self.template_manager_available = False
        
        # Query Analyzer
        try:
            from .query_analyzer import DynamicQueryAnalyzer
            self.query_analyzer = DynamicQueryAnalyzer()
            self.query_analyzer_available = True
        except ImportError as e:
            self.logger.warning(f"QueryAnalyzer not available: {e}")
            self.query_analyzer = None
            self.query_analyzer_available = False
        
        # Prompt Assembler
        try:
            from ..assemblers.prompt_assembler import PromptAssembler
            self.assembler = PromptAssembler()
            self.assembler_available = True
        except ImportError as e:
            self.logger.warning(f"PromptAssembler not available: {e}")
            self.assembler = None
            self.assembler_available = False

    async def build_sophisticated_prompt(
        self,
        query: str,
        schema_context: Dict[str, Any],
        request_id: str = None  # pyright: ignore[reportArgumentType]
    ) -> Dict[str, Any]:
        """
        MAIN ENTRY POINT: Build sophisticated prompt using centralized system WITH XML SUPPORT
        
        Returns dict with:
        - prompt: Generated prompt string (now with XML examples)
        - quality: Quality assessment
        - metadata: Generation metadata
        """
        
        try:
            self.logger.info(f"Building sophisticated prompt with XML support for: {query[:50]}...")
            start_time = time.time()
            
            # STEP 1: Query Analysis
            query_intent = await self._analyze_query(query)
            
            # STEP 2: Generate Prompt (now XML-aware)
            if self.assembler_available:
                # Use full assembler
                structured_prompt = await self._use_full_assembler(query, query_intent, schema_context)
                prompt_text = structured_prompt.get_full_prompt()
                quality = "sophisticated"
            else:
                # Use simplified generation (now with XML support)
                prompt_text = self._generate_simple_prompt(query, schema_context)
                quality = "basic"
            
            # STEP 3: Create response
            processing_time = (time.time() - start_time) * 1000
            
            # Check for XML content in schema
            has_xml_fields = bool(
                schema_context.get('xml_enhanced_tables') or 
                schema_context.get('xml_field_mappings') or 
                schema_context.get('xml_schema_loaded', False)
            )
            
            result = {
                "prompt": prompt_text,
                "quality": quality,
                "success": True,
                "metadata": {
                    "request_id": request_id,
                    "processing_time_ms": processing_time,
                    "query_length": len(query),
                    "prompt_length": len(prompt_text),
                    "schema_tables": len(schema_context.get("tables", [])),
                    "xml_fields_included": has_xml_fields,
                    "centralized_builder_used": True,
                    "defensive_schema_formatting": True,
                    "join_display_fixed": True,
                    "xml_integration_enabled": True,
                    "components_used": {
                        "template_manager": self.template_manager_available,
                        "query_analyzer": self.query_analyzer_available,
                        "assembler": self.assembler_available
                    }
                }
            }
            
            self.logger.info(f"Sophisticated prompt built: {len(prompt_text)} chars, {quality} quality, XML: {has_xml_fields}")
            return result
            
        except Exception as e:
            self.logger.error(f"Sophisticated prompt building failed: {e}")
            # Emergency fallback
            return {
                "prompt": self._create_emergency_prompt(query, schema_context),
                "quality": "fallback",
                "success": False,
                "error": str(e),
                "metadata": {
                    "request_id": request_id,
                    "centralized_builder_used": True,
                    "fallback_used": True
                }
            }

    async def _analyze_query(self, query: str):
        """Analyze query intent"""
        
        if self.query_analyzer_available:
            try:
                return self.query_analyzer.analyze_query_with_intelligence(query, True)  # pyright: ignore[reportOptionalMemberAccess]
            except Exception as e:
                self.logger.warning(f"Query analysis failed: {e}")
        
        # Basic intent
        from .data_models import QueryIntent, PromptType, QueryComplexity
        
        return QueryIntent(
            query_type=PromptType.SIMPLE_SELECT,
            complexity=QueryComplexity.MEDIUM,
            confidence=0.7
        )

    async def _use_full_assembler(self, query: str, query_intent, schema_context: Dict[str, Any]):
        """Use full assembler if available"""
        
        try:
            # Convert schema_context to proper format
            from .data_models import PromptOptions
            
            options = PromptOptions(
                enable_intelligent_filtering=True,
                optimization_level="sophisticated"
            )
            
            return await self.assembler.assemble_intelligent_prompt(  # pyright: ignore[reportOptionalMemberAccess]
                query, query_intent, schema_context, options  # pyright: ignore[reportArgumentType]
            )  # type: ignore
            
        except Exception as e:
            self.logger.warning(f"Full assembler failed: {e}")
            # Return basic structured prompt
            from .data_models import StructuredPrompt
            
            return StructuredPrompt(
                system_context="You are an expert SQL generator with XML field expertise.",
                schema_context=self._format_schema_context_robustly(schema_context),
                user_query=query,
                instructions="Generate SQL based on the provided schema, using proper XML extraction for XML fields.",
                prompt_type=query_intent.query_type
            )

    def _generate_simple_prompt(self, query: str, schema_context: Dict[str, Any]) -> str:
        """Generate simple prompt when assembler not available - NOW WITH XML SUPPORT"""
        
        schema_text = self._format_schema_context_robustly(schema_context)
        
        # Check if XML fields are present
        has_xml_fields = bool(
            schema_context.get('xml_enhanced_tables') or 
            schema_context.get('xml_field_mappings') or 
            schema_context.get('xml_schema_loaded', False)
        )
        
        xml_instructions = ""
        if has_xml_fields:
            xml_instructions = """
⚠️  IMPORTANT XML FIELD HANDLING:
- XML fields require special .value() syntax as shown in the schema
- NEVER reference XML field names directly as columns
- Use the exact SQL expressions provided for XML fields
- XML data is stored in dedicated XML columns (like CTPT_XML, XML_Collateral)
"""

        return f"""You are an expert SQL generator with XML field expertise. Generate accurate, efficient SQL queries based on the user's natural language request.

SCHEMA CONTEXT:
{schema_text}

INSTRUCTIONS:
- Generate clean, readable SQL
- Use proper table and column names exactly as provided
- For XML fields, use the exact SQL expressions shown (with .value() method)
- Follow SQL best practices
- Optimize for performance when possible{xml_instructions}

USER REQUEST: {query}

Generate the SQL query:"""

    def _format_schema_context_robustly(self, schema_context: Any) -> str:
        """
        ENHANCED: Robust schema formatter that handles list, dict, and complex schema formats WITH XML SUPPORT
        
        Args:
            schema_context: Can be dict, list, or complex schema object
            
        Returns:
            str: Formatted schema string with XML field support
        """
        
        try:
            # Handle None or empty
            if not schema_context:
                return "No schema information available"
            
            # Format 1: Bridge/Complex format {"tables": [...], "columns_by_table": {...}} WITH XML
            if isinstance(schema_context, dict):
                if "tables" in schema_context and "columns_by_table" in schema_context:
                    return self._format_bridge_schema(schema_context)
                elif "columns_by_table" in schema_context:
                    return self._format_columns_by_table_schema(schema_context)
                else:
                    # Format 2: Direct mapping {"table_name": ["col1", "col2"]}
                    return self._format_mapping_schema(schema_context)
            
            # Format 3: Raw list [{"name": "table", "columns": [...]}]
            elif isinstance(schema_context, list):
                return self._format_list_schema(schema_context)
            
            # Fallback: Convert to string
            else:
                return f"Schema: {str(schema_context)[:200]}..."
                
        except Exception as e:
            self.logger.warning(f"Schema formatting failed with fallback: {e}")
            return f"Schema available (format error: {type(schema_context).__name__})"

    def _format_bridge_schema(self, schema_context: dict) -> str:
        """
        ENHANCED: Format bridge-style schema with XML field support and mixed query example
        
        Uses proper join keys: source_table, source_column, target_table, target_column
        Separates relational columns from XML fields with usage examples
        """
        
        parts = []
        
        # Tables
        tables = schema_context.get("tables", [])
        if tables:
            parts.append("=== TABLES ===")
            for table in tables[:10]:  # Limit to 10 tables
                parts.append(f"- {table}")
            if len(tables) > 10:
                parts.append(f"... and {len(tables) - 10} more tables")
        
        # ENHANCED: Columns by table WITH XML field separation
        columns_by_table = schema_context.get("columns_by_table", {})
        xml_enhanced_tables = schema_context.get("xml_enhanced_tables", {})
        xml_field_mappings = schema_context.get("xml_field_mappings", {})
        xml_schema_loaded = schema_context.get("xml_schema_loaded", False)
        
        if columns_by_table or xml_enhanced_tables:
            parts.append("\n=== COLUMNS & XML FIELDS ===")
            
            for table in tables[:8]:  # Limit to 8 tables
                # Regular database columns
                if table in columns_by_table:
                    columns = columns_by_table[table]
                    if isinstance(columns, list):
                        col_names = []
                        for col in columns[:12]:  # Limit columns
                            if isinstance(col, dict):
                                col_name = col.get('column', col.get('name', str(col)))
                            else:
                                col_name = str(col)
                            col_names.append(col_name)
                        
                        parts.append(f"\n{table} (Database Columns):")
                        parts.append(f"  {', '.join(col_names)}")
                        
                        if len(columns) > 12:
                            parts.append(f"  ... and {len(columns) - 12} more columns")
                    else:
                        parts.append(f"\n{table} (Database): {str(columns)[:50]}")
                
                # NEW: XML fields for this table
                if table in xml_enhanced_tables and xml_schema_loaded:
                    xml_data = xml_enhanced_tables[table]
                    xml_column = xml_data['xml_column']
                    xml_fields = xml_data['fields']
                    
                    parts.append(f"\n{table} (XML Fields in {xml_column}):")
                    
                    for field in xml_fields[:8]:  # Show first 8 XML fields
                        field_name = field['name']
                        sql_expr = field['sql_expression']
                        parts.append(f"  {field_name}: {sql_expr}")
                    
                    if len(xml_fields) > 8:
                        parts.append(f"  ... and {len(xml_fields) - 8} more XML fields")

        # XML Field Usage Examples (NEW)
        if xml_field_mappings and xml_schema_loaded:
            parts.append(f"\n=== XML FIELD USAGE EXAMPLES ===")
            parts.append("⚠️  XML fields require special SQL syntax:")
            
            # Show examples from different tables
            shown_tables = set()
            example_count = 0
            
            for field_key, mapping in xml_field_mappings.items():
                if example_count >= 4:  # Limit examples
                    break
                
                table = mapping['table']
                if table not in shown_tables or len(shown_tables) < 2:
                    shown_tables.add(table)
                    field_name = mapping['field_name']
                    sql_expression = mapping['sql_expression']
                    
                    parts.append(f"  {table}.{field_name}:")
                    parts.append(f"    {sql_expression}")
                    example_count += 1
            
            if len(xml_field_mappings) > 4:
                parts.append(f"  ... and {len(xml_field_mappings) - 4} more XML fields available")

        # FIXED: Joins section with correct key mapping
        joins = schema_context.get("joins", [])
        if joins:
            parts.append(f"\n=== JOINS ({len(joins)} available) ===")
            for join in joins[:6]:  # Show first 6 joins
                if isinstance(join, dict):
                    # FIXED: Use correct keys from your join data structure
                    source_table = join.get('source_table', 'unknown')
                    source_column = join.get('source_column', 'unknown')
                    target_table = join.get('target_table', 'unknown')
                    target_column = join.get('target_column', 'unknown')
                    
                    # Display full join information
                    parts.append(f"- {source_table}.{source_column} -> {target_table}.{target_column}")
                else:
                    parts.append(f"- {str(join)[:50]}")
            
            if len(joins) > 6:
                parts.append(f"... and {len(joins) - 6} more joins available")

        # NEW: Mixed SQL and XML Query Example
        if xml_enhanced_tables and xml_schema_loaded:
            parts.append(f"\n=== MIXED SQL & XML QUERY EXAMPLE ===")
            parts.append("📋 Example showing both regular columns AND XML field extraction:")
            
            # Find a table with both regular columns and XML fields
            example_table = None
            for table_name in tables:
                if table_name in columns_by_table and table_name in xml_enhanced_tables:
                    example_table = table_name
                    break
            
            if not example_table and xml_enhanced_tables:
                example_table = list(xml_enhanced_tables.keys())[0]
            
            if example_table:
                # Get regular columns
                regular_cols = columns_by_table.get(example_table, [])
                reg_col_names = []
                if isinstance(regular_cols, list) and regular_cols:
                    for col in regular_cols[:3]:  # First 3 regular columns
                        if isinstance(col, dict):
                            col_name = col.get('column', col.get('name', str(col)))
                        else:
                            col_name = str(col)
                        reg_col_names.append(col_name)
                
                # Get XML fields
                xml_data = xml_enhanced_tables.get(example_table, {})
                xml_fields = xml_data.get('fields', [])
                
                parts.append(f"")
                parts.append(f"-- Example: Select regular columns AND XML fields from {example_table}")
                parts.append(f"SELECT")
                
                # Add regular columns
                if reg_col_names:
                    for col in reg_col_names:
                        parts.append(f"    {col},  -- Regular column")
                
                # Add XML fields
                xml_examples_shown = 0
                for field in xml_fields[:2]:  # Show 2 XML fields
                    field_name = field['name']
                    sql_expr = field['sql_expression']
                    comma = "," if xml_examples_shown < 1 and len(xml_fields) > 1 else ""
                    parts.append(f"    {sql_expr} AS {field_name}{comma}  -- XML field")
                    xml_examples_shown += 1
                
                parts.append(f"FROM {example_table};")
                parts.append(f"")
                parts.append(f"✅ This example shows:")
                parts.append(f"   - Regular columns: Direct table.column reference")
                parts.append(f"   - XML fields: Use .value('xpath', 'datatype') syntax")
                parts.append(f"   - Always alias XML extractions with AS fieldname")
        
        # XML Usage Instructions (NEW)
        if xml_field_mappings and xml_schema_loaded:
            parts.append(f"\n=== XML FIELD INSTRUCTIONS ===")
            parts.append("🎯 CRITICAL XML Rules:")
            parts.append("1. Use EXACT SQL expressions shown above for XML fields")
            parts.append("2. NEVER reference XML field names as direct columns")
            parts.append("3. XML data is in dedicated columns (like CTPT_XML, XML_Collateral)")
            parts.append("4. Always use .value('(xpath)[1]', 'datatype') for extraction")
            parts.append("5. Regular columns use normal table.column syntax")
            parts.append("6. Mix regular and XML columns freely in same query")
        
        return "\n".join(parts) if parts else "Bridge schema structure available"

    def _format_columns_by_table_schema(self, schema_context: dict) -> str:
        """Format columns_by_table style schema with XML awareness"""
        
        columns_by_table = schema_context.get("columns_by_table", {})
        xml_enhanced_tables = schema_context.get("xml_enhanced_tables", {})
        
        if not columns_by_table and not xml_enhanced_tables:
            return "No columns information available"
        
        parts = ["=== TABLES & COLUMNS ==="]
        
        # Process regular tables
        for table, columns in list(columns_by_table.items())[:10]:  # Limit tables
            if isinstance(columns, list):
                col_names = []
                for col in columns[:8]:  # Limit columns
                    if isinstance(col, dict):
                        col_name = col.get('column', col.get('name', str(col)))
                    else:
                        col_name = str(col)
                    col_names.append(col_name)
                
                parts.append(f"{table} (Database): {', '.join(col_names)}")
                
                if len(columns) > 8:
                    parts.append(f"  ... and {len(columns) - 8} more columns")
            else:
                parts.append(f"{table}: {str(columns)[:50]}")
            
            # Add XML fields for this table if available
            if table in xml_enhanced_tables:
                xml_data = xml_enhanced_tables[table]
                xml_fields = xml_data.get('fields', [])
                if xml_fields:
                    xml_names = [field['name'] for field in xml_fields[:5]]
                    parts.append(f"{table} (XML Fields): {', '.join(xml_names)}")
                    if len(xml_fields) > 5:
                        parts.append(f"  ... and {len(xml_fields) - 5} more XML fields")
        
        return "\n".join(parts)

    def _format_mapping_schema(self, schema_context: dict) -> str:
        """Format direct mapping: {"table": ["col1", "col2"]} or other dict formats"""
        
        parts = ["=== SCHEMA MAPPING ==="]
        
        for key, value in list(schema_context.items())[:10]:  # Limit entries
            if isinstance(value, list):
                if value and isinstance(value[0], dict):
                    # List of column objects
                    col_names = []
                    for item in value[:8]:
                        if isinstance(item, dict):
                            col_name = item.get('column', item.get('name', str(item)))
                        else:
                            col_name = str(item)
                        col_names.append(col_name)
                    
                    parts.append(f"{key}: {', '.join(col_names)}")
                    
                    if len(value) > 8:
                        parts.append(f"  ... and {len(value) - 8} more items")
                else:
                    # Simple list
                    parts.append(f"{key}: {', '.join(str(v) for v in value[:8])}")
                    if len(value) > 8:
                        parts.append(f"  ... and {len(value) - 8} more items")
            else:
                parts.append(f"{key}: {str(value)[:50]}")
        
        return "\n".join(parts)

    def _format_list_schema(self, schema_context: list) -> str:
        """Format list-style schema: [{"name": "table", "columns": [...]}]"""
        
        if not schema_context:
            return "Empty schema list"
        
        parts = ["=== LIST-BASED SCHEMA ==="]
        
        for i, item in enumerate(schema_context[:10]):  # Limit to 10 items
            if isinstance(item, dict):
                # Extract table name
                table_name = (
                    item.get("name") or
                    item.get("table") or
                    item.get("table_name") or
                    f"item_{i}"
                )
                
                # Extract columns
                columns = (
                    item.get("columns") or
                    item.get("fields") or
                    item.get("column_list") or
                    []
                )
                
                if isinstance(columns, list) and columns:
                    col_names = []
                    for col in columns[:8]:  # Limit columns
                        if isinstance(col, dict):
                            col_name = col.get('column', col.get('name', str(col)))
                        else:
                            col_name = str(col)
                        col_names.append(col_name)
                    
                    parts.append(f"{table_name}: {', '.join(col_names)}")
                    
                    if len(columns) > 8:
                        parts.append(f"  ... and {len(columns) - 8} more columns")
                else:
                    parts.append(f"{table_name}: {str(columns)[:50]}")
            else:
                parts.append(f"Item {i}: {str(item)[:50]}")
        
        if len(schema_context) > 10:
            parts.append(f"... and {len(schema_context) - 10} more items")
        
        return "\n".join(parts)

    def _format_schema_context(self, schema_context: Dict[str, Any]) -> str:
        """Legacy method - now calls robust formatter"""
        
        return self._format_schema_context_robustly(schema_context)

    def _create_emergency_prompt(self, query: str, schema_context: Dict[str, Any]) -> str:
        """Emergency fallback prompt with XML awareness"""
        
        has_xml = bool(schema_context.get('xml_schema_loaded', False))
        xml_note = "\nNote: XML fields detected - use .value() syntax for extraction." if has_xml else ""
        
        return f"""Generate SQL for: {query}

Available schema context: {len(schema_context)} items{xml_note}

Please generate appropriate SQL based on the user's request."""

    async def health_check(self) -> Dict[str, Any]:
        """Health check for PromptBuilder with XML integration status"""
        
        return {
            "status": "healthy",
            "components": {
                "template_manager": self.template_manager_available,
                "query_analyzer": self.query_analyzer_available,
                "assembler": self.assembler_available
            },
            "centralized_prompt_builder": True,
            "async_client_manager": self.async_client_manager is not None,
            "defensive_schema_formatting": True,
            "join_display_fixed": True,  # NEW: Indicates join fix is applied
            "xml_integration_enabled": True,  # NEW: XML support enabled
            "mixed_query_examples": True,  # NEW: Provides mixed SQL/XML examples
            "schema_format_support": [
                "bridge_format",
                "columns_by_table_format",
                "mapping_format",
                "list_format",
                "mixed_format",
                "xml_enhanced_format"  # NEW
            ],
            "xml_features": {  # NEW: XML-specific features
                "xml_field_separation": True,
                "xml_usage_examples": True,
                "mixed_query_samples": True,
                "xml_syntax_instructions": True
            }
        }

    def is_available(self) -> bool:
        """Check if PromptBuilder is available"""
        
        return True  # Always available as it has fallbacks


# Factory function for easy import
def create_prompt_builder(async_client_manager=None) -> PromptBuilder:
    """Factory function to create PromptBuilder"""
    
    return PromptBuilder(async_client_manager)


# Export
__all__ = ['PromptBuilder', 'create_prompt_builder']
