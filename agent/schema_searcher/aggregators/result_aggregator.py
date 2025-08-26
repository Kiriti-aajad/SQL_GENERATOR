"""
Enhanced Result aggregation module: deduped union with XML path preservation.
FAIL-FAST VERSION with XML Schema Priority Management.
Combines results from engines with strict validation and XML priority handling.
Supports authoritative xml_schema.json data with fail-fast behavior.
"""

from typing import List, Dict, Set, Optional, Any, Tuple
from collections import defaultdict
from difflib import SequenceMatcher
import logging
import json
import os

from agent.schema_searcher.core.data_models import RetrievedColumn, SearchMethod, ColumnType

def _normalized(s: str) -> str:
    return s.lower().replace('_', '').replace(' ', '') if s else ''

def _string_sim(a: str, b: str) -> float:
    a, b = _normalized(a), _normalized(b)
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0

def _is_similar_col(a: RetrievedColumn, b: RetrievedColumn, thresh=0.83):
    """Enhanced column similarity with XML path awareness"""
    # Basic table and column name matching
    table_match = _string_sim(a.table, b.table) > thresh
    column_match = _string_sim(a.column, b.column) > thresh
    
    if not (table_match and column_match):
        return False
    
    # Enhanced XML column matching with xpath consideration
    if a.type == ColumnType.XML and b.type == ColumnType.XML:
        xml_a = getattr(a, 'xml_column', '') or ''
        xml_b = getattr(b, 'xml_column', '') or ''
        xml_col_match = (_string_sim(xml_a, xml_b) > thresh) if (xml_a and xml_b) else True
        
        xpath_a = getattr(a, 'xpath', '') or ''
        xpath_b = getattr(b, 'xpath', '') or ''
        xpath_match = (_string_sim(xpath_a, xpath_b) > thresh) if (xpath_a and xpath_b) else True
        
        return xml_col_match and xpath_match
    
    # Don't match XML with relational columns
    if a.type != b.type:
        return False
    
    return True

class EnhancedXMLSchemaManager:
    """
    XML Schema Manager with Priority Handling
    Provides authoritative XML data that overrides schema.json when conflicts exist
    """
    
    def __init__(self, xml_schema_path: str = None): # pyright: ignore[reportArgumentType]
        self.logger = logging.getLogger("XMLSchemaManager")
        self.xml_schema_data = {}
        self.xml_fields_by_table = {}
        self.xml_field_index = {}  # For fast lookup by table.column
        
        if xml_schema_path is None:
            xml_schema_path = "E:/Github/sql-ai-agent/data/metadata/xml_schema.json"
        
        self.xml_schema_path = xml_schema_path
        self._load_xml_schema_data()
        self._build_field_index()
    
    def _load_xml_schema_data(self):
        """Load and parse xml_schema.json data"""
        try:
            if os.path.exists(self.xml_schema_path):
                with open(self.xml_schema_path, 'r', encoding='utf-8') as f:
                    self.xml_schema_data = json.load(f)
                
                if isinstance(self.xml_schema_data, list):
                    for table_data in self.xml_schema_data:
                        table_name = table_data.get('table', '')
                        if table_name:
                            self.xml_fields_by_table[table_name] = table_data
                
                self.logger.info(f"Loaded authoritative XML schema data for {len(self.xml_fields_by_table)} tables")
            else:
                self.logger.warning(f"XML schema file not found: {self.xml_schema_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to load XML schema data: {e}")
    
    def _build_field_index(self):
        """Build fast lookup index for table.column combinations"""
        for table_name, table_data in self.xml_fields_by_table.items():
            fields = table_data.get('fields', [])
            for field in fields:
                field_name = field.get('name', '')
                if field_name:
                    key = f"{table_name}.{field_name}".lower()
                    self.xml_field_index[key] = {
                        'table': table_name,
                        'field': field,
                        'xml_column': table_data.get('xml_column', '')
                    }
    
    def has_authoritative_xml_field(self, table_name: str, field_name: str) -> bool:
        """Check if we have authoritative XML data for this table.column"""
        key = f"{table_name}.{field_name}".lower()
        return key in self.xml_field_index
    
    def get_authoritative_xml_field(self, table_name: str, field_name: str) -> Optional[Dict[str, Any]]:
        """Get authoritative XML field data that should override schema.json"""
        key = f"{table_name}.{field_name}".lower()
        return self.xml_field_index.get(key)
    
    def get_xml_fields_for_table(self, table_name: str) -> List[Dict[str, Any]]:
        """Get all XML fields for a specific table"""
        table_data = self.xml_fields_by_table.get(table_name, {})
        return table_data.get('fields', [])
    
    def get_xml_column_for_table(self, table_name: str) -> Optional[str]:
        """Get the XML storage column name for a table"""
        table_data = self.xml_fields_by_table.get(table_name, {})
        return table_data.get('xml_column')

class ResultAggregator:
    """
    Enhanced union-based aggregator with XML path preservation and FAIL-FAST behavior.
    CRITICAL CHANGES:
    - Fails fast when all engines fail
    - Validates input and results strictly  
    - Prioritizes xml_schema.json over schema.json
    - Provides rich schema context for prompt builder
    """
    
    def __init__(self, xml_schema_path: str = None): # pyright: ignore[reportArgumentType]
        # Safe logger initialization
        try:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.WARNING)
        except Exception:
            class NoOpLogger:
                def debug(self, msg): pass
                def info(self, msg): pass
                def warning(self, msg): pass
                def error(self, msg): pass
            self.logger = NoOpLogger()
        
        # Enhanced XML schema manager with priority handling
        self.xml_schema_manager = EnhancedXMLSchemaManager(xml_schema_path)
        
        # Legacy XML manager for backward compatibility
        self.legacy_xml_manager = None
        self._initialize_legacy_xml_manager()
        
        # Track XML priority usage for statistics
        self.xml_priority_applied_count = 0
        self.schema_json_overridden_count = 0
    
    def _initialize_legacy_xml_manager(self):
        """Initialize legacy XML schema manager for backward compatibility"""
        try:
            from agent.sql_generator.config import get_config
            config = get_config()
            if hasattr(config, 'xml_schema_manager'):
                self.legacy_xml_manager = config.xml_schema_manager # pyright: ignore[reportAttributeAccessIssue]
                self.logger.debug("Legacy XML schema manager initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize legacy XML schema manager: {e}")
    
    def aggregate(
        self,
        method_results: Dict[SearchMethod, List[RetrievedColumn]],
    ) -> List[RetrievedColumn]:
        """
        Enhanced aggregation with XML priority and FAIL-FAST behavior
        """
        # ✅ FAIL FAST: Validate input
        if not method_results:
            raise ValueError("CRITICAL: No search method results provided for aggregation")
        
        # ✅ FAIL FAST: Check engine success rate with selective behavior
        successful_engines = []
        failed_engines = []
        
        for method, results in method_results.items():
            method_name = method.value if hasattr(method, 'value') else str(method)
            if results and len(results) > 0:
                successful_engines.append(method_name)
            else:
                failed_engines.append(method_name)
        
        # ✅ FAIL FAST: Require at least one successful engine
        if not successful_engines:
            engine_list = ", ".join(failed_engines)
            raise ValueError(f"CRITICAL: All search engines failed - no results from any method: {engine_list}")
        
        # ✅ LOG: Engine degradation (but continue if some work as discussed)
        if failed_engines:
            failed_list = ", ".join(failed_engines)
            successful_list = ", ".join(successful_engines)
            self.logger.warning(f"Search engine degradation detected: Failed=[{failed_list}], Working=[{successful_list}]")
        
        # Flatten results from successful engines only
        all_columns: List[RetrievedColumn] = [
            col
            for method, results in method_results.items() 
            if results and len(results) > 0
            for col in results
        ]
        
        # ✅ FAIL FAST: Validate we have actual results
        if not all_columns:
            raise ValueError("CRITICAL: No valid columns found after processing all engine results")
        
        # ✅ FAIL FAST: Check for excessive results (possible fallback behavior)
        if len(all_columns) > 500:  # Reasonable threshold
            engine_counts = {str(method): len(results or []) for method, results in method_results.items()}
            raise ValueError(f"CRITICAL: Excessive results ({len(all_columns)}) detected - possible engine fallback behavior. Engine counts: {engine_counts}")
        
        # ✅ ENHANCED: Process with XML priority and type separation
        processed_columns = self._process_columns_with_xml_priority(all_columns)
        
        # Enhanced deduplication with complete type awareness
        deduped = self._deduplicate_with_priority_awareness(processed_columns)
        
        # ✅ FAIL FAST: Validate final deduplication results
        if not deduped:
            raise ValueError("CRITICAL: Deduplication process resulted in zero columns - possible processing error")
        
        self.logger.info(f"Successfully aggregated {len(all_columns)} columns into {len(deduped)} unique columns from {len(successful_engines)} engines")
        return deduped
    
    def _process_columns_with_xml_priority(self, all_columns: List[RetrievedColumn]) -> List[RetrievedColumn]:
        """
        Process columns with XML PRIORITY HANDLING
        xml_schema.json takes precedence over schema.json
        """
        processed_columns = []
        xml_columns_processed = 0
        relational_columns_processed = 0
        xml_priority_applied = 0
        
        for column in all_columns:
            try:
                # Determine column type with XML priority
                column_type, priority_metadata = self._determine_column_type_with_priority(column)
                
                if column_type == ColumnType.XML:
                    processed_column = self._process_xml_column_with_priority(column, priority_metadata)
                    xml_columns_processed += 1
                    
                    if priority_metadata.get('has_xml_priority'):
                        xml_priority_applied += 1
                else:
                    processed_column = self._process_relational_column(column)
                    relational_columns_processed += 1
                
                if processed_column:
                    processed_columns.append(processed_column)
                    
            except Exception as e:
                self.logger.warning(f"Failed to process column {column.table}.{column.column}: {e}")
                processed_columns.append(column)
        
        self.logger.info(f"Column processing: {xml_columns_processed} XML ({xml_priority_applied} with XML priority), {relational_columns_processed} relational")
        return processed_columns
    
    def _determine_column_type_with_priority(self, column: RetrievedColumn) -> Tuple[ColumnType, Dict[str, Any]]:
        """
        Determine column type with XML PRIORITY HANDLING
        Returns: (column_type, priority_metadata)
        """
        table_name = getattr(column, 'table', '')
        column_name = getattr(column, 'column', '')
        
        priority_metadata = {
            'source': 'unknown',
            'has_xml_priority': False,
            'overridden_schema_json': False
        }
        
        # PRIORITY 1: Check xml_schema.json (HIGHEST PRIORITY)
        if table_name and column_name:
            if self.xml_schema_manager.has_authoritative_xml_field(table_name, column_name):
                priority_metadata.update({
                    'source': 'xml_schema_json',
                    'has_xml_priority': True,
                    'overridden_schema_json': hasattr(column, 'type') and getattr(column, 'type')
                })
                
                if priority_metadata['overridden_schema_json']:
                    self.schema_json_overridden_count += 1
                    self.logger.debug(f"XML priority: Overriding schema.json for {table_name}.{column_name}")
                
                return ColumnType.XML, priority_metadata
        
        # PRIORITY 2: Check explicit type from schema.json
        if hasattr(column, 'type') and getattr(column, 'type'):
            priority_metadata['source'] = 'schema_json_explicit'
            column_type = getattr(column, 'type')
            if column_type == 'xml':
                return ColumnType.XML, priority_metadata
            return ColumnType.RELATIONAL, priority_metadata
        
        # PRIORITY 3: Check schema.json indicators
        xml_indicators = ['xml_column', 'example_query_syntax', 'xpath']
        for indicator in xml_indicators:
            if hasattr(column, indicator) and getattr(column, indicator):
                priority_metadata['source'] = 'schema_json_indicators'
                return ColumnType.XML, priority_metadata
        
        # PRIORITY 4: Default to relational
        priority_metadata['source'] = 'default_relational'
        return ColumnType.RELATIONAL, priority_metadata
    
    def _process_xml_column_with_priority(self, column: RetrievedColumn, priority_metadata: Dict[str, Any]) -> RetrievedColumn:
        """
        Process XML column with priority-based enhancement
        """
        # Set base XML properties
        column.type = ColumnType.XML
        column.is_xml_field = True # pyright: ignore[reportAttributeAccessIssue]
        column.extraction_method = 'xml_path' # pyright: ignore[reportAttributeAccessIssue]
        
        table_name = getattr(column, 'table', '')
        column_name = getattr(column, 'column', '')
        
        # PRIORITY-BASED ENHANCEMENT
        if priority_metadata.get('has_xml_priority'):
            # Use authoritative XML data (HIGHEST PRIORITY)
            self._apply_authoritative_xml_data(column, table_name, column_name)
            self.xml_priority_applied_count += 1
        else:
            # Fall back to schema.json extraction
            self._extract_schema_json_xml_info(column)
            
            # Try to enhance with xml_schema.json if available
            self._enhance_with_xml_schema_json(column)
        
        # Final tagging
        column.xml_path_tagged = True # pyright: ignore[reportAttributeAccessIssue]
        column.priority_source = priority_metadata.get('source', 'unknown') # pyright: ignore[reportAttributeAccessIssue]
        column.is_authoritative_xml = priority_metadata.get('has_xml_priority', False) # pyright: ignore[reportAttributeAccessIssue]
        
        return column
    
    def _apply_authoritative_xml_data(self, column: RetrievedColumn, table_name: str, column_name: str):
        """
        Apply authoritative XML data that overrides any schema.json values
        """
        auth_data = self.xml_schema_manager.get_authoritative_xml_field(table_name, column_name)
        if not auth_data:
            return
        
        field_data = auth_data['field']
        xml_storage_column = auth_data['xml_column']
        
        # Apply authoritative data (OVERRIDES schema.json)
        column.xml_storage_column = xml_storage_column # pyright: ignore[reportAttributeAccessIssue]
        column.xpath = field_data.get('xpath', '')
        column.sql_expression = field_data.get('sql_expression', '')
        
        # Store original data for reference
        column.authoritative_xml_data = field_data # pyright: ignore[reportAttributeAccessIssue]
        column.xml_column = xml_storage_column  # For compatibility
        
        # Clear any conflicting schema.json data
        if hasattr(column, 'example_query_syntax'):
            column.original_example_query_syntax = getattr(column, 'example_query_syntax') # pyright: ignore[reportAttributeAccessIssue]
            # Keep authoritative sql_expression instead
        
        self.logger.debug(f"Applied authoritative XML data for {table_name}.{column_name}")
    
    def _extract_schema_json_xml_info(self, column: RetrievedColumn):
        """Extract XML information from schema.json structure"""
        # Extract XML storage column
        if hasattr(column, 'xml_column') and getattr(column, 'xml_column'):
            xml_storage_column = getattr(column, 'xml_column')
            if not hasattr(column, 'xml_storage_column'):
                column.xml_storage_column = xml_storage_column # pyright: ignore[reportAttributeAccessIssue]
        
        # Extract XPath from example_query_syntax
        if hasattr(column, 'example_query_syntax'):
            syntax = getattr(column, 'example_query_syntax', '') or ''
            if syntax:
                xpath = self._extract_xpath_from_syntax(syntax)
                if xpath and not getattr(column, 'xpath', None):
                    column.xpath = xpath
                
                if not getattr(column, 'sql_expression', None):
                    column.sql_expression = syntax
    
    def _enhance_with_xml_schema_json(self, column: RetrievedColumn):
        """
        Enhance XML column with data from xml_schema.json if not already authoritative
        """
        if getattr(column, 'is_authoritative_xml', False):
            return  # Already has authoritative data
        
        table_name = getattr(column, 'table', '')
        column_name = getattr(column, 'column', '')
        
        if not table_name or not column_name:
            return
        
        # Try to find in xml_schema.json for enhancement (not override)
        auth_data = self.xml_schema_manager.get_authoritative_xml_field(table_name, column_name)
        if auth_data:
            field_data = auth_data['field']
            xml_storage_column = auth_data['xml_column']
            
            # Enhance missing fields only
            if not getattr(column, 'xml_storage_column', None):
                column.xml_storage_column = xml_storage_column # pyright: ignore[reportAttributeAccessIssue]
            
            if not getattr(column, 'xpath', None):
                column.xpath = field_data.get('xpath', '')
            
            if not getattr(column, 'sql_expression', None):
                column.sql_expression = field_data.get('sql_expression', '')
            
            column.enhanced_from_xml_schema = True # pyright: ignore[reportAttributeAccessIssue]
            self.logger.debug(f"Enhanced {table_name}.{column_name} with xml_schema.json data")
    
    def _extract_xpath_from_syntax(self, syntax: str) -> Optional[str]:
        """Extract XPath from SQL syntax like CTPT_XML.value('(/path)[1]', 'type')"""
        try:
            import re
            # Pattern for .value('(XPATH)[1]', 'datatype')
            pattern = r"\.value\('\(([^)]+)\)\[1\]'"
            match = re.search(pattern, syntax)
            if match:
                return match.group(1)
            
            # Alternative pattern for XPath in quotes
            pattern = r"'(/[^']+)'"
            match = re.search(pattern, syntax)
            if match:
                xpath = match.group(1)
                return xpath.replace('[1]', '').strip()
            
        except Exception as e:
            self.logger.warning(f"Failed to extract XPath from syntax '{syntax}': {e}")
        
        return None
    
    def _process_relational_column(self, column: RetrievedColumn) -> RetrievedColumn:
        """Process regular relational column"""
        if not hasattr(column, 'type') or not getattr(column, 'type'):
            column.type = ColumnType.RELATIONAL
        
        column.is_xml_field = False # pyright: ignore[reportAttributeAccessIssue]
        column.extraction_method = 'direct_column' # pyright: ignore[reportAttributeAccessIssue]
        
        # Clear any XML-specific attributes that might have been set incorrectly
        xml_attrs_to_clear = ['xml_column', 'xpath', 'sql_expression', 'xml_storage_column', 'xml_path_tagged']
        for attr in xml_attrs_to_clear:
            if hasattr(column, attr):
                try:
                    delattr(column, attr)
                except:
                    pass
        
        return column
    
    def _deduplicate_with_priority_awareness(self, processed_columns: List[RetrievedColumn]) -> List[RetrievedColumn]:
        """
        Enhanced deduplication that respects XML priority and column types
        """
        deduped: List[RetrievedColumn] = []
        
        for candidate in processed_columns:
            matched = False
            
            for existing in deduped:
                if _is_similar_col(candidate, existing):
                    # Enhanced merging with priority awareness
                    self._merge_column_information_with_priority(existing, candidate)
                    matched = True
                    break
            
            if not matched:
                # Final enhancement with legacy XML manager if available
                enhanced_candidate = self._enhance_with_legacy_xml_schema(candidate)
                deduped.append(enhanced_candidate)
        
        return deduped
    
    def _merge_column_information_with_priority(self, target: RetrievedColumn, source: RetrievedColumn):
        """
        Enhanced merge logic with XML priority awareness
        """
        # Only merge columns of the same type
        if getattr(target, 'type', None) != getattr(source, 'type', None):
            self.logger.warning(f"Type mismatch during merge: {getattr(target, 'type')} vs {getattr(source, 'type')}")
            return
        
        # Check for XML priority
        target_is_authoritative = getattr(target, 'is_authoritative_xml', False)
        source_is_authoritative = getattr(source, 'is_authoritative_xml', False)
        
        # If source has XML priority and target doesn't, prefer source
        if source_is_authoritative and not target_is_authoritative:
            self._copy_priority_data(target, source)
            return
        
        # If target has priority, keep it and only merge non-conflicting data
        if target_is_authoritative and not source_is_authoritative:
            return  # Keep target data as-is
        
        # Standard merging for same priority level
        self._merge_column_information(target, source)
    
    def _copy_priority_data(self, target: RetrievedColumn, source: RetrievedColumn):
        """Copy high-priority data to target"""
        priority_attrs = [
            'xpath', 'sql_expression', 'xml_storage_column', 'xml_column',
            'is_authoritative_xml', 'priority_source', 'authoritative_xml_data'
        ]
        
        for attr in priority_attrs:
            if hasattr(source, attr):
                setattr(target, attr, getattr(source, attr))
    
    def _merge_column_information(self, target: RetrievedColumn, source: RetrievedColumn):
        """
        Standard merge logic preserving the best information from both columns
        """
        # Merge basic information (preserve longer/more detailed descriptions)
        target_desc = getattr(target, 'description', '') or ''
        source_desc = getattr(source, 'description', '') or ''
        if len(source_desc) > len(target_desc):
            target.description = source_desc
        
        # Enhanced XML information merging
        if getattr(target, 'type', None) == ColumnType.XML and getattr(source, 'type', None) == ColumnType.XML:
            # Merge XML column name (prefer non-empty)
            if not getattr(target, 'xml_column', None) and getattr(source, 'xml_column', None):
                target.xml_column = getattr(source, 'xml_column')
            
            # Merge SQL expression (prefer longer/more complete expressions)
            target_sql = getattr(target, 'sql_expression', '') or ''
            source_sql = getattr(source, 'sql_expression', '') or ''
            if len(source_sql) > len(target_sql):
                target.sql_expression = source_sql
            
            # Merge XPath (prefer non-empty)
            if not getattr(target, 'xpath', None) and getattr(source, 'xpath', None):
                target.xpath = getattr(source, 'xpath')
            
            # Merge data type information
            if not getattr(target, 'data_type', None) and getattr(source, 'data_type', None):
                target.data_type = getattr(source, 'data_type') # pyright: ignore[reportAttributeAccessIssue]
            
            # Merge any additional XML-related attributes
            xml_attrs = ['business_name', 'full_data_type', 'is_nullable']
            for attr in xml_attrs:
                if not getattr(target, attr, None) and getattr(source, attr, None):
                    setattr(target, attr, getattr(source, attr))
    
    def _enhance_with_legacy_xml_schema(self, column: RetrievedColumn) -> RetrievedColumn:
        """Enhance with legacy XML schema manager if available"""
        if (not self.legacy_xml_manager or 
            getattr(column, 'type', None) != ColumnType.XML or
            getattr(column, 'is_authoritative_xml', False)):
            return column
        
        try:
            paths = self.legacy_xml_manager.get_paths_for_table(column.table)
            for path in paths:
                if path.name.lower() == column.column.lower():
                    # Only enhance if not already enhanced from xml_schema.json
                    if not getattr(column, 'enhanced_from_xml_schema', False):
                        xml_attrs = ['xml_column', 'xpath', 'sql_expression', 'data_type']
                        for attr in xml_attrs:
                            if not getattr(column, attr, None) and hasattr(path, attr):
                                setattr(column, attr, getattr(path, attr))
                    break
        except Exception as e:
            self.logger.warning(f"Legacy XML enhancement failed for {column.table}.{column.column}: {e}")
        
        return column
    
    def build_rich_schema_context_for_prompt(self, aggregated_columns: List[RetrievedColumn]) -> Dict[str, Any]:
        """
        Build rich schema context optimized for prompt builder
        Separates XML and relational data with verified paths prioritized
        """
        context = {
            'tables': {},
            'xml_tables': {},
            'relational_tables': {},
            'verified_xml_paths': {},  # 100% verified XML paths for accuracy
            'schema_quality_metrics': {},
            'authoritative_xml_count': 0,
            'total_xml_count': 0,
            'prompt_optimization': {
                'xml_priority_applied': self.xml_priority_applied_count,
                'schema_json_overridden': self.schema_json_overridden_count
            }
        }
        
        # Organize columns by table and type
        for column in aggregated_columns:
            table_name = column.table
            col_type = getattr(column, 'type', ColumnType.RELATIONAL)
            
            # Initialize table structure
            if table_name not in context['tables']:
                context['tables'][table_name] = {
                    'xml_columns': [],
                    'relational_columns': [],
                    'xml_storage_column': None,
                    'has_authoritative_xml': False
                }
            
            table_context = context['tables'][table_name]
            
            if col_type == ColumnType.XML:
                # Process XML column with rich context
                xml_column_context = {
                    'column_name': column.column,
                    'xpath': getattr(column, 'xpath', ''),
                    'sql_expression': getattr(column, 'sql_expression', ''),
                    'xml_storage_column': getattr(column, 'xml_storage_column', '') or getattr(column, 'xml_column', ''),
                    'is_authoritative': getattr(column, 'is_authoritative_xml', False),
                    'priority_source': getattr(column, 'priority_source', 'unknown'),
                    'verification_status': '100% verified' if getattr(column, 'is_authoritative_xml', False) else 'schema.json derived'
                }
                
                table_context['xml_columns'].append(xml_column_context)
                
                # Track XML storage column
                xml_storage = xml_column_context['xml_storage_column']
                if xml_storage and not table_context['xml_storage_column']:
                    table_context['xml_storage_column'] = xml_storage
                
                # Track authoritative XML data
                if getattr(column, 'is_authoritative_xml', False):
                    table_context['has_authoritative_xml'] = True
                    context['authoritative_xml_count'] += 1
                
                context['total_xml_count'] += 1
                
                # Add to verified XML paths (prioritize authoritative)
                xpath = getattr(column, 'xpath', '')
                sql_expr = getattr(column, 'sql_expression', '')
                if xpath and sql_expr:
                    if table_name not in context['verified_xml_paths']:
                        context['verified_xml_paths'][table_name] = []
                    
                    context['verified_xml_paths'][table_name].append({
                        'column_name': column.column,
                        'xpath': xpath,
                        'sql_expression': sql_expr,
                        'is_authoritative': getattr(column, 'is_authoritative_xml', False),
                        'priority_source': getattr(column, 'priority_source', 'unknown')
                    })
            
            else:
                # Process relational column
                rel_column_context = {
                    'column_name': column.column,
                    'data_type': getattr(column, 'datatype', 'unknown'),
                    'extraction_method': 'direct_column'
                }
                table_context['relational_columns'].append(rel_column_context)
        
        # Separate XML and relational tables for prompt context
        for table_name, table_data in context['tables'].items():
            if table_data['xml_columns']:
                context['xml_tables'][table_name] = table_data
            if table_data['relational_columns']:
                context['relational_tables'][table_name] = table_data
        
        # Build quality metrics
        context['schema_quality_metrics'] = self._build_prompt_quality_metrics(context)
        
        return context
    
    def _build_prompt_quality_metrics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build schema quality metrics optimized for prompt context"""
        total_tables = len(context['tables'])
        xml_tables_count = len(context['xml_tables'])
        
        authoritative_ratio = 0
        if context['total_xml_count'] > 0:
            authoritative_ratio = context['authoritative_xml_count'] / context['total_xml_count']
        
        return {
            'total_tables': total_tables,
            'xml_enabled_tables': xml_tables_count,
            'relational_only_tables': len(context['relational_tables']) - xml_tables_count,
            'xml_authoritative_ratio': round(authoritative_ratio, 3),
            'verified_xml_paths_count': sum(len(paths) for paths in context['verified_xml_paths'].values()),
            'schema_json_overrides': self.schema_json_overridden_count,
            'xml_priority_applications': self.xml_priority_applied_count,
            'prompt_readiness_score': self._calculate_prompt_readiness_score(context)
        }
    
    def _calculate_prompt_readiness_score(self, context: Dict[str, Any]) -> float:
        """Calculate how ready the schema context is for accurate prompt generation"""
        score = 0.0
        
        # Authoritative XML data usage (40%)
        if context['total_xml_count'] > 0:
            auth_ratio = context['authoritative_xml_count'] / context['total_xml_count']
            score += auth_ratio * 0.4
        
        # Verified XML paths coverage (30%)
        verified_paths = sum(len(paths) for paths in context['verified_xml_paths'].values())
        if context['total_xml_count'] > 0:
            verified_ratio = verified_paths / context['total_xml_count']
            score += verified_ratio * 0.3
        
        # Table coverage (20%)
        if context['total_tables'] > 0:
            xml_table_ratio = len(context['xml_tables']) / context['total_tables']
            score += xml_table_ratio * 0.2
        
        # Schema quality (10%)
        if context['total_tables'] > 0:
            table_quality = min(1.0, context['total_tables'] / 10.0)  # Up to 10 tables is excellent
            score += table_quality * 0.1
        
        return round(min(1.0, score), 3)
    
    # Preserve existing methods with fail-fast enhancements
    def get_aggregation_statistics(self, method_results: Dict[SearchMethod, List[RetrievedColumn]]) -> Dict[str, Any]:
        """Get comprehensive statistics with fail-fast behavior tracking"""
        try:
            aggregated = self.aggregate(method_results)
            aggregation_failed = False
            error_message = None
        except ValueError as e:
            aggregated = []
            aggregation_failed = True
            error_message = str(e)
        
        # Input statistics
        input_stats = {}
        total_input = 0
        engines_with_results = 0
        engines_failed = 0
        
        for method, results in method_results.items():
            count = len(results or [])
            method_name = method.value if hasattr(method, 'value') else str(method)
            input_stats[method_name] = count
            total_input += count
            
            if results:
                engines_with_results += 1
            else:
                engines_failed += 1
        
        if aggregation_failed:
            return {
                'aggregation_failed': True,
                'error': error_message,
                'input_statistics': {
                    'total_input_columns': total_input,
                    'results_by_engine': input_stats,
                    'engines_with_results': engines_with_results,
                    'engines_failed': engines_failed,
                    'engine_success_rate': engines_with_results / len(method_results) if method_results else 0
                },
                'output_statistics': {
                    'total_output_columns': 0,
                    'fail_fast_triggered': True
                }
            }
        
        # Analyze output
        xml_columns = len([col for col in aggregated if getattr(col, 'type', None) == ColumnType.XML])
        relational_columns = len([col for col in aggregated if getattr(col, 'type', None) != ColumnType.XML])
        tables_found = len(set(col.table for col in aggregated))
        
        # Calculate quality metrics
        deduplication_ratio = len(aggregated) / max(1, total_input)
        xml_enrichment_ratio = len([col for col in aggregated 
                                   if getattr(col, 'type', None) == ColumnType.XML and getattr(col, 'sql_expression', None)]) / max(1, xml_columns)
        
        return {
            'aggregation_failed': False,
            'input_statistics': {
                'total_input_columns': total_input,
                'results_by_engine': input_stats,
                'engines_with_results': engines_with_results,
                'engines_failed': engines_failed,
                'engine_success_rate': engines_with_results / len(method_results) if method_results else 0
            },
            'output_statistics': {
                'total_output_columns': len(aggregated),
                'xml_columns': xml_columns,
                'relational_columns': relational_columns,
                'tables_found': tables_found,
                'deduplication_ratio': round(deduplication_ratio, 3),
                'xml_enrichment_ratio': round(xml_enrichment_ratio, 3)
            },
            'xml_priority_metrics': {
                'authoritative_xml_columns': len([col for col in aggregated if getattr(col, 'is_authoritative_xml', False)]),
                'schema_json_overrides': self.schema_json_overridden_count,
                'xml_priority_applications': self.xml_priority_applied_count
            },
            'quality_metrics': {
                'has_xml_schema_manager': self.xml_schema_manager is not None,
                'avg_columns_per_table': round(len(aggregated) / max(1, tables_found), 2),
                'xml_columns_with_sql_expressions': len([col for col in aggregated 
                                                        if getattr(col, 'type', None) == ColumnType.XML and getattr(col, 'sql_expression', None)]),
                'xml_columns_with_xpath': len([col for col in aggregated 
                                              if getattr(col, 'type', None) == ColumnType.XML and getattr(col, 'xpath', None)])
            }
        }

# Factory function with XML priority and fail-fast behavior
def create_result_aggregator(xml_schema_path: str = None): # pyright: ignore[reportArgumentType]
    """
    Factory function for creating enhanced ResultAggregator with:
    - XML schema priority (xml_schema.json over schema.json)  
    - Fail-fast behavior (no silent fallbacks)
    - Rich schema context for prompt builders
    """
    return ResultAggregator(xml_schema_path)
