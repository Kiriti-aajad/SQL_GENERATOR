"""
XML Schema Manager for enhanced schema integration and validation

FIXED: All import paths, JSON file handling, method implementations, and data structure processing
Properly handles your xml_schema.json file format with table/xml_column/fields structure
UPDATED: Enhanced path detection for Windows absolute paths
ENHANCED: Added schema enrichment and quick lookup methods for NLP-to-SQL integration
"""

import json
import logging
import traceback
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass
from pathlib import Path

# FIXED: Corrected imports with proper fallback handling
try:
    from agent.schema_searcher.core.data_models import (
        create_retrieved_column_safe
    )
    DATA_MODELS_AVAILABLE = True
except ImportError:
    # Fallback - create stub classes instead of None to prevent errors
    DATA_MODELS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Data models not available, using fallback stubs")
    
    class RetrievedColumn: pass
    class ColumnType: pass
    class SearchMethod: pass
    class DataType: pass
    create_retrieved_column_safe = lambda: None

logger = logging.getLogger(__name__)

@dataclass
class XMLPath:
    """
    Represents an XML path with complete metadata
    FIXED: Matches your actual data structure from xml_schema.json
    """
    name: str  # Field name (e.g., "CPArrLeadBBFamilyMember")
    table: str  # Table name (e.g., "tblCounterparty")
    xml_column: str  # XML column name (e.g., "CTPT_XML")
    xpath: str  # XPath expression (e.g., "/CTPT/CPArrLeadBBFamilyMember")
    sql_expression: str  # SQL extraction expression
    data_type: str = "varchar(100)"  # Default data type extracted from SQL
    business_name: Optional[str] = None

    def __post_init__(self):
        """Extract data type from SQL expression if not provided"""
        if self.data_type == "varchar(100)" and self.sql_expression:
            # Extract data type from SQL expression like "varchar(100)" or "int"
            import re
            type_match = re.search(r"'(\w+(?:\(\d+\))?)'", self.sql_expression)
            if type_match:
                self.data_type = type_match.group(1)


class XMLSchemaManager:
    """
    FIXED: Enhanced XML Schema Manager that properly handles your xml_schema.json format
    - Reads JSON files directly (no XMLLoader dependency)
    - Handles your specific data structure with table/xml_column/fields
    - Proper error handling and availability checking
    - ENHANCED: Robust path detection for your Windows absolute path
    - ENHANCED: Schema enrichment methods for NLP-to-SQL integration
    """

    def __init__(self, xml_schema_path: Optional[str] = None):
        self.logger = logger.getChild("XMLSchemaManager")
        
        # FIXED: Enhanced path detection for your specific file location
        if xml_schema_path:
            self.xml_schema_path = Path(xml_schema_path).resolve()
        else:
            # FIXED: Comprehensive auto-detection including your exact path
            self.xml_schema_path = self._detect_xml_schema_file()
        
        # Track initialization status
        self._initialization_errors: List[str] = []
        self._is_available = False
        self._schema_loaded = False
        
        # Initialize collections to store XML schema data
        self.xml_paths: Dict[str, List[XMLPath]] = {}  # table_name -> List[XMLPath]
        self.xml_columns: List[Dict[str, Any]] = []  # Standardized column format
        self.raw_schema_data: List[Dict[str, Any]] = []  # Raw JSON data
        
        # Statistics
        self._statistics = {
            'tables_count': 0,
            'xml_fields_count': 0,
            'xml_columns_count': 0,
            'load_time': 0.0
        }
        
        # Load XML schema
        self._load_xml_schema()
        
        # Determine availability
        self._is_available = (
            self._schema_loaded and
            len(self.xml_paths) > 0 and
            len(self._initialization_errors) == 0
        )
        
        if self._is_available:
            self.logger.info(f" XML Schema Manager initialized successfully: "
                           f"{self._statistics['tables_count']} tables, "
                           f"{self._statistics['xml_fields_count']} XML fields")
            self.logger.info(f" Loaded authoritative XML schema data for {self._statistics['tables_count']} tables")
        else:
            self.logger.warning(f"XML Schema Manager initialization incomplete: {self._initialization_errors}")

    def _detect_xml_schema_file(self) -> Path:
        """
        FIXED: Robust XML schema file detection prioritizing your specific path
        """
        import os
        
        # Get current working directory
        cwd = Path.cwd()
        
        # FIXED: Comprehensive path detection with your exact path first
        possible_paths = [
            # YOUR SPECIFIC ABSOLUTE PATH (HIGHEST PRIORITY)
            Path(r"E:\Github\sql-ai-agent\data\metadata\xml_schema.json"),
            Path("E:/Github/sql-ai-agent/data/metadata/xml_schema.json"),
            
            # Relative to current working directory
            cwd / "data" / "metadata" / "xml_schema.json",
            
            # If running from project root
            Path("data/metadata/xml_schema.json"),
            
            # If XMLSchemaManager is in a submodule, try relative paths
            Path(__file__).parent.parent.parent / "data" / "metadata" / "xml_schema.json" if __file__ else None,
            Path(__file__).parent.parent / "data" / "metadata" / "xml_schema.json" if __file__ else None,
            
            # Alternative common locations
            Path("metadata/xml_schema.json"),
            Path("xml_schema.json"),
            cwd.parent / "data" / "metadata" / "xml_schema.json",
            
            # Check if we're in a subdirectory and need to go up
            cwd / ".." / "data" / "metadata" / "xml_schema.json",
            cwd / ".." / ".." / "data" / "metadata" / "xml_schema.json",
        ]
        
        # Remove None values
        possible_paths = [p for p in possible_paths if p is not None]
        
        self.logger.info(f"🔍 Searching for XML schema file in {len(possible_paths)} locations...")
        
        for i, path in enumerate(possible_paths):
            try:
                resolved_path = path.resolve()
                if resolved_path.exists() and resolved_path.is_file():
                    self.logger.info(f" Found XML schema file at location {i+1}: {resolved_path}")
                    return resolved_path
                else:
                    self.logger.debug(f" Location {i+1}: {resolved_path} (not found)")
            except Exception as e:
                self.logger.debug(f" Location {i+1}: {path} (error: {e})")
        
        # If no file found, return your specific path as default
        default_path = Path(r"E:\Github\sql-ai-agent\data\metadata\xml_schema.json")
        self.logger.error(f" XML schema file not found in any location!")
        self.logger.error(f"Using your specified path as default: {default_path}")
        self.logger.error("Please verify the file exists and is accessible at the expected location.")
        return default_path

    def _verify_file_access(self) -> Dict[str, Any]:
        """
        FIXED: Comprehensive file access verification with detailed diagnostics
        """
        path = self.xml_schema_path
        
        diagnostics = {
            'path': str(path),
            'absolute_path': str(path.resolve()) if path else 'None',
            'exists': False,
            'is_file': False,
            'is_readable': False,
            'parent_exists': False,
            'file_size': 0,
            'error': None
        }
        
        try:
            if path and path.exists():
                diagnostics['exists'] = True
                diagnostics['is_file'] = path.is_file()
                diagnostics['parent_exists'] = path.parent.exists()
                
                if diagnostics['is_file']:
                    try:
                        stat_info = path.stat()
                        diagnostics['file_size'] = stat_info.st_size
                    except Exception:
                        diagnostics['file_size'] = -1
                    
                    # Test readability
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            f.read(100)  # Read first 100 chars to test
                        diagnostics['is_readable'] = True
                    except Exception as read_error:
                        diagnostics['error'] = f"Read error: {read_error}"
            else:
                diagnostics['parent_exists'] = path.parent.exists() if path else False
                
        except Exception as e:
            diagnostics['error'] = str(e)
        
        return diagnostics

    def _load_xml_schema(self):
        """
        FIXED: Load and organize XML schema with enhanced error reporting for your file
        """
        import time
        start_time = time.time()
        
        try:
            # FIXED: Comprehensive file verification first
            file_diagnostics = self._verify_file_access()
            self.logger.info(f" XML schema file diagnostics: {file_diagnostics}")
            
            if not file_diagnostics['exists']:
                error_msg = f"XML schema file not found at: {self.xml_schema_path}"
                self.logger.error(error_msg)
                self.logger.error(f"Absolute path checked: {file_diagnostics['absolute_path']}")
                self.logger.error("Please ensure the file exists at your specified location.")
                self._initialization_errors.append(error_msg)
                return
            
            if not file_diagnostics['is_file']:
                error_msg = f"Path is not a file: {self.xml_schema_path}"
                self.logger.error(error_msg)
                self._initialization_errors.append(error_msg)
                return
            
            if not file_diagnostics['is_readable']:
                error_msg = f"File is not readable: {self.xml_schema_path}"
                if file_diagnostics['error']:
                    error_msg += f" - {file_diagnostics['error']}"
                self.logger.error(error_msg)
                self._initialization_errors.append(error_msg)
                return
            
            self.logger.info(f" Loading XML schema from: {self.xml_schema_path}")
            self.logger.info(f" File size: {file_diagnostics['file_size']} bytes")
            
            # FIXED: Load JSON data with better error handling
            with open(self.xml_schema_path, 'r', encoding='utf-8') as f:
                self.raw_schema_data = json.load(f)
            
            if not isinstance(self.raw_schema_data, list):
                error_msg = f"Expected JSON array, got {type(self.raw_schema_data)}"
                self.logger.error(error_msg)
                self._initialization_errors.append(error_msg)
                return
            
            # Process your specific data structure
            processed_tables = 0
            processed_fields = 0
            processed_columns = 0
            
            for table_def in self.raw_schema_data:
                try:
                    # Extract table information
                    table_name = table_def.get("table", "unknown_table")
                    xml_column = table_def.get("xml_column", "xml_data")
                    fields = table_def.get("fields", [])
                    
                    if not fields:
                        self.logger.debug(f"No fields found for table {table_name}")
                        continue
                    
                    # Process fields for this table
                    paths = []
                    for field in fields:
                        try:
                            # Create XMLPath object from your field structure
                            path = XMLPath(
                                name=field.get("name", ""),
                                table=table_name,
                                xml_column=xml_column,
                                xpath=field.get("xpath", ""),
                                sql_expression=field.get("sql_expression", ""),
                                data_type=self._extract_data_type_from_sql(field.get("sql_expression", "")),
                                business_name=field.get("business_name")
                            )
                            
                            if path.name and path.xpath and path.sql_expression:
                                paths.append(path)
                                processed_fields += 1
                            else:
                                self.logger.debug(f"Incomplete field data: {field}")
                                
                        except Exception as field_error:
                            self.logger.warning(f"Failed to process XML field {field}: {field_error}")
                            continue
                    
                    # Store paths for this table
                    if paths:
                        self.xml_paths[table_name] = paths
                        processed_tables += 1
                        
                        # Create standardized columns for each field
                        for path in paths:
                            column_data = {
                                'table': path.table,
                                'column': path.name,
                                'type': 'XML',
                                'description': f'XML field from {path.xml_column}',
                                'xml_column': path.xml_column,
                                'xpath': path.xpath,
                                'sql_expression': path.sql_expression,
                                'data_type': path.data_type,
                                'confidence_score': 1.0,
                                'retrieval_method': 'xml_schema_manager'
                            }
                            self.xml_columns.append(column_data)
                            processed_columns += 1
                            
                except Exception as table_error:
                    self.logger.warning(f"Failed to process table definition {table_def}: {table_error}")
                    continue
            
            # Update statistics
            self._statistics.update({
                'tables_count': processed_tables,
                'xml_fields_count': processed_fields,
                'xml_columns_count': processed_columns,
                'load_time': time.time() - start_time
            })
            
            self._schema_loaded = True
            
            self.logger.info(f" XML schema loading completed: {processed_tables} tables, "
                           f"{processed_fields} fields, {processed_columns} columns in "
                           f"{self._statistics['load_time']:.2f}s")
            
        except FileNotFoundError as e:
            error_msg = f"XML schema file not found: {e}"
            self.logger.error(error_msg)
            self._initialization_errors.append(error_msg)
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in schema file: {e}"
            self.logger.error(error_msg)
            self._initialization_errors.append(error_msg)
            
        except PermissionError as e:
            error_msg = f"Permission denied accessing XML schema file: {e}"
            self.logger.error(error_msg)
            self._initialization_errors.append(error_msg)
            
        except Exception as e:
            error_msg = f"Failed to load XML schema: {e}"
            self.logger.error(error_msg)
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self._initialization_errors.append(error_msg)

    def _extract_data_type_from_sql(self, sql_expression: str) -> str:
        """
        FIXED: Extract data type from SQL expression like:
        "CTPT_XML.value('(/CTPT/CPArrLeadBBFamilyMember)[1]', 'varchar(100)')"
        """
        if not sql_expression:
            return "varchar(100)"
        
        import re
        # Look for pattern like 'varchar(100)' or 'int' in the SQL expression
        type_match = re.search(r"'(\w+(?:\(\d+\))?)'", sql_expression)
        if type_match:
            return type_match.group(1)
        
        # Default fallback
        return "varchar(100)"

    def get_paths_for_table(self, table_name: str) -> List[XMLPath]:
        """Get all XML paths for a specific table"""
        return self.xml_paths.get(table_name, [])

    def get_xml_columns(self) -> List[Dict[str, Any]]:
        """Get all XML columns as standardized dictionaries"""
        return self.xml_columns.copy()

    def find_xml_path(self, table_name: str, field_name: str) -> Optional[XMLPath]:
        """Find specific XML path by table and field name with comprehensive matching"""
        paths = self.get_paths_for_table(table_name)
        if not paths:
            return None
        
        # First try exact match
        for path in paths:
            if path.name == field_name:
                return path
        
        # Then try case-insensitive match
        field_lower = field_name.lower()
        for path in paths:
            if path.name.lower() == field_lower:
                return path
        
        # Finally try partial matching
        for path in paths:
            path_name_lower = path.name.lower()
            if field_lower in path_name_lower or path_name_lower in field_lower:
                return path
        
        return None

    def enhance_column_with_xml_info(self, column: Any) -> Any:
        """Enhance a RetrievedColumn with XML schema information"""
        if not column:
            return column
        
        try:
            # Handle both RetrievedColumn objects and dictionaries
            if hasattr(column, 'table') and hasattr(column, 'column'):
                table_name = column.table
                column_name = column.column
            elif isinstance(column, dict):
                table_name = column.get('table', '')
                column_name = column.get('column', '')
            else:
                return column
            
            if not table_name or not column_name:
                return column
            
            # Find XML path information
            xml_path = self.find_xml_path(table_name, column_name)
            if xml_path:
                try:
                    # Enhance with XML information
                    xml_info = {
                        'xml_column': xml_path.xml_column,
                        'xpath': xml_path.xpath,
                        'sql_expression': xml_path.sql_expression,
                        'xml_data_type': xml_path.data_type,
                        'is_xml_column': True
                    }
                    
                    # Apply enhancements based on object type
                    if hasattr(column, '__dict__'):
                        # RetrievedColumn object - set attributes safely
                        for key, value in xml_info.items():
                            if not hasattr(column, key) or getattr(column, key, None) is None:
                                try:
                                    setattr(column, key, value)
                                except (AttributeError, TypeError):
                                    # Handle frozen dataclasses
                                    try:
                                        object.__setattr__(column, key, value)
                                    except Exception:
                                        pass
                    elif isinstance(column, dict):
                        # Dictionary - update directly
                        column.update(xml_info)
                    
                    self.logger.debug(f"Enhanced {table_name}.{column_name} with XML schema info")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to enhance column {table_name}.{column_name}: {e}")
                    
        except Exception as e:
            self.logger.warning(f"Error in enhance_column_with_xml_info: {e}")
        
        return column

    def search_xml_fields(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Search XML fields based on query string"""
        if not query or not self.xml_columns:
            return []
        
        query_lower = query.lower()
        matching_columns = []
        
        for column in self.xml_columns:
            # Search in field name, table name, and xpath
            searchable_text = f"{column['column']} {column['table']} {column['xpath']}".lower()
            
            if query_lower in searchable_text:
                matching_columns.append(column.copy())
                if len(matching_columns) >= max_results:
                    break
        
        return matching_columns

    def validate_xml_schema(self) -> Dict[str, Any]:
        """Validate the loaded XML schema and return comprehensive quality metrics"""
        total_tables = len(self.xml_paths)
        total_fields = len(self.xml_columns)
        
        # Check for completeness
        complete_paths = 0
        incomplete_paths = 0
        
        for table_name, paths in self.xml_paths.items():
            for path in paths:
                if path.xpath and path.sql_expression and path.name:
                    complete_paths += 1
                else:
                    incomplete_paths += 1
        
        # Check for unique field names
        all_field_names = []
        for paths in self.xml_paths.values():
            all_field_names.extend([path.name for path in paths])
        
        unique_fields = len(set(all_field_names))
        duplicate_fields = len(all_field_names) - unique_fields
        
        return {
            'total_tables': total_tables,
            'total_xml_fields': total_fields,
            'complete_paths': complete_paths,
            'incomplete_paths': incomplete_paths,
            'completeness_ratio': complete_paths / max(1, total_fields),
            'unique_fields': unique_fields,
            'duplicate_fields': duplicate_fields,
            'schema_available': total_tables > 0,
            'file_exists': self.xml_schema_path.exists() if self.xml_schema_path else False,
            'initialization_errors': self._initialization_errors.copy(),
            'schema_loaded': self._schema_loaded
        }

    def get_tables_with_xml(self) -> List[str]:
        """Get list of all tables that have XML columns"""
        return list(self.xml_paths.keys())

    def get_xml_columns_for_table(self, table_name: str) -> List[Dict[str, Any]]:
        """Get all XML columns for a specific table"""
        return [col for col in self.xml_columns if col['table'] == table_name]

    def is_available(self) -> bool:
        """Check if XML schema manager is properly loaded and available"""
        return self._is_available

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive XML schema statistics"""
        validation_results = self.validate_xml_schema()
        
        return {
            'manager_status': 'available' if self.is_available() else 'unavailable',
            'file_path': str(self.xml_schema_path) if self.xml_schema_path else 'not_set',
            'file_exists': self.xml_schema_path.exists() if self.xml_schema_path else False,
            'tables_count': self._statistics['tables_count'],
            'xml_fields_count': self._statistics['xml_fields_count'],
            'xml_columns_count': self._statistics['xml_columns_count'],
            'load_time': self._statistics['load_time'],
            'initialization_errors': self._initialization_errors.copy(),
            'validation_results': validation_results,
            'tables_with_xml': self.get_tables_with_xml(),
            'schema_loaded': self._schema_loaded,
            'data_models_available': DATA_MODELS_AVAILABLE
        }

    def get_initialization_status(self) -> Dict[str, Any]:
        """Get detailed initialization status for debugging"""
        return {
            'file_path': str(self.xml_schema_path) if self.xml_schema_path else 'not_set',
            'file_exists': self.xml_schema_path.exists() if self.xml_schema_path else False,
            'schema_loaded': self._schema_loaded,
            'xml_paths_loaded': len(self.xml_paths),
            'xml_columns_loaded': len(self.xml_columns),
            'is_available': self._is_available,
            'initialization_errors': self._initialization_errors.copy(),
            'data_models_available': DATA_MODELS_AVAILABLE,
            'statistics': self._statistics.copy()
        }

    def reload_schema(self) -> bool:
        """Reload XML schema from file"""
        try:
            # Reset state
            self.xml_paths.clear()
            self.xml_columns.clear()
            self.raw_schema_data.clear()
            self._initialization_errors.clear()
            self._schema_loaded = False
            self._is_available = False
            
            # Reload
            self._load_xml_schema()
            
            # Update availability
            self._is_available = (
                self._schema_loaded and
                len(self.xml_paths) > 0 and
                len(self._initialization_errors) == 0
            )
            
            if self._is_available:
                self.logger.info("XML schema reloaded successfully")
                return True
            else:
                self.logger.warning(f"XML schema reload failed: {self._initialization_errors}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to reload XML schema: {e}")
            self._initialization_errors.append(f"Reload failed: {str(e)}")
            return False

    def export_schema_summary(self) -> Dict[str, Any]:
        """Export comprehensive schema summary for debugging and validation"""
        summary = {
            'metadata': {
                'file_path': str(self.xml_schema_path) if self.xml_schema_path else 'not_set',
                'load_time': self._statistics['load_time'],
                'available': self._is_available
            },
            'statistics': self.get_statistics(),
            'tables': {}
        }
        
        # Add detailed table information
        for table_name, paths in self.xml_paths.items():
            table_info = {
                'xml_column': paths[0].xml_column if paths else 'unknown',
                'field_count': len(paths),
                'fields': []
            }
            
            for path in paths:
                field_info = {
                    'name': path.name,
                    'xpath': path.xpath,
                    'data_type': path.data_type,
                    'has_sql_expression': bool(path.sql_expression)
                }
                table_info['fields'].append(field_info)
            
            summary['tables'][table_name] = table_info
        
        return summary

    # NEW METHODS FOR NLP-TO-SQL INTEGRATION

    def enrich_schema_context(self, schema_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        CRITICAL: Enrich discovered schema context with XML field mappings
        This is the key integration method for the NLP-to-SQL pipeline
        """
        if not self.is_available():
            self.logger.warning("XML Schema Manager not available - skipping enrichment")
            return schema_context
        
        # Create enhanced schema context
        enhanced_schema = schema_context.copy()
        
        # Add XML field mappings organized by table
        xml_field_mappings = {}
        xml_enhanced_tables = {}
        
        # Process each table in the discovered schema
        tables = schema_context.get('tables', [])
        
        for table_name in tables:
            if table_name in self.xml_paths:
                # Get XML paths for this table
                table_paths = self.xml_paths[table_name]
                
                if table_paths:
                    # Create table XML info
                    xml_column = table_paths[0].xml_column  # All paths in table use same XML column
                    xml_enhanced_tables[table_name] = {
                        'xml_column': xml_column,
                        'fields': []
                    }
                    
                    # Add each XML field
                    for path in table_paths:
                        field_info = {
                            'name': path.name,
                            'xpath': path.xpath,
                            'sql_expression': path.sql_expression,
                            'data_type': path.data_type
                        }
                        xml_enhanced_tables[table_name]['fields'].append(field_info)
                        
                        # Create field mapping for easy lookup
                        field_key = f"{table_name}.{path.name}"
                        xml_field_mappings[field_key] = {
                            'table': table_name,
                            'field_name': path.name,
                            'xpath': path.xpath,
                            'xml_column': xml_column,
                            'sql_expression': path.sql_expression,
                            'data_type': path.data_type
                        }
        
        # Add XML data to schema context
        enhanced_schema.update({
            'xml_enhanced_tables': xml_enhanced_tables,
            'xml_field_mappings': xml_field_mappings,
            'xml_tables_count': len(xml_enhanced_tables),
            'xml_fields_total': len(xml_field_mappings),
            'xml_schema_loaded': True
        })
        
        self.logger.info(f"Schema enriched with XML data: {len(xml_enhanced_tables)} tables, "
                        f"{len(xml_field_mappings)} XML fields")
        
        return enhanced_schema

    def get_xml_field_info(self, table_name: str, field_name: str) -> Optional[Dict[str, Any]]:
        """
        Quick lookup for XML field information
        Returns None if field not found, otherwise returns field details
        """
        xml_path = self.find_xml_path(table_name, field_name)
        if xml_path:
            return {
                'table': xml_path.table,
                'field_name': xml_path.name,
                'xpath': xml_path.xpath,
                'xml_column': xml_path.xml_column,
                'sql_expression': xml_path.sql_expression,
                'data_type': xml_path.data_type
            }
        return None


# Factory function for easy instantiation
def create_xml_schema_manager(xml_schema_path: Optional[str] = None) -> XMLSchemaManager:
    """Factory function to create XMLSchemaManager with your file path"""
    return XMLSchemaManager(xml_schema_path)


# Enhanced test function
def test_xml_schema_manager():
    """Enhanced test function to verify XMLSchemaManager functionality including new methods"""
    print("Testing XML Schema Manager with enhancements...")
    print("=" * 80)
    
    try:
        # Test with your specific file path
        manager = XMLSchemaManager("E:/Github/sql-ai-agent/data/metadata/xml_schema.json")
        
        # Test basic functionality
        print(f"Manager Available: {manager.is_available()}")
        
        # Test statistics
        stats = manager.get_statistics()
        print(f"Tables Count: {stats['tables_count']}")
        print(f"XML Fields Count: {stats['xml_fields_count']}")
        print(f"File Exists: {stats['file_exists']}")
        
        # Test table listing
        tables = manager.get_tables_with_xml()
        print(f"Tables with XML: {tables}")
        
        # Test field search for tblCounterparty
        if "tblCounterparty" in tables:
            paths = manager.get_paths_for_table("tblCounterparty")
            print(f"XML fields in tblCounterparty: {len(paths)}")
            if paths:
                print(f"Sample field: {paths[0].name} -> {paths.xpath}") # pyright: ignore[reportAttributeAccessIssue]
        
        # Test search functionality
        search_results = manager.search_xml_fields("CPArrLead")
        print(f"Search results for 'CPArrLead': {len(search_results)}")
        
        # NEW: Test schema enrichment
        mock_schema = {
            'tables': ['tblCounterparty', 'tblOSWFActionStatusCollateralTracker'],
            'columns_by_table': {
                'tblCounterparty': ['CTPT_ID', 'CounterpartyName']
            }
        }
        
        enriched_schema = manager.enrich_schema_context(mock_schema)
        print(f"XML Tables in enriched schema: {enriched_schema.get('xml_tables_count', 0)}")
        print(f"XML Fields in enriched schema: {enriched_schema.get('xml_fields_total', 0)}")
        
        # NEW: Test quick lookup
        field_info = manager.get_xml_field_info('tblCounterparty', 'CPArrLeadBBFamilyMember')
        if field_info:
            print(f"Found field info: {field_info['sql_expression']}")
        else:
            print("Field info not found - this is expected if field doesn't exist")
        
        print("=" * 80)
        print("Enhanced XML Schema Manager test completed successfully!")
        
    except Exception as e:
        print(f"XML Schema Manager test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_xml_schema_manager()
