"""
Metadata Loader for NLP Processor
Loads rich metadata infrastructure from existing data sources
Leverages schema.json, joins_verified.json, xml_schema.json, tables.json
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)

class MetadataLoader:
    """
    Load and manage rich metadata from existing data infrastructure
    Handles schema, joins, XML mappings, and table metadata
    """
    
    def __init__(self, data_root_path: Optional[str] = None):
        """Initialize metadata loader with data paths"""
        self.data_root = Path(data_root_path) if data_root_path else self._get_default_data_path()
        self.metadata_path = self.data_root / "metadata"
        
        # Cache for loaded metadata
        self._schema_cache = None
        self._joins_cache = None
        self._xml_cache = None
        self._tables_cache = None
        self._cache_timestamp = None
        
        # Statistics
        self.load_stats = {
            "schema_columns": 0,
            "verified_joins": 0,
            "xml_fields": 0,
            "tables_loaded": 0,
            "last_loaded": None
        }
        
        logger.info(f"MetadataLoader initialized with data path: {self.data_root}")
    
    def _get_default_data_path(self) -> Path:
        """Get default data path relative to project structure"""
        # From agent/nlp_processor/utils/ go to data/
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent  # Up to sql-ai-agent/
        return project_root / "data"
    
    def load_all_metadata(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load all metadata from existing data sources
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            Complete metadata dictionary
        """
        try:
            if self._is_cache_valid() and not force_reload:
                logger.info("Using cached metadata")
                return self._get_cached_metadata()
            
            logger.info("Loading fresh metadata from data sources...")
            
            # Load all metadata components
            schema_data = self.load_schema_metadata()
            joins_data = self.load_join_intelligence()
            xml_data = self.load_xml_mappings()
            tables_data = self.load_table_metadata()
            
            # Update cache
            self._update_cache(schema_data, joins_data, xml_data, tables_data)
            
            # Update statistics
            self._update_statistics(schema_data, joins_data, xml_data, tables_data)
            
            complete_metadata = {
                "schema": schema_data,
                "joins": joins_data,
                "xml_mappings": xml_data,
                "tables": tables_data,
                "load_timestamp": datetime.now(),
                "statistics": self.load_stats
            }
            
            logger.info(f"Metadata loaded successfully: {self.load_stats}")
            return complete_metadata
            
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise
    
    def load_schema_metadata(self) -> List[Dict[str, Any]]:
        """
        Load schema metadata from schema.json
        Contains column descriptions, data types, business context
        
        Returns:
            List of column metadata with business descriptions
        """
        schema_file = self.metadata_path / "schema.json"
        
        try:
            if not schema_file.exists():
                logger.warning(f"Schema file not found: {schema_file}")
                return []
            
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)
            
            # Ensure we have a list
            if not isinstance(schema_data, list):
                logger.warning(f"Schema data is not a list, got: {type(schema_data)}")
                return []
            
            # Process and enhance schema data
            enhanced_schema = self._enhance_schema_data(schema_data)
            
            logger.info(f"Loaded schema metadata: {len(enhanced_schema)} columns")
            return enhanced_schema
            
        except Exception as e:
            logger.error(f"Error loading schema metadata: {e}")
            return []
    
    def load_join_intelligence(self) -> List[Dict[str, Any]]:
        """
        Load join intelligence from joins_verified.json
        Contains verified relationships with confidence scores
        
        Returns:
            List of verified join relationships
        """
        joins_file = self.metadata_path / "joins_verified.json"
        
        try:
            if not joins_file.exists():
                logger.warning(f"Joins file not found: {joins_file}")
                return []
            
            with open(joins_file, 'r', encoding='utf-8') as f:
                joins_data = json.load(f)
            
            # Ensure we have a list
            if not isinstance(joins_data, list):
                logger.warning(f"Joins data is not a list, got: {type(joins_data)}")
                return []
            
            # Process and enhance join data
            enhanced_joins = self._enhance_join_data(joins_data)
            
            # Sort by confidence score (highest first)
            enhanced_joins.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            logger.info(f"Loaded join intelligence: {len(enhanced_joins)} verified joins")
            return enhanced_joins
            
        except Exception as e:
            logger.error(f"Error loading join intelligence: {e}")
            return []
    
    def load_xml_mappings(self) -> Dict[str, Any]:
        """
        Load XML schema mappings from xml_schema.json
        Contains 5,603 XML fields with SQL expressions
        
        Returns:
            Dictionary of XML mappings by table
        """
        xml_file = self.metadata_path / "xml_schema.json"
        
        try:
            if not xml_file.exists():
                logger.warning(f"XML schema file not found: {xml_file}")
                return {}
            
            with open(xml_file, 'r', encoding='utf-8') as f:
                xml_data = json.load(f)
            
            # Ensure we have a list
            if not isinstance(xml_data, list):
                logger.warning(f"XML data is not a list, got: {type(xml_data)}")
                return {}
            
            # Organize XML data by table for efficient access
            organized_xml = self._organize_xml_by_table(xml_data)
            
            logger.info(f"Loaded XML mappings: {len(xml_data)} XML fields across {len(organized_xml)} tables")
            return organized_xml
            
        except Exception as e:
            logger.error(f"Error loading XML mappings: {e}")
            return {}
    
    def load_table_metadata(self) -> Dict[str, Any]:
        """
        Load table metadata from tables.json
        Handles both table name arrays and table metadata objects
        
        Returns:
            Dictionary of table metadata
        """
        tables_file = self.metadata_path / "tables.json"
        
        try:
            if not tables_file.exists():
                logger.warning(f"Tables file not found: {tables_file}")
                return {}
            
            with open(tables_file, 'r', encoding='utf-8') as f:
                tables_data = json.load(f)
            
            # Handle different table data formats
            if isinstance(tables_data, list):
                # Check if list contains strings (table names) or objects (metadata)
                if all(isinstance(item, str) for item in tables_data):
                    # Convert list of table names to dictionary with default metadata
                    tables_dict = {table_name: {"name": table_name} for table_name in tables_data}
                else:
                    # List of objects - convert to dictionary
                    tables_dict = {}
                    for table_info in tables_data:
                        if isinstance(table_info, dict):
                            table_name = table_info.get('table', table_info.get('name', 'unknown'))
                            tables_dict[table_name] = table_info
                        else:
                            # Fallback for mixed types
                            table_name = str(table_info)
                            tables_dict[table_name] = {"name": table_name}
            elif isinstance(tables_data, dict):
                # Already a dictionary
                tables_dict = tables_data
            else:
                # Handle case where tables_data is a single string or other type
                logger.warning(f"Unexpected tables data type: {type(tables_data)}")
                tables_dict = {}
            
            # Enhance table data with business context
            enhanced_tables = self._enhance_table_data(tables_dict)
            
            logger.info(f"Loaded table metadata: {len(enhanced_tables)} tables")
            return enhanced_tables
            
        except Exception as e:
            logger.error(f"Error loading table metadata: {e}")
            return {}
    
    def _enhance_schema_data(self, schema_data: List[Dict]) -> List[Dict]:
        """Enhance schema data with additional business intelligence"""
        enhanced = []
        
        for column_info in schema_data:
            if not isinstance(column_info, dict):
                logger.warning(f"Column info is not a dict: {type(column_info)}")
                continue
                
            enhanced_column = {
                **column_info,
                "business_searchable": self._extract_business_keywords(column_info.get('description', '')),
                "is_lookup_key": self._is_lookup_key(column_info),
                "is_foreign_key": self._is_foreign_key(column_info),
                "aggregatable": self._is_aggregatable(column_info),
                "temporal": self._is_temporal_column(column_info)
            }
            enhanced.append(enhanced_column)
        
        return enhanced
    
    def _enhance_join_data(self, joins_data: List[Dict]) -> List[Dict]:
        """Enhance join data with additional intelligence"""
        enhanced = []
        
        for join_info in joins_data:
            if not isinstance(join_info, dict):
                logger.warning(f"Join info is not a dict: {type(join_info)}")
                continue
                
            enhanced_join = {
                **join_info,
                "join_type_recommended": self._recommend_join_type(join_info),
                "business_context": self._get_join_business_context(join_info),
                "query_frequency": self._estimate_query_frequency(join_info),
                "performance_score": self._calculate_performance_score(join_info)
            }
            enhanced.append(enhanced_join)
        
        return enhanced
    
    def _organize_xml_by_table(self, xml_data: List[Dict]) -> Dict[str, Dict]:
        """Organize XML data by table for efficient access"""
        organized = {}
        
        for xml_field in xml_data:
            if not isinstance(xml_field, dict):
                logger.warning(f"XML field is not a dict: {type(xml_field)}")
                continue
                
            table = xml_field.get('table', 'unknown')
            
            if table not in organized:
                organized[table] = {
                    'xml_column': xml_field.get('xml_column'),
                    'fields': [],
                    'field_count': 0,
                    'business_domains': set()
                }
            
            # Add field with enhanced metadata
            enhanced_field = {
                **xml_field,
                "business_searchable": self._extract_xml_business_keywords(xml_field.get('name', '')),
                "data_type_inferred": self._infer_xml_data_type(xml_field.get('sql_expression', '')),
                "aggregatable": self._is_xml_aggregatable(xml_field)
            }
            
            organized[table]['fields'].append(enhanced_field)
            organized[table]['field_count'] += 1
            
            # Extract business domain
            business_domain = self._extract_business_domain(xml_field.get('name', ''))
            if business_domain:
                organized[table]['business_domains'].add(business_domain)
        
        # Convert sets to lists for JSON serialization
        for table_data in organized.values():
            table_data['business_domains'] = list(table_data['business_domains'])
        
        return organized
    
    def _enhance_table_data(self, tables_data: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Enhance table data with business intelligence
        Handles both simple table names and complex metadata objects
        """
        enhanced = {}
        
        for table_name, table_info in tables_data.items():
            # Ensure table_info is a dictionary
            if isinstance(table_info, str):
                table_info = {"name": table_info}
            elif not isinstance(table_info, dict):
                table_info = {"name": str(table_info)}
            
            # Add enhanced metadata
            enhanced[table_name] = {
                **table_info,
                "query_patterns": self._extract_query_patterns(table_name),
                "business_domain": self._extract_table_business_domain(table_name),
                "analyst_relevance": self._calculate_analyst_relevance(table_name),
                "table_type": self._determine_table_type(table_name),
                "common_joins": self._predict_common_joins(table_name)
            }
        
        return enhanced
    
    def _determine_table_type(self, table_name: str) -> str:
        """Determine the type/category of the table"""
        table_lower = table_name.lower()
        
        if 'counterparty' in table_lower or 'ctpt' in table_lower:
            return 'counterparty_related'
        elif 'application' in table_lower or 'swf' in table_lower:
            return 'application_workflow'
        elif 'user' in table_lower or 'role' in table_lower:
            return 'user_management'
        elif 'tracker' in table_lower:
            return 'tracking_workflow'
        elif 'schedule' in table_lower:
            return 'scheduling'
        else:
            return 'operational'
    
    def _predict_common_joins(self, table_name: str) -> List[str]:
        """Predict common join patterns for this table"""
        table_lower = table_name.lower()
        common_joins = []
        
        join_patterns = {
            'tblCounterparty': ['tblCTPTAddress', 'tblCTPTContactDetails', 'tblOApplicationMaster'],
            'tblCTPTAddress': ['tblCounterparty'],
            'tblCTPTContactDetails': ['tblCounterparty'],
            'tblOApplicationMaster': ['tblCounterparty', 'tblOSWFActionStatusApplicationTracker']
        }
        
        # Direct mapping
        if table_name in join_patterns:
            common_joins.extend(join_patterns[table_name])
        
        # Pattern-based predictions
        if 'ctpt' in table_lower and table_name != 'tblCounterparty':
            common_joins.append('tblCounterparty')
        
        if 'application' in table_lower and 'tblCounterparty' not in common_joins:
            common_joins.append('tblCounterparty')
        
        return list(set(common_joins))  # Remove duplicates
    
    # Helper methods for enhancement
    def _extract_business_keywords(self, description: str) -> List[str]:
        """Extract business searchable keywords from column descriptions"""
        keywords = []
        if not isinstance(description, str):
            return keywords
            
        description_lower = description.lower()
        
        # Common business terms
        business_terms = [
            'name', 'id', 'identifier', 'code', 'status', 'date', 'amount', 
            'address', 'email', 'phone', 'region', 'country', 'state',
            'application', 'counterparty', 'customer', 'client', 'owner'
        ]
        
        for term in business_terms:
            if term in description_lower:
                keywords.append(term)
        
        return keywords
    
    def _is_lookup_key(self, column_info: Dict) -> bool:
        """Determine if column is a lookup key"""
        column_name = column_info.get('column', '').lower()
        return any(suffix in column_name for suffix in ['_id', '_code', 'id', 'code'])
    
    def _is_foreign_key(self, column_info: Dict) -> bool:
        """Determine if column is a foreign key"""
        column_name = column_info.get('column', '').lower()
        return 'ctpt_id' in column_name or '_id' in column_name
    
    def _is_aggregatable(self, column_info: Dict) -> bool:
        """Determine if column can be aggregated"""
        datatype = column_info.get('datatype', '').lower()
        numeric_types = ['int', 'decimal', 'float', 'money', 'numeric']
        return any(num_type in datatype for num_type in numeric_types)
    
    def _is_temporal_column(self, column_info: Dict) -> bool:
        """Determine if column contains temporal data"""
        column_name = column_info.get('column', '').lower()
        datatype = column_info.get('datatype', '').lower()
        
        return ('date' in column_name or 'time' in column_name or 
                'created' in column_name or 'updated' in column_name or
                'date' in datatype or 'time' in datatype)
    
    def _recommend_join_type(self, join_info: Dict) -> str:
        """Recommend optimal join type based on confidence and context"""
        confidence = join_info.get('confidence', 0)
        if confidence >= 90:
            return 'INNER'
        elif confidence >= 70:
            return 'LEFT'
        else:
            return 'LEFT_WITH_VALIDATION'
    
    def _get_join_business_context(self, join_info: Dict) -> str:
        """Extract business context for join relationship"""
        source = join_info.get('source', '')
        target = join_info.get('target', '')
        
        context_map = {
            'tblCounterparty': 'counterparty_data',
            'tblCTPTAddress': 'geographic_context',
            'tblCTPTContactDetails': 'communication_info',
            'tblOApplicationMaster': 'application_workflow'
        }
        
        source_context = context_map.get(source, 'unknown')
        target_context = context_map.get(target, 'unknown')
        
        return f"{source_context}_to_{target_context}"
    
    def _estimate_query_frequency(self, join_info: Dict) -> str:
        """Estimate how frequently this join is used in queries"""
        if join_info.get('verified', False) and join_info.get('confidence', 0) >= 90:
            return 'high'
        elif join_info.get('confidence', 0) >= 70:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_performance_score(self, join_info: Dict) -> int:
        """Calculate performance score for join (0-100)"""
        score = 50  # Base score
        
        if join_info.get('verified', False):
            score += 30
        
        confidence = join_info.get('confidence', 0)
        score += min(confidence * 0.2, 20)  # Max 20 points from confidence
        
        return min(score, 100)
    
    def _extract_xml_business_keywords(self, field_name: str) -> List[str]:
        """Extract business keywords from XML field names"""
        keywords = []
        if not isinstance(field_name, str):
            return keywords
            
        field_lower = field_name.lower()
        
        xml_business_terms = [
            'family', 'member', 'turnover', 'profit', 'supplier', 'lead',
            'arrangement', 'business', 'financial', 'legal', 'compliance'
        ]
        
        for term in xml_business_terms:
            if term in field_lower:
                keywords.append(term)
        
        return keywords
    
    def _infer_xml_data_type(self, sql_expression: str) -> str:
        """Infer data type from XML SQL expression"""
        if not isinstance(sql_expression, str):
            return 'string'
            
        sql_expr_lower = sql_expression.lower()
        if 'varchar' in sql_expr_lower:
            return 'string'
        elif 'int' in sql_expr_lower:
            return 'integer'
        elif 'decimal' in sql_expr_lower or 'money' in sql_expr_lower:
            return 'decimal'
        elif 'date' in sql_expr_lower:
            return 'date'
        else:
            return 'string'
    
    def _is_xml_aggregatable(self, xml_field: Dict) -> bool:
        """Determine if XML field can be aggregated"""
        sql_expr = xml_field.get('sql_expression', '').lower()
        return 'decimal' in sql_expr or 'money' in sql_expr or 'int' in sql_expr
    
    def _extract_business_domain(self, field_name: str) -> str:
        """Extract business domain from XML field name"""
        if not isinstance(field_name, str):
            return 'general'
            
        field_lower = field_name.lower()
        
        if 'family' in field_lower or 'member' in field_lower:
            return 'family_structure'
        elif 'turnover' in field_lower or 'profit' in field_lower:
            return 'financial_metrics'
        elif 'supplier' in field_lower:
            return 'supply_chain'
        elif 'lead' in field_lower:
            return 'lead_management'
        else:
            return 'general'
    
    def _extract_query_patterns(self, table_name: str) -> List[str]:
        """Extract likely query patterns for table"""
        patterns = []
        if not isinstance(table_name, str):
            return ['general queries']
            
        table_lower = table_name.lower()
        
        pattern_map = {
            'counterparty': ['customer details', 'counterparty info', 'client lookup'],
            'address': ['location queries', 'regional analysis', 'geographic filtering'],
            'contact': ['contact information', 'communication details'],
            'application': ['application status', 'workflow tracking', 'loan processing'],
            'deviation': ['deviation analysis', 'exception tracking'],
            'collateral': ['collateral analysis', 'security valuation']
        }
        
        for key, pattern_list in pattern_map.items():
            if key in table_lower:
                patterns.extend(pattern_list)
        
        return patterns if patterns else ['general queries']
    
    def _extract_table_business_domain(self, table_name: str) -> str:
        """Extract business domain for table"""
        if not isinstance(table_name, str):
            return 'operational'
            
        table_lower = table_name.lower()
        
        if 'counterparty' in table_lower or 'ctpt' in table_lower:
            return 'counterparty_management'
        elif 'application' in table_lower or 'swf' in table_lower:
            return 'application_workflow'
        elif 'user' in table_lower or 'role' in table_lower:
            return 'user_management'
        else:
            return 'operational'
    
    def _calculate_analyst_relevance(self, table_name: str) -> str:
        """Calculate how relevant table is for analyst queries"""
        if not isinstance(table_name, str):
            return 'low'
            
        table_lower = table_name.lower()
        
        high_relevance = ['counterparty', 'application', 'deviation', 'collateral']
        medium_relevance = ['address', 'contact', 'owner']
        
        if any(term in table_lower for term in high_relevance):
            return 'high'
        elif any(term in table_lower for term in medium_relevance):
            return 'medium'
        else:
            return 'low'
    
    def _update_cache(self, schema_data, joins_data, xml_data, tables_data):
        """Update internal cache"""
        self._schema_cache = schema_data
        self._joins_cache = joins_data
        self._xml_cache = xml_data
        self._tables_cache = tables_data
        self._cache_timestamp = datetime.now()
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is valid (not older than 1 hour)"""
        if not self._cache_timestamp:
            return False
        
        cache_age = (datetime.now() - self._cache_timestamp).total_seconds()
        return cache_age < 3600  # 1 hour
    
    def _get_cached_metadata(self) -> Dict[str, Any]:
        """Get cached metadata"""
        return {
            "schema": self._schema_cache,
            "joins": self._joins_cache,
            "xml_mappings": self._xml_cache,
            "tables": self._tables_cache,
            "load_timestamp": self._cache_timestamp,
            "statistics": self.load_stats
        }
    
    def _update_statistics(self, schema_data, joins_data, xml_data, tables_data):
        """Update loading statistics"""
        self.load_stats.update({
            "schema_columns": len(schema_data) if schema_data else 0,
            "verified_joins": len(joins_data) if joins_data else 0,
            "xml_fields": sum(len(table_data.get('fields', [])) for table_data in xml_data.values()) if xml_data else 0,
            "tables_loaded": len(tables_data) if tables_data else 0,
            "last_loaded": datetime.now().isoformat()
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get loader statistics"""
        return {
            **self.load_stats,
            "cache_valid": self._is_cache_valid(),
            "data_path": str(self.data_root),
            "metadata_path": str(self.metadata_path)
        }
    
    def get_table_schema(self, table_name: str) -> List[Dict]:
        """Get schema for specific table"""
        if not self._schema_cache:
            self.load_all_metadata()
        
        return [col for col in self._schema_cache if col.get('table') == table_name] # type: ignore
    
    def get_table_joins(self, table_name: str) -> List[Dict]:
        """Get joins for specific table"""
        if not self._joins_cache:
            self.load_all_metadata()
        
        return [join for join in self._joins_cache  # type: ignore
                if join.get('source') == table_name or join.get('target') == table_name]
    
    def get_xml_fields(self, table_name: str) -> Dict[str, Any]:
        """Get XML fields for specific table"""
        if not self._xml_cache:
            self.load_all_metadata()
        
        return self._xml_cache.get(table_name, {}) # type: ignore

# Singleton instance for global access
_metadata_loader_instance = None

def get_metadata_loader() -> MetadataLoader:
    """Get singleton metadata loader instance"""
    global _metadata_loader_instance
    if _metadata_loader_instance is None:
        _metadata_loader_instance = MetadataLoader()
    return _metadata_loader_instance

# Test function
def main():
    """Test metadata loader functionality"""
    loader = MetadataLoader()
    
    try:
        print("Starting metadata loading test...")
        metadata = loader.load_all_metadata()
        print("Metadata loaded successfully!")
        print(f"Statistics: {loader.get_statistics()}")
        
        # Test specific queries
        counterparty_schema = loader.get_table_schema('tblCounterparty')
        print(f"Counterparty columns: {len(counterparty_schema)}")
        
        counterparty_joins = loader.get_table_joins('tblCounterparty')
        print(f"Counterparty joins: {len(counterparty_joins)}")
        
        counterparty_xml = loader.get_xml_fields('tblCounterparty')
        print(f"Counterparty XML fields: {counterparty_xml.get('field_count', 0)}")
        
        # Show sample data
        if counterparty_schema:
            print(f"Sample counterparty column: {counterparty_schema[0].get('column', 'unknown')}")
        
        # Show table information
        print("\nLoaded tables:")
        for table_name in list(metadata['tables'].keys())[:5]:  # Show first 5 tables
            table_info = metadata['tables'][table_name]
            print(f"- {table_name}: {table_info.get('business_domain', 'unknown')} domain")
        
    except Exception as e:
        print(f"Error testing metadata loader: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
