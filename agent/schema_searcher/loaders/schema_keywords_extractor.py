"""
Schema Keyword Extractor - Generate meaningful keywords from database schema
Creates a YAML file with schema-derived keywords for accurate query validation
"""

import yaml
import re
import json
import os
import sys
from typing import List, Dict, Set, Any, Optional
from pathlib import Path
import argparse

class SchemaKeywordExtractor:
    """Extract meaningful keywords from database schema files"""
    
    def __init__(self):
        self.keywords: Set[str] = set()
        self.schema_stats = {
            'tables_processed': 0,
            'columns_processed': 0,
            'xml_fields_processed': 0,
            'keywords_generated': 0
        }
    
    def extract_from_schema_loader_data(self, schema_data: List[Dict[str, Any]]) -> Set[str]:
        """Extract keywords from SchemaLoader data format"""
        keywords = set()
        
        for item in schema_data:
            table_name = item.get('table', '')
            column_name = item.get('column', '')
            description = item.get('description', '')
            datatype = item.get('datatype', '')
            
            # Extract from table name
            if table_name:
                keywords.update(self._extract_words_from_identifier(table_name))
                self.schema_stats['tables_processed'] += 1
            
            # Extract from column name
            if column_name:
                keywords.update(self._extract_words_from_identifier(column_name))
                self.schema_stats['columns_processed'] += 1
            
            # Extract from description
            if description:
                keywords.update(self._extract_words_from_text(description))
            
            # Extract from datatype
            if datatype:
                keywords.add(datatype.lower())
        
        return keywords
    
    def extract_from_xml_loader_data(self, xml_data: List[Dict[str, Any]]) -> Set[str]:
        """Extract keywords from XMLLoader data format"""
        keywords = set()
        
        for xml_table in xml_data:
            table_name = xml_table.get('table', '')
            xml_column = xml_table.get('xml_column', '')
            
            # Extract from table and xml column names
            if table_name:
                keywords.update(self._extract_words_from_identifier(table_name))
            
            if xml_column:
                keywords.update(self._extract_words_from_identifier(xml_column))
            
            # Extract from XML fields
            fields = xml_table.get('fields', [])
            for field in fields:
                field_name = field.get('name', '')
                xpath = field.get('xpath', '')
                sql_expression = field.get('sql_expression', '')
                
                if field_name:
                    keywords.update(self._extract_words_from_identifier(field_name))
                    self.schema_stats['xml_fields_processed'] += 1
                
                if xpath:
                    # Extract element names from XPath
                    xpath_elements = re.findall(r'/([^/\[\]]+)', xpath)
                    for element in xpath_elements:
                        keywords.update(self._extract_words_from_identifier(element))
                
                if sql_expression:
                    keywords.update(self._extract_words_from_text(sql_expression))
        
        return keywords
    
    def _extract_words_from_identifier(self, identifier: str) -> Set[str]:
        """Extract meaningful words from database identifiers (table/column names)"""
        words = set()
        
        if not identifier:
            return words
        
        # Add the full identifier (lowercase)
        words.add(identifier.lower())
        
        # Split camelCase and PascalCase: "CustomerName" -> ["customer", "name"]
        camel_words = re.findall(r'[A-Z][a-z]*|[a-z]+', identifier)
        words.update(word.lower() for word in camel_words if len(word) > 1)
        
        # Split snake_case: "customer_name" -> ["customer", "name"]
        snake_words = identifier.split('_')
        words.update(word.lower() for word in snake_words if len(word) > 1)
        
        # Split by common separators
        for separator in ['-', '.', ' ']:
            if separator in identifier:
                separated_words = identifier.split(separator)
                words.update(word.lower() for word in separated_words if len(word) > 1)
        
        # Remove common prefixes/suffixes to get root words
        root_words = set()
        for word in words:
            # Remove common database prefixes
            for prefix in ['tbl', 'tb', 'table', 'col', 'column', 'fk', 'pk', 'idx']:
                if word.startswith(prefix) and len(word) > len(prefix):
                    root_words.add(word[len(prefix):])
            
            # Remove common suffixes
            for suffix in ['id', 'key', 'num', 'nbr', 'cd', 'code']:
                if word.endswith(suffix) and len(word) > len(suffix):
                    root_words.add(word[:-len(suffix)])
        
        words.update(root_words)
        
        # Filter out very short or meaningless words
        meaningful_words = {
            word for word in words 
            if len(word) >= 2 and not word.isdigit() and word.isalpha()
        }
        
        return meaningful_words
    
    def _extract_words_from_text(self, text: str) -> Set[str]:
        """Extract meaningful words from description text"""
        words = set()
        
        if not text:
            return words
        
        # Extract words using regex (3+ characters, alphabetic)
        text_words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy',
            'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'this', 'that',
            'with', 'have', 'from', 'they', 'know', 'want', 'been', 'good', 'much',
            'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long',
            'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'
        }
        
        meaningful_words = {word for word in text_words if word not in stop_words}
        words.update(meaningful_words)
        
        return words
    
    def load_schema_from_loaders(self) -> Set[str]:
        """Load schema data using SchemaLoader and XMLLoader"""
        all_keywords = set()
        
        try:
            # Import your schema loaders
            from agent.schema_searcher.loaders.schema_loader import SchemaLoader
            from agent.schema_searcher.loaders.xml_loader import XMLLoader
            
            # Load relational schema data
            print("Loading relational schema data...")
            schema_loader = SchemaLoader()
            schema_data = schema_loader.load()
            relational_keywords = self.extract_from_schema_loader_data(schema_data)
            all_keywords.update(relational_keywords)
            print(f"  - Extracted {len(relational_keywords)} keywords from relational schema")
            
            # Load XML schema data
            print("Loading XML schema data...")
            xml_loader = XMLLoader()
            xml_data = xml_loader.load()
            xml_keywords = self.extract_from_xml_loader_data(xml_data)
            all_keywords.update(xml_keywords)
            print(f"  - Extracted {len(xml_keywords)} keywords from XML schema")
            
        except ImportError as e:
            print(f"Could not import schema loaders: {e}")
            print("Using sample data instead...")
            # Fallback to sample data
            all_keywords.update(self._get_sample_keywords())
        
        except Exception as e:
            print(f"Error loading schema data: {e}")
            print("Using sample data instead...")
            all_keywords.update(self._get_sample_keywords())
        
        return all_keywords
    
    def _get_sample_keywords(self) -> Set[str]:
        """Fallback sample keywords for demonstration"""
        return {
            'customer', 'account', 'order', 'product', 'user', 'name', 'id', 
            'date', 'amount', 'balance', 'email', 'address', 'phone', 'status',
            'code', 'type', 'category', 'description', 'number', 'reference'
        }
    
    def load_from_sql_files(self, sql_dir: str) -> Set[str]:
        """Extract keywords from SQL schema files"""
        keywords = set()
        
        sql_path = Path(sql_dir)
        if not sql_path.exists():
            print(f"SQL directory not found: {sql_dir}")
            return keywords
        
        for sql_file in sql_path.glob('*.sql'):
            print(f"Processing SQL file: {sql_file.name}")
            try:
                content = sql_file.read_text(encoding='utf-8')
                file_keywords = self._extract_from_sql_content(content)
                keywords.update(file_keywords)
            except Exception as e:
                print(f"Error processing {sql_file}: {e}")
        
        return keywords
    
    def _extract_from_sql_content(self, sql_content: str) -> Set[str]:
        """Extract table and column names from SQL content"""
        keywords = set()
        
        # Extract CREATE TABLE statements
        create_table_pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:`?)(\w+)(?:`?)'
        table_matches = re.findall(create_table_pattern, sql_content, re.IGNORECASE)
        
        for table_name in table_matches:
            keywords.update(self._extract_words_from_identifier(table_name))
        
        # Extract column names from CREATE TABLE statements
        # This is a simplified pattern - you might need more sophisticated parsing
        column_pattern = r'(?:`?)(\w+)(?:`?)\s+(?:INT|VARCHAR|TEXT|DATETIME|DECIMAL|BOOLEAN|CHAR)'
        column_matches = re.findall(column_pattern, sql_content, re.IGNORECASE)
        
        for column_name in column_matches:
            keywords.update(self._extract_words_from_identifier(column_name))
        
        return keywords
    
    def generate_yaml_config(self, keywords: Set[str], output_path: str) -> Dict[str, Any]:
        """Generate YAML configuration with schema keywords"""
        
        # Sort keywords for consistent output
        sorted_keywords = sorted(keywords)
        self.schema_stats['keywords_generated'] = len(sorted_keywords)
        
        # Create comprehensive YAML structure
        yaml_config = {
            'schema_keywords': {
                'version': '1.0',
                'generated_at': str(Path().resolve()),
                'statistics': self.schema_stats,
                'keywords': {
                    'core_terms': sorted_keywords,
                    'common_database_terms': [
                        'table', 'column', 'data', 'field', 'record', 'schema', 'database',
                        'index', 'key', 'primary', 'foreign', 'constraint', 'relation'
                    ],
                    'search_action_terms': [
                        'search', 'find', 'get', 'show', 'list', 'display', 'retrieve',
                        'select', 'filter', 'sort', 'group', 'aggregate', 'calculate'
                    ]
                },
                'validation_rules': {
                    'min_query_length': 3,
                    'require_core_terms': True,
                    'allow_action_terms': True,
                    'reject_patterns': ['xyz', 'abc', 'qwerty', 'asdf', 'lorem', 'ipsum', 'nonsense', 'random']
                }
            }
        }
        
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, indent=2)
        
        return yaml_config

def main():
    """Main function to run schema keyword extraction"""
    parser = argparse.ArgumentParser(description='Extract keywords from database schema')
    parser.add_argument('--output', '-o', default='schema_keywords.yaml', help='Output YAML file')
    parser.add_argument('--sql-dir', help='Directory containing SQL schema files')
    parser.add_argument('--use-loaders', action='store_true', help='Use SchemaLoader and XMLLoader')
    
    args = parser.parse_args()
    
    extractor = SchemaKeywordExtractor()
    
    print("🎯 Schema Keyword Extractor")
    print("=" * 50)
    
    # Extract keywords from different sources
    all_keywords = set()
    
    if args.use_loaders:
        print("📊 Loading schema data from loaders...")
        loader_keywords = extractor.load_schema_from_loaders()
        all_keywords.update(loader_keywords)
    
    if args.sql_dir:
        print(f"📁 Loading schema data from SQL files in {args.sql_dir}...")
        sql_keywords = extractor.load_from_sql_files(args.sql_dir)
        all_keywords.update(sql_keywords)
    
    if not all_keywords:
        print("⚠️ No schema sources specified. Using sample keywords for demonstration.")
        all_keywords.update(extractor._get_sample_keywords())
    
    # Generate YAML configuration
    print("📝 Generating YAML configuration...")
    yaml_config = extractor.generate_yaml_config(all_keywords, args.output)
    
    print("✅ Schema keyword extraction completed!")
    print(f"   📊 Statistics:")
    print(f"   - Tables processed: {extractor.schema_stats['tables_processed']}")
    print(f"   - Columns processed: {extractor.schema_stats['columns_processed']}")
    print(f"   - XML fields processed: {extractor.schema_stats['xml_fields_processed']}")
    print(f"   - Keywords generated: {extractor.schema_stats['keywords_generated']}")
    print(f"   📄 Output file: {args.output}")
    
    # Show sample keywords
    print(f"   🔍 Sample keywords:")
    sample_keywords = sorted(all_keywords)[:20]
    for i, keyword in enumerate(sample_keywords, 1):
        print(f"      {i:2d}. {keyword}")
    
    if len(all_keywords) > 20:
        print(f"      ... and {len(all_keywords) - 20} more")

if __name__ == "__main__":
    main()
