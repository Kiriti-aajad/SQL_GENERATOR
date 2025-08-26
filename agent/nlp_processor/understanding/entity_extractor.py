"""
Business Entity Extraction for Professional Analyst Queries
Extracts and maps business entities to your database schema
Handles counterparties, temporal expressions, regions, amounts, etc.
FIXED: Added JSON serialization support to prevent BusinessEntity errors
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from agent.nlp_processor.core.data_models import BusinessEntity, DatabaseField, FieldType
from agent.nlp_processor.config_module import get_config
from agent.nlp_processor.utils.metadata_loader import get_metadata_loader

logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    Extract business entities from professional analyst queries
    Maps entities to your database schema and field definitions
    """
    
    def __init__(self):
        """Initialize entity extractor with schema intelligence"""
        self.config = get_config()
        self.metadata_loader = get_metadata_loader()
        
        # Load metadata for intelligent field mapping
        self.metadata = self.metadata_loader.load_all_metadata()
        
        # Entity extraction thresholds
        self.entity_threshold = self.config.get('understanding.entity_threshold', 0.6)
        
        # Initialize extraction patterns
        self._initialize_extraction_patterns()
        self._build_schema_mappings()
        
        logger.info("Entity extractor initialized with schema mappings")
    
    def _initialize_extraction_patterns(self):
        """Initialize entity extraction patterns for banking domain"""
        
        # Counterparty name patterns
        self.counterparty_patterns = {
            'company_suffixes': [
                r'\b\w+\s+(corp|corporation|company|co|ltd|limited|inc|incorporated|bank|financial|services)\b',
                r'\b\w+\s+(pvt|private|public|holding|group|enterprises|solutions)\b'
            ],
            'business_indicators': [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+(corp|ltd|inc|bank)\b',
                r'\b[A-Z]{2,}\s+(corp|ltd|inc|bank)\b'
            ],
            'quoted_names': [
                r'"([^"]+)"',
                r"'([^']+)'"
            ]
        }
        
        # Temporal expression patterns
        self.temporal_patterns = {
            'relative_dates': [
                (r'last\s+(\d+)\s+(days?|weeks?|months?|years?)', 'relative_past'),
                (r'past\s+(\d+)\s+(days?|weeks?|months?|years?)', 'relative_past'),
                (r'within\s+(\d+)\s+(days?|weeks?|months?|years?)', 'relative_within'),
                (r'recent(?:ly)?', 'relative_recent'),
                (r'today', 'absolute_today'),
                (r'yesterday', 'absolute_yesterday')
            ],
            'absolute_dates': [
                (r'\b(\d{4}-\d{2}-\d{2})\b', 'date_iso'),
                (r'\b(\d{1,2}/\d{1,2}/\d{4})\b', 'date_us'),
                (r'\b(\d{1,2}-\d{1,2}-\d{4})\b', 'date_hyphen')
            ]
        }
        
        # Geographic/Regional patterns
        self.geographic_patterns = {
            'indian_states': [
                'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh',
                'goa', 'gujarat', 'haryana', 'himachal pradesh', 'jharkhand',
                'karnataka', 'kerala', 'madhya pradesh', 'maharashtra', 'manipur',
                'meghalaya', 'mizoram', 'nagaland', 'odisha', 'punjab',
                'rajasthan', 'sikkim', 'tamil nadu', 'telangana', 'tripura',
                'uttar pradesh', 'uttarakhand', 'west bengal'
            ],
            'major_cities': [
                'delhi', 'mumbai', 'bangalore', 'hyderabad', 'chennai', 'kolkata',
                'pune', 'ahmedabad', 'surat', 'jaipur', 'lucknow', 'kanpur',
                'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam', 'patna'
            ],
            'regional_terms': [
                'north', 'south', 'east', 'west', 'central', 'northeast', 'northwest',
                'southeast', 'southwest', 'northern', 'southern', 'eastern', 'western'
            ]
        }
        
        # Numerical/Amount patterns
        self.numerical_patterns = {
            'amounts': [
                (r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(lakh|crore|thousand|million|billion)', 'indian_currency'),
                (r'rs\.?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', 'rupees'),
                (r'inr\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', 'inr'),
                (r'(\d+(?:,\d{3})*(?:\.\d{2})?)', 'plain_number')
            ],
            'percentages': [
                (r'(\d+(?:\.\d+)?)\s*%', 'percentage'),
                (r'(\d+(?:\.\d+)?)\s*percent', 'percentage')
            ]
        }
        
        # Application/ID patterns
        self.identifier_patterns = {
            'application_ids': [
                (r'\b(APP\d+)\b', 'application_id'),
                (r'\b(APPL\d+)\b', 'application_id'),
                (r'\b([A-Z]{2,}\d{4,})\b', 'possible_id')
            ],
            'counterparty_ids': [
                (r'\b(CTPT\d+)\b', 'counterparty_id'),
                (r'\b(CP\d+)\b', 'counterparty_id'),
                (r'\b(CUST\d+)\b', 'customer_id')
            ]
        }
    
    def _build_schema_mappings(self):
        """Build mappings from business terms to database fields"""
        
        self.field_mappings = {
            'counterparty_fields': {},
            'temporal_fields': {},
            'geographic_fields': {},
            'amount_fields': {},
            'identifier_fields': {}
        }
        
        # Map schema fields to categories
        for column_info in self.metadata.get('schema', []):
            table = column_info.get('table', '')
            column = column_info.get('column', '')
            description = column_info.get('description', '').lower()
            
            # Create database field object
            field = DatabaseField(
                name=column,
                table=table,
                field_type=FieldType.PHYSICAL_COLUMN,
                data_type=column_info.get('datatype', ''),
                description=column_info.get('description', ''),
                aggregatable=column_info.get('aggregatable', False),
                temporal=column_info.get('temporal', False),
                business_keywords=column_info.get('business_searchable', [])
            )
            
            # Categorize fields
            if 'counterparty' in description or 'ctpt' in column.lower():
                self.field_mappings['counterparty_fields'][column.lower()] = field
            
            if column_info.get('temporal', False):
                self.field_mappings['temporal_fields'][column.lower()] = field
            
            if any(geo in description for geo in ['address', 'state', 'region', 'location']):
                self.field_mappings['geographic_fields'][column.lower()] = field
            
            if column_info.get('aggregatable', False):
                self.field_mappings['amount_fields'][column.lower()] = field
            
            if any(id_term in column.lower() for id_term in ['id', 'code', 'number']):
                self.field_mappings['identifier_fields'][column.lower()] = field
        
        # Add XML fields to mappings
        for table_name, xml_info in self.metadata.get('xml_mappings', {}).items():
            for xml_field in xml_info.get('fields', []):
                field = DatabaseField(
                    name=xml_field.get('name', ''), # type: ignore
                    table=table_name,
                    field_type=FieldType.XML_FIELD,
                    data_type=xml_field.get('data_type_inferred', 'string'),
                    sql_expression=xml_field.get('sql_expression', ''),
                    xpath=xml_field.get('xpath', ''),
                    aggregatable=xml_field.get('aggregatable', False),
                    business_keywords=xml_field.get('business_searchable', [])
                )
                
                # Categorize XML fields
                field_name_lower = field.name.lower()
                if any(term in field_name_lower for term in ['family', 'member', 'lead']):
                    self.field_mappings['counterparty_fields'][field_name_lower] = field
                
                if any(term in field_name_lower for term in ['amount', 'value', 'turnover']):
                    self.field_mappings['amount_fields'][field_name_lower] = field
    
    # CRITICAL FIX: Changed return type and added serialization
    def extract(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Extract business entities from analyst query
        
        Args:
            query_text: Natural language query from analyst
            context: Optional context including intent information
            
        Returns:
            List of extracted business entities as JSON-serializable dictionaries
        """
        entities = []
        query_lower = query_text.lower().strip()
        
        # Extract different types of entities
        entities.extend(self._extract_counterparty_entities(query_text, query_lower))
        entities.extend(self._extract_temporal_entities(query_text, query_lower))
        entities.extend(self._extract_geographic_entities(query_text, query_lower))
        entities.extend(self._extract_numerical_entities(query_text, query_lower))
        entities.extend(self._extract_identifier_entities(query_text, query_lower))
        
        # Filter entities by confidence threshold
        filtered_entities = [e for e in entities if e.confidence >= self.entity_threshold]
        
        # CRITICAL FIX: Convert BusinessEntity objects to JSON-serializable dictionaries
        serializable_entities = [entity.to_dict() for entity in filtered_entities]
        
        logger.info(f"Extracted {len(serializable_entities)} entities from query")
        return serializable_entities
    
    def _extract_counterparty_entities(self, original_query: str, query_lower: str) -> List[BusinessEntity]:
        """Extract counterparty/company name entities"""
        entities = []
        
        # Check for quoted company names first (highest confidence)
        for pattern in self.counterparty_patterns['quoted_names']:
            matches = re.finditer(pattern, original_query, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                if len(name) > 2:  # Minimum length check
                    entities.append(BusinessEntity(
                        entity_type="counterparty",
                        entity_value=name,
                        confidence=0.9,
                        table_mapping="tblCounterparty",
                        field_mapping=self._find_best_field_mapping('counterparty', 'CounterpartyName'),
                        field_type=FieldType.PHYSICAL_COLUMN
                    ))
        
        # Check for company suffix patterns
        for pattern in self.counterparty_patterns['company_suffixes']:
            matches = re.finditer(pattern, original_query, re.IGNORECASE)
            for match in matches:
                company_name = match.group(0).strip()
                entities.append(BusinessEntity(
                    entity_type="counterparty",
                    entity_value=company_name,
                    confidence=0.8,
                    table_mapping="tblCounterparty",
                    field_mapping=self._find_best_field_mapping('counterparty', 'CounterpartyName'),
                    field_type=FieldType.PHYSICAL_COLUMN
                ))
        
        # Check for business indicator patterns
        for pattern in self.counterparty_patterns['business_indicators']:
            matches = re.finditer(pattern, original_query, re.IGNORECASE)
            for match in matches:
                company_name = match.group(0).strip()
                entities.append(BusinessEntity(
                    entity_type="counterparty",
                    entity_value=company_name,
                    confidence=0.7,
                    table_mapping="tblCounterparty",
                    field_mapping=self._find_best_field_mapping('counterparty', 'CounterpartyName'),
                    field_type=FieldType.PHYSICAL_COLUMN
                ))
        
        return entities
    
    def _extract_temporal_entities(self, original_query: str, query_lower: str) -> List[BusinessEntity]:
        """Extract temporal/date entities"""
        entities = []
        
        # Extract relative dates
        for pattern, date_type in self.temporal_patterns['relative_dates']:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                temporal_expr = match.group(0)
                
                # Calculate actual date if possible
                calculated_date = self._calculate_temporal_value(temporal_expr, date_type)
                
                entities.append(BusinessEntity(
                    entity_type="temporal",
                    entity_value=temporal_expr,
                    confidence=0.9,
                    table_mapping="multiple",
                    field_mapping=self._find_best_field_mapping('temporal', 'created_date'),
                    field_type=FieldType.PHYSICAL_COLUMN
                ))
                
                # Add calculated date as additional context
                if calculated_date:
                    entities.append(BusinessEntity(
                        entity_type="calculated_date",
                        entity_value=calculated_date,
                        confidence=0.8,
                        table_mapping="multiple",
                        field_mapping=self._find_best_field_mapping('temporal', 'created_date'),
                        field_type=FieldType.PHYSICAL_COLUMN
                    ))
        
        # Extract absolute dates
        for pattern, date_type in self.temporal_patterns['absolute_dates']:
            matches = re.finditer(pattern, original_query)
            for match in matches:
                date_value = match.group(1)
                entities.append(BusinessEntity(
                    entity_type="absolute_date",
                    entity_value=date_value,
                    confidence=0.95,
                    table_mapping="multiple",
                    field_mapping=self._find_best_field_mapping('temporal', 'created_date'),
                    field_type=FieldType.PHYSICAL_COLUMN
                ))
        
        return entities
    
    def _extract_geographic_entities(self, original_query: str, query_lower: str) -> List[BusinessEntity]:
        """Extract geographic/regional entities"""
        entities = []
        
        # Check for Indian states (highest confidence for specific states)
        for state in self.geographic_patterns['indian_states']:
            if state in query_lower:
                entities.append(BusinessEntity(
                    entity_type="state",
                    entity_value=state.title(),
                    confidence=0.9,
                    table_mapping="tblCTPTAddress",
                    field_mapping=self._find_best_field_mapping('geographic', 'StateAddress'),
                    field_type=FieldType.PHYSICAL_COLUMN
                ))
        
        # Check for major cities
        for city in self.geographic_patterns['major_cities']:
            if city in query_lower:
                entities.append(BusinessEntity(
                    entity_type="city",
                    entity_value=city.title(),
                    confidence=0.8,
                    table_mapping="tblCTPTAddress",
                    field_mapping=self._find_best_field_mapping('geographic', 'StateAddress'),
                    field_type=FieldType.PHYSICAL_COLUMN
                ))
        
        # Check for regional terms
        for region in self.geographic_patterns['regional_terms']:
            if region in query_lower:
                entities.append(BusinessEntity(
                    entity_type="region",
                    entity_value=region.title(),
                    confidence=0.7,
                    table_mapping="tblCTPTAddress",
                    field_mapping=self._find_best_field_mapping('geographic', 'StateAddress'),
                    field_type=FieldType.PHYSICAL_COLUMN
                ))
        
        return entities
    
    def _extract_numerical_entities(self, original_query: str, query_lower: str) -> List[BusinessEntity]:
        """Extract numerical/amount entities"""
        entities = []
        
        # Extract amounts with currency indicators
        for pattern, amount_type in self.numerical_patterns['amounts']:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                amount_value = match.group(0)
                entities.append(BusinessEntity(
                    entity_type="amount",
                    entity_value=amount_value,
                    confidence=0.8,
                    table_mapping="multiple",
                    field_mapping=self._find_best_field_mapping('amount', 'amount'),
                    field_type=FieldType.PHYSICAL_COLUMN
                ))
        
        # Extract percentages
        for pattern, percentage_type in self.numerical_patterns['percentages']:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                percentage_value = match.group(0)
                entities.append(BusinessEntity(
                    entity_type="percentage",
                    entity_value=percentage_value,
                    confidence=0.8,
                    table_mapping="multiple",
                    field_mapping=self._find_best_field_mapping('amount', 'percentage'),
                    field_type=FieldType.PHYSICAL_COLUMN
                ))
        
        return entities
    
    def _extract_identifier_entities(self, original_query: str, query_lower: str) -> List[BusinessEntity]:
        """Extract ID/identifier entities"""
        entities = []
        
        # Extract application IDs
        for pattern, id_type in self.identifier_patterns['application_ids']:
            matches = re.finditer(pattern, original_query, re.IGNORECASE)
            for match in matches:
                id_value = match.group(1)
                entities.append(BusinessEntity(
                    entity_type="application_id",
                    entity_value=id_value,
                    confidence=0.9,
                    table_mapping="tblOApplicationMaster",
                    field_mapping=self._find_best_field_mapping('identifier', 'ApplicationID'),
                    field_type=FieldType.PHYSICAL_COLUMN
                ))
        
        # Extract counterparty IDs
        for pattern, id_type in self.identifier_patterns['counterparty_ids']:
            matches = re.finditer(pattern, original_query, re.IGNORECASE)
            for match in matches:
                id_value = match.group(1)
                entities.append(BusinessEntity(
                    entity_type="counterparty_id",
                    entity_value=id_value,
                    confidence=0.9,
                    table_mapping="tblCounterparty",
                    field_mapping=self._find_best_field_mapping('identifier', 'CTPT_ID'),
                    field_type=FieldType.PHYSICAL_COLUMN
                ))
        
        return entities
    
    def _find_best_field_mapping(self, category: str, preferred_field: str = None) -> Optional[DatabaseField]: # type: ignore
        """Find best field mapping for entity category"""
        
        category_key = f"{category}_fields"
        if category_key not in self.field_mappings:
            return None
        
        available_fields = self.field_mappings[category_key]
        
        # Try preferred field first
        if preferred_field and preferred_field.lower() in available_fields:
            return available_fields[preferred_field.lower()]
        
        # Return first available field if any
        if available_fields:
            return list(available_fields.values())[0]
        
        return None
    
    def _calculate_temporal_value(self, temporal_expr: str, date_type: str) -> Optional[str]:
        """Calculate actual date from temporal expression"""
        try:
            current_date = datetime.now()
            
            if date_type == 'relative_past' or date_type == 'relative_within':
                # Extract number and unit from expression
                match = re.search(r'(\d+)\s+(days?|weeks?|months?|years?)', temporal_expr)
                if match:
                    num = int(match.group(1))
                    unit = match.group(2).rstrip('s')  # Remove plural 's'
                    
                    if unit == 'day':
                        target_date = current_date - timedelta(days=num)
                    elif unit == 'week':
                        target_date = current_date - timedelta(weeks=num)
                    elif unit == 'month':
                        target_date = current_date - timedelta(days=num * 30)  # Approximation
                    elif unit == 'year':
                        target_date = current_date - timedelta(days=num * 365)  # Approximation
                    else:
                        return None
                    
                    return target_date.strftime('%Y-%m-%d')
            
            elif date_type == 'relative_recent':
                # Default to last 7 days for "recent"
                target_date = current_date - timedelta(days=7)
                return target_date.strftime('%Y-%m-%d')
            
            elif date_type == 'absolute_today':
                return current_date.strftime('%Y-%m-%d')
            
            elif date_type == 'absolute_yesterday':
                target_date = current_date - timedelta(days=1)
                return target_date.strftime('%Y-%m-%d')
        
        except Exception as e:
            logger.warning(f"Could not calculate temporal value for '{temporal_expr}': {e}")
            return None
        
        return None
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get entity extraction capability statistics"""
        return {
            "field_mappings": {
                "counterparty_fields": len(self.field_mappings['counterparty_fields']),
                "temporal_fields": len(self.field_mappings['temporal_fields']),
                "geographic_fields": len(self.field_mappings['geographic_fields']),
                "amount_fields": len(self.field_mappings['amount_fields']),
                "identifier_fields": len(self.field_mappings['identifier_fields'])
            },
            "pattern_counts": {
                "counterparty_patterns": sum(len(patterns) for patterns in self.counterparty_patterns.values()),
                "temporal_patterns": sum(len(patterns) for patterns in self.temporal_patterns.values()),
                "geographic_locations": sum(len(locations) for locations in self.geographic_patterns.values()),
                "numerical_patterns": sum(len(patterns) for patterns in self.numerical_patterns.values()),
                "identifier_patterns": sum(len(patterns) for patterns in self.identifier_patterns.values())
            },
            "confidence_threshold": self.entity_threshold
        }

def main():
    """Test entity extractor functionality"""
    try:
        extractor = EntityExtractor()
        print("Entity extractor initialized successfully!")
        
        # Get extraction statistics
        stats = extractor.get_extraction_statistics()
        print(f"Extraction capabilities: {stats}")
        
        # Test with sample analyst queries
        test_queries = [
            "Give me last 10 days created customers",
            "Which regions have maximum defaulters for ABC Corporation",
            "Show me deviations for XYZ Ltd in Delhi region",
            "What is the sum of collateral for CTPT123",
            "Recent applications in Mumbai with amount above 50 lakhs",
            "Customer details for Tata Motors Limited"
        ]
        
        print(f"\nTesting entity extraction with {len(test_queries)} queries:")
        
        for query in test_queries:
            entities = extractor.extract(query)
            print(f"\nQuery: '{query}'")
            print(f"Extracted {len(entities)} entities:")
            
            for entity_dict in entities:
                print(f"  - {entity_dict['entity_type']}: '{entity_dict['entity_value']}' "
                      f"(confidence: {entity_dict['confidence']:.2f}, table: {entity_dict.get('table_mapping', 'N/A')})")
        
    except Exception as e:
        print(f"Error testing entity extractor: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
