"""
Core Data Models for NLP Processor
Properly distinguishes between physical columns and XML fields
Leverages loaded metadata for enhanced query processing
FIXED: Added JSON serialization support to resolve BusinessEntity serialization errors
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
import json

# Fix the import issue - use absolute import or add path handling
try:
    from utils.metadata_loader import get_metadata_loader
except ImportError:
    # Fallback for when running from different contexts
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.metadata_loader import get_metadata_loader

class QueryType(Enum):
    """Professional analyst query types"""
    TEMPORAL_ANALYSIS = "temporal_analysis"
    REGIONAL_AGGREGATION = "regional_aggregation" 
    CUSTOMER_ANALYSIS = "customer_analysis"
    DEVIATION_ANALYSIS = "deviation_analysis"
    DEFAULTER_ANALYSIS = "defaulter_analysis"
    COLLATERAL_ANALYSIS = "collateral_analysis"

class FieldType(Enum):
    """Distinguish between field types"""
    PHYSICAL_COLUMN = "physical_column"
    XML_FIELD = "xml_field"
    COMPUTED_FIELD = "computed_field"

# ADD MISSING SEMANTIC PARSING ENUMS
class SemanticRelationType(Enum):
    """Types of semantic relationships"""
    SUBJECT_OBJECT = "subject_object"
    ATTRIBUTE = "attribute"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    COMPARATIVE = "comparative"
    AGGREGATION = "aggregation"
    FILTERING = "filtering"
    GROUPING = "grouping"
    ORDERING = "ordering"
    CAUSAL = "causal"
    CONDITIONAL = "conditional"

class ConceptType(Enum):
    """Types of semantic concepts"""
    ENTITY = "entity"
    ACTION = "action"
    ATTRIBUTE = "attribute"
    QUANTIFIER = "quantifier"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CONDITION = "condition"
    MEASURE = "measure"

@dataclass
class DatabaseField:
    """Represents a queryable field (column or XML field)"""
    name: str
    table: str
    field_type: FieldType
    data_type: str
    description: Optional[str] = None
    sql_expression: Optional[str] = None  # For XML fields
    xpath: Optional[str] = None  # For XML fields
    aggregatable: bool = False
    temporal: bool = False
    business_keywords: List[str] = field(default_factory=list)
    
    def __hash__(self):
        """Make DatabaseField hashable for use in sets and dictionaries"""
        return hash((
            self.name,
            self.table,
            self.field_type,
            self.data_type,
            self.sql_expression,
            self.xpath
        ))
    
    def __eq__(self, other):
        """Define equality for DatabaseField objects"""
        if not isinstance(other, DatabaseField):
            return False
        return (
            self.name == other.name and
            self.table == other.table and
            self.field_type == other.field_type and
            self.data_type == other.data_type and
            self.sql_expression == other.sql_expression and
            self.xpath == other.xpath
        )
    
    def __repr__(self):
        """Better string representation for debugging"""
        return f"DatabaseField({self.table}.{self.name}[{self.field_type.value}])"
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert DatabaseField to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert enum to string value
        if isinstance(self.field_type, FieldType):
            result['field_type'] = self.field_type.value
        return result
    
    def to_json(self):
        """Convert DatabaseField to JSON string"""
        return json.dumps(self.to_dict())

@dataclass
class TableStructure:
    """Complete table structure with physical and XML fields"""
    table_name: str
    physical_columns: List[DatabaseField]
    xml_fields: List[DatabaseField]
    business_domain: str
    analyst_relevance: str
    common_joins: List[str]
    
    @property
    def total_fields(self) -> int:
        return len(self.physical_columns) + len(self.xml_fields)
    
    @property
    def all_fields(self) -> List[DatabaseField]:
        return self.physical_columns + self.xml_fields
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert TableStructure to dictionary for JSON serialization"""
        return {
            'table_name': self.table_name,
            'physical_columns': [col.to_dict() for col in self.physical_columns],
            'xml_fields': [field.to_dict() for field in self.xml_fields],
            'business_domain': self.business_domain,
            'analyst_relevance': self.analyst_relevance,
            'common_joins': self.common_joins,
            'total_fields': self.total_fields
        }

@dataclass
class AnalystQuery:
    """Professional analyst query input"""
    query_text: str
    context: Dict[str, Any]
    query_type: str = "analyst_professional"
    timestamp: datetime = field(default_factory=datetime.now)
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert AnalystQuery to dictionary for JSON serialization"""
        return {
            'query_text': self.query_text,
            'context': self.context,
            'query_type': self.query_type,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

@dataclass
class IntentResult:
    """Intent classification result"""
    query_type: QueryType
    confidence: float
    temporal_context: Optional[str] = None
    aggregation_type: Optional[str] = None
    target_tables: List[str] = field(default_factory=list)
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert IntentResult to dictionary for JSON serialization"""
        return {
            'query_type': self.query_type.value if isinstance(self.query_type, QueryType) else str(self.query_type),
            'confidence': self.confidence,
            'temporal_context': self.temporal_context,
            'aggregation_type': self.aggregation_type,
            'target_tables': self.target_tables
        }

@dataclass
class BusinessEntity:
    """Extracted business entity with field mapping - FIXED: Added JSON serialization"""
    entity_type: str
    entity_value: str
    confidence: float
    table_mapping: Optional[str] = None
    field_mapping: Optional[DatabaseField] = None
    field_type: Optional[FieldType] = None
    
    # CRITICAL FIX: JSON Serialization Support
    def to_dict(self):
        """Convert BusinessEntity to dictionary for JSON serialization"""
        result = {
            'entity_type': self.entity_type,
            'entity_value': self.entity_value,
            'confidence': self.confidence,
            'table_mapping': self.table_mapping
        }
        
        # Handle nested DatabaseField object
        if self.field_mapping is not None:
            if hasattr(self.field_mapping, 'to_dict'):
                result['field_mapping'] = self.field_mapping.to_dict()
            else:
                result['field_mapping'] = str(self.field_mapping)
        else:
            result['field_mapping'] = None
        
        # Handle FieldType enum
        if self.field_type is not None:
            if isinstance(self.field_type, FieldType):
                result['field_type'] = self.field_type.value
            else:
                result['field_type'] = str(self.field_type)
        else:
            result['field_type'] = None
        
        return result
    
    def to_json(self):
        """Convert BusinessEntity to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create BusinessEntity from dictionary (for deserialization)"""
        # Handle field_type conversion back to enum
        field_type = None
        if data.get('field_type'):
            try:
                field_type = FieldType(data['field_type'])
            except (ValueError, KeyError):
                field_type = None
        
        # Handle field_mapping - simplified for now
        field_mapping = data.get('field_mapping')
        if isinstance(field_mapping, dict):
            # Could reconstruct DatabaseField here if needed
            field_mapping = None  # Simplified for now
        
        return cls(
            entity_type=data['entity_type'],
            entity_value=data['entity_value'],
            confidence=data['confidence'],
            table_mapping=data.get('table_mapping'),
            field_mapping=field_mapping,
            field_type=field_type
        )

@dataclass
class Token:
    """Represents a linguistic token with analysis"""
    text: str
    lemma: str
    pos_tag: str
    is_stopword: bool
    is_banking_term: bool
    is_entity: bool
    entity_type: Optional[str] = None
    confidence: float = 1.0
    semantic_role: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert Token to dictionary for JSON serialization"""
        return asdict(self)

@dataclass
class SemanticRelation:
    """Represents semantic relationship between tokens (linguistic analysis)"""
    relation_type: str
    head_token: str
    dependent_token: str
    confidence: float
    context: Optional[str] = None
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert SemanticRelation to dictionary for JSON serialization"""
        return asdict(self)

# ADD MISSING SEMANTIC PARSING CLASSES
@dataclass
class SemanticConcept:
    """Represents a semantic concept in the query"""
    concept_id: str
    concept_type: ConceptType
    surface_form: str  # Original text
    canonical_form: str  # Normalized form
    confidence: float
    properties: Dict[str, Any] = field(default_factory=dict)
    database_mapping: Optional[DatabaseField] = None
    constraints: List[str] = field(default_factory=list)
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert SemanticConcept to dictionary for JSON serialization"""
        result = asdict(self)
        # Handle enum
        if isinstance(self.concept_type, ConceptType):
            result['concept_type'] = self.concept_type.value
        # Handle nested DatabaseField
        if self.database_mapping and hasattr(self.database_mapping, 'to_dict'):
            result['database_mapping'] = self.database_mapping.to_dict()
        elif self.database_mapping:
            result['database_mapping'] = str(self.database_mapping)
        return result

@dataclass
class SemanticParseRelation:
    """Represents a semantic relationship between concepts (semantic parsing)"""
    relation_id: str
    relation_type: SemanticRelationType
    head_concept: str  # concept_id
    tail_concept: str  # concept_id
    confidence: float
    properties: Dict[str, Any] = field(default_factory=dict)
    linguistic_evidence: List[str] = field(default_factory=list)
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert SemanticParseRelation to dictionary for JSON serialization"""
        result = asdict(self)
        # Handle enum
        if isinstance(self.relation_type, SemanticRelationType):
            result['relation_type'] = self.relation_type.value
        return result

@dataclass
class SemanticFrame:
    """Represents a semantic frame (structured meaning representation)"""
    frame_id: str
    frame_type: str  # e.g., "query_intent", "aggregation", "comparison"
    core_concepts: List[str]  # concept_ids
    frame_elements: Dict[str, str] = field(default_factory=dict)  # role -> concept_id
    constraints: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert SemanticFrame to dictionary for JSON serialization"""
        return asdict(self)

@dataclass
class SemanticParseTree:
    """Hierarchical semantic representation"""
    root_concept: str
    concepts: Dict[str, SemanticConcept]
    relations: List[SemanticParseRelation]
    frames: List[SemanticFrame]
    discourse_structure: Dict[str, Any] = field(default_factory=dict)
    pragmatic_context: Dict[str, Any] = field(default_factory=dict)
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert SemanticParseTree to dictionary for JSON serialization"""
        return {
            'root_concept': self.root_concept,
            'concepts': {k: v.to_dict() for k, v in self.concepts.items()},
            'relations': [r.to_dict() for r in self.relations],
            'frames': [f.to_dict() for f in self.frames],
            'discourse_structure': self.discourse_structure,
            'pragmatic_context': self.pragmatic_context
        }

@dataclass 
class QuerySemantics:
    """Complete semantic representation of a banking query"""
    original_query: str
    parse_tree: SemanticParseTree
    query_intent_frame: SemanticFrame
    execution_semantics: Dict[str, Any]
    business_semantics: Dict[str, Any]
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    ambiguity_resolutions: List[str] = field(default_factory=list)
    semantic_validation: Dict[str, Any] = field(default_factory=dict)
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert QuerySemantics to dictionary for JSON serialization"""
        return {
            'original_query': self.original_query,
            'parse_tree': self.parse_tree.to_dict() if self.parse_tree else None,
            'query_intent_frame': self.query_intent_frame.to_dict() if self.query_intent_frame else None,
            'execution_semantics': self.execution_semantics,
            'business_semantics': self.business_semantics,
            'confidence_scores': self.confidence_scores,
            'ambiguity_resolutions': self.ambiguity_resolutions,
            'semantic_validation': self.semantic_validation
        }

@dataclass
class LinguisticAnalysis:
    """Complete linguistic analysis result"""
    original_query: str
    normalized_query: str
    tokens: List[Token]
    sentences: List[str]
    semantic_relations: List[SemanticRelation]
    banking_concepts: List[str]
    query_patterns: List[str]
    complexity_score: float
    readability_score: float
    ambiguity_indicators: List[str] = field(default_factory=list)
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert LinguisticAnalysis to dictionary for JSON serialization"""
        return {
            'original_query': self.original_query,
            'normalized_query': self.normalized_query,
            'tokens': [t.to_dict() for t in self.tokens],
            'sentences': self.sentences,
            'semantic_relations': [r.to_dict() for r in self.semantic_relations],
            'banking_concepts': self.banking_concepts,
            'query_patterns': self.query_patterns,
            'complexity_score': self.complexity_score,
            'readability_score': self.readability_score,
            'ambiguity_indicators': self.ambiguity_indicators
        }

@dataclass
class TemporalExpression:
    """Represents temporal expressions in queries"""
    original_text: str
    normalized_form: str
    temporal_type: str  # relative, absolute, range
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    confidence: float = 1.0
    granularity: str = "day"  # day, week, month, quarter, year
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert TemporalExpression to dictionary for JSON serialization"""
        return {
            'original_text': self.original_text,
            'normalized_form': self.normalized_form,
            'temporal_type': self.temporal_type,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'confidence': self.confidence,
            'granularity': self.granularity
        }

@dataclass
class GeographicEntity:
    """Represents geographic entities in queries"""
    original_text: str
    normalized_form: str
    geographic_type: str  # state, city, region, branch
    mapped_code: Optional[str] = None
    confidence: float = 1.0
    parent_geography: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert GeographicEntity to dictionary for JSON serialization"""
        return asdict(self)

@dataclass
class NumericalEntity:
    """Represents numerical entities in queries"""
    original_text: str
    normalized_value: float
    value_type: str  # amount, percentage, count, ratio
    currency: Optional[str] = None
    scale: Optional[str] = None  # thousand, lakh, crore
    confidence: float = 1.0
    context: Optional[str] = None
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert NumericalEntity to dictionary for JSON serialization"""
        return asdict(self)

@dataclass
class ValidationResult:
    """Result of query validation"""
    is_valid: bool
    confidence_score: float
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert ValidationResult to dictionary for JSON serialization"""
        return asdict(self)

@dataclass
class ProcessingMetrics:
    """Metrics for query processing performance"""
    total_processing_time: float
    component_times: Dict[str, float] = field(default_factory=dict)
    memory_usage: Optional[float] = None
    cache_hits: int = 0
    cache_misses: int = 0
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert ProcessingMetrics to dictionary for JSON serialization"""
        return asdict(self)

@dataclass
class EnhancedSchemaContext:
    """Schema context enhanced with proper field distinction"""
    table_structures: Dict[str, TableStructure]
    join_intelligence: List[Dict[str, Any]]
    total_physical_columns: int
    total_xml_fields: int
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert EnhancedSchemaContext to dictionary for JSON serialization"""
        return {
            'table_structures': {k: v.to_dict() for k, v in self.table_structures.items()},
            'join_intelligence': self.join_intelligence,
            'total_physical_columns': self.total_physical_columns,
            'total_xml_fields': self.total_xml_fields
        }
    
    @classmethod
    def from_metadata_loader(cls):
        """Create from your loaded metadata with proper field classification"""
        loader = get_metadata_loader()
        metadata = loader.load_all_metadata()
        
        table_structures = {}
        total_physical = 0
        total_xml = 0
        
        # Process each table
        for table_name, table_info in metadata.get('tables', {}).items():
            # Get physical columns from schema
            schema_columns = loader.get_table_schema(table_name)
            physical_fields = []
            
            for col_info in schema_columns:
                field = DatabaseField(
                    name=col_info.get('column', ''),
                    table=table_name,
                    field_type=FieldType.PHYSICAL_COLUMN,
                    data_type=col_info.get('datatype', ''),
                    description=col_info.get('description', ''),
                    aggregatable=col_info.get('aggregatable', False),
                    temporal=col_info.get('temporal', False),
                    business_keywords=col_info.get('business_searchable', [])
                )
                physical_fields.append(field)
                total_physical += 1
            
            # Get XML fields
            xml_info = loader.get_xml_fields(table_name)
            xml_fields = []
            
            if xml_info and 'fields' in xml_info:
                for xml_field in xml_info['fields']:
                    field = DatabaseField(
                        name=xml_field.get('name', ''),
                        table=table_name,
                        field_type=FieldType.XML_FIELD,
                        data_type=xml_field.get('data_type_inferred', 'string'),
                        sql_expression=xml_field.get('sql_expression', ''),
                        xpath=xml_field.get('xpath', ''),
                        aggregatable=xml_field.get('aggregatable', False),
                        business_keywords=xml_field.get('business_searchable', [])
                    )
                    xml_fields.append(field)
                    total_xml += 1
            
            # Create table structure
            table_structures[table_name] = TableStructure(
                table_name=table_name,
                physical_columns=physical_fields,
                xml_fields=xml_fields,
                business_domain=table_info.get('business_domain', 'operational'),
                analyst_relevance=table_info.get('analyst_relevance', 'medium'),
                common_joins=table_info.get('common_joins', [])
            )
        
        return cls(
            table_structures=table_structures,
            join_intelligence=metadata.get('joins', []),
            total_physical_columns=total_physical,
            total_xml_fields=total_xml
        )
    
    def get_table_structure(self, table_name: str) -> Optional[TableStructure]:
        """Get complete structure for a specific table"""
        return self.table_structures.get(table_name)
    
    def find_fields_by_keyword(self, keyword: str) -> List[DatabaseField]:
        """Find all fields (physical and XML) that match business keyword"""
        matching_fields = []
        keyword_lower = keyword.lower()
        
        for table_structure in self.table_structures.values():
            for field in table_structure.all_fields:
                if keyword_lower in [kw.lower() for kw in field.business_keywords]:
                    matching_fields.append(field)
                elif keyword_lower in field.name.lower():
                    matching_fields.append(field)
                elif field.description and keyword_lower in field.description.lower():
                    matching_fields.append(field)
        
        return matching_fields
    
    def get_aggregatable_fields(self, table_name: Optional[str] = None) -> List[DatabaseField]:
        """Get all aggregatable fields, optionally filtered by table"""
        aggregatable = []
        
        tables_to_check = [table_name] if table_name else self.table_structures.keys()
        
        for table in tables_to_check:
            if table and table in self.table_structures:
                for field in self.table_structures[table].all_fields:
                    if field.aggregatable:
                        aggregatable.append(field)
        
        return aggregatable
    
    def get_temporal_fields(self, table_name: Optional[str] = None) -> List[DatabaseField]:
        """Get all temporal fields, optionally filtered by table"""
        temporal = []
        
        tables_to_check = [table_name] if table_name else self.table_structures.keys()
        
        for table in tables_to_check:
            if table and table in self.table_structures:
                for field in self.table_structures[table].all_fields:
                    if field.temporal:
                        temporal.append(field)
        
        return temporal

@dataclass
class ProcessedQuery:
    """Complete processed query result"""
    original_query: AnalystQuery
    intent: IntentResult
    entities: List[BusinessEntity]
    temporal_expressions: List[TemporalExpression] = field(default_factory=list)
    geographic_entities: List[GeographicEntity] = field(default_factory=list)
    numerical_entities: List[NumericalEntity] = field(default_factory=list)
    linguistic_analysis: Optional[LinguisticAnalysis] = None
    relevant_tables: List[str] = field(default_factory=list)
    relevant_fields: List[DatabaseField] = field(default_factory=list)
    business_context: Dict[str, Any] = field(default_factory=dict)
    validation_result: Optional[ValidationResult] = None
    processing_metrics: Optional[ProcessingMetrics] = None
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert ProcessedQuery to dictionary for JSON serialization"""
        return {
            'original_query': self.original_query.to_dict() if self.original_query else None,
            'intent': self.intent.to_dict() if self.intent else None,
            'entities': [e.to_dict() for e in self.entities],
            'temporal_expressions': [t.to_dict() for t in self.temporal_expressions],
            'geographic_entities': [g.to_dict() for g in self.geographic_entities],
            'numerical_entities': [n.to_dict() for n in self.numerical_entities],
            'linguistic_analysis': self.linguistic_analysis.to_dict() if self.linguistic_analysis else None,
            'relevant_tables': self.relevant_tables,
            'relevant_fields': [f.to_dict() for f in self.relevant_fields],
            'business_context': self.business_context,
            'validation_result': self.validation_result.to_dict() if self.validation_result else None,
            'processing_metrics': self.processing_metrics.to_dict() if self.processing_metrics else None
        }

# Continue with remaining classes with similar serialization additions...

@dataclass
class EnhancedQueryResult:
    """Enhanced result for schema searcher integration"""
    structured_query: Dict[str, Any]
    field_mappings: List[DatabaseField]
    join_requirements: List[Dict[str, Any]]
    business_context: Dict[str, Any]
    xml_extractions_needed: List[DatabaseField]
    performance_hints: Dict[str, Any]
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    alternative_interpretations: List[Dict[str, Any]] = field(default_factory=list)
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert EnhancedQueryResult to dictionary for JSON serialization"""
        return {
            'structured_query': self.structured_query,
            'field_mappings': [f.to_dict() for f in self.field_mappings],
            'join_requirements': self.join_requirements,
            'business_context': self.business_context,
            'xml_extractions_needed': [f.to_dict() for f in self.xml_extractions_needed],
            'performance_hints': self.performance_hints,
            'confidence_scores': self.confidence_scores,
            'alternative_interpretations': self.alternative_interpretations
        }

@dataclass
class SQLGenerationContext:
    """Context for SQL generation"""
    selected_fields: List[DatabaseField]
    join_tables: List[str]
    where_conditions: List[str]
    groupby_fields: List[str] = field(default_factory=list)
    having_conditions: List[str] = field(default_factory=list)
    orderby_fields: List[str] = field(default_factory=list)
    limit_clause: Optional[int] = None
    xml_extractions: Dict[str, str] = field(default_factory=dict)
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert SQLGenerationContext to dictionary for JSON serialization"""
        return {
            'selected_fields': [f.to_dict() for f in self.selected_fields],
            'join_tables': self.join_tables,
            'where_conditions': self.where_conditions,
            'groupby_fields': self.groupby_fields,
            'having_conditions': self.having_conditions,
            'orderby_fields': self.orderby_fields,
            'limit_clause': self.limit_clause,
            'xml_extractions': self.xml_extractions
        }

@dataclass
class QueryExecutionPlan:
    """Execution plan for complex queries"""
    main_query: str
    subqueries: List[str] = field(default_factory=list)
    temp_tables: List[str] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)
    estimated_cost: Optional[float] = None
    optimization_hints: List[str] = field(default_factory=list)
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert QueryExecutionPlan to dictionary for JSON serialization"""
        return asdict(self)

@dataclass
class BankingBusinessContext:
    """Banking-specific business context"""
    regulatory_framework: Optional[str] = None
    reporting_period: Optional[str] = None
    business_segment: Optional[str] = None
    risk_category: Optional[str] = None
    compliance_requirements: List[str] = field(default_factory=list)
    audit_trail_needed: bool = False
    data_sensitivity: str = "medium"  # low, medium, high, confidential
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert BankingBusinessContext to dictionary for JSON serialization"""
        return asdict(self)

# ADD ORCHESTRATION-SPECIFIC CLASSES
@dataclass
class ComponentResult:
    """Result from individual component processing"""
    component_name: str
    success: bool
    result: Any
    processing_time: float
    error_message: Optional[str] = None
    confidence_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert ComponentResult to dictionary for JSON serialization"""
        result_data = self.result
        
        # Handle complex result objects that might have to_dict methods
        if hasattr(self.result, 'to_dict'):
            result_data = self.result.to_dict()
        elif isinstance(self.result, (list, tuple)):
            result_data = []
            for item in self.result:
                if hasattr(item, 'to_dict'):
                    result_data.append(item.to_dict())
                else:
                    result_data.append(str(item) if not isinstance(item, (str, int, float, bool, type(None))) else item)
        elif not isinstance(self.result, (str, int, float, bool, dict, list, type(None))):
            result_data = str(self.result)
        
        return {
            'component_name': self.component_name,
            'success': self.success,
            'result': result_data,
            'processing_time': self.processing_time,
            'error_message': self.error_message,
            'confidence_score': self.confidence_score,
            'metadata': self.metadata
        }

@dataclass
class OrchestrationResult:
    """Complete orchestration result"""
    query_id: str
    processed_query: Optional[ProcessedQuery]
    query_semantics: Optional[QuerySemantics]
    component_results: Dict[str, ComponentResult]
    overall_confidence: float
    processing_metrics: ProcessingMetrics
    validation_results: ValidationResult
    recommendations: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    
    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert OrchestrationResult to dictionary for JSON serialization"""
        return {
            'query_id': self.query_id,
            'processed_query': self.processed_query.to_dict() if self.processed_query else None,
            'query_semantics': self.query_semantics.to_dict() if self.query_semantics else None,
            'component_results': {k: v.to_dict() for k, v in self.component_results.items()},
            'overall_confidence': self.overall_confidence,
            'processing_metrics': self.processing_metrics.to_dict(),
            'validation_results': self.validation_results.to_dict(),
            'recommendations': self.recommendations,
            'next_steps': self.next_steps
        }

# GLOBAL SERIALIZATION HELPER FUNCTION
def make_serializable(obj: Any) -> Any:
    """
    Global helper function to make any object JSON serializable
    This can be used as a fallback for objects that don't have to_dict() methods
    """
    if obj is None:
        return None
    
    if hasattr(obj, 'to_dict') and callable(obj.to_dict):
        return obj.to_dict()
    
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    if isinstance(obj, Enum):
        return obj.value
    
    if isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    
    if hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                result[key] = make_serializable(value)
        return result
    
    # Fallback to string representation
    return str(obj)
