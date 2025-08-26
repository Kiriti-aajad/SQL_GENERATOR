"""
Core data models for the prompt builder system.
Defines all data structures used throughout the prompt building pipeline.
Enhanced with FilteredIntelligentRetrievalAgent integration.
FIXED: Added missing enum values to resolve ENTITY_SPECIFIC error.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set
from enum import Enum
from datetime import datetime

class PromptType(Enum):
    """Types of SQL prompts based on query complexity and requirements"""
    SIMPLE_SELECT = "simple_select"
    JOIN_QUERY = "join_query"
    XML_EXTRACTION = "xml_extraction"
    AGGREGATION = "aggregation"
    COMPLEX_FILTER = "complex_filter"
    MULTI_TABLE = "multi_table"
    
    # CRITICAL FIX: Add missing enum values that were causing ENTITY_SPECIFIC error
    ENTITY_SPECIFIC = "entity_specific"
    INTELLIGENT_OPTIMIZED = "intelligent_optimized"
    ENHANCED_JOIN = "enhanced_join"
    SCHEMA_AWARE = "schema_aware"

class QueryComplexity(Enum):
    """Query complexity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# CRITICAL FIX: Add ColumnType enum (was missing)
class ColumnType(Enum):
    """Column types for enhanced classification"""
    REGULAR = "regular"
    XML = "xml"
    KEY = "key"
    RELATIONAL = "relational"  # Add additional types for compatibility
    COMPUTED = "computed"
    VIRTUAL = "virtual"

@dataclass
class QueryIntent:
    """
    Analysis result of user's natural language query.
    Determines prompt building strategy.
    Enhanced with schema intelligence integration.
    """
    query_type: PromptType
    complexity: QueryComplexity
    involves_joins: bool = False
    involves_xml: bool = False
    involves_aggregation: bool = False
    target_tables: Set[str] = field(default_factory=set)
    keywords: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    # NEW: Schema intelligence integration
    schema_entities_detected: List[str] = field(default_factory=list)
    schema_confidence: float = 0.0
    recommended_tables: List[str] = field(default_factory=list)
    filtering_applied: bool = False
    entity_priorities: Dict[str, int] = field(default_factory=dict)
    
    def requires_relationship_context(self) -> bool:
        """Check if query needs detailed relationship information"""
        return self.involves_joins or len(self.target_tables) > 1
    
    def requires_xml_context(self) -> bool:
        """Check if query needs XML-specific context"""
        return self.involves_xml or self.query_type == PromptType.XML_EXTRACTION
    
    def has_intelligent_schema_support(self) -> bool:
        """Check if intelligent schema retrieval was successfully applied"""
        return self.filtering_applied and self.schema_confidence > 0.0

@dataclass
class SchemaContext:
    """
    Processed schema information ready for prompt insertion.
    Maintains exact table/column names from aggregator.
    """
    # Core schema information
    tables: Dict[str, List[str]] = field(default_factory=dict)  # table -> columns
    column_details: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    
    # Specialized contexts
    xml_mappings: List[Dict[str, str]] = field(default_factory=list)
    primary_keys: Dict[str, str] = field(default_factory=dict)  # table -> pk_column
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)
    
    # Context metadata
    total_columns: int = 0
    total_tables: int = 0
    has_xml_fields: bool = False
    confidence_range: tuple = (0.0, 1.0)
    
    def get_table_names(self) -> List[str]:
        """Get list of all table names (exact names preserved)"""
        return list(self.tables.keys())
    
    def get_column_count(self) -> int:
        """Get total number of columns across all tables"""
        return sum(len(columns) for columns in self.tables.values())
    
    def has_relationships(self) -> bool:
        """Check if schema context includes relationship information"""
        return len(self.relationships) > 0 or len(self.foreign_keys) > 0

@dataclass
class IntelligentSchemaContext:
    """
    Enhanced schema context that integrates FilteredIntelligentRetrievalAgent results
    Supports the 60-80% table reduction achieved in testing
    """
    # Original schema context (inherit from SchemaContext)
    base_context: SchemaContext
    
    # Intelligence metadata
    reasoning_applied: bool = False
    original_table_count: int = 0
    filtered_table_count: int = 0
    confidence_score: float = 0.0
    table_reduction_rate: float = 0.0
    
    # Entity detection results (from your successful tests)
    detected_entities: List[str] = field(default_factory=list)  # ['director'], ['contact'], etc.
    entity_priorities: Dict[str, int] = field(default_factory=dict)  # entity -> priority score
    
    # Reasoning metadata
    reasoning_iterations: int = 0
    convergence_achieved: bool = False
    processing_duration: float = 0.0
    
    # Quality metrics (from your test results)
    completeness_score: float = 0.0  # 40-70% range in your tests
    consistency_score: float = 0.0   # 100% in your tests
    coverage_score: float = 0.0      # 70-80% in your tests
    
    # Table priority scores (from filtering results)
    table_priority_scores: Dict[str, int] = field(default_factory=dict)
    
    def get_reduction_summary(self) -> str:
        """Get human-readable summary of table reduction"""
        if self.original_table_count > 0:
            reduction_pct = (self.original_table_count - self.filtered_table_count) / self.original_table_count * 100
            return f"Reduced from {self.original_table_count} to {self.filtered_table_count} tables ({reduction_pct:.1f}% reduction)"
        return "No reduction applied"
    
    def get_top_priority_table(self) -> Optional[str]:
        """Get the highest priority table from filtering"""
        if self.table_priority_scores:
            return max(self.table_priority_scores.items(), key=lambda x: x[1])[0]
        return None
    
    def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get summary of intelligence applied"""
        return {
            'reasoning_applied': self.reasoning_applied,
            'entities_detected': self.detected_entities,
            'confidence_score': self.confidence_score,
            'table_reduction_rate': self.table_reduction_rate,
            'top_priority_table': self.get_top_priority_table(),
            'quality_scores': {
                'completeness': self.completeness_score,
                'consistency': self.consistency_score,
                'coverage': self.coverage_score
            }
        }

@dataclass
class SchemaRetrievalResult:
    """
    Result from FilteredIntelligentRetrievalAgent integration
    Bridges schema searcher output with prompt builder input
    """
    # Raw intelligent retrieval results
    raw_results: Dict[str, Any]
    
    # Processed for prompt building
    schema_context: IntelligentSchemaContext
    query_intent: QueryIntent
    
    # Integration metadata
    retrieval_successful: bool = True
    error_message: Optional[str] = None
    fallback_used: bool = False
    
    # Performance metrics (based on your test results: 12-40s processing time)
    total_processing_time: float = 0.0
    schema_retrieval_time: float = 0.0
    context_building_time: float = 0.0
    
    # Filtering results (your 60-80% reduction success)
    filtering_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def was_successful(self) -> bool:
        """Check if retrieval was successful"""
        return self.retrieval_successful and not self.fallback_used
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance metrics summary"""
        return {
            'total_time': self.total_processing_time,
            'schema_retrieval_time': self.schema_retrieval_time,
            'context_building_time': self.context_building_time,
            'efficiency_ratio': self.schema_retrieval_time / max(self.total_processing_time, 0.01)
        }

@dataclass
class PromptOptions:
    """
    Configuration options for prompt generation.
    Allows customization without hardcoding.
    Enhanced with intelligent schema options.
    """
    # Content control
    max_context_length: int = 2000
    include_examples: bool = True
    include_descriptions: bool = True
    include_data_types: bool = True
    
    # Filtering options
    confidence_threshold: float = 0.7
    max_tables: int = 10
    max_columns_per_table: int = 15
    
    # Template preferences
    template_type: Optional[PromptType] = None
    prioritize_relationships: bool = True
    emphasize_xml_handling: bool = False
    
    # LLM-specific optimizations
    target_llm: str = "gpt"  # "gpt", "claude", "gemini"
    
    # NEW: Intelligent schema options
    enable_intelligent_filtering: bool = True
    entity_detection_enabled: bool = True
    table_reduction_enabled: bool = True
    reasoning_iterations: int = 3
    schema_confidence_threshold: float = 0.6
    
    # CRITICAL FIX: Add missing fields for compatibility
    optimization_level: str = "intelligent"
    preserve_join_information: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert options to dictionary for template rendering"""
        return {
            "max_context_length": self.max_context_length,
            "include_examples": self.include_examples,
            "include_descriptions": self.include_descriptions,
            "include_data_types": self.include_data_types,
            "confidence_threshold": self.confidence_threshold,
            "max_tables": self.max_tables,
            "max_columns_per_table": self.max_columns_per_table,
            "prioritize_relationships": self.prioritize_relationships,
            "emphasize_xml_handling": self.emphasize_xml_handling,
            "target_llm": self.target_llm,
            # Intelligent schema options
            "enable_intelligent_filtering": self.enable_intelligent_filtering,
            "entity_detection_enabled": self.entity_detection_enabled,
            "table_reduction_enabled": self.table_reduction_enabled,
            "reasoning_iterations": self.reasoning_iterations,
            "schema_confidence_threshold": self.schema_confidence_threshold,
            "optimization_level": self.optimization_level,
            "preserve_join_information": self.preserve_join_information
        }

@dataclass
class TemplateConfig:
    """
    Configuration for template selection and rendering.
    Loaded from YAML configuration files.
    Enhanced with intelligent schema support.
    """
    template_id: str
    template_type: PromptType
    base_template: str
    specializations: List[str] = field(default_factory=list)
    triggers: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    
    # NEW: Intelligence-aware template selection
    supports_intelligent_schema: bool = True
    entity_specific_templates: Dict[str, str] = field(default_factory=dict)  # entity -> template_path
    
    def matches_intent(self, intent: QueryIntent) -> bool:
        """Check if this template configuration matches the query intent"""
        if self.template_type != intent.query_type:
            return False
        
        # Check additional triggers
        if "involves_joins" in self.triggers:
            if self.triggers["involves_joins"] != intent.involves_joins:
                return False
                
        if "involves_xml" in self.triggers:
            if self.triggers["involves_xml"] != intent.involves_xml:
                return False
        
        # NEW: Check intelligent schema compatibility
        if intent.filtering_applied and not self.supports_intelligent_schema:
            return False
                
        return True
    
    def get_entity_template(self, entity: str) -> Optional[str]:
        """Get entity-specific template if available"""
        return self.entity_specific_templates.get(entity)

@dataclass
class StructuredPrompt:
    """
    Final structured prompt ready for LLM consumption.
    Contains all components assembled from templates and context.
    Enhanced with intelligent schema metadata.
    """
    # Core prompt components
    system_context: str
    schema_context: str
    user_query: str
    instructions: str
    
    # Optional components
    examples: Optional[str] = None
    validation_rules: Optional[str] = None
    
    # Metadata
    prompt_type: PromptType = PromptType.SIMPLE_SELECT
    generated_at: datetime = field(default_factory=datetime.now)
    schema_tables_used: List[str] = field(default_factory=list)
    total_length: int = 0
    
    # Assembly metadata
    template_id: str = ""
    specializations_applied: List[str] = field(default_factory=list)
    
    # NEW: Intelligence integration metadata
    intelligent_schema_used: bool = False
    schema_confidence: float = 0.0
    table_reduction_applied: bool = False
    original_vs_filtered_tables: tuple = (0, 0)  # (original_count, filtered_count)
    reasoning_metadata: Dict[str, Any] = field(default_factory=dict)
    detected_entities: List[str] = field(default_factory=list)
    entity_priorities: Dict[str, int] = field(default_factory=dict)
    
    def get_full_prompt(self) -> str:
        """
        Assemble all components into a single prompt string.
        OPTIMIZED for orchestrator integration with intelligent schema context.
        """
        components = []
        
        # Add system context if available
        if self.system_context and self.system_context.strip():
            components.append("=== SYSTEM CONTEXT ===")
            components.append(self.system_context.strip())
            components.append("")
        
        # Add intelligent schema context if available
        if self.schema_context and self.schema_context.strip():
            if self.intelligent_schema_used:
                components.append("=== INTELLIGENT SCHEMA CONTEXT ===")
                # Add intelligence metadata
                if self.table_reduction_applied:
                    orig, filtered = self.original_vs_filtered_tables
                    reduction_pct = ((orig - filtered) / max(orig, 1)) * 100
                    components.append(f"# Schema Intelligence Applied: {orig} → {filtered} tables ({reduction_pct:.1f}% reduction)")
                if self.detected_entities:
                    components.append(f"# Detected Entities: {', '.join(self.detected_entities)}")
                if self.schema_confidence > 0:
                    components.append(f"# Schema Confidence: {self.schema_confidence:.1%}")
                components.append("")
            else:
                components.append("=== SCHEMA CONTEXT ===")
            
            components.append(self.schema_context.strip())
            components.append("")
        
        # Add instructions if available
        if self.instructions and self.instructions.strip():
            components.append("=== INSTRUCTIONS ===")
            components.append(self.instructions.strip())
            components.append("")
        
        # Add examples if available
        if self.examples and self.examples.strip():
            components.append("=== EXAMPLES ===")
            components.append(self.examples.strip())
            components.append("")
        
        # Add validation rules if available
        if self.validation_rules and self.validation_rules.strip():
            components.append("=== VALIDATION RULES ===")
            components.append(self.validation_rules.strip())
            components.append("")
        
        # Add user query at the end
        components.append("=== USER QUERY ===")
        components.append(self.user_query.strip() if self.user_query else "No query provided")
        
        full_prompt = "\n".join(components)
        
        # Update length metadata
        self.total_length = len(full_prompt)
        
        return full_prompt
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get prompt generation metadata for logging/debugging"""
        metadata = {
            "prompt_type": self.prompt_type.value,
            "generated_at": self.generated_at.isoformat(),
            "schema_tables_used": self.schema_tables_used,
            "total_length": self.total_length,
            "template_id": self.template_id,
            "specializations_applied": self.specializations_applied,
            "has_examples": self.examples is not None,
            "has_validation_rules": self.validation_rules is not None
        }
        
        # Add intelligent schema metadata
        if self.intelligent_schema_used:
            metadata.update({
                "intelligent_schema_used": self.intelligent_schema_used,
                "schema_confidence": self.schema_confidence,
                "table_reduction_applied": self.table_reduction_applied,
                "original_vs_filtered_tables": self.original_vs_filtered_tables,
                "detected_entities": self.detected_entities,
                "entity_priorities": self.entity_priorities,
                "reasoning_metadata": self.reasoning_metadata
            })
        
        return metadata
    
    def validate_name_preservation(self, original_table_names: Set[str]) -> bool:
        """
        Validate that all original table names are preserved exactly in the prompt.
        Critical for maintaining database schema accuracy.
        """
        full_prompt = self.get_full_prompt()
        
        for table_name in original_table_names:
            if table_name not in full_prompt:
                return False
        
        return True
    
    def get_intelligence_summary(self) -> Optional[str]:
        """Get human-readable summary of applied intelligence"""
        if not self.intelligent_schema_used:
            return None
        
        summary_parts = []
        
        if self.table_reduction_applied:
            orig, filtered = self.original_vs_filtered_tables
            reduction_pct = ((orig - filtered) / max(orig, 1)) * 100
            summary_parts.append(f"Table reduction: {orig} → {filtered} ({reduction_pct:.1f}%)")
        
        if self.detected_entities:
            summary_parts.append(f"Entities detected: {', '.join(self.detected_entities)}")
        
        if self.schema_confidence > 0:
            summary_parts.append(f"Confidence: {self.schema_confidence:.1%}")
        
        return " | ".join(summary_parts) if summary_parts else "Intelligence applied"

@dataclass
class ContextSection:
    """
    Individual section of schema context (tables, columns, relationships, etc.)
    Used for modular prompt assembly.
    Enhanced with intelligence metadata.
    """
    section_type: str  # "tables", "columns", "relationships", "xml_mappings", "intelligence"
    content: str
    priority: int = 0
    length: int = 0
    table_names_used: Set[str] = field(default_factory=set)
    
    # NEW: Intelligence-specific metadata
    entity_related: bool = False
    confidence_score: float = 0.0
    priority_score: int = 0
    
    def __post_init__(self):
        """Calculate length after initialization"""
        self.length = len(self.content)

# CRITICAL FIX: Updated RetrievedColumn with xml_path compatibility
@dataclass
class RetrievedColumn:
    """
    Single column metadata passed into prompt template.
    Used for generating the SCHEMA CONTEXT section.
    Enhanced with intelligent retrieval metadata and FIXED xml_path compatibility.
    """
    table: str
    column: str
    datatype: str = "unknown"
    description: str = ""
    confidence_score: float = 1.0
    type: Optional[ColumnType] = None
    
    # XML-related fields - FIXED for backward compatibility
    xml_column: Optional[str] = None
    xpath: Optional[str] = None  # New standard field (replaces xml_path)
    sql_expression: Optional[str] = None
    
    # Legacy compatibility fields
    source: Optional[str] = None
    confidence: Optional[float] = None
    is_xml: bool = False
    
    # NEW: Intelligence metadata
    entity_relevance: Optional[str] = None  # Which entity this column relates to
    priority_score: int = 0  # Priority score from filtering
    reasoning_applied: bool = False
    
    # BACKWARD COMPATIBILITY for xml_path - this was causing the error!
    def __post_init__(self):
        """Handle backward compatibility and field normalization"""
        # Normalize confidence fields
        if self.confidence is not None and self.confidence_score == 1.0:
            self.confidence_score = self.confidence
        
        # Set type based on is_xml flag if not explicitly set
        if self.type is None:
            if self.is_xml or self.xpath or self.xml_column:
                self.type = ColumnType.XML
            else:
                self.type = ColumnType.REGULAR

# CRITICAL FIX: Helper function for safe creation
def create_retrieved_column_safe(**kwargs):
    """
    Create RetrievedColumn safely handling xml_path parameter.
    This prevents the xml_path error that was breaking your pipeline!
    """
    # Handle the xml_path -> xpath conversion
    if 'xml_path' in kwargs:
        kwargs['xpath'] = kwargs.pop('xml_path')
    
    # Handle other legacy parameter mappings
    if 'confidence' in kwargs and 'confidence_score' not in kwargs:
        kwargs['confidence_score'] = kwargs['confidence']
    
    return RetrievedColumn(**kwargs)

# Helper classes for intelligent schema integration

@dataclass
class EntityDetectionResult:
    """Result of entity detection from intelligent schema retrieval"""
    detected_entities: List[str]
    entity_confidence: Dict[str, float]
    entity_keywords: Dict[str, List[str]]
    query_complexity: QueryComplexity

@dataclass
class TableFilteringResult:
    """Result of table filtering from intelligent schema retrieval"""
    original_tables: List[str]
    filtered_tables: List[str]
    table_scores: Dict[str, int]
    reduction_rate: float
    filtering_rationale: Dict[str, str]

# Factory functions for easy integration

def create_intelligent_schema_context(
    raw_retrieval_results: Dict[str, Any]
) -> IntelligentSchemaContext:
    """
    Factory function to create IntelligentSchemaContext from FilteredIntelligentRetrievalAgent results
    """
    # Extract base schema context
    tables = {}
    for table, columns in raw_retrieval_results.get('columns_by_table', {}).items():
        tables[table] = [col.get('column', '') for col in columns]
    
    base_context = SchemaContext(
        tables=tables,
        column_details=list(raw_retrieval_results.get('columns_by_table', {}).values()),
        relationships=[str(join) for join in raw_retrieval_results.get('joins', [])],
        total_tables=len(raw_retrieval_results.get('tables', [])),
        total_columns=raw_retrieval_results.get('total_columns', 0)
    )
    
    # Extract intelligence metadata
    reasoning_results = raw_retrieval_results.get('reasoning_results', {})
    filtering_metadata = raw_retrieval_results.get('filtering_metadata', {})
    quality_metrics = raw_retrieval_results.get('quality_metrics', {})
    
    return IntelligentSchemaContext(
        base_context=base_context,
        reasoning_applied=raw_retrieval_results.get('reasoning_metadata', {}).get('reasoning_applied', False),
        original_table_count=filtering_metadata.get('original_table_count', 0),
        filtered_table_count=filtering_metadata.get('filtered_table_count', 0),
        confidence_score=reasoning_results.get('final_confidence', 0.0),
        table_reduction_rate=filtering_metadata.get('original_table_count', 0) - filtering_metadata.get('filtered_table_count', 0),
        reasoning_iterations=reasoning_results.get('iterations_completed', 0),
        convergence_achieved=reasoning_results.get('convergence_achieved', False),
        completeness_score=quality_metrics.get('completeness_score', 0.0),
        consistency_score=quality_metrics.get('consistency_score', 0.0),
        coverage_score=quality_metrics.get('coverage_score', 0.0),
        table_priority_scores=raw_retrieval_results.get('table_priority_scores', {})
    )

def create_schema_retrieval_result(
    raw_results: Dict[str, Any],
    processing_times: Dict[str, float]
) -> SchemaRetrievalResult:
    """
    Factory function to create SchemaRetrievalResult from intelligent retrieval
    """
    intelligent_context = create_intelligent_schema_context(raw_results)
    
    # Create query intent (simplified for now)
    query_intent = QueryIntent(
        query_type=PromptType.SIMPLE_SELECT,  # Will be determined by query analysis
        complexity=QueryComplexity.MEDIUM,
        filtering_applied=intelligent_context.reasoning_applied,
        schema_confidence=intelligent_context.confidence_score
    )
    
    return SchemaRetrievalResult(
        raw_results=raw_results,
        schema_context=intelligent_context,
        query_intent=query_intent,
        retrieval_successful=True,
        total_processing_time=processing_times.get('total', 0.0),
        schema_retrieval_time=processing_times.get('retrieval', 0.0),
        context_building_time=processing_times.get('context_building', 0.0),
        filtering_metadata=raw_results.get('filtering_metadata', {})
    )
