"""
Core data models for Schema Searcher Agent

This module defines all data structures, enums, and types used throughout
the schema retrieval system. All models are immutable dataclasses with
comprehensive validation and serialization support.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from enum import Enum, auto
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)

class SearchMethod(Enum):
    """Available search methods for schema retrieval"""
    SEMANTIC = "semantic"
    BM25 = "bm25"
    FUZZY = "fuzzy"
    CHROMA = "chroma"
    FAISS = "faiss"
    NLP = "nlp"

    @classmethod
    def all_methods(cls) -> List[SearchMethod]:
        """Get all available search methods"""
        return list(cls)

    @classmethod
    def default_methods(cls) -> List[SearchMethod]:
        """Get default search methods for typical queries"""
        return [cls.SEMANTIC, cls.BM25, cls.CHROMA, cls.FAISS]

# CRITICAL FIX: Add ColumnType enum (was missing)
class ColumnType(Enum):
    """Column types for enhanced classification"""
    REGULAR = "regular"
    XML = "xml"
    KEY = "key"
    RELATIONAL = "relational"  # Keep your existing values
    COMPUTED = "computed"
    VIRTUAL = "virtual"

# CRITICAL FIX: Add missing PromptType enum (fixes ENTITY_SPECIFIC error)
class PromptType(Enum):
    """Query prompt types for different SQL generation scenarios"""
    SIMPLE_SELECT = "simple_select"
    JOIN_QUERY = "join_query"
    XML_EXTRACTION = "xml_extraction"
    AGGREGATION = "aggregation"
    COMPLEX_FILTER = "complex_filter"
    MULTI_TABLE = "multi_table"
    
    # These were causing the ENTITY_SPECIFIC error - now properly defined
    ENTITY_SPECIFIC = "entity_specific"
    INTELLIGENT_OPTIMIZED = "intelligent_optimized"
    ENHANCED_JOIN = "enhanced_join"
    SCHEMA_AWARE = "schema_aware"

# CRITICAL FIX: Add missing QueryComplexity enum
class QueryComplexity(Enum):
    """Query complexity levels for processing optimization"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class JoinType(Enum):
    """Types of table joins"""
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    FULL = "full"
    CROSS = "cross"

class Priority(Enum):
    """Priority levels for operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DataType(Enum):
    """Common database data types"""
    INT = "int"
    BIGINT = "bigint"
    VARCHAR = "varchar"
    NVARCHAR = "nvarchar"
    TEXT = "text"
    DATETIME = "datetime"
    DATE = "date"
    FLOAT = "float"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    UUID = "uniqueidentifier"
    JSON = "json"
    XML = "xml"
    BINARY = "binary"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, type_str: str) -> DataType:
        """Convert string to DataType enum, fallback to UNKNOWN"""
        try:
            # Handle common variations
            type_str = type_str.lower().strip()
            type_mapping = {
                "integer": cls.INT,
                "string": cls.VARCHAR,
                "str": cls.VARCHAR,
                "bool": cls.BOOLEAN,
                "guid": cls.UUID,
                "uniqueidentifier": cls.UUID,
                "timestamp": cls.DATETIME,
            }
            
            if type_str in type_mapping:
                return type_mapping[type_str]
            
            # Try direct match
            for data_type in cls:
                if data_type.value == type_str:
                    return data_type
            
            return cls.UNKNOWN
        except Exception:
            return cls.UNKNOWN

@dataclass(frozen=True)
class TableReference:
    """Reference to a database table"""
    name: str
    schema: Optional[str] = None
    database: Optional[str] = None
    
    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("Table name cannot be empty")
    
    @property
    def full_name(self) -> str:
        """Get fully qualified table name"""
        parts = []
        if self.database:
            parts.append(self.database)
        if self.schema:
            parts.append(self.schema)
        parts.append(self.name)
        return ".".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class ColumnReference:
    """Reference to a database column"""
    table: str
    column: str
    datatype: DataType = DataType.UNKNOWN
    nullable: bool = True
    primary_key: bool = False
    foreign_key: bool = False
    
    def __post_init__(self):
        if not self.table or not self.table.strip():
            raise ValueError("Table name cannot be empty")
        if not self.column or not self.column.strip():
            raise ValueError("Column name cannot be empty")
    
    @property
    def full_name(self) -> str:
        """Get fully qualified column name"""
        return f"{self.table}.{self.column}"
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['datatype'] = self.datatype.value
        return data

@dataclass(frozen=True)
class XMLFieldReference:
    """Reference to an XML field within a column"""
    table: str
    xml_column: str
    field_name: str
    xpath: str
    sql_expression: str
    datatype: DataType = DataType.UNKNOWN
    
    def __post_init__(self):
        if not all([self.table, self.xml_column, self.field_name]):
            raise ValueError("Table, XML column, and field name are required")
        if not self.xpath.startswith('/'):
            raise ValueError("XPath must start with '/'")
    
    @property
    def full_reference(self) -> str:
        """Get full XML field reference"""
        return f"{self.table}.{self.xml_column}::{self.field_name}"
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['datatype'] = self.datatype.value
        return data

# CRITICAL FIX: Updated RetrievedColumn with all missing methods
@dataclass(frozen=False)
class RetrievedColumn:
    """
    Single column metadata passed into prompt template.
    Used for generating the SCHEMA CONTEXT section.
    Enhanced with intelligent retrieval metadata and FIXED for all compatibility issues.
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
    retrieval_method: SearchMethod = SearchMethod.SEMANTIC
    nullable: bool = True
    primary_key: bool = False
    foreign_key: bool = False
    retrieved_at: datetime = field(default_factory=datetime.now)
    
    # NEW: Intelligence metadata
    entity_relevance: Optional[str] = None  # Which entity this column relates to
    priority_score: int = 0  # Priority score from filtering
    reasoning_applied: bool = False
    
    # CRITICAL FIX: Private field for XML field tracking
    _is_xml_field: bool = field(default=False, init=False)
    
    def __post_init__(self):
        """Handle backward compatibility and field normalization"""
        if not all([self.table.strip(), self.column.strip()]):
            raise ValueError("Table and column names cannot be empty")
        
        # Normalize confidence fields
        if self.confidence is not None and self.confidence_score == 1.0:
            self.confidence_score = self.confidence
        
        # Auto-fix invalid confidence scores
        if not (0.0 <= self.confidence_score <= 1.0):
            self.confidence_score = max(0.0, min(1.0, self.confidence_score))
        
        # Set type based on is_xml flag if not explicitly set
        if self.type is None:
            if self.is_xml or self.xpath or self.xml_column:
                self.type = ColumnType.XML
                self._is_xml_field = True
            else:
                self.type = ColumnType.REGULAR
                self._is_xml_field = False
        else:
            self._is_xml_field = (self.type == ColumnType.XML)
    
    # CRITICAL FIX: Add missing .get() method for dict-like access
    def get(self, key: str, default=None):
        """Get attribute value with default fallback - dict-like access"""
        try:
            return getattr(self, key, default)
        except (AttributeError, TypeError):
            return default
    
    # CRITICAL FIX: Add is_xml_field property with setter
    @property
    def is_xml_field(self) -> bool:
        """Check if this is an XML field"""
        return self._is_xml_field
    
    @is_xml_field.setter
    def is_xml_field(self, value: bool) -> None:
        """Set XML field status"""
        self._is_xml_field = bool(value)
        if value:
            self.type = ColumnType.XML
        elif self.type == ColumnType.XML:
            self.type = ColumnType.REGULAR
    
    def update_confidence(self, new_score: float) -> None:
        """Update confidence score safely"""
        self.confidence_score = max(0.0, min(1.0, new_score))
    
    @property
    def full_name(self) -> str:
        """Get fully qualified column name"""
        return f"{self.table}.{self.column}"
    
    @property
    def parsed_datatype(self) -> DataType:
        """Get parsed datatype enum"""
        return DataType.from_string(self.datatype)
    
    # BACKWARD COMPATIBILITY: Support xml_path access
    @property
    def xml_path(self) -> Optional[str]:
        """Backward compatibility property for xml_path"""
        return self.xpath
    
    @xml_path.setter
    def xml_path(self, value: Optional[str]) -> None:
        """Backward compatibility setter for xml_path"""
        self.xpath = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['type'] = self.type.value if self.type else None
        data['retrieval_method'] = self.retrieval_method.value
        data['retrieved_at'] = self.retrieved_at.isoformat()
        data['is_xml_field'] = self.is_xml_field  # Include computed property
        # Remove private field from dict
        data.pop('_is_xml_field', None)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RetrievedColumn:
        """Create instance from dictionary"""
        # Handle legacy xml_path parameter
        if 'xml_path' in data and 'xpath' not in data:
            data['xpath'] = data.pop('xml_path')
        
        # Remove computed property from data
        data.pop('is_xml_field', None)
        
        # Parse enums
        if 'type' in data and data['type']:
            data['type'] = ColumnType(data['type'])
        if 'retrieval_method' in data:
            data['retrieval_method'] = SearchMethod(data['retrieval_method'])
        
        # Parse datetime
        if 'retrieved_at' in data:
            data['retrieved_at'] = datetime.fromisoformat(data['retrieved_at'])
        
        return cls(**data)
    
    def __hash__(self) -> int:
        """Custom hash for deduplication"""
        return hash((self.table, self.column, self.type))

@dataclass(frozen=True)
class RetrievedJoin:
    """Represents a join relationship between tables"""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    join_type: JoinType
    confidence: int
    verified: bool
    priority: Priority
    comment: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not all([
            self.source_table.strip(),
            self.source_column.strip(),
            self.target_table.strip(),
            self.target_column.strip()
        ]):
            raise ValueError("All join fields must be non-empty")
        
        if not (0 <= self.confidence <= 100):
            raise ValueError("Confidence must be between 0 and 100")
        
        if self.source_table == self.target_table:
            raise ValueError("Source and target tables cannot be the same")
    
    @property
    def source_reference(self) -> str:
        """Get source column reference"""
        return f"{self.source_table}.{self.source_column}"
    
    @property
    def target_reference(self) -> str:
        """Get target column reference"""
        return f"{self.target_table}.{self.target_column}"
    
    @property
    def join_key(self) -> str:
        """Get unique join identifier"""
        return f"{self.source_reference}={self.target_reference}"
    
    @property
    def reverse_join(self) -> RetrievedJoin:
        """Get the reverse of this join"""
        return RetrievedJoin(
            source_table=self.target_table,
            source_column=self.target_column,
            target_table=self.source_table,
            target_column=self.source_column,
            join_type=self.join_type,
            confidence=self.confidence,
            verified=self.verified,
            priority=self.priority,
            comment=f"Reverse of: {self.comment}" if self.comment else None
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['join_type'] = self.join_type.value
        data['priority'] = self.priority.value
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RetrievedJoin:
        """Create instance from dictionary"""
        # Parse enums
        data['join_type'] = JoinType(data.get('join_type', 'inner'))
        data['priority'] = Priority(data.get('priority', 'medium'))
        
        # Parse datetime
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)
    
    def __hash__(self) -> int:
        """Custom hash for deduplication"""
        return hash((self.source_reference, self.target_reference))

@dataclass
class SearchConfig:
    """Configuration for schema search operations"""
    methods: List[SearchMethod] = field(default_factory=lambda: SearchMethod.default_methods())
    top_k: int = 10
    confidence_threshold: float = 0.3
    parallel_execution: bool = True
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    timeout: int = 30  # seconds
    max_retries: int = 3
    include_xml_fields: bool = True
    include_joins: bool = True
    
    def __post_init__(self):
        if not self.methods:
            self.methods = SearchMethod.default_methods()
        
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
    
    @property
    def cache_key_suffix(self) -> str:
        """Generate cache key suffix from config"""
        config_str = f"{sorted([m.value for m in self.methods])}-{self.top_k}-{self.confidence_threshold}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['methods'] = [method.value for method in self.methods]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SearchConfig:
        """Create instance from dictionary"""
        if 'methods' in data:
            data['methods'] = [SearchMethod(method) for method in data['methods']]
        return cls(**data)

@dataclass
class SearchMetrics:
    """Performance and quality metrics for search operations"""
    total_results: int = 0
    execution_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    method_timings: Dict[str, float] = field(default_factory=dict)
    method_result_counts: Dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    errors: List[str] = field(default_factory=list)
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    @property
    def average_method_time(self) -> float:
        """Calculate average method execution time"""
        if not self.method_timings:
            return 0.0
        return sum(self.method_timings.values()) / len(self.method_timings)
    
    def add_error(self, error: str) -> None:
        """Add an error to the metrics"""
        self.error_count += 1
        self.errors.append(error)
        if len(self.errors) > 10:  # Keep only last 10 errors
            self.errors = self.errors[-10:]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SchemaRetrievalResult:
    """Complete result from schema retrieval operation"""
    query: str
    tables: Set[str]
    relational_columns: List[RetrievedColumn]
    xml_fields: List[RetrievedColumn]
    joins: List[RetrievedJoin]
    execution_time: float
    retrieval_methods_used: List[SearchMethod]
    confidence_scores: Dict[str, float]
    metrics: SearchMetrics = field(default_factory=SearchMetrics)
    cached: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.query.strip():
            raise ValueError("Query cannot be empty")
        
        if self.execution_time < 0:
            raise ValueError("Execution time cannot be negative")
    
    @property
    def total_columns(self) -> int:
        """Total number of retrieved columns"""
        return len(self.relational_columns) + len(self.xml_fields)
    
    @property
    def all_columns(self) -> List[RetrievedColumn]:
        """Get all columns (relational + XML)"""
        return self.relational_columns + self.xml_fields
    
    @property
    def unique_tables(self) -> Set[str]:
        """Get all unique table names from results"""
        return self.tables.union({col.table for col in self.all_columns})
    
    @property
    def average_confidence(self) -> float:
        """Calculate average confidence score"""
        all_scores = [col.confidence_score for col in self.all_columns]
        return sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    @property
    def max_confidence(self) -> float:
        """Get maximum confidence score"""
        all_scores = [col.confidence_score for col in self.all_columns]
        return max(all_scores) if all_scores else 0.0
    
    def get_columns_by_table(self, table_name: str) -> List[RetrievedColumn]:
        """Get all columns for a specific table"""
        return [col for col in self.all_columns if col.table == table_name]
    
    def get_high_confidence_columns(self, threshold: float = 0.7) -> List[RetrievedColumn]:
        """Get columns with confidence above threshold"""
        return [col for col in self.all_columns if col.confidence_score >= threshold]
    
    def get_joins_for_tables(self, table_names: Set[str]) -> List[RetrievedJoin]:
        """Get joins that connect the specified tables"""
        return [
            join for join in self.joins
            if join.source_table in table_names and join.target_table in table_names
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "query": self.query,
            "tables": list(self.tables),
            "relational_columns": [col.to_dict() for col in self.relational_columns],
            "xml_fields": [field.to_dict() for field in self.xml_fields],
            "joins": [join.to_dict() for join in self.joins],
            "execution_time": self.execution_time,
            "retrieval_methods_used": [method.value for method in self.retrieval_methods_used],
            "confidence_scores": self.confidence_scores,
            "metrics": self.metrics.to_dict(),
            "cached": self.cached,
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total_columns": self.total_columns,
                "unique_tables_count": len(self.unique_tables),
                "joins_count": len(self.joins),
                "average_confidence": self.average_confidence,
                "max_confidence": self.max_confidence
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SchemaRetrievalResult:
        """Create instance from dictionary"""
        # Parse complex fields
        data['tables'] = set(data['tables'])
        data['relational_columns'] = [RetrievedColumn.from_dict(col) for col in data['relational_columns']]
        data['xml_fields'] = [RetrievedColumn.from_dict(field) for field in data['xml_fields']]
        data['joins'] = [RetrievedJoin.from_dict(join) for join in data['joins']]
        data['retrieval_methods_used'] = [SearchMethod(method) for method in data['retrieval_methods_used']]
        
        # Parse datetime
        if 'timestamp' in data:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Remove summary if present (calculated property)
        data.pop('summary', None)
        
        return cls(**data)

@dataclass
class EngineStatus:
    """Status information for a search engine"""
    name: str
    method: SearchMethod
    initialized: bool
    available: bool
    last_error: Optional[str] = None
    total_searches: int = 0
    successful_searches: int = 0
    average_response_time: float = 0.0
    last_used: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return self.successful_searches / self.total_searches if self.total_searches > 0 else 0.0
    
    @property
    def status(self) -> str:
        """Get overall status string"""
        if not self.available:
            return "unavailable"
        if not self.initialized:
            return "not_initialized"
        if self.last_error:
            return "error"
        return "healthy"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['method'] = self.method.value
        data['status'] = self.status
        data['success_rate'] = self.success_rate
        if self.last_used:
            data['last_used'] = self.last_used.isoformat()
        return data

# CRITICAL FIX: Helper function for safe creation (FIXED INDENTATION!)
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

# Type aliases for better code readability
ColumnDict = Dict[str, Any]
JoinDict = Dict[str, Any]
SchemaDict = Dict[str, Any]
ConfigDict = Dict[str, Any]
MetricsDict = Dict[str, Any]

# Validation constants
MAX_QUERY_LENGTH = 1000
MAX_TABLE_NAME_LENGTH = 128
MAX_COLUMN_NAME_LENGTH = 128
MAX_DESCRIPTION_LENGTH = 2000
MIN_CONFIDENCE_SCORE = 0.0
MAX_CONFIDENCE_SCORE = 1.0

# Default configurations
DEFAULT_SEARCH_CONFIG = SearchConfig()
DEFAULT_TOP_K = 10
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_TIMEOUT = 30

# UPDATED: Export list including the new enums
__all__ = [
    # Enums
    'SearchMethod',
    'ColumnType', 
    'PromptType',        # ADDED: This was missing and causing ENTITY_SPECIFIC error
    'QueryComplexity',   # ADDED: This was missing too
    'JoinType',
    'Priority',
    'DataType',
    
    # Core data models
    'TableReference',
    'ColumnReference', 
    'XMLFieldReference',
    'RetrievedColumn',
    'RetrievedJoin',
    
    # Configuration
    'SearchConfig',
    
    # Results and metrics
    'SchemaRetrievalResult',
    'SearchMetrics',
    'EngineStatus',
    
    # Helper functions
    'create_retrieved_column_safe',
    
    # Type aliases
    'ColumnDict',
    'JoinDict', 
    'SchemaDict',
    'ConfigDict',
    'MetricsDict',
    
    # Constants
    'DEFAULT_SEARCH_CONFIG',
    'DEFAULT_TOP_K',
    'DEFAULT_CONFIDENCE_THRESHOLD',
    'DEFAULT_TIMEOUT'
]
