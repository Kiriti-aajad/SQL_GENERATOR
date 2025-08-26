r"""
SearchOrchestrator - ENHANCED: Schema-Aware + Smart Convergence + Safe Path Handling

CRITICAL ENHANCEMENTS:
- Schema-aware validation with your exact data paths
- Intelligent convergence detection (dynamic thresholds)
- Query complexity-based engine selection
- Multi-criteria result quality scoring
- Enhanced AI enhancement with schema constraints
- Safe path handling for all data files
- XML integration support

SAFE PATH HANDLING: Handles all your data file paths with proper fallbacks
- E:/Github/sql-ai-agent/data/metadata/schema.json
- E:/Github/sql-ai-agent/data/metadata/xml_schema.json  
- E:/Github/sql-ai-agent/data/metadata/tables.json
- E:/Github/sql-ai-agent/data/metadata/joins_verified.json

Version: 2.5.0 - COMPLETE ACCURACY ENHANCEMENT
Date: 2025-08-21
"""

import asyncio
import logging
import time
import os
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import re
import traceback

# FIXED: Import AsyncClientManager class, not instance
try:
    from agent.sql_generator.async_client_manager import AsyncClientManager, get_client_context
    ASYNC_CLIENT_MANAGER_AVAILABLE = True
except ImportError:
    ASYNC_CLIENT_MANAGER_AVAILABLE = False

# SAFE XML IMPORTS - handles if XML components are not available
try:
    from agent.schema_searcher.managers.xml_schema_manager import XMLSchemaManager
    XML_SCHEMA_MANAGER_AVAILABLE = True
except ImportError:
    XML_SCHEMA_MANAGER_AVAILABLE = False
    XMLSchemaManager = None

logger = logging.getLogger(__name__)

class SearchError(Exception):
    """Base exception for search errors"""
    pass

class SearchConfigurationError(SearchError):
    """Configuration-related errors that should fail fast"""
    pass

class SearchExecutionError(SearchError):
    """Execution errors that may allow partial recovery"""
    pass

class SearchMethod(Enum):
    """Search method enumeration"""
    SEMANTIC = "semantic"
    FUZZY = "fuzzy"
    BM25 = "bm25"
    FAISS = "faiss"
    CHROMA = "chroma"
    NLP = "nlp"

class QueryComplexity(Enum):
    """Query complexity levels for intelligent engine selection"""
    SIMPLE = "simple"          # Basic table/column lookup
    MODERATE = "moderate"      # Multi-table queries
    COMPLEX = "complex"        # Complex joins and relationships
    ANALYTICAL = "analytical"  # Advanced analysis queries

@dataclass
class SearchResult:
    """Enhanced search result with accuracy tracking"""
    results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_time: float = 0.0
    iterations_performed: int = 0
    convergence_achieved: bool = False
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    async_client_stats: Dict[str, Any] = field(default_factory=dict)
    ai_enhancement_used: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # NEW: Enhanced accuracy tracking
    schema_compliance_score: float = 0.0
    query_complexity: Optional[QueryComplexity] = None
    engines_selected_intelligently: bool = False
    xml_fields_found: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert SearchResult to dict for serialization compatibility"""
        return {
            'results': self.results,
            'metadata': self.metadata,
            'total_time': self.total_time,
            'success': len(self.results) > 0,
            'iterations_performed': self.iterations_performed,
            'convergence_achieved': self.convergence_achieved,
            'ai_enhancement_used': self.ai_enhancement_used,
            'errors': self.errors,
            'warnings': self.warnings,
            'schema_compliance_score': self.schema_compliance_score,
            'query_complexity': self.query_complexity.value if self.query_complexity else None,
            'engines_selected_intelligently': self.engines_selected_intelligently,
            'xml_fields_found': self.xml_fields_found
        }

@dataclass
class SearchStrategy:
    """Search strategy configuration"""
    methods: List[SearchMethod]
    weights: Dict[SearchMethod, float]
    max_results_per_method: int = 10
    enable_ai_enhancement: bool = True
    ai_enhancement_threshold: float = 0.7

class SafeDataPathManager:
    """SAFE PATH MANAGER: Handles all your data file paths with proper fallbacks"""
    
    def __init__(self, base_path: str = None): # pyright: ignore[reportArgumentType]
        self.logger = logging.getLogger(f"{__name__}.SafeDataPathManager")
        
        # Base path handling with multiple fallbacks
        if base_path:
            self.base_path = Path(base_path)
        else:
            # Try multiple base path options
            possible_bases = [
                Path(r"E:\Github\sql-ai-agent"),  # Your exact path
                Path(r"E:\Github\sqlAgentAI"),    # Alternative
                Path.cwd(),                        # Current directory
                Path("."),                         # Relative
                Path(__file__).parent.parent.parent if '__file__' in globals() else Path(".")
            ]
            
            self.base_path = None
            for base in possible_bases:
                if base.exists():
                    self.base_path = base
                    break
            
            if not self.base_path:
                self.base_path = Path(".")
                self.logger.warning("Could not find base path, using current directory")
        
        self.logger.info(f"SafeDataPathManager initialized with base: {self.base_path}")
        
        # Your exact data file paths with fallbacks
        self.data_paths = {
            'schema': [
                self.base_path / 'data' / 'metadata' / 'schema.json',
                self.base_path / 'data' / 'schema.json',
                self.base_path / 'schema.json',
                Path('data/metadata/schema.json'),
                Path('schema.json')
            ],
            'xml_schema': [
                self.base_path / 'data' / 'metadata' / 'xml_schema.json',
                self.base_path / 'data' / 'xml_schema.json',
                Path('data/metadata/xml_schema.json'),
                Path('xml_schema.json')
            ],
            'tables': [
                self.base_path / 'data' / 'metadata' / 'tables.json',
                self.base_path / 'data' / 'tables.json',
                Path('data/metadata/tables.json'),
                Path('tables.json')
            ],
            'joins': [
                self.base_path / 'data' / 'metadata' / 'joins_verified.json',
                self.base_path / 'data' / 'joins_verified.json',
                Path('data/metadata/joins_verified.json'),
                Path('joins_verified.json')
            ]
        }
    
    def get_safe_path(self, file_type: str) -> Optional[Path]:
        """Get safe path for file type with fallback handling"""
        if file_type not in self.data_paths:
            self.logger.warning(f"Unknown file type: {file_type}")
            return None
        
        for path in self.data_paths[file_type]:
            try:
                if path.exists() and path.is_file():
                    self.logger.debug(f"Found {file_type} at: {path}")
                    return path
            except Exception as e:
                self.logger.debug(f"Path check failed for {path}: {e}")
                continue
        
        self.logger.warning(f"No valid path found for {file_type}")
        return None
    
    def load_json_safe(self, file_type: str) -> Optional[Dict[str, Any]]:
        """Safely load JSON data with comprehensive error handling"""
        path = self.get_safe_path(file_type)
        if not path:
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.logger.info(f"Successfully loaded {file_type} from: {path}")
                return data
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in {path}: {e}")
            return None
        except FileNotFoundError:
            self.logger.warning(f"File not found: {path}")
            return None
        except PermissionError:
            self.logger.error(f"Permission denied: {path}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error loading {path}: {e}")
            return None

class EnhancedSchemaContextManager:
    """ENHANCED SCHEMA MANAGER: Uses your exact data files with XML integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.EnhancedSchemaContextManager")
        self.path_manager = SafeDataPathManager()
        self._schema_cache = None
        self._cache_timestamp = 0
        self._cache_ttl = 300  # 5 minutes
        
        # XML Schema Manager integration
        self.xml_manager = None
        self._init_xml_manager()
    
    def _init_xml_manager(self):
        """Initialize XML Schema Manager safely"""
        if not XML_SCHEMA_MANAGER_AVAILABLE:
            self.logger.info("XML Schema Manager not available")
            return
        
        try:
            xml_schema_path = self.path_manager.get_safe_path('xml_schema')
            if xml_schema_path:
                self.xml_manager = XMLSchemaManager(str(xml_schema_path)) # pyright: ignore[reportOptionalCall]
                self.logger.info("XML Schema Manager initialized successfully")
            else:
                self.logger.warning("XML schema file not found")
        except Exception as e:
            self.logger.warning(f"XML Schema Manager initialization failed: {e}")
    
    def get_schema_context(self) -> Dict[str, Any]:
        """Get comprehensive schema context from your data files"""
        current_time = time.time()
        
        # Return cached version if available and fresh
        if (self._schema_cache and 
            (current_time - self._cache_timestamp) < self._cache_ttl):
            return self._schema_cache
        
        try:
            schema_context = self._load_comprehensive_schema()
            if schema_context and schema_context.get('available', False):
                self._schema_cache = schema_context
                self._cache_timestamp = current_time
                self.logger.info(f"Schema context loaded: {len(schema_context.get('all_tables', []))} tables")
                return schema_context
        except Exception as e:
            self.logger.error(f"Schema loading failed: {e}")
        
        # Safe fallback - doesn't break existing functionality
        return {
            'available': False,
            'all_tables': set(),
            'all_columns': set(),
            'tables': {},
            'xml_fields': {},
            'joins': [],
            'error': 'Schema not available - using fallback mode'
        }
    
    def _load_comprehensive_schema(self) -> Dict[str, Any]:
        """Load comprehensive schema from all your data files"""
        schema_context = {
            'available': False,
            'tables': {},
            'all_tables': set(),
            'all_columns': set(),
            'xml_fields': {},
            'joins': [],
            'data_sources': {}
        }
        
        # 1. Load main schema.json
        schema_data = self.path_manager.load_json_safe('schema')
        if schema_data:
            self._process_schema_data(schema_data, schema_context)
            schema_context['data_sources']['schema'] = True
        
        # 2. Load tables.json (additional table info)
        tables_data = self.path_manager.load_json_safe('tables')
        if tables_data:
            self._process_tables_data(tables_data, schema_context)
            schema_context['data_sources']['tables'] = True
        
        # 3. Load XML schema data
        xml_data = self.path_manager.load_json_safe('xml_schema')
        if xml_data:
            self._process_xml_data(xml_data, schema_context)
            schema_context['data_sources']['xml_schema'] = True
        
        # 4. Load joins data
        joins_data = self.path_manager.load_json_safe('joins')
        if joins_data:
            self._process_joins_data(joins_data, schema_context)
            schema_context['data_sources']['joins'] = True
        
        # 5. XML Manager integration
        if self.xml_manager:
            try:
                xml_stats = self.xml_manager.get_statistics()
                schema_context['xml_manager_stats'] = xml_stats
                schema_context['data_sources']['xml_manager'] = True
            except Exception as e:
                self.logger.warning(f"XML manager stats failed: {e}")
        
        schema_context['available'] = len(schema_context['all_tables']) > 0
        return schema_context
    
    def _process_schema_data(self, schema_data: Any, context: Dict[str, Any]):
        """Process main schema.json data"""
        if isinstance(schema_data, list):
            for item in schema_data:
                if isinstance(item, dict):
                    table = item.get('table', '').strip()
                    column = item.get('column', '').strip()
                    
                    if table and column:
                        if table not in context['tables']:
                            context['tables'][table] = {'columns': [], 'metadata': {}}
                        
                        if column not in context['tables'][table]['columns']:
                            context['tables'][table]['columns'].append(column)
                        
                        context['all_tables'].add(table)
                        context['all_columns'].add(column)
    
    def _process_tables_data(self, tables_data: Any, context: Dict[str, Any]):
        """Process tables.json data"""
        # Add any additional table processing logic here
        if isinstance(tables_data, dict):
            for table, info in tables_data.items():
                if table not in context['tables']:
                    context['tables'][table] = {'columns': [], 'metadata': {}}
                
                if isinstance(info, dict):
                    context['tables'][table]['metadata'].update(info)
                
                context['all_tables'].add(table)
    
    def _process_xml_data(self, xml_data: Any, context: Dict[str, Any]):
        """Process xml_schema.json data"""
        if isinstance(xml_data, list):
            for item in xml_data:
                if isinstance(item, dict):
                    table = item.get('table', '').strip()
                    xml_column = item.get('xml_column', '').strip()
                    xpath = item.get('xpath', '')
                    
                    if table and xml_column:
                        if table not in context['xml_fields']:
                            context['xml_fields'][table] = []
                        
                        context['xml_fields'][table].append({
                            'xml_column': xml_column,
                            'xpath': xpath,
                            'metadata': item
                        })
                        
                        context['all_tables'].add(table)
    
    def _process_joins_data(self, joins_data: Any, context: Dict[str, Any]):
        """Process joins_verified.json data"""
        if isinstance(joins_data, list):
            context['joins'] = joins_data
        elif isinstance(joins_data, dict):
            context['joins'] = joins_data.get('joins', [])
    
    def validate_schema_compliance(self, results: List[Dict[str, Any]]) -> float:
        """Calculate schema compliance score for results"""
        if not results:
            return 0.0
            
        schema_context = self.get_schema_context()
        if not schema_context['available']:
            return 1.0  # No schema available, assume compliance
        
        valid_tables = schema_context['all_tables']
        valid_columns = schema_context['all_columns']
        
        compliant_results = 0
        for result in results:
            table_name = result.get('table_name', '').strip()
            column_name = result.get('column_name', '').strip()
            
            # Check if table and column exist in schema
            table_match = table_name in valid_tables or any(
                table_name.lower() in valid_table.lower() for valid_table in valid_tables
            )
            column_match = column_name in valid_columns or any(
                column_name.lower() in valid_column.lower() for valid_column in valid_columns
            )
            
            if table_match and column_match:
                compliant_results += 1
        
        return compliant_results / len(results) if results else 0.0
    
    def find_xml_fields(self, results: List[Dict[str, Any]]) -> int:
        """Find and mark XML fields in results"""
        if not results:
            return 0
            
        schema_context = self.get_schema_context()
        xml_fields = schema_context.get('xml_fields', {})
        
        xml_count = 0
        for result in results:
            table_name = result.get('table_name', '').strip()
            column_name = result.get('column_name', '').strip()
            
            if table_name in xml_fields:
                for xml_field in xml_fields[table_name]:
                    if column_name == xml_field['xml_column']:
                        result['is_xml_field'] = True
                        result['xml_xpath'] = xml_field['xpath']
                        xml_count += 1
                        break
        
        return xml_count

class SearchOrchestrator:
    """
    COMPLETE ENHANCED: SearchOrchestrator with all accuracy improvements + safe path handling
    
    KEY ENHANCEMENTS:
    - Schema-aware validation using your exact data files
    - Intelligent convergence detection with dynamic thresholds
    - Query complexity analysis for smart engine selection
    - Multi-criteria result quality scoring
    - Enhanced AI enhancement with schema constraints
    - XML integration support
    - Safe path handling for all data files
    - Comprehensive error handling and fallbacks
    
    BACKWARD COMPATIBILITY: 100% maintained - all existing functionality preserved
    """

    def __init__(
        self,
        engines: Union[Dict[SearchMethod, Any], List[Any]],
        async_client_manager: Optional['AsyncClientManager'] = None,
        max_iterations: int = 5,  # Keep existing default - can be overridden dynamically
        convergence_threshold: float = 0.8,  # Keep existing default - can be adjusted dynamically
        min_improvement_threshold: float = 0.05,
        enable_ai_enhancement: bool = True,
        # NEW: Enhanced configuration options
        enable_schema_validation: bool = True,
        enable_intelligent_engine_selection: bool = True,
        enable_dynamic_convergence: bool = True,
        enable_xml_integration: bool = True
    ):
        # CRITICAL FIX 1: Initialize logger FIRST
        self.logger = logging.getLogger("SearchOrchestrator")
        
        # CRITICAL FIX 2: Store AsyncClientManager IMMEDIATELY
        self.async_client_manager = async_client_manager
        
        # CRITICAL FIX 3: Add comprehensive debug logging
        self.logger.info("DEBUG: Enhanced SearchOrchestrator.__init__ called")
        self.logger.info(f"DEBUG: - Received AsyncClientManager: {async_client_manager is not None}")
        if async_client_manager:
            self.logger.info(f"DEBUG: - AsyncClientManager ID: {id(async_client_manager)}")
            self.logger.info(f"DEBUG: - AsyncClientManager type: {type(async_client_manager)}")
            # Check if it has expected methods
            has_methods = all(hasattr(async_client_manager, method) for method in ['generate_sql_async', 'health_check', 'get_client_status'])
            self.logger.info(f"DEBUG: - Has expected methods: {has_methods}")
        else:
            self.logger.warning("DEBUG: AsyncClientManager is None in SearchOrchestrator init!")

        # Normalize engines to dict format
        self.engines = self._normalize_engines(engines)
        
        # Validate critical configuration
        if not self.engines:
            raise SearchConfigurationError("No search engines provided after normalization")
        if max_iterations <= 0:
            raise SearchConfigurationError("max_iterations must be positive")
        if not (0.0 <= convergence_threshold <= 1.0):
            raise SearchConfigurationError("convergence_threshold must be between 0 and 1")

        # Store configuration (existing)
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.min_improvement_threshold = min_improvement_threshold
        self.enable_ai_enhancement = enable_ai_enhancement
        
        # NEW: Enhanced configuration options
        self.enable_schema_validation = enable_schema_validation
        self.enable_intelligent_engine_selection = enable_intelligent_engine_selection
        self.enable_dynamic_convergence = enable_dynamic_convergence
        self.enable_xml_integration = enable_xml_integration
        
        # NEW: Initialize enhanced schema context manager with safe path handling
        self.schema_manager = EnhancedSchemaContextManager() if enable_schema_validation else None
        
        # NEW: Query complexity patterns for intelligent engine selection
        self._complexity_patterns = {
            QueryComplexity.SIMPLE: [
                r'\b(show|list|get|find)\s+\w+\s*$',
                r'^\w+\s*(table|column)s?\b',
                r'^\w+\s+\w+\s*$'
            ],
            QueryComplexity.MODERATE: [
                r'\b(with|having|where|join)\b',
                r'\b\w+\s+and\s+\w+\b',
                r'customers?\s+with\s+\w+'
            ],
            QueryComplexity.COMPLEX: [
                r'\b(relationship|related|association)\b',
                r'\bcomplex\s+\w+\b',
                r'\b(analysis|analytics|analyze)\b'
            ],
            QueryComplexity.ANALYTICAL: [
                r'\b(aggregate|sum|count|average|trend)\b',
                r'\b(report|dashboard|metric)\b',
                r'\b(over\s+time|historical|timeline)\b'
            ]
        }
        
        # NEW: Engine performance profiles for intelligent selection
        self._engine_profiles = {
            SearchMethod.BM25: {
                'best_for': [QueryComplexity.SIMPLE, QueryComplexity.MODERATE],
                'strengths': ['exact_match', 'keyword_search'],
                'avg_time': 2.0,
                'reliability': 0.9
            },
            SearchMethod.SEMANTIC: {
                'best_for': [QueryComplexity.MODERATE, QueryComplexity.COMPLEX],
                'strengths': ['conceptual_search', 'domain_understanding'],
                'avg_time': 5.0,
                'reliability': 0.8
            },
            SearchMethod.FAISS: {
                'best_for': [QueryComplexity.MODERATE, QueryComplexity.COMPLEX],
                'strengths': ['similarity_search', 'fast_retrieval'],
                'avg_time': 3.0,
                'reliability': 0.85
            },
            SearchMethod.FUZZY: {
                'best_for': [QueryComplexity.SIMPLE, QueryComplexity.MODERATE],
                'strengths': ['typo_tolerance', 'partial_match'],
                'avg_time': 4.0,
                'reliability': 0.7
            },
            SearchMethod.NLP: {
                'best_for': [QueryComplexity.COMPLEX, QueryComplexity.ANALYTICAL],
                'strengths': ['natural_language', 'complex_understanding'],
                'avg_time': 8.0,
                'reliability': 0.75
            },
            SearchMethod.CHROMA: {
                'best_for': [QueryComplexity.MODERATE, QueryComplexity.COMPLEX],
                'strengths': ['vector_search', 'semantic_similarity'],
                'avg_time': 4.5,
                'reliability': 0.8
            }
        }

        # Initialize components with proper error handling
        self._initialize_components()

        # Final validation
        self.logger.info("DEBUG: Enhanced SearchOrchestrator final init status:")
        self.logger.info(f"DEBUG: - Final AsyncClientManager available: {hasattr(self, 'async_client_manager') and self.async_client_manager is not None}")
        if hasattr(self, 'async_client_manager') and self.async_client_manager:
            self.logger.info(f"DEBUG: - Final AsyncClientManager ID: {id(self.async_client_manager)}")
            # Validate the object hasn't been corrupted
            try:
                status = self.async_client_manager.get_client_status()
                self.logger.info("DEBUG: - AsyncClientManager functional test: SUCCESS")
            except Exception as e:
                self.logger.error(f"DEBUG: - AsyncClientManager functional test: FAILED - {e}")

        # NEW: Log enhancement status
        self.logger.info(f"Enhanced SearchOrchestrator initialized successfully with {len(self.engines)} engines")
        self.logger.info(f"Enhancements: Schema={self.enable_schema_validation}, "
                        f"Intelligent={self.enable_intelligent_engine_selection}, "
                        f"Dynamic={self.enable_dynamic_convergence}, "
                        f"XML={self.enable_xml_integration}")

    # [Keep all existing methods for backward compatibility...]
    def _normalize_engines(self, engines: Union[Dict[SearchMethod, Any], List[Any]]) -> Dict[SearchMethod, Any]:
        """EXISTING METHOD - unchanged for backward compatibility"""
        try:
            # Already a dict - validate and return
            if isinstance(engines, dict):
                normalized: Dict[SearchMethod, Any] = {}
                for key, engine in engines.items():
                    if isinstance(key, SearchMethod):
                        normalized[key] = engine
                    elif isinstance(key, str):
                        try:
                            method = SearchMethod(key.lower())
                            normalized[method] = engine
                        except ValueError:
                            self.logger.warning(f"Unknown search method '{key}', using generic mapping")
                            method = SearchMethod.SEMANTIC
                            normalized[method] = engine
                    else:
                        normalized[SearchMethod.SEMANTIC] = engine
                self.logger.info(f"Normalized dict engines: {list(normalized.keys())}")
                return normalized

            # List format - convert to dict
            elif isinstance(engines, list):
                self.logger.info(f"Converting list of {len(engines)} engines to dict format")
                normalized = {}
                method_priority = [
                    SearchMethod.SEMANTIC, SearchMethod.FAISS, SearchMethod.CHROMA,
                    SearchMethod.BM25, SearchMethod.FUZZY, SearchMethod.NLP
                ]

                for i, engine in enumerate(engines):
                    # Try to detect engine type from class name
                    engine_class_name = getattr(engine, '__class__', type(engine)).__name__.lower()
                    method = None

                    # Smart method detection based on class name
                    if 'semantic' in engine_class_name:
                        method = SearchMethod.SEMANTIC
                    elif 'faiss' in engine_class_name:
                        method = SearchMethod.FAISS
                    elif 'chroma' in engine_class_name:
                        method = SearchMethod.CHROMA
                    elif 'bm25' in engine_class_name:
                        method = SearchMethod.BM25
                    elif 'fuzzy' in engine_class_name:
                        method = SearchMethod.FUZZY
                    elif 'nlp' in engine_class_name:
                        method = SearchMethod.NLP
                    else:
                        if i < len(method_priority):
                            method = method_priority[i]
                        else:
                            method = SearchMethod.SEMANTIC

                    # Ensure no duplicate keys
                    original_method = method
                    counter = 1
                    while method in normalized:
                        self.logger.warning(f"Duplicate method {method}, using fallback assignment")
                        if counter < len(method_priority):
                            method = method_priority[counter]
                        else:
                            method = SearchMethod.SEMANTIC
                        counter += 1

                    normalized[method] = engine
                    self.logger.info(f"Assigned engine {i} ({engine_class_name}) to method {method.value}")

                return normalized

            # Single engine - convert to dict
            else:
                self.logger.info("Converting single engine to dict format")
                engine_class_name = getattr(engines, '__class__', type(engines)).__name__.lower()
                if 'semantic' in engine_class_name:
                    method = SearchMethod.SEMANTIC
                elif 'faiss' in engine_class_name:
                    method = SearchMethod.FAISS
                elif 'chroma' in engine_class_name:
                    method = SearchMethod.CHROMA
                else:
                    method = SearchMethod.SEMANTIC
                return {method: engines}

        except Exception as e:
            self.logger.error(f"Engine normalization failed: {e}")
            return {}

    def _initialize_components(self) -> None:
        """EXISTING METHOD - unchanged for backward compatibility"""
        # Re-verify AsyncClientManager at start
        self.logger.info("DEBUG: _initialize_components called")
        self.logger.info(f"DEBUG: - AsyncClientManager at start: {self.async_client_manager is not None}")
        if self.async_client_manager:
            self.logger.info(f"DEBUG: - AsyncClientManager ID at start: {id(self.async_client_manager)}")

        # AsyncClientManager integration
        if self.async_client_manager:
            self.logger.info("AsyncClientManager integration enabled")
        else:
            self.logger.warning("AsyncClientManager not available - AI features disabled")

        # Initialize reasoning components - non-critical
        self.gap_detector = None
        self.keyword_generator = None
        self.schema_analyzer = None

        try:
            from agent.schema_searcher.reasoning.gap_detector import GapDetector
            from agent.schema_searcher.reasoning.keyword_generator import KeywordGenerator
            from agent.schema_searcher.reasoning.schema_analyzer import SchemaAnalyzer

            self.gap_detector = GapDetector()
            self.keyword_generator = KeywordGenerator()
            self.schema_analyzer = SchemaAnalyzer()
            self.logger.info("Reasoning components initialized")
        except ImportError as e:
            self.logger.warning(f"Reasoning components not available: {e}")

        # Performance monitoring
        self.search_history: List[Dict[str, Any]] = []
        self.performance_stats = {
            'total_searches': 0,
            'total_errors': 0,
            'average_response_time': 0.0
        }

        # Final AsyncClientManager verification
        self.logger.info("DEBUG: _initialize_components finished")
        self.logger.info(f"DEBUG: - AsyncClientManager at end: {self.async_client_manager is not None}")
        if self.async_client_manager:
            self.logger.info(f"DEBUG: - AsyncClientManager ID at end: {id(self.async_client_manager)}")

    # NEW: Query complexity analysis
    def _analyze_query_complexity(self, query: str) -> QueryComplexity:
        """Analyze query complexity for intelligent engine selection"""
        if not self.enable_intelligent_engine_selection:
            return QueryComplexity.MODERATE  # Default fallback
        
        query_lower = query.lower()
        
        # Check patterns in order of complexity
        for complexity, patterns in self._complexity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    self.logger.debug(f"Query classified as {complexity.value}: '{query}'")
                    return complexity
        
        # Default to moderate complexity
        return QueryComplexity.MODERATE

    # NEW: Intelligent engine selection
    def _select_engines_intelligently(
        self, 
        query_complexity: QueryComplexity, 
        available_engines: Dict[SearchMethod, Any]
    ) -> Dict[SearchMethod, Any]:
        """Select engines based on query complexity and performance profiles"""
        if not self.enable_intelligent_engine_selection:
            return available_engines  # Use all engines (existing behavior)
        
        # Score engines based on complexity and profiles
        engine_scores = {}
        for method, engine in available_engines.items():
            profile = self._engine_profiles.get(method, {})
            best_for = profile.get('best_for', [])
            reliability = profile.get('reliability', 0.5)
            
            # Base score from complexity match
            complexity_score = 1.0 if query_complexity in best_for else 0.5
            
            # Combined score
            total_score = (complexity_score * 0.7) + (reliability * 0.3)
            engine_scores[method] = total_score
        
        # Select top engines (at least 2, at most 4)
        sorted_engines = sorted(engine_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Ensure we have at least 2 engines for reliability
        min_engines = min(2, len(available_engines))
        max_engines = min(4, len(available_engines))
        
        # Select engines with score > 0.6 or top performers
        selected_count = min(max_engines, max(min_engines, 
            len([score for _, score in sorted_engines if score > 0.6])))
        
        selected_engines = {
            method: available_engines[method] 
            for method, _ in sorted_engines[:selected_count]
        }
        
        self.logger.info(f"Intelligent engine selection ({query_complexity.value}): "
                        f"{[m.value for m in selected_engines.keys()]} "
                        f"(from {len(available_engines)} available)")
        
        return selected_engines

    # NEW: Dynamic convergence detection
    def _should_continue_iteration(
        self, 
        iteration: int, 
        all_results: List[Dict[str, Any]], 
        keywords: List[str],
        last_iteration_count: int
    ) -> Tuple[bool, str]:
        """Enhanced convergence detection with dynamic thresholds"""
        if not self.enable_dynamic_convergence:
            # Use existing behavior
            coverage = self._calculate_coverage(all_results, keywords)
            return coverage < self.convergence_threshold, f"coverage={coverage:.2f}"
        
        # Enhanced dynamic convergence logic
        current_count = len(all_results)
        
        # Check for diminishing returns
        if last_iteration_count > 0:
            improvement_rate = (current_count - last_iteration_count) / last_iteration_count
            if improvement_rate < 0.1 and iteration > 1:  # Less than 10% improvement
                return False, f"diminishing_returns={improvement_rate:.2f}"
        
        # Calculate quality-adjusted coverage
        coverage = self._calculate_coverage(all_results, keywords)
        
        # Dynamic threshold based on iteration number (more lenient over time)
        dynamic_threshold = max(0.3, self.convergence_threshold - (iteration * 0.1))
        
        # Check schema compliance if available
        schema_compliance = 1.0
        if self.schema_manager:
            schema_compliance = self.schema_manager.validate_schema_compliance(all_results)
            # Boost coverage with schema compliance
            adjusted_coverage = (coverage * 0.7) + (schema_compliance * 0.3)
        else:
            adjusted_coverage = coverage
        
        should_continue = adjusted_coverage < dynamic_threshold
        
        self.logger.debug(f"Dynamic convergence: coverage={coverage:.2f}, "
                         f"schema_compliance={schema_compliance:.2f}, "
                         f"adjusted={adjusted_coverage:.2f}, threshold={dynamic_threshold:.2f}")
        
        return should_continue, f"adjusted_coverage={adjusted_coverage:.2f}"

    # ENHANCED: Main orchestration method with ALL improvements
    async def _orchestrate_search_async(self, query: str) -> SearchResult:
        """ENHANCED: Complete orchestration with all accuracy improvements + safe path handling"""
        start_time = time.time()
        search_errors: List[str] = []
        search_warnings: List[str] = []

        self.logger.info(f"Starting comprehensive enhanced search for: '{query}'")

        # NEW: Analyze query complexity
        query_complexity = self._analyze_query_complexity(query)
        
        # Validate engines with detailed reporting
        available_engines = self._validate_engines_enhanced()
        if not available_engines:
            raise SearchExecutionError("No healthy search engines available after validation")

        # NEW: Intelligent engine selection
        selected_engines = self._select_engines_intelligently(query_complexity, available_engines)
        engines_selected_intelligently = len(selected_engines) < len(available_engines)

        # Process query (existing logic)
        keywords = self._process_query(query)
        if not keywords:
            search_warnings.append("No meaningful keywords extracted from query")
            keywords = [query]

        # NEW: Enhanced AI enhancement with schema context
        ai_enhanced_query: str = query
        ai_enhancement_used = False
        ai_stats: Dict[str, Any] = {}

        if self.enable_ai_enhancement and self.async_client_manager:
            try:
                ai_result = await self._enhance_query_with_ai_enhanced(query)
                if self._safe_get(ai_result, 'success'):
                    enhanced_query_raw = self._safe_get(ai_result, 'enhanced_query', query)
                    ai_enhanced_query = self._ensure_string(enhanced_query_raw)
                    ai_enhancement_used = True
                    ai_stats = ai_result
                    self.logger.info(f"Schema-aware AI enhancement successful: '{ai_enhanced_query}'")
                else:
                    search_warnings.append(f"AI enhancement failed: {self._safe_get(ai_result, 'error', 'Unknown error')}")
            except Exception as e:
                search_warnings.append(f"AI enhancement error: {str(e)}")
                self.logger.warning(f"AI enhancement failed: {e}")

        # Enhanced iterative search with dynamic convergence
        all_results: List[Dict[str, Any]] = []
        iterations_completed = 0
        convergence_achieved = False
        last_iteration_count = 0

        for iteration in range(self.max_iterations):
            try:
                self.logger.info(f"=== ENHANCED ITERATION {iteration + 1} START ===")
                self.logger.info(f"Query: '{ai_enhanced_query}' (complexity: {query_complexity.value})")
                self.logger.info(f"Keywords: {keywords}")
                self.logger.info(f"Selected engines: {[method.value for method in selected_engines.keys()]}")
                
                iteration_results = await self._execute_search_iteration_enhanced(
                    selected_engines, keywords, ai_enhanced_query, iteration
                )

                all_results.extend(iteration_results['results'])
                iterations_completed += 1

                # Log iteration results
                self.logger.info(f"=== ENHANCED ITERATION {iteration + 1} RESULTS ===")
                self.logger.info(f"Results from this iteration: {len(iteration_results['results'])}")
                self.logger.info(f"Total results so far: {len(all_results)}")

                # NEW: Enhanced convergence detection
                should_continue, convergence_reason = self._should_continue_iteration(
                    iteration, all_results, keywords, last_iteration_count
                )
                
                self.logger.info(f"Convergence status: {convergence_reason}")
                
                if not should_continue:
                    convergence_achieved = True
                    self.logger.info(f"Enhanced convergence achieved at iteration {iteration + 1}: {convergence_reason}")
                    break

                last_iteration_count = len(all_results)

                if iteration_results.get('warnings'):
                    search_warnings.extend(iteration_results['warnings'])

            except SearchExecutionError as e:
                search_errors.append(f"Iteration {iteration + 1} failed: {str(e)}")
                self.logger.error(f"Search iteration {iteration + 1} failed: {e}")
                
                # Only fail completely if first iteration fails with no results
                if not all_results and iteration == 0:
                    raise SearchExecutionError(f"Initial search iteration failed: {e}")
                else:
                    self.logger.warning(f"Continuing with partial results from previous iterations")
                    break

            except Exception as e:
                search_errors.append(f"Unexpected error in iteration {iteration + 1}: {str(e)}")
                self.logger.error(f"Unexpected iteration error: {e}")
                break

        # NEW: Enhanced result finalization with schema scoring and XML integration
        if all_results:
            final_results = await self._finalize_results_with_comprehensive_scoring(all_results)
            if not final_results and all_results:
                search_warnings.append("Result finalization filtered all results - preserving raw results")
                final_results = all_results[:10]
        else:
            final_results = []
            search_errors.append("No results from any search engine")

        total_time = time.time() - start_time

        # NEW: Calculate enhanced metrics
        schema_compliance_score = 0.0
        xml_fields_found = 0
        
        if self.schema_manager and final_results:
            schema_compliance_score = self.schema_manager.validate_schema_compliance(final_results)
            if self.enable_xml_integration:
                xml_fields_found = self.schema_manager.find_xml_fields(final_results)

        # Update performance stats
        self.performance_stats['total_searches'] += 1
        self._update_performance_stats(total_time, len(final_results), bool(search_errors))

        # Enhanced comprehensive metadata
        result_metadata = {
            'query_original': query,
            'query_enhanced': ai_enhanced_query,
            'query_complexity': query_complexity.value,
            'keywords_used': keywords,
            'engines_used': [method.value for method in selected_engines.keys()],
            'engines_attempted': len(self.engines),
            'engines_successful': len(available_engines),
            'engines_selected_intelligently': engines_selected_intelligently,
            'ai_stats': ai_stats,
            'raw_results_count': len(all_results),
            'final_results_count': len(final_results),
            'has_valid_results': len(final_results) > 0,
            'schema_compliance_score': schema_compliance_score,
            'xml_fields_found': xml_fields_found,
            'search_orchestrator_version': 'v2.5.0_complete_enhancement',
            'enhancements_applied': {
                'schema_validation': self.enable_schema_validation,
                'intelligent_engine_selection': engines_selected_intelligently,
                'dynamic_convergence': self.enable_dynamic_convergence,
                'xml_integration': self.enable_xml_integration,
                'safe_path_handling': True,
                'comprehensive_data_loading': True
            },
            'data_sources_used': self.schema_manager.get_schema_context().get('data_sources', {}) if self.schema_manager else {}
        }

        # Build comprehensive result with all enhancements
        result = SearchResult(
            results=final_results,
            metadata=result_metadata,
            total_time=total_time,
            iterations_performed=iterations_completed,
            convergence_achieved=convergence_achieved,
            ai_enhancement_used=ai_enhancement_used,
            errors=search_errors,
            warnings=search_warnings,
            performance_stats=self._get_performance_summary(),
            schema_compliance_score=schema_compliance_score,
            query_complexity=query_complexity,
            engines_selected_intelligently=engines_selected_intelligently,
            xml_fields_found=xml_fields_found
        )

        # Final comprehensive logging
        self.logger.info(f"=== ENHANCED SEARCH COMPLETE ===")
        self.logger.info(f"Final results: {len(final_results)}")
        self.logger.info(f"Total time: {total_time:.2f}s")
        self.logger.info(f"Schema compliance: {schema_compliance_score:.2f}")
        self.logger.info(f"XML fields found: {xml_fields_found}")
        self.logger.info(f"Query complexity: {query_complexity.value}")
        self.logger.info(f"Engines used intelligently: {engines_selected_intelligently}")
        self.logger.info(f"Errors: {len(search_errors)}")
        self.logger.info(f"Warnings: {len(search_warnings)}")

        if final_results:
            sample_tables = list(set(r.get('table_name', 'unknown') for r in final_results[:5]))
            self.logger.info(f"Sample tables found: {sample_tables}")

        return result

    # NEW: Enhanced AI enhancement with schema context from your data files
    async def _enhance_query_with_ai_enhanced(self, query: str) -> Dict[str, Any]:
        """Enhanced AI query enhancement with schema context from your actual data files"""
        if not self.async_client_manager:
            return {'success': False, 'error': 'AsyncClientManager not available'}

        try:
            # Get comprehensive schema context from your data files
            schema_context = ""
            if self.schema_manager:
                schema_info = self.schema_manager.get_schema_context()
                if schema_info['available']:
                    # Provide actual schema context to AI from your data files
                    table_info = []
                    tables_processed = 0
                    
                    for table_name, table_data in schema_info['tables'].items():
                        if tables_processed >= 10:  # Limit for prompt size
                            break
                        
                        columns = table_data['columns'][:8]  # First 8 columns
                        table_info.append(f"  {table_name}: {', '.join(columns)}")
                        tables_processed += 1
                    
                    schema_context = f"\nActual Database Schema (from {', '.join(schema_info.get('data_sources', {}).keys())}):\n" + "\n".join(table_info)
                    
                    # Add XML fields info if available
                    if schema_info.get('xml_fields') and self.enable_xml_integration:
                        xml_info = []
                        for table, xml_fields in list(schema_info['xml_fields'].items())[:3]:
                            xml_columns = [f['xml_column'] for f in xml_fields[:3]]
                            xml_info.append(f"  {table} (XML): {', '.join(xml_columns)}")
                        if xml_info:
                            schema_context += f"\nXML Fields:\n" + "\n".join(xml_info)
                    
                    schema_context += "\n"

            enhancement_prompt = f"""Enhance this database schema search query using ONLY the provided actual schema:

Original Query: {query}
{schema_context}
STRICT RULES:
1. ONLY use table and column names from the Actual Database Schema above
2. DO NOT create, invent, or assume any table/column names not listed above
3. Focus on database schema elements (tables, columns, relationships)
4. Expand abbreviations and technical terms using only available schema elements
5. Include related concepts and synonyms from available schema only
6. If no relevant schema elements are found, return the original query unchanged
7. Keep banking/financial domain context if relevant to available schema

Enhanced Query using ONLY the actual schema provided:"""

            # Try multiple methods to find working AI interface
            result = None
            for method_name in ['generate_sql_async', 'generate_async', 'query_async']:
                if hasattr(self.async_client_manager, method_name):
                    try:
                        method = getattr(self.async_client_manager, method_name)
                        if method_name == 'generate_sql_async':
                            result = await method(enhancement_prompt, target_llm='mathstral')
                        else:
                            result = await method(enhancement_prompt)
                        break
                    except Exception as e:
                        self.logger.warning(f"AI method {method_name} failed: {e}")
                        continue

            if not result:
                return {'success': False, 'error': 'No compatible AI method found'}

            if result and self._safe_get(result, 'success'):
                enhanced_query = (
                    self._safe_get(result, 'sql') or
                    self._safe_get(result, 'response') or
                    self._safe_get(result, 'content') or
                    ''
                ).strip()

                if enhanced_query and enhanced_query != query and len(enhanced_query) > 5:
                    # NEW: Validate enhancement against actual schema
                    if self.schema_manager:
                        schema_info = self.schema_manager.get_schema_context()
                        if schema_info['available']:
                            # Check for fictional table names that shouldn't exist
                            fictional_tables = ['bankingtransactions', 'accountbalances', 'userprofiles', 'transactionhistory']
                            enhanced_lower = enhanced_query.lower()
                            
                            for fictional in fictional_tables:
                                if fictional in enhanced_lower:
                                    self.logger.warning(f"AI enhancement contains fictional table: {fictional}, rejecting")
                                    return {
                                        'success': False,
                                        'error': f'AI enhancement rejected - contains fictional table: {fictional}',
                                        'fallback_query': query
                                    }

                    return {
                        'success': True,
                        'enhanced_query': enhanced_query,
                        'original_query': query,
                        'client_used': self._safe_get(result, 'model_used', 'unknown'),
                        'enhancement_method': 'ai_powered_schema_constrained',
                        'schema_context_used': bool(schema_context),
                        'data_sources_used': list(self.schema_manager.get_schema_context().get('data_sources', {}).keys()) if self.schema_manager else []
                    }

            return {
                'success': False,
                'error': 'AI enhancement produced invalid result',
                'fallback_query': query
            }

        except Exception as e:
            self.logger.error(f"Enhanced AI enhancement failed: {e}")
            return {
                'success': False,
                'error': f'AI enhancement failed: {str(e)}',
                'fallback_query': query
            }

    # NEW: Enhanced result finalization with comprehensive scoring
    async def _finalize_results_with_comprehensive_scoring(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced result finalization with comprehensive scoring including XML integration"""
        if not results:
            return []

        # Apply schema-aware scoring if available
        if self.schema_manager:
            schema_info = self.schema_manager.get_schema_context()
            if schema_info['available']:
                for result in results:
                    # Add comprehensive schema compliance scoring
                    table_name = result.get('table_name', '').strip()
                    column_name = result.get('column_name', '').strip()
                    
                    # Check schema compliance
                    table_match = table_name in schema_info['all_tables']
                    column_match = column_name in schema_info['all_columns']
                    
                    # Calculate schema score
                    schema_score = 0.0
                    if table_match and column_match:
                        schema_score = 1.0
                    elif table_match:
                        schema_score = 0.7
                    elif any(table_name.lower() in t.lower() for t in schema_info['all_tables']):
                        schema_score = 0.5
                    
                    # Check for XML fields if XML integration is enabled
                    xml_boost = 0.0
                    if self.enable_xml_integration and table_name in schema_info.get('xml_fields', {}):
                        for xml_field in schema_info['xml_fields'][table_name]:
                            if column_name == xml_field['xml_column']:
                                xml_boost = 0.2  # Boost for XML fields
                                result['is_xml_field'] = True
                                result['xml_xpath'] = xml_field['xpath']
                                break
                    
                    # Comprehensive enhanced scoring
                    original_score = result.get('score', 0.5)
                    result['schema_compliance_score'] = schema_score
                    result['xml_boost'] = xml_boost
                    result['enhanced_score'] = (original_score * 0.5) + (schema_score * 0.4) + (xml_boost * 0.1)

        # Use enhanced result finalization (existing logic with improvements)
        return self._finalize_results_enhanced(results)

    # Keep ALL existing methods for backward compatibility
    async def search_schema(self, query: str) -> SearchResult:
        """EXISTING METHOD - enhanced internally but API unchanged"""
        if not query or not query.strip():
            raise SearchConfigurationError("Query cannot be empty")

        try:
            return await self._orchestrate_search_async(query)
        except SearchConfigurationError:
            raise
        except Exception as e:
            self.logger.error(f"Search failed with unexpected error: {e}")
            self.performance_stats['total_errors'] += 1
            return SearchResult(
                results=[],
                metadata={"query": query},
                errors=[f"Search failed: {str(e)}"],
                total_time=0.0
            )

    def search_schema_sync(self, query: str) -> SearchResult:
        """EXISTING METHOD - unchanged for backward compatibility"""
        return asyncio.run(self.search_schema(query))

    def _safe_get(self, obj: Any, key: str, default: Any = None) -> Any:
        """EXISTING METHOD - unchanged"""
        if isinstance(obj, dict):
            return obj.get(key, default)
        try:
            return getattr(obj, key, default)
        except Exception:
            return default

    def _safe_int_comparison(self, value: Any, comparison_value: int) -> bool:
        """EXISTING METHOD - unchanged"""
        if value is None:
            return False
        try:
            return int(value) > comparison_value
        except (TypeError, ValueError):
            return False

    def _ensure_string(self, value: Any) -> str:
        """EXISTING METHOD - unchanged"""
        if value is None:
            return ""
        return str(value)

    # [All other existing methods remain unchanged for backward compatibility]
    # Including: _validate_engines_enhanced, _process_query, _execute_search_iteration_enhanced,
    # _search_single_engine_enhanced, _standardize_engine_results_enhanced, _calculate_coverage,
    # _finalize_results_enhanced, _update_performance_stats, _get_performance_summary, etc.

    def _validate_engines_enhanced(self) -> Dict[SearchMethod, Any]:
        """EXISTING METHOD - unchanged for backward compatibility"""
        healthy_engines: Dict[SearchMethod, Any] = {}
        
        self.logger.info(f"=== ENGINE VALIDATION START ===")
        self.logger.info(f"Total engines to validate: {len(self.engines)}")

        for method, engine in self.engines.items():
            self.logger.info(f"[{method.value}] Validating engine: {type(engine).__name__}")
            
            try:
                # Check for health_check method first
                if hasattr(engine, 'health_check'):
                    try:
                        health_result = engine.health_check()
                        if health_result:
                            self.logger.info(f"[{method.value}] Health check: PASSED")
                            healthy_engines[method] = engine
                            continue
                        else:
                            self.logger.warning(f"[{method.value}] Health check: FAILED")
                            continue
                    except Exception as e:
                        self.logger.warning(f"[{method.value}] Health check error: {e}")

                # Check for any search-compatible method
                search_methods = ['search_async', 'search', 'retrieve', 'retrieve_async', 
                                'find', 'query', 'retrieve_complete_schema', 'retrieve_complete_schema_json']
                
                has_search_method = any(hasattr(engine, method_name) for method_name in search_methods)
                available_methods = [m for m in search_methods if hasattr(engine, m)]
                
                if has_search_method:
                    self.logger.info(f"[{method.value}] Compatible methods found: {available_methods}")
                    healthy_engines[method] = engine
                else:
                    all_methods = [attr for attr in dir(engine) if not attr.startswith('_') and callable(getattr(engine, attr))]
                    self.logger.error(f"[{method.value}] No compatible search method found")
                    self.logger.error(f"[{method.value}] Available methods: {all_methods}")
                    
            except Exception as e:
                self.logger.error(f"[{method.value}] Validation completely failed: {e}")

        self.logger.info(f"=== ENGINE VALIDATION COMPLETE ===")
        self.logger.info(f"Healthy engines: {len(healthy_engines)}/{len(self.engines)}")
        self.logger.info(f"Validated engines: {[method.value for method in healthy_engines.keys()]}")
        
        if not healthy_engines:
            self.logger.error("CRITICAL: No healthy search engines available!")
            raise SearchExecutionError("No healthy search engines available after validation")

        return healthy_engines

    def _process_query(self, query: str) -> List[str]:
        """EXISTING METHOD - unchanged"""
        keywords = re.findall(r'\b\w+\b', query.lower())
        keywords = [k for k in keywords if len(k) > 2]
        if not keywords:
            return [query.strip()]
        return keywords

    # [Include all other existing methods here...]

    async def _execute_search_iteration_enhanced(
        self,
        engines: Dict[SearchMethod, Any],
        keywords: List[str],
        query: str,
        iteration: int
    ) -> Dict[str, Any]:
        """EXISTING METHOD - unchanged for backward compatibility"""
        
        self.logger.info(f"Query: '{query}'")
        self.logger.info(f"Keywords: {keywords}")
        self.logger.info(f"Engines to try: {[method.value for method in engines.keys()]}")
        
        iteration_results: List[Dict[str, Any]] = []
        iteration_warnings: List[str] = []
        successful_engines: List[str] = []
        failed_engines: List[Tuple[str, str]] = []
        
        # Try engines one by one with detailed reporting
        for method, engine in engines.items():
            engine_start_time = time.time()
            
            try:
                self.logger.info(f"[{method.value}] Starting engine search...")
                
                engine_results = await self._search_single_engine_enhanced(engine, method, keywords, query)
                
                engine_time = time.time() - engine_start_time
                
                if engine_results:
                    iteration_results.extend(engine_results)
                    successful_engines.append(method.value)
                    self.logger.info(f"[{method.value}] SUCCESS: {len(engine_results)} results in {engine_time:.2f}s")
                else:
                    failed_engines.append((method.value, "No results returned"))
                    self.logger.warning(f"[{method.value}] FAILED: No results in {engine_time:.2f}s")
                    
            except Exception as e:
                engine_time = time.time() - engine_start_time
                error_msg = f"{type(e).__name__}: {str(e)}"
                failed_engines.append((method.value, error_msg))
                self.logger.error(f"[{method.value}] FAILED: {error_msg} in {engine_time:.2f}s")
                
        # Log iteration summary
        self.logger.info(f"=== ITERATION {iteration + 1} SUMMARY ===")
        self.logger.info(f"Successful engines: {successful_engines}")
        self.logger.info(f"Failed engines: {[f'{name}: {error}' for name, error in failed_engines]}")
        self.logger.info(f"Total results: {len(iteration_results)}")
        
        # CRITICAL FIX: Only fail if ALL engines fail
        if not successful_engines:
            error_summary = f"All {len(engines)} engines failed: {dict(failed_engines)}"
            self.logger.error(f"ITERATION {iteration + 1} FAILED: {error_summary}")
            raise SearchExecutionError(error_summary)
        
        # Add warnings for failed engines but continue
        for engine_name, error in failed_engines:
            iteration_warnings.append(f"Engine {engine_name} failed: {error}")

        return {
            'results': iteration_results,
            'successful_engines': len(successful_engines),
            'failed_engines': len(failed_engines),
            'warnings': iteration_warnings,
            'engine_summary': {
                'successful': successful_engines,
                'failed': dict(failed_engines)
            }
        }

    async def _search_single_engine_enhanced(
        self,
        engine: Any,
        method: SearchMethod,
        keywords: List[str],
        query: str
    ) -> List[Dict[str, Any]]:
        """EXISTING METHOD - unchanged for backward compatibility"""
        
        self.logger.info(f"[{method.value}] Starting engine search for: '{query}'")
        
        try:
            # COMPREHENSIVE method detection and calling
            search_methods = [
                ('search_async', True),
                ('search', False),
                ('retrieve', False),
                ('retrieve_async', True),
                ('find', False),
                ('find_async', True),
                ('query', False),
                ('query_async', True),
                ('retrieve_complete_schema', False),
                ('retrieve_complete_schema_json', False)
            ]
            
            method_used = None
            for method_name, is_async in search_methods:
                if hasattr(engine, method_name):
                    method_used = (method_name, is_async)
                    self.logger.info(f"[{method.value}] Using method: {method_name} (async: {is_async})")
                    break
            
            if not method_used:
                self.logger.error(f"[{method.value}] FAILED: No compatible search method found")
                return []
            
            method_name, is_async = method_used
            engine_method = getattr(engine, method_name)
            
            # Try different parameter combinations
            param_combinations = [
                {'query': query},
                {'request': {'query': query}},
                {'query': query, 'keywords': keywords},
                {'query': query, 'limit': 10},
                {'query': query, 'context': {'keywords': keywords}},
                query,
                {'q': query},
                {'search_query': query},
                {'text': query}
            ]
            
            raw_results = None
            for i, params in enumerate(param_combinations):
                try:
                    self.logger.debug(f"[{method.value}] Trying parameter combination {i+1}: {type(params)}")
                    
                    if is_async:
                        if isinstance(params, dict):
                            raw_results = await engine_method(**params)
                        else:
                            raw_results = await engine_method(params)
                    else:
                        loop = asyncio.get_running_loop()
                        if isinstance(params, dict):
                            raw_results = await loop.run_in_executor(None, lambda: engine_method(**params))
                        else:
                            raw_results = await loop.run_in_executor(None, engine_method, params)
                    
                    if raw_results is not None:
                        self.logger.info(f"[{method.value}] SUCCESS with parameter combination {i+1}")
                        break
                    else:
                        self.logger.debug(f"[{method.value}] Parameter combination {i+1} returned None")
                        
                except Exception as e:
                    self.logger.debug(f"[{method.value}] Parameter combination {i+1} failed: {e}")
                    continue
            
            if raw_results is None:
                self.logger.error(f"[{method.value}] FAILED: All parameter combinations returned None")
                return []
                
            result_count = len(raw_results) if isinstance(raw_results, list) else 1
            self.logger.info(f"[{method.value}] SUCCESS: {result_count} raw results retrieved")
            
        except Exception as e:
            self.logger.error(f"[{method.value}] FAILED: Engine search completely failed - {e}")
            return []

        # Process and standardize results
        standardized = self._standardize_engine_results_enhanced(raw_results, method)
        self.logger.info(f"[{method.value}] Standardized to {len(standardized)} results")
        
        return standardized

    def _standardize_engine_results_enhanced(self, raw_results: Any, method: SearchMethod) -> List[Dict[str, Any]]:
        """EXISTING METHOD - unchanged for backward compatibility"""
        
        if not raw_results:
            return []

        standardized_results: List[Dict[str, Any]] = []
        
        try:
            # Handle different result formats
            if hasattr(raw_results, 'to_dict'):
                dict_result = raw_results.to_dict()
                if isinstance(dict_result, dict):
                    if 'results' in dict_result:
                        raw_results = dict_result['results']
                    elif 'data' in dict_result:
                        raw_results = dict_result['data']
                    else:
                        raw_results = [dict_result]
                else:
                    raw_results = dict_result
            elif hasattr(raw_results, 'results'):
                raw_results = raw_results.results
            elif hasattr(raw_results, 'data'):
                raw_results = raw_results.data

            if not isinstance(raw_results, list):
                raw_results = [raw_results]

            for i, result in enumerate(raw_results):
                try:
                    if isinstance(result, dict):
                        standardized_result = {
                            'table_name': (
                                result.get('table_name') or 
                                result.get('table') or 
                                result.get('table_schema') or 
                                f'table_{i}'
                            ),
                            'column_name': (
                                result.get('column_name') or 
                                result.get('column') or 
                                result.get('field') or 
                                result.get('field_name') or 
                                f'column_{i}'
                            ),
                            'description': (
                                result.get('description') or 
                                result.get('desc') or 
                                result.get('comment') or 
                                result.get('details') or 
                                ''
                            ),
                            'score': float(result.get('score', result.get('confidence', result.get('relevance', 1.0)))),
                            'source_engine': method.value,
                            'metadata': result.get('metadata', result),
                            'is_valid_schema': True,
                            'engine_source': method.value,
                            'raw_result': result
                        }
                    else:
                        result_str = str(result) if result else f'result_{i}'
                        standardized_result = {
                            'table_name': f'parsed_table_{i}',
                            'column_name': result_str[:50] if len(result_str) > 50 else result_str,
                            'description': f'Raw result from {method.value}: {result_str}',
                            'score': 0.5,
                            'source_engine': method.value,
                            'metadata': {'raw_type': type(result).__name__},
                            'is_valid_schema': True,
                            'engine_source': method.value,
                            'raw_result': result
                        }

                    standardized_results.append(standardized_result)
                    
                except Exception as e:
                    self.logger.warning(f"[{method.value}] Failed to standardize result {i}: {e}")
                    standardized_results.append({
                        'table_name': f'error_table_{i}',
                        'column_name': f'error_column_{i}',
                        'description': f'Standardization error: {e}',
                        'score': 0.1,
                        'source_engine': method.value,
                        'metadata': {'standardization_error': str(e)},
                        'is_valid_schema': False,
                        'engine_source': method.value,
                        'raw_result': result
                    })

            return standardized_results

        except Exception as e:
            self.logger.error(f"[{method.value}] Standardization completely failed: {e}")
            return [{
                'table_name': f'{method.value}_error_table',
                'column_name': 'error_column',
                'description': f'Complete standardization failure: {e}',
                'score': 0.1,
                'source_engine': method.value,
                'metadata': {'critical_error': str(e)},
                'is_valid_schema': False,
                'engine_source': method.value,
                'raw_result': raw_results
            }]

    def _calculate_coverage(self, results: List[Dict[str, Any]], keywords: List[str]) -> float:
        """EXISTING METHOD - unchanged"""
        if not results or not keywords:
            return 0.0

        total_matches = 0
        for result in results:
            result_text = ' '.join([
                str(result.get('table_name', '')),
                str(result.get('column_name', '')),
                str(result.get('description', ''))
            ]).lower()

            for keyword in keywords:
                if keyword.lower() in result_text:
                    total_matches += 1

        max_possible = len(results) * len(keywords)
        return total_matches / max_possible if max_possible > 0 else 0.0

    def _finalize_results_enhanced(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """EXISTING METHOD - unchanged for backward compatibility"""
        if not results:
            return []

        # Group by table
        table_groups = defaultdict(list)
        for result in results:
            table_name = result.get('table_name', '').lower()
            if table_name:
                table_groups[table_name].append(result)

        scored_results: List[Dict[str, Any]] = []

        # Process each table group
        for table_name, table_results in table_groups.items():
            total_score = sum(r.get('score', 0) for r in table_results)
            avg_score = total_score / len(table_results) if table_results else 0

            # Boost banking table scores
            if table_name.startswith('tbl'):
                avg_score *= 1.5

            # Add table summary
            table_summary = {
                'table_name': table_name,
                'column_name': 'TABLE_SUMMARY',
                'description': f'Table with {len(table_results)} relevant columns',
                'score': avg_score,
                'source_engine': 'aggregated',
                'metadata': {
                    'column_count': len(table_results),
                    'engines_found': list(set(r.get('source_engine', '') for r in table_results))
                }
            }
            scored_results.append(table_summary)

            # Add individual results with boosted scores
            for result in table_results:
                result['score'] = result.get('score', 0) * 1.2 if table_name.startswith('tbl') else result.get('score', 0)
                scored_results.append(result)

        # Deduplicate
        seen: Set[str] = set()
        unique_results: List[Dict[str, Any]] = []

        for result in sorted(scored_results, key=lambda x: x.get('score', 0), reverse=True):
            table_name = result.get('table_name', 'unknown')
            column_name = result.get('column_name', 'unknown')
            primary_key = f"{table_name}:{column_name}"
            fallback_key = f"{table_name}:*"

            if primary_key not in seen:
                seen.add(primary_key)
                seen.add(fallback_key)
                unique_results.append(result)
            elif fallback_key not in seen and column_name == 'unknown':
                seen.add(fallback_key)
                unique_results.append(result)

        # Guarantee some results if we had any input
        if results and not unique_results:
            self.logger.warning("All results filtered out - preserving top result to prevent fallback")
            top_result = max(results, key=lambda x: x.get('score', 0))
            unique_results = [top_result]

        # Mark as processed
        for result in unique_results:
            result['search_orchestrator_processed'] = True
            result['total_engines_searched'] = len(self.engines)
            result['deduplication_applied'] = True

        return unique_results[:50]

    def _update_performance_stats(self, duration: float, result_count: int, had_errors: bool) -> None:
        """EXISTING METHOD - unchanged"""
        total = self.performance_stats['total_searches']
        current_avg = self.performance_stats['average_response_time']
        self.performance_stats['average_response_time'] = (
            (current_avg * (total - 1) + duration) / total
        )

        if had_errors:
            self.performance_stats['total_errors'] += 1

    def _get_performance_summary(self) -> Dict[str, Any]:
        """EXISTING METHOD - unchanged"""
        return {
            'total_searches': self.performance_stats['total_searches'],
            'total_errors': self.performance_stats['total_errors'],
            'error_rate': (self.performance_stats['total_errors'] /
                          max(self.performance_stats['total_searches'], 1)),
            'average_response_time': self.performance_stats['average_response_time']
        }

    # NEW: Enhanced health check with comprehensive status
    def health_check(self) -> Dict[str, Any]:
        """ENHANCED: Health check with all enhancement status + data file validation"""
        # Get existing health check
        health_status = self._get_base_health_check()
        
        # Add comprehensive enhancement status
        health_status['accuracy_enhancements'] = {
            'schema_validation_enabled': self.enable_schema_validation,
            'intelligent_engine_selection_enabled': self.enable_intelligent_engine_selection,
            'dynamic_convergence_enabled': self.enable_dynamic_convergence,
            'xml_integration_enabled': self.enable_xml_integration,
            'safe_path_handling_enabled': True,
            'schema_manager_available': self.schema_manager is not None,
            'xml_manager_available': XML_SCHEMA_MANAGER_AVAILABLE,
            'data_files_status': {}
        }
        
        # Check data files status
        if self.schema_manager:
            try:
                schema_context = self.schema_manager.get_schema_context()
                health_status['accuracy_enhancements']['schema_context_loaded'] = schema_context['available']
                health_status['accuracy_enhancements']['schema_tables_count'] = len(schema_context['all_tables'])
                health_status['accuracy_enhancements']['schema_columns_count'] = len(schema_context['all_columns'])
                health_status['accuracy_enhancements']['xml_fields_count'] = len(schema_context.get('xml_fields', {}))
                health_status['accuracy_enhancements']['data_sources_loaded'] = schema_context.get('data_sources', {})
                
                # Individual data file status
                path_manager = self.schema_manager.path_manager
                for file_type in ['schema', 'xml_schema', 'tables', 'joins']:
                    file_path = path_manager.get_safe_path(file_type)
                    health_status['accuracy_enhancements']['data_files_status'][file_type] = {
                        'available': file_path is not None,
                        'path': str(file_path) if file_path else None
                    }
                    
            except Exception as e:
                health_status['accuracy_enhancements']['schema_error'] = str(e)
        
        return health_status

    def _get_base_health_check(self) -> Dict[str, Any]:
        """Get base health check (existing logic)"""
        health_status: Dict[str, Any] = {
            'overall_status': 'unknown',
            'engines': {},
            'async_client_manager': {},
            'diagnostics': {},
            'recommendations': []
        }

        healthy_engines = 0
        total_engines = len(self.engines)

        # Check each engine individually (existing logic)
        for method, engine in self.engines.items():
            engine_status: Dict[str, Any] = {'status': 'unknown', 'details': {}}

            try:
                if hasattr(engine, 'health_check'):
                    engine_status['status'] = 'healthy' if engine.health_check() else 'unhealthy'
                elif hasattr(engine, 'is_healthy'):
                    engine_status['status'] = 'healthy' if engine.is_healthy() else 'unhealthy'
                elif hasattr(engine, 'ping'):
                    engine_status['status'] = 'healthy' if engine.ping() else 'unhealthy'
                else:
                    # Test for search methods
                    search_methods = ['search_async', 'search', 'retrieve', 'retrieve_async']
                    has_method = any(hasattr(engine, m) for m in search_methods)
                    engine_status['status'] = 'healthy' if has_method else 'unknown'

                if engine_status['status'] == 'healthy':
                    healthy_engines += 1

            except Exception as e:
                engine_status['status'] = 'error'
                engine_status['error'] = str(e)

            health_status['engines'][method.value] = engine_status

        # AsyncClientManager health check (existing logic)
        if self.async_client_manager:
            try:
                client_status = self.async_client_manager.get_client_status()
                healthy_count_raw = self._safe_get(client_status, 'healthy_count', 0)
                total_clients_raw = self._safe_get(client_status, 'total_clients', 0)

                healthy_count = 0 if healthy_count_raw is None else int(healthy_count_raw)
                total_clients = 0 if total_clients_raw is None else int(total_clients_raw)

                health_status['async_client_manager'] = {
                    'available': True,
                    'healthy_clients': healthy_count,
                    'total_clients': total_clients,
                    'status': 'healthy' if healthy_count > 0 else 'unhealthy'
                }

            except Exception as e:
                health_status['async_client_manager'] = {
                    'available': False,
                    'error': str(e),
                    'status': 'error'
                }
        else:
            health_status['async_client_manager'] = {
                'available': False,
                'status': 'not_configured'
            }

               # Overall status determination (existing logic)
        engine_health_rate = healthy_engines / total_engines if total_engines > 0 else 0
        ai_healthy = health_status['async_client_manager'].get('status') == 'healthy'

        if engine_health_rate >= 0.8 and ai_healthy:
            health_status['overall_status'] = 'healthy'
        elif engine_health_rate >= 0.5:
            health_status['overall_status'] = 'degraded'
        else:
            health_status['overall_status'] = 'unhealthy'

        health_status['diagnostics'] = {
            'engine_health_rate': engine_health_rate,
            'healthy_engines': healthy_engines,
            'total_engines': total_engines,
            'ai_available': ai_healthy,
            'complete_enhancement_version': 'v2.5.0',
            'safe_path_handling': True,
            'comprehensive_data_integration': True
        }

        # Enhanced recommendations
        if engine_health_rate < 0.8:
            health_status['recommendations'].append("Some search engines are unhealthy - check engine configurations")
        if not ai_healthy:
            health_status['recommendations'].append("AsyncClientManager is not healthy - check AI client connections")
        if engine_health_rate == 0:
            health_status['recommendations'].append("CRITICAL: No search engines are healthy - system will not function")

        return health_status

    # NEW: Get comprehensive system status
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including all enhancements"""
        status = {
            'orchestrator_version': 'v2.5.0_complete_enhancement',
            'timestamp': time.time(),
            'enhancements_active': {
                'schema_validation': self.enable_schema_validation,
                'intelligent_engine_selection': self.enable_intelligent_engine_selection,
                'dynamic_convergence': self.enable_dynamic_convergence,
                'xml_integration': self.enable_xml_integration,
                'ai_enhancement': self.enable_ai_enhancement
            },
            'components_available': {
                'schema_manager': self.schema_manager is not None,
                'xml_manager': XML_SCHEMA_MANAGER_AVAILABLE,
                'async_client_manager': self.async_client_manager is not None
            },
            'performance_stats': self.performance_stats.copy()
        }
        
        if self.schema_manager:
            try:
                schema_context = self.schema_manager.get_schema_context()
                status['schema_status'] = {
                    'available': schema_context['available'],
                    'tables_loaded': len(schema_context['all_tables']),
                    'columns_loaded': len(schema_context['all_columns']),
                    'xml_fields_loaded': len(schema_context.get('xml_fields', {})),
                    'data_sources': schema_context.get('data_sources', {})
                }
            except Exception as e:
                status['schema_status'] = {'error': str(e)}
        
        return status

# FACTORY FUNCTIONS AND UTILITIES

def create_enhanced_search_orchestrator(
    engines: Union[Dict[SearchMethod, Any], List[Any]],
    async_client_manager: Optional['AsyncClientManager'] = None,
    enable_all_enhancements: bool = True,
    **kwargs
) -> SearchOrchestrator:
    """
    Factory function to create enhanced SearchOrchestrator with all improvements
    
    Args:
        engines: Search engines to use
        async_client_manager: Optional AsyncClientManager for AI features
        enable_all_enhancements: Enable all accuracy enhancements (default: True)
        **kwargs: Additional configuration options
    
    Returns:
        Fully configured enhanced SearchOrchestrator
    """
    if enable_all_enhancements:
        # Enable all enhancements by default
        config = {
            'enable_schema_validation': True,
            'enable_intelligent_engine_selection': True, 
            'enable_dynamic_convergence': True,
            'enable_xml_integration': True,
            'enable_ai_enhancement': True,
            **kwargs
        }
    else:
        # Use provided configuration
        config = kwargs
    
    orchestrator = SearchOrchestrator(
        engines=engines,
        async_client_manager=async_client_manager,
        **config
    )
    
    logger.info(f"Enhanced SearchOrchestrator created with config: {config}")
    return orchestrator

def create_compatible_search_orchestrator(
    engines: Union[Dict[SearchMethod, Any], List[Any]],
    async_client_manager: Optional['AsyncClientManager'] = None,
    **kwargs
) -> SearchOrchestrator:
    """
    Factory function to create SearchOrchestrator with backward compatibility mode
    (All enhancements disabled for maximum compatibility)
    
    Args:
        engines: Search engines to use
        async_client_manager: Optional AsyncClientManager for AI features
        **kwargs: Additional configuration options
    
    Returns:
        SearchOrchestrator with enhancements disabled for compatibility
    """
    config = {
        'enable_schema_validation': False,
        'enable_intelligent_engine_selection': False,
        'enable_dynamic_convergence': False,
        'enable_xml_integration': False,
        **kwargs
    }
    
    orchestrator = SearchOrchestrator(
        engines=engines,
        async_client_manager=async_client_manager,
        **config
    )
    
    logger.info("Compatible SearchOrchestrator created (enhancements disabled)")
    return orchestrator

# CONFIGURATION VALIDATION UTILITIES

def validate_data_files_availability() -> Dict[str, Any]:
    """
    Validate availability of data files for enhanced features
    
    Returns:
        Dictionary with file availability status
    """
    path_manager = SafeDataPathManager()
    
    validation_result = {
        'timestamp': time.time(),
        'base_path': str(path_manager.base_path),
        'files': {},
        'recommendations': []
    }
    
    file_types = ['schema', 'xml_schema', 'tables', 'joins']
    
    for file_type in file_types:
        file_path = path_manager.get_safe_path(file_type)
        data = path_manager.load_json_safe(file_type)
        
        validation_result['files'][file_type] = {
            'available': file_path is not None,
            'path': str(file_path) if file_path else None,
            'loadable': data is not None,
            'record_count': len(data) if isinstance(data, list) else (
                len(data) if isinstance(data, dict) else 0
            ) if data else 0
        }
        
        if not file_path:
            validation_result['recommendations'].append(f"Consider creating {file_type}.json for enhanced features")
        elif not data:
            validation_result['recommendations'].append(f"Check {file_type}.json format - failed to load")
    
    # Overall assessment
    available_files = sum(1 for f in validation_result['files'].values() if f['available'])
    validation_result['summary'] = {
        'total_files': len(file_types),
        'available_files': available_files,
        'availability_rate': available_files / len(file_types),
        'enhancement_readiness': 'ready' if available_files >= 2 else 'partial' if available_files >= 1 else 'not_ready'
    }
    
    return validation_result

# SYSTEM INTEGRATION HELPERS

class SearchOrchestratorIntegrator:
    """Helper class for integrating enhanced SearchOrchestrator into existing systems"""
    
    @staticmethod
    def integrate_with_existing_system(
        existing_orchestrator: Any,
        engines: Union[Dict[SearchMethod, Any], List[Any]],
        async_client_manager: Optional['AsyncClientManager'] = None,
        migration_mode: str = 'enhanced'
    ) -> SearchOrchestrator:
        """
        Integrate enhanced SearchOrchestrator with existing system
        
        Args:
            existing_orchestrator: Existing orchestrator instance (for config extraction)
            engines: Search engines to use
            async_client_manager: Optional AsyncClientManager
            migration_mode: 'enhanced', 'compatible', or 'hybrid'
        
        Returns:
            New SearchOrchestrator configured based on migration mode
        """
        # Extract configuration from existing orchestrator if available
        config = {}
        
        if hasattr(existing_orchestrator, 'max_iterations'):
            config['max_iterations'] = existing_orchestrator.max_iterations
        if hasattr(existing_orchestrator, 'convergence_threshold'):
            config['convergence_threshold'] = existing_orchestrator.convergence_threshold
        if hasattr(existing_orchestrator, 'enable_ai_enhancement'):
            config['enable_ai_enhancement'] = existing_orchestrator.enable_ai_enhancement
        
        # Apply migration mode settings
        if migration_mode == 'enhanced':
            # Enable all enhancements
            config.update({
                'enable_schema_validation': True,
                'enable_intelligent_engine_selection': True,
                'enable_dynamic_convergence': True,
                'enable_xml_integration': True
            })
        elif migration_mode == 'compatible':
            # Disable enhancements for compatibility
            config.update({
                'enable_schema_validation': False,
                'enable_intelligent_engine_selection': False,
                'enable_dynamic_convergence': False,
                'enable_xml_integration': False
            })
        elif migration_mode == 'hybrid':
            # Enable only safe enhancements
            config.update({
                'enable_schema_validation': True,
                'enable_intelligent_engine_selection': False,  # May change engine behavior
                'enable_dynamic_convergence': False,  # May change iteration behavior
                'enable_xml_integration': True
            })
        
        new_orchestrator = SearchOrchestrator(
            engines=engines,
            async_client_manager=async_client_manager,
            **config
        )
        
        logger.info(f"SearchOrchestrator integrated with mode: {migration_mode}")
        return new_orchestrator

# XML INTEGRATION DETAILS

def get_xml_integration_info() -> Dict[str, Any]:
    """
    Get information about XML integration capabilities
    
    Returns:
        Dictionary with XML integration information
    """
    xml_info = {
        'xml_schema_manager_available': XML_SCHEMA_MANAGER_AVAILABLE,
        'xml_integration_files': [
            'agent/schema_searcher/managers/xml_schema_manager.py',
            'agent/schema_searcher/loaders/xml_loader.py'
        ],
        'xml_data_files': [
            'data/metadata/xml_schema.json'
        ],
        'xml_integration_features': [
            'XML field detection and tagging',
            'XPath information for XML columns',
            'XML-aware result scoring and ranking',
            'Integration with XMLSchemaManager for advanced XML operations'
        ]
    }
    
    if XML_SCHEMA_MANAGER_AVAILABLE:
        xml_info['status'] = 'available'
        xml_info['description'] = 'Full XML integration available with XMLSchemaManager'
    else:
        xml_info['status'] = 'unavailable'
        xml_info['description'] = 'XML integration requires XMLSchemaManager component'
        xml_info['setup_instructions'] = [
            '1. Ensure agent.schema_searcher.managers.xml_schema_manager is available',
            '2. Create data/metadata/xml_schema.json with your XML schema information',
            '3. Enable XML integration in SearchOrchestrator configuration'
        ]
    
    return xml_info

# EXPORT ALL COMPONENTS

__all__ = [
    # Main classes
    'SearchOrchestrator',
    'SearchResult',
    'SearchStrategy',
    'SearchMethod',
    'QueryComplexity',
    'SearchError',
    'SearchConfigurationError', 
    'SearchExecutionError',
    
    # Enhanced components
    'SafeDataPathManager',
    'EnhancedSchemaContextManager',
    'SearchOrchestratorIntegrator',
    
    # Factory functions
    'create_enhanced_search_orchestrator',
    'create_compatible_search_orchestrator',
    
    # Utility functions
    'validate_data_files_availability',
    'get_xml_integration_info'
]

# INITIALIZATION AND FINAL SETUP

def initialize_enhanced_search_system() -> Dict[str, Any]:
    """
    Initialize and validate the enhanced search system
    
    Returns:
        Initialization status and recommendations
    """
    init_status = {
        'timestamp': time.time(),
        'version': 'v2.5.0_complete_enhancement',
        'components': {},
        'data_files': {},
        'recommendations': [],
        'ready_for_production': True
    }
    
    # Check component availability
    init_status['components'] = {
        'async_client_manager': ASYNC_CLIENT_MANAGER_AVAILABLE,
        'xml_schema_manager': XML_SCHEMA_MANAGER_AVAILABLE,
        'safe_path_manager': True,  # Always available
        'enhanced_schema_manager': True  # Always available
    }
    
    # Validate data files
    init_status['data_files'] = validate_data_files_availability()
    
    # Generate recommendations
    if not ASYNC_CLIENT_MANAGER_AVAILABLE:
        init_status['recommendations'].append("Consider enabling AsyncClientManager for AI features")
    
    if not XML_SCHEMA_MANAGER_AVAILABLE:
        init_status['recommendations'].append("Consider enabling XMLSchemaManager for XML integration")
    
    if init_status['data_files']['summary']['availability_rate'] < 0.5:
        init_status['recommendations'].append("Create more data files for optimal performance")
        init_status['ready_for_production'] = False
    
    # Final status
    if init_status['ready_for_production'] and not init_status['recommendations']:
        init_status['status'] = 'fully_ready'
    elif init_status['ready_for_production']:
        init_status['status'] = 'ready_with_recommendations'
    else:
        init_status['status'] = 'needs_setup'
    
    return init_status

# LOG SYSTEM INITIALIZATION
if __name__ == "__main__":
    # System validation when run directly
    print("Enhanced SearchOrchestrator System Validation")
    print("=" * 60)
    
    init_result = initialize_enhanced_search_system()
    
    print(f"Status: {init_result['status']}")
    print(f"Version: {init_result['version']}")
    print(f"Ready for production: {init_result['ready_for_production']}")
    
    print("\nComponent Status:")
    for component, available in init_result['components'].items():
        status = "✅" if available else "❌"
        print(f"  {status} {component}")
    
    print(f"\nData Files: {init_result['data_files']['summary']['available_files']}/{init_result['data_files']['summary']['total_files']} available")
    
    if init_result['recommendations']:
        print("\nRecommendations:")
        for rec in init_result['recommendations']:
            print(f"  • {rec}")
    
    print("\n" + "=" * 60)
    print("Enhanced SearchOrchestrator ready for use!")
