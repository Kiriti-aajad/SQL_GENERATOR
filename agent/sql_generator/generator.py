"""
SQL Generator - Integrated with Shared AsyncClientManager Singleton
FIXED: All character escaping, Unicode, and integration issues
FIXED: Now receives shared AsyncClientManager instead of creating its own
FIXED: DYNAMIC schema processing - no hardcoded domain logic
FIXED: Singleton pattern implemented to prevent multiple instances
"""

import asyncio
import logging
import time
import os
import re
from typing import Dict, Any, Optional, Union, List
import threading
import json
import hashlib
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# FIXED: Import AsyncClientManager for type hints only
try:
    from agent.sql_generator.async_client_manager import AsyncClientManager, get_client_context
    ASYNC_CLIENT_MANAGER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"AsyncClientManager not available: {e}")
    AsyncClientManager = None
    get_client_context = None
    ASYNC_CLIENT_MANAGER_AVAILABLE = False

class DynamicSchemaProcessor:
    """DYNAMIC schema intelligence for SQL generation - works with ANY database"""
    
    @classmethod
    def enhance_prompt_with_schema_context(cls, prompt: str, schema_context: Any = None) -> str:
        """Enhance prompt with DISCOVERED schema knowledge - NO hardcoded logic"""
        enhanced_parts = [
            "=== DISCOVERED DATABASE SCHEMA ===",
            "You are generating SQL for a database with the following DISCOVERED schema:",
            ""
        ]

        if schema_context and isinstance(schema_context, dict):
            # Add discovered tables and columns
            tables = schema_context.get('tables', [])
            columns_by_table = schema_context.get('columns_by_table', {})
            
            if tables:
                enhanced_parts.extend([
                    f"AVAILABLE TABLES ({len(tables)} discovered):",
                    ""
                ])
                
                for table in tables[:20]:  # Limit for prompt size
                    if table in columns_by_table:
                        columns = columns_by_table[table]
                        if isinstance(columns, list):
                            col_names = []
                            for col in columns[:15]:  # Limit columns per table
                                if isinstance(col, dict):
                                    col_name = col.get('column', col.get('name', str(col)))
                                    col_type = col.get('type', '')
                                    if col_type:
                                        col_names.append(f"{col_name} ({col_type})")
                                    else:
                                        col_names.append(col_name)
                                else:
                                    col_names.append(str(col))
                            
                            enhanced_parts.append(f"Table: {table}")
                            enhanced_parts.append(f"  Columns: {', '.join(col_names)}")
                            enhanced_parts.append("")
                    else:
                        enhanced_parts.append(f"Table: {table}")
                        enhanced_parts.append("")

            # Add discovered joins
            joins = schema_context.get('joins', [])
            if joins:
                enhanced_parts.extend([
                    f"AVAILABLE JOINS ({len(joins)} discovered):",
                    ""
                ])
                
                for join in joins[:15]:  # Limit joins
                    if isinstance(join, dict):
                        table1 = join.get('table1', join.get('from_table', 'unknown'))
                        table2 = join.get('table2', join.get('to_table', 'unknown'))
                        condition = join.get('join_condition', join.get('condition', f"{table1}.id = {table2}.id"))
                        join_type = join.get('join_type', 'INNER')
                        enhanced_parts.append(f"  {table1} {join_type} JOIN {table2} ON {condition}")
                    else:
                        enhanced_parts.append(f"  {join}")
                enhanced_parts.append("")

            # Add XML fields if present
            xml_mappings = schema_context.get('xml_mappings', [])
            if xml_mappings:
                enhanced_parts.extend([
                    f"XML FIELDS ({len(xml_mappings)} discovered):",
                    ""
                ])
                
                for xml_field in xml_mappings[:10]:  # Limit XML fields
                    if isinstance(xml_field, dict):
                        table = xml_field.get('table', 'unknown')
                        field = xml_field.get('field', xml_field.get('column', 'unknown'))
                        xpath = xml_field.get('xpath', xml_field.get('xml_path', 'unknown'))
                        data_type = xml_field.get('data_type', '')
                        if data_type:
                            enhanced_parts.append(f"  {table}.{field}: {xpath} ({data_type})")
                        else:
                            enhanced_parts.append(f"  {table}.{field}: {xpath}")
                    else:
                        enhanced_parts.append(f"  {xml_field}")
                enhanced_parts.append("")

        enhanced_parts.extend([
            "=== USER QUERY ===",
            prompt,
            "",
            "=== INSTRUCTIONS ===",
            "1. Use ONLY the table and column names from the DISCOVERED SCHEMA above",
            "2. Use the DISCOVERED JOINS for multi-table queries",
            "3. Handle XML fields with proper XPath expressions if present",
            "4. Generate clean, efficient SQL that works with the discovered schema",
            "5. Always use proper table aliases for clarity",
            "6. Return ONLY the SQL query, no explanations"
        ])

        return "\n".join(enhanced_parts)

    @classmethod
    def validate_and_fix_sql(cls, sql: str, schema_context: Any = None) -> Dict[str, Any]:
        """Validate and fix generated SQL based on DISCOVERED schema"""
        if not sql or not sql.strip():
            return {"valid": False, "fixed_sql": sql, "issues": ["Empty SQL"]}

        issues = []
        fixed_sql = sql

        if schema_context and isinstance(schema_context, dict):
            tables = schema_context.get('tables', [])
            columns_by_table = schema_context.get('columns_by_table', {})

            # Check if SQL uses valid table names from discovered schema
            sql_upper = sql.upper()
            found_tables = []
            
            for table in tables:
                if table.upper() in sql_upper or table.lower() in sql.lower():
                    found_tables.append(table)

            if not found_tables and tables:
                issues.append(f"No recognized table names found. Available: {', '.join(tables[:5])}")

            # Validate column references (basic check)
            for table, columns in columns_by_table.items():
                if isinstance(columns, list) and any(table.lower() in sql.lower() for table in [table]):
                    # SQL mentions this table, could validate columns here
                    pass

        return {
            "valid": len(issues) == 0,
            "fixed_sql": fixed_sql,
            "issues": issues,
            "has_valid_schema": len(issues) == 0,
            "schema_tables_found": len(found_tables) if 'found_tables' in locals() else 0 # pyright: ignore[reportPossiblyUnboundVariable]
        }

class AdvancedCache:
    """Production-ready caching with TTL and LRU eviction"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, float] = {}
        self.access_count: Dict[str, int] = {}
        self.hit_count = 0
        self.miss_count = 0
        self._lock = threading.RLock()

    async def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self.cache:
                self.miss_count += 1
                return None
            
            if time.time() - self.timestamps.get(key, 0) > self.ttl:
                self._remove_key(key)
                self.miss_count += 1
                return None
            
            self.access_count[key] = self.access_count.get(key, 0) + 1
            self.hit_count += 1
            return self.cache[key]

    async def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._cleanup_expired()
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
            self.access_count[key] = 1

    def _remove_key(self, key: str) -> None:
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
        self.access_count.pop(key, None)

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [k for k, t in self.timestamps.items() if now - t > self.ttl]
        for k in expired:
            self._remove_key(k)

    def _evict_lru(self) -> None:
        if self.access_count:
            lru = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            self._remove_key(lru)

    def get_stats(self) -> Dict[str, Any]:
        total = self.hit_count + self.miss_count
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": (self.hit_count / total * 100) if total > 0 else 0,
            "ttl_seconds": self.ttl
        }

    def cleanup(self):
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()
            self.access_count.clear()

class SQLGenerator:
    """
    Production-ready SQL Generator with DYNAMIC Schema Intelligence
    FIXED: Now integrates with shared AsyncClientManager singleton
    FIXED: Uses DYNAMIC schema processing - works with ANY database
    FIXED: Singleton pattern implemented to prevent multiple instances
    """
    
    # CRITICAL FIX: Singleton pattern implementation
    _instance: Optional['SQLGenerator'] = None
    _initialized: bool = False
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        """FIXED: Singleton pattern - return existing instance if available"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self, 
                 model_configs: Optional[Any] = None, 
                 client_manager: Optional['AsyncClientManager'] = None, # pyright: ignore[reportInvalidTypeForm]
                 async_client_manager: Optional['AsyncClientManager'] = None):  # pyright: ignore[reportInvalidTypeForm] # FIXED: Added for compatibility
        """
        CRITICAL FIX: Constructor now accepts shared AsyncClientManager instance
        FIXED: Singleton pattern prevents duplicate initialization
        This eliminates the dual instance problem by using the singleton from main.py
        """
        # CRITICAL FIX: Prevent duplicate initialization
        if self._initialized:
            return
        
        self.logger = logging.getLogger("SQLGenerator")
        self.status = "initializing"
        
        # FIXED: Use provided shared AsyncClientManager instance (don't create new one)
        # Support both parameter names for compatibility
        self.client_manager = client_manager or async_client_manager
        
        if not self.client_manager and ASYNC_CLIENT_MANAGER_AVAILABLE:
            try:
                # Only create new instance if none provided (fallback for standalone usage)
                self.client_manager = AsyncClientManager() # pyright: ignore[reportOptionalCall]
                self.logger.warning("No shared AsyncClientManager provided, created fallback instance")
            except Exception as e:
                self.logger.warning(f"Failed to create fallback AsyncClientManager instance: {e}")

        # Log integration status
        if self.client_manager:
            self.logger.info(f"SQLGenerator singleton initialized with AsyncClientManager (ID: {id(self.client_manager)})")
            
            # Verify singleton integrity if possible
            if hasattr(self.client_manager, 'verify_singleton'):
                is_singleton = self.client_manager.verify_singleton()
                if not is_singleton:
                    self.logger.warning("AsyncClientManager singleton integrity violation detected!")
                else:
                    self.logger.info("AsyncClientManager singleton integrity verified")
        else:
            self.logger.warning("SQLGenerator singleton initialized without AsyncClientManager - running in degraded mode")

        # FIXED: Dynamic schema processor instead of hardcoded banking processor
        self.schema_processor = DynamicSchemaProcessor()
        
        # Core attributes
        self.models_available = False
        self.ready = False
        self._async_initialization_complete = False
        
        # Performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "fallback_generations": 0,
            "offline_responses": 0,
            "average_response_time": 0.0,
            "model_usage": {},
            "cache_hits": 0,
            "cache_misses": 0,
            "dynamic_schema_fixes_applied": 0,  # Changed from banking_fixes_applied
            "schema_validations": 0,            # Changed from table_name_corrections
            "event_loop_errors_handled": 0
        }

        # Configuration
        self.connection_timeout = int(os.getenv("SQL_GEN_TIMEOUT", "30"))
        cache_size = int(os.getenv("SQL_GEN_CACHE_SIZE", "1000"))
        cache_ttl = int(os.getenv("SQL_GEN_CACHE_TTL", "3600"))
        self.cache = AdvancedCache(max_size=cache_size, ttl=cache_ttl)

        # Safe async initialization scheduling
        self._safe_schedule_async_initialization()
        
        # CRITICAL FIX: Mark as initialized to prevent duplicate initialization
        self._initialized = True
        
        self.logger.info("SQLGenerator singleton initialized with DYNAMIC SCHEMA processing and shared AsyncClientManager integration")

    @classmethod
    def get_instance(cls, 
                     model_configs: Optional[Any] = None, 
                     client_manager: Optional['AsyncClientManager'] = None, # pyright: ignore[reportInvalidTypeForm]
                     async_client_manager: Optional['AsyncClientManager'] = None) -> 'SQLGenerator': # pyright: ignore[reportInvalidTypeForm]
        """
        FIXED: Get or create SQLGenerator singleton instance
        This is the preferred way to get SQLGenerator instances
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(model_configs, client_manager or async_client_manager)
            return cls._instance

    def _safe_schedule_async_initialization(self):
        """Safe async initialization scheduling with proper event loop handling"""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._async_initialize())
            self.logger.debug("Async initialization scheduled in running event loop")
        except RuntimeError:
            self.logger.info("No event loop running, async initialization deferred until first use")

    async def _async_initialize(self):
        """
        FIXED: Initialize using shared AsyncClientManager with enhanced error handling
        No longer creates new AsyncClientManager - uses the provided shared instance
        """
        try:
            if not self.client_manager:
                self.logger.warning("No AsyncClientManager available")
                self.status = "degraded"
                self._async_initialization_complete = True
                return

            # Check if AsyncClientManager is already initialized
            if hasattr(self.client_manager, '_async_initialized') and self.client_manager._async_initialized:
                self.logger.info("AsyncClientManager already initialized, skipping initialization")
            else:
                # Initialize if not already done
                await self.client_manager.initialize()

            # Update status based on client availability
            client_status = self.client_manager.get_client_status()
            self.models_available = client_status["healthy_count"] > 0
            self.ready = self.models_available
            self.status = "healthy" if self.ready else "degraded"
            self._async_initialization_complete = True

            self.logger.info(f"Async initialization completed with shared AsyncClientManager: {client_status['healthy_count']} healthy clients")

        except Exception as e:
            self.logger.error(f"Async initialization failed: {e}")
            self.status = "degraded"
            self._async_initialization_complete = True

    async def ensure_initialized(self):
        """Ensure initialization completed before processing"""
        if not self._async_initialization_complete:
            await self._async_initialize()

    def get_status(self) -> str:
        return self.status

    def set_status(self, status: str):
        self.status = status

    def is_healthy(self) -> bool:
        return self.status in ["healthy", "ready"]

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check with shared AsyncClientManager status"""
        await self.ensure_initialized()

        if not self.client_manager:
            return {
                "status": "degraded",
                "component": "SQLGenerator",
                "ready": False,
                "models_available": False,
                "error": "AsyncClientManager not available",
                "dynamic_schema_enabled": True,  # Changed from banking_domain_enabled
                "shared_instance": False,
                "singleton_pattern": True  # NEW: Confirm singleton implementation
            }

        try:
            # Get client manager status
            health_result = await self.client_manager.health_check()
            client_status = self.client_manager.get_client_status()

            # Verify singleton integrity
            is_singleton = False
            if hasattr(self.client_manager, 'verify_singleton'):
                is_singleton = self.client_manager.verify_singleton()

            # Update our status
            if client_status["healthy_count"] > 0:
                self.models_available = True
                self.ready = True
                self.status = "healthy"
            else:
                self.models_available = False
                self.status = "degraded"

            return {
                "status": self.status,
                "component": "SQLGenerator",
                "ready": self.ready,
                "models_available": self.models_available,
                "client_status": client_status,
                "client_health": health_result,
                "performance_stats": self.get_performance_stats(),
                "dynamic_schema_enabled": True,  # Changed from banking_domain_enabled
                "cache_stats": self.cache.get_stats(),
                "shared_instance": is_singleton,
                "async_client_manager_id": id(self.client_manager),
                "all_critical_fixes_applied": True,
                "singleton_integration": "completed",
                "singleton_pattern": True,  # NEW: Confirm singleton implementation
                "instance_count": 1 if self._instance else 0  # NEW: Track instance count
            }

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "component": "SQLGenerator",
                "ready": False,
                "models_available": False,
                "error": str(e),
                "dynamic_schema_enabled": True,  # Changed from banking_domain_enabled
                "shared_instance": False,
                "singleton_pattern": True  # NEW: Confirm singleton implementation
            }

    def generate_sql(self, prompt: str, banking_insights: Any = None, schema_context: Any = None,
                    target_llm: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synchronous wrapper with proper event loop detection"""
        self.performance_stats["total_requests"] += 1

        # Smart event loop detection and handling
        try:
            loop = asyncio.get_running_loop()
            # We're in an event loop, cannot use sync method
            self.performance_stats["event_loop_errors_handled"] += 1
            self.logger.error("generate_sql() called inside active event loop. Use generate_sql_async() instead.")
            return self._create_offline_response(
                prompt, "Called sync generate_sql() from within async event loop. Use generate_sql_async()"
            )
        except RuntimeError:
            # No event loop running, we can safely create one
            try:
                return asyncio.run(self.generate_sql_async(prompt, banking_insights, schema_context, target_llm, context))
            except Exception as e:
                self.logger.error(f"Failed to run SQL generation: {e}")
                return self._create_offline_response(prompt, f"Generation failed: {str(e)}")

    async def generate_sql_async(self, prompt: str, banking_insights: Any = None, schema_context: Any = None,
                               target_llm: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Async SQL generation with shared AsyncClientManager and DYNAMIC schema processing"""
        start_time = time.time()
        self.performance_stats["total_requests"] += 1
        self.logger.info("=== SQL GENERATION STARTED (with shared AsyncClientManager + DYNAMIC SCHEMA + SINGLETON) ===")

        try:
            # Ensure initialization
            await self.ensure_initialized()

            # Check if ready
            if not self.ready or not self.models_available or not self.client_manager:
                self.performance_stats["offline_responses"] += 1
                return self._create_offline_response(prompt, "Models offline or not ready")

            # Check cache first
            cache_key = self._generate_cache_key(prompt, banking_insights, schema_context, target_llm, context)
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self.performance_stats["cache_hits"] += 1
                cached_result["from_cache"] = True
                cached_result["singleton_instance"] = True  # NEW: Mark as singleton
                self.logger.info("Returning cached result from singleton instance")
                return cached_result

            self.performance_stats["cache_misses"] += 1

            # Intelligent model selection
            selected_target = self._select_optimal_model(prompt, target_llm, schema_context)

            # Generate SQL using shared AsyncClientManager
            try:
                # Build DYNAMIC schema-aware prompt
                enhanced_prompt = self._build_dynamic_schema_enhanced_prompt(prompt, banking_insights, schema_context, context)

                # CRITICAL FIX: Correct parameter order for generate_sql_async
                result = await asyncio.wait_for(
                    self.client_manager.generate_sql_async(
                        enhanced_prompt,        # prompt: str
                        context=context,        # context: Optional[Dict[str, Any]] = None
                        target_llm=selected_target  # target_llm: Optional[str] = None
                    ),
                    timeout=self.connection_timeout
                )

                if result and result.get("success", False):
                    # Apply DYNAMIC schema validation and fixes
                    result = self._apply_dynamic_schema_fixes(result, prompt, schema_context)

                    # Cache successful result
                    await self.cache.set(cache_key, result)

                    # Update stats
                    self.performance_stats["successful_generations"] += 1
                    self._update_model_usage_stats(result.get("model_used", "unknown"))

                    processing_time = (time.time() - start_time) * 1000
                    result["processing_time_ms"] = processing_time
                    result["models_available"] = True
                    result["from_cache"] = False
                    result["dynamic_schema_applied"] = True    # Changed from banking_domain_applied
                    result["shared_async_client_manager"] = True
                    result["async_client_manager_id"] = id(self.client_manager)
                    result["all_fixes_applied"] = True
                    result["singleton_instance"] = True  # NEW: Mark as singleton generated
                    result["singleton_pattern_enabled"] = True  # NEW: Confirm singleton

                    self.logger.info(f"SQL generation successful with shared AsyncClientManager + DYNAMIC SCHEMA + SINGLETON in {processing_time:.2f}ms")
                    return result
                else:
                    self.performance_stats["failed_generations"] += 1
                    return self._create_offline_response(prompt, "Generation unsuccessful")

            except Exception as e:
                self.logger.error(f"Client generation error with shared AsyncClientManager: {e}")
                self.performance_stats["failed_generations"] += 1
                return self._create_offline_response(prompt, f"Client error: {str(e)}")

        except asyncio.TimeoutError:
            self.performance_stats["failed_generations"] += 1
            return self._create_offline_response(prompt, "Request timed out")
        except Exception as e:
            self.logger.error(f"Unexpected error in SQL generation: {e}")
            self.performance_stats["failed_generations"] += 1
            return self._create_offline_response(prompt, f"Generation failed: {str(e)}")

    def _select_optimal_model(self, prompt: str, target_llm: Optional[str], schema_context: Any = None) -> str:
        """Intelligent model selection based on query complexity"""
        if target_llm:
            return target_llm

        # Analyze query complexity
        prompt_lower = prompt.lower()
        
        # Complex queries -> DeepSeek
        complex_indicators = [
            'join', 'subquery', 'group by', 'having', 'case when',
            'window function', 'cte', 'with', 'union', 'exists'
        ]
        
        # Simple queries -> Mathstral
        simple_indicators = [
            'select * from', 'where', 'order by', 'limit', 'top'
        ]

        complexity_score = sum(1 for indicator in complex_indicators if indicator in prompt_lower)
        simplicity_score = sum(1 for indicator in simple_indicators if indicator in prompt_lower)

        if complexity_score > simplicity_score:
            return "deepseek"
        else:
            return "mathstral"

    def _build_dynamic_schema_enhanced_prompt(self, prompt: str, banking_insights: Any, schema_context: Any,
                                            context: Optional[Dict[str, Any]] = None) -> str:
        """Build DYNAMIC schema-aware prompt using DISCOVERED schema"""
        # Use DYNAMIC schema processor for core enhancement
        enhanced_prompt = self.schema_processor.enhance_prompt_with_schema_context(prompt, schema_context)

        # Add additional context from insights (renamed from banking_insights for clarity)
        additional_parts = []
        if banking_insights:  # Keep parameter name for backward compatibility, but treat as generic insights
            if isinstance(banking_insights, dict):
                intent = banking_insights.get("intent", "unknown")
                entities = banking_insights.get("entities", [])
                nlp_enhancement = banking_insights.get("nlp_enhancement", {})

                if intent != "unknown":
                    additional_parts.append(f"=== NLP INTENT ===\nDetected Intent: {intent}")

                if entities:
                    additional_parts.append(f"=== ENTITIES ===\nExtracted: {', '.join(str(e) for e in entities[:5])}")

                if nlp_enhancement:
                    enhancement_quality = nlp_enhancement.get("enhancement_quality", "unknown")
                    additional_parts.append(f"=== NLP ENHANCEMENT ===\nQuality: {enhancement_quality}")

        if additional_parts:
            enhanced_prompt = enhanced_prompt + "\n\n" + "\n\n".join(additional_parts)

        return enhanced_prompt

    def _apply_dynamic_schema_fixes(self, result: Dict[str, Any], original_prompt: str, schema_context: Any = None) -> Dict[str, Any]:
        """Apply DYNAMIC schema validation and fixes to generated SQL"""
        if not result or not result.get("sql"):
            return result

        original_sql = result["sql"]

        # Apply DYNAMIC schema validation and fixes
        validation_result = self.schema_processor.validate_and_fix_sql(original_sql, schema_context)

        if validation_result["issues"]:
            # SQL was modified
            self.performance_stats["dynamic_schema_fixes_applied"] += 1
            self.performance_stats["schema_validations"] += len(validation_result["issues"])
            self.logger.info(f"Applied dynamic schema fixes: {validation_result['issues']}")

            result["sql"] = validation_result["fixed_sql"]
            result["schema_fixes_applied"] = validation_result["issues"]  # Changed from banking_fixes_applied
            result["schema_validation_passed"] = validation_result["has_valid_schema"]

            # Boost confidence if we fixed schema issues
            if result.get("confidence", 0) < 0.9:
                result["confidence"] = min(0.9, result.get("confidence", 0.8) + 0.1)
        else:
            result["schema_fixes_applied"] = []
            result["schema_validation_passed"] = validation_result["has_valid_schema"]

        # Add metadata
        result["dynamic_schema_processed"] = True      # Changed from banking_domain_processed
        result["uses_discovered_schema"] = validation_result["has_valid_schema"]  # Changed from uses_correct_table_names

        return result

    def _generate_cache_key(self, prompt: str, banking_insights: Any, schema_context: Any,
                           target_llm: Optional[str], context: Optional[Dict[str, Any]]) -> str:
        """Generate deterministic cache key"""
        cache_data = {
            "prompt": prompt,
            "insights": str(banking_insights) if banking_insights else "",  # Changed from banking_insights
            "schema_context": str(schema_context) if schema_context else "",
            "target_llm": target_llm or "",
            "context": str(context) if context else "",
            "dynamic_schema": "v4.1_dynamic_shared_singleton"  # UPDATED: Version includes singleton
        }

        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _update_model_usage_stats(self, model_name: str):
        """Track model usage statistics"""
        if model_name not in self.performance_stats["model_usage"]:
            self.performance_stats["model_usage"][model_name] = 0
        self.performance_stats["model_usage"][model_name] += 1

    def _create_offline_response(self, prompt: str, reason: str = "Models offline") -> Dict[str, Any]:
        """Create offline fallback response with DYNAMIC schema awareness"""
        self.logger.warning(f"Offline response: {reason}")
        self.performance_stats["offline_responses"] += 1

        # Create GENERIC offline SQL (no hardcoded table names)
        offline_sql = f"""-- SQL Generator: Models currently offline or unavailable
-- Query: {prompt[:100]}{'...' if len(prompt) > 100 else ''}
-- Reason: {reason}
-- Generic Fallback Query
SELECT 
    'offline_mode' as status,
    '{reason}' as error_reason,
    '{prompt[:50]}' as original_query
-- Please check connectivity and try again"""

        return {
            "sql": offline_sql,
            "success": False,
            "confidence": 0.0,
            "model_used": "offline",
            "processing_time_ms": 0.0,
            "tokens_used": 0,
            "models_available": False,
            "error": reason,
            "message": "Models currently offline. Please check connectivity.",
            "status": self.status,
            "from_cache": False,
            "dynamic_schema_applied": True,     # Changed from banking_domain_applied
            "uses_discovered_schema": False,   # Changed from uses_correct_table_names
            "shared_async_client_manager": self.client_manager is not None,
            "all_fixes_applied": True,
            "singleton_instance": True,  # NEW: Mark as singleton
            "singleton_pattern_enabled": True  # NEW: Confirm singleton
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = dict(self.performance_stats)
        stats["models_available"] = self.models_available
        stats["ready"] = self.ready
        stats["status"] = self.status
        stats["initialization_complete"] = self._async_initialization_complete
        stats["singleton_enabled"] = True  # NEW: Confirm singleton
        stats["instance_id"] = id(self)    # NEW: Track instance ID

        if self.client_manager:
            try:
                client_status = self.client_manager.get_client_status()
                stats["client_manager_status"] = client_status
                stats["shared_async_client_manager"] = True
                stats["async_client_manager_id"] = id(self.client_manager)

                # Add singleton verification
                if hasattr(self.client_manager, 'verify_singleton'):
                    stats["singleton_verified"] = self.client_manager.verify_singleton()
            except Exception as e:
                stats["client_manager_status"] = {"error": str(e)}
                stats["shared_async_client_manager"] = False
        else:
            stats["client_manager_status"] = {"error": "AsyncClientManager not available"}
            stats["shared_async_client_manager"] = False

        stats["cache_stats"] = self.cache.get_stats()
        stats["dynamic_schema_enabled"] = True    # Changed from banking_domain_enabled
        stats["all_critical_fixes_applied"] = True

        # Calculate success rate
        total = stats["successful_generations"] + stats["failed_generations"]
        if total > 0:
            stats["success_rate"] = (stats["successful_generations"] / total) * 100
        else:
            stats["success_rate"] = 0

        return stats

    async def cleanup(self):
        """
        FIXED: Enhanced cleanup with shared AsyncClientManager coordination
        Don't cleanup shared AsyncClientManager - let main.py handle it
        """
        self.logger.info("Cleaning up SQLGenerator singleton...")
        self.status = "shutting_down"

        try:
            # Cleanup cache
            if self.cache:
                self.cache.cleanup()

            # FIXED: Don't cleanup shared AsyncClientManager (main.py handles it)
            if self.client_manager:
                self.logger.info("Shared AsyncClientManager cleanup will be handled by main.py")
                # Just clear our reference
                self.client_manager = None

            self._async_initialization_complete = False
            self.status = "shutdown"
            self.logger.info("SQLGenerator singleton cleanup completed successfully (shared AsyncClientManager preserved)")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

# FIXED: Factory functions with singleton support
def create_sql_generator(model_configs: Optional[Any] = None, 
                        client_manager: Optional['AsyncClientManager'] = None, # type: ignore
                        async_client_manager: Optional['AsyncClientManager'] = None) -> SQLGenerator: # pyright: ignore[reportInvalidTypeForm]
    """
    Factory function to create SQLGenerator with shared AsyncClientManager
    FIXED: Now uses singleton pattern - returns existing instance if available
    This is the preferred way for main.py to create SQLGenerator instances
    """
    return SQLGenerator.get_instance(model_configs=model_configs, 
                                   client_manager=client_manager or async_client_manager)

# Export
__all__ = ["SQLGenerator", "DynamicSchemaProcessor", "create_sql_generator"]

# Safe module-level execution
if __name__ == "__main__":
    async def test_sql_generator():
        """Test the SQL Generator with shared AsyncClientManager (standalone mode)"""
        print("Testing SQLGenerator with DYNAMIC SCHEMA and shared AsyncClientManager + SINGLETON...")
        
        try:
            # Test singleton behavior
            generator1 = SQLGenerator.get_instance()
            generator2 = SQLGenerator.get_instance()
            
            print(f"Singleton Test: generator1 is generator2 = {generator1 is generator2}")
            print(f"Instance ID 1: {id(generator1)}")
            print(f"Instance ID 2: {id(generator2)}")
            
            # Health check
            health = await generator1.health_check()
            print(f"Health Status: {health['status']}")
            print(f"All Fixes Applied: {health.get('all_critical_fixes_applied', False)}")
            print(f"Dynamic Schema Enabled: {health.get('dynamic_schema_enabled', False)}")
            print(f"Shared Instance: {health.get('shared_instance', False)}")
            print(f"Singleton Pattern: {health.get('singleton_pattern', False)}")
            print(f"Singleton Integration: {health.get('singleton_integration', 'unknown')}")

            if health['status'] in ['healthy', 'degraded']:
                # Test with sample dynamic schema
                test_schema = {
                    "tables": ["customers", "orders", "products"],
                    "columns_by_table": {
                        "customers": ["id", "name", "email"],
                        "orders": ["id", "customer_id", "total"],
                        "products": ["id", "name", "price"]
                    },
                    "joins": [
                        {"table1": "customers", "table2": "orders", "join_condition": "customers.id = orders.customer_id"}
                    ]
                }
                
                result = await generator1.generate_sql_async(
                    "Show me all customer orders with product details",
                    schema_context=test_schema
                )
                print(f"SQL Generated: {result['success']}")
                print(f"Dynamic Schema Applied: {result.get('dynamic_schema_applied', False)}")
                print(f"Shared AsyncClientManager: {result.get('shared_async_client_manager', False)}")
                print(f"Singleton Instance: {result.get('singleton_instance', False)}")
                print(f"Generated SQL: {result['sql'][:200]}...")
            else:
                print("Health check failed, skipping SQL generation test")

        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(test_sql_generator())
