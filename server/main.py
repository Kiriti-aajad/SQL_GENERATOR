"""

Enhanced FastAPI Application for SQL Generator with Dynamic Intent Routing

FIXED: AsyncClientManager singleton integration, component sharing, QueryResponse attribute error, and type safety

UPGRADED: Dynamic AI-powered intent classification system integrated

ENHANCED: Complete A-to-Z pipeline debugging integrated

FIXED: SQL field name mapping issue (sql vs generated_sql)

FIXED: Timeout increased to 180 seconds for complex queries - NO MORE 30s TIMEOUTS

NEW: Added /execute-sql endpoint for direct SQL execution

"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextlib import asynccontextmanager
import traceback
import time
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

# NEW: Additional imports for SQL execution
import pyodbc
import os
from dotenv import load_dotenv

# Load environment variables for database connection
load_dotenv()

# Configuration and models imports
try:
    from config import get_config, ServerConfig
    from models import (
        ProcessingMode, QueryRequest, QueryResponse, ConfidenceLevel,
        create_success_response, create_error_response
    )
except ImportError as e:
    # Fallback for different project structures
    sys.path.insert(0, str(Path(__file__).parent))
    from config import get_config, ServerConfig
    from models import (
        ProcessingMode, QueryRequest, QueryResponse, ConfidenceLevel,
        create_success_response, create_error_response
    )

# Global configuration
config = get_config()
CONFIG_AVAILABLE = True

# FORCE TIMEOUT OVERRIDES - NO MORE 30 SECOND TIMEOUTS
FORCED_QUERY_TIMEOUT = 180  # 3 minutes for complex queries
FORCED_HEALTH_CHECK_TIMEOUT = 120  # 2 minutes for health checks
FORCED_SQL_EXECUTION_TIMEOUT = 120  # 2 minutes for SQL execution
FORCED_COMPONENT_TIMEOUT = 150  # 2.5 minutes for component operations

def setup_logging():
    """Setup enhanced logging with UTF-8 encoding support for comprehensive debugging"""
    try:
        if hasattr(config, 'get_logging_config'):
            logging_config = config.get_logging_config()
        else:
            raise AttributeError("get_logging_config method not available")
    except AttributeError:
        # Fallback if get_logging_config doesn't exist
        logging_config = {
            'level': 'DEBUG', # Changed to DEBUG for comprehensive logging
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'enable_console': True,
            'enable_file': True,
            'log_file': 'debug_pipeline.log',
            'log_dir': 'logs'
        }

    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG, # Force DEBUG level for pipeline tracing
        format=logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[]
    )

    logger = logging.getLogger()
    logger.handlers.clear()

    # Console handler with UTF-8 encoding for Windows
    if logging_config.get('enable_console', True):
        console_handler = logging.StreamHandler(sys.stdout)
        if hasattr(console_handler.stream, 'reconfigure'):
            try:
                console_handler.stream.reconfigure(encoding='utf-8', errors='replace') # pyright: ignore[reportAttributeAccessIssue]
            except Exception:
                pass
        console_handler.setFormatter(
            logging.Formatter(logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        )
        logger.addHandler(console_handler)

    # File handler with UTF-8 encoding for debugging
    log_dir = Path(logging_config.get('log_dir', 'logs'))
    log_dir.mkdir(exist_ok=True)
    try:
        # Always create debug log file
        debug_file_handler = logging.FileHandler(
            log_dir / 'debug_pipeline.log',
            encoding='utf-8',
            errors='replace'
        )
        debug_file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(debug_file_handler)
        logging.info(f"Debug logging enabled: {log_dir / 'debug_pipeline.log'}")

        # Original log file if specified
        if logging_config.get('enable_file', False) and logging_config.get('log_file'):
            log_file_path = log_dir / logging_config['log_file']
            if log_file_path.name != 'debug_pipeline.log': # Avoid duplicate handlers
                file_handler = logging.FileHandler(
                    log_file_path,
                    encoding='utf-8',
                    errors='replace'
                )
                file_handler.setFormatter(
                    logging.Formatter(logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                )
                logger.addHandler(file_handler)
                logging.info(f"File logging enabled: {log_file_path}")
    except Exception as e:
        logging.warning(f"Could not setup file logging: {e}")

# Setup enhanced logging BEFORE imports
setup_logging()

# Log timeout configuration on startup
logging.info("=" * 80)
logging.info("TIMEOUT CONFIGURATION - FORCED OVERRIDES ACTIVE")
logging.info(f"FORCED_QUERY_TIMEOUT: {FORCED_QUERY_TIMEOUT} seconds")
logging.info(f"FORCED_HEALTH_CHECK_TIMEOUT: {FORCED_HEALTH_CHECK_TIMEOUT} seconds") 
logging.info(f"FORCED_SQL_EXECUTION_TIMEOUT: {FORCED_SQL_EXECUTION_TIMEOUT} seconds")
logging.info(f"FORCED_COMPONENT_TIMEOUT: {FORCED_COMPONENT_TIMEOUT} seconds")
logging.info("NO MORE 30 SECOND TIMEOUTS!")
logging.info("=" * 80)

# Component imports with error handling
COMPONENTS_AVAILABLE = {}
components = {}

def safe_import_component(module_name: str, component_name: str):
    """Safely import components and track availability"""
    try:
        module = __import__(module_name, fromlist=[component_name])
        component = getattr(module, component_name)
        COMPONENTS_AVAILABLE[component_name] = component
        logging.info(f"Successfully imported {component_name}")
        return component
    except ImportError as e:
        logging.warning(f"Component {component_name} not available: {e}")
        COMPONENTS_AVAILABLE[component_name] = None
        return None
    except Exception as e:
        logging.error(f"Error importing {component_name}: {e}")
        COMPONENTS_AVAILABLE[component_name] = None
        return None

def safe_get_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get attribute from object or dict"""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    else:
        return getattr(obj, attr, default)

# Enhanced SQL extraction function
def extract_sql_from_result(result: Any) -> str:
    """
    FIXED: Extract SQL from result, checking all possible field names
    This fixes the critical issue where SQL was stored in 'generated_sql' but we were looking for 'sql'
    """
    possible_sql_fields = ['sql', 'generated_sql', 'query_result', 'final_sql', 'query']
    
    # Check if result is a dictionary
    if isinstance(result, dict):
        for field in possible_sql_fields:
            if field in result and result[field]:
                logging.info(f"DEBUG: Found SQL in result['{field}']: {result[field][:100]}...")
                return result[field]
    
    # Check if result has a data attribute with dictionary
    elif hasattr(result, 'data') and isinstance(result.data, dict):
        for field in possible_sql_fields:
            if field in result.data and result.data[field]:
                logging.info(f"DEBUG: Found SQL in result.data['{field}']: {result.data[field][:100]}...")
                return result.data[field]
    
    # Check for direct attributes
    elif hasattr(result, 'sql') or hasattr(result, 'generated_sql'):
        sql_val = getattr(result, 'sql', None) or getattr(result, 'generated_sql', None)
        if sql_val:
            logging.info(f"DEBUG: Found SQL in direct attribute: {sql_val[:100]}...")
            return sql_val

    logging.error("DEBUG: No SQL found in any expected field!")
    logging.error(f"DEBUG: Result type: {type(result)}")
    logging.error(f"DEBUG: Result keys/attrs: {list(result.keys()) if isinstance(result, dict) else [attr for attr in dir(result) if not attr.startswith('_')]}")
    return ""

def extract_explanation_from_result(result: Any) -> str:
    """Extract explanation from result, checking multiple possible sources"""
    possible_explanation_fields = ['explanation', 'description', 'details', 'summary']
    
    if isinstance(result, dict):
        for field in possible_explanation_fields:
            if field in result and result[field]:
                return result[field]
    elif hasattr(result, 'data') and isinstance(result.data, dict):
        for field in possible_explanation_fields:
            if field in result.data and result.data[field]:
                return result.data[field]
    elif hasattr(result, 'explanation'):
        return safe_get_attr(result, 'explanation', "")
    return ""

def extract_confidence_from_result(result: Any) -> str:
    """Extract confidence from result, with proper fallback handling"""
    if isinstance(result, dict):
        return result.get("confidence", "medium")
    elif hasattr(result, 'data') and isinstance(result.data, dict):
        return result.data.get("confidence", "medium")
    elif hasattr(result, 'confidence'):
        return safe_get_attr(result, 'confidence', "medium")
    else:
        return "medium"

# Import components
HybridAIAgentOrchestrator = safe_import_component(
    "orchestrator.hybrid_orchestrator", "HybridAIAgentOrchestrator"
)

HybridConfig = safe_import_component(
    "orchestrator.hybrid_orchestrator", "HybridConfig"
)

IntelligentOrchestrator = safe_import_component(
    "orchestrator.intelligent_orchestrator", "IntelligentOrchestrator"
)

CreateSchemaRetrievalAgent = safe_import_component(
    "agent.schema_searcher.core.retrieval_agent", "create_schema_retrieval_agent"
)

CreateIntelligentRetrievalAgent = safe_import_component(
    "agent.schema_searcher.core.intelligent_retrieval_agent", "create_intelligent_retrieval_agent"
)

AsyncClientManager = safe_import_component(
    "agent.sql_generator.async_client_manager", "AsyncClientManager"
)

async def initialize_components():
    """Initialize components with enhanced debugging and shared AsyncClientManager singleton"""
    logging.info("=" * 80)
    logging.info("INITIALIZING SQL GENERATOR API WITH COMPREHENSIVE DEBUG LOGGING")
    logging.info("=" * 80)
    initialization_start = time.time()

    # Create and initialize AsyncClientManager FIRST
    shared_client_manager = None
    if AsyncClientManager:
        try:
            logging.info("DEBUG: Creating shared AsyncClientManager singleton...")
            shared_client_manager = AsyncClientManager()
            
            # Verify singleton integrity
            if hasattr(shared_client_manager, 'verify_singleton'):
                if not shared_client_manager.verify_singleton():
                    logging.error("DEBUG: AsyncClientManager singleton integrity compromised!")
                    raise RuntimeError("AsyncClientManager singleton integrity compromised")
            
            # Initialize the AsyncClientManager with extended timeout
            logging.info(f"DEBUG: Initializing AsyncClientManager with {FORCED_COMPONENT_TIMEOUT}s timeout...")
            try:
                await asyncio.wait_for(
                    shared_client_manager.initialize(), 
                    timeout=FORCED_COMPONENT_TIMEOUT
                )
            except asyncio.TimeoutError:
                logging.error(f"DEBUG: AsyncClientManager initialization timed out after {FORCED_COMPONENT_TIMEOUT}s")
                shared_client_manager = None
                components['client_manager'] = None
                raise
            
            # Store the shared instance
            components['client_manager'] = shared_client_manager
            logging.info(f"DEBUG: AsyncClientManager initialized successfully (ID: {id(shared_client_manager)})")

            # Log client status for debugging
            if hasattr(shared_client_manager, 'get_client_status'):
                try:
                    status = shared_client_manager.get_client_status()
                    logging.info(f"DEBUG: AsyncClientManager status: {status.get('healthy_count', 0)}/{status.get('total_clients', 0)} clients healthy")
                    logging.info(f"DEBUG: Available clients: {status.get('available_clients', [])}")
                except Exception as e:
                    logging.warning(f"DEBUG: Could not get client status: {e}")
                    
        except Exception as e:
            logging.error(f"DEBUG: Failed to initialize AsyncClientManager: {e}")
            if logging.getLogger().level <= logging.DEBUG:
                logging.debug(traceback.format_exc())
            shared_client_manager = None
            components['client_manager'] = None
    else:
        logging.warning("DEBUG: AsyncClientManager class not available")
        components['client_manager'] = None

    # Initialize HybridAIAgentOrchestrator with dynamic routing capabilities
    if HybridAIAgentOrchestrator and HybridConfig:
        try:
            logging.info("DEBUG: Initializing HybridAIAgentOrchestrator...")
            hybrid_config = HybridConfig()
            
            # Pass shared AsyncClientManager to enable dynamic routing
            if shared_client_manager and hasattr(HybridAIAgentOrchestrator, '__init__'):
                import inspect
                try:
                    sig = inspect.signature(HybridAIAgentOrchestrator.__init__)
                    if 'async_client_manager' in sig.parameters:
                        components['hybrid_orchestrator'] = HybridAIAgentOrchestrator(
                            hybrid_config,
                            async_client_manager=shared_client_manager
                        )
                        logging.info("DEBUG: HybridAIAgentOrchestrator initialized with DYNAMIC ROUTING enabled")
                    else:
                        components['hybrid_orchestrator'] = HybridAIAgentOrchestrator(hybrid_config)
                        logging.info("DEBUG: HybridAIAgentOrchestrator initialized in standard mode")
                except Exception as e:
                    logging.warning(f"DEBUG: Could not inspect HybridAIAgentOrchestrator signature: {e}")
                    components['hybrid_orchestrator'] = HybridAIAgentOrchestrator(hybrid_config)
            else:
                components['hybrid_orchestrator'] = HybridAIAgentOrchestrator(hybrid_config)
                
            if hasattr(components['hybrid_orchestrator'], 'initialize'):
                logging.info(f"DEBUG: Initializing HybridAIAgentOrchestrator with {FORCED_COMPONENT_TIMEOUT}s timeout...")
                try:
                    await asyncio.wait_for(
                        components['hybrid_orchestrator'].initialize(),
                        timeout=FORCED_COMPONENT_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logging.error(f"DEBUG: HybridAIAgentOrchestrator initialization timed out after {FORCED_COMPONENT_TIMEOUT}s")
                    components['hybrid_orchestrator'] = None
                    raise
            
            # Check if dynamic routing is enabled
            if components['hybrid_orchestrator'] and hasattr(components['hybrid_orchestrator'], 'dynamic_router'):
                logging.info("DEBUG: DYNAMIC AI-POWERED INTENT ROUTING ENABLED")
                logging.info("DEBUG: Adaptive query classification active")
                logging.info("DEBUG: Banking domain optimization active")
            else:
                logging.info("DEBUG: Standard orchestration mode active")
                
        except Exception as e:
            logging.error(f"DEBUG: Failed to initialize HybridAIAgentOrchestrator: {e}")
            if logging.getLogger().level <= logging.DEBUG:
                logging.debug(traceback.format_exc())
            components['hybrid_orchestrator'] = None
    else:
        logging.warning("DEBUG: HybridAIAgentOrchestrator or HybridConfig not available")
        components['hybrid_orchestrator'] = None

    # Initialize other components with extended timeouts
    if IntelligentOrchestrator:
        try:
            if shared_client_manager:
                import inspect
                try:
                    sig = inspect.signature(IntelligentOrchestrator.__init__)
                    if 'async_client_manager' in sig.parameters:
                        components['intelligent_orchestrator'] = IntelligentOrchestrator(
                            async_client_manager=shared_client_manager
                        )
                    else:
                        components['intelligent_orchestrator'] = IntelligentOrchestrator()
                except Exception as e:
                    logging.warning(f"DEBUG: Could not inspect IntelligentOrchestrator signature: {e}")
                    components['intelligent_orchestrator'] = IntelligentOrchestrator()
            else:
                components['intelligent_orchestrator'] = IntelligentOrchestrator()
                
            if hasattr(components['intelligent_orchestrator'], 'initialize'):
                logging.info(f"DEBUG: Initializing IntelligentOrchestrator with {FORCED_COMPONENT_TIMEOUT}s timeout...")
                try:
                    await asyncio.wait_for(
                        components['intelligent_orchestrator'].initialize(),
                        timeout=FORCED_COMPONENT_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logging.error(f"DEBUG: IntelligentOrchestrator initialization timed out after {FORCED_COMPONENT_TIMEOUT}s")
                    components['intelligent_orchestrator'] = None
                    
            logging.info("DEBUG: IntelligentOrchestrator initialized with shared AsyncClientManager")
        except Exception as e:
            logging.error(f"DEBUG: Failed to initialize IntelligentOrchestrator: {e}")
            components['intelligent_orchestrator'] = None
    else:
        logging.warning("DEBUG: IntelligentOrchestrator not available")
        components['intelligent_orchestrator'] = None

    # Initialize retrieval agents with extended timeouts
    if CreateSchemaRetrievalAgent:
        try:
            components['retrieval_agent'] = CreateSchemaRetrievalAgent(json_mode=True)
            if hasattr(components['retrieval_agent'], 'initialize'):
                logging.info(f"DEBUG: Initializing Schema RetrievalAgent with {FORCED_COMPONENT_TIMEOUT}s timeout...")
                try:
                    await asyncio.wait_for(
                        components['retrieval_agent'].initialize(),
                        timeout=FORCED_COMPONENT_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logging.error(f"DEBUG: Schema RetrievalAgent initialization timed out after {FORCED_COMPONENT_TIMEOUT}s")
                    components['retrieval_agent'] = None
            logging.info("DEBUG: Schema RetrievalAgent initialized successfully")
        except Exception as e:
            logging.error(f"DEBUG: Failed to initialize Schema RetrievalAgent: {e}")
            components['retrieval_agent'] = None
    else:
        logging.warning("DEBUG: CreateSchemaRetrievalAgent not available")
        components['retrieval_agent'] = None

    if CreateIntelligentRetrievalAgent:
        try:
            if shared_client_manager:
                components['intelligent_retrieval_agent'] = CreateIntelligentRetrievalAgent(
                    async_client_manager=shared_client_manager
                )
            else:
                components['intelligent_retrieval_agent'] = CreateIntelligentRetrievalAgent()
            if hasattr(components['intelligent_retrieval_agent'], 'initialize'):
                logging.info(f"DEBUG: Initializing Intelligent RetrievalAgent with {FORCED_COMPONENT_TIMEOUT}s timeout...")
                try:
                    await asyncio.wait_for(
                        components['intelligent_retrieval_agent'].initialize(),
                        timeout=FORCED_COMPONENT_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logging.error(f"DEBUG: Intelligent RetrievalAgent initialization timed out after {FORCED_COMPONENT_TIMEOUT}s")
                    components['intelligent_retrieval_agent'] = None
            logging.info("DEBUG: Intelligent RetrievalAgent initialized with shared AsyncClientManager")
        except Exception as e:
            logging.error(f"DEBUG: Failed to initialize Intelligent RetrievalAgent: {e}")
            components['intelligent_retrieval_agent'] = None
    else:
        logging.warning("DEBUG: CreateIntelligentRetrievalAgent not available")
        components['intelligent_retrieval_agent'] = None

    initialization_time = time.time() - initialization_start
    available_components = [name for name, comp in components.items() if comp is not None]
    
    logging.info("=" * 80)
    logging.info(f"DEBUG: INITIALIZATION COMPLETE IN {initialization_time:.2f}s")
    logging.info(f"DEBUG: Available components: {available_components}")
    
    # Log dynamic routing capabilities
    if components.get('hybrid_orchestrator') and hasattr(components['hybrid_orchestrator'], 'dynamic_router'):
        logging.info("DEBUG: DYNAMIC ROUTING CAPABILITIES:")
        logging.info("DEBUG: AI-Powered Intent Classification")
        logging.info("DEBUG: Adaptive Query Routing")
        logging.info("DEBUG: Banking Domain Optimization")
        logging.info("DEBUG: Historical Pattern Learning")
        logging.info("=" * 80)

    # Enhanced validation with AsyncClientManager status
    if not available_components:
        logging.warning("DEBUG: No components were successfully initialized - server will run in minimal mode")
    elif 'client_manager' not in available_components:
        logging.warning("DEBUG: AsyncClientManager not available - SQL generation may be limited")
    else:
        if shared_client_manager and hasattr(shared_client_manager, 'get_client_status'):
            try:
                status = shared_client_manager.get_client_status()
                logging.info(f"DEBUG: AsyncClientManager integration complete: {status.get('available_clients', [])} clients available")
                if status.get('healthy_count', 0) > 0:
                    logging.info(f"DEBUG: AI features enabled: {status.get('healthy_count', 0)} healthy clients")
                else:
                    logging.warning("DEBUG: AI features may be limited: no healthy clients available")
            except Exception as e:
                logging.warning(f"DEBUG: Could not log AsyncClientManager integration status: {e}")

async def validate_component_health() -> Dict[str, Dict[str, Any]]:
    """Validate health with FORCED extended timeouts and proper error handling"""
    health_status = {}
    
    for component_name, component in components.items():
        if component is None:
            health_status[component_name] = {
                "status": "unavailable",
                "message": "Component not initialized",
                "available": False
            }
            continue

        try:
            if hasattr(component, 'health_check'):
                # FORCED: Use extended health check timeout instead of config value
                health_check_timeout = FORCED_HEALTH_CHECK_TIMEOUT
                logging.info(f"DEBUG: Health check for {component_name} with {health_check_timeout}s timeout (FORCED)")
                
                try:
                    health_result = await asyncio.wait_for(
                        component.health_check(),
                        timeout=health_check_timeout
                    )
                except asyncio.TimeoutError:
                    health_status[component_name] = {
                        "status": "error",
                        "message": f"Health check timed out after {health_check_timeout}s (FORCED TIMEOUT)",
                        "available": False
                    }
                    logging.warning(f"DEBUG: {component_name} health check timed out after {health_check_timeout}s")
                    continue

                # Handle different return types
                if isinstance(health_result, dict):
                    health_result.setdefault("available", True)
                    health_result.setdefault("status", "healthy")
                    health_status[component_name] = health_result
                elif isinstance(health_result, bool):
                    health_status[component_name] = {
                        "status": "healthy" if health_result else "unhealthy",
                        "available": health_result,
                        "message": "Health check completed"
                    }
                else:
                    is_healthy = bool(health_result)
                    health_status[component_name] = {
                        "status": "healthy" if is_healthy else "unhealthy",
                        "available": is_healthy,
                        "message": "Health check completed"
                    }
            else:
                health_status[component_name] = {
                    "status": "healthy",
                    "available": True,
                    "message": "No health check method - assuming healthy"
                }
        except Exception as e:
            health_status[component_name] = {
                "status": "error",
                "available": False,
                "message": str(e),
                "error_type": type(e).__name__
            }

    return health_status

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with proper error handling"""
    startup_start = time.time()
    try:
        logging.info("DEBUG: Starting Enhanced SQL Generator API with Dynamic Routing...")
        await initialize_components()
        startup_time = time.time() - startup_start
        logging.info(f"DEBUG: Application startup completed in {startup_time:.2f}s")
        yield
    except Exception as e:
        logging.error(f"DEBUG: Startup failed: {e}")
        if logging.getLogger().level <= logging.DEBUG:
            logging.debug(traceback.format_exc())
        yield
    finally:
        logging.info("DEBUG: Shutting down Enhanced SQL Generator API...")
        for component_name, component in components.items():
            if component and hasattr(component, 'cleanup'):
                try:
                    await component.cleanup()
                    logging.info(f"DEBUG: Cleaned up {component_name}")
                except Exception as e:
                    logging.warning(f"DEBUG: Error cleaning up {component_name}: {e}")

# Create FastAPI application
app = FastAPI(
    title="Enhanced SQL Generator API with Dynamic Routing",
    description="API for generating SQL from natural language queries using AI orchestration with dynamic intent routing",
    version="2.0.0",
    docs_url="/docs" if not safe_get_attr(config, 'is_production', lambda: False)() else None,
    redoc_url="/redoc" if not safe_get_attr(config, 'is_production', lambda: False)() else None,
    lifespan=lifespan
)

# CORS middleware configuration
try:
    allowed_origins = safe_get_attr(config, 'ALLOWED_ORIGINS', ['*'])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception as e:
    logging.warning(f"DEBUG: Could not configure CORS middleware: {e}")

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 3))
    return response

# Exception handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Validation error",
            "details": exc.errors()
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "error_code": f"HTTP_{exc.status_code}"
        }
    )

def get_current_config() -> ServerConfig:
    return config

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with enhanced system information"""
    is_dev = not safe_get_attr(config, 'is_production', lambda: False)()
    
    # Check if dynamic routing is enabled
    dynamic_routing_enabled = False
    if components.get('hybrid_orchestrator') and hasattr(components['hybrid_orchestrator'], 'dynamic_router'):
        dynamic_routing_enabled = True

    return {
        "message": "Enhanced SQL Generator API with Dynamic Routing",
        "version": "2.0.0",
        "status": "running",
        "environment": "development" if is_dev else "production",
        "components_available": len([c for c in components.values() if c is not None]),
        "ai_features_enabled": bool(components.get('client_manager')),
        "dynamic_routing_enabled": dynamic_routing_enabled,
        "debug_logging_enabled": True,
        "timeout_increased": f"{FORCED_QUERY_TIMEOUT} seconds (FORCED)",
        "timeout_configuration": {
            "query_timeout": f"{FORCED_QUERY_TIMEOUT}s",
            "health_check_timeout": f"{FORCED_HEALTH_CHECK_TIMEOUT}s",
            "sql_execution_timeout": f"{FORCED_SQL_EXECUTION_TIMEOUT}s",
            "component_timeout": f"{FORCED_COMPONENT_TIMEOUT}s"
        },
        "features": [
            "SQL Generation from Natural Language",
            "AI-Powered Query Processing",
            "AsyncClientManager Integration",
            "Dynamic Intent Classification" if dynamic_routing_enabled else "Standard Query Routing",
            "Banking Domain Optimization" if dynamic_routing_enabled else "General Domain Processing",
            "Adaptive Learning" if dynamic_routing_enabled else "Static Processing",
            "Comprehensive Pipeline Debugging",
            "Extended Timeout for Complex Queries (180s)",
            "Direct SQL Execution"  # NEW feature
        ],
        "endpoints": [
            "/query - Process SQL generation queries",
            "/execute-sql - Execute SQL queries directly",  # NEW endpoint
            "/health - System health check",
            "/config - Configuration information",
            "/components - Component status",
            "/routing-stats - Dynamic routing statistics" if dynamic_routing_enabled else None,
            "/intent-analysis/{query} - AI intent analysis" if dynamic_routing_enabled else None,
            "/docs - API documentation"
        ]
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with FORCED extended timeouts"""
    try:
        health_start = time.time()
        logging.info(f"DEBUG: Starting health check with {FORCED_HEALTH_CHECK_TIMEOUT}s timeout (FORCED)")
        
        component_health = await validate_component_health()
        health_time = time.time() - health_start

        # Determine overall status
        available_count = sum(1 for comp in component_health.values() if comp.get("available", False))
        total_count = len(component_health)
        
        if available_count == total_count:
            overall_status = "healthy"
        elif available_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        # Add AsyncClientManager specific health info
        client_manager_info = {}
        if components.get('client_manager'):
            try:
                client_status = components['client_manager'].get_client_status()
                client_manager_info = {
                    "available_clients": client_status.get('available_clients', []),
                    "healthy_count": client_status.get('healthy_count', 0),
                    "total_clients": client_status.get('total_clients', 0),
                    "is_singleton": client_status.get('is_singleton', False),
                    "instance_id": client_status.get('instance_id')
                }
            except Exception as e:
                client_manager_info = {"error": str(e)}

        # Add dynamic routing health info
        dynamic_routing_info = {}
        if components.get('hybrid_orchestrator') and hasattr(components['hybrid_orchestrator'], 'dynamic_router'):
            try:
                routing_stats = components['hybrid_orchestrator'].dynamic_router.get_routing_statistics()
                dynamic_routing_info = {
                    "enabled": True,
                    "ai_classification_success_rate": routing_stats.get("ai_success_rate", 0.0),
                    "total_classifications": routing_stats.get("total_classifications", 0),
                    "route_distribution": routing_stats.get("route_distribution", {}),
                    "most_used_route": routing_stats.get("most_used_route", "none")
                }
            except Exception as e:
                dynamic_routing_info = {"enabled": True, "error": str(e)}
        else:
            dynamic_routing_info = {"enabled": False, "message": "Standard routing active"}

        return {
            "success": True,
            "status": overall_status,
            "timestamp": time.time(),
            "execution_time": round(health_time, 3),
            "components": component_health,
            "async_client_manager": client_manager_info,
            "dynamic_routing": dynamic_routing_info,
            "debug_logging": True,
            "timeout_settings": f"{FORCED_QUERY_TIMEOUT} seconds for queries (FORCED)",
            "sql_execution_enabled": True,  # NEW: Indicate SQL execution is available
            "timeout_configuration": {
                "query_timeout": f"{FORCED_QUERY_TIMEOUT}s",
                "health_check_timeout": f"{FORCED_HEALTH_CHECK_TIMEOUT}s",
                "sql_execution_timeout": f"{FORCED_SQL_EXECUTION_TIMEOUT}s",
                "component_timeout": f"{FORCED_COMPONENT_TIMEOUT}s"
            },
            "performance_metrics": {
                "available_components": available_count,
                "total_components": total_count,
                "availability_ratio": round(available_count / total_count, 2) if total_count > 0 else 0
            },
            "configuration": {
                "max_concurrent_requests": safe_get_attr(config, 'MAX_CONCURRENT_REQUESTS', 10),
                "caching_enabled": safe_get_attr(config, 'ENABLE_CACHING', True),
                "graceful_fallbacks": safe_get_attr(config, 'ENABLE_GRACEFUL_FALLBACKS', True),
                "debug_mode": safe_get_attr(config, 'DEBUG_MODE', False)
            }
        }
    except Exception as e:
        logging.error(f"DEBUG: Health check failed: {e}")
        return {
            "success": False,
            "status": "error",
            "timestamp": time.time(),
            "error": str(e),
            "components": {"error": {"status": "error", "message": str(e)}}
        }

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    ENHANCED: Process query with comprehensive A-to-Z debugging and FORCED extended timeouts
    TIMEOUT FIXED: Now uses FORCED 180 seconds timeout - NO MORE 30s TIMEOUTS!
    """
    start_time = time.time()
    
    # TIMEOUT CONFIGURATION DEBUG - FORCED VALUES
    logging.info("=" * 80)
    logging.info("DEBUG TIMEOUT CONFIGURATION CHECK - FORCED VALUES")
    logging.info(f"FORCED_QUERY_TIMEOUT: {FORCED_QUERY_TIMEOUT} seconds")
    logging.info(f"Config DEFAULT_TIMEOUT: {safe_get_attr(config, 'DEFAULT_TIMEOUT', 'NOT_SET')}")
    logging.info(f"Request timeout: {safe_get_attr(request, 'timeout', 'NOT_SET')}")
    logging.info("=" * 80)
    
    # PIPELINE START DEBUG
    logging.info("=" * 80)
    logging.info("DEBUG PIPELINE START: Full A-to-Z Debug Trace")
    logging.info(f"DEBUG RAW QUERY RECEIVED: '{request.query}'")
    logging.info(f"DEBUG REQUEST TIMESTAMP: {time.time()}")
    logging.info(f"DEBUG QUERY LENGTH: {len(request.query)} characters")
    logging.info(f"DEBUG SESSION ID: {safe_get_attr(request, 'session_id', 'None')}")

    try:
        logging.info(f"DEBUG: Processing query with enhanced routing: {request.query[:100]}...")
        
        # Check if we have any orchestrator available
        orchestrator = (
            components.get('hybrid_orchestrator') or
            components.get('intelligent_orchestrator')
        )

        # ORCHESTRATOR SELECTION DEBUG
        logging.info("=" * 60)
        logging.info("DEBUG ORCHESTRATOR SELECTION")
        logging.info(f"DEBUG Available Components: {list(components.keys())}")
        logging.info(f"DEBUG Hybrid Available: {components.get('hybrid_orchestrator') is not None}")
        logging.info(f"DEBUG Intelligent Available: {components.get('intelligent_orchestrator') is not None}")
        
        if not orchestrator:
            logging.error("DEBUG NO ORCHESTRATOR AVAILABLE!")
            return create_error_response(
                error_message="No orchestrator available - system in degraded mode",
                error_code="NO_ORCHESTRATOR",
                session_id=safe_get_attr(request, 'session_id', None)
            ).model_copy(update={"execution_time": round(time.time() - start_time, 3)})

        orchestrator_type = type(orchestrator).__name__
        logging.info(f"DEBUG Selected Orchestrator: {orchestrator_type}")
        logging.info(f"DEBUG Has Dynamic Router: {hasattr(orchestrator, 'dynamic_router')}")

        # Log routing method being used
        if hasattr(orchestrator, 'dynamic_router'):
            logging.info("DEBUG Using AI-powered dynamic routing for query classification")
            logging.info(f"DEBUG AsyncClientManager Available: {orchestrator.async_client_manager is not None}")
        else:
            logging.info("DEBUG Using standard orchestration routing")

        # ORCHESTRATOR INPUT DEBUG
        logging.info("=" * 60)
        logging.info("DEBUG ORCHESTRATOR PROCESSING - ENTRY")
        logging.info(f"DEBUG Input Query: '{request.query}'")
        logging.info(f"DEBUG Input Context: {safe_get_attr(request, 'context', {})}")
        logging.info(f"DEBUG Available Methods: {[m for m in dir(orchestrator) if 'process' in m or 'orchestrate' in m]}")

        # CRITICAL FIX: FORCED timeout - ignore all config values and use hardcoded 180 seconds
        query_timeout = FORCED_QUERY_TIMEOUT  # FORCED to 180 seconds
        logging.info(f"DEBUG Query Timeout: {query_timeout}s (FORCED - IGNORING ALL CONFIG VALUES)")
        logging.info("DEBUG: NO MORE 30 SECOND TIMEOUTS!")

        orchestrator_start = time.time()
        logging.info(f"DEBUG STARTING ORCHESTRATOR at {orchestrator_start}")

        try:
            # Use correct field names in orchestrator calls
            if hasattr(orchestrator, 'process_query'):
                logging.info("DEBUG Calling orchestrator.process_query()")
                result = await asyncio.wait_for(
                    orchestrator.process_query(request.query, safe_get_attr(request, 'context', {})),
                    timeout=query_timeout  # FORCED 180 seconds
                )
            elif hasattr(orchestrator, 'orchestrate_user_query'):
                logging.info("DEBUG Calling orchestrator.orchestrate_user_query()")
                result = await asyncio.wait_for(
                    orchestrator.orchestrate_user_query(request.query, safe_get_attr(request, 'context', {})),
                    timeout=query_timeout  # FORCED 180 seconds
                )
            else:
                logging.error("DEBUG NO VALID ORCHESTRATOR METHOD FOUND!")
                result = {
                    "sql": f"-- Query: {request.query}\nSELECT 'Query processing not fully available' AS message;",
                    "explanation": "Using fallback response - full processing not available",
                    "confidence": "medium"
                }

            orchestrator_time = time.time() - orchestrator_start
            logging.info(f"DEBUG ORCHESTRATOR COMPLETED in {orchestrator_time:.2f}s")
            
        except asyncio.TimeoutError:
            timeout_time = time.time() - orchestrator_start
            logging.error(f"DEBUG ORCHESTRATOR TIMEOUT after {timeout_time:.2f}s (FORCED TIMEOUT: {query_timeout}s)")
            return create_error_response(
                error_message=f"Query processing timed out after {query_timeout} seconds (FORCED TIMEOUT)",
                error_code="TIMEOUT",
                session_id=safe_get_attr(request, 'session_id', None)
            ).model_copy(update={"execution_time": round(time.time() - start_time, 3)})

        # ORCHESTRATOR OUTPUT DEBUG
        logging.info("=" * 60)
        logging.info("DEBUG ORCHESTRATOR PROCESSING - EXIT")
        logging.info(f"DEBUG Processing Time: {orchestrator_time:.2f}s")
        logging.info(f"DEBUG Result Type: {type(result)}")
        logging.info(f"DEBUG Result Structure: {type(result).__name__}")

        # Debug result contents based on type
        if isinstance(result, dict):
            logging.info("DEBUG RESULT IS DICTIONARY:")
            logging.info(f"DEBUG Dict Keys: {list(result.keys())}")
            for key, value in result.items():
                if key in ['sql', 'generated_sql']:
                    logging.info(f"DEBUG SQL Content ({key}): '{value}' (length: {len(str(value))})")
                elif key == 'explanation':
                    logging.info(f"DEBUG Explanation: '{value}' (length: {len(str(value))})")
                else:
                    logging.info(f"DEBUG {key}: {value}")
        elif hasattr(result, 'data') and isinstance(result.data, dict):
            logging.info("DEBUG RESULT HAS DATA DICT:")
            logging.info(f"DEBUG Data Keys: {list(result.data.keys())}")
            for key, value in result.data.items():
                if key in ['sql', 'generated_sql']:
                    logging.info(f"DEBUG Data SQL ({key}): '{value}' (length: {len(str(value))})")
                elif key == 'explanation':
                    logging.info(f"DEBUG Data Explanation: '{value}' (length: {len(str(value))})")
                else:
                    logging.info(f"DEBUG Data {key}: {value}")
        elif hasattr(result, 'sql') or hasattr(result, 'generated_sql'):
            logging.info("DEBUG RESULT HAS DIRECT SQL ATTRIBUTE:")
            sql_value = getattr(result, 'sql', '') or getattr(result, 'generated_sql', '')
            logging.info(f"DEBUG Direct SQL: '{sql_value}' (length: {len(str(sql_value))})")
            logging.info(f"DEBUG Direct Explanation: '{getattr(result, 'explanation', '')}' (length: {len(str(getattr(result, 'explanation', '')))})")
            logging.info(f"DEBUG Direct Confidence: '{getattr(result, 'confidence', '')}'")
        else:
            logging.info("DEBUG UNKNOWN RESULT STRUCTURE:")
            logging.info(f"DEBUG Result String: '{str(result)}'")
            logging.info(f"DEBUG Result Attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")

        # AI ROUTING DECISION DEBUG (if available)
        logging.info("=" * 60)
        logging.info("DEBUG AI ROUTING ANALYSIS")
        if hasattr(result, 'data') and isinstance(result.data, dict): # pyright: ignore[reportAttributeAccessIssue]
            routing_info = result.data.get('ai_routing_decision') # pyright: ignore[reportAttributeAccessIssue]
            if routing_info:
                logging.info("DEBUG AI ROUTING FOUND:")
                logging.info(f"DEBUG Selected Route: {routing_info.get('selected_route', 'unknown')}")
                logging.info(f"DEBUG Confidence: {routing_info.get('confidence', 0.0):.3f}")
                logging.info(f"DEBUG Classification Time: {routing_info.get('classification_time', 0)}ms")
                logging.info(f"DEBUG Primary Intent: {routing_info.get('primary_intent', 'unknown')}")
            else:
                logging.info("DEBUG No AI routing decision found in result.data")
        else:
            logging.info("DEBUG No AI routing decision data available")

        # FIXED: SQL EXTRACTION AND PROCESSING DEBUG
        logging.info("=" * 60)
        logging.info("DEBUG SQL EXTRACTION - CRITICAL PHASE")
        
        execution_time = round(time.time() - start_time, 3)
        confidence = ConfidenceLevel.MEDIUM # Default

        # Use the enhanced extraction functions
        sql_result = extract_sql_from_result(result)
        explanation_result = extract_explanation_from_result(result)
        confidence_raw = extract_confidence_from_result(result)
        
        # Track which path we took for debugging
        extraction_path = "enhanced_extraction"

        # Final extraction summary
        logging.info("DEBUG EXTRACTION SUMMARY:")
        logging.info(f"DEBUG Path Used: {extraction_path}")
        logging.info(f"DEBUG SQL Length: {len(sql_result)} characters")
        logging.info(f"DEBUG SQL Empty: {sql_result == ''}")
        logging.info(f"DEBUG SQL Content: '{sql_result[:200]}{'...' if len(sql_result) > 200 else ''}'")

        # Enhanced problem detection
        if sql_result == "":
            logging.error("DEBUG CRITICAL ISSUE: SQL RESULT IS EMPTY!")
            logging.error("DEBUG This indicates the extraction failed even with enhanced logic!")
            logging.error(f"DEBUG Original result type: {type(result)}")
            logging.error(f"DEBUG Original result content: {result}")
            logging.error(f"DEBUG Extraction path: {extraction_path}")
            
            # Last resort: try to extract from string representation
            result_str = str(result)
            if 'SELECT' in result_str.upper():
                logging.info("DEBUG: Attempting to extract SQL from string representation")
                import re
                sql_match = re.search(r'(SELECT.*?;?)', result_str, re.IGNORECASE | re.DOTALL)
                if sql_match:
                    sql_result = sql_match.group(1).strip()
                    logging.info(f"DEBUG: Extracted SQL from string: {sql_result}")
        else:
            logging.info("DEBUG: SQL extraction successful!")

        # Safe confidence processing
        try:
            if isinstance(confidence_raw, (int, float)):
                conf_score = float(confidence_raw)
            elif isinstance(confidence_raw, str):
                conf_map = {'high': 0.9, 'medium': 0.6, 'low': 0.3}
                conf_score = conf_map.get(confidence_raw.lower(), 0.5)
            else:
                conf_score = 0.5

            if conf_score >= 0.8:
                confidence = ConfidenceLevel.HIGH
            elif conf_score >= 0.5:
                confidence = ConfidenceLevel.MEDIUM
            else:
                confidence = ConfidenceLevel.LOW
        except (ValueError, TypeError, AttributeError):
            confidence = ConfidenceLevel.MEDIUM

        # FINAL RESPONSE ASSEMBLY DEBUG
        logging.info("=" * 60)
        logging.info("DEBUG FINAL RESPONSE ASSEMBLY")
        logging.info(f"DEBUG Final SQL: '{sql_result}' (length: {len(sql_result)})")
        logging.info(f"DEBUG Final Explanation: '{explanation_result}' (length: {len(explanation_result)})")
        logging.info(f"DEBUG Final Confidence: {confidence}")
        logging.info(f"DEBUG Total Execution Time: {execution_time}s")

        response = create_success_response(
            sql=sql_result,
            explanation=explanation_result,
            confidence=confidence,
            execution_time=execution_time
        )

        # Final response validation
        response_dict = response.dict() if hasattr(response, 'dict') else response
        logging.info("DEBUG RESPONSE VALIDATION:")
        logging.info(f"DEBUG Response SQL Empty: {response_dict.get('sql', '') == ''}") # pyright: ignore[reportAttributeAccessIssue]
        logging.info(f"DEBUG Response Success: {response_dict.get('success', False)}") # pyright: ignore[reportAttributeAccessIssue]
        logging.info("DEBUG PIPELINE COMPLETE - END OF TRACE")
        logging.info("=" * 80)

        return response

    except Exception as e:
        execution_time = round(time.time() - start_time, 3)
        logging.error(f"DEBUG Query processing failed: {e}")
        logging.error(f"DEBUG Exception type: {type(e)}")
        logging.error(f"DEBUG Traceback: {traceback.format_exc()}")
        
        return create_error_response(
            error_message=str(e),
            error_code="UNHANDLED_ERROR",
            session_id=safe_get_attr(request, 'session_id', None)
        ).model_copy(update={"execution_time": execution_time})

# NEW: SQL Execution Endpoint with CORRECT pyodbc timeout handling
@app.post("/execute-sql")
async def execute_sql(request: dict):
    """
    Execute SQL query on the database and return results
    FIXED: Correct pyodbc timeout handling (connection.timeout, not cursor.settimeout)
    """
    start_time = time.time()
    
    try:
        sql = request.get("sql", "")
        session_id = request.get("session_id", "streamlit-session")
        max_rows = request.get("max_rows", 1000)
        timeout = request.get("timeout", FORCED_SQL_EXECUTION_TIMEOUT)  # FORCED timeout
        
        # FORCED: Use extended timeout instead of request value
        if timeout < FORCED_SQL_EXECUTION_TIMEOUT:
            timeout = FORCED_SQL_EXECUTION_TIMEOUT
            logging.info(f"SQL EXECUTION: Forced timeout to {timeout}s (was {request.get('timeout', 'default')})")
        
        logging.info(f"SQL EXECUTION: Starting execution for session {session_id}")
        logging.info(f"SQL EXECUTION: Query length: {len(sql)} characters")
        logging.info(f"SQL EXECUTION: Max rows: {max_rows}")
        logging.info(f"SQL EXECUTION: Timeout: {timeout}s (FORCED)")
        
        # Basic safety check - only allow SELECT queries
        sql_upper = sql.upper().strip()
        if not sql_upper.startswith("SELECT") and not sql_upper.startswith("WITH"):
            logging.warning(f"SQL EXECUTION: Rejected non-SELECT query: {sql[:100]}...")
            return {
                "success": False,
                "error": "Only SELECT and WITH (CTE) queries are allowed for security reasons",
                "error_type": "SECURITY_ERROR",
                "execution_time": time.time() - start_time
            }
        
        # Check for dangerous operations
        dangerous_ops = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'TRUNCATE', 'ALTER', 'CREATE', 'EXEC', 'xp_', 'sp_']
        for op in dangerous_ops:
            if op in sql_upper:
                logging.warning(f"SQL EXECUTION: Rejected dangerous operation '{op}': {sql[:100]}...")
                return {
                    "success": False,
                    "error": f"Dangerous SQL operation '{op}' not allowed. Only SELECT queries are permitted.",
                    "error_type": "SECURITY_ERROR", 
                    "execution_time": time.time() - start_time
                }
        
        # Create database connection using your existing .env credentials
        logging.info("SQL EXECUTION: Creating database connection...")
        conn_str = (
            f"DRIVER={{{os.getenv('db_driver', 'ODBC Driver 17 for SQL Server')}}};"
            f"SERVER={os.getenv('db_server')};"
            f"DATABASE={os.getenv('db_name')};"
            f"UID={os.getenv('db_user')};"
            f"PWD={os.getenv('db_password')};"
            "TrustServerCertificate=yes;"
            "Encrypt=yes;"
        )
        
        execution_start = time.time()
        
        # FIXED: Correct pyodbc timeout handling
        with pyodbc.connect(conn_str, timeout=timeout) as conn:  # Set timeout on connection
            # FIXED: Set query timeout on connection, not cursor
            conn.timeout = timeout  # This affects all cursors created from this connection
            logging.info(f"SQL EXECUTION: Connection timeout set to {timeout}s")
            
            with conn.cursor() as cursor:
                # REMOVED: cursor.settimeout(timeout) - This method doesn't exist!
                
                logging.info(f"SQL EXECUTION: Executing query with {timeout}s timeout (FORCED)...")
                cursor.execute(sql)
                
                # Get column names
                columns = [column[0] for column in cursor.description] if cursor.description else []
                
                # Fetch results (limited by max_rows)
                rows = cursor.fetchmany(max_rows)
                
                # Convert to list of dictionaries
                result = []
                for row in rows:
                    row_dict = {}
                    for i, value in enumerate(row):
                        # Handle special data types
                        if value is None:
                            row_dict[columns[i]] = None
                        elif isinstance(value, (int, float, str, bool)):
                            row_dict[columns[i]] = value
                        else:
                            # Convert other types to string
                            row_dict[columns[i]] = str(value)
                    result.append(row_dict)
                
                execution_time = time.time() - start_time
                query_execution_time = time.time() - execution_start
                
                logging.info(f"SQL EXECUTION: Success! Retrieved {len(result)} rows in {query_execution_time:.2f}s")
                logging.info(f"SQL EXECUTION: Columns: {columns}")
                
                return {
                    "success": True,
                    "data": result,
                    "row_count": len(result),
                    "columns": columns,
                    "execution_time": execution_time,
                    "query_execution_time": query_execution_time,
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "max_rows_applied": len(rows) == max_rows,
                    "sql_length": len(sql),
                    "timeout_used": f"{timeout}s (FORCED - PYODBC FIXED)"
                }
                
    except pyodbc.Error as db_error:
        execution_time = time.time() - start_time
        logging.error(f"SQL EXECUTION: Database error: {str(db_error)}")
        return {
            "success": False,
            "error": f"Database error: {str(db_error)}",
            "error_type": "DATABASE_ERROR",
            "execution_time": execution_time,
            "timestamp": time.time()
        }
    
    except Exception as e:
        execution_time = time.time() - start_time
        logging.error(f"SQL EXECUTION: Unexpected error: {str(e)}")
        logging.error(f"SQL EXECUTION: Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": f"Execution error: {str(e)}",
            "error_type": "EXECUTION_ERROR",
            "execution_time": execution_time,
            "timestamp": time.time()
        }


@app.get("/config")
async def get_app_config(current_config: ServerConfig = Depends(get_current_config)):
    """Get application configuration (non-sensitive parts) with FORCED timeout values"""
    component_timeouts = {
        "query_timeout": f"{FORCED_QUERY_TIMEOUT}s (FORCED)",
        "health_check_timeout": f"{FORCED_HEALTH_CHECK_TIMEOUT}s (FORCED)",
        "sql_execution_timeout": f"{FORCED_SQL_EXECUTION_TIMEOUT}s (FORCED)",
        "component_timeout": f"{FORCED_COMPONENT_TIMEOUT}s (FORCED)",
        "nlp_timeout": safe_get_attr(current_config, 'NLP_TIMEOUT', 30),
        "schema_timeout": safe_get_attr(current_config, 'SCHEMA_TIMEOUT', 30),
        "prompt_timeout": safe_get_attr(current_config, 'PROMPT_TIMEOUT', 30),
        "sql_timeout": safe_get_attr(current_config, 'SQL_TIMEOUT', 30)
    }

    dynamic_routing_config = {}
    if components.get('hybrid_orchestrator') and hasattr(components['hybrid_orchestrator'], 'dynamic_router'):
        dynamic_routing_config = {
            "enabled": True,
            "ai_classification_enabled": components['hybrid_orchestrator'].async_client_manager is not None,
            "adaptive_learning_enabled": True,
            "banking_domain_optimization": True
        }
    else:
        dynamic_routing_config = {"enabled": False, "method": "standard_orchestration"}

    return {
        "host": safe_get_attr(current_config, 'HOST', '0.0.0.0'),
        "port": safe_get_attr(current_config, 'PORT', 8000),
        "environment": "development" if not safe_get_attr(current_config, 'is_production', lambda: False)() else "production",
        "max_concurrent_requests": safe_get_attr(current_config, 'MAX_CONCURRENT_REQUESTS', 10),
        "caching_enabled": safe_get_attr(current_config, 'ENABLE_CACHING', True),
        "graceful_fallbacks": safe_get_attr(current_config, 'ENABLE_GRACEFUL_FALLBACKS', True),
        "components_available": list(COMPONENTS_AVAILABLE.keys()),
        "components_loaded": list(components.keys()),
        "timeouts": component_timeouts,
        "dynamic_routing": dynamic_routing_config,
        "debug_logging": True,
        "sql_execution_enabled": True,  # NEW: Show SQL execution capability
        "performance_monitoring": safe_get_attr(current_config, 'ENABLE_PERFORMANCE_MONITORING', False),
        "debug_mode": safe_get_attr(current_config, 'DEBUG_MODE', False),
        "timeout_improvements": f"FORCED to {FORCED_QUERY_TIMEOUT} seconds - NO MORE 30s TIMEOUTS!"
    }

@app.get("/components")
async def get_component_status():
    """Get detailed component status including AsyncClientManager and dynamic routing"""
    component_status = {
        "available_imports": {name: comp is not None for name, comp in COMPONENTS_AVAILABLE.items()},
        "initialized_components": {name: comp is not None for name, comp in components.items()},
        "component_count": len(components),
        "healthy_components": len([c for c in components.values() if c is not None]),
        "timeout_settings": f"{FORCED_QUERY_TIMEOUT} seconds for query processing (FORCED)",
        "sql_execution_available": True,  # NEW: Show SQL execution status
        "timeout_configuration": {
            "query_timeout": f"{FORCED_QUERY_TIMEOUT}s",
            "health_check_timeout": f"{FORCED_HEALTH_CHECK_TIMEOUT}s",
            "sql_execution_timeout": f"{FORCED_SQL_EXECUTION_TIMEOUT}s",
            "component_timeout": f"{FORCED_COMPONENT_TIMEOUT}s"
        }
    }

    if components.get('client_manager'):
        try:
            client_manager_status = components['client_manager'].get_client_status()
            component_status["client_manager"] = client_manager_status
            if hasattr(components['client_manager'], 'verify_singleton'):
                component_status["client_manager"]["singleton_verified"] = components['client_manager'].verify_singleton()
        except Exception as e:
            component_status["client_manager"] = {"error": str(e)}

    if components.get('hybrid_orchestrator') and hasattr(components['hybrid_orchestrator'], 'dynamic_router'):
        try:
            routing_stats = components['hybrid_orchestrator'].dynamic_router.get_routing_statistics()
            component_status["dynamic_routing"] = {
                "enabled": True,
                "statistics": routing_stats
            }
        except Exception as e:
            component_status["dynamic_routing"] = {"enabled": True, "error": str(e)}
    else:
        component_status["dynamic_routing"] = {"enabled": False}

    return component_status

# Log startup information with FORCED timeout values
logging.info("DEBUG: Enhanced FastAPI application initialized with comprehensive debugging")
logging.info(f"DEBUG: Configuration loaded: {config.__class__.__name__}")
logging.info(f"DEBUG: Available component imports: {list(COMPONENTS_AVAILABLE.keys())}")
logging.info(f"DEBUG: Query timeout FORCED to {FORCED_QUERY_TIMEOUT} seconds - NO MORE 30s TIMEOUTS!")
logging.info("DEBUG: SQL execution endpoint added at /execute-sql")  # NEW

if safe_get_attr(config, 'DEBUG_MODE', False):
    logging.info("DEBUG: Debug mode enabled")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=safe_get_attr(config, 'HOST', '0.0.0.0'),
        port=safe_get_attr(config, 'PORT', 8000),
        reload=safe_get_attr(config, 'DEBUG_MODE', False)
    )
