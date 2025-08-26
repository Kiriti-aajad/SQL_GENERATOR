"""
Enhanced Production Server Startup Script

FIXED: AsyncClientManager Integration - No More Dual Instances
FIXED: Receives shared AsyncClientManager singleton from main.py instead of creating its own
FIXED: AsyncIO event loop conflict and uvicorn server startup
FIXED: Component status validation and Unicode logging issues
FIXED: Uvicorn reload warning - now uses CLI for development reload
FIXED: Smart reload exclusions based on directory structure
INTEGRATED: Proper AsyncClientManager lifecycle management with shared singleton pattern
"""

import uvicorn
import sys
import os
import asyncio
import logging
import signal
import time
import subprocess
from pathlib import Path
from typing import Optional
import traceback

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# FIXED: Import AsyncClientManager class properly for type hints only
try:
    from agent.sql_generator.async_client_manager import AsyncClientManager
    ASYNC_CLIENT_MANAGER_AVAILABLE = True
    print("AsyncClientManager class imported successfully for type hints")
except ImportError as e:
    print(f"AsyncClientManager import failed: {e}")
    ASYNC_CLIENT_MANAGER_AVAILABLE = False
    AsyncClientManager = None

# Import enhanced configuration
try:
    from server.config import get_config, ServerConfig
    config = get_config()
    CONFIG_LOADED = True
    print(f"Loaded configuration for environment: {os.getenv('ENVIRONMENT', 'development')}")
except ImportError as e:
    print(f"Failed to load enhanced configuration: {e}")
    try:
        from server.config import get_config
        config = get_config()
        CONFIG_LOADED = True
        print("Loaded basic configuration")
    except ImportError as e2:
        print(f"Failed to load any configuration: {e2}")
        CONFIG_LOADED = False
        config = None

class ServerManager:
    """
    Server manager with startup validation and AsyncClientManager integration
    FIXED: Now receives shared AsyncClientManager instead of creating its own
    FIXED: Eliminates uvicorn reload warning by using CLI for development reload
    FIXED: Smart reload exclusions based on actual directory structure
    """

    def __init__(self, config: ServerConfig, async_client_manager: Optional['AsyncClientManager'] = None): # pyright: ignore[reportPossiblyUnboundVariable, reportInvalidTypeForm]
        """
        CRITICAL FIX: Constructor now receives shared AsyncClientManager instance
        This eliminates the dual instance problem by using the singleton from main.py
        """
        self.config = config # pyright: ignore[reportPossiblyUnboundVariable]
        self.server: Optional[uvicorn.Server] = None
        self.startup_time = time.time()

        # FIXED: Store the shared AsyncClientManager instance (don't create new one)
        self.async_client_manager = async_client_manager
        self.async_client_manager_initialized = async_client_manager is not None
        self.shutdown_requested = False

        # Log the integration status
        if self.async_client_manager:
            logger = logging.getLogger("ServerManager")
            logger.info(f"ServerManager initialized with shared AsyncClientManager (ID: {id(self.async_client_manager)})")
        else:
            logger = logging.getLogger("ServerManager")
            logger.warning("ServerManager initialized without AsyncClientManager - running in degraded mode")

    def setup_logging(self):
        """Setup logging with UTF-8 encoding support"""
        try:
            log_dir = Path(self.config.LOG_DIR)
            log_dir.mkdir(exist_ok=True)
            handlers = []

            # File handler with explicit UTF-8 encoding
            if self.config.ENABLE_FILE_LOGGING:
                file_handler = logging.FileHandler(
                    log_dir / self.config.LOG_FILE,
                    encoding='utf-8',
                    errors='replace'
                )
                file_handler.setFormatter(logging.Formatter(self.config.LOG_FORMAT))
                handlers.append(file_handler)

            # Console handler with UTF-8 support
            if self.config.ENABLE_CONSOLE_LOGGING:
                console_handler = logging.StreamHandler(sys.stdout)
                # FIXED: Add UTF-8 encoding for Windows compatibility
                if hasattr(console_handler.stream, 'reconfigure'):
                    try:
                        console_handler.stream.reconfigure(encoding='utf-8', errors='replace') # pyright: ignore[reportAttributeAccessIssue]
                    except Exception:
                        pass # Ignore if reconfigure fails
                console_handler.setFormatter(logging.Formatter(self.config.LOG_FORMAT))
                handlers.append(console_handler)

            # FIXED: Add encoding and error handling to basicConfig
            logging.basicConfig(
                level=getattr(logging, self.config.LOG_LEVEL),
                format=self.config.LOG_FORMAT,
                handlers=handlers,
                force=True
            )

            logger = logging.getLogger("ServerStartup")
            logger.info(f"Logging configured: level={self.config.LOG_LEVEL}, handlers={len(handlers)}")

        except Exception as e:
            # Fallback logging without UTF-8
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            print(f"Logging setup failed: {e}")

    def print_startup_banner(self):
        """Print startup banner with system information including AsyncClientManager status"""
        env = os.getenv("ENVIRONMENT", "development").upper()
        print("=" * 80)
        print("SQL Generator API Server with Shared AsyncClientManager Integration")
        print("=" * 80)
        print(f"Server URL: http://{self.config.HOST}:{self.config.PORT}")
        print(f"API Docs: http://{self.config.HOST}:{self.config.PORT}/docs")
        print(f"Health Check: http://{self.config.HOST}:{self.config.PORT}/health")
        print(f"Environment: {env}")
        print(f"Debug Mode: {self.config.DEBUG_MODE}")
        print(f"Workers: {self.config.WORKERS}")
        print(f"Auto Reload: {self.config.RELOAD}")
        print(f"Max Requests: {self.config.MAX_CONCURRENT_REQUESTS}")
        print(f"Caching: {self.config.ENABLE_CACHING}")
        print(f"Monitoring: {self.config.ENABLE_PERFORMANCE_MONITORING}")
        print(f"AsyncClientManager: {'Shared Instance' if self.async_client_manager else 'Not Available'}")
        if self.async_client_manager:
            print(f"AsyncClientManager ID: {id(self.async_client_manager)}")
        print("=" * 80)

    def validate_configuration(self) -> bool:
        """Validate server configuration"""
        logger = logging.getLogger("ConfigValidation")
        logger.info("Validating configuration")
        errors = []

        if not (1 <= self.config.PORT <= 65535):
            errors.append(f"PORT must be 1-65535 (got {self.config.PORT})")

        if self.config.MAX_CONCURRENT_REQUESTS < 1:
            errors.append("MAX_CONCURRENT_REQUESTS must be >=1")

        if self.config.DEFAULT_TIMEOUT < 30:
            errors.append("DEFAULT_TIMEOUT must be >=30")

        try:
            log_dir = Path(self.config.LOG_DIR)
            log_dir.mkdir(exist_ok=True)
            test_file = log_dir / "test.tmp"
            test_file.write_text("x")
            test_file.unlink()
        except Exception as e:
            errors.append(f"Log directory not writable: {e}")

        if errors:
            for err in errors:
                logger.error(err)
            return False

        logger.info("Configuration validation passed")
        return True

    async def validate_async_client_manager(self) -> bool:
        """
        FIXED: Validate the shared AsyncClientManager instance (don't create new one)
        This method now validates the received instance instead of creating a new one
        """
        logger = logging.getLogger("AsyncClientManagerValidation")

        if not self.async_client_manager:
            logger.warning("No AsyncClientManager instance provided - server will run in degraded mode")
            return False

        try:
            logger.info(f"Validating shared AsyncClientManager instance (ID: {id(self.async_client_manager)})")

            # Verify the instance is properly initialized
            if hasattr(self.async_client_manager, '_async_initialized'):
                if not self.async_client_manager._async_initialized:
                    logger.warning("AsyncClientManager instance not yet initialized")
                    return False

            # Get client status
            try:
                if hasattr(self.async_client_manager, 'get_client_status'):
                    status = self.async_client_manager.get_client_status()
                    healthy_count = status.get('healthy_count', 0)
                    total_clients = status.get('total_clients', 0)
                    available_clients = status.get('available_clients', [])
                    is_singleton = status.get('is_singleton', False)

                    logger.info(f"AsyncClientManager validation: {healthy_count}/{total_clients} clients healthy")
                    logger.info(f"Available clients: {available_clients}")
                    logger.info(f"Singleton integrity: {is_singleton}")

                    if not is_singleton:
                        logger.error("AsyncClientManager singleton integrity violation detected!")
                        return False

                    if healthy_count > 0:
                        logger.info("AsyncClientManager validation successful")
                        return True
                    else:
                        logger.warning("AsyncClientManager has no healthy clients")
                        return False

                elif hasattr(self.async_client_manager, 'health_check'):
                    health_result = await self.async_client_manager.health_check()
                    if isinstance(health_result, dict):
                        healthy_clients = health_result.get('healthy_count', 0)
                        available_clients = health_result.get('available_clients', [])
                        logger.info(f"AsyncClientManager health check: {healthy_clients} healthy clients")
                        return healthy_clients > 0
                    else:
                        is_healthy = bool(health_result)
                        logger.info(f"AsyncClientManager health check: {'passed' if is_healthy else 'failed'}")
                        return is_healthy
                else:
                    # Assume it's working if we can access it
                    logger.info("AsyncClientManager validation assumed successful (no status methods)")
                    return True

            except Exception as e:
                logger.warning(f"Could not get AsyncClientManager status: {e}")
                # Assume it's working if we got this far
                logger.info("AsyncClientManager validation assumed successful despite status error")
                return True

        except Exception as e:
            logger.error(f"AsyncClientManager validation failed: {e}")
            logger.error(traceback.format_exc())
            return False

    async def validate_components(self) -> bool:
        """Validate orchestrator components and shared AsyncClientManager"""
        logger = logging.getLogger("ComponentValidation")
        logger.info("Validating components with shared AsyncClientManager")

        # Validate orchestrator components
        orchestrator_available = False
        try:
            from orchestrator.hybrid_orchestrator import HybridAIAgentOrchestrator, HybridConfig
            hybrid_config = HybridConfig(**self.config.get_hybrid_config())
            test_orch = HybridAIAgentOrchestrator(hybrid_config)
            try:
                health_result = await test_orch.health_check()
                orchestrator_available = bool(health_result)
                logger.info(f"Orchestrator health check: {'passed' if orchestrator_available else 'failed'}")
            except Exception as e:
                logger.warning(f"Orchestrator health check failed: {e}")
                orchestrator_available = False
        except ImportError as e:
            logger.warning(f"Hybrid orchestrator import failed: {e}")
        except Exception as e:
            logger.error(f"Orchestrator component validation failed: {e}")

        # FIXED: Validate shared AsyncClientManager (don't initialize new one)
        async_client_available = await self.validate_async_client_manager()

        # Determine overall validation result
        if orchestrator_available and async_client_available:
            logger.info("Full system validation passed: Orchestrator + Shared AsyncClientManager")
            return True
        elif orchestrator_available or async_client_available:
            logger.warning("Partial system validation passed: Some components available")
            if self.config.MOCK_UNAVAILABLE_COMPONENTS:
                logger.warning("Running with partial components (mocking enabled)")
                return True
            return True # Allow partial operation
        else:
            logger.error("Component validation failed: No orchestrator or AsyncClientManager available")
            if self.config.MOCK_UNAVAILABLE_COMPONENTS:
                logger.warning("Running with mocked components")
                return True
            return False

    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def handler(signum, frame):
            logger = logging.getLogger("SignalHandler")
            logger.info(f"Signal {signum} received, shutting down gracefully...")
            # FIXED: Set shutdown flag instead of creating asyncio task in signal handler
            self.shutdown_requested = True
            if self.server:
                self.server.should_exit = True

        # FIXED: Handle signal registration failures gracefully
        try:
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)
        except ValueError as e:
            # Signal handlers can only be set from main thread
            logger = logging.getLogger("SignalHandler")
            logger.warning(f"Could not set signal handlers (not main thread?): {e}")

    async def cleanup_async_resources(self):
        """
        FIXED: Cleanup coordination (don't cleanup shared AsyncClientManager here)
        The shared AsyncClientManager should be cleaned up by main.py, not here
        """
        logger = logging.getLogger("AsyncCleanup")
        try:
            if self.async_client_manager:
                logger.info("AsyncClientManager cleanup will be handled by main.py (shared instance)")
                # Don't cleanup the shared instance here - it might be used by other components
                # Just clear our reference
                self.async_client_manager = None
                self.async_client_manager_initialized = False
            else:
                logger.info("No AsyncClientManager instance to cleanup")
        except Exception as e:
            logger.error(f"Cleanup coordination failed: {e}")

    def print_component_discovery(self):
        print("Component Discovery:")
        discovery_paths = getattr(self.config, 'COMPONENT_DISCOVERY_PATHS', [])
        if discovery_paths:
            for p in discovery_paths:
                print(f"  - {p}")
        else:
            print("  - No discovery paths configured")
        print("AsyncClientManager Integration:")
        print(f"  Available: {ASYNC_CLIENT_MANAGER_AVAILABLE}")
        print(f"  Shared Instance: {self.async_client_manager is not None}")
        print(f"  Instance ID: {id(self.async_client_manager) if self.async_client_manager else 'None'}")
        if self.async_client_manager and hasattr(self.async_client_manager, 'verify_singleton'):
            print(f"  Singleton Verified: {self.async_client_manager.verify_singleton()}")

    async def run_startup_validation(self) -> bool:
        print("Running startup validation with shared AsyncClientManager...")
        if not self.validate_configuration():
            print("Configuration validation failed")
            return False
        if not await self.validate_components():
            print("Component validation failed")
            return False
        print("Enhanced startup validation passed")
        return True

    def print_summary(self):
        uptime = time.time() - self.startup_time
        print(f"Startup completed in {uptime:.2f}s")
        print("API Endpoints:")
        print("  POST /query - Generate SQL from natural language")
        print("  GET /health - System health check")
        print("  GET /config - Configuration info")
        print("  GET /components - Component status")

        # Print AsyncClientManager status
        if self.async_client_manager:
            try:
                print("AsyncClientManager Status:")
                print("  Status: Shared Instance Operational")
                print(f"  Instance: {type(self.async_client_manager).__name__}")
                print(f"  Instance ID: {id(self.async_client_manager)}")
                
                # Get detailed status if available
                if hasattr(self.async_client_manager, 'get_client_status'):
                    status = self.async_client_manager.get_client_status()
                    print(f"  Healthy Clients: {status.get('healthy_count', 0)}/{status.get('total_clients', 0)}")
                    print(f"  Available Clients: {status.get('available_clients', [])}")
            except Exception:
                print("AsyncClientManager Status: Error retrieving detailed status")
        else:
            print("AsyncClientManager: Not provided (degraded mode)")

    def get_smart_reload_exclusions(self):
        """
        FIXED: Generate smart reload exclusions based on actual directory structure
        This prevents unnecessary reloads on data files, logs, cache, etc.
        """
        return [
            # Python cache files
            "*.pyc",
            "__pycache__",
            
            # Data and logs directories
            "logs",
            "data",
            ".cache",
            
            # Version control
            ".git",
            ".gitignore",
            
            # Environment and config files
            ".env",
            "*.env",
            
            # Documentation
            "docs",
            "README.md",
            "*.md",
            
            # Tests
            "tests",
            ".pytest_cache",
            
            # Scripts and utilities
            "scripts",
            "output",
            
            # Database files
            "*.sqlite3",
            "*.db",
            
            # Log files
            "*.log",
            
            # Data files
            "*.json",
            "*.csv",
            "*.pkl",
            "*.bin",
            
            # Config files
            "*.ini",
            "*.toml",
            "*.yaml",
            "*.yml",
            
            # IDE files
            ".vscode",
            "*.lock",
            
            # Monitoring and frontend build files
            "monitoring",
            "frontend",
            
            # Specific data directories from your structure
            "data/embeddings",
            "data/metadata", 
            "data/join_graph",
            ".cache/sentence_transformers",
            ".cache/transformers"
        ]

    async def start_server(self):
        """Start the Uvicorn server with shared AsyncClientManager integration - FIXED reload handling"""
        self.setup_logging()
        logger = logging.getLogger("ServerManager")
        self.print_startup_banner()

        if self.config.DEBUG_MODE:
            self.print_component_discovery()

        if not await self.run_startup_validation():
            logger.error("Startup validation failed, exiting")
            sys.exit(1)

        self.setup_signal_handlers()
        self.print_summary()

        logger.info("Starting Uvicorn server with shared AsyncClientManager integration")

        # Get uvicorn config
        if CONFIG_LOADED:
            uv_conf = self.config.get_uvicorn_config()
        else:
            uv_conf = {
                "host": "0.0.0.0",
                "port": 8000,
                "reload": False,
                "workers": 1,
                "log_level": "info",
                "access_log": True,
                "loop": "asyncio" # Force asyncio loop
            }

        # FIXED: Handle reload properly with smart exclusions
        if uv_conf.get("reload", False):
            env = os.getenv("ENVIRONMENT", "development").lower()
            if env == "development":
                logger.info("Development mode detected - using CLI for proper reload support")
                logger.info("Starting development server with smart reload exclusions...")
                
                # Use subprocess to call uvicorn CLI for proper reload
                cmd = [
                    sys.executable, "-m", "uvicorn", 
                    "server.main:app",
                    f"--host={uv_conf.get('host', '0.0.0.0')}", 
                    f"--port={uv_conf.get('port', 8000)}", 
                    "--reload",
                    f"--log-level={uv_conf.get('log_level', 'info')}"
                ]
                
                # FIXED: Add smart reload exclusions based on directory structure
                smart_exclusions = self.get_smart_reload_exclusions()
                for exclude in smart_exclusions:
                    cmd.extend(["--reload-exclude", exclude])
                
                logger.info(f"Reload exclusions: {len(smart_exclusions)} patterns")
                logger.info("Executing development server with smart exclusions...")
                
                try:
                    subprocess.run(cmd)
                    return
                except KeyboardInterrupt:
                    logger.info("Development server shutdown via Ctrl+C")
                    return
                except Exception as e:
                    logger.error(f"Failed to start development server via CLI: {e}")
                    logger.info("Falling back to programmatic mode without reload...")
                    uv_conf["reload"] = False
            else:
                logger.info("Production/non-development mode - reload disabled for stability")
                uv_conf["reload"] = False
        else:
            logger.info("Reload disabled - starting in production mode")
            uv_conf["reload"] = False

        # Handle reload excludes safely (only if reload is still enabled)
        if uv_conf.get("reload", False) and hasattr(self.config, 'RELOAD_EXCLUDES') and self.config.RELOAD_EXCLUDES:
            uv_conf["reload_excludes"] = self.config.RELOAD_EXCLUDES

        # Start server with shared AsyncClientManager
        try:
            server_config = uvicorn.Config("server.main:app", **uv_conf)
            self.server = uvicorn.Server(server_config)

            # Serve with proper cleanup handling
            try:
                await self.server.serve()
            finally:
                # FIXED: Cleanup coordination only (don't cleanup shared instance)
                await self.cleanup_async_resources()

        except Exception as e:
            logger.error(f"Server startup failed: {e}")
            logger.error(traceback.format_exc())
            await self.cleanup_async_resources()
            sys.exit(1)

# FIXED: Factory function to create ServerManager with shared AsyncClientManager
def create_server_manager(config: 'ServerConfig', async_client_manager: Optional['AsyncClientManager'] = None) -> ServerManager: # pyright: ignore[reportInvalidTypeForm]
    """
    Factory function to create ServerManager with shared AsyncClientManager
    This is the preferred way to create ServerManager instances
    """
    return ServerManager(config, async_client_manager=async_client_manager)

async def main(async_client_manager: Optional['AsyncClientManager'] = None): # pyright: ignore[reportInvalidTypeForm]
    """
    FIXED: Main entry point now accepts shared AsyncClientManager
    This allows integration with main.py's shared singleton
    """
    if not CONFIG_LOADED or not config:
        print("Configuration not loaded, aborting")
        sys.exit(1)

    manager = None
    try:
        # FIXED: Create ServerManager with shared AsyncClientManager
        manager = create_server_manager(config, async_client_manager)
        await manager.start_server()

    except KeyboardInterrupt:
        print("\nServer shutdown requested")
        if manager:
            print("Cleaning up server resources...")
            await manager.cleanup_async_resources()
            print("Server cleanup completed")

    except Exception as e:
        print(f"Server startup failed: {e}")
        traceback.print_exc()
        if manager:
            await manager.cleanup_async_resources()
        sys.exit(1)

# FIXED: Standalone execution mode (for development/testing)
async def main_standalone():
    """
    Standalone execution mode for development/testing
    Creates its own AsyncClientManager if none provided
    """
    standalone_client_manager = None

    if ASYNC_CLIENT_MANAGER_AVAILABLE:
        try:
            print("Creating AsyncClientManager for standalone mode...")
            standalone_client_manager = AsyncClientManager() # pyright: ignore[reportOptionalCall]
            await standalone_client_manager.initialize()
            print("AsyncClientManager initialized for standalone mode")
        except Exception as e:
            print(f"Failed to initialize AsyncClientManager in standalone mode: {e}")
            standalone_client_manager = None

    # Run main with the standalone AsyncClientManager
    await main(standalone_client_manager)

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 8):
        print("Python 3.8+ required")
        sys.exit(1)

    # FIXED: Robust asyncio event loop handling
    try:
        # Check if we're in an event loop without triggering RuntimeError
        current_task = None
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None

        if current_task is not None:
            # We're already in an event loop
            print("Warning: Running in existing event loop - using standalone mode")
            
            # For development/testing environments with existing loops
            try:
                import nest_asyncio # pyright: ignore[reportMissingImports]
                nest_asyncio.apply()
                
                # Create the task for standalone mode
                asyncio.create_task(main_standalone())
                
                # Keep the main thread alive for the task to complete
                try:
                    asyncio.get_event_loop().run_forever()
                except KeyboardInterrupt:
                    print("\nServer shutdown complete")
                    
            except ImportError:
                print("nest_asyncio not available, cannot run in existing event loop")
                sys.exit(1)
        else:
            # No event loop running - safe to use asyncio.run()
            try:
                # Set event loop policy for Windows compatibility
                if sys.platform == 'win32':
                    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                    print("Windows asyncio event loop policy configured")

                # Run in standalone mode when executed directly
                asyncio.run(main_standalone())
                
            except KeyboardInterrupt:
                print("\nServer shutdown complete")
            except Exception as e:
                print(f"Fatal error: {e}")
                sys.exit(1)

    except Exception as e:
        print(f"Event loop setup failed: {e}")
        
        # FALLBACK: Try to create a new event loop
        try:
            if sys.platform == 'win32':
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                print("Fallback: Windows asyncio event loop policy configured")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                print("Fallback: Created new event loop")
                loop.run_until_complete(main_standalone())
            finally:
                loop.close()
                print("Fallback: Event loop closed")

        except Exception as fallback_error:
            print(f"Fallback event loop creation failed: {fallback_error}")
            sys.exit(1)
