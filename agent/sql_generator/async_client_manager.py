"""
Fixed AsyncClientManager - Complete Script
FIXED: Race condition in async initialization 
FIXED: Better separation of concerns with modular design
MAINTAINS: All original functionality with improved architecture
"""

import asyncio
import logging
import time
import threading
import os
from typing import Dict, Optional, Any, List, Union
from dataclasses import dataclass
from datetime import datetime
import json
from dotenv import load_dotenv
import aiohttp

# Load environment variables
load_dotenv()

# Import dedicated clients with fallback
try:
    from orchestrator.clients.mathstral_client import MathstralClient
    from orchestrator.clients.deepseek_client import DeepSeekClient
    DEDICATED_CLIENTS_AVAILABLE = True
except ImportError:
    try:
        from orchestrator.clients.mathstral_client import MathstralClient
        from orchestrator.clients.deepseek_client import DeepSeekClient
        DEDICATED_CLIENTS_AVAILABLE = True
    except ImportError:
        logging.warning("Dedicated clients not available, falling back to NgrokClient")
        MathstralClient = None
        DeepSeekClient = None
        DEDICATED_CLIENTS_AVAILABLE = False

# ============================================================================
# COMPONENT 1: Health Tracking (Separated Concern)
# ============================================================================

@dataclass
class ClientHealth:
    """Focused health tracking for clients"""
    name: str
    failures: int = 0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    is_available: bool = True

    def record_failure(self):
        self.failures += 1
        self.last_failure = datetime.utcnow()
        # Circuit breaker: disable after 5 consecutive failures
        if self.failures >= 5:
            self.is_available = False

    def record_success(self):
        self.failures = 0
        self.last_success = datetime.utcnow()
        self.is_available = True

    def should_retry(self) -> bool:
        if not self.is_available and self.last_failure:
            # Re-enable after 2 minutes
            if (datetime.utcnow() - self.last_failure).total_seconds() > 120:
                self.is_available = True
        return self.is_available

# ============================================================================
# COMPONENT 2: Ngrok Client (Separated Concern)
# ============================================================================

class NgrokClient:
    """Focused HTTP client for ngrok endpoints"""
    
    def __init__(self, name: str, model_name: str, ngrok_url: str):
        self.client_name = name
        self.model_name = model_name
        self.ngrok_url = ngrok_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(f"NgrokClient.{name}")

    def set_session(self, session: aiohttp.ClientSession):
        """Set the aiohttp session"""
        self.session = session

    async def generate_sql_async(self, prompt: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Generate SQL using ngrok endpoint"""
        if not self.session:
            raise Exception(f"No session available for {self.client_name}")

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a SQL generation assistant specialized for Microsoft SQL Server. "
                        "Generate SPECIFIC queries that target only the requested data columns. "
                        "NEVER use SELECT * unless explicitly requested. "
                        "Always include appropriate WHERE clauses and LIMIT clauses for data safety. "
                        "Return ONLY a valid SQL query starting with SELECT, WITH, INSERT, UPDATE, or DELETE. "
                        "Do NOT include any explanation, apology, or additional text."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        # Add context if provided
        if context:
            payload["messages"][1]["content"] = f"Context: {context}\n\nQuery: {prompt}"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        try:
            self.logger.debug(f"Calling {self.ngrok_url} with model {self.model_name}")
            timeout = aiohttp.ClientTimeout(total=60)
            
            async with self.session.post(
                self.ngrok_url,
                json=payload,
                headers=headers,
                timeout=timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"HTTP error {response.status}: {error_text}")
                    raise Exception(f"HTTP {response.status}: {error_text}")

                try:
                    data = await response.json()
                except aiohttp.ContentTypeError as e:
                    error_text = await response.text()
                    self.logger.error(f"Invalid JSON response: {error_text}")
                    raise Exception(f"Invalid JSON response: {e}")

                # Extract SQL from response
                sql_query = data["choices"][0]["message"]["content"].strip()
                
                # Remove markdown code fences if present
                if sql_query.startswith("```"):
                    lines = sql_query.split("\n")
                    if len(lines) > 2:
                        sql_query = "\n".join(lines[1:-1]).strip()
                    else:
                        sql_query = sql_query.replace("```", "").strip()

                return {
                    "success": True,
                    "sql": sql_query,
                    "generated_sql": sql_query,
                    "confidence": 0.85,
                    "confidence_score": 0.85,
                    "model_used": self.model_name,
                    "tokens_used": data.get("usage", {}).get("total_tokens", 0)
                }

        except Exception as e:
            self.logger.error(f"Error calling {self.client_name}: {e}")
            return {
                "success": False,
                "sql": "",
                "error": str(e),
                "confidence": 0.0,
                "model_used": self.model_name
            }

    async def health_check(self) -> bool:
        """Simple health check for NgrokClient"""
        try:
            return bool(self.ngrok_url and self.ngrok_url.startswith('http'))
        except Exception:
            return False

    async def cleanup(self):
        """Cleanup client resources"""
        # Session cleanup is handled by the manager
        pass

# ============================================================================
# COMPONENT 3: Client Factory (Separated Concern)  
# ============================================================================

class ClientFactory:
    """Focused factory for creating different client types"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    async def create_dedicated_client(self, client_type: str) -> Optional[Any]:
        """Create dedicated clients (MathstralClient, DeepSeekClient)"""
        try:
            if client_type == 'mathstral' and MathstralClient:
                client = MathstralClient()
                self.logger.info("MathstralClient created successfully")
                return client
            elif client_type == 'deepseek' and DeepSeekClient:
                client = DeepSeekClient()
                self.logger.info("DeepSeekClient created successfully")
                return client
            else:
                raise ImportError(f"{client_type} client not available")
        except Exception as e:
            self.logger.error(f"Failed to create {client_type} client: {e}")
            return None

    async def create_ngrok_client(self, name: str, model_name: str, ngrok_url: str) -> Optional[NgrokClient]:
        """Create NgrokClient with session"""
        try:
            if not ngrok_url:
                raise ValueError(f"No ngrok URL provided for {name}")
                
            client = NgrokClient(name, model_name, ngrok_url)
            self.logger.info(f"{name} NgrokClient created with URL: {ngrok_url}")
            return client
        except Exception as e:
            self.logger.error(f"Failed to create {name} NgrokClient: {e}")
            return None

    async def create_session(self, name: str) -> Optional[aiohttp.ClientSession]:
        """Create aiohttp session for NgrokClient"""
        try:
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30
            )
            
            session_timeout = aiohttp.ClientTimeout(total=60)
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=session_timeout,
                headers={'User-Agent': 'SQL-AI-Agent/1.0'}
            )
            
            self.logger.debug(f"Session created for {name}")
            return session
        except Exception as e:
            self.logger.error(f"Failed to create session for {name}: {e}")
            return None

# ============================================================================
# COMPONENT 4: Initialization Manager (Separated Concern)
# ============================================================================

class InitializationManager:
    """Focused initialization logic with race condition fixes"""
    
    def __init__(self, client_manager):
        self.client_manager = client_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self.factory = ClientFactory()
        
        # RACE CONDITION FIX: Use Event and Task for thread-safe initialization
        self._init_event = asyncio.Event()
        self._init_task: Optional[asyncio.Task] = None
        self._init_exception: Optional[Exception] = None
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """FIXED: Thread-safe initialization without race conditions"""
        # Quick check without lock for performance
        if self._init_event.is_set():
            if self._init_exception:
                raise self._init_exception
            return

        async with self._init_lock:
            # Double-check after acquiring lock
            if self._init_event.is_set():
                if self._init_exception:
                    raise self._init_exception
                return

            # Start initialization task if not already started
            if self._init_task is None:
                self._init_task = asyncio.create_task(self._do_initialization())

        # Wait for initialization to complete
        try:
            await self._init_event.wait()
            if self._init_exception:
                raise self._init_exception
        except Exception as e:
            self._init_exception = e
            raise

    async def _do_initialization(self) -> None:
        """Actual initialization implementation"""
        try:
            self.logger.info("Starting AsyncClientManager initialization...")
            start_time = time.time()

            if DEDICATED_CLIENTS_AVAILABLE:
                await self._initialize_dedicated_clients()
            else:
                await self._initialize_ngrok_clients()

            initialization_time = time.time() - start_time
            successful_clients = len([h for h in self.client_manager.client_health.values() if h.is_available])
            
            self.logger.info(
                f"Initialization completed in {initialization_time:.2f}s. "
                f"Successfully initialized {successful_clients}/{len(self.client_manager.client_health)} clients"
            )

            # Signal completion
            self._init_event.set()

        except Exception as e:
            self._init_exception = e
            self._init_event.set()
            raise

    async def _initialize_dedicated_clients(self):
        """Initialize dedicated clients"""
        self.logger.info("Initializing dedicated clients...")
        
        # Initialize MathstralClient
        client = await self.factory.create_dedicated_client('mathstral')
        if client:
            self.client_manager.clients['mathstral'] = client
            self.client_manager.client_health['mathstral'] = ClientHealth('mathstral')
        else:
            self.client_manager.client_health['mathstral'] = ClientHealth('mathstral', is_available=False)

        # Initialize DeepSeekClient
        client = await self.factory.create_dedicated_client('deepseek')
        if client:
            self.client_manager.clients['deepseek'] = client
            self.client_manager.client_health['deepseek'] = ClientHealth('deepseek')
        else:
            self.client_manager.client_health['deepseek'] = ClientHealth('deepseek', is_available=False)

    async def _initialize_ngrok_clients(self):
        """Initialize NgrokClient fallbacks"""
        self.logger.info("Initializing NgrokClient fallbacks...")
        
        ngrok_urls = {
            'mathstral': os.getenv('MATHSTRAL_NGROK_URL'),
            'deepseek': os.getenv('DEEPSEEK_NGROK_URL')
        }
        
        default_ngrok_url = os.getenv('NGROK_URL', 'https://85bf6540c0f8.ngrok-free.app/v1/chat/completions')
        
        model_mappings = {
            'mathstral': 'mathstral-7b-v0.1',
            'deepseek': 'deepseek-coder-6.7b-instruct'
        }

        for client_name, model_name in model_mappings.items():
            ngrok_url = ngrok_urls.get(client_name) or default_ngrok_url
            
            if ngrok_url:
                client = await self.factory.create_ngrok_client(client_name, model_name, ngrok_url)
                session = await self.factory.create_session(client_name)
                
                if client and session:
                    client.set_session(session)
                    self.client_manager.clients[client_name] = client
                    self.client_manager.sessions[client_name] = session
                    self.client_manager.client_health[client_name] = ClientHealth(client_name)
                else:
                    self.client_manager.client_health[client_name] = ClientHealth(client_name, is_available=False)
            else:
                self.client_manager.client_health[client_name] = ClientHealth(client_name, is_available=False)

# ============================================================================
# COMPONENT 5: Main AsyncClientManager (Focused Core Logic)
# ============================================================================

class AsyncClientManager:
    """
    FIXED: Thread-safe singleton with race condition fixes
    IMPROVED: Better separation of concerns with focused responsibilities
    MAINTAINS: All original functionality
    """
    
    # Singleton implementation
    _instance: Optional['AsyncClientManager'] = None
    _instance_lock = threading.Lock()
    _class_initialized = False

    def __new__(cls):
        """Thread-safe singleton pattern"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Prevent multiple initialization of singleton instance"""
        if getattr(self, '_instance_initialized', False):
            return

        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Core data structures
        self.clients: Dict[str, Any] = {}
        self.client_health: Dict[str, ClientHealth] = {}
        self.sessions: Dict[str, aiohttp.ClientSession] = {}
        
        # SEPARATION OF CONCERNS: Delegate initialization to specialized manager
        self.init_manager = InitializationManager(self)
        
        # Cleanup management
        self._cleanup_lock = asyncio.Lock()
        self._cleanup_done = False
        self._cleanup_tasks: List[asyncio.Task] = []
        
        self._instance_initialized = True
        self.logger.info(f"AsyncClientManager singleton created (dedicated_clients={DEDICATED_CLIENTS_AVAILABLE})")

    # ========================================================================
    # SINGLETON MANAGEMENT METHODS
    # ========================================================================

    @classmethod
    def get_singleton_instance(cls) -> Optional['AsyncClientManager']:
        """Get the singleton instance if it exists"""
        return cls._instance

    @classmethod
    def ensure_singleton(cls) -> 'AsyncClientManager':
        """Get singleton instance or create if not exists"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def get_or_create_singleton(cls) -> 'AsyncClientManager':
        """Alias for ensure_singleton"""
        return cls.ensure_singleton()

    @classmethod
    def reset_singleton(cls):
        """Reset singleton (testing only)"""
        with cls._instance_lock:
            if cls._instance is not None:
                if hasattr(cls._instance, 'init_manager') and cls._instance.init_manager._init_event.is_set():
                    logging.warning("Resetting singleton without proper cleanup - may cause resource leaks")
                cls._instance = None
                cls._class_initialized = False

    def verify_singleton(self) -> bool:
        """Verify this is the correct singleton instance"""
        return self is AsyncClientManager._instance

    # ========================================================================
    # INITIALIZATION METHODS (DELEGATED)
    # ========================================================================

    async def initialize(self):
        """FIXED: Delegate to InitializationManager - no race conditions"""
        await self.init_manager.initialize()

    # ========================================================================
    # CLIENT MANAGEMENT METHODS
    # ========================================================================

    async def get_healthy_client(self, preferred_name: Optional[str] = None) -> Optional[Any]:
        """Get a healthy client, preferring the specified one"""
        await self.initialize()
        
        # Try preferred client first
        if preferred_name and preferred_name in self.clients:
            health = self.client_health[preferred_name]
            if health.should_retry():
                return self.clients[preferred_name]

        # Find any healthy client
        for name, client in self.clients.items():
            health = self.client_health[name]
            if health.should_retry():
                return client

        self.logger.warning("No healthy clients available")
        return None

    async def generate_sql_async(self, prompt: str, context: Optional[Dict[str, Any]] = None, target_llm: Optional[str] = None) -> Dict[str, Any]:
        """Generate SQL using available clients with fallback"""
        await self.initialize()
        
        client = await self.get_healthy_client(target_llm)
        if not client:
            return {
                "success": False,
                "sql": "",
                "error": "No healthy clients available",
                "confidence": 0.0,
                "model_used": "none"
            }

        try:
            if hasattr(client, 'generate_sql_async'):
                # Check client type and call appropriately
                if hasattr(client, 'client_name') or type(client).__name__ in ['MathstralClient', 'DeepSeekClient']:
                    result = await client.generate_sql_async(prompt, context=context)
                else:
                    result = await client.generate_sql_async(prompt, context)
            else:
                raise Exception(f"Client does not support generate_sql_async method")

            # Record success
            client_name = getattr(client, 'client_name', target_llm or 'unknown')
            if client_name in self.client_health:
                self.client_health[client_name].record_success()

            return result

        except Exception as e:
            # Record failure
            client_name = getattr(client, 'client_name', target_llm or 'unknown')
            if client_name in self.client_health:
                self.client_health[client_name].record_failure()

            return {
                "success": False,
                "sql": "",
                "error": str(e),
                "confidence": 0.0,
                "model_used": client_name
            }

    # ========================================================================
    # HEALTH AND STATUS METHODS
    # ========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        if not self.init_manager._init_event.is_set():
            return {
                "status": "not_initialized",
                "available": False,
                "clients": {},
                "healthy_count": 0,
                "total_clients": 0
            }

        client_status = {}
        healthy_count = 0

        for name, client in self.clients.items():
            try:
                # Check client health
                if hasattr(client, 'health_check'):
                    if asyncio.iscoroutinefunction(client.health_check):
                        is_healthy = await client.health_check()
                    else:
                        is_healthy = client.health_check()
                elif hasattr(client, 'async_health_check'):
                    is_healthy = await client.async_health_check()
                else:
                    is_healthy = True

                if is_healthy:
                    healthy_count += 1
                    if name in self.client_health:
                        self.client_health[name].record_success()
                else:
                    if name in self.client_health:
                        self.client_health[name].record_failure()

                # Build status info
                health_obj = self.client_health.get(name)
                client_status[name] = {
                    "healthy": is_healthy,
                    "failures": health_obj.failures if health_obj else 0,
                    "last_success": health_obj.last_success.isoformat() if health_obj and health_obj.last_success else None,
                    "last_failure": health_obj.last_failure.isoformat() if health_obj and health_obj.last_failure else None,
                    "available": health_obj.is_available if health_obj else True,
                    "client_type": type(client).__name__,
                    "dedicated_client": hasattr(client, 'is_offline')
                }

            except Exception as e:
                self.logger.error(f"Health check failed for {name}: {e}")
                if name in self.client_health:
                    self.client_health[name].record_failure()
                
                client_status[name] = {
                    "healthy": False,
                    "error": str(e),
                    "available": False,
                    "client_type": type(client).__name__ if client else "Unknown"
                }

        overall_status = "healthy" if healthy_count == len(self.clients) else ("degraded" if healthy_count > 0 else "unhealthy")

        return {
            "status": overall_status,
            "available": healthy_count > 0,
            "clients": client_status,
            "healthy_count": healthy_count,
            "total_clients": len(self.client_health),
            "sessions_active": len([s for s in self.sessions.values() if not s.closed]),
            "using_dedicated_clients": DEDICATED_CLIENTS_AVAILABLE,
            "singleton_instance_id": id(self),
            "is_singleton": self.verify_singleton()
        }

    def get_client_status(self) -> Dict[str, Any]:
        """Synchronous client status"""
        healthy_count = len([h for h in self.client_health.values() if h.should_retry()])
        healthy_clients = [name for name, h in self.client_health.items() if h.should_retry()]
        all_clients = list(self.clients.keys())

        return {
            "healthy_count": healthy_count,
            "total_clients": len(self.client_health),
            "healthy_clients": healthy_clients,
            "available_clients": healthy_clients,
            "all_clients": all_clients,
            "initialized": self.init_manager._init_event.is_set(),
            "is_singleton": self.verify_singleton(),
            "instance_id": id(self),
            "using_dedicated_clients": DEDICATED_CLIENTS_AVAILABLE
        }

    def get_available_clients(self) -> List[str]:
        """Get list of available client names"""
        available = [
            name for name, health in self.client_health.items()
            if health.should_retry()
        ]
        return [name for name in available if name in self.clients]

    def get_client_by_name(self, name: str) -> Optional[Any]:
        """Get specific client by name"""
        return self.clients.get(name)

    # ========================================================================
    # CLEANUP METHODS
    # ========================================================================

    async def cleanup(self):
        """Clean up all resources"""
        async with self._cleanup_lock:
            if self._cleanup_done:
                return

            self.logger.info("Cleaning up AsyncClientManager...")

            # Cancel cleanup tasks
            for task in self._cleanup_tasks:
                if not task.done():
                    task.cancel()

            # Close sessions
            for name, session in self.sessions.items():
                try:
                    if session and not session.closed:
                        await session.close()
                        self.logger.debug(f"Closed session for {name}")
                except Exception as e:
                    self.logger.warning(f"Error closing session for {name}: {e}")

            # Clean up clients
            for name, client in self.clients.items():
                try:
                    if hasattr(client, 'cleanup'):
                        await client.cleanup()
                    elif hasattr(client, 'close'):
                        await client.close()
                    self.logger.debug(f"Cleaned up {name} client")
                except Exception as e:
                    self.logger.warning(f"Error cleaning up {name} client: {e}")

            # Clear collections
            self.sessions.clear()
            self.clients.clear()
            self.client_health.clear()
            self._cleanup_done = True

            # Reset initialization state
            self.init_manager._init_event.clear()
            self.init_manager._init_task = None
            self.init_manager._init_exception = None

            self.logger.info("AsyncClientManager cleanup completed")

    # ========================================================================
    # DEBUGGING AND STATUS METHODS
    # ========================================================================

    @classmethod
    def get_instance_info(cls) -> Dict[str, Any]:
        """Get singleton instance information"""
        return {
            "instance_exists": cls._instance is not None,
            "instance_id": id(cls._instance) if cls._instance else None,
            "class_initialized": cls._class_initialized,
            "singleton_methods_available": True
        }

    def get_singleton_status(self) -> Dict[str, Any]:
        """Get detailed singleton status"""
        return {
            "is_singleton_instance": self.verify_singleton(),
            "instance_id": id(self),
            "class_instance_id": id(self._instance) if self._instance else None,
            "instance_initialized": getattr(self, '_instance_initialized', False),
            "async_initialized": self.init_manager._init_event.is_set(),
            "clients_count": len(self.clients),
            "health_tracking_count": len(self.client_health)
        }

    # ========================================================================
    # CONTEXT MANAGER SUPPORT
    # ========================================================================

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()

# ============================================================================
# FACTORY FUNCTIONS AND UTILITIES
# ============================================================================

async def create_client_manager() -> AsyncClientManager:
    """Factory function to create and initialize the client manager"""
    manager = AsyncClientManager()
    await manager.initialize()
    
    if not manager.verify_singleton():
        raise RuntimeError("AsyncClientManager singleton integrity compromised")
    
    return manager

def get_client_manager() -> AsyncClientManager:
    """Get the singleton client manager instance"""
    return AsyncClientManager()

def get_singleton_async_client_manager() -> Optional[AsyncClientManager]:
    """Get existing singleton instance without creating new one"""
    return AsyncClientManager.get_singleton_instance()

def ensure_async_client_manager() -> AsyncClientManager:
    """Ensure AsyncClientManager singleton exists and return it"""
    return AsyncClientManager.ensure_singleton()

# ============================================================================
# CONTEXT MANAGER FOR CLIENT ACCESS
# ============================================================================

async def get_client_context(preferred_name: Optional[str] = None):
    """Async context manager for getting a client"""
    class ClientContext:
        def __init__(self, manager: AsyncClientManager, preferred_name: Optional[str] = None):
            self.manager = manager
            self.preferred_name = preferred_name
            self.client = None

        async def __aenter__(self):
            self.client = await self.manager.get_healthy_client(self.preferred_name)
            return self.client

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass  # No specific cleanup needed

    manager = AsyncClientManager()
    await manager.initialize()
    return ClientContext(manager, preferred_name)

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'AsyncClientManager',
    'NgrokClient', 
    'ClientHealth',
    'ClientFactory',
    'InitializationManager',
    'create_client_manager',
    'get_client_manager',
    'get_client_context',
    'get_singleton_async_client_manager',
    'ensure_async_client_manager'
]

# ============================================================================
# TEST FUNCTION
# ============================================================================

async def test_async_client_manager():
    """Test function to validate functionality"""
    print(f"Testing AsyncClientManager with {'DEDICATED' if DEDICATED_CLIENTS_AVAILABLE else 'NGROK'} clients...")
    
    try:
        # Test singleton methods
        print("\n=== Testing Singleton Methods ===")
        initial_instance = AsyncClientManager.get_singleton_instance()
        print(f"Initial singleton instance: {initial_instance}")

        manager1 = AsyncClientManager.ensure_singleton()
        print(f"First ensure_singleton call: {id(manager1)}")

        manager2 = AsyncClientManager.ensure_singleton()
        print(f"Second ensure_singleton call: {id(manager2)}")
        print(f"Same instance: {manager1 is manager2}")

        # Initialize manager
        await manager1.initialize()

        # Test client status
        status = manager1.get_client_status()
        print(f"\n=== Client Status ===")
        print(f"Healthy clients: {status['healthy_clients']}")
        print(f"Available clients: {status['available_clients']}")
        print(f"Is singleton: {status['is_singleton']}")

        # Test health check
        health = await manager1.health_check()
        print(f"\n=== Health Check ===")
        print(f"Overall status: {health['status']}")
        print(f"Total clients: {health['total_clients']}")

        # Test SQL generation if clients available
        if status['healthy_clients']:
            print("\n=== Testing SQL Generation ===")
            result = await manager1.generate_sql_async("Show me all customers")
            print(f"SQL Generation Success: {result['success']}")
            if result['success']:
                print(f"Generated SQL: {result['sql'][:100]}...")
            else:
                print(f"Error: {result['error']}")

        # Test cleanup
        await manager1.cleanup()
        print("\nAsyncClientManager test completed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(test_async_client_manager())
