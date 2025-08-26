"""
Connection management for SQL Executor.
Handles MS SQL Server connections with pooling, health monitoring, and error recovery.
"""

import asyncio
import aioodbc
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from dataclasses import dataclass
import threading
import time
import pyodbc  # For direct sync testing if needed

from .config import get_database_config, get_executor_config
from .models import ConnectionInfo, ErrorType
from .error_handler import ErrorHandler

logger = logging.getLogger(__name__)  # type: ignore

@dataclass
class PoolStats:
    """
    Connection pool statistics.

    Attributes:
        total_connections (int): Total number of connections created.
        active_connections (int): Number of currently active (in use) connections.
        idle_connections (int): Number of idle (available) connections.
        failed_connections (int): Number of failed connection attempts.
        total_queries_executed (int): Total number of queries executed.
        average_connection_time_ms (float): Average time in milliseconds to create a connection.
        last_health_check (Optional[datetime]): Timestamp of last health check.
    """
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    total_queries_executed: int = 0
    average_connection_time_ms: float = 0.0
    last_health_check: Optional[datetime] = None

class Connection:
    """
    Wrapper for database connection with metadata.

    Args:
        connection: The underlying aioodbc connection object.
        pool_name (str): The name of the connection pool this connection belongs to.

    Attributes:
        connection: The raw aioodbc connection.
        pool_name (str): Pool name.
        created_at (datetime): Timestamp when the connection was created.
        last_used (datetime): Timestamp when the connection was last used.
        query_count (int): Number of queries executed on this connection.
        is_healthy (bool): Connection health status.
        connection_id (int): Unique identifier for the connection instance.
    """

    def __init__(self, connection, pool_name: str):
        self.connection = connection
        self.pool_name = pool_name
        self.created_at = datetime.utcnow()
        self.last_used = datetime.utcnow()
        self.query_count = 0
        self.is_healthy = True
        self.connection_id = id(connection)

    async def execute(self, query: str, params=None):
        """
        Execute a SQL query using this connection.

        Args:
            query (str): SQL query string.
            params (optional): Query parameters.

        Returns:
            Cursor object after executing the query.

        Raises:
            Exception: If query execution fails.
        """
        self.last_used = datetime.utcnow()
        self.query_count += 1
        
        cursor = await self.connection.cursor()
        try:
            if params:
                await cursor.execute(query, params)
            else:
                await cursor.execute(query)
            return cursor
        except Exception as e:
            await cursor.close()
            raise e

    async def close(self):
        """
        Close the database connection.

        Logs a warning if an error occurs during closing.
        """
        try:
            await self.connection.close()
        except Exception as e:
            logger.warning(f"Error closing connection {self.connection_id}: {e}")

class ConnectionPool:
    """
    Async connection pool for MS SQL Server.

    Args:
        name (str): Name of the connection pool.
        connection_string (str): ODBC connection string for database.
        min_size (int): Minimum number of connections to maintain.
        max_size (int): Maximum number of connections allowed.

    Attributes:
        available_connections (List[Connection]): List of idle connections.
        active_connections (Dict[int, Connection]): Dictionary of active connections keyed by connection_id.
        pool_lock (asyncio.Lock): Async lock for synchronizing pool operations.
        total_created (int): Total number of connections created.
        stats (PoolStats): Pool statistics.
    """

    def __init__(self, name: str, connection_string: str, min_size: int = 2, max_size: int = 10):
        self.name = name
        self.connection_string = connection_string
        self.min_size = min_size
        self.max_size = max_size
        
        self.available_connections: List[Connection] = []
        self.active_connections: Dict[int, Connection] = {}
        self.pool_lock = asyncio.Lock()
        self.total_created = 0
        self.stats = PoolStats()
        
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = None
        self._health_check_task = None
        
        self._creation_semaphore = asyncio.Semaphore(max_size)

    async def initialize(self):
        """
        Initialize the connection pool by creating minimum connections and starting health check.

        Raises:
            Exception: If pool initialization fails.
        """
        logger.info(f"Initializing connection pool '{self.name}' (min={self.min_size}, max={self.max_size})")
        
        try:
            for _ in range(self.min_size):
                connection = await self._create_connection()
                if connection:
                    self.available_connections.append(connection)
                    self.total_created += 1
            
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            logger.info(f"Connection pool '{self.name}' initialized with {len(self.available_connections)} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool '{self.name}': {e}")
            raise

    async def _create_connection(self) -> Optional[Connection]:
        """
        Create a new database connection with fallback for SSL issues.

        Returns:
            Connection: New Connection object if successful, else None.
        """
        attempts = [
            self.connection_string,  # Original string
            self.connection_string.replace("Encrypt=yes;", ""),  # Remove encryption
            self.connection_string.replace("Encrypt=yes;", "Encrypt=no;")  # Explicitly disable
        ]

        for attempt_str in attempts:
            try:
                start_time = time.time()
                
                raw_connection = await aioodbc.connect(dsn=attempt_str)
                
                connection_time = (time.time() - start_time) * 1000  # ms
                
                await self._test_connection(raw_connection)
                
                connection = Connection(raw_connection, self.name)
                
                self.stats.average_connection_time_ms = (
                    (self.stats.average_connection_time_ms * self.total_created + connection_time) /
                    (self.total_created + 1)
                )
                
                logger.debug(f"Created new connection {connection.connection_id} in {connection_time:.2f}ms using string: {attempt_str}")
                return connection
                
            except Exception as e:
                logger.warning(f"Connection attempt failed with string '{attempt_str}': {e}")

        logger.error(f"Failed to create connection for pool '{self.name}' after all attempts")
        self.stats.failed_connections += 1
        return None

    async def _test_connection(self, connection):
        """
        Test if the connection is alive by executing a simple query.

        Args:
            connection: The raw aioodbc connection.

        Raises:
            Exception: If connection test fails.
        """
        try:
            cursor = await connection.cursor()
            await cursor.execute("SELECT 1")
            await cursor.fetchone()
            await cursor.close()
        except Exception as e:
            await connection.close()
            raise e

    @asynccontextmanager
    async def acquire(self):
        """
        Async context manager to acquire a connection from the pool.

        Yields:
            Connection: A healthy connection from the pool.

        Raises:
            Exception: If no connection is available.
        """
        connection = None
        try:
            connection = await self._get_connection()
            if connection:
                yield connection
            else:
                raise Exception(f"Could not acquire connection from pool '{self.name}'")
        finally:
            if connection:
                await self._return_connection(connection)

    async def _get_connection(self) -> Optional[Connection]:
        """
        Retrieve a healthy connection from the pool or create a new one.

        Returns:
            Connection: Healthy connection if available, else None.
        """
        async with self.pool_lock:
            while self.available_connections:
                connection = self.available_connections.pop(0)
                if await self._is_connection_healthy(connection):
                    self.active_connections[connection.connection_id] = connection
                    self.stats.active_connections = len(self.active_connections)
                    self.stats.idle_connections = len(self.available_connections)
                    return connection
                else:
                    await connection.close()
            
            if len(self.active_connections) < self.max_size:
                async with self._creation_semaphore:
                    connection = await self._create_connection()
                    if connection:
                        self.active_connections[connection.connection_id] = connection
                        self.total_created += 1
                        self.stats.total_connections += 1
                        self.stats.active_connections = len(self.active_connections)
                        return connection
            
            logger.warning(f"Connection pool '{self.name}' exhausted (active: {len(self.active_connections)}, max: {self.max_size})")
            return None

    async def _return_connection(self, connection: Connection):
        """
        Return a connection to the pool or close it if unhealthy or pool is full.

        Args:
            connection (Connection): The connection to return.
        """
        async with self.pool_lock:
            if connection.connection_id in self.active_connections:
                del self.active_connections[connection.connection_id]
                if (await self._is_connection_healthy(connection) and 
                    len(self.available_connections) < self.max_size):
                    self.available_connections.append(connection)
                else:
                    await connection.close()
                
                self.stats.active_connections = len(self.active_connections)
                self.stats.idle_connections = len(self.available_connections)

    async def _is_connection_healthy(self, connection: Connection) -> bool:
        """
        Check if a connection is healthy by age and test query.

        Args:
            connection (Connection): The connection to test.

        Returns:
            bool: True if healthy, False otherwise.
        """
        try:
            age = datetime.utcnow() - connection.created_at
            if age > timedelta(hours=1):
                return False
            
            cursor = await connection.connection.cursor()
            await cursor.execute("SELECT 1")
            result = await cursor.fetchone()
            await cursor.close()
            
            connection.is_healthy = result is not None
            return connection.is_healthy
        except Exception as e:
            logger.debug(f"Connection {connection.connection_id} health check failed: {e}")
            connection.is_healthy = False
            return False

    async def _health_check_loop(self):
        """
        Background task to perform periodic health checks on pool connections.
        """
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error for pool '{self.name}': {e}")

    async def _perform_health_check(self):
        """
        Check all available connections and maintain minimum pool size.
        """
        logger.debug(f"Performing health check for pool '{self.name}'")
        
        async with self.pool_lock:
            healthy_connections = []
            for connection in self.available_connections:
                if await self._is_connection_healthy(connection):
                    healthy_connections.append(connection)
                else:
                    await connection.close()
            
            self.available_connections = healthy_connections
            self.stats.last_health_check = datetime.utcnow()
            
            while len(self.available_connections) < self.min_size:
                connection = await self._create_connection()
                if connection:
                    self.available_connections.append(connection)
                else:
                    break
            
            self.stats.idle_connections = len(self.available_connections)

    async def close_all(self):
        """
        Close all connections and cancel health checks.
        """
        logger.info(f"Closing connection pool '{self.name}'")
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        async with self.pool_lock:
            for connection in self.available_connections:
                await connection.close()
            self.available_connections.clear()
            
            for connection in self.active_connections.values():
                await connection.close()
            self.active_connections.clear()
        
        logger.info(f"Connection pool '{self.name}' closed")

    def get_stats(self) -> PoolStats:
        """
        Get current statistics of the connection pool.

        Returns:
            PoolStats: Current pool statistics.
        """
        self.stats.total_connections = len(self.available_connections) + len(self.active_connections)
        self.stats.active_connections = len(self.active_connections)
        self.stats.idle_connections = len(self.available_connections)
        return self.stats

class ConnectionManager:
    """
    Main connection manager orchestrator handling connection pools and queries.

    Attributes:
        db_config: Database configuration.
        executor_config: Executor configuration.
        error_handler: Error handling instance.
        read_pool (Optional[ConnectionPool]): Read-only connection pool.
        pools_initialized (bool): Whether pools are initialized.
        _shutdown_event (asyncio.Event): Shutdown event flag.
    """

    def __init__(self):
        self.db_config = get_database_config()
        self.executor_config = get_executor_config()
        self.error_handler = ErrorHandler()
        
        self.read_pool: Optional[ConnectionPool] = None
        self.pools_initialized = False
        
        self._shutdown_event = asyncio.Event()

    async def initialize(self):
        """
        Initialize all connection pools.

        Raises:
            Exception: If initialization fails.
        """
        if self.pools_initialized:
            logger.warning("Connection manager already initialized")
            return
        
        logger.info("Initializing connection manager...")
        
        try:
            self.read_pool = ConnectionPool(
                name="read_pool",
                connection_string=self.db_config.get_async_connection_string(),
                min_size=self.db_config.min_pool_size,
                max_size=self.db_config.max_pool_size
            )
            
            await self.read_pool.initialize()
            
            self.pools_initialized = True
            logger.info("Connection manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection manager: {e}")
            raise

    @asynccontextmanager
    async def get_read_connection(self):
        """
        Async context manager to get a read-only connection.

        Yields:
            Connection: A connection from the read pool.

        Raises:
            Exception: If pool is not initialized.
        """
        if not self.pools_initialized:
            await self.initialize()
        
        if not self.read_pool:
            raise Exception("Read pool not initialized")
        
        async with self.read_pool.acquire() as connection:
            yield connection

    @asynccontextmanager
    async def get_connection(self):
        """
        Async context manager to get a connection (alias for read connection).

        Yields:
            Connection: A database connection.
        """
        async with self.get_read_connection() as connection:
            yield connection

    async def execute_query(self, sql_query: str, params=None, request_id: Optional[str] = None):
        """
        Execute a SQL query with connection management and error handling.

        Args:
            sql_query (str): The SQL query to execute.
            params (optional): Query parameters.
            request_id (Optional[str]): Identifier for the request.

        Returns:
            List[Dict]: Query results as list of dicts.

        Raises:
            Exception: If query execution fails.
        """
        async def _execute_operation():
            async with self.get_connection() as connection:
                cursor = await connection.execute(sql_query, params)
                try:
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    rows = await cursor.fetchall()
                    results = []
                    for row in rows:
                        results.append({columns[i]: row[i] for i in range(len(columns))})
                    return results, columns
                finally:
                    await cursor.close()
        
        return await self.error_handler.execute_with_error_handling(
            _execute_operation, sql_query, request_id
        )

    async def test_connection(self) -> bool:
        """
        Test database connectivity by executing a simple query.

        Returns:
            bool: True if connection test passes, False otherwise.
        """
        try:
            async with self.get_connection() as connection:
                cursor = await connection.execute("SELECT 1 AS test_value")
                result = await cursor.fetchone()
                await cursor.close()
                return result is not None
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_connection_info(self) -> ConnectionInfo:
        """
        Get information about the current connection state.

        Returns:
            ConnectionInfo: Data object with connection metadata.
        """
        return ConnectionInfo(
            server=self.db_config.server,
            database=self.db_config.database,
            connected_at=datetime.utcnow(),
            is_healthy=self.pools_initialized
        )

    def get_pool_stats(self) -> Dict[str, PoolStats]:
        """
        Get statistics for all connection pools.

        Returns:
            Dict[str, PoolStats]: Mapping from pool names to their stats.
        """
        stats = {}
        if self.read_pool:
            stats['read_pool'] = self.read_pool.get_stats()
        return stats

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check of the connection manager.

        Returns:
            Dict[str, Any]: Health status including overall status and pool stats.
        """
        health_status = {
            'overall_healthy': True,
            'pools_initialized': self.pools_initialized,
            'connection_test': False,
            'pool_stats': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            health_status['connection_test'] = await self.test_connection()
            health_status['pool_stats'] = self.get_pool_stats()
            health_status['overall_healthy'] = (
                self.pools_initialized and
                health_status['connection_test']
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status['overall_healthy'] = False
            health_status['error'] = str(e)
        
        return health_status

    async def shutdown(self):
        """
        Gracefully shut down connection manager and close all pools.
        """
        logger.info("Shutting down connection manager...")
        
        self._shutdown_event.set()
        
        if self.read_pool:
            await self.read_pool.close_all()
        
        self.pools_initialized = False
        logger.info("Connection manager shutdown complete")

# Global connection manager instance
connection_manager = ConnectionManager()

async def get_connection():
    """
    Async generator to get a database connection.

    Yields:
        Connection: A database connection from the pool.
    """
    async with connection_manager.get_connection() as conn:
        yield conn

async def execute_query(sql: str, params=None):
    """
    Execute a query using the connection manager.

    Args:
        sql (str): SQL query string.
        params (optional): Query parameters.

    Returns:
        Query result as list of dictionaries.
    """
    return await connection_manager.execute_query(sql, params)

async def test_connectivity():
    """
    Test connectivity to the database.

    Returns:
        bool: True if connection is successful, False otherwise.
    """
    return await connection_manager.test_connection()

__all__ = [
    'ConnectionManager', 'ConnectionPool', 'Connection', 'PoolStats',
    'connection_manager', 'get_connection', 'execute_query', 'test_connectivity'
]
