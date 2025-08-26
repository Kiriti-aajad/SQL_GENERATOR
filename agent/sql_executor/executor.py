"""
Main SQL Executor Engine with robust DB connection handling and debugging
"""


import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import uuid
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import pyodbc  # For direct connection test


# Load env vars early
load_dotenv()


from .models import (
    ExecutionRequest, ExecutionResult, ExecutionStatus, ErrorType, OutputFormat,
    QueryMetadata, ColumnInfo, PerformanceStats,
    create_success_result, create_empty_result, create_error_result
)
from .connection_manager import ConnectionManager
from .formatters import ResultFormatter
from .error_handler import ErrorHandler
from .config import get_executor_config, get_database_config


logger = logging.getLogger(__name__) # type: ignore


class QueryProcessor:
    """Processes SQL queries and extracts metadata"""


    def __init__(self):
        self.config = get_executor_config()


    def preprocess_query(self, sql_query: str) -> str:
        processed_query = sql_query.strip()
        if processed_query.endswith(';'):
            processed_query = processed_query[:-1]
        if self.config.max_result_rows and self._should_add_limit(processed_query):
            processed_query = self._add_row_limit(processed_query, self.config.max_result_rows)
        return processed_query


    def _should_add_limit(self, query: str) -> bool:
        query_upper = query.upper()
        if 'TOP ' in query_upper or 'OFFSET' in query_upper or 'FETCH' in query_upper:
            return False
        return query_upper.startswith('SELECT')


    def _add_row_limit(self, query: str, max_rows: int) -> str:
        if query.upper().startswith('SELECT'):
            if query.upper().startswith('SELECT DISTINCT'):
                return query.replace('SELECT DISTINCT', f'SELECT DISTINCT TOP {max_rows}', 1)
            else:
                return query.replace('SELECT', f'SELECT TOP {max_rows}', 1)
        return query


    def extract_column_info(self, cursor) -> List[ColumnInfo]:
        columns = []
        if cursor.description:
            for i, desc in enumerate(cursor.description):
                column = ColumnInfo(
                    name=desc[0],
                    data_type=self._get_data_type_name(desc[1]) if len(desc) > 1 else 'unknown',
                    max_length=desc[3] if len(desc) > 3 else None,
                    is_nullable=desc[6] if len(desc) > 6 else True,
                    ordinal_position=i + 1
                )
                columns.append(column)
        return columns


    def extract_column_info_from_data(self, columns: List[str]) -> List[ColumnInfo]:
        return [
            ColumnInfo(
                name=col,
                data_type="unknown",
                max_length=None,
                is_nullable=True,
                ordinal_position=i + 1
            )
            for i, col in enumerate(columns)
        ]


    def _get_data_type_name(self, type_code) -> str:
        type_mapping = {
            -5: 'bigint',
            -7: 'bit',
            1: 'varchar',
            -9: 'nvarchar',
            2: 'numeric',
            3: 'decimal',
            4: 'int',
            5: 'smallint',
            6: 'float',
            8: 'double',
            9: 'datetime',
        }
        return type_mapping.get(type_code, 'unknown')


class ExecutionMonitor:
    def __init__(self):
        self.execution_stats: Dict[str, Any] = {}


    def start_execution(self, request_id: str) -> Dict[str, Any]:
        start_time = time.time()
        execution_context = {
            'request_id': request_id,
            'start_time': start_time,
            'start_datetime': datetime.utcnow()
        }
        self.execution_stats[request_id] = execution_context
        return execution_context


    def end_execution(self, request_id: str, row_count: int = 0) -> PerformanceStats:
        if request_id not in self.execution_stats:
            return PerformanceStats(execution_time_ms=0, rows_returned=row_count)


        context = self.execution_stats[request_id]
        end_time = time.time()


        execution_time_ms = int((end_time - context['start_time']) * 1000)


        del self.execution_stats[request_id]


        return PerformanceStats(
            execution_time_ms=execution_time_ms,
            rows_returned=row_count,
            connection_time_ms=0
        )


class SQLExecutor:
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.error_handler = ErrorHandler()
        self.formatter = ResultFormatter()
        self.query_processor = QueryProcessor()
        self.monitor = ExecutionMonitor()
        self.config = get_executor_config()
        self.db_config = get_database_config()


        self.is_initialized = False
        self.total_queries_executed = 0
        self.successful_queries = 0
        self.failed_queries = 0


        self.thread_pool = ThreadPoolExecutor(max_workers=5)


    def _build_connection_string(self, encrypt: Optional[str] = None, trust_cert: Optional[str] = None) -> str:
        db_user = os.getenv("db_user")
        db_password = os.getenv("db_password")
        db_server = os.getenv("db_server")
        db_name = os.getenv("db_name")
        db_driver = os.getenv("db_driver", "ODBC Driver 17 for SQL Server")


        if not all([db_user, db_password, db_server, db_name, db_driver]):
            raise ValueError("Missing required DB credentials in environment variables")


        parts = [
            f"DRIVER={{{db_driver}}}",
            f"SERVER={db_server}",
            f"DATABASE={db_name}",
            f"UID={db_user}",
            f"PWD={db_password}"
        ]


        if encrypt is not None:
            parts.append(f"Encrypt={encrypt}")


        if trust_cert is not None:
            parts.append(f"TrustServerCertificate={trust_cert}")


        connection_string = ";".join(parts) + ";"
        return connection_string


    def _test_pyodbc_connection(self, connection_string: str) -> bool:
        try:
            masked = connection_string.replace(os.getenv('db_password', ''), '***')
            logger.debug(f"Testing DB connection with connection string (masked): {masked}")
            conn = pyodbc.connect(connection_string, timeout=5)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            logger.debug(f"DB connection test successful, query result: {row[0]}") # type: ignore
            return True
        except Exception as e:
            logger.warning(f"DB connection test failed: {e}")
            return False


    async def initialize(self):
        if self.is_initialized:
            logger.warning("SQL Executor already initialized")
            return


        logger.info("Initializing SQL Executor with robust connection testing...")


        attempts = [
            {"encrypt": None, "trust_cert": None},  # No encryption (default unencrypted)
            {"encrypt": "no", "trust_cert": None},  # Explicitly disable encryption
            {"encrypt": "yes", "trust_cert": "yes"}  # Encrypted with trust bypass
        ]


        last_error = None
        for attempt_no, params in enumerate(attempts, 1):
            try:
                conn_str = self._build_connection_string(params.get("encrypt"), params.get("trust_cert"))
                if self._test_pyodbc_connection(conn_str):
                    logger.info(f"Connection successful on attempt #{attempt_no} with params: {params}")


                    self.connection_manager.connection_string = conn_str # type: ignore
                    await self.connection_manager.initialize()


                    is_connected = False
                    retries = 5
                    for i in range(retries):
                        logger.debug(f"Testing connection via connection manager (try {i+1}/{retries})...")
                        try:
                            is_connected = await self.connection_manager.test_connection()
                            if is_connected:
                                logger.info("Connection manager test connection succeeded!")
                                break
                        except Exception as err:
                            logger.warning(f"Connection manager test connection failed: {err}")
                        await asyncio.sleep(1)


                    if not is_connected:
                        raise ConnectionError("Connection manager failed to establish connection after retries")


                    self.is_initialized = True
                    logger.info("SQL Executor initialized successfully")
                    return
                else:
                    logger.warning(f"Connection test failed on attempt #{attempt_no} with params: {params}")
            except Exception as e:
                last_error = e
                logger.error(f"Error during connection attempt #{attempt_no}: {e}")


        raise ConnectionError(f"Failed to establish database connectivity after retries. Last error: {last_error}")


    async def execute_query(self, request: ExecutionRequest) -> ExecutionResult:
        if not self.is_initialized:
            await self.initialize()


        if not request.request_id:
            request.request_id = str(uuid.uuid4())


        logger.info(f"Executing query for user {request.user_id} [Request: {request.request_id}]")


        self.total_queries_executed += 1


        try:
            validation_result = await self._validate_request(request)
            if validation_result:
                self.failed_queries += 1
                return validation_result


            # ✅ FIXED: Handle both string and enum types for execution_mode
            execution_mode = request.execution_mode
            if hasattr(execution_mode, 'value'):
                mode_value = execution_mode.value
            else:
                mode_value = str(execution_mode)

            if mode_value == 'dry_run':
                return await self._dry_run_execution(request)
            elif mode_value == 'explain_plan':
                return await self._explain_plan_execution(request)
            else:
                return await self._execute_query_full(request)


        except Exception as e:
            self.failed_queries += 1
            logger.exception("Unexpected error in execute_query")
            return self.error_handler.handle_execution_error(e, request.sql_query, request.request_id)


    async def execute(self, sql_query: str) -> Dict[str, Any]:
        request = ExecutionRequest(
            sql_query=sql_query,
            user_id="orchestrator",
            request_id=str(uuid.uuid4()),
            output_format=OutputFormat.JSON,
            execution_mode="full", # type: ignore
            timeout_seconds=30,
        )


        response = await self.execute_query(request)


        return {
            "status": response.status,
            "data": response.data,
            "message": response.message,
            "row_count": response.row_count,
        }


    async def _validate_request(self, request: ExecutionRequest) -> Optional[ExecutionResult]:
        readonly_result = self.error_handler.validate_readonly_query(
            request.sql_query,
            request.request_id
        )
        if readonly_result:
            return readonly_result


        if not request.sql_query.strip():
            return create_error_result(
                error_type=ErrorType.SYNTAX_ERROR,
                error_message="Empty query provided",
                suggestions=["Provide a valid SQL query"],
                request_id=request.request_id
            )


        if request.output_format not in [fmt for fmt in OutputFormat]:
            return create_error_result(
                error_type=ErrorType.SYNTAX_ERROR,
                error_message=f"Unsupported output format: {request.output_format.value}",
                suggestions=["Use supported formats: json, csv, html, excel"],
                request_id=request.request_id
            )


        return None


    async def _dry_run_execution(self, request: ExecutionRequest) -> ExecutionResult:
        processed_query = self.query_processor.preprocess_query(request.sql_query)


        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            success=True,
            message="Query validation successful (dry run)",
            data=[],
            row_count=0,
            request_id=request.request_id,
            suggestions=[
                "Query syntax appears valid",
                "No READ-ONLY violations detected",
                f"Processed query: {processed_query[:100]}{'...' if len(processed_query) > 100 else ''}"
            ]
        )


        return result


    async def _explain_plan_execution(self, request: ExecutionRequest) -> ExecutionResult:
        try:
            explain_query = f"SET SHOWPLAN_ALL ON; {request.sql_query}"


            async with self.connection_manager.get_connection() as connection:
                cursor = await connection.execute(explain_query)


                plan_data = []
                async for row in cursor:
                    row_dict = {cursor.description[i][0]: row[i] for i in range(len(cursor.description))}
                    plan_data.append(row_dict)


                await cursor.close()


            columns = self.query_processor.extract_column_info(cursor) if cursor.description else []


            result = create_success_result(
                data=plan_data,
                columns=columns,
                execution_time_ms=0,
                request_id=request.request_id
            )
            result.message = "Query execution plan retrieved successfully"


            return result


        except Exception as e:
            logger.exception("Failed to retrieve execution plan")
            return self.error_handler.handle_execution_error(e, request.sql_query, request.request_id)


    async def _execute_query_full(self, request: ExecutionRequest) -> ExecutionResult:
        execution_context = self.monitor.start_execution(request.request_id) # type: ignore


        try:
            processed_query = self.query_processor.preprocess_query(request.sql_query)
            logger.debug(f"Executing processed query: {processed_query}")


            result_data, columns = await asyncio.wait_for(
                self._execute_with_connection(processed_query),
                timeout=request.timeout_seconds or 30
            )


            performance = self.monitor.end_execution(request.request_id, len(result_data)) # type: ignore


            if not result_data:
                result = self.error_handler.handle_empty_result(
                    processed_query,
                    performance.execution_time_ms,
                    request.request_id
                )
                self.successful_queries += 1
                return result


            column_info = self.query_processor.extract_column_info_from_data(columns)


            result = create_success_result(
                data=result_data,
                columns=column_info,
                execution_time_ms=performance.execution_time_ms,
                request_id=request.request_id
            )


            result.performance = performance
            self.successful_queries += 1


            logger.info(f"Query executed successfully: {len(result_data)} rows in {performance.execution_time_ms}ms")
            return result


        except asyncio.TimeoutError:
            self.failed_queries += 1
            return self.error_handler.handle_timeout_error(
                processed_query, # type: ignore
                request.timeout_seconds or 30,
                request.request_id
            )


        except Exception as e:
            logger.exception("Error during full query execution")
            self.failed_queries += 1
            return self.error_handler.handle_execution_error(e, processed_query, request.request_id) # type: ignore


    async def _execute_with_connection(self, query: str) -> tuple:
        """✅ FIXED: Proper async database connection handling"""
        async with self.connection_manager.get_connection() as connection:
            # Execute query directly with async connection
            cursor = await connection.execute(query)
            
            try:
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                
                # Fetch all rows
                rows = await cursor.fetchall()

                result_data = []
                for row in rows:
                    row_dict = {}
                    for i, column_name in enumerate(columns):
                        value = row[i] if i < len(row) else None
                        if isinstance(value, datetime):
                            row_dict[column_name] = value.isoformat()
                        else:
                            row_dict[column_name] = value
                    result_data.append(row_dict)

                return result_data, columns

            finally:
                await cursor.close()


    async def format_result(self, result: ExecutionResult, output_format: OutputFormat) -> Union[str, bytes]:
        try:
            return self.formatter.format_result(result, output_format)
        except Exception as e:
            logger.exception("Error formatting result, falling back to JSON")
            return self.formatter.format_result(result, OutputFormat.JSON)


    async def execute_and_format(self, request: ExecutionRequest) -> Union[str, bytes]:
        result = await self.execute_query(request)
        formatted_result = await self.format_result(result, request.output_format)
        return formatted_result


    def get_execution_stats(self) -> Dict[str, Any]:
        success_rate = (self.successful_queries / self.total_queries_executed * 100) if self.total_queries_executed > 0 else 0


        return {
            'total_queries_executed': self.total_queries_executed,
            'successful_queries': self.successful_queries,
            'failed_queries': self.failed_queries,
            'success_rate_percent': round(success_rate, 2),
            'is_initialized': self.is_initialized,
            'connection_stats': self.connection_manager.get_pool_stats(),
            'supported_formats': self.formatter.get_supported_formats()
        }


    async def health_check(self) -> Dict[str, Any]:
        health_status = {
            'executor_healthy': self.is_initialized,
            'executor_stats': self.get_execution_stats(),
            'connection_health': await self.connection_manager.health_check(),
            'timestamp': datetime.utcnow().isoformat()
        }


        health_status['overall_healthy'] = (
            self.is_initialized and
            health_status['connection_health'].get('overall_healthy', False)
        )


        return health_status


    async def shutdown(self):
        logger.info("Shutting down SQL Executor...")
        await self.connection_manager.shutdown()
        self.is_initialized = False
        logger.info("SQL Executor shutdown complete")


# Global executor instance
sql_executor = SQLExecutor()


# Convenience async functions
async def execute_query(request: ExecutionRequest) -> ExecutionResult:
    return await sql_executor.execute_query(request)


async def execute_and_format(request: ExecutionRequest) -> Union[str, bytes]:
    return await sql_executor.execute_and_format(request)


async def health_check() -> Dict[str, Any]:
    return await sql_executor.health_check()


__all__ = [
    'SQLExecutor', 'QueryProcessor', 'ExecutionMonitor',
    'sql_executor', 'execute_query', 'execute_and_format', 'health_check'
]
