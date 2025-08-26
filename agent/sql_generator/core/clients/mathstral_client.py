"""
Mathstral Client - FIXED VERSION with Proper Async Timeout Handling
- FIXED: All async timeout context manager issues resolved
- FIXED: Proper session management and error handling
- FIXED: JSON parsing with proper [0] indexing
- FIXED: Environment variables only, no hardcoded URLs
- FIXED: NO API keys or authentication (direct NGROK calls)
- FIXED: Complete async/await compatibility
"""

import logging
import aiohttp
import asyncio
import time
import os
from typing import Dict, Any, Optional

# Handle dotenv import gracefully
try:
    from dotenv import load_dotenv  # pyright: ignore[reportAssignmentType]
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    def load_dotenv(*args, **kwargs):
        pass

class MathstralClient:
    """
    FIXED: Mathstral client with proper async timeout handling
    """

    def __init__(self, base_url=None, api_key=None, config=None, logger=None):
        if DOTENV_AVAILABLE:
            load_dotenv(override=True)
        
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info("INITIALIZING MATHSTRAL CLIENT")
        
        # Get URL from environment variables only
        self.base_url = self._get_env_url(base_url)
        self.is_offline = not bool(self.base_url)
        
        # No API key needed for NGROK calls
        self.api_key = None
        self.model = "mathstral-7b-v0.1"

        self.session: Optional[aiohttp.ClientSession] = None

        # Required attributes for compatibility
        self.last_health_check: Optional[float] = None
        self.health_check_status: bool = False
        self.client_name = "mathstral"

        # Config object for compatibility
        self.config = type('Config', (), {
            'name': 'Mathstral',
            'base_url': self.base_url,
            'offline': self.is_offline
        })()

        if self.is_offline:
            self.logger.warning("MathstralClient initialized in OFFLINE mode - no environment URL found")
        else:
            self.logger.info(f"MathstralClient initialized with endpoint: {self.base_url}")

    def _get_env_url(self, base_url=None) -> Optional[str]:
        """Get URL from environment variables only"""
        if base_url:
            return base_url
            
        url_candidates = [
            os.getenv("MATHSTRAL_ENDPOINT"),
            os.getenv("MATHSTRAL_BASE_URL"),
            os.getenv("MATHSTRAL_API_URL"),
            os.getenv("MATHSTRAL_URL"),  # Added this common variant
        ]
        
        # Try constructing from base NGROK URL
        ngrok_url = os.getenv("NGROK_URL")
        if ngrok_url:
            url_candidates.append(f"{ngrok_url}/v1/chat/completions")
        
        for url in url_candidates:
            if url and url.strip():
                self.logger.info(f"Using Mathstral URL from environment: {url}")
                return url.strip()
        
        self.logger.error("No Mathstral endpoint found in environment variables")
        return None

    async def __aenter__(self):
        if not self.is_offline and (not self.session or self.session.closed):
            await self._create_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _create_session(self):
        """FIXED: Create session with proper timeout handling"""
        try:
            if not self.is_offline and (not self.session or self.session.closed):
                connector = aiohttp.TCPConnector(
                    limit=10,
                    limit_per_host=5,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                    keepalive_timeout=30
                )
                
                # FIXED: Add default timeout to session
                timeout = aiohttp.ClientTimeout(
                    total=30,      # Total timeout
                    connect=10,    # Connection timeout
                    sock_read=20   # Socket read timeout
                )
                
                self.session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,  # FIXED: Add timeout to session
                    headers={'User-Agent': 'SQL-AI-Agent-Mathstral/1.0'}
                )
                self.logger.info("Mathstral session created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create Mathstral session: {e}")
            self.session = None

    def health_check(self) -> bool:
        """Synchronous health check"""
        try:
            if self.is_offline:
                self.health_check_status = False
                return False
                
            is_healthy = bool(self.base_url and self.base_url.startswith('http'))
            self.last_health_check = time.time()
            self.health_check_status = is_healthy
            return is_healthy
        except Exception as e:
            self.logger.error(f"Mathstral health check error: {e}")
            self.health_check_status = False
            return False

    async def async_health_check(self) -> bool:
        """FIXED: Async health check with proper timeout handling"""
        try:
            if self.is_offline:
                self.logger.warning("Health check failed: Model is offline")
                self.health_check_status = False
                return False
                
            if not self.session or self.session.closed:
                await self._create_session()

            test_payload = {
                "model": "mathstral-7b-v0.1",
                "messages": [
                    {"role": "system", "content": "You are a SQL assistant. Return only SQL."},
                    {"role": "user", "content": "SELECT 1"}
                ]
            }

            headers = {"Content-Type": "application/json"}

            # FIXED: Proper timeout handling
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with self.session.post(  # pyright: ignore[reportOptionalMemberAccess]
                    self.base_url,  # pyright: ignore[reportArgumentType]
                    json=test_payload, 
                    headers=headers, 
                    timeout=timeout
                ) as response:
                    if response.status == 200:
                        self.logger.info("Mathstral health check passed")
                        self.health_check_status = True
                        self.last_health_check = time.time()
                        return True
                    else:
                        response_text = await response.text()
                        self.logger.error(f"Health check failed: HTTP {response.status}")
                        self.logger.error(f"Response: {response_text}")
                        self.health_check_status = False
                        return False
            
            except asyncio.TimeoutError:
                self.logger.error("Health check timeout")
                self.health_check_status = False
                return False
            except aiohttp.ServerTimeoutError: # pyright: ignore[reportUnusedExcept]
                self.logger.error("Health check server timeout")
                self.health_check_status = False
                return False

        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            self.health_check_status = False
            return False

    async def generate_sql_async(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """FIXED: Generate SQL with proper async timeout handling"""
        try:
            if self.is_offline:
                self.logger.warning("SQL generation failed: Model is offline")
                return {
                    "success": False,
                    "generated_sql": "",
                    "sql": "",
                    "error": "Model is offline - no environment URL configured",
                    "confidence_score": 0.0,
                    "confidence": 0.0,
                    "model_used": "mathstral-7b-v0.1",
                    "offline": True
                }
            
            # FIXED: Ensure session is created properly in async context
            if not self.session or self.session.closed:
                await self._create_session()
                
            # If session creation failed, return error
            if not self.session:
                self.logger.error("Failed to create session for SQL generation")
                return {
                    "success": False,
                    "generated_sql": "",
                    "sql": "",
                    "error": "Failed to create HTTP session",
                    "confidence_score": 0.0,
                    "confidence": 0.0,
                    "model_used": "mathstral-7b-v0.1",
                    "offline": True
                }

            payload = {
                "model": "mathstral-7b-v0.1",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert SQL generation assistant specialized for Microsoft SQL Server. "
                            "Return ONLY a valid SQL query starting with SELECT, WITH, INSERT, UPDATE, or DELETE. "
                            "Do NOT include any explanation, apology, or additional text."
                        )
                    },
                    {"role": "user", "content": prompt}
                ]
            }

            headers = {"Content-Type": "application/json"}
            self.logger.debug(f"Generating SQL for prompt: {prompt}")

            # FIXED: Proper timeout handling with specific exception handling
            try:
                # FIXED: Create timeout object properly
                timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=20)
                
                async with self.session.post(  # pyright: ignore[reportOptionalMemberAccess]
                    self.base_url,  # pyright: ignore[reportArgumentType]
                    json=payload, 
                    headers=headers, 
                    timeout=timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.logger.debug(f"API Response: {data}")
                        
                        if data and "choices" in data and len(data["choices"]) > 0:
                            first_choice = data["choices"][0]  # FIXED:  index
                            
                            if "message" in first_choice and "content" in first_choice["message"]:
                                sql_content = first_choice["message"]["content"].strip()
                                
                                # Remove markdown if present
                                if sql_content.startswith("```"):
                                    lines = sql_content.split("\n")
                                    if len(lines) > 2:
                                        sql_content = "\n".join(lines[1:-1]).strip()
                                    else:
                                        sql_content = sql_content.replace("```", "").strip()
                                
                                self.logger.info(f"Successfully generated SQL: {sql_content[:100]}...")
                                
                                return {
                                    "success": True,
                                    "generated_sql": sql_content,
                                    "sql": sql_content,
                                    "confidence_score": 0.85,
                                    "confidence": 0.85,
                                    "model_used": "mathstral-7b-v0.1",
                                    "tokens_used": data.get("usage", {}).get("total_tokens", 0),
                                    "offline": False
                                }
                        
                        return {
                            "success": False,
                            "generated_sql": "",
                            "sql": "",
                            "error": f"Invalid response structure: {data}",
                            "confidence_score": 0.0,
                            "confidence": 0.0,
                            "model_used": "mathstral-7b-v0.1",
                            "offline": False
                        }
                    else:
                        error_text = await response.text()
                        self.logger.error(f"API error: HTTP {response.status} - {error_text}")
                        return {
                            "success": False,
                            "generated_sql": "",
                            "sql": "",
                            "error": f"HTTP {response.status}: {error_text}",
                            "confidence_score": 0.0,
                            "confidence": 0.0,
                            "model_used": "mathstral-7b-v0.1",
                            "offline": False
                        }

            except asyncio.TimeoutError:
                # FIXED: Specific timeout error handling
                self.logger.error("Request timeout (30 seconds)")
                return {
                    "success": False,
                    "generated_sql": "",
                    "sql": "",
                    "error": "Request timeout",
                    "confidence_score": 0.0,
                    "confidence": 0.0,
                    "model_used": "mathstral-7b-v0.1",
                    "offline": False
                }
            except aiohttp.ServerTimeoutError: # pyright: ignore[reportUnusedExcept]
                # FIXED: Server timeout handling
                self.logger.error("Server timeout")
                return {
                    "success": False,
                    "generated_sql": "",
                    "sql": "",
                    "error": "Server timeout",
                    "confidence_score": 0.0,
                    "confidence": 0.0,
                    "model_used": "mathstral-7b-v0.1",
                    "offline": False
                }
            except aiohttp.ClientError as e:
                # FIXED: Client error handling
                self.logger.error(f"Client error: {e}")
                return {
                    "success": False,
                    "generated_sql": "",
                    "sql": "",
                    "error": f"Client error: {e}",
                    "confidence_score": 0.0,
                    "confidence": 0.0,
                    "model_used": "mathstral-7b-v0.1",
                    "offline": False
                }

        except Exception as e:
            self.logger.error(f"Mathstral SQL generation error: {e}")
            return {
                "success": False,
                "generated_sql": "",
                "sql": "",
                "error": str(e),
                "confidence_score": 0.0,
                "confidence": 0.0,
                "model_used": "mathstral-7b-v0.1",
                "offline": self.is_offline
            }

    async def close(self):
        """FIXED: Close session safely"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
                self.logger.info("Mathstral session closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing Mathstral session: {e}")
        finally:
            self.session = None

    async def cleanup(self):
        """Cleanup resources"""
        await self.close()

    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        return {
            "base_url": self.base_url,
            "has_api_key": False,  # NO API keys used
            "last_health_check": self.last_health_check,
            "health_status": self.health_check_status,
            "session_active": self.session is not None and not getattr(self.session, 'closed', True),
            "offline": self.is_offline
        }

    def is_healthy(self) -> bool:
        """Check if client is healthy"""
        return self.health_check_status and not self.is_offline

    @property
    def endpoint_url(self) -> str:
        """Get endpoint URL"""
        return self.base_url or "offline"

    def get_model_name(self) -> str:
        """Get model name"""
        return "mathstral-7b-v0.1"

    async def test_connection(self) -> bool:
        """Test connection"""
        return await self.async_health_check()

    async def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration"""
        return {
            "valid": bool(self.base_url and self.base_url.startswith('http')) and not self.is_offline,
            "base_url": self.base_url or "not_configured",
            "model": "mathstral-7b-v0.1",
            "api_key_configured": False,  # NO API keys
            "session_ready": self.is_healthy(),
            "offline": self.is_offline
        }

    def __repr__(self):
        """String representation"""
        status = "offline" if self.is_offline else "online"
        return f"MathstralClient(base_url='{self.base_url or 'not_configured'}', status='{status}')"

__all__ = ["MathstralClient"]
