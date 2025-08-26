"""
DeepSeek Client - Environment Variable Only Version  
- FIXED: All syntax errors corrected
- FIXED: Uses environment variables only, no hardcoded URLs
- FIXED: Returns "model is offline" when environment URL not available
- FIXED: Proper fallback handling for offline scenarios
- FIXED: Clean code without emojis
"""

import logging
import aiohttp
import asyncio
import time
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

class DeepSeekClient:
    def __init__(self, base_url=None, api_key=None, config=None, logger=None):
        # Load environment variables first
        load_dotenv(override=True)
        
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Get URL from environment variables only - no hardcoded fallbacks
        self.base_url = self._get_env_url(base_url)
        self.is_offline = not bool(self.base_url)
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_health_check: Optional[float] = None
        self.health_check_status: bool = False
        self.client_name = "deepseek"
        
        # Config for compatibility
        self.config = type('Config', (), {
            'name': 'DeepSeek',
            'base_url': self.base_url,
            'offline': self.is_offline
        })()
        
        if self.is_offline:
            self.logger.warning("DeepSeekClient initialized in OFFLINE mode - no environment URL found")
        else:
            self.logger.info(f"DeepSeekClient initialized with endpoint: {self.base_url}")

    def _get_env_url(self, base_url=None) -> Optional[str]:
        """Get URL from environment variables only"""
        if base_url:
            return base_url
            
        # Try different environment variable names
        url_candidates = [
            os.getenv("DEEPSEEK_ENDPOINT"),
            os.getenv("DEEPSEEK_BASE_URL"),
            os.getenv("DEEPSEEK_API_URL"),
        ]
        
        # Try constructing from base NGROK URL
        ngrok_url = os.getenv("NGROK_URL")
        if ngrok_url:
            url_candidates.append(f"{ngrok_url}/v1/chat/completions")
        
        for url in url_candidates:
            if url and url.strip():
                self.logger.info(f"Using DeepSeek URL from environment: {url}")
                return url.strip()
        
        self.logger.error("No DeepSeek endpoint found in environment variables")
        return None

    async def __aenter__(self):
        if not self.is_offline and (not self.session or self.session.closed):
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                headers={'User-Agent': 'SQL-AI-Agent-DeepSeek/1.0'}
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

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
            self.logger.error(f"Health check error: {e}")
            self.health_check_status = False
            return False

    async def async_health_check(self) -> bool:
        """Async health check with offline detection"""
        try:
            if self.is_offline:
                self.logger.warning("Health check failed: Model is offline (no environment URL)")
                self.health_check_status = False
                return False
                
            if not self.session or self.session.closed:
                await self.__aenter__()

            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a SQL assistant. Return only SQL."},
                    {"role": "user", "content": "SELECT 1"}
                ]
            }

            headers = {"Content-Type": "application/json"}

            async with self.session.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    self.logger.info("DeepSeek health check passed")
                    self.health_check_status = True
                    self.last_health_check = time.time()
                    return True
                else:
                    response_text = await response.text()
                    self.logger.error(f"Health check failed: HTTP {response.status}")
                    self.logger.error(f"Response: {response_text}")
                    self.health_check_status = False
                    return False

        except aiohttp.ClientTimeout: # type: ignore
            self.logger.error("Health check timed out")
            self.health_check_status = False
            return False
        except aiohttp.ClientError as e:
            self.logger.error(f"Health check client error: {e}")
            self.health_check_status = False
            return False
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            self.health_check_status = False
            return False

    async def generate_sql_async(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate SQL with offline mode support"""
        try:
            # Check if model is offline
            if self.is_offline:
                self.logger.warning("SQL generation failed: Model is offline")
                return {
                    "success": False,
                    "generated_sql": "",
                    "sql": "",
                    "error": "Model is offline - no environment URL configured",
                    "confidence_score": 0.0,
                    "confidence": 0.0,
                    "model_used": "deepseek-chat",
                    "offline": True
                }
            
            if not self.session or self.session.closed:
                await self.__aenter__()

            payload = {
                "model": "deepseek-chat",
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

            async with self.session.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data and "choices" in data and len(data["choices"]) > 0:
                        sql_content = data["choices"][0]["message"]["content"].strip()
                        
                        # Remove markdown if present
                        if sql_content.startswith("```"):
                            lines = sql_content.split("\n")
                            if len(lines) > 2:
                                sql_content = "\n".join(lines[1:-1]).strip()
                            else:
                                sql_content = sql_content.replace("```", "").strip()
                        
                        self.logger.info(f"SQL generated successfully: {sql_content[:100]}...")
                        
                        return {
                            "success": True,
                            "generated_sql": sql_content,
                            "sql": sql_content,
                            "confidence_score": 0.82,
                            "confidence": 0.82,
                            "model_used": "deepseek-chat",
                            "tokens_used": data.get("usage", {}).get("total_tokens", 0),
                            "offline": False
                        }
                    else:
                        return {
                            "success": False,
                            "generated_sql": "",
                            "sql": "",
                            "error": "Invalid response structure",
                            "confidence_score": 0.0,
                            "confidence": 0.0,
                            "model_used": "deepseek-chat",
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
                        "model_used": "deepseek-chat",
                        "offline": False
                    }

        except aiohttp.ClientTimeout:
            self.logger.error("SQL generation timed out")
            return {
                "success": False,
                "generated_sql": "",
                "sql": "",
                "error": "Request timed out",
                "confidence_score": 0.0,
                "confidence": 0.0,
                "model_used": "deepseek-chat",
                "offline": False
            }
        except aiohttp.ClientError as e:
            self.logger.error(f"SQL generation client error: {e}")
            return {
                "success": False,
                "generated_sql": "",
                "sql": "",
                "error": str(e),
                "confidence_score": 0.0,
                "confidence": 0.0,
                "model_used": "deepseek-chat",
                "offline": False
            }
        except Exception as e:
            self.logger.error(f"SQL generation error: {e}")
            return {
                "success": False,
                "generated_sql": "",
                "sql": "",
                "error": str(e),
                "confidence_score": 0.0,
                "confidence": 0.0,
                "model_used": "deepseek-chat",
                "offline": self.is_offline
            }

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
        self.session = None

    # Compatibility methods
    def is_healthy(self) -> bool:
        return self.health_check_status and not self.is_offline

    def get_connection_status(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "has_api_key": False,
            "last_health_check": self.last_health_check,
            "health_status": self.health_check_status,
            "session_active": self.session is not None and not getattr(self.session, 'closed', True),
            "offline": self.is_offline
        }

    @property
    def endpoint_url(self) -> str:
        return self.base_url or "offline"

    def get_model_name(self) -> str:
        return "deepseek-chat"

    async def test_connection(self) -> bool:
        return await self.async_health_check()

    async def validate_configuration(self) -> Dict[str, Any]:
        return {
            "valid": bool(self.base_url and self.base_url.startswith('http')) and not self.is_offline,
            "base_url": self.base_url or "not_configured",
            "model": "deepseek-chat",
            "api_key_configured": False,
            "session_ready": self.is_healthy(),
            "offline": self.is_offline
        }

    def __repr__(self):
        status = "offline" if self.is_offline else "online"
        return f"DeepSeekClient(base_url='{self.base_url or 'not_configured'}', status='{status}')"

__all__ = ["DeepSeekClient"]
