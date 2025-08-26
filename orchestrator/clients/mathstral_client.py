"""
Mathstral Client - FULLY FIXED VERSION
- FIXED: Timeout context manager issues (uses aiohttp.ClientTimeout properly)
- FIXED: Proper environment variable reading and URL configuration
- FIXED: Proper model name usage and streamlined implementation
- FIXED: Session management and response parsing
"""

import logging
import aiohttp
import asyncio
import json
import time
import os
from typing import Dict, Any, Optional

class MathstralClient:
    """
    Fixed Mathstral client using mathstral-7b-v0.1 model via NGROK endpoint.
    CRITICAL FIX: Now uses proper aiohttp.ClientTimeout for all requests.
    """

    def __init__(self, base_url=None, api_key=None, config=None, logger=None):
        # FIXED: Corrected environment variable names (MATHSTRAL not MATHRAL)
        self.base_url = (
            base_url or 
            (config.get("base_url") if config else None) or
            os.getenv("MATHSTRAL_ENDPOINT") or
            os.getenv("MATHSTRAL_BASE_URL") or
            f"{os.getenv('NGROK_URL', 'https://c5804745dbc1.ngrok-free.app')}/v1/chat/completions"
        )
        
        self.api_key = (
            api_key or 
            (config.get("api_key") if config else None) or
            os.getenv("MATHSTRAL_API_KEY")
        )
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # Add required attributes for SQLGenerator compatibility
        self.last_health_check: Optional[float] = None
        self.health_check_status: bool = False
        
        # Add client_name for AsyncClientManager compatibility
        self.client_name = "mathstral"

        # Simple config object for compatibility
        self.config = type('Config', (), {
            'name': 'Mathstral',
            'base_url': self.base_url
        })()

        self.logger.info(f"MathstralClient initialized with endpoint: {self.base_url}")

    async def __aenter__(self):
        """Async context manager entry"""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session and not self.session.closed:
            await self.session.close()
        self.session = None

    def health_check(self) -> bool:
        """Synchronous health check for SQLGenerator compatibility"""
        try:
            # Basic validation - check if endpoint is configured and valid
            is_healthy = bool(self.base_url and self.base_url.startswith('http'))
            
            # Update tracking attributes
            self.last_health_check = time.time()
            self.health_check_status = is_healthy
            
            self.logger.debug(f"Mathstral health check: {'PASSED' if is_healthy else 'FAILED'}")
            return is_healthy
            
        except Exception as e:
            self.logger.error(f"Mathstral health check error: {e}")
            self.health_check_status = False
            return False

    async def async_health_check(self) -> bool:
        """FIXED: Async health check with proper timeout handling"""
        try:
            if not self.session or self.session.closed:
                self.session = aiohttp.ClientSession()

            # Use correct model name for health check
            test_payload = {
                "model": "mathstral-7b-v0.1",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a SQL assistant. Return only SQL."
                    },
                    {
                        "role": "user", 
                        "content": "SELECT 1"
                    }
                ]
            }

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # CRITICAL FIX: Use aiohttp.ClientTimeout inline
            async with self.session.post(
                self.base_url, 
                json=test_payload, 
                headers=headers, 
                timeout=aiohttp.ClientTimeout(total=10)  # FIXED!
            ) as response:
                if response.status == 200:
                    self.logger.info("Mathstral async health check passed")
                    self.health_check_status = True
                    self.last_health_check = time.time()
                    return True
                else:
                    self.logger.warning(f"Mathstral async health check failed: {response.status}")
                    self.health_check_status = False
                    return False

        except asyncio.TimeoutError:
            self.logger.error("Mathstral health check timed out")
            self.health_check_status = False
            return False
        except Exception as e:
            self.logger.error(f"Mathstral async health check error: {e}")
            self.health_check_status = False
            return False

    async def generate_sql_async(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        FIXED: Generate SQL using mathstral-7b-v0.1 with proper timeout handling.
        """
        try:
            if not self.session or self.session.closed:
                self.session = aiohttp.ClientSession()

            # Use mathstral-7b-v0.1 consistently
            payload = {
                "model": kwargs.get("model", "mathstral-7b-v0.1"),
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert SQL generation assistant specialized for Microsoft SQL Server. "
                            "Return ONLY a valid SQL query starting with SELECT, WITH, INSERT, UPDATE, or DELETE. "
                            "Do NOT include any explanation, apology, or additional text. "
                            "Focus on generating clean, efficient SQL queries."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.1)
            }

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # CRITICAL FIX: Use aiohttp.ClientTimeout inline
            async with self.session.post(
                self.base_url, 
                json=payload, 
                headers=headers, 
                timeout=aiohttp.ClientTimeout(total=30)  # FIXED!
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # FIXED: Extract SQL with proper error handling
                    try:
                        # CRITICAL FIX: Correct array indexing
                        sql_query = data["choices"][0]["message"]["content"].strip()
                        
                        # Clean SQL output
                        sql_query = self._clean_sql_output(sql_query)

                        return {
                            "sql": sql_query,
                            "generated_sql": sql_query,  # SQLGenerator compatibility
                            "success": True,
                            "confidence": 0.85,
                            "confidence_score": 0.85,
                            "model_used": payload["model"],
                            "tokens_used": data.get("usage", {}).get("total_tokens", 0),
                            "response_metadata": {
                                "api_response": data,
                                "processing_time": time.time()
                            }
                        }

                    except (KeyError, IndexError, AttributeError) as parse_error:
                        self.logger.error(f"Error parsing Mathstral response: {parse_error}")
                        self.logger.error(f"Response data: {data}")
                        return {
                            "sql": "",
                            "generated_sql": "",
                            "success": False,
                            "confidence": 0.0,
                            "confidence_score": 0.0,
                            "error": f"Parsing error: {parse_error}",
                            "model_used": payload["model"]
                        }

                else:
                    error_text = await response.text()
                    self.logger.error(f"Mathstral API request failed with status {response.status}: {error_text}")
                    return {
                        "sql": "",
                        "generated_sql": "",
                        "success": False,
                        "confidence": 0.0,
                        "confidence_score": 0.0,
                        "error": f"HTTP {response.status}: {error_text}",
                        "model_used": payload["model"]
                    }

        except asyncio.TimeoutError:
            self.logger.error("Mathstral API request timed out")
            return {
                "sql": "",
                "generated_sql": "",
                "success": False,
                "confidence": 0.0,
                "confidence_score": 0.0,
                "error": "Request timed out",
                "model_used": kwargs.get("model", "mathstral-7b-v0.1")
            }

        except Exception as e:
            self.logger.error(f"Exception during Mathstral API request: {e}")
            return {
                "sql": "",
                "generated_sql": "",
                "success": False,
                "confidence": 0.0,
                "confidence_score": 0.0,
                "error": str(e),
                "model_used": kwargs.get("model", "mathstral-7b-v0.1")
            }

    def _clean_sql_output(self, sql_output: str) -> str:
        """Clean and format SQL output from Mathstral"""
        if not sql_output:
            return ""
        
        # Remove markdown code blocks
        if sql_output.startswith("```"):
            lines = sql_output.split('\n')
            if len(lines) > 2:
                sql_output = '\n'.join(lines[1:-1]).strip()
            else:
                sql_output = sql_output.replace("```", "").strip()
        
        # Remove common response prefixes
        prefixes_to_remove = [
            "Here's the SQL query:",
            "Here is the SQL:",
            "SQL:",
            "Query:",
            "The SQL query is:",
            "Here's your SQL:"
        ]
        
        for prefix in prefixes_to_remove:
            if sql_output.lower().startswith(prefix.lower()):
                sql_output = sql_output[len(prefix):].strip()
        
        return sql_output

    async def close(self):
        """Close the HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
        self.session = None
        self.logger.info("MathstralClient session closed")

    async def cleanup(self):
        """Cleanup method for AsyncClientManager compatibility"""
        await self.close()

    # Compatibility methods for SQLGenerator and AsyncClientManager
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status information"""
        return {
            "base_url": self.base_url,
            "has_api_key": bool(self.api_key),
            "last_health_check": self.last_health_check,
            "health_status": self.health_check_status,
            "session_active": self.session is not None and not getattr(self.session, 'closed', True)
        }

    def is_healthy(self) -> bool:
        """Quick health status check"""
        return self.health_check_status

    @property
    def endpoint_url(self) -> str:
        return self.base_url

    def get_model_name(self) -> str:
        return "mathstral-7b-v0.1"

    async def test_connection(self) -> bool:
        """Test connection compatibility method"""
        return await self.async_health_check()

    def get_client_info(self) -> Dict[str, Any]:
        """Get client information for debugging"""
        return {
            "client_name": self.client_name,
            "model": "mathstral-7b-v0.1",
            "base_url": self.base_url,
            "is_healthy": self.is_healthy(),
            "last_health_check": self.last_health_check,
            "session_active": self.session is not None and not getattr(self.session, 'closed', True)
        }

    async def validate_configuration(self) -> Dict[str, Any]:
        """Validate client configuration"""
        return {
            "valid": bool(self.base_url and "mathstral" in self.base_url.lower()),
            "base_url": self.base_url,
            "model": "mathstral-7b-v0.1",
            "api_key_configured": bool(self.api_key),
            "session_ready": self.is_healthy()
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": "mathstral-7b-v0.1",
            "endpoint": self.base_url,
            "client_type": "MathstralClient",
            "supports_async": True
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current client status"""
        return {
            "healthy": self.is_healthy(),
            "initialized": self.session is not None,
            "last_check": self.last_health_check,
            "model": "mathstral-7b-v0.1",
            "endpoint": self.base_url
        }

    def __del__(self):
        """Cleanup on deletion"""
        try:
            if self.session and hasattr(self.session, 'closed') and not self.session.closed:
                self.logger.warning("MathstralClient session was not properly closed - use 'async with' or call close() explicitly")
        except Exception:
            pass

    def __repr__(self):
        return f"MathstralClient(base_url='{self.base_url}', model='mathstral-7b-v0.1')"

# Export the main class
__all__ = ["MathstralClient"]

# Test function
async def test_mathstral_client():
    """Test the fixed MathstralClient"""
    print("Testing FIXED MathstralClient...")
    print(f"NGROK_URL from env: {os.getenv('NGROK_URL')}")
    print(f"MATHSTRAL_ENDPOINT from env: {os.getenv('MATHSTRAL_ENDPOINT')}")
    
    try:
        async with MathstralClient() as client:
            print(f"Initialized with endpoint: {client.base_url}")
            
            # Test health check
            health = await client.async_health_check()
            print(f"Health check: {'PASSED' if health else 'FAILED'}")
            
            if health:
                # Test SQL generation
                result = await client.generate_sql_async(
                    "Write SQL Server query to get all customers with their phone numbers"
                )
                print("SQL Generation Result:")
                print(f"  Success: {result.get('success')}")
                print(f"  Confidence: {result.get('confidence')}")
                print(f"  SQL: {result.get('generated_sql', '')[:100]}...")
            else:
                print("Health check failed - skipping SQL generation test")
                
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_mathstral_client())
