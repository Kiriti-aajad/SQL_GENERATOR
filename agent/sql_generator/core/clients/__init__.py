"""
SQL Generator Clients - Simplified Architecture
Exports the corrected MathstralClient and DeepSeekClient without BaseModelClient dependencies
"""

from .mathstral_client import MathstralClient
from .deepseek_client import DeepSeekClient

# Export only the simplified clients
__all__ = ["MathstralClient", "DeepSeekClient"]
