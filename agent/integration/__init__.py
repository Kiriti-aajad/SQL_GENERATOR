"""
Agent Integration Package - Core Banking SQL AI Integration Components

This package provides the main integration layer for the banking SQL AI agent system,
connecting NLP processing, schema discovery, and SQL generation components.

Components:
- OrchestratorCompatibleSystem: Main orchestrator interface  
- NLPSchemaIntegrationBridge: NLP-Schema integration bridge
- GeneratorExecutorBridge: SQL generation and execution bridge

Note: This package has been restructured to eliminate circular import dependencies.
"""

from .orchestrator_interface import (
    create_orchestrator_compatible_system,
    OrchestratorCompatibleSystem
)

from .nlp_schema_integration_bridge import NLPSchemaIntegrationBridge
from .generator_bridge import GeneratorExecutorBridge

__all__ = [
    "create_orchestrator_compatible_system",
    "OrchestratorCompatibleSystem", 
    "NLPSchemaIntegrationBridge",
    "GeneratorExecutorBridge"
]
