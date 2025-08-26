"""
Orchestrator Exception Definitions
All custom exceptions used by the orchestrator system
"""

class ComponentInitializationError(Exception):
    """
    Raised when orchestrator component initialization fails
    
    This exception is thrown when:
    - Required components cannot be imported
    - Component dependencies are missing
    - Configuration is invalid for component setup
    """
    pass

class OrchestrationError(Exception):
    """
    Raised during orchestration processing
    
    This exception is thrown when:
    - Query routing fails
    - Agent execution encounters errors
    - Processing pipeline breaks down
    """
    pass

class ConfigurationError(Exception):
    """
    Raised when configuration is invalid
    
    This exception is thrown when:
    - Configuration validation fails
    - Required configuration parameters are missing
    - Configuration values are out of valid range
    """
    pass

class AgentUnavailableError(Exception):
    """
    Raised when a required agent is not available
    
    This exception is thrown when:
    - Attempting to use an agent that hasn't been initialized
    - Agent is in unhealthy state
    - Agent dependencies are not met
    """
    pass

class QueryProcessingError(Exception):
    """
    Raised during query processing
    
    This exception is thrown when:
    - Query validation fails
    - Processing timeout occurs
    - Response formatting fails
    """
    pass
