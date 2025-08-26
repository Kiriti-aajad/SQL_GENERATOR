"""
Configuration management for the prompt builder system.
Handles loading and validation of YAML configuration files.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Configuration file paths
CONFIG_DIR = Path(__file__).parent
TEMPLATE_CONFIG_PATH = CONFIG_DIR / "template_config.yaml"
CONTEXT_CONFIG_PATH = CONFIG_DIR / "context_config.yaml"
LLM_OPTIMIZATIONS_PATH = CONFIG_DIR / "llm_optimizations.yaml"


class ConfigManager:
    """
    Centralized configuration manager for prompt builder.
    Loads and caches YAML configurations.
    """
    
    def __init__(self):
        self._template_config: Optional[Dict[str, Any]] = None
        self._context_config: Optional[Dict[str, Any]] = None
        self._llm_config: Optional[Dict[str, Any]] = None
    
    def get_template_config(self) -> Dict[str, Any]:
        """Load and cache template configuration"""
        if self._template_config is None:
            self._template_config = self._load_yaml_config(TEMPLATE_CONFIG_PATH)
        return self._template_config
    
    def get_context_config(self) -> Dict[str, Any]:
        """Load and cache context configuration"""
        if self._context_config is None:
            self._context_config = self._load_yaml_config(CONTEXT_CONFIG_PATH)
        return self._context_config
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Load and cache LLM optimization configuration"""
        if self._llm_config is None:
            try:
                self._llm_config = self._load_yaml_config(LLM_OPTIMIZATIONS_PATH)
            except FileNotFoundError:
                logger.warning(f"LLM config file not found: {LLM_OPTIMIZATIONS_PATH}")
                self._llm_config = {}
        return self._llm_config
    
    def reload_configs(self):
        """Force reload of all configurations"""
        self._template_config = None
        self._context_config = None
        self._llm_config = None
        logger.info("Configuration cache cleared - will reload on next access")
    
    def _load_yaml_config(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                logger.debug(f"Loaded configuration from {file_path}")
                return config or {}
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {file_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration {file_path}: {e}")
            raise


# Global configuration manager instance
config_manager = ConfigManager()

# Convenience functions for easy access
def get_template_config() -> Dict[str, Any]:
    """Get template configuration"""
    return config_manager.get_template_config()

def get_context_config() -> Dict[str, Any]:
    """Get context building configuration"""
    return config_manager.get_context_config()

def get_llm_config() -> Dict[str, Any]:
    """Get LLM optimization configuration"""
    return config_manager.get_llm_config()

def reload_all_configs():
    """Reload all configuration files"""
    config_manager.reload_configs()
