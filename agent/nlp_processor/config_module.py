"""
Global Configuration for NLP Processor - FAIL-FAST VERSION
Integrates with your existing data infrastructure and metadata

FIXES IMPLEMENTED:
- Returns dictionary from get_config() (not object)
- Fail-fast configuration loading (no silent fallbacks)
- Immediate metadata validation
- Clear error messages when dependencies missing
- Proper exception propagation
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing"""
    pass

class MetadataError(Exception):
    """Raised when metadata integration fails"""
    pass

class NLPConfig:
    """
    FAIL-FAST Configuration manager for NLP Processor
    NO SILENT FALLBACKS - If configuration is broken, system fails clearly
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        
        # FAIL-FAST: Load configuration immediately, no lazy loading
        self._config = self._load_config_strict()
        
        # FAIL-FAST: Validate metadata integration immediately
        self._validate_metadata_integration()
        
        # FAIL-FAST: Enhance with metadata immediately (no lazy loading)
        self._enhance_config_with_metadata_strict()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration path"""
        config_dir = Path(__file__).parent / "config"
        config_file = config_dir / "nlp_config.yaml"
        
        # Ensure config directory exists
        config_dir.mkdir(exist_ok=True)
        
        return str(config_file)
    
    def _load_config_strict(self) -> Dict[str, Any]:
        """
        Load configuration with STRICT validation - NO SILENT FALLBACKS
        """
        config = None
        
        # Try to load from file first
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                if config is None:
                    raise ConfigurationError(f"Configuration file {self.config_path} is empty or invalid")
                    
                if not isinstance(config, dict):
                    raise ConfigurationError(f"Configuration file {self.config_path} must contain a dictionary")
                    
            except yaml.YAMLError as e:
                raise ConfigurationError(f"Invalid YAML in configuration file {self.config_path}: {e}")
            except Exception as e:
                raise ConfigurationError(f"Cannot load configuration file {self.config_path}: {e}")
        
        # If no file exists, create default config and save it
        if config is None:
            config = self._get_default_config()
            self._save_default_config(config)
        
        # Validate required sections
        self._validate_config_structure(config)
        
        return config
    
    def _save_default_config(self, config: Dict[str, Any]) -> None:
        """Save default configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False, indent=2)
        except Exception as e:
            # Don't fail on save error, just warn
            print(f"Warning: Could not save default config to {self.config_path}: {e}")
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> None:
        """Validate that configuration has required structure"""
        required_sections = [
            'data_sources',
            'processing', 
            'understanding',
            'business_intelligence',
            'schema_integration'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
        
        if missing_sections:
            raise ConfigurationError(f"Configuration missing required sections: {missing_sections}")
    
    def _validate_metadata_integration(self) -> None:
        """
        Validate metadata integration is available - FAIL FAST if broken
        """
        try:
            from agent.nlp_processor.utils.metadata_loader import get_metadata_loader
            metadata_loader = get_metadata_loader()
            
            if metadata_loader is None:
                raise MetadataError("Metadata loader returned None")
                
            # Test that metadata can be loaded
            metadata = metadata_loader.load_all_metadata()
            if not metadata:
                raise MetadataError("Metadata loader returned empty metadata")
                
        except ImportError as e:
            raise MetadataError(f"Cannot import metadata loader: {e}")
        except Exception as e:
            raise MetadataError(f"Metadata integration validation failed: {e}")
    
    def _enhance_config_with_metadata_strict(self) -> None:
        """
        Enhance configuration with metadata - FAIL FAST if enhancement fails
        """
        try:
            from agent.nlp_processor.utils.metadata_loader import get_metadata_loader
            metadata_loader = get_metadata_loader()
            metadata = metadata_loader.load_all_metadata()
            
            # Add enhanced configurations
            self._config["table_configs"] = self._generate_table_configs(metadata)
            self._config["field_mappings"] = self._generate_field_mappings(metadata)
            self._config["join_configs"] = self._generate_join_configs(metadata)
            self._config["xml_configs"] = self._generate_xml_configs(metadata)
            
            # Validate enhanced configuration
            self._validate_enhanced_config()
            
        except Exception as e:
            raise ConfigurationError(f"Failed to enhance configuration with metadata: {e}")
    
    def _validate_enhanced_config(self) -> None:
        """Validate that enhanced configuration is complete"""
        required_enhanced_sections = ['table_configs', 'field_mappings', 'join_configs', 'xml_configs']
        missing_sections = []
        
        for section in required_enhanced_sections:
            if section not in self._config:
                missing_sections.append(section)
        
        if missing_sections:
            raise ConfigurationError(f"Enhanced configuration missing sections: {missing_sections}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration optimized for your system"""
        return {
            "data_sources": {
                "metadata_path": "data/metadata",
                "schema_file": "schema.json",
                "joins_file": "joins_verified.json",
                "xml_schema_file": "xml_schema.json",
                "tables_file": "tables.json"
            },
            "processing": {
                "max_query_length": 500,
                "timeout_seconds": 30,
                "enable_caching": True,
                "performance_target_seconds": 5
            },
            "understanding": {
                "intent_threshold": 0.7,
                "entity_threshold": 0.6,
                "enable_fuzzy_matching": True,
                "temporal_patterns": [
                    "last \\d+ days",
                    "recent", 
                    "within \\d+ \\w+"
                ]
            },
            "business_intelligence": {
                "auto_complete_context": True,
                "resolve_codes_to_names": True,
                "add_temporal_context": True,
                "banking_domain": {
                    "counterparty_context": "always_include_contact_address",
                    "application_context": "include_workflow_status",
                    "region_context": "resolve_to_readable_names",
                    "xml_field_priority": "high"
                }
            },
            "schema_integration": {
                "use_verified_joins": True,
                "join_confidence_threshold": 80,
                "prefer_high_relevance_tables": True,
                "xml_fields_enabled": True,
                "physical_columns_priority": "equal"
            },
            "aggregation": {
                "default_aggregations": ["SUM", "AVG", "MAX", "COUNT"],
                "numerical_fields_auto_aggregate": True,
                "group_by_auto_suggest": True,
                "temporal_aggregation_enabled": True
            },
            "analyst_patterns": {
                "temporal_analysis": {
                    "patterns": ["last \\d+ days", "recent", "within"],
                    "default_columns": ["created_date", "application_date"],
                    "auto_ordering": True
                },
                "regional_analysis": {
                    "patterns": ["region", "state", "location", "geographic"],
                    "default_grouping": "StateAddress",
                    "resolve_codes": True
                },
                "aggregation_analysis": {
                    "patterns": ["sum", "total", "average", "maximum", "count"],
                    "auto_group_by": True,
                    "include_context": True
                }
            },
            "integration": {
                "schema_searcher_timeout": 10,
                "xml_manager_enabled": True,
                "maintain_performance": True,
                "bridge_retry_attempts": 3
            },
            "monitoring": {
                "enable_performance_tracking": True,
                "enable_accuracy_tracking": True,
                "log_level": "INFO"
            },
            "nlp_processor": {
                "enabled": True,
                "timeout": 30,
                "max_retries": 3
            }
        }
    
    def _generate_table_configs(self, metadata: Dict[str, Any]) -> Dict[str, Dict]:
        """Generate table-specific configurations from metadata"""
        table_configs = {}
        
        for table_name, table_info in metadata.get('tables', {}).items():
            table_configs[table_name] = {
                "business_domain": table_info.get('business_domain', 'operational'),
                "analyst_relevance": table_info.get('analyst_relevance', 'medium'),
                "query_patterns": table_info.get('query_patterns', []),
                "auto_joins": table_info.get('common_joins', []),
                "processing_priority": self._calculate_processing_priority(table_info)
            }
        
        return table_configs
    
    def _generate_field_mappings(self, metadata: Dict[str, Any]) -> Dict[str, List]:
        """Generate field mappings for business terms"""
        mappings = {
            "temporal_fields": [],
            "aggregatable_fields": [],
            "lookup_fields": [],
            "foreign_key_fields": []
        }
        
        for column_info in metadata.get('schema', []):
            if column_info.get('temporal', False):
                mappings["temporal_fields"].append({
                    "table": column_info.get('table'),
                    "column": column_info.get('column'),
                    "type": "temporal"
                })
            
            if column_info.get('aggregatable', False):
                mappings["aggregatable_fields"].append({
                    "table": column_info.get('table'),
                    "column": column_info.get('column'),
                    "data_type": column_info.get('datatype')
                })
        
        return mappings
    
    def _generate_join_configs(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate join configurations from verified joins"""
        join_configs = {
            "high_confidence_joins": [],
            "medium_confidence_joins": [],
            "join_priorities": {}
        }
        
        for join_info in metadata.get('joins', []):
            confidence = join_info.get('confidence', 0)
            join_config = {
                "source": join_info.get('source'),
                "target": join_info.get('target'),
                "source_column": join_info.get('source_column'),
                "target_column": join_info.get('target_column'),
                "confidence": confidence
            }
            
            if confidence >= 90:
                join_configs["high_confidence_joins"].append(join_config)
            elif confidence >= 70:
                join_configs["medium_confidence_joins"].append(join_config)
        
        return join_configs
    
    def _generate_xml_configs(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate XML processing configurations"""
        xml_configs = {
            "enabled_tables": [],
            "field_priorities": {},
            "business_domains": {}
        }
        
        for table_name, xml_info in metadata.get('xml_mappings', {}).items():
            xml_configs["enabled_tables"].append(table_name)
        
        return xml_configs
    
    def _calculate_processing_priority(self, table_info: Dict) -> int:
        """Calculate processing priority based on table metadata"""
        priority = 50  # Base priority
        
        relevance = table_info.get('analyst_relevance', 'medium')
        if relevance == 'high':
            priority += 30
        elif relevance == 'medium':
            priority += 10
        
        return min(priority, 100)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary - THIS IS WHAT get_config() SHOULD RETURN
        """
        return self._config.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if not isinstance(value, dict) or k not in value:
                return default
            value = value[k]
        return value
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get configuration statistics"""
        return {
            "total_tables_configured": len(self._config.get('table_configs', {})),
            "high_confidence_joins": len(self._config.get('join_configs', {}).get('high_confidence_joins', [])),
            "xml_enabled_tables": len(self._config.get('xml_configs', {}).get('enabled_tables', [])),
            "temporal_fields": len(self._config.get('field_mappings', {}).get('temporal_fields', [])),
            "aggregatable_fields": len(self._config.get('field_mappings', {}).get('aggregatable_fields', [])),
            "config_file": self.config_path
        }


# Global configuration instance
try:
    _config_instance = NLPConfig()
except Exception as e:
    # FAIL FAST - Don't create broken config instance
    raise ConfigurationError(f"CRITICAL: Cannot initialize NLP configuration: {e}")


def get_config() -> Dict[str, Any]:
    """
    Get global configuration as DICTIONARY (not object)
    ✅ CRITICAL FIX: Returns dictionary that orchestrator expects
    """
    try:
        return _config_instance.to_dict()
    except Exception as e:
        raise ConfigurationError(f"CRITICAL: Cannot get configuration dictionary: {e}")


def get_config_object() -> NLPConfig:
    """Get configuration object for advanced operations"""
    return _config_instance


# Export both functions
__all__ = ["get_config", "get_config_object", "NLPConfig", "ConfigurationError", "MetadataError"]
