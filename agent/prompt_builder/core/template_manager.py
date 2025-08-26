"""
Template Manager for dynamic template loading and selection.
Handles YAML template files and provides intelligent template selection
based on query analysis and configuration.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

from .data_models import (
    TemplateConfig, QueryIntent, PromptType, PromptOptions, QueryComplexity
)

logger = logging.getLogger(__name__)

@dataclass
class LoadedTemplate:
    """Container for a loaded template with metadata"""
    template_id: str
    template_type: PromptType
    content: Dict[str, Any]
    file_path: Path
    loaded_at: datetime
    specializations: List[str]
    priority: int
    
    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        """Check if template cache is expired"""
        return (datetime.now() - self.loaded_at).total_seconds() > ttl_seconds

class TemplateManager:
    """
    Manages template loading, caching, and selection for prompt building.
    Provides intelligent template selection based on query analysis.
    """
    
    def __init__(self, templates_directory: Optional[Path] = None):
        """
        Initialize template manager.
        
        Args:
            templates_directory: Custom template directory path
        """
        self.templates_directory = templates_directory or self._get_default_template_dir()
        self.template_cache: Dict[str, LoadedTemplate] = {}
        self.config_cache: Optional[Dict[str, Any]] = None
        self.config_loaded_at: Optional[datetime] = None
        
        # CRITICAL FIX: Add template type mapping to handle warnings
        self.template_type_mapping = {
            "simple_select": PromptType.SIMPLE_SELECT,
            "join_query": PromptType.JOIN_QUERY,
            "xml_extraction": PromptType.XML_EXTRACTION,
            "aggregation": PromptType.AGGREGATION,
            "complex_filter": PromptType.COMPLEX_FILTER,
            "multi_table": PromptType.MULTI_TABLE,
            
            # CRITICAL FIX: Add missing mappings to fix warnings
            "entity_specific": PromptType.ENTITY_SPECIFIC,
            "intelligent_optimized": PromptType.INTELLIGENT_OPTIMIZED,
            "enhanced_join": PromptType.ENHANCED_JOIN,
            "schema_aware": PromptType.SCHEMA_AWARE,
        }
        
        logger.info(f"TemplateManager initialized with directory: {self.templates_directory}")
    
    def _get_default_template_dir(self) -> Path:
        """Get default templates directory path"""
        return Path(__file__).parent.parent / "templates"
    
    def select_template(
        self, 
        query_type: Union[PromptType, QueryIntent], 
        target_llm: str = "gpt",
        context: Optional[Dict[str, Any]] = None
    ) -> TemplateConfig:
        """
        Select the most appropriate template based on query type and target LLM.
        UPDATED: Fixed signature to match prompt assembler calls.
        
        Args:
            query_type: PromptType enum or QueryIntent object
            target_llm: Target LLM identifier
            context: Additional context for template selection
            
        Returns:
            TemplateConfig for the selected template
        """
        # Handle both PromptType and QueryIntent inputs
        if isinstance(query_type, QueryIntent):
            prompt_type = query_type.query_type
            query_intent = query_type
        else:
            prompt_type = query_type
            # Create a basic QueryIntent for internal processing
            query_intent = QueryIntent(
                query_type=prompt_type,
                complexity=QueryComplexity.MEDIUM,
                involves_joins=False,
                involves_xml=prompt_type == PromptType.XML_EXTRACTION,
                involves_aggregation=prompt_type == PromptType.AGGREGATION,
                target_tables=set(),
                keywords=[],
                confidence=0.8
            )
        
        logger.info(f"Selecting template for query type: {prompt_type.value}, target_llm: {target_llm}")
        
        try:
            # Get template selection configuration
            config = self._get_template_config()
            
            # Find matching templates
            matching_templates = self._find_matching_templates(query_intent, config, target_llm)
            
            if not matching_templates:
                logger.warning(f"No matching templates found for {prompt_type.value}, using default")
                return self._get_default_template(config, prompt_type, target_llm)
            
            # Sort by priority and confidence
            best_template = self._select_best_template(matching_templates, query_intent, target_llm)
            
            logger.info(f"Selected template: {best_template.template_id}")
            return best_template
            
        except Exception as e:
            logger.error(f"Template selection failed: {e}, using fallback")
            return self._get_fallback_template(prompt_type, target_llm)
    
    def _find_matching_templates(
        self, 
        query_intent: QueryIntent, 
        config: Dict[str, Any],
        target_llm: str
    ) -> List[TemplateConfig]:
        """Find all templates that match the query intent"""
        matching_templates = []
        template_selection = config.get("template_selection", {})
        
        for template_id, template_spec in template_selection.items():
            if self._template_matches_intent(template_spec, query_intent, target_llm):
                template_config = self._create_template_config(template_id, template_spec)
                if template_config:  # Only add valid configs
                    matching_templates.append(template_config)
        
        return matching_templates
    
    def _template_matches_intent(
        self, 
        template_spec: Dict[str, Any], 
        query_intent: QueryIntent,
        target_llm: str
    ) -> bool:
        """Check if a template specification matches the query intent"""
        triggers = template_spec.get("triggers", {})
        
        # Check template type with improved validation
        template_type_str = template_spec.get("template_type", "")
        try:
            # CRITICAL FIX: Use mapping to handle all template types
            if template_type_str in self.template_type_mapping:
                template_type = self.template_type_mapping[template_type_str]
            else:
                template_type = PromptType(template_type_str)
                
            if template_type != query_intent.query_type:
                return False
        except ValueError:
            logger.warning(f"Invalid template type: {template_type_str}")
            return False
        
        # Check target LLM compatibility
        supported_llms = template_spec.get("supported_llms", ["gpt", "claude", "gemini", "mistral", "defog", "deepseek", "mathstral"])
        if target_llm not in supported_llms:
            return False
        
        # Check complexity requirements
        complexity_requirements = triggers.get("complexity", [])
        if complexity_requirements:
            if isinstance(complexity_requirements, str):
                complexity_requirements = [complexity_requirements]
            if query_intent.complexity.value not in complexity_requirements:
                return False
        
        # Check table count requirements
        table_count_req = triggers.get("table_count", "")
        if table_count_req:
            actual_count = len(query_intent.target_tables)
            if not self._evaluate_count_condition(actual_count, table_count_req):
                return False
        
        # Check boolean flags
        bool_flags = ["involves_joins", "involves_xml", "involves_aggregation"]
        for flag in bool_flags:
            if flag in triggers:
                required_value = triggers[flag]
                actual_value = getattr(query_intent, flag, False)
                if required_value != actual_value:
                    return False
        
        # Check keyword requirements
        required_keywords = triggers.get("keywords", [])
        if required_keywords:
            query_keywords_lower = [kw.lower() for kw in query_intent.keywords]
            for required_keyword in required_keywords:
                if required_keyword.lower() not in query_keywords_lower:
                    return False
        
        return True
    
    def _evaluate_count_condition(self, actual_count: int, condition: str) -> bool:
        """Evaluate count-based conditions like '> 1', '<= 2', etc."""
        condition = condition.strip()
        
        try:
            if condition.startswith("<="):
                threshold = int(condition[2:].strip())
                return actual_count <= threshold
            elif condition.startswith(">="):
                threshold = int(condition[2:].strip())
                return actual_count >= threshold
            elif condition.startswith("<"):
                threshold = int(condition[1:].strip())
                return actual_count < threshold
            elif condition.startswith(">"):
                threshold = int(condition[1:].strip())
                return actual_count > threshold
            elif condition.startswith("==") or condition.startswith("="):
                threshold = int(condition.lstrip("=").strip())
                return actual_count == threshold
            else:
                # Try direct integer comparison
                threshold = int(condition)
                return actual_count == threshold
        except ValueError:
            logger.warning(f"Invalid count condition: {condition}")
            return True
    
    def _create_template_config(
        self, 
        template_id: str, 
        template_spec: Dict[str, Any]
    ) -> Optional[TemplateConfig]:
        """Create TemplateConfig from specification with validation"""
        try:
            template_type_str = template_spec.get("template_type", "simple_select")
            
            # CRITICAL FIX: Use mapping for template type conversion
            if template_type_str in self.template_type_mapping:
                template_type = self.template_type_mapping[template_type_str]
            else:
                template_type = PromptType(template_type_str)
            
            return TemplateConfig(
                template_id=template_id,
                template_type=template_type,
                base_template=template_spec.get("base_template", "base/system_context.yaml"),
                specializations=template_spec.get("specializations", []),
                triggers=template_spec.get("triggers", {}),
                priority=template_spec.get("priority", 1)
            )
        except (ValueError, KeyError) as e:
            logger.error(f"Failed to create template config for {template_id}: {e}")
            return None
    
    def _select_best_template(
        self, 
        matching_templates: List[TemplateConfig], 
        query_intent: QueryIntent,
        target_llm: str
    ) -> TemplateConfig:
        """Select the best template from matching candidates"""
        if len(matching_templates) == 1:
            return matching_templates[0]
        
        # Score templates based on various factors
        scored_templates = []
        for template in matching_templates:
            score = self._calculate_template_score(template, query_intent, target_llm)
            scored_templates.append((template, score))
        
        # Sort by score (higher is better)
        scored_templates.sort(key=lambda x: x[1], reverse=True)
        
        best_template = scored_templates[0][0]
        logger.debug(f"Template scoring results: {[(t.template_id, s) for t, s in scored_templates]}")
        
        return best_template
    
    def _calculate_template_score(
        self, 
        template: TemplateConfig, 
        query_intent: QueryIntent,
        target_llm: str
    ) -> float:
        """Calculate a score for template selection"""
        score = float(template.priority)
        
        # Boost for exact template type match
        if template.template_type == query_intent.query_type:
            score += 10.0
        
        # Boost for specializations that match intent
        if template.specializations:
            specialization_boost = len(template.specializations) * 2.0
            score += specialization_boost
        
        # Boost for XML specialization when needed
        if query_intent.involves_xml and any("xml" in spec.lower() for spec in template.specializations):
            score += 5.0
        
        # Boost for join specialization when needed
        if query_intent.involves_joins and any("join" in spec.lower() for spec in template.specializations):
            score += 3.0
        
        # Boost for LLM-specific templates
        if target_llm.lower() in template.template_id.lower():
            score += 4.0
        
        return score
    
    def _get_default_template(
        self, 
        config: Dict[str, Any], 
        prompt_type: PromptType, 
        target_llm: str
    ) -> TemplateConfig:
        """Get the default fallback template"""
        default_config = config.get("default_template", {})
        return TemplateConfig(
            template_id=f"default_{prompt_type.value}_{target_llm}",
            template_type=prompt_type,
            base_template=default_config.get("base_template", "base/system_context.yaml"),
            specializations=[],
            triggers={},
            priority=0
        )
    
    def _get_fallback_template(self, prompt_type: PromptType, target_llm: str) -> TemplateConfig:
        """Emergency fallback template when everything fails"""
        return TemplateConfig(
            template_id=f"fallback_{prompt_type.value}",
            template_type=prompt_type,
            base_template="base/system_context.yaml",
            specializations=[],
            triggers={},
            priority=0
        )
    
    def get_template_config(self) -> Dict[str, Any]:
        """Get template configuration - called by config.py"""
        return self._get_template_config()
    
    def load_template(self, template_path: str) -> Dict[str, Any]:
        """
        Load a template file and return its content.
        
        Args:
            template_path: Relative path to template file
            
        Returns:
            Template content as dictionary
        """
        # Check cache first
        if template_path in self.template_cache:
            cached_template = self.template_cache[template_path]
            if not cached_template.is_expired():
                logger.debug(f"Using cached template: {template_path}")
                return cached_template.content
        
        # Load from file
        full_path = self.templates_directory / template_path
        
        if not full_path.exists():
            logger.warning(f"Template file not found: {full_path}, using default content")
            return self._get_default_template_content()
        
        try:
            with open(full_path, 'r', encoding='utf-8') as file:
                content = yaml.safe_load(file) or {}
                
            # Cache the loaded template
            loaded_template = LoadedTemplate(
                template_id=template_path,
                template_type=PromptType.SIMPLE_SELECT,  # Will be updated by caller
                content=content,
                file_path=full_path,
                loaded_at=datetime.now(),
                specializations=[],
                priority=1
            )
            
            self.template_cache[template_path] = loaded_template
            logger.info(f"Loaded template: {template_path}")
            
            return content
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML template {full_path}: {e}")
            return self._get_default_template_content()
        except Exception as e:
            logger.error(f"Error loading template {full_path}: {e}")
            return self._get_default_template_content()
    
    def _get_default_template_content(self) -> Dict[str, Any]:
        """Get default template content when files are missing"""
        return {
            "system_prompts": {
                "primary_context": "You are an expert SQL query generator.",
                "core_instructions": [
                    "Generate accurate SQL queries based on provided schema",
                    "Preserve all table and column names exactly",
                    "Use proper JOIN syntax when needed"
                ],
                "name_preservation_directive": "CRITICAL: Preserve all database names exactly as provided"
            },
            "instructions": "Generate SQL based on the provided schema context.",
            "validation_checklist": {
                "final_checks": [
                    "Verify table names are preserved exactly",
                    "Check SQL syntax is correct",
                    "Ensure all required columns are included"
                ]
            },
            "examples": {
                "simple_select": [
                    "SELECT * FROM Users WHERE active = 1",
                    "SELECT name, email FROM Users ORDER BY name"
                ],
                "xml_extraction": [
                    "SELECT table.column.value('(/path/to/element)[1]', 'VARCHAR(100)') as extracted_value FROM table"
                ]
            }
        }
    
    def load_template_set(self, template_config: TemplateConfig) -> Dict[str, Dict[str, Any]]:
        """
        Load a complete template set including base template and specializations.
        
        Args:
            template_config: Configuration specifying templates to load
            
        Returns:
            Dictionary of template content keyed by template path
        """
        template_set = {}
        
        # Load base template
        base_content = self.load_template(template_config.base_template)
        template_set["base"] = base_content
        
        # Load specializations
        for specialization in template_config.specializations:
            try:
                spec_content = self.load_template(specialization)
                template_set[specialization] = spec_content
            except Exception as e:
                logger.warning(f"Specialization template not found: {specialization}, using default")
                template_set[specialization] = self._get_default_template_content()
        
        return template_set
    
    def get_available_templates(self) -> List[str]:
        """Get list of available template files"""
        templates = []
        
        if not self.templates_directory.exists():
            logger.warning(f"Templates directory does not exist: {self.templates_directory}")
            return templates
        
        # Walk through templates directory
        for root, dirs, files in os.walk(self.templates_directory):
            for file in files:
                if file.endswith('.yaml') or file.endswith('.yml'):
                    # Get relative path from templates directory
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(self.templates_directory)
                    templates.append(str(rel_path))
        
        return sorted(templates)
    
    def _get_template_config(self) -> Dict[str, Any]:
        """Get template configuration with caching"""
        # Check if cache is expired (5 minutes TTL)
        if (self.config_cache is None or 
            self.config_loaded_at is None or
            (datetime.now() - self.config_loaded_at).total_seconds() > 300):
            
            # Try to load from config, fall back to default if needed
            try:
                from ..config import get_template_config
                self.config_cache = get_template_config()
            except Exception as e:
                logger.warning(f"Failed to load template config: {e}, using default")
                self.config_cache = self._get_default_config()
            
            self.config_loaded_at = datetime.now()
            logger.debug("Reloaded template configuration")
        
        return self.config_cache
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when config files are missing"""
        return {
            "template_selection": {
                "sql_generation_gpt": {
                    "template_type": "simple_select",
                    "base_template": "base/system_context.yaml",
                    "priority": 5,
                    "supported_llms": ["gpt", "claude", "gemini"],
                    "triggers": {}
                },
                "xml_extraction_local": {
                    "template_type": "xml_extraction",
                    "base_template": "base/system_context.yaml",
                    "priority": 8,
                    "supported_llms": ["mistral", "defog", "deepseek", "mathstral"],
                    "triggers": {
                        "involves_xml": True
                    },
                    "specializations": ["xml/extraction_patterns.yaml"]
                },
                "join_query_optimized": {
                    "template_type": "join_query",
                    "base_template": "base/system_context.yaml",
                    "priority": 7,
                    "supported_llms": ["gpt", "claude", "gemini", "mistral", "defog", "deepseek", "mathstral"],
                    "triggers": {
                        "involves_joins": True
                    },
                    "specializations": ["joins/relationship_patterns.yaml"]
                },
                # CRITICAL FIX: Add missing template types to prevent warnings
                "entity_specific_analyzer": {
                    "template_type": "entity_specific",
                    "base_template": "base/system_context.yaml",
                    "priority": 9,
                    "supported_llms": ["mistral", "defog", "deepseek", "mathstral"],
                    "triggers": {
                        "complexity": ["high"]
                    },
                    "specializations": ["entity/analysis_patterns.yaml"]
                },
                "intelligent_schema_processor": {
                    "template_type": "intelligent_optimized",
                    "base_template": "base/system_context.yaml",
                    "priority": 10,
                    "supported_llms": ["gpt", "claude", "gemini", "mistral", "defog", "deepseek", "mathstral"],
                    "triggers": {
                        "table_count": "> 5"
                    },
                    "specializations": ["intelligent/optimization_patterns.yaml"]
                },
                "enhanced_join_resolver": {
                    "template_type": "enhanced_join",
                    "base_template": "base/system_context.yaml",
                    "priority": 8,
                    "supported_llms": ["mistral", "defog", "deepseek", "mathstral"],
                    "triggers": {
                        "involves_joins": True,
                        "table_count": "> 3"
                    },
                    "specializations": ["joins/enhanced_patterns.yaml"]
                },
                "schema_aware_generator": {
                    "template_type": "schema_aware",
                    "base_template": "base/system_context.yaml",
                    "priority": 6,
                    "supported_llms": ["gpt", "claude", "gemini", "mistral", "defog", "deepseek", "mathstral"],
                    "triggers": {},
                    "specializations": ["schema/awareness_patterns.yaml"]
                }
            },
            "default_template": {
                "template_type": "simple_select",
                "base_template": "base/system_context.yaml",
                "supported_llms": ["gpt", "claude", "gemini", "mistral", "defog", "deepseek", "mathstral"]
            }
        }
    
    def clear_cache(self):
        """Clear template cache to force reload"""
        self.template_cache.clear()
        self.config_cache = None
        self.config_loaded_at = None
        logger.info("Template cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return {
            "cached_templates": len(self.template_cache),
            "config_cached": self.config_cache is not None,
            "config_age_seconds": (
                (datetime.now() - self.config_loaded_at).total_seconds() 
                if self.config_loaded_at else None
            ),
            "template_files_available": len(self.get_available_templates()),
            "template_type_mappings": len(self.template_type_mapping)
        }
    
    def validate_template_types(self) -> Dict[str, Any]:
        """Validate all template types in configuration"""
        validation_results = {
            "valid_types": [],
            "invalid_types": [],
            "missing_mappings": []
        }
        
        config = self._get_template_config()
        template_selection = config.get("template_selection", {})
        
        for template_id, template_spec in template_selection.items():
            template_type_str = template_spec.get("template_type", "")
            
            if template_type_str in self.template_type_mapping:
                validation_results["valid_types"].append(template_type_str)
            else:
                try:
                    PromptType(template_type_str)
                    validation_results["valid_types"].append(template_type_str)
                except ValueError:
                    validation_results["invalid_types"].append(template_type_str)
        
        return validation_results
