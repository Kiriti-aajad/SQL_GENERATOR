"""
Instruction Builder for injecting user query-specific instructions into the prompt.
Dynamically generates instructions based on query type, schema context,
and template configuration.
"""


from typing import Dict, List, Optional, Any
import logging


from ..core.data_models import QueryIntent, SchemaContext, PromptOptions
from ..core.template_manager import TemplateManager


logger = logging.getLogger(__name__) # type: ignore



class InstructionBuilder:
    """
    Dynamically builds SQL generation instructions for LLM prompts.
    """
    
    def __init__(self):
        self.template_manager = TemplateManager()
    
    def build_instructions(
        self,
        query_intent: QueryIntent,
        schema_context: SchemaContext,
        options: PromptOptions,
        template_set: Dict[str, Dict]
    ) -> str:
        """
        Build instruction block for final prompt - FIXED VERSION
        """
        logger.debug("Building instructions section")
        
        try:
            parts = []

            # ✅ FIXED: Validate template_set
            if not template_set or not isinstance(template_set, dict):
                logger.warning("Invalid or empty template_set provided")
                return self._get_fallback_instructions()

            # ✅ FIXED: Safe access to base template
            base_template = template_set.get("base", {})
            if not isinstance(base_template, dict):
                logger.warning("Invalid base template structure")
                return self._get_fallback_instructions()

            # ✅ FIXED: Safe access to generation instructions
            base_instructions = base_template.get("generation_instructions", {})
            if not isinstance(base_instructions, dict):
                logger.warning("Invalid generation_instructions structure")
                return self._get_fallback_instructions()

            # ✅ FIXED: Safe primary directive processing
            primary_directive = base_instructions.get("primary_directive")
            if primary_directive and isinstance(primary_directive, str) and primary_directive.strip():
                clean_directive = self._clean_content(primary_directive)
                if clean_directive:
                    parts.append(clean_directive)

            # ✅ FIXED: Safe generation steps processing
            generation_steps = base_instructions.get("generation_steps")
            if generation_steps and isinstance(generation_steps, list):
                valid_steps = []
                for step in generation_steps:
                    if isinstance(step, str) and step.strip():
                        clean_step = self._clean_content(step)
                        if clean_step:
                            valid_steps.append(f"- {clean_step}")
                
                if valid_steps:
                    parts.append("GENERATION STEPS:")
                    parts.extend(valid_steps)

            # ✅ FIXED: Safe type-specific instruction processing
            type_instructions = base_instructions.get("query_type_instructions", {})
            if isinstance(type_instructions, dict) and query_intent.query_type:
                query_type_key = query_intent.query_type.value
                hint = type_instructions.get(query_type_key, {})
                
                if isinstance(hint, dict):
                    # Process approach
                    approach = hint.get("approach")
                    if approach and isinstance(approach, str) and approach.strip():
                        clean_approach = self._clean_content(approach)
                        if clean_approach:
                            parts.append(f"APPROACH: {clean_approach}")
                    
                    # Process considerations
                    considerations = hint.get("considerations")
                    if considerations and isinstance(considerations, list):
                        valid_considerations = []
                        for item in considerations:
                            if isinstance(item, str) and item.strip():
                                clean_item = self._clean_content(item)
                                if clean_item:
                                    valid_considerations.append(f"- {clean_item}")
                        
                        if valid_considerations:
                            parts.append("CONSIDERATIONS:")
                            parts.extend(valid_considerations)

            # ✅ FIXED: Safe name preservation processing
            name_preservation = base_instructions.get("name_preservation_instructions", {})
            if isinstance(name_preservation, dict):
                absolute_requirements = name_preservation.get("absolute_requirements")
                if absolute_requirements and isinstance(absolute_requirements, str) and absolute_requirements.strip():
                    clean_requirements = self._clean_content(absolute_requirements)
                    if clean_requirements:
                        parts.append("\nMANDATORY NAMING REQUIREMENTS:")
                        parts.append(clean_requirements)

            # ✅ FIXED: Validate final result
            if not parts:
                logger.warning("No valid instruction parts found, using fallback")
                return self._get_fallback_instructions()

            result = "\n".join(parts)
            
            # ✅ FIXED: Final validation
            if not result or not result.strip():
                logger.warning("Empty instruction result, using fallback")
                return self._get_fallback_instructions()

            logger.debug(f"Successfully built instructions: {len(result)} characters")
            return result

        except Exception as e:
            logger.error(f"Error building instructions: {e}")
            logger.exception("Full traceback:")
            return self._get_fallback_instructions()
    
    def _clean_content(self, content: str) -> str:
        """
        Clean template content to prevent regex issues downstream
        """
        if not content:
            return ""
        
        try:
            # Remove potential problematic characters that might be interpreted as regex
            # Keep the content readable but safe
            cleaned = content.strip()
            
            # Log if content contains potential regex metacharacters
            regex_chars = ['[', ']', '(', ')', '{', '}', '^', '$', '*', '+', '?', '|', '\\']
            if any(char in cleaned for char in regex_chars):
                logger.debug(f"Content contains potential regex metacharacters: {cleaned[:50]}...")
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Error cleaning content: {e}")
            return ""
    
    def _get_fallback_instructions(self) -> str:
        """
        Provide fallback instructions when template processing fails
        """
        return """Generate an SQL query to answer the user's natural language request using the schema context and guidelines below.
Use table and column names exactly as provided. Follow all naming conventions precisely.

GENERATION STEPS:
- Analyze the user's natural language query to understand the intent
- Identify the required tables from the provided schema context
- Select appropriate columns that fulfill the query requirements
- Generate syntactically correct SQL for the target database system

MANDATORY NAMING REQUIREMENTS:
- Use table names EXACTLY as provided (preserve case, prefixes, suffixes)
- Use column names EXACTLY as provided (no modifications or translations)
- Maintain exact naming conventions throughout the query"""
    
    def validate_template_content(self, template_set: Dict[str, Dict]) -> bool:
        """
        Validate template content structure
        """
        try:
            if not isinstance(template_set, dict):
                return False
            
            base_template = template_set.get("base", {})
            if not isinstance(base_template, dict):
                return False
            
            generation_instructions = base_template.get("generation_instructions", {})
            if not isinstance(generation_instructions, dict):
                return False
            
            # Check for required sections
            required_sections = ["primary_directive", "generation_steps", "name_preservation_instructions"]
            for section in required_sections:
                if section not in generation_instructions:
                    logger.warning(f"Missing required instruction section: {section}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating template content: {e}")
            return False
    
    def get_instruction_stats(self) -> Dict[str, Any]:
        """
        Get statistics about instruction building
        """
        return {
            "builder_initialized": True,
            "template_manager_available": self.template_manager is not None,
            "fallback_instructions_length": len(self._get_fallback_instructions())
        }
