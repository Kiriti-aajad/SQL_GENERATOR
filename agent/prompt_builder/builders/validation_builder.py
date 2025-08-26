"""
Validation Builder for generating validation checklists in prompts.
Ensures correctness of field references, joins, conditions, and name preservation.
"""

from typing import Dict, List
import logging

from ..core.data_models import PromptType, SchemaContext, QueryIntent

logger = logging.getLogger(__name__) # type: ignore


class ValidationBuilder:
    """
    Generates final validation checks for the prompt before sending to LLM.
    """
    
    def build_validation_checks(
        self,
        schema_context: SchemaContext,
        query_intent: QueryIntent
    ) -> str:
        """
        Generate final checklist block covering required fields and validation criteria.
        """
        logger.debug("Building validation checklist")
        checklist = [
            "VALIDATION CHECKLIST:",
            "- All table names MUST match those in the schema context",
            "- All column names MUST be from the specified tables",
            "- JOIN conditions must use defined relationships (if applicable)",
            "- SQL syntax must be valid and complete",
            "- Prompt output must maintain all schema naming exactly (case & prefix)"
        ]
        
        if query_intent.involves_joins:
            checklist.extend([
                "- JOINs must prevent cross joins (Cartesian products)",
                "- Table aliases must be used in multi-table queries"
            ])
        
        if query_intent.involves_xml:
            checklist.extend([
                "- XML extraction must use the provided SQL expressions and xpaths",
                "- Do not invent your own extraction logic"
            ])
        
        if query_intent.query_type == PromptType.AGGREGATION:
            checklist.extend([
                "- Include GROUP BY where necessary",
                "- Apply aggregate functions correctly",
                "- Use HAVING for filtering aggregates"
            ])

        return "\n".join(checklist)
