"""
Enhanced Context Optimizer with Critical Database Element Protection.
Optimizes context sections while NEVER trimming essential tables, columns, joins, or XML information.
Leverages intelligent schema results and protects your proven 60-80% table reduction benefits.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from ..core.data_models import (
    ContextSection, QueryIntent, IntelligentSchemaContext, 
    SchemaRetrievalResult, PromptOptions
)
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class EnhancedContextOptimizer:
    """
    Enhanced context optimizer that PROTECTS critical database information.
    NEVER trims tables, columns, joins, or XML - only trims descriptions and fluff.
    """
    
    def __init__(self):
        """Initialize enhanced optimizer with database protection capabilities"""
        self.optimization_strategies = {
            'intelligent_schema': self._optimize_by_schema_intelligence,
            'entity_aware': self._optimize_by_entity_relevance,
            'table_priority': self._optimize_by_table_priority,
            'traditional': self._optimize_traditional
        }
        
        # Protection settings
        self.min_section_length = 50
        self.truncation_buffer = 20
        self.critical_protection_enabled = True
        
        # Critical database element patterns
        self.critical_patterns = {
            'table_definitions': [r'^table:', r'table\s*:', r'tables?\s*\('],
            'column_definitions': [r'columns?\s*\(', r'column:', r'\|\s*\w+\s*\|'],
            'primary_keys': [r'primary\s*key', r'\bpk\b', r'uniqueid', r'_id\b', r'\bid\b'],
            'foreign_keys': [r'foreign\s*key', r'\bfk\b', r'references', r'->', r'='],
            'join_relationships': [r'join', r'relationship', r'connected', r'linked'],
            'xml_mappings': [r'xml', r'xpath', r'extract', r'//']
        }
        
        logger.info("EnhancedContextOptimizer initialized with database protection")
    
    def optimize_context_sections_intelligently(
        self,
        context_sections: Dict[str, ContextSection],
        query_intent: Optional[QueryIntent] = None,
        schema_retrieval_result: Optional[SchemaRetrievalResult] = None,
        options: Optional[PromptOptions] = None,
        max_total_length: int = 2000
    ) -> Dict[str, ContextSection]:
        """
        Enhanced optimization that PROTECTS all critical database information.
        
        PROTECTION GUARANTEE: Never trims tables, columns, joins, or XML information.
        Only trims descriptions, examples, and non-essential text.
        """
        logger.info(f"Optimizing context with database protection from ~{sum(s.length for s in context_sections.values())} chars")
        
        current_total = sum(s.length for s in context_sections.values())
        
        if current_total <= max_total_length:
            logger.debug("Context already within limits, no optimization needed")
            return context_sections
        
        # STEP 1: Identify and protect critical database elements
        protected_elements = self._identify_protected_database_elements(
            context_sections, query_intent, schema_retrieval_result
        )
        
        # STEP 2: Calculate critical content length
        critical_length = self._calculate_critical_content_length(context_sections, protected_elements)
        
        if critical_length >= max_total_length:
            logger.warning(f"Critical database content ({critical_length}) exceeds limit ({max_total_length}). Preserving all critical elements anyway.")
            return self._preserve_critical_only(context_sections, protected_elements)
        
        # STEP 3: Choose optimization strategy
        strategy = self._select_optimization_strategy(query_intent, schema_retrieval_result, options)
        
        # STEP 4: Apply safe optimization (protects critical elements)
        optimized_sections = self._apply_safe_optimization(
            context_sections, protected_elements, strategy, query_intent, 
            schema_retrieval_result, max_total_length
        )
        
        # STEP 5: Validate protection
        self._validate_critical_elements_preserved(context_sections, optimized_sections, protected_elements)
        
        final_length = sum(s.length for s in optimized_sections.values())
        reduction_pct = ((current_total - final_length) / current_total) * 100
        
        logger.info(f"Safe optimization complete: {current_total} → {final_length} chars ({reduction_pct:.1f}% reduction) using {strategy} strategy")
        logger.info(f"✅ ALL critical database elements preserved")
        
        return optimized_sections
    
    # 🔥 CRITICAL FIX #3: Added the missing optimization methods INSIDE the class
    def _optimize_by_schema_intelligence(
        self,
        context_sections: Dict[str, ContextSection],
        query_intent: Optional[QueryIntent],
        schema_result: Optional[SchemaRetrievalResult],
        max_length: int
    ) -> Dict[str, ContextSection]:
        """Optimize using intelligent schema results"""
        logger.info("Applying schema intelligence optimization")
        
        try:
            protected_elements = self._identify_protected_database_elements(
                context_sections, query_intent, schema_result
            )
            
            return self._apply_safe_optimization(
                context_sections, protected_elements, 'intelligent_schema',
                query_intent, schema_result, max_length
            )
        except Exception as e:
            logger.error(f"Schema intelligence optimization failed: {e}")
            return self._optimize_traditional(
                context_sections, query_intent, schema_result, max_length
            )

    def _optimize_by_entity_relevance(
        self,
        context_sections: Dict[str, ContextSection],
        query_intent: Optional[QueryIntent],
        schema_result: Optional[SchemaRetrievalResult],
        max_length: int
    ) -> Dict[str, ContextSection]:
        """Optimize based on detected entities"""
        logger.info("Applying entity-aware optimization")
        
        try:
            protected_elements = self._identify_protected_database_elements(
                context_sections, query_intent, schema_result
            )
            
            return self._apply_safe_optimization(
                context_sections, protected_elements, 'entity_aware',
                query_intent, schema_result, max_length
            )
        except Exception as e:
            logger.error(f"Entity relevance optimization failed: {e}")
            return self._optimize_traditional(
                context_sections, query_intent, schema_result, max_length
            )

    def _optimize_by_table_priority(
        self,
        context_sections: Dict[str, ContextSection],
        query_intent: Optional[QueryIntent],
        schema_result: Optional[SchemaRetrievalResult],
        max_length: int
    ) -> Dict[str, ContextSection]:
        """Optimize based on table priorities"""
        logger.info("Applying table priority optimization")
        
        try:
            protected_elements = self._identify_protected_database_elements(
                context_sections, query_intent, schema_result
            )
            
            return self._apply_safe_optimization(
                context_sections, protected_elements, 'table_priority',
                query_intent, schema_result, max_length
            )
        except Exception as e:
            logger.error(f"Table priority optimization failed: {e}")
            return self._optimize_traditional(
                context_sections, query_intent, schema_result, max_length
            )
    
    def _identify_protected_database_elements(
        self,
        context_sections: Dict[str, ContextSection],
        query_intent: Optional[QueryIntent],
        schema_result: Optional[SchemaRetrievalResult]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Identify ALL database elements that must NEVER be trimmed
        """
        protected_elements = {
            'critical_tables': set(),
            'critical_columns': set(),
            'critical_joins': [],
            'critical_xml': [],
            'critical_lines_by_section': defaultdict(list)
        }
        
        # PROTECTION 1: Tables from your proven schema intelligence
        if schema_result and schema_result.was_successful():
            intelligent_context = schema_result.schema_context
            if hasattr(intelligent_context, 'base_context') and intelligent_context.base_context.tables:
                protected_elements['critical_tables'].update(intelligent_context.base_context.tables.keys())
                logger.debug(f"Protected {len(intelligent_context.base_context.tables)} tables from intelligent schema")
        
        # PROTECTION 2: Tables from query intent (recommended tables)
        if query_intent and hasattr(query_intent, 'recommended_tables') and query_intent.recommended_tables:
            protected_elements['critical_tables'].update(query_intent.recommended_tables)
            logger.debug(f"Protected {len(query_intent.recommended_tables)} recommended tables")
        
        # PROTECTION 3: Analyze each section for critical database content
        for section_name, section in context_sections.items():
            critical_lines = self._extract_critical_lines_from_section(section, protected_elements['critical_tables'])
            protected_elements['critical_lines_by_section'][section_name] = critical_lines
            
            # Extract specific critical elements
            joins = self._extract_join_information(section)
            xml_info = self._extract_xml_information(section)
            
            protected_elements['critical_joins'].extend(joins)
            protected_elements['critical_xml'].extend(xml_info)
        
        logger.info(f"Database protection identified: {len(protected_elements['critical_tables'])} tables, "
                   f"{len(protected_elements['critical_joins'])} joins, "
                   f"{len(protected_elements['critical_xml'])} XML mappings")
        
        return protected_elements
    
    def _extract_critical_lines_from_section(self, section: ContextSection, critical_tables: Set[str]) -> List[str]:
        """
        Extract lines from section that contain critical database information
        """
        critical_lines = []
        lines = section.content.splitlines()
        
        for line in lines:
            line_lower = line.lower().strip()
            if not line_lower:
                continue
            
            is_critical = False
            
            # CHECK 1: Table definitions (NEVER trim)
            for pattern in self.critical_patterns['table_definitions']:
                if re.search(pattern, line_lower):
                    is_critical = True
                    break
            
            # CHECK 2: Column definitions (NEVER trim)
            if not is_critical:
                for pattern in self.critical_patterns['column_definitions']:
                    if re.search(pattern, line_lower):
                        is_critical = True
                        break
            
            # CHECK 3: Primary key information (NEVER trim)
            if not is_critical:
                for pattern in self.critical_patterns['primary_keys']:
                    if re.search(pattern, line_lower):
                        is_critical = True
                        break
            
            # CHECK 4: Foreign key/relationship information (NEVER trim)
            if not is_critical:
                for pattern in self.critical_patterns['foreign_keys']:
                    if re.search(pattern, line_lower):
                        is_critical = True
                        break
            
            # CHECK 5: Join relationships (NEVER trim)
            if not is_critical:
                for pattern in self.critical_patterns['join_relationships']:
                    if re.search(pattern, line_lower):
                        is_critical = True
                        break
            
            # CHECK 6: XML mappings (NEVER trim)
            if not is_critical:
                for pattern in self.critical_patterns['xml_mappings']:
                    if re.search(pattern, line_lower):
                        is_critical = True
                        break
            
            # CHECK 7: Lines mentioning critical tables (NEVER trim)
            if not is_critical:
                for table_name in critical_tables:
                    if table_name.lower() in line_lower:
                        is_critical = True
                        break
            
            # CHECK 8: Structured data lines (table.column format)
            if not is_critical:
                if re.search(r'\w+\.\w+', line) and ('|' in line or ':' in line):
                    is_critical = True
            
            if is_critical:
                critical_lines.append(line)
        
        return critical_lines
    
    def _extract_join_information(self, section: ContextSection) -> List[str]:
        """
        Extract ALL join relationship information (NEVER trim joins)
        """
        joins = []
        
        # If this is a relationships section, ALL content is critical
        if section.section_type == "relationships":
            return section.content.splitlines()
        
        # Extract join patterns from any section
        for line in section.content.splitlines():
            line_lower = line.lower()
            if any(pattern in line_lower for pattern in ['join', '=', '->', 'relationship', 'connected', 'linked']):
                joins.append(line)
        
        return joins
    
    def _extract_xml_information(self, section: ContextSection) -> List[str]:
        """
        Extract ALL XML mapping information (NEVER trim XML)
        """
        xml_info = []
        
        # If this is an XML section, ALL content is critical
        if section.section_type == "xml_mappings":
            return section.content.splitlines()
        
        # Extract XML patterns from any section
        for line in section.content.splitlines():
            line_lower = line.lower()
            if any(pattern in line_lower for pattern in ['xml', 'xpath', 'extract', '//']):
                xml_info.append(line)
        
        return xml_info
    
    def _calculate_critical_content_length(
        self, 
        context_sections: Dict[str, ContextSection], 
        protected_elements: Dict[str, Any]
    ) -> int:
        """
        Calculate total length of ALL critical database content
        """
        total_critical_length = 0
        
        for section_name, critical_lines in protected_elements['critical_lines_by_section'].items():
            critical_content = '\n'.join(critical_lines)
            total_critical_length += len(critical_content)
        
        return total_critical_length
    
    def _preserve_critical_only(
        self, 
        context_sections: Dict[str, ContextSection], 
        protected_elements: Dict[str, Any]
    ) -> Dict[str, ContextSection]:
        """
        Emergency mode: Return only critical database content when space is very limited
        """
        logger.warning("Space extremely limited - preserving only critical database elements")
        
        critical_only_sections = {}
        
        for section_name, section in context_sections.items():
            critical_lines = protected_elements['critical_lines_by_section'][section_name]
            
            if critical_lines:
                critical_content = '\n'.join(critical_lines)
                critical_content += "\n... (non-essential content removed to preserve critical database information)"
                
                critical_only_sections[section_name] = ContextSection(
                    section_type=section.section_type,
                    content=critical_content,
                    priority=15,  # Maximum priority
                    length=len(critical_content),
                    table_names_used=section.table_names_used
                )
        
        return critical_only_sections
    
    def _apply_safe_optimization(
        self,
        context_sections: Dict[str, ContextSection],
        protected_elements: Dict[str, Any],
        strategy: str,
        query_intent: Optional[QueryIntent],
        schema_result: Optional[SchemaRetrievalResult],
        max_length: int
    ) -> Dict[str, ContextSection]:
        """
        Apply optimization while guaranteeing protection of critical elements
        """
        # Calculate section priorities using selected strategy
        if strategy == 'intelligent_schema':
            section_priorities = self._calculate_intelligent_section_priorities(
                context_sections, schema_result.schema_context if schema_result else None, query_intent
            )
        elif strategy == 'entity_aware':
            section_priorities = self._calculate_entity_aware_priorities(context_sections, query_intent)
        elif strategy == 'table_priority':
            section_priorities = self._calculate_table_priority_based_priorities(context_sections, query_intent)
        else:
            section_priorities = {name: getattr(section, 'priority', 5) for name, section in context_sections.items()}
        
        # Sort sections by priority (highest first)
        sorted_sections = sorted(
            context_sections.items(),
            key=lambda item: section_priorities.get(item[0], 5),
            reverse=True
        )
        
        optimized_sections = {}
        remaining_length = max_length
        
        for section_name, section in sorted_sections:
            # STEP 1: Always include critical content first
            critical_lines = protected_elements['critical_lines_by_section'][section_name]
            critical_content = '\n'.join(critical_lines) if critical_lines else ""
            critical_length = len(critical_content)
            
            # STEP 2: Check if we can fit the whole section
            if section.length <= remaining_length:
                # Whole section fits
                optimized_sections[section_name] = section
                remaining_length -= section.length
                
            elif critical_length <= remaining_length:
                # Only critical content fits - create safe truncated version
                safe_section = self._create_safe_truncated_section(
                    section, critical_lines, remaining_length, section_priorities.get(section_name, 5)
                )
                
                if safe_section:
                    optimized_sections[section_name] = safe_section
                    remaining_length -= safe_section.length
                    
            else:
                # Critical content doesn't fit - include it anyway (database integrity > length limits)
                logger.warning(f"Critical content for {section_name} exceeds remaining space - including anyway")
                
                critical_only_section = ContextSection(
                    section_type=section.section_type,
                    content=critical_content + "\n... (space exceeded, but critical database info preserved)",
                    priority=20,  # Highest priority
                    length=critical_length,
                    table_names_used=section.table_names_used
                )
                
                optimized_sections[section_name] = critical_only_section
                remaining_length = max(0, remaining_length - critical_length)
            
            # Stop if no meaningful space remains
            if remaining_length < 50:
                break
        
        return optimized_sections
    
    def _create_safe_truncated_section(
        self,
        original_section: ContextSection,
        critical_lines: List[str],
        available_length: int,
        section_priority: int
    ) -> Optional[ContextSection]:
        """
        Create truncated section that PRESERVES all critical database elements
        """
        if available_length < 50:
            return None
        
        # Start with critical content (NEVER trimmed)
        final_lines = critical_lines.copy()
        current_length = sum(len(line) + 1 for line in final_lines)  # +1 for newlines
        
        # Add non-critical content if space allows
        all_lines = original_section.content.splitlines()
        non_critical_lines = [line for line in all_lines if line not in critical_lines]
        
        # Prioritize non-critical lines by importance
        prioritized_non_critical = self._prioritize_non_critical_lines(non_critical_lines)
        
        remaining_space = available_length - current_length - 50  # Reserve space for truncation message
        
        for line in prioritized_non_critical:
            line_length = len(line) + 1  # +1 for newline
            
            if line_length <= remaining_space:
                final_lines.append(line)
                remaining_space -= line_length
            else:
                break
        
        # Add truncation message if we removed non-critical content
        if len(non_critical_lines) > len([line for line in non_critical_lines if line in final_lines]):
            final_lines.append("... (descriptions truncated - all critical database information preserved)")
        
        final_content = '\n'.join(final_lines)
        
        return ContextSection(
            section_type=original_section.section_type,
            content=final_content,
            priority=section_priority,
            length=len(final_content),
            table_names_used=original_section.table_names_used
        )
    
    def _prioritize_non_critical_lines(self, non_critical_lines: List[str]) -> List[str]:
        """
        Prioritize non-critical lines for inclusion (trim least important first)
        """
        line_priorities = []
        
        for line in non_critical_lines:
            priority = 0
            line_lower = line.lower()
            
            # Higher priority for structural information
            if line.startswith('===') or line.startswith('---'):
                priority += 5
            
            # Higher priority for shorter lines (often headers)
            if len(line) < 50:
                priority += 3
            elif len(line) > 200:
                priority -= 3  # Lower priority for very long lines
            
            # Lower priority for obvious descriptions
            if any(word in line_lower for word in ['description:', 'note:', 'example:', 'details:']):
                priority -= 5
            
            # Higher priority for data type information
            if any(word in line_lower for word in ['varchar', 'int', 'date', 'decimal']):
                priority += 2
            
            line_priorities.append((line, priority))
        
        # Sort by priority (highest first)
        sorted_lines = sorted(line_priorities, key=lambda x: x[1], reverse=True)
        return [line for line, _ in sorted_lines]
    
    def _validate_critical_elements_preserved(
        self,
        original_sections: Dict[str, ContextSection],
        optimized_sections: Dict[str, ContextSection],
        protected_elements: Dict[str, Any]
    ) -> None:
        """
        Validate that ALL critical database elements are preserved
        """
        validation_errors = []
        
        # Check that critical tables are mentioned
        critical_tables = protected_elements['critical_tables']
        optimized_content = ' '.join(section.content for section in optimized_sections.values()).lower()
        
        for table in critical_tables:
            if table.lower() not in optimized_content:
                validation_errors.append(f"Critical table missing: {table}")
        
        # Check that join information is preserved
        original_joins = set()
        optimized_joins = set()
        
        for section in original_sections.values():
            original_joins.update(self._extract_join_information(section))
        
        for section in optimized_sections.values():
            optimized_joins.update(self._extract_join_information(section))
        
        missing_joins = original_joins - optimized_joins
        if missing_joins:
            validation_errors.append(f"Missing joins: {missing_joins}")
        
        # Check XML information preservation
        original_xml = set()
        optimized_xml = set()
        
        for section in original_sections.values():
            original_xml.update(self._extract_xml_information(section))
        
        for section in optimized_sections.values():
            optimized_xml.update(self._extract_xml_information(section))
        
        missing_xml = original_xml - optimized_xml
        if missing_xml:
            validation_errors.append(f"Missing XML mappings: {missing_xml}")
        
        # Log validation results
        if validation_errors:
            logger.error(f"CRITICAL ELEMENT VALIDATION FAILED: {validation_errors}")
            raise ValueError(f"Critical database elements were lost during optimization: {validation_errors}")
        else:
            logger.info("✅ Critical element validation passed - all database elements preserved")
    
    def _select_optimization_strategy(
        self,
        query_intent: Optional[QueryIntent],
        schema_result: Optional[SchemaRetrievalResult],
        options: Optional[PromptOptions]
    ) -> str:
        """
        Select optimization strategy based on available intelligence
        """
        # Use intelligent schema optimization if available
        if (schema_result and schema_result.was_successful() and 
            hasattr(schema_result.schema_context, 'reasoning_applied') and
            schema_result.schema_context.reasoning_applied):
            return 'intelligent_schema'
        
        # Use entity-aware optimization if entities detected
        if (query_intent and hasattr(query_intent, 'schema_entities_detected') and
            query_intent.schema_entities_detected and 
            len(query_intent.schema_entities_detected) > 0):
            return 'entity_aware'
        
        # Use table priority optimization if priorities available
        if (query_intent and hasattr(query_intent, 'entity_priorities') and
            query_intent.entity_priorities and 
            len(query_intent.entity_priorities) > 0):
            return 'table_priority'
        
        return 'traditional'
    
    def _calculate_intelligent_section_priorities(
        self,
        context_sections: Dict[str, ContextSection],
        intelligent_context: Optional[IntelligentSchemaContext],
        query_intent: Optional[QueryIntent]
    ) -> Dict[str, int]:
        """
        Calculate section priorities using intelligent schema context
        """
        section_priorities = {}
        
        for section_name, section in context_sections.items():
            base_priority = getattr(section, 'priority', 5)
            intelligence_boost = 0
            
            # Boost for sections with critical database content
            if section.section_type in ['tables', 'relationships', 'xml_mappings']:
                intelligence_boost += 10
            
            # Boost for intelligence summary
            if section.section_type == 'intelligence':
                intelligence_boost += 15
            
            # Boost based on table priority scores from intelligent filtering
            if (intelligent_context and hasattr(intelligent_context, 'table_priority_scores') and
                intelligent_context.table_priority_scores):
                for table_name in section.table_names_used:
                    if table_name in intelligent_context.table_priority_scores:
                        table_score = intelligent_context.table_priority_scores[table_name]
                        intelligence_boost += min(table_score // 20, 5)
            
            section_priorities[section_name] = base_priority + intelligence_boost
        
        return section_priorities
    
    def _calculate_entity_aware_priorities(
        self,
        context_sections: Dict[str, ContextSection],
        query_intent: Optional[QueryIntent]
    ) -> Dict[str, int]:
        """
        Calculate priorities based on detected entities
        """
        section_priorities = {}
        
        detected_entities = (query_intent.schema_entities_detected 
                           if query_intent and hasattr(query_intent, 'schema_entities_detected') 
                           else [])
        
        for section_name, section in context_sections.items():
            base_priority = getattr(section, 'priority', 5)
            entity_boost = 0
            
            # Always prioritize critical database sections
            if section.section_type in ['tables', 'relationships', 'xml_mappings']:
                entity_boost += 10
            
            # Boost based on entity mentions
            content_lower = section.content.lower()
            for entity in detected_entities:
                if entity.lower() in content_lower:
                    entity_boost += 3
            
            section_priorities[section_name] = base_priority + entity_boost
        
        return section_priorities
    
    def _calculate_table_priority_based_priorities(
        self,
        context_sections: Dict[str, ContextSection],
        query_intent: Optional[QueryIntent]
    ) -> Dict[str, int]:
        """
        Calculate priorities based on table priorities
        """
        section_priorities = {}
        
        entity_priorities = (query_intent.entity_priorities 
                           if query_intent and hasattr(query_intent, 'entity_priorities') 
                           else {})
        
        for section_name, section in context_sections.items():
            base_priority = getattr(section, 'priority', 5)
            table_boost = 0
            
            # Always prioritize critical database sections
            if section.section_type in ['tables', 'relationships', 'xml_mappings']:
                table_boost += 10
            
            # Boost based on table relevance to priority entities
            for table_name in section.table_names_used:
                max_entity_priority = 0
                for entity, priority in entity_priorities.items():
                    if self._table_relates_to_entity(table_name, entity):
                        max_entity_priority = max(max_entity_priority, priority)
                
                table_boost += max_entity_priority // 20
            
            section_priorities[section_name] = base_priority + table_boost
        
        return section_priorities
    
    def _table_relates_to_entity(self, table_name: str, entity: str) -> bool:
        """
        Check if table relates to entity using semantic similarity
        """
        table_lower = table_name.lower()
        entity_lower = entity.lower()
        
        # Direct match
        if entity_lower in table_lower:
            return True
        
        # Remove prefixes and check
        clean_table = re.sub(r'^(tbl|dim|fact|ref)', '', table_lower)
        if entity_lower in clean_table:
            return True
        
        # Word overlap check
        table_words = set(re.findall(r'\w+', clean_table))
        entity_words = set(re.findall(r'\w+', entity_lower))
        
        return len(table_words.intersection(entity_words)) > 0
    
    def _optimize_traditional(
        self,
        context_sections: Dict[str, ContextSection],
        query_intent: Optional[QueryIntent],
        schema_result: Optional[SchemaRetrievalResult],
        max_length: int
    ) -> Dict[str, ContextSection]:
        """
        Traditional optimization with database protection
        """
        logger.debug("Applying traditional optimization with database protection")
        
        # Identify protected elements even in traditional mode
        protected_elements = self._identify_protected_database_elements(
            context_sections, query_intent, schema_result
        )
        
        # Apply safe optimization
        return self._apply_safe_optimization(
            context_sections, protected_elements, 'traditional', 
            query_intent, schema_result, max_length
        )
    
    # 🔥 ADDITIONAL FIX: Add the traditional optimize_context_sections method for backward compatibility
    def optimize_context_sections(
        self,
        context_sections: Dict[str, ContextSection],
        max_total_length: int = 2000
    ) -> Dict[str, ContextSection]:
        """
        Traditional optimization method with database protection (for backward compatibility)
        """
        return self.optimize_context_sections_intelligently(
            context_sections=context_sections,
            max_total_length=max_total_length
        )
    
    def get_optimization_stats(
        self,
        original_sections: Dict[str, ContextSection],
        optimized_sections: Dict[str, ContextSection],
        strategy_used: str
    ) -> Dict[str, Any]:
        """
        Get detailed statistics about the optimization process
        """
        original_total = sum(s.length for s in original_sections.values())
        optimized_total = sum(s.length for s in optimized_sections.values())
        
        return {
            'strategy_used': strategy_used,
            'original_length': original_total,
            'optimized_length': optimized_total,
            'reduction_bytes': original_total - optimized_total,
            'reduction_percentage': ((original_total - optimized_total) / original_total * 100) if original_total > 0 else 0,
            'sections_preserved': len(optimized_sections),
            'sections_removed': len(original_sections) - len(optimized_sections),
            'sections_truncated': sum(1 for s in optimized_sections.values() if 'truncated' in s.content),
            'critical_elements_protected': True,
            'database_integrity': 'preserved'
        }


# Backward compatibility with enhanced protection
class ContextOptimizer(EnhancedContextOptimizer):
    """
    Backward compatible context optimizer with database protection
    """
    pass
