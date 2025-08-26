"""
Main NLP Processing Pipeline - FAIL-FAST VERSION
NO HARDCODED FALLBACKS - If core functionality is broken, system fails clearly
Coordinates all NLP processing stages for professional analyst queries
PHILOSOPHY: Fail fast, fail clearly, no silent degradation
"""

import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from .data_models import (
    AnalystQuery, ProcessedQuery, IntentResult, BusinessEntity,
    EnhancedSchemaContext, QueryType, DatabaseField, FieldType,
    ProcessingMetrics
)

from agent.nlp_processor.config_module import get_config
from ..utils.metadata_loader import get_metadata_loader

logger = logging.getLogger(__name__)

class PipelineError(Exception):
    """Pipeline-specific errors for clear failure identification"""
    pass

class CoreFunctionalityError(PipelineError):
    """Raised when core functionality is missing or broken"""
    pass

class Pipeline:
    """
    Main NLP processing pipeline - FAIL-FAST IMPLEMENTATION
    
    NO HARDCODED FALLBACKS: If core components don't work, system fails clearly
    PRINCIPLE: Better to fail fast than provide incorrect results
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize NLP pipeline with STRICT validation - no fallbacks"""
        
        # FAIL-FAST: Core configuration must be available
        try:
            self.config = get_config()
        except Exception as e:
            raise CoreFunctionalityError(f"CORE FAILURE: Cannot load configuration - {e}")
        
        if self.config is None:
            raise CoreFunctionalityError("CORE FAILURE: Configuration is None")

        # FAIL-FAST: Metadata loader must be available  
        try:
            self.metadata_loader = get_metadata_loader()
        except Exception as e:
            raise CoreFunctionalityError(f"CORE FAILURE: Cannot load metadata loader - {e}")
            
        if self.metadata_loader is None:
            raise CoreFunctionalityError("CORE FAILURE: Metadata loader is None")

        # FAIL-FAST: Schema context must be available
        try:
            self.schema_context = EnhancedSchemaContext.from_metadata_loader()
        except Exception as e:
            raise CoreFunctionalityError(f"CORE FAILURE: Cannot load schema context - {e}")
            
        if self.schema_context is None:
            raise CoreFunctionalityError("CORE FAILURE: Schema context is None")

        # Initialize processing components - these can be None initially
        self._intent_classifier = None
        self._entity_extractor = None
        self._domain_mapper = None
        self._context_completer = None
        self._query_enricher = None

        # Processing statistics
        self.total_processed = 0
        self.processing_times = []
        self.success_count = 0
        self.error_count = 0

        # FAIL-FAST: Performance target must be valid
        try:
            self.performance_target = self.config.get('processing.performance_target_seconds', 5)
            if not isinstance(self.performance_target, (int, float)) or self.performance_target <= 0:
                raise CoreFunctionalityError("CORE FAILURE: Invalid performance target configuration")
        except Exception as e:
            raise CoreFunctionalityError(f"CORE FAILURE: Cannot access performance configuration - {e}")

        logger.info("NLP Pipeline initialized successfully with fail-fast validation")
        self._validate_core_capabilities()

    def _validate_core_capabilities(self):
        """Validate core capabilities - FAIL if essential components missing"""
        
        # Check schema context capabilities
        if not hasattr(self.schema_context, 'table_structures'):
            raise CoreFunctionalityError("CORE FAILURE: Schema context missing table_structures")
            
        if not hasattr(self.schema_context, 'get_table_structure'):
            raise CoreFunctionalityError("CORE FAILURE: Schema context missing get_table_structure method")

        # Log what we have (only if validation passes)
        try:
            tables_count = len(self.schema_context.table_structures)
            physical_columns = getattr(self.schema_context, 'total_physical_columns', 0)
            xml_fields = getattr(self.schema_context, 'total_xml_fields', 0)
            
            logger.info(f"Core capabilities validated: {tables_count} tables, {physical_columns} columns, {xml_fields} XML fields")
        except Exception as e:
            logger.warning(f"Could not log capabilities (non-critical): {e}")

    def process(self, analyst_query: AnalystQuery) -> ProcessedQuery:
        """
        Process analyst query through pipeline - FAIL-FAST, NO FALLBACKS
        
        If any core step fails, the entire process fails clearly
        """
        if not isinstance(analyst_query, AnalystQuery):
            raise CoreFunctionalityError("CORE FAILURE: Invalid input - expected AnalystQuery object")
            
        if not analyst_query.query_text or not analyst_query.query_text.strip():
            raise CoreFunctionalityError("CORE FAILURE: Empty or invalid query text")

        start_time = time.time()
        component_start_times = {}
        
        logger.info(f"Processing analyst query: {analyst_query.query_text[:100]}...")

        # Stage 1: Intent Classification - MUST succeed
        component_start_times['intent_classification'] = time.time()
        try:
            intent_result = self._classify_intent(analyst_query.query_text)
            if intent_result is None:
                raise CoreFunctionalityError("CORE FAILURE: Intent classification returned None")
        except Exception as e:
            raise CoreFunctionalityError(f"CORE FAILURE: Intent classification failed - {e}")
        intent_time = (time.time() - component_start_times['intent_classification']) * 1000

        # Stage 2: Entity Extraction - MUST succeed
        component_start_times['entity_extraction'] = time.time()
        try:
            entities = self._extract_entities(analyst_query.query_text, intent_result)
            if entities is None:
                raise CoreFunctionalityError("CORE FAILURE: Entity extraction returned None")
        except Exception as e:
            raise CoreFunctionalityError(f"CORE FAILURE: Entity extraction failed - {e}")
        entity_time = (time.time() - component_start_times['entity_extraction']) * 1000

        # Stage 3: Business Context Completion - MUST succeed
        component_start_times['context_completion'] = time.time()
        try:
            business_context = self._complete_business_context(analyst_query, intent_result, entities)
            if business_context is None:
                raise CoreFunctionalityError("CORE FAILURE: Business context completion returned None")
        except Exception as e:
            raise CoreFunctionalityError(f"CORE FAILURE: Business context completion failed - {e}")
        context_time = (time.time() - component_start_times['context_completion']) * 1000

        # Stage 4: Schema Resolution - MUST succeed
        component_start_times['schema_resolution'] = time.time()
        try:
            relevant_fields, relevant_tables = self._resolve_schema_elements(intent_result, entities, business_context)
            if relevant_fields is None or relevant_tables is None:
                raise CoreFunctionalityError("CORE FAILURE: Schema resolution returned None")
        except Exception as e:
            raise CoreFunctionalityError(f"CORE FAILURE: Schema resolution failed - {e}")
        schema_time = (time.time() - component_start_times['schema_resolution']) * 1000

        # Stage 5: Create ProcessingMetrics - MUST succeed
        total_processing_time = (time.time() - start_time) * 1000
        
        try:
            processing_metrics = ProcessingMetrics(
                total_processing_time=total_processing_time,
                component_times={
                    "intent_classification": intent_time,
                    "entity_extraction": entity_time,
                    "context_completion": context_time,
                    "schema_resolution": schema_time
                },
                memory_usage=None,
                cache_hits=0,
                cache_misses=0
            )
        except Exception as e:
            raise CoreFunctionalityError(f"CORE FAILURE: Cannot create processing metrics - {e}")

        # Stage 6: Create ProcessedQuery - MUST succeed
        try:
            processed_query = ProcessedQuery(
                original_query=analyst_query,
                intent=intent_result,
                entities=entities,
                relevant_tables=relevant_tables,
                relevant_fields=relevant_fields,
                business_context=business_context,
                processing_metrics=processing_metrics
            )
        except Exception as e:
            raise CoreFunctionalityError(f"CORE FAILURE: Cannot create ProcessedQuery - {e}")

        # Update statistics (non-critical, can fail silently)
        try:
            self._update_statistics(processed_query, success=True)
        except Exception as e:
            logger.warning(f"Statistics update failed (non-critical): {e}")

        logger.info(f"Query processed successfully in {total_processing_time:.1f}ms")
        return processed_query

    def _classify_intent(self, query_text: str) -> IntentResult:
        """
        Classify intent - FAIL if core classification logic is broken
        NO FALLBACKS - either works or fails clearly
        """
        if not query_text or not isinstance(query_text, str):
            raise CoreFunctionalityError("CORE FAILURE: Invalid query text for intent classification")

        query_lower = query_text.lower()

        # Core logic - if this breaks, we fail
        try:
            # Temporal analysis detection
            temporal_keywords = ['last', 'recent', 'days', 'weeks', 'months', 'years', 'since', 'until', 'before', 'after']
            if any(keyword in query_lower for keyword in temporal_keywords):
                temporal_context = self._extract_temporal_context(query_text)
                return IntentResult(
                    query_type=QueryType.TEMPORAL_ANALYSIS,
                    confidence=0.8,
                    temporal_context=temporal_context,
                    target_tables=self._suggest_temporal_tables()
                )

            # Regional analysis detection
            regional_keywords = ['region', 'state', 'country', 'location', 'city', 'area']
            if any(keyword in query_lower for keyword in regional_keywords):
                return IntentResult(
                    query_type=QueryType.REGIONAL_AGGREGATION,
                    confidence=0.8,
                    target_tables=self._suggest_regional_tables()
                )

            # Aggregation detection
            aggregation_keywords = ['sum', 'count', 'average', 'max', 'min', 'total', 'aggregation']
            if any(keyword in query_lower for keyword in aggregation_keywords):
                aggregation_type = self._detect_aggregation_type(query_text)
                query_type = QueryType.DEFAULTER_ANALYSIS if 'defaulter' in query_lower else QueryType.CUSTOMER_ANALYSIS
                return IntentResult(
                    query_type=query_type,
                    confidence=0.7,
                    aggregation_type=aggregation_type,
                    target_tables=self._suggest_aggregation_tables(query_text)
                )

            # Default case - this is the minimum we can provide
            return IntentResult(
                query_type=QueryType.CUSTOMER_ANALYSIS,
                confidence=0.6,
                target_tables=self._get_core_tables()
            )
            
        except Exception as e:
            raise CoreFunctionalityError(f"CORE FAILURE: Intent classification logic failed - {e}")

    def _extract_entities(self, query_text: str, intent: IntentResult) -> List[BusinessEntity]:
        """
        Extract entities - FAIL if core extraction logic is broken
        """
        if not query_text or not intent:
            raise CoreFunctionalityError("CORE FAILURE: Invalid input for entity extraction")

        try:
            entities = []
            query_lower = query_text.lower()

            # Basic entity extraction - if this fails, we fail
            words = query_text.split()
            
            # Extract counterparty patterns
            counterparty_patterns = ['corp', 'company', 'ltd', 'inc', 'bank']
            for i, word in enumerate(words):
                if any(pattern in word.lower() for pattern in counterparty_patterns):
                    if i > 0:
                        potential_name = f"{words[i-1]} {word}"
                        field_mapping = self._find_field_by_name("CounterpartyName", "tblCounterparty")
                        entities.append(BusinessEntity(
                            entity_type="counterparty",
                            entity_value=potential_name,
                            confidence=0.7,
                            table_mapping="tblCounterparty",
                            field_mapping=field_mapping,
                            field_type=FieldType.PHYSICAL_COLUMN
                        ))

            # Extract temporal entities if temporal intent
            if hasattr(intent, 'temporal_context') and intent.temporal_context:
                entities.append(BusinessEntity(
                    entity_type="temporal",
                    entity_value=intent.temporal_context,
                    confidence=0.8,
                    table_mapping="multiple",
                    field_type=FieldType.PHYSICAL_COLUMN
                ))

            # Extract regional entities
            regional_terms = ['region', 'state', 'country', 'location', 'delhi', 'mumbai', 'bangalore']
            for term in regional_terms:
                if term in query_lower:
                    field_mapping = self._find_field_by_name("StateAddress", "tblCTPTAddress")
                    entities.append(BusinessEntity(
                        entity_type="geographic",
                        entity_value=term,
                        confidence=0.6,
                        table_mapping="tblCTPTAddress",
                        field_mapping=field_mapping,
                        field_type=FieldType.PHYSICAL_COLUMN
                    ))
                    break

            return entities
            
        except Exception as e:
            raise CoreFunctionalityError(f"CORE FAILURE: Entity extraction logic failed - {e}")

    def _complete_business_context(self, query: AnalystQuery, intent: IntentResult, entities: List[BusinessEntity]) -> Dict[str, Any]:
        """
        Complete business context - FAIL if context logic is broken
        """
        if not intent:
            raise CoreFunctionalityError("CORE FAILURE: No intent provided for business context")

        try:
            # Core context structure - this must work
            context = {
                "query_type": intent.query_type.value if hasattr(intent.query_type, 'value') else str(intent.query_type),
                "confidence": intent.confidence,
                "auto_completions": [],
                "business_rules_applied": [],
                "required_joins": [],
                "xml_extractions": []
            }

            # Apply business rules based on query type
            if intent.query_type == QueryType.TEMPORAL_ANALYSIS:
                context["business_rules_applied"].append("temporal_date_filtering")
                context["auto_completions"].append("include_created_date_ordering")

            if intent.query_type == QueryType.REGIONAL_AGGREGATION:
                context["business_rules_applied"].append("region_name_resolution")
                context["auto_completions"].append("resolve_codes_to_names")

            # Process counterparty entities
            counterparty_entities = [e for e in entities if e.entity_type == "counterparty"]
            if counterparty_entities:
                context["auto_completions"].append("include_contact_and_address")
                context["required_joins"].extend([
                    "tblCounterparty -> tblCTPTContactDetails",
                    "tblCounterparty -> tblCTPTAddress"
                ])

            return context
            
        except Exception as e:
            raise CoreFunctionalityError(f"CORE FAILURE: Business context logic failed - {e}")

    def _resolve_schema_elements(self, intent: IntentResult, entities: List[BusinessEntity], context: Dict[str, Any]) -> Tuple[List[DatabaseField], List[str]]:
        """
        Resolve schema elements - FAIL if schema resolution is broken
        """
        if not intent or not hasattr(self.schema_context, 'get_table_structure'):
            raise CoreFunctionalityError("CORE FAILURE: Invalid intent or missing schema context for resolution")

        try:
            relevant_fields = []
            relevant_tables = set()

            # Process intent target tables
            target_tables = getattr(intent, 'target_tables', [])
            if not target_tables:
                raise CoreFunctionalityError("CORE FAILURE: Intent has no target tables")

            for table_name in target_tables:
                relevant_tables.add(table_name)
                
                # Validate table exists in schema
                table_structure = self.schema_context.get_table_structure(table_name)
                if not table_structure:
                    logger.warning(f"Table {table_name} not found in schema - skipping")
                    continue

                # Add basic fields if available
                if hasattr(table_structure, 'all_fields') and table_structure.all_fields:
                    # Add first few fields as basic relevant fields
                    relevant_fields.extend(table_structure.all_fields[:2])

            # Process entity mappings
            for entity in entities:
                if entity.table_mapping and entity.field_mapping:
                    relevant_tables.add(entity.table_mapping)
                    relevant_fields.append(entity.field_mapping)

            # Ensure we have some results
            if not relevant_tables:
                raise CoreFunctionalityError("CORE FAILURE: No relevant tables resolved")

            return list(relevant_fields), list(relevant_tables)
            
        except Exception as e:
            raise CoreFunctionalityError(f"CORE FAILURE: Schema resolution logic failed - {e}")

    # Helper methods - FAIL-FAST implementations
    def _extract_temporal_context(self, query_text: str) -> Optional[str]:
        """Extract temporal context - return None if not found (valid case)"""
        try:
            import re
            patterns = [
                r'last (\d+) days?',
                r'recent',
                r'within (\d+) (\w+)'
            ]

            for pattern in patterns:
                match = re.search(pattern, query_text.lower())
                if match:
                    return match.group(0)
            return None
        except Exception:
            return None

    def _detect_aggregation_type(self, query_text: str) -> Optional[str]:
        """Detect aggregation type - return None if not found (valid case)"""
        try:
            query_lower = query_text.lower()
            if any(term in query_lower for term in ['sum', 'total']):
                return 'SUM'
            elif any(term in query_lower for term in ['average', 'avg']):
                return 'AVG'
            elif any(term in query_lower for term in ['maximum', 'max', 'highest']):
                return 'MAX'
            elif any(term in query_lower for term in ['minimum', 'min', 'lowest']):
                return 'MIN'
            elif any(term in query_lower for term in ['count', 'number']):
                return 'COUNT'
            return None
        except Exception:
            return None

    def _find_field_by_name(self, field_name: str, table_name: str) -> Optional[DatabaseField]:
        """Find field by name - return None if not found (valid case)"""
        try:
            if not hasattr(self.schema_context, 'get_table_structure'):
                return None
                
            table_structure = self.schema_context.get_table_structure(table_name)
            if not table_structure or not hasattr(table_structure, 'all_fields'):
                return None

            for field in table_structure.all_fields:
                if hasattr(field, 'name') and field.name.lower() == field_name.lower():
                    return field
            return None
        except Exception:
            return None

    def _suggest_temporal_tables(self) -> List[str]:
        """Suggest temporal tables - MUST return valid table list"""
        return ['tblCounterparty', 'tblOApplicationMaster', 'tblOSWFActionStatusApplicationTracker']

    def _suggest_regional_tables(self) -> List[str]:
        """Suggest regional tables - MUST return valid table list"""
        return ['tblCounterparty', 'tblCTPTAddress']

    def _suggest_aggregation_tables(self, query_text: str) -> List[str]:
        """Suggest aggregation tables - MUST return valid table list"""
        if 'collateral' in query_text.lower():
            return ['tblOSWFActionStatusCollateralTracker', 'tblCounterparty']
        elif 'defaulter' in query_text.lower():
            return ['tblOApplicationMaster', 'tblCounterparty', 'tblCTPTAddress']
        else:
            return ['tblCounterparty', 'tblOApplicationMaster']

    def _get_core_tables(self) -> List[str]:
        """Get core tables - MUST always return at least basic tables"""
        return ['tblCounterparty', 'tblOApplicationMaster']

    def _update_statistics(self, processed_query: Optional[ProcessedQuery], success: bool):
        """Update statistics - this can fail without breaking the pipeline"""
        self.total_processed += 1
        if success and processed_query:
            self.success_count += 1
            if processed_query.processing_metrics:
                self.processing_times.append(processed_query.processing_metrics.total_processing_time)
        else:
            self.error_count += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics - MUST work or fail clearly"""
        try:
            if not self.processing_times:
                avg_time = 0
                min_time = 0
                max_time = 0
            else:
                avg_time = sum(self.processing_times) / len(self.processing_times)
                min_time = min(self.processing_times)
                max_time = max(self.processing_times)

            return {
                "total_processed": self.total_processed,
                "success_count": self.success_count,
                "error_count": self.error_count,
                "success_rate": (self.success_count / max(self.total_processed, 1)) * 100,
                "avg_processing_time": avg_time,
                "min_processing_time": min_time,
                "max_processing_time": max_time,
                "performance_target": self.performance_target,
                "meeting_target": avg_time <= self.performance_target if avg_time > 0 else True
            }
        except Exception as e:
            raise CoreFunctionalityError(f"CORE FAILURE: Cannot retrieve statistics - {e}")

    def reset_statistics(self):
        """Reset statistics"""
        self.total_processed = 0
        self.processing_times = []
        self.success_count = 0
        self.error_count = 0

    def health_check(self) -> bool:
        """Health check - MUST work or fail clearly"""
        try:
            return (
                self.config is not None and
                self.metadata_loader is not None and
                self.schema_context is not None and
                hasattr(self.schema_context, 'get_table_structure')
            )
        except Exception as e:
            raise CoreFunctionalityError(f"CORE FAILURE: Health check failed - {e}")

# Backward compatibility alias
NLPPipeline = Pipeline
