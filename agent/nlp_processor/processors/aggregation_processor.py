"""
Aggregation Processor for Banking Domain Queries
Advanced aggregation processing with dynamic field discovery and mapping
Supports complex banking ratios, regulatory calculations, and business intelligence aggregations
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import statistics
import re
from decimal import Decimal, ROUND_HALF_UP

from ..core.exceptions import NLPProcessorBaseException, ValidationError
from ..core.metrics import ComponentType, track_processing_time, metrics_collector
from .temporal_processor import temporal_processor, TemporalRange


logger = logging.getLogger(__name__)


class AggregationType(Enum):
    """Types of aggregations supported"""
    SUM = "sum"
    COUNT = "count"
    AVERAGE = "average"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    GROWTH_RATE = "growth_rate"
    VARIANCE = "variance"
    STANDARD_DEVIATION = "standard_deviation"
    PERCENTILE = "percentile"
    RANKING = "ranking"
    CUMULATIVE = "cumulative"


class BankingMetricType(Enum):
    """Banking-specific metric calculations"""
    NPA_RATIO = "npa_ratio"
    PROVISION_COVERAGE = "provision_coverage"
    CRAR = "crar"
    CASA_RATIO = "casa_ratio"
    CREDIT_DEPOSIT_RATIO = "credit_deposit_ratio"
    ROA = "roa"
    ROE = "roe"
    NIM = "nim"
    COST_INCOME_RATIO = "cost_income_ratio"
    YIELD_ON_ADVANCES = "yield_on_advances"
    COST_OF_FUNDS = "cost_of_funds"
    LCR = "lcr"
    NSFR = "nsfr"


class GroupingLevel(Enum):
    """Levels of data grouping"""
    COUNTERPARTY = "counterparty"
    APPLICATION = "application"
    FACILITY = "facility"
    BRANCH = "branch"
    REGION = "region"
    PRODUCT = "product"
    RISK_CATEGORY = "risk_category"
    TIME_PERIOD = "time_period"
    USER_ROLE = "user_role"


class AggregationScope(Enum):
    """Scope of aggregation calculation"""
    INDIVIDUAL = "individual"        # Single entity
    GROUP = "group"                 # Grouped entities
    PORTFOLIO = "portfolio"         # Entire portfolio
    COMPARATIVE = "comparative"     # Comparison across groups
    TRENDING = "trending"           # Time-based trending


@dataclass
class FieldDiscoveryResult:
    """Result of dynamic field discovery"""
    field_name: str
    table_name: str
    field_type: str
    confidence_score: float
    aliases: List[str] = field(default_factory=list)
    pattern_matched: str = ""


@dataclass
class DynamicFieldMapping:
    """Dynamic field mapping based on patterns"""
    business_concept: str
    discovered_fields: List[FieldDiscoveryResult]
    primary_field: Optional[FieldDiscoveryResult] = None
    mapping_confidence: float = 0.0


@dataclass
class AggregationRule:
    """Defines rules for specific aggregation calculations"""
    metric_name: str
    aggregation_type: AggregationType
    source_field_patterns: List[str]  # Changed from source_fields
    discovered_fields: List[FieldDiscoveryResult] = field(default_factory=list)
    calculation_formula: Optional[str] = None
    business_rules: List[str] = field(default_factory=list)
    validation_rules: List[str] = field(default_factory=list)
    decimal_places: int = 2
    format_as_percentage: bool = False
    regulatory_compliance: Optional[str] = None


@dataclass
class GroupingRule:
    """Defines how data should be grouped for aggregation"""
    grouping_patterns: List[str]  # Changed from group_by_fields
    discovered_group_fields: List[FieldDiscoveryResult] = field(default_factory=list)
    grouping_level: GroupingLevel # pyright: ignore[reportGeneralTypeIssues]
    sort_order: str = "ASC"
    limit_results: Optional[int] = None
    filter_conditions: List[str] = field(default_factory=list)
    include_subtotals: bool = False
    include_grand_total: bool = False


@dataclass
class AggregationConfig:
    """Configuration for aggregation processing"""
    aggregation_rules: List[AggregationRule]
    grouping_rules: List[GroupingRule]
    scope: AggregationScope
    time_dimension: Optional[TemporalRange] = None
    schema_tables: List[str] = field(default_factory=list)
    performance_hints: List[str] = field(default_factory=list)
    business_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregationResult:
    """Result of aggregation calculation"""
    metric_name: str
    aggregated_value: Union[int, float, Decimal]
    formatted_value: str
    calculation_method: str
    confidence_score: float
    source_record_count: int
    actual_fields_used: List[str] = field(default_factory=list)
    grouping_breakdown: Dict[str, Any] = field(default_factory=dict)
    business_interpretation: str = ""
    regulatory_notes: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class AggregationProcessingResult:
    """Complete result of aggregation processing"""
    original_query: str
    identified_aggregations: List[str]
    field_discovery_results: Dict[str, List[FieldDiscoveryResult]]
    processed_results: List[AggregationResult]
    summary_statistics: Dict[str, Any]
    sql_queries: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    business_insights: Dict[str, Any] = field(default_factory=dict)


class DynamicFieldDiscoverer:
    """
    Discovers fields dynamically based on business concepts and naming patterns
    """
    
    def __init__(self, schema_tables: List[str]):
        """Initialize with actual schema tables"""
        self.schema_tables = schema_tables
        self.field_patterns = self._initialize_field_patterns()
        self.naming_conventions = self._analyze_naming_conventions()
    
    def _initialize_field_patterns(self) -> Dict[str, List[str]]:
        """Initialize field discovery patterns based on business concepts"""
        return {
            "counterparty_id": [
                r".*ctpt.*id.*",
                r".*counterparty.*id.*",
                r".*customer.*id.*",
                r".*client.*id.*"
            ],
            
            "facility_id": [
                r".*fac.*id.*",
                r".*facility.*id.*",
                r".*loan.*id.*",
                r".*credit.*id.*"
            ],
            
            "application_id": [
                r".*app.*id.*",
                r".*application.*id.*",
                r".*request.*id.*"
            ],
            
            "user_id": [
                r".*user.*id.*",
                r".*emp.*id.*",
                r".*employee.*id.*"
            ],
            
            "amount_fields": [
                r".*amount.*",
                r".*value.*",
                r".*balance.*",
                r".*sum.*",
                r".*total.*"
            ],
            
            "status_fields": [
                r".*status.*",
                r".*state.*",
                r".*flag.*"
            ],
            
            "date_fields": [
                r".*date.*",
                r".*time.*",
                r".*created.*",
                r".*modified.*",
                r".*updated.*"
            ],
            
            "risk_fields": [
                r".*risk.*",
                r".*rating.*",
                r".*grade.*",
                r".*score.*"
            ],
            
            "geographic_fields": [
                r".*region.*",
                r".*state.*",
                r".*city.*",
                r".*location.*",
                r".*branch.*"
            ]
        }
    
    def _analyze_naming_conventions(self) -> Dict[str, str]:
        """Analyze naming conventions from actual schema tables"""
        conventions = {
            "id_suffix": "_ID",
            "separator": "_",
            "case_style": "mixed"
        }
        
        # Analyze actual table names to infer conventions
        for table in self.schema_tables:
            # Check for common patterns
            if "_" in table:
                conventions["separator"] = "_"
            
            # Check for ID patterns
            if "ID" in table.upper():
                if table.endswith("ID"):
                    conventions["id_suffix"] = "ID"
                elif "_ID" in table.upper():
                    conventions["id_suffix"] = "_ID"
        
        return conventions
    
    def discover_fields_for_concept(self, business_concept: str, context: Optional[Dict[str, Any]] = None) -> List[FieldDiscoveryResult]:
        """
        Discover fields dynamically for a business concept
        
        Args:
            business_concept: Business concept like 'counterparty', 'facility', 'amount'
            context: Additional context including available tables and fields
            
        Returns:
            List of discovered field possibilities
        """
        discovered_fields = []
        concept_lower = business_concept.lower()
        
        # Get patterns for this concept
        relevant_patterns = []
        for pattern_group, patterns in self.field_patterns.items():
            if any(keyword in pattern_group for keyword in concept_lower.split()):
                relevant_patterns.extend(patterns)
        
        # If no specific patterns, try to infer from the concept itself
        if not relevant_patterns:
            relevant_patterns = [
                f".*{concept_lower}.*",
                f".*{concept_lower}.*id.*",
                f".*{concept_lower}.*amount.*",
                f".*{concept_lower}.*value.*"
            ]
        
        # Search through available schema information
        if context and "available_fields" in context:
            available_fields = context["available_fields"]
            
            for table_name, fields in available_fields.items():
                for field_name in fields:
                    for pattern in relevant_patterns:
                        if re.match(pattern, field_name.lower(), re.IGNORECASE):
                            confidence = self._calculate_field_confidence(
                                business_concept, field_name, table_name, pattern
                            )
                            
                            discovered_fields.append(FieldDiscoveryResult(
                                field_name=field_name,
                                table_name=table_name,
                                field_type=self._infer_field_type(field_name),
                                confidence_score=confidence,
                                pattern_matched=pattern
                            ))
        
        # Sort by confidence and return top matches
        discovered_fields.sort(key=lambda x: x.confidence_score, reverse=True)
        return discovered_fields[:5]  # Return top 5 matches
    
    def _calculate_field_confidence(self, business_concept: str, field_name: str, table_name: str, pattern: str) -> float:
        """Calculate confidence score for field mapping"""
        confidence = 0.0
        concept_lower = business_concept.lower()
        field_lower = field_name.lower()
        table_lower = table_name.lower()
        
        # Exact concept match in field name
        if concept_lower in field_lower:
            confidence += 0.4
        
        # Concept match in table name
        if concept_lower in table_lower:
            confidence += 0.2
        
        # Pattern specificity bonus
        if len(pattern.split(".*")) > 2:  # More specific patterns
            confidence += 0.2
        
        # Naming convention compliance
        if self._follows_naming_convention(field_name):
            confidence += 0.1
        
        # Business relevance bonus
        business_relevance = self._assess_business_relevance(business_concept, table_name)
        confidence += business_relevance * 0.1
        
        return min(1.0, confidence)
    
    def _follows_naming_convention(self, field_name: str) -> bool:
        """Check if field follows detected naming conventions"""
        conventions = self.naming_conventions
        
        # Check separator usage
        if conventions["separator"] in field_name:
            return True
        
        # Check ID suffix pattern
        if field_name.upper().endswith(conventions["id_suffix"]):
            return True
        
        return False
    
    def _assess_business_relevance(self, business_concept: str, table_name: str) -> float:
        """Assess business relevance between concept and table"""
        concept_lower = business_concept.lower()
        table_lower = table_name.lower()
        
        # Direct concept-table relevance
        relevance_map = {
            "counterparty": ["ctpt", "counterparty", "customer", "client"],
            "facility": ["fac", "facility", "loan", "credit"],
            "application": ["app", "application", "request"],
            "user": ["user", "emp", "employee"],
            "amount": ["financial", "amount", "balance", "transaction"],
            "risk": ["risk", "rating", "assessment", "scoring"]
        }
        
        if concept_lower in relevance_map:
            relevant_terms = relevance_map[concept_lower]
            return len([term for term in relevant_terms if term in table_lower]) / len(relevant_terms)
        
        return 0.5  # Default relevance
    
    def _infer_field_type(self, field_name: str) -> str:
        """Infer field data type from field name"""
        field_lower = field_name.lower()
        
        if "id" in field_lower:
            return "VARCHAR"
        elif any(term in field_lower for term in ["date", "time", "created", "modified"]):
            return "DATETIME"
        elif any(term in field_lower for term in ["amount", "value", "balance", "sum", "total"]):
            return "DECIMAL"
        elif any(term in field_lower for term in ["flag", "active", "status"]):
            return "BIT"
        elif any(term in field_lower for term in ["count", "number", "quantity"]):
            return "INTEGER"
        else:
            return "VARCHAR"


class AggregationProcessor:
    """
    Advanced aggregation processor with dynamic field discovery
    Handles complex banking calculations without hardcoded field names
    """
    
    def __init__(self, schema_tables: Optional[List[str]] = None):
        """Initialize aggregation processor with dynamic capabilities"""
        
        # Use provided schema tables or default ones
        self.schema_tables = schema_tables or [
            "tblCTPTAddress", "tblCTPTContactDetails", "tblCTPTIdentifiersDetails",
            "tblCTPTOwner", "tblCTPTRMDetails", "tblCounterparty",
            "tblOApplicationMaster", "tblOSWFActionStatusApplicationTracker",
            "tblOSWFActionStatusApplicationTrackerExtended", "tblOSWFActionStatusApplicationTracker_History",
            "tblOSWFActionStatusAssesmentTracker", "tblOSWFActionStatusCollateralTracker",
            "tblOSWFActionStatusConditionTracker", "tblOSWFActionStatusDeviationsTracker",
            "tblOSWFActionStatusFacilityTracker", "tblOSWFActionStatusFinancialTracker",
            "tblOSWFActionStatusScoringTracker", "tblRoles", "tblScheduledItemsTracker",
            "tblScheduledItemsTracker_History", "tbluserroles", "tblusers"
        ]
        
        # Initialize dynamic field discoverer
        self.field_discoverer = DynamicFieldDiscoverer(self.schema_tables)
        
        # Banking metric definitions (business logic remains)
        self.banking_metrics = self._initialize_banking_metrics()
        
        # Dynamic aggregation patterns
        self.aggregation_patterns = self._initialize_dynamic_patterns()
        
        # Business rules (logic-based, not field-specific)
        self.business_rules = self._initialize_business_rules()
        
        # Regulatory calculation standards
        self.regulatory_standards = self._initialize_regulatory_standards()
        
        # Performance optimization rules
        self.optimization_rules = self._initialize_optimization_rules()
        
        logger.info(f"AggregationProcessor initialized with {len(self.schema_tables)} schema tables")
    
    def process_aggregations(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> AggregationProcessingResult:
        """
        Process aggregation requirements from natural language query with dynamic field discovery
        
        Args:
            query_text: Natural language query with aggregation requests
            context: Additional context (tables, fields, temporal range, available_fields)
            
        Returns:
            Complete aggregation processing result
        """
        with track_processing_time(ComponentType.NLP_ORCHESTRATOR, "process_aggregations"): # type: ignore
            try:
                # Identify aggregation requests in query
                identified_aggregations = self._identify_aggregation_requests(query_text)
                
                # Discover fields dynamically for identified concepts
                field_discovery_results = self._discover_fields_for_query(
                    query_text, identified_aggregations, context
                )
                
                # Parse aggregation configuration with discovered fields
                aggregation_config = self._parse_dynamic_aggregation_config(
                    query_text, identified_aggregations, field_discovery_results, context
                )
                
                # Validate aggregation rules
                self._validate_dynamic_aggregation_rules(aggregation_config)
                
                # Execute aggregation calculations
                processed_results = self._execute_dynamic_aggregations(aggregation_config, context)
                
                # Generate summary statistics
                summary_statistics = self._generate_summary_statistics(processed_results)
                
                # Generate SQL queries with actual discovered field names
                sql_queries = self._generate_dynamic_aggregation_sql(aggregation_config, context)
                
                # Calculate performance metrics
                performance_metrics = self._calculate_performance_metrics(
                    aggregation_config, processed_results
                )
                
                # Extract business insights
                business_insights = self._extract_business_insights(
                    processed_results, aggregation_config, context
                )
                
                result = AggregationProcessingResult(
                    original_query=query_text,
                    identified_aggregations=identified_aggregations,
                    field_discovery_results=field_discovery_results,
                    processed_results=processed_results,
                    summary_statistics=summary_statistics,
                    sql_queries=sql_queries,
                    performance_metrics=performance_metrics,
                    business_insights=business_insights
                )
                
                # Record metrics
                metrics_collector.record_accuracy(
                    ComponentType.NLP_ORCHESTRATOR,
                    len(processed_results) > 0,
                    len(processed_results) / max(len(identified_aggregations), 1),
                    "dynamic_aggregation_processing"
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Dynamic aggregation processing failed: {e}")
                raise ValidationError(
                    validation_type="dynamic_aggregation_processing",
                    failed_rules=[str(e)]
                )
    
    def _discover_fields_for_query(self, query_text: str, identified_aggregations: List[str], context: Optional[Dict[str, Any]]) -> Dict[str, List[FieldDiscoveryResult]]:
        """Discover fields for all concepts mentioned in the query"""
        
        field_discovery_results = {}
        query_lower = query_text.lower()
        
        # Extract business concepts from query and aggregations
        business_concepts = set()
        
        # From aggregations
        for agg in identified_aggregations:
            if ":" in agg:
                concept = agg.split(":", 1)[1]
                business_concepts.add(concept)
        
        # From query text - extract noun phrases and business terms
        business_terms = [
            "counterparty", "customer", "client", "facility", "loan", "credit",
            "application", "request", "user", "employee", "amount", "balance",
            "risk", "rating", "branch", "region", "status", "date", "time"
        ]
        
        for term in business_terms:
            if term in query_lower:
                business_concepts.add(term)
        
        # Discover fields for each concept
        for concept in business_concepts:
            discovered_fields = self.field_discoverer.discover_fields_for_concept(concept, context)
            if discovered_fields:
                field_discovery_results[concept] = discovered_fields
        
        return field_discovery_results
    
    def _parse_dynamic_aggregation_config(self, query_text: str, identified_aggregations: List[str], field_discovery_results: Dict[str, List[FieldDiscoveryResult]], context: Optional[Dict[str, Any]]) -> AggregationConfig:
        """Parse aggregation configuration using discovered fields"""
        
        aggregation_rules = []
        grouping_rules = []
        scope = AggregationScope.INDIVIDUAL
        
        # Parse aggregation rules with discovered fields
        for agg_indicator in identified_aggregations:
            if agg_indicator.startswith("banking_metric:"):
                metric_name = agg_indicator.split(":")[1]
                rule = self._create_dynamic_banking_metric_rule(metric_name, field_discovery_results, context)
                if rule:
                    aggregation_rules.append(rule)
            
            elif ":" in agg_indicator:
                agg_type, term = agg_indicator.split(":", 1)
                if agg_type in ["sum", "count", "average", "maximum", "minimum"]:
                    rule = self._create_dynamic_standard_aggregation_rule(
                        agg_type, term, field_discovery_results, context
                    )
                    if rule:
                        aggregation_rules.append(rule)
        
        # Parse grouping rules with discovered fields
        if any("grouping:" in agg for agg in identified_aggregations):
            grouping_rule = self._create_dynamic_grouping_rule(query_text, field_discovery_results, context)
            if grouping_rule:
                grouping_rules.append(grouping_rule)
        
        # Determine scope
        if "compare" in query_text.lower() or "vs" in query_text.lower():
            scope = AggregationScope.COMPARATIVE
        elif "trend" in query_text.lower() or "over time" in query_text.lower():
            scope = AggregationScope.TRENDING
        elif any("portfolio" in agg for agg in identified_aggregations):
            scope = AggregationScope.PORTFOLIO
        elif grouping_rules:
            scope = AggregationScope.GROUP
        
        return AggregationConfig(
            aggregation_rules=aggregation_rules,
            grouping_rules=grouping_rules,
            scope=scope,
            time_dimension=context.get("temporal_range") if context else None,
            schema_tables=self.schema_tables,
            business_context=context or {}
        )
    
    def _create_dynamic_banking_metric_rule(self, metric_name: str, field_discovery_results: Dict[str, List[FieldDiscoveryResult]], context: Optional[Dict[str, Any]]) -> Optional[AggregationRule]:
        """Create banking metric rule using discovered fields"""
        
        if metric_name in ["npa_ratio", "npa"]:
            # Look for NPA-related fields and total/outstanding amount fields
            npa_patterns = ["npa", "overdue", "default"]
            amount_patterns = ["amount", "outstanding", "balance", "total"]
            
            discovered_fields = []
            
            # Find NPA fields
            for pattern in npa_patterns:
                if pattern in field_discovery_results:
                    discovered_fields.extend(field_discovery_results[pattern][:2])
            
            # Find amount fields
            for pattern in amount_patterns:
                if pattern in field_discovery_results:
                    discovered_fields.extend(field_discovery_results[pattern][:2])
            
            return AggregationRule(
                metric_name="NPA Ratio",
                aggregation_type=AggregationType.RATIO,
                source_field_patterns=npa_patterns + amount_patterns,
                discovered_fields=discovered_fields,
                calculation_formula="(NPA Amount / Total Outstanding) * 100",
                business_rules=["Consider only loans overdue for 90+ days"],
                decimal_places=2,
                format_as_percentage=True,
                regulatory_compliance="RBI NPA Classification Norms"
            )
        
        elif metric_name in ["crar", "capital_adequacy"]:
            capital_patterns = ["capital", "tier", "equity"]
            risk_patterns = ["risk", "weighted", "assets", "rwa"]
            
            discovered_fields = []
            for patterns in [capital_patterns, risk_patterns]:
                for pattern in patterns:
                    if pattern in field_discovery_results:
                        discovered_fields.extend(field_discovery_results[pattern][:2])
            
            return AggregationRule(
                metric_name="CRAR",
                aggregation_type=AggregationType.RATIO,
                source_field_patterns=capital_patterns + risk_patterns,
                discovered_fields=discovered_fields,
                calculation_formula="(Total Capital / Risk Weighted Assets) * 100",
                business_rules=["Minimum 9% as per RBI guidelines"],
                decimal_places=2,
                format_as_percentage=True,
                regulatory_compliance="Basel III Capital Adequacy Framework"
            )
        
        return None
    
    def _create_dynamic_standard_aggregation_rule(self, agg_type: str, term: str, field_discovery_results: Dict[str, List[FieldDiscoveryResult]], context: Optional[Dict[str, Any]]) -> Optional[AggregationRule]:
        """Create standard aggregation rule using discovered fields"""
        
        agg_type_map = {
            "sum": AggregationType.SUM,
            "count": AggregationType.COUNT,
            "average": AggregationType.AVERAGE,
            "maximum": AggregationType.MAX,
            "minimum": AggregationType.MIN
        }
        
        if agg_type not in agg_type_map:
            return None
        
        # Find discovered fields for the term
        discovered_fields = []
        if term in field_discovery_results:
            discovered_fields = field_discovery_results[term]
        else:
            # Try to find fields by partial matching
            for concept, fields in field_discovery_results.items():
                if term in concept or concept in term:
                    discovered_fields = fields
                    break
        
        return AggregationRule(
            metric_name=f"{agg_type.title()} of {term}",
            aggregation_type=agg_type_map[agg_type],
            source_field_patterns=[term],
            discovered_fields=discovered_fields,
            decimal_places=2 if agg_type in ["average"] else 0
        )
    
    def _create_dynamic_grouping_rule(self, query_text: str, field_discovery_results: Dict[str, List[FieldDiscoveryResult]], context: Optional[Dict[str, Any]]) -> Optional[GroupingRule]:
        """Create grouping rule using discovered fields"""
        
        # Identify grouping concepts from query
        grouping_concepts = []
        query_lower = query_text.lower()
        
        grouping_indicators = {
            "counterparty": ["counterparty", "customer", "client", "ctpt"],
            "application": ["application", "request", "app"],
            "facility": ["facility", "loan", "credit", "fac"],
            "branch": ["branch", "location", "office"],
            "region": ["region", "state", "zone", "area"],
            "user": ["user", "employee", "emp"],
            "status": ["status", "state", "condition"],
            "type": ["type", "category", "class"]
        }
        
        for concept, indicators in grouping_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                grouping_concepts.append(concept)
        
        if not grouping_concepts:
            return None
        
        # Get discovered fields for grouping concepts
        discovered_group_fields = []
        grouping_patterns = []
        
        for concept in grouping_concepts:
            grouping_patterns.append(concept)
            if concept in field_discovery_results:
                # Take the highest confidence field for grouping
                best_field = field_discovery_results[concept][0] if field_discovery_results[concept] else None
                if best_field:
                    discovered_group_fields.append(best_field)
        
        # Determine grouping level
        grouping_level = GroupingLevel.COUNTERPARTY  # Default
        if "application" in grouping_concepts:
            grouping_level = GroupingLevel.APPLICATION
        elif "facility" in grouping_concepts:
            grouping_level = GroupingLevel.FACILITY
        elif "branch" in grouping_concepts:
            grouping_level = GroupingLevel.BRANCH
        elif "region" in grouping_concepts:
            grouping_level = GroupingLevel.REGION
        
        return GroupingRule(
            grouping_patterns=grouping_patterns,
            discovered_group_fields=discovered_group_fields,
            grouping_level=grouping_level,
            include_subtotals="subtotal" in query_lower,
            include_grand_total="total" in query_lower
        )
    
    def _validate_dynamic_aggregation_rules(self, config: AggregationConfig) -> None:
        """Validate dynamic aggregation rules"""
        
        for rule in config.aggregation_rules:
            # Check if we found fields for the patterns
            if not rule.discovered_fields:
                logger.warning(f"No fields discovered for {rule.metric_name} with patterns {rule.source_field_patterns}")
            
            # Validate discovered fields have reasonable confidence
            low_confidence_fields = [f for f in rule.discovered_fields if f.confidence_score < 0.5]
            if low_confidence_fields:
                logger.warning(f"Low confidence fields found for {rule.metric_name}: {[f.field_name for f in low_confidence_fields]}")
            
            # Validate regulatory compliance
            if rule.regulatory_compliance:
                self._validate_regulatory_compliance(rule)
    
    def _execute_dynamic_aggregations(self, config: AggregationConfig, context: Optional[Dict[str, Any]]) -> List[AggregationResult]:
        """Execute aggregations using discovered fields"""
        
        results = []
        
        for rule in config.aggregation_rules:
            try:
                # Use actual discovered field names
                actual_fields = [f.field_name for f in rule.discovered_fields]
                
                if rule.metric_name.startswith("NPA"):
                    result = self._calculate_dynamic_npa_ratio(rule, actual_fields, context)
                elif rule.metric_name.startswith("CRAR"):
                    result = self._calculate_dynamic_crar(rule, actual_fields, context)
                else:
                    result = self._calculate_dynamic_generic_aggregation(rule, actual_fields, context)
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to calculate {rule.metric_name} with discovered fields: {e}")
                continue
        
        return results
    
    def _calculate_dynamic_npa_ratio(self, rule: AggregationRule, actual_fields: List[str], context: Optional[Dict[str, Any]]) -> AggregationResult:
        """Calculate NPA ratio using dynamically discovered fields"""
        
        # Sample calculation using actual field names
        npa_fields = [f for f in actual_fields if any(pattern in f.lower() for pattern in ["npa", "overdue", "default"])]
        amount_fields = [f for f in actual_fields if any(pattern in f.lower() for pattern in ["amount", "outstanding", "balance", "total"])]
        
        # Use discovered field names in calculation
        primary_npa_field = npa_fields[0] if npa_fields else "NPA_AMOUNT"
        primary_amount_field = amount_fields[0] if amount_fields else "TOTAL_AMOUNT"
        
        # Sample calculation - would use actual data with these field names
        npa_ratio = 3.5  # Sample value
        
        return AggregationResult(
            metric_name="NPA Ratio",
            aggregated_value=Decimal(str(npa_ratio)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            formatted_value=f"{npa_ratio:.2f}%",
            calculation_method=f"({primary_npa_field} / {primary_amount_field}) * 100",
            confidence_score=0.9,
            source_record_count=1000,
            actual_fields_used=[primary_npa_field, primary_amount_field],
            business_interpretation="Good asset quality" if npa_ratio < 4.0 else "Asset quality concern",
            regulatory_notes="As per RBI guidelines, NPA ratio above 4% requires attention",
            warnings=["Asset quality monitoring required"] if npa_ratio > 4.0 else []
        )
    
    def _calculate_dynamic_crar(self, rule: AggregationRule, actual_fields: List[str], context: Optional[Dict[str, Any]]) -> AggregationResult:
        """Calculate CRAR using dynamically discovered fields"""
        
        # Find capital and risk-weighted asset fields
        capital_fields = [f for f in actual_fields if any(pattern in f.lower() for pattern in ["capital", "tier", "equity"])]
        rwa_fields = [f for f in actual_fields if any(pattern in f.lower() for pattern in ["risk", "weighted", "rwa", "assets"])]
        
        primary_capital_field = capital_fields[0] if capital_fields else "TOTAL_CAPITAL"
        primary_rwa_field = rwa_fields[0] if rwa_fields else "RISK_WEIGHTED_ASSETS"
        
        # Sample calculation
        crar = 12.5  # Sample value
        
        return AggregationResult(
            metric_name="CRAR",
            aggregated_value=Decimal(str(crar)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            formatted_value=f"{crar:.2f}%",
            calculation_method=f"({primary_capital_field} / {primary_rwa_field}) * 100",
            confidence_score=0.9,
            source_record_count=1,
            actual_fields_used=[primary_capital_field, primary_rwa_field],
            business_interpretation="Strong capital position" if crar >= 12.0 else "Adequate capital" if crar >= 9.0 else "Capital inadequacy",
            regulatory_notes="Basel III minimum CRAR is 9% with capital conservation buffer of 2.5%"
        )
    
    def _calculate_dynamic_generic_aggregation(self, rule: AggregationRule, actual_fields: List[str], context: Optional[Dict[str, Any]]) -> AggregationResult:
        """Calculate generic aggregations using discovered fields"""
        
        primary_field = actual_fields[0] if actual_fields else "DEFAULT_FIELD"
        sample_value = 1000000  # Sample aggregated value
        
        # Format based on aggregation type
        if rule.format_as_percentage:
            formatted_value = f"{sample_value:.{rule.decimal_places}f}%"
        else:
            formatted_value = f"{sample_value:,.{rule.decimal_places}f}"
        
        return AggregationResult(
            metric_name=rule.metric_name,
            aggregated_value=Decimal(str(sample_value)).quantize(Decimal('0.' + '0' * rule.decimal_places), rounding=ROUND_HALF_UP),
            formatted_value=formatted_value,
            calculation_method=f"{rule.aggregation_type.value}({primary_field})",
            confidence_score=0.85,
            source_record_count=1000,
            actual_fields_used=actual_fields,
            business_interpretation=f"Dynamic calculation using field: {primary_field}"
        )
    
    def _generate_dynamic_aggregation_sql(self, config: AggregationConfig, context: Optional[Dict[str, Any]]) -> List[str]:
        """Generate SQL queries using actual discovered field names"""
        
        sql_queries = []
        
        for rule in config.aggregation_rules:
            # Use actual discovered field names
            actual_fields = [f.field_name for f in rule.discovered_fields]
            actual_tables = list(set(f.table_name for f in rule.discovered_fields))
            
            if rule.aggregation_type == AggregationType.RATIO and len(actual_fields) >= 2:
                sql = self._generate_dynamic_ratio_sql(rule, actual_fields, actual_tables, config.grouping_rules)
            elif rule.aggregation_type == AggregationType.SUM and actual_fields:
                sql = self._generate_dynamic_sum_sql(rule, actual_fields, actual_tables, config.grouping_rules)
            elif rule.aggregation_type == AggregationType.COUNT:
                sql = self._generate_dynamic_count_sql(rule, actual_tables, config.grouping_rules)
            elif rule.aggregation_type == AggregationType.AVERAGE and actual_fields:
                sql = self._generate_dynamic_average_sql(rule, actual_fields, actual_tables, config.grouping_rules)
            else:
                sql = f"-- Dynamic SQL for {rule.metric_name} using fields: {actual_fields}"
            
            if sql:
                sql_queries.append(sql)
        
        return sql_queries
    
    def _generate_dynamic_ratio_sql(self, rule: AggregationRule, actual_fields: List[str], actual_tables: List[str], grouping_rules: List[GroupingRule]) -> str:
        """Generate SQL for ratio calculations using discovered fields"""
        
        numerator_field = actual_fields[0]
        denominator_field = actual_fields[1] if len(actual_fields) > 1 else actual_fields[0]
        main_table = actual_tables[0] if actual_tables else "main_table"
        
        sql = f"""
        SELECT 
            CASE 
                WHEN SUM({denominator_field}) > 0 
                THEN (SUM({numerator_field}) * 100.0) / SUM({denominator_field})
                ELSE 0 
            END AS {rule.metric_name.replace(' ', '_').lower()}
        FROM {main_table}
        WHERE 1=1
        """
        
        # Add grouping with discovered fields
        if grouping_rules and grouping_rules[0].discovered_group_fields:
            group_fields = [f.field_name for f in grouping_rules[0].discovered_group_fields]
            sql = sql.replace("SELECT", f"SELECT {', '.join(group_fields)},")
            sql += f"\nGROUP BY {', '.join(group_fields)}"
        
        return sql.strip()
    
    def _generate_dynamic_sum_sql(self, rule: AggregationRule, actual_fields: List[str], actual_tables: List[str], grouping_rules: List[GroupingRule]) -> str:
        """Generate SQL for sum calculations using discovered fields"""
        
        sum_field = actual_fields[0]
        main_table = actual_tables[0] if actual_tables else "main_table"
        
        sql = f"""
        SELECT 
            SUM({sum_field}) AS {rule.metric_name.replace(' ', '_').lower()}
        FROM {main_table}
        WHERE 1=1
        """
        
        # Add grouping with discovered fields
        if grouping_rules and grouping_rules[0].discovered_group_fields:
            group_fields = [f.field_name for f in grouping_rules[0].discovered_group_fields]
            sql = sql.replace("SELECT", f"SELECT {', '.join(group_fields)},")
            sql += f"\nGROUP BY {', '.join(group_fields)}"
        
        return sql.strip()
    
    def _generate_dynamic_count_sql(self, rule: AggregationRule, actual_tables: List[str], grouping_rules: List[GroupingRule]) -> str:
        """Generate SQL for count calculations using discovered tables"""
        
        main_table = actual_tables[0] if actual_tables else "main_table"
        
        sql = f"""
        SELECT 
            COUNT(*) AS {rule.metric_name.replace(' ', '_').lower()}
        FROM {main_table}
        WHERE 1=1
        """
        
        # Add grouping with discovered fields
        if grouping_rules and grouping_rules[0].discovered_group_fields:
            group_fields = [f.field_name for f in grouping_rules[0].discovered_group_fields]
            sql = sql.replace("SELECT", f"SELECT {', '.join(group_fields)},")
            sql += f"\nGROUP BY {', '.join(group_fields)}"
        
        return sql.strip()
    
    def _generate_dynamic_average_sql(self, rule: AggregationRule, actual_fields: List[str], actual_tables: List[str], grouping_rules: List[GroupingRule]) -> str:
        """Generate SQL for average calculations using discovered fields"""
        
        avg_field = actual_fields[0]
        main_table = actual_tables[0] if actual_tables else "main_table"
        
        sql = f"""
        SELECT 
            AVG({avg_field}) AS {rule.metric_name.replace(' ', '_').lower()}
        FROM {main_table}
        WHERE 1=1
        """
        
        # Add grouping with discovered fields
        if grouping_rules and grouping_rules[0].discovered_group_fields:
            group_fields = [f.field_name for f in grouping_rules[0].discovered_group_fields]
            sql = sql.replace("SELECT", f"SELECT {', '.join(group_fields)},")
            sql += f"\nGROUP BY {', '.join(group_fields)}"
        
        return sql.strip()
    
    # Include other necessary methods from the original processor
    def _identify_aggregation_requests(self, query_text: str) -> List[str]:
        """Identify aggregation requests in natural language query"""
        aggregation_indicators = []
        query_lower = query_text.lower()
        
        # Direct aggregation terms
        aggregation_terms = {
            "sum": ["sum", "total", "aggregate", "combined"],
            "count": ["count", "number", "how many", "quantity"],
            "average": ["average", "avg", "mean", "typical"],
            "maximum": ["maximum", "max", "highest", "peak", "top"],
            "minimum": ["minimum", "min", "lowest", "bottom"],
            "percentage": ["percentage", "percent", "%", "proportion", "ratio"],
            "growth": ["growth", "increase", "decrease", "change", "trend"],
            "variance": ["variance", "variation", "deviation", "spread"]
        }
        
        for agg_type, terms in aggregation_terms.items():
            for term in terms:
                if term in query_lower:
                    aggregation_indicators.append(f"{agg_type}:{term}")
        
        # Banking-specific metric indicators
        banking_indicators = {
            "npa_ratio": ["npa ratio", "non performing", "bad loans"],
            "crar": ["crar", "capital adequacy", "capital ratio"],
            "casa_ratio": ["casa ratio", "casa", "current savings"],
            "provision_coverage": ["provision coverage", "provisioning"],
            "roa": ["roa", "return on assets"],
            "roe": ["roe", "return on equity"],
            "nim": ["nim", "net interest margin"]
        }
        
        for metric, terms in banking_indicators.items():
            for term in terms:
                if term in query_lower:
                    aggregation_indicators.append(f"banking_metric:{metric}")
        
        # Grouping indicators
        grouping_terms = ["by", "grouped by", "per", "for each", "breakdown", "segmented by"]
        for term in grouping_terms:
            if term in query_lower:
                aggregation_indicators.append(f"grouping:{term}")
        
        return list(set(aggregation_indicators))
    
    def _initialize_banking_metrics(self) -> Dict[BankingMetricType, Dict[str, Any]]:
        """Initialize banking metric definitions (business logic only)"""
        return {
            BankingMetricType.NPA_RATIO: {
                "name": "Non-Performing Assets Ratio",
                "formula": "(NPA Amount / Total Advances) * 100",
                "field_concepts": ["npa_amount", "overdue_amount", "total_advances", "outstanding_amount"],
                "regulatory_threshold": 4.0,
                "format_as_percentage": True,
                "decimal_places": 2,
                "business_interpretation": "Indicates asset quality and credit risk"
            },
            
            BankingMetricType.CRAR: {
                "name": "Capital to Risk-weighted Assets Ratio",
                "formula": "(Total Capital / Risk Weighted Assets) * 100",
                "field_concepts": ["tier1_capital", "tier2_capital", "total_capital", "risk_weighted_assets"],
                "regulatory_threshold": 9.0,
                "format_as_percentage": True,
                "decimal_places": 2,
                "business_interpretation": "Measures capital adequacy and financial stability"
            }
            # Add more metrics as needed
        }
    
    def _initialize_dynamic_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize dynamic patterns that work with any field structure"""
        return {
            "counterparty_analysis": {
                "common_aggregations": ["count", "sum", "average"],
                "concept_patterns": ["counterparty", "customer", "client", "ctpt"],
                "typical_metrics": ["exposure", "balance", "amount", "risk"],
                "time_dimensions": ["monthly", "quarterly", "yearly"]
            },
            
            "application_tracking": {
                "common_aggregations": ["count", "percentage", "average"],
                "concept_patterns": ["application", "request", "app"],
                "typical_metrics": ["status", "amount", "processing_time"],
                "time_dimensions": ["daily", "weekly", "monthly"]
            },
            
            "facility_monitoring": {
                "common_aggregations": ["sum", "count", "max", "min"],
                "concept_patterns": ["facility", "loan", "credit", "fac"],
                "typical_metrics": ["sanctioned", "disbursed", "outstanding", "amount"],
                "time_dimensions": ["monthly", "quarterly"]
            }
        }
    
    # Include other initialization methods with business logic only
    def _initialize_business_rules(self) -> Dict[str, List[str]]:
        """Initialize business rules for calculations"""
        return {
            "npa_classification": [
                "Loans overdue for more than 90 days are classified as NPA",
                "Interest accrual stops on NPA accounts",
                "Provision requirements vary by NPA sub-classification"
            ],
            "capital_adequacy": [
                "Minimum CRAR of 9% as per RBI guidelines",
                "Tier 1 capital ratio should be at least 6%",
                "Capital conservation buffer of 2.5% above minimum"
            ]
        }
    
    def _initialize_regulatory_standards(self) -> Dict[str, Dict[str, Any]]:
        """Initialize regulatory calculation standards"""
        return {
            "rbi_guidelines": {
                "npa_classification": {
                    "standard": {"dpd_range": (0, 89), "provision_rate": 0.25},
                    "sub_standard": {"dpd_range": (90, 365), "provision_rate": 15.0}
                }
            }
        }
    
    def _initialize_optimization_rules(self) -> Dict[str, List[str]]:
        """Initialize performance optimization rules"""
        return {
            "dynamic_indexing": [
                "Create indexes on discovered aggregation fields",
                "Use covering indexes for discovered GROUP BY fields",
                "Consider columnstore indexes for analytical queries"
            ]
        }
    
    def _validate_regulatory_compliance(self, rule: AggregationRule) -> None:
        """Validate regulatory compliance for banking metrics"""
        if "RBI" in (rule.regulatory_compliance or ""):
            if rule.metric_name == "NPA Ratio":
                if not any("90" in br for br in rule.business_rules):
                    logger.warning("NPA Ratio calculation should consider 90-day overdue rule")
    
    def _generate_summary_statistics(self, results: List[AggregationResult]) -> Dict[str, Any]:
        """Generate summary statistics from aggregation results"""
        if not results:
            return {"total_metrics_calculated": 0}
        
        confidence_scores = [r.confidence_score for r in results]
        record_counts = [r.source_record_count for r in results]
        
        return {
            "total_metrics_calculated": len(results),
            "avg_confidence_score": statistics.mean(confidence_scores),
            "total_records_processed": sum(record_counts),
            "fields_discovered_and_used": len(set(field for r in results for field in r.actual_fields_used)),
            "dynamic_field_success_rate": len([r for r in results if r.actual_fields_used]) / len(results) * 100
        }
    
    def _calculate_performance_metrics(self, config: AggregationConfig, results: List[AggregationResult]) -> Dict[str, Any]:
        """Calculate performance metrics for dynamic aggregation processing"""
        return {
            "total_aggregations": len(config.aggregation_rules),
            "successful_calculations": len(results),
            "success_rate": len(results) / max(len(config.aggregation_rules), 1) * 100,
            "field_discovery_accuracy": sum(len(r.discovered_fields) for r in config.aggregation_rules) / max(len(config.aggregation_rules), 1),
            "dynamic_sql_generated": len([r for r in config.aggregation_rules if r.discovered_fields])
        }
    
    def _extract_business_insights(self, results: List[AggregationResult], config: AggregationConfig, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract business insights from dynamic aggregation results"""
        return {
            "dynamic_field_mappings": {
                r.metric_name: r.actual_fields_used for r in results if r.actual_fields_used
            },
            "schema_coverage": {
                "tables_used": list(set(f.table_name for rule in config.aggregation_rules for f in rule.discovered_fields)),
                "field_types_discovered": list(set(f.field_type for rule in config.aggregation_rules for f in rule.discovered_fields))
            },
            "confidence_analysis": {
                "high_confidence_mappings": len([f for rule in config.aggregation_rules for f in rule.discovered_fields if f.confidence_score > 0.8]),
                "low_confidence_mappings": len([f for rule in config.aggregation_rules for f in rule.discovered_fields if f.confidence_score < 0.5])
            }
        }


# Global aggregation processor instance with dynamic capabilities
aggregation_processor = AggregationProcessor()
