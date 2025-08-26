"""
Context Completer for Banking Business Intelligence
Auto-completes incomplete analyst queries with intelligent context inference
Provides missing context inference and professional query enhancements
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from datetime import datetime, timedelta

from ..core.exceptions import NLPProcessorBaseException, ValidationError
from ..core.metrics import ComponentType, track_processing_time, metrics_collector
from .domain_mapper import domain_mapper, BusinessDomain, FieldMapping


logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context that can be inferred or added"""
    TEMPORAL = "temporal"           # Time-related context
    GEOGRAPHIC = "geographic"       # Location/region context
    BUSINESS_RULE = "business_rule" # Banking business rules
    METRIC = "metric"              # Missing metrics or calculations
    SCOPE = "scope"                # Analysis scope (tables, filters)
    RELATIONSHIP = "relationship"   # Data relationships and joins


class CompletionConfidence(Enum):
    """Confidence levels for context completion"""
    HIGH = "high"           # Very confident completion
    MEDIUM = "medium"       # Good completion with some assumptions
    LOW = "low"            # Uncertain completion
    SUGGESTED = "suggested" # Suggested completion for user review


@dataclass
class ContextGap:
    """Represents a gap in query context"""
    gap_type: ContextType
    description: str
    priority: int = 1  # 1=high, 2=medium, 3=low
    suggested_completions: List[str] = field(default_factory=list)
    business_impact: str = ""
    examples: List[str] = field(default_factory=list)


@dataclass
class ContextCompletion:
    """A single context completion suggestion"""
    completion_text: str
    context_type: ContextType
    confidence: CompletionConfidence
    reasoning: str
    business_rule: Optional[str] = None
    impact_description: str = ""
    alternative_completions: List[str] = field(default_factory=list)


@dataclass
class CompletionResult:
    """Result of context completion process"""
    original_query: str
    completed_query: str
    identified_gaps: List[ContextGap]
    applied_completions: List[ContextCompletion]
    confidence_score: float
    requires_user_confirmation: bool = False
    enhancement_suggestions: List[str] = field(default_factory=list)
    business_context: Dict[str, Any] = field(default_factory=dict)


class ContextCompleter:
    """
    Intelligent context completion for banking analyst queries
    Identifies missing context and provides smart completions
    """
    
    def __init__(self):
        """Initialize context completer with banking knowledge"""
        
        # Banking business rules for context completion
        self.business_rules = self._initialize_business_rules()
        
        # Temporal context patterns
        self.temporal_patterns = self._initialize_temporal_patterns()
        
        # Geographic context knowledge
        self.geographic_context = self._initialize_geographic_context()
        
        # Professional analyst query patterns
        self.analyst_patterns = self._initialize_analyst_patterns()
        
        # Metric completion rules
        self.metric_rules = self._initialize_metric_rules()
        
        # Scope completion patterns
        self.scope_patterns = self._initialize_scope_patterns()
        
        # Context inference cache
        self.completion_cache: Dict[str, CompletionResult] = {}
        
        logger.info("ContextCompleter initialized with banking business rules")
    
    def complete_query_context(self, query_text: str, user_context: Optional[Dict[str, Any]] = None) -> CompletionResult:
        """
        Complete missing context in analyst query
        
        Args:
            query_text: Original analyst query
            user_context: Additional user context (role, preferences, history)
            
        Returns:
            Completion result with enhanced query
        """
        with track_processing_time(ComponentType.NLP_ORCHESTRATOR, "complete_context"):
            try:
                # Check cache
                cache_key = f"{query_text.lower()}_{hash(str(user_context))}"
                if cache_key in self.completion_cache:
                    return self.completion_cache[cache_key]
                
                # Identify context gaps
                context_gaps = self._identify_context_gaps(query_text, user_context)
                
                # Generate completions for each gap
                completions = []
                enhanced_query = query_text
                
                for gap in sorted(context_gaps, key=lambda x: x.priority):
                    completion = self._generate_completion_for_gap(gap, enhanced_query, user_context)
                    if completion:
                        completions.append(completion)
                        enhanced_query = self._apply_completion(enhanced_query, completion)
                
                # Calculate overall confidence
                confidence_score = self._calculate_completion_confidence(completions)
                
                # Determine if user confirmation is needed
                requires_confirmation = any(
                    c.confidence in [CompletionConfidence.LOW, CompletionConfidence.SUGGESTED] 
                    for c in completions
                )
                
                # Generate enhancement suggestions
                enhancement_suggestions = self._generate_enhancement_suggestions(
                    enhanced_query, context_gaps, user_context
                )
                
                # Extract business context
                business_context = self._extract_business_context(enhanced_query, completions)
                
                result = CompletionResult(
                    original_query=query_text,
                    completed_query=enhanced_query,
                    identified_gaps=context_gaps,
                    applied_completions=completions,
                    confidence_score=confidence_score,
                    requires_user_confirmation=requires_confirmation,
                    enhancement_suggestions=enhancement_suggestions,
                    business_context=business_context
                )
                
                # Cache result
                self.completion_cache[cache_key] = result
                
                # Record metrics
                metrics_collector.record_accuracy(
                    ComponentType.NLP_ORCHESTRATOR,
                    confidence_score > 0.7,
                    confidence_score,
                    "context_completion"
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Context completion failed: {e}")
                # Return minimal result on failure
                return CompletionResult(
                    original_query=query_text,
                    completed_query=query_text,
                    identified_gaps=[],
                    applied_completions=[],
                    confidence_score=0.0,
                    requires_user_confirmation=True
                )
    
    def enhance_professional_query(self, query_text: str, analyst_profile: Optional[Dict[str, Any]] = None) -> str:
        """
        Enhance query with professional banking terminology and best practices
        
        Args:
            query_text: Original query
            analyst_profile: Analyst's role, experience, preferences
            
        Returns:
            Enhanced professional query
        """
        enhanced_query = query_text
        
        # Apply professional terminology
        enhanced_query = self._apply_professional_terminology(enhanced_query)
        
        # Add banking-specific qualifiers
        enhanced_query = self._add_banking_qualifiers(enhanced_query, analyst_profile)
        
        # Apply best practice patterns
        enhanced_query = self._apply_best_practices(enhanced_query, analyst_profile)
        
        return enhanced_query
    
    def suggest_query_improvements(self, query_text: str, performance_context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Suggest improvements for query performance and accuracy
        
        Args:
            query_text: Query to improve
            performance_context: Performance requirements and constraints
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Performance optimization suggestions
        if performance_context:
            suggestions.extend(self._get_performance_suggestions(query_text, performance_context))
        
        # Accuracy improvement suggestions
        suggestions.extend(self._get_accuracy_suggestions(query_text))
        
        # Business intelligence suggestions
        suggestions.extend(self._get_business_intelligence_suggestions(query_text))
        
        return suggestions
    
    def _identify_context_gaps(self, query_text: str, user_context: Optional[Dict[str, Any]]) -> List[ContextGap]:
        """Identify missing context in the query"""
        gaps = []
        query_lower = query_text.lower()
        
        # Temporal context gaps
        temporal_gaps = self._identify_temporal_gaps(query_lower)
        gaps.extend(temporal_gaps)
        
        # Geographic context gaps
        geographic_gaps = self._identify_geographic_gaps(query_lower)
        gaps.extend(geographic_gaps)
        
        # Metric context gaps
        metric_gaps = self._identify_metric_gaps(query_lower)
        gaps.extend(metric_gaps)
        
        # Scope context gaps
        scope_gaps = self._identify_scope_gaps(query_lower, user_context)
        gaps.extend(scope_gaps)
        
        # Business rule gaps
        business_rule_gaps = self._identify_business_rule_gaps(query_lower)
        gaps.extend(business_rule_gaps)
        
        # Relationship gaps
        relationship_gaps = self._identify_relationship_gaps(query_lower)
        gaps.extend(relationship_gaps)
        
        return gaps
    
    def _identify_temporal_gaps(self, query_lower: str) -> List[ContextGap]:
        """Identify missing temporal context"""
        gaps = []
        
        # Vague time references
        vague_temporal = [
            "recent", "latest", "current", "new", "old", "previous",
            "today", "yesterday", "soon", "now"
        ]
        
        for term in vague_temporal:
            if term in query_lower:
                gap = ContextGap(
                    gap_type=ContextType.TEMPORAL,
                    description=f"Vague time reference: '{term}' needs specific date range",
                    priority=1,
                    suggested_completions=[
                        f"last 30 days instead of '{term}'",
                        f"Q1 2024 instead of '{term}'",
                        f"January 2024 instead of '{term}'"
                    ],
                    business_impact="Unclear time periods can lead to incorrect analysis",
                    examples=[
                        f"Replace '{term} transactions' with 'transactions in last 30 days'",
                        f"Replace '{term} customers' with 'customers onboarded in Q1 2024'"
                    ]
                )
                gaps.append(gap)
        
        # Missing time context for time-sensitive queries
        time_sensitive_terms = ["growth", "trend", "change", "increase", "decrease", "performance"]
        if any(term in query_lower for term in time_sensitive_terms):
            if not any(temporal in query_lower for temporal in ["last", "previous", "from", "to", "between", "q1", "q2", "q3", "q4", "month", "year", "day"]):
                gap = ContextGap(
                    gap_type=ContextType.TEMPORAL,
                    description="Time-sensitive analysis requires specific time period",
                    priority=1,
                    suggested_completions=[
                        "Add 'over last 12 months'",
                        "Add 'compared to previous quarter'",
                        "Add 'year-over-year'"
                    ],
                    business_impact="Performance analysis requires time comparison context",
                    examples=[
                        "Replace 'customer growth' with 'customer growth over last 12 months'",
                        "Replace 'loan performance' with 'loan performance in Q1 vs Q4'"
                    ]
                )
                gaps.append(gap)
        
        return gaps
    
    def _identify_geographic_gaps(self, query_lower: str) -> List[ContextGap]:
        """Identify missing geographic context"""
        gaps = []
        
        # Vague location references
        vague_geographic = ["here", "there", "local", "nearby", "regional"]
        
        for term in vague_geographic:
            if term in query_lower:
                gap = ContextGap(
                    gap_type=ContextType.GEOGRAPHIC,
                    description=f"Vague location reference: '{term}' needs specific geography",
                    priority=2,
                    suggested_completions=[
                        f"specific state name instead of '{term}'",
                        f"specific city name instead of '{term}'",
                        f"specific region (North, South, East, West) instead of '{term}'"
                    ],
                    business_impact="Geographic analysis requires specific location context",
                    examples=[
                        f"Replace '{term} branches' with 'branches in Maharashtra'",
                        f"Replace '{term} customers' with 'customers in North region'"
                    ]
                )
                gaps.append(gap)
        
        # Missing geographic scope for multi-location queries
        location_sensitive = ["branch", "region", "state", "city", "zone", "territory"]
        if any(term in query_lower for term in location_sensitive):
            if not any(geo in query_lower for geo in ["mumbai", "delhi", "bangalore", "chennai", "north", "south", "east", "west", "maharashtra", "karnataka", "tamil nadu"]):
                gap = ContextGap(
                    gap_type=ContextType.GEOGRAPHIC,
                    description="Multi-location analysis could benefit from specific geographic scope",
                    priority=3,
                    suggested_completions=[
                        "Add specific state or region filter",
                        "Add 'across all regions' for comprehensive analysis",
                        "Add 'in tier-1 cities' for urban focus"
                    ],
                    business_impact="Geographic scope helps focus analysis and improves performance",
                    examples=[
                        "Add 'in Maharashtra and Gujarat' to branch analysis",
                        "Add 'across North and West regions' for broader scope"
                    ]
                )
                gaps.append(gap)
        
        return gaps
    
    def _identify_metric_gaps(self, query_lower: str) -> List[ContextGap]:
        """Identify missing metric context"""
        gaps = []
        
        # Vague performance terms
        vague_metrics = ["performance", "efficiency", "good", "bad", "high", "low", "better", "worse"]
        
        for term in vague_metrics:
            if term in query_lower:
                gap = ContextGap(
                    gap_type=ContextType.METRIC,
                    description=f"Vague metric: '{term}' needs specific measurement",
                    priority=1,
                    suggested_completions=[
                        f"specific KPI instead of '{term}' (ROI, NPA ratio, growth rate)",
                        f"quantified threshold instead of '{term}' (>5%, <2%)",
                        f"comparative metric instead of '{term}' (vs industry average)"
                    ],
                    business_impact="Clear metrics enable accurate analysis and decision-making",
                    examples=[
                        f"Replace '{term} performance' with 'ROI > 15%'",
                        f"Replace '{term} customers' with 'customers with balance > 1 lakh'"
                    ]
                )
                gaps.append(gap)
        
        # Missing aggregation context
        potential_aggregation = ["customer", "loan", "account", "transaction", "branch"]
        if any(term in query_lower for term in potential_aggregation):
            if not any(agg in query_lower for agg in ["total", "sum", "count", "average", "max", "min", "top", "bottom"]):
                gap = ContextGap(
                    gap_type=ContextType.METRIC,
                    description="Query could benefit from specific aggregation method",
                    priority=2,
                    suggested_completions=[
                        "Add 'total count of' for counting",
                        "Add 'sum of amounts for' for financial totals",
                        "Add 'top 10' for rankings"
                    ],
                    business_impact="Aggregation methods clarify the required calculation",
                    examples=[
                        "Replace 'customers' with 'total count of customers'",
                        "Replace 'loans' with 'sum of loan amounts'"
                    ]
                )
                gaps.append(gap)
        
        return gaps
    
    def _identify_scope_gaps(self, query_lower: str, user_context: Optional[Dict[str, Any]]) -> List[ContextGap]:
        """Identify missing scope context"""
        gaps = []
        
        # Missing customer type scope
        if "customer" in query_lower:
            if not any(ctype in query_lower for ctype in ["retail", "corporate", "sme", "individual", "business"]):
                gap = ContextGap(
                    gap_type=ContextType.SCOPE,
                    description="Customer analysis could specify customer type",
                    priority=2,
                    suggested_completions=[
                        "Add 'retail customers' for individual customers",
                        "Add 'corporate customers' for business clients",
                        "Add 'SME customers' for small-medium enterprises"
                    ],
                    business_impact="Customer segmentation provides more targeted insights",
                    examples=[
                        "Replace 'customers' with 'retail customers'",
                        "Replace 'customer loans' with 'SME customer loans'"
                    ]
                )
                gaps.append(gap)
        
        # Missing product scope
        if "loan" in query_lower:
            if not any(ltype in query_lower for ltype in ["home", "personal", "auto", "business", "agriculture", "education"]):
                gap = ContextGap(
                    gap_type=ContextType.SCOPE,
                    description="Loan analysis could specify loan product type",
                    priority=2,
                    suggested_completions=[
                        "Add 'home loans' for housing finance",
                        "Add 'personal loans' for unsecured lending",
                        "Add 'business loans' for commercial lending"
                    ],
                    business_impact="Product-specific analysis provides clearer insights",
                    examples=[
                        "Replace 'loans' with 'home loans'",
                        "Replace 'loan performance' with 'personal loan performance'"
                    ]
                )
                gaps.append(gap)
        
        # Missing status scope
        status_relevant = ["account", "loan", "customer", "transaction"]
        if any(term in query_lower for term in status_relevant):
            if not any(status in query_lower for status in ["active", "inactive", "closed", "dormant", "npa", "standard"]):
                gap = ContextGap(
                    gap_type=ContextType.SCOPE,
                    description="Analysis could specify entity status for filtering",
                    priority=3,
                    suggested_completions=[
                        "Add 'active' for currently operational entities",
                        "Add 'closed' for terminated entities",
                        "Add 'NPA' for non-performing assets"
                    ],
                    business_impact="Status filters help focus on relevant data",
                    examples=[
                        "Add 'active accounts' instead of just 'accounts'",
                        "Add 'standard loans' to exclude NPAs"
                    ]
                )
                gaps.append(gap)
        
        return gaps
    
    def _identify_business_rule_gaps(self, query_lower: str) -> List[ContextGap]:
        """Identify missing business rule context"""
        gaps = []
        
        # Risk analysis without risk parameters
        if any(risk_term in query_lower for risk_term in ["risk", "default", "npa"]):
            if not any(param in query_lower for param in ["ratio", "percentage", "dpd", "provision", "exposure"]):
                gap = ContextGap(
                    gap_type=ContextType.BUSINESS_RULE,
                    description="Risk analysis should include specific risk parameters",
                    priority=1,
                    suggested_completions=[
                        "Add 'NPA ratio' for portfolio quality measurement",
                        "Add 'DPD > 90' for default identification",
                        "Add 'provision coverage ratio' for risk provisioning"
                    ],
                    business_impact="Risk parameters are essential for meaningful risk analysis",
                    examples=[
                        "Replace 'risk analysis' with 'NPA ratio analysis'",
                        "Replace 'default customers' with 'customers with DPD > 90'"
                    ]
                )
                gaps.append(gap)
        
        # Compliance queries without regulatory context
        compliance_terms = ["compliance", "regulatory", "audit"]
        if any(term in query_lower for term in compliance_terms):
            if not any(reg in query_lower for reg in ["rbi", "basel", "crar", "slr", "crr", "casa"]):
                gap = ContextGap(
                    gap_type=ContextType.BUSINESS_RULE,
                    description="Compliance analysis should specify regulatory framework",
                    priority=1,
                    suggested_completions=[
                        "Add 'RBI guidelines' for central bank regulations",
                        "Add 'Basel III norms' for capital adequacy",
                        "Add 'CRAR requirements' for capital ratios"
                    ],
                    business_impact="Regulatory context ensures compliance accuracy",
                    examples=[
                        "Replace 'compliance check' with 'RBI compliance check'",
                        "Replace 'regulatory audit' with 'Basel III compliance audit'"
                    ]
                )
                gaps.append(gap)
        
        return gaps
    
    def _identify_relationship_gaps(self, query_lower: str) -> List[ContextGap]:
        """Identify missing relationship context"""
        gaps = []
        
        # Multi-entity queries without relationship specification
        entities = ["customer", "account", "loan", "transaction", "branch", "collateral"]
        entity_count = sum(1 for entity in entities if entity in query_lower)
        
        if entity_count > 1:
            if not any(rel in query_lower for rel in ["with", "having", "linked", "associated", "related", "belonging"]):
                gap = ContextGap(
                    gap_type=ContextType.RELATIONSHIP,
                    description="Multi-entity query could specify relationships more clearly",
                    priority=2,
                    suggested_completions=[
                        "Add 'customers with loans' to specify relationship",
                        "Add 'accounts linked to' for clearer connections",
                        "Add 'transactions associated with' for better joins"
                    ],
                    business_impact="Clear relationships improve query accuracy and performance",
                    examples=[
                        "Replace 'customers loans' with 'customers with loans'",
                        "Replace 'account transactions' with 'transactions for accounts'"
                    ]
                )
                gaps.append(gap)
        
        return gaps
    
    def _generate_completion_for_gap(self, gap: ContextGap, query_text: str, user_context: Optional[Dict[str, Any]]) -> Optional[ContextCompletion]:
        """Generate a specific completion for a context gap"""
        
        if gap.gap_type == ContextType.TEMPORAL:
            return self._generate_temporal_completion(gap, query_text, user_context)
        elif gap.gap_type == ContextType.GEOGRAPHIC:
            return self._generate_geographic_completion(gap, query_text, user_context)
        elif gap.gap_type == ContextType.METRIC:
            return self._generate_metric_completion(gap, query_text, user_context)
        elif gap.gap_type == ContextType.SCOPE:
            return self._generate_scope_completion(gap, query_text, user_context)
        elif gap.gap_type == ContextType.BUSINESS_RULE:
            return self._generate_business_rule_completion(gap, query_text, user_context)
        elif gap.gap_type == ContextType.RELATIONSHIP:
            return self._generate_relationship_completion(gap, query_text, user_context)
        
        return None
    
    def _generate_temporal_completion(self, gap: ContextGap, query_text: str, user_context: Optional[Dict[str, Any]]) -> Optional[ContextCompletion]:
        """Generate temporal context completion"""
        
        # Default to last 30 days for most queries
        default_period = "last 30 days"
        confidence = CompletionConfidence.MEDIUM
        
        # Adjust based on query type
        if "growth" in query_text.lower() or "trend" in query_text.lower():
            default_period = "over last 12 months"
            confidence = CompletionConfidence.HIGH
        elif "quarterly" in query_text.lower() or "quarter" in query_text.lower():
            default_period = "in current quarter"
            confidence = CompletionConfidence.HIGH
        elif "performance" in query_text.lower():
            default_period = "compared to previous period"
            confidence = CompletionConfidence.MEDIUM
        
        # Use user context for better completion
        if user_context:
            preferred_period = user_context.get("default_time_period")
            if preferred_period:
                default_period = preferred_period
                confidence = CompletionConfidence.HIGH
        
        return ContextCompletion(
            completion_text=default_period,
            context_type=ContextType.TEMPORAL,
            confidence=confidence,
            reasoning=f"Added '{default_period}' to provide specific temporal scope",
            business_rule="Time-bound analysis provides more actionable insights",
            impact_description="Enables accurate time-series analysis and trend identification",
            alternative_completions=[
                "last 90 days",
                "current financial year",
                "year-to-date",
                "previous quarter"
            ]
        )
    
    def _generate_geographic_completion(self, gap: ContextGap, query_text: str, user_context: Optional[Dict[str, Any]]) -> Optional[ContextCompletion]:
        """Generate geographic context completion"""
        
        default_scope = "across all regions"
        confidence = CompletionConfidence.SUGGESTED
        
        # Use user context for better completion
        if user_context:
            user_region = user_context.get("default_region")
            if user_region:
                default_scope = f"in {user_region}"
                confidence = CompletionConfidence.HIGH
        
        return ContextCompletion(
            completion_text=default_scope,
            context_type=ContextType.GEOGRAPHIC,
            confidence=confidence,
            reasoning=f"Added '{default_scope}' for geographic clarity",
            business_rule="Geographic scope helps focus analysis and improve performance",
            impact_description="Enables region-specific insights and comparisons",
            alternative_completions=[
                "in tier-1 cities",
                "in North and West regions",
                "in Maharashtra and Gujarat",
                "excluding metro cities"
            ]
        )
    
    def _generate_metric_completion(self, gap: ContextGap, query_text: str, user_context: Optional[Dict[str, Any]]) -> Optional[ContextCompletion]:
        """Generate metric context completion"""
        
        # Determine appropriate metric based on query content
        if "customer" in query_text.lower():
            metric = "total count of"
            confidence = CompletionConfidence.HIGH
        elif "amount" in query_text.lower() or "loan" in query_text.lower():
            metric = "sum of"
            confidence = CompletionConfidence.HIGH
        elif "performance" in query_text.lower():
            metric = "average"
            confidence = CompletionConfidence.MEDIUM
        else:
            metric = "total"
            confidence = CompletionConfidence.SUGGESTED
        
        return ContextCompletion(
            completion_text=metric,
            context_type=ContextType.METRIC,
            confidence=confidence,
            reasoning=f"Added '{metric}' for clear aggregation method",
            business_rule="Specific metrics enable precise calculations and comparisons",
            impact_description="Clarifies the type of analysis and expected output format",
            alternative_completions=[
                "average of",
                "maximum",
                "minimum",
                "top 10",
                "bottom 5"
            ]
        )
    
    def _generate_scope_completion(self, gap: ContextGap, query_text: str, user_context: Optional[Dict[str, Any]]) -> Optional[ContextCompletion]:
        """Generate scope context completion"""
        
        scope = "active"
        confidence = CompletionConfidence.MEDIUM
        
        # Determine appropriate scope
        if "customer" in query_text.lower():
            scope = "active customers"
            confidence = CompletionConfidence.HIGH
        elif "loan" in query_text.lower():
            scope = "standard loans"  # Exclude NPAs by default
            confidence = CompletionConfidence.HIGH
        elif "account" in query_text.lower():
            scope = "active accounts"
            confidence = CompletionConfidence.HIGH
        
        return ContextCompletion(
            completion_text=scope,
            context_type=ContextType.SCOPE,
            confidence=confidence,
            reasoning=f"Added '{scope}' for focused analysis scope",
            business_rule="Scope filtering ensures relevant and actionable results",
            impact_description="Focuses analysis on operationally relevant entities",
            alternative_completions=[
                "all entities (including inactive)",
                "high-value segments only",
                "retail segment only",
                "corporate segment only"
            ]
        )
    
    def _generate_business_rule_completion(self, gap: ContextGap, query_text: str, user_context: Optional[Dict[str, Any]]) -> Optional[ContextCompletion]:
        """Generate business rule context completion"""
        
        if "risk" in query_text.lower():
            completion = "with NPA ratio calculation"
            rule = "RBI guidelines for NPA classification"
        elif "compliance" in query_text.lower():
            completion = "as per RBI guidelines"
            rule = "Reserve Bank of India regulatory framework"
        elif "capital" in query_text.lower():
            completion = "meeting CRAR requirements"
            rule = "Basel III capital adequacy norms"
        else:
            completion = "as per banking regulations"
            rule = "Standard banking compliance framework"
        
        return ContextCompletion(
            completion_text=completion,
            context_type=ContextType.BUSINESS_RULE,
            confidence=CompletionConfidence.HIGH,
            reasoning=f"Added '{completion}' for regulatory compliance",
            business_rule=rule,
            impact_description="Ensures analysis meets regulatory standards and banking best practices",
            alternative_completions=[
                "as per internal policies",
                "following industry benchmarks",
                "meeting Basel III standards",
                "per RBI master directions"
            ]
        )
    
    def _generate_relationship_completion(self, gap: ContextGap, query_text: str, user_context: Optional[Dict[str, Any]]) -> Optional[ContextCompletion]:
        """Generate relationship context completion"""
        
        # Infer relationship based on entities mentioned
        if "customer" in query_text.lower() and "loan" in query_text.lower():
            completion = "customers with loans"
        elif "account" in query_text.lower() and "transaction" in query_text.lower():
            completion = "transactions for accounts"
        elif "loan" in query_text.lower() and "collateral" in query_text.lower():
            completion = "loans secured by collateral"
        else:
            completion = "related entities"
        
        return ContextCompletion(
            completion_text=completion,
            context_type=ContextType.RELATIONSHIP,
            confidence=CompletionConfidence.MEDIUM,
            reasoning=f"Clarified relationship as '{completion}'",
            business_rule="Clear entity relationships improve query performance and accuracy",
            impact_description="Ensures proper data joins and relationship handling",
            alternative_completions=[
                "all linked entities",
                "directly associated records",
                "hierarchically related data",
                "cross-referenced entities"
            ]
        )
    
    def _apply_completion(self, query_text: str, completion: ContextCompletion) -> str:
        """Apply a completion to enhance the query"""
        
        enhanced_query = query_text
        
        if completion.context_type == ContextType.TEMPORAL:
            # Add temporal context at the end
            enhanced_query = f"{query_text} {completion.completion_text}"
        
        elif completion.context_type == ContextType.GEOGRAPHIC:
            # Add geographic context appropriately
            enhanced_query = f"{query_text} {completion.completion_text}"
        
        elif completion.context_type == ContextType.METRIC:
            # Add metric at the beginning
            enhanced_query = f"{completion.completion_text} {query_text}"
        
        elif completion.context_type == ContextType.SCOPE:
            # Replace general terms with scoped terms
            enhanced_query = query_text.replace("customers", completion.completion_text)
            enhanced_query = enhanced_query.replace("loans", completion.completion_text)
            enhanced_query = enhanced_query.replace("accounts", completion.completion_text)
        
        elif completion.context_type == ContextType.BUSINESS_RULE:
            # Add business rule context
            enhanced_query = f"{query_text} {completion.completion_text}"
        
        elif completion.context_type == ContextType.RELATIONSHIP:
            # Replace entity combinations with clearer relationships
            enhanced_query = re.sub(r'customers?\s+loans?', completion.completion_text, enhanced_query, flags=re.IGNORECASE)
            enhanced_query = re.sub(r'accounts?\s+transactions?', completion.completion_text, enhanced_query, flags=re.IGNORECASE)
        
        return enhanced_query.strip()
    
    def _calculate_completion_confidence(self, completions: List[ContextCompletion]) -> float:
        """Calculate overall confidence score for completions"""
        if not completions:
            return 1.0  # No completions needed
        
        confidence_values = {
            CompletionConfidence.HIGH: 0.9,
            CompletionConfidence.MEDIUM: 0.7,
            CompletionConfidence.LOW: 0.5,
            CompletionConfidence.SUGGESTED: 0.3
        }
        
        total_confidence = sum(confidence_values[c.confidence] for c in completions)
        return total_confidence / len(completions)
    
    def _generate_enhancement_suggestions(self, query: str, gaps: List[ContextGap], user_context: Optional[Dict[str, Any]]) -> List[str]:
        """Generate suggestions for further query enhancement"""
        suggestions = []
        
        # High-priority gap suggestions
        high_priority_gaps = [g for g in gaps if g.priority == 1]
        if high_priority_gaps:
            suggestions.append(f"Consider addressing {len(high_priority_gaps)} high-priority context items for better accuracy")
        
        # Performance suggestions
        if len(gaps) > 3:
            suggestions.append("Query has multiple context gaps - consider breaking into smaller, more focused queries")
        
        # Business intelligence suggestions
        if "performance" in query.lower():
            suggestions.append("Add comparative metrics (vs previous period, vs benchmark) for better insights")
        
        if "customer" in query.lower():
            suggestions.append("Consider customer segmentation (retail/corporate/SME) for more targeted analysis")
        
        return suggestions
    
    def _extract_business_context(self, query: str, completions: List[ContextCompletion]) -> Dict[str, Any]:
        """Extract business context from completed query"""
        context = {
            "domains_involved": [],
            "analysis_type": "descriptive",
            "time_sensitivity": "medium",
            "compliance_relevant": False,
            "risk_relevant": False
        }
        
        # Identify domains
        if any(term in query.lower() for term in ["customer", "client"]):
            context["domains_involved"].append("customer_management")
        if any(term in query.lower() for term in ["loan", "credit"]):
            context["domains_involved"].append("loan_portfolio")
        if any(term in query.lower() for term in ["account", "deposit"]):
            context["domains_involved"].append("deposit_management")
        if any(term in query.lower() for term in ["risk", "npa", "default"]):
            context["domains_involved"].append("risk_management")
            context["risk_relevant"] = True
        if any(term in query.lower() for term in ["compliance", "regulatory"]):
            context["compliance_relevant"] = True
        
        # Analysis type
        if any(term in query.lower() for term in ["trend", "growth", "change"]):
            context["analysis_type"] = "trend_analysis"
        elif any(term in query.lower() for term in ["compare", "versus", "vs"]):
            context["analysis_type"] = "comparative"
        elif any(term in query.lower() for term in ["predict", "forecast"]):
            context["analysis_type"] = "predictive"
        
        # Time sensitivity
        temporal_completions = [c for c in completions if c.context_type == ContextType.TEMPORAL]
        if temporal_completions:
            context["time_sensitivity"] = "high"
        
        return context
    
    # Initialize methods for various rule sets
    def _initialize_business_rules(self) -> Dict[str, Any]:
        """Initialize banking business rules"""
        return {
            "npa_classification": {
                "rule": "Loans overdue for more than 90 days are classified as NPA",
                "application": "risk_analysis",
                "parameters": {"dpd_threshold": 90}
            },
            "capital_adequacy": {
                "rule": "CRAR should be minimum 9% as per RBI norms",
                "application": "compliance_check",
                "parameters": {"crar_minimum": 9.0}
            },
            "customer_segmentation": {
                "rule": "Retail customers have exposure < 5 crores, Corporate > 5 crores",
                "application": "customer_analysis",
                "parameters": {"retail_threshold": 50000000}
            }
        }
    
    def _initialize_temporal_patterns(self) -> Dict[str, Any]:
        """Initialize temporal completion patterns"""
        return {
            "default_periods": {
                "performance_analysis": "last 12 months",
                "transaction_analysis": "last 30 days",
                "customer_analysis": "current quarter",
                "risk_analysis": "last 90 days"
            },
            "seasonal_adjustments": {
                "march": "financial year end analysis",
                "december": "calendar year analysis",
                "quarterly": "quarterly comparison"
            }
        }
    
    def _initialize_geographic_context(self) -> Dict[str, Any]:
        """Initialize geographic completion knowledge"""
        return {
            "regions": ["North", "South", "East", "West", "Central"],
            "tier_cities": {
                "tier1": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad"],
                "tier2": ["Pune", "Ahmedabad", "Surat", "Jaipur", "Lucknow", "Kanpur"]
            },
            "states": ["Maharashtra", "Karnataka", "Tamil Nadu", "Gujarat", "Rajasthan"]
        }
    
    def _initialize_analyst_patterns(self) -> Dict[str, Any]:
        """Initialize professional analyst patterns"""
        return {
            "risk_analyst": {
                "preferred_metrics": ["NPA ratio", "provision coverage", "credit loss"],
                "default_scope": "portfolio level",
                "time_horizon": "quarterly"
            },
            "credit_analyst": {
                "preferred_metrics": ["disbursement", "approval rate", "ticket size"],
                "default_scope": "product level",
                "time_horizon": "monthly"
            },
            "relationship_manager": {
                "preferred_metrics": ["customer acquisition", "portfolio growth", "cross-sell"],
                "default_scope": "region level",
                "time_horizon": "monthly"
            }
        }
    
    def _initialize_metric_rules(self) -> Dict[str, Any]:
        """Initialize metric completion rules"""
        return {
            "aggregation_defaults": {
                "customer": "count",
                "amount": "sum",
                "rate": "average",
                "ratio": "percentage"
            },
            "business_kpis": {
                "portfolio_quality": ["NPA ratio", "provision coverage ratio"],
                "profitability": ["ROA", "ROE", "NIM"],
                "efficiency": ["cost to income ratio", "productivity ratios"]
            }
        }
    
    def _initialize_scope_patterns(self) -> Dict[str, Any]:
        """Initialize scope completion patterns"""
        return {
            "default_filters": {
                "customer": "active customers",
                "loan": "standard loans",
                "account": "operational accounts"
            },
            "segmentation_options": {
                "customer": ["retail", "corporate", "SME"],
                "loan": ["secured", "unsecured", "priority sector"],
                "geography": ["urban", "semi-urban", "rural"]
            }
        }
    
    # Helper methods for various suggestion types
    def _apply_professional_terminology(self, query: str) -> str:
        """Apply professional banking terminology"""
        # This would be expanded with comprehensive terminology mapping
        professional_terms = {
            "bad loans": "non-performing assets",
            "money": "amount",
            "profit": "net income",
            "loss": "provision"
        }
        
        enhanced = query
        for casual, professional in professional_terms.items():
            enhanced = enhanced.replace(casual, professional)
        
        return enhanced
    
    def _add_banking_qualifiers(self, query: str, analyst_profile: Optional[Dict[str, Any]]) -> str:
        """Add banking-specific qualifiers"""
        # Add qualifiers based on analyst profile and query content
        return query  # Placeholder for detailed implementation
    
    def _apply_best_practices(self, query: str, analyst_profile: Optional[Dict[str, Any]]) -> str:
        """Apply banking query best practices"""
        # Apply industry best practices
        return query  # Placeholder for detailed implementation
    
    def _get_performance_suggestions(self, query: str, performance_context: Dict[str, Any]) -> List[str]:
        """Get performance optimization suggestions"""
        return ["Consider adding date range filters to improve query performance"]
    
    def _get_accuracy_suggestions(self, query: str) -> List[str]:
        """Get accuracy improvement suggestions"""
        return ["Add specific metrics for more precise analysis"]
    
    def _get_business_intelligence_suggestions(self, query: str) -> List[str]:
        """Get business intelligence enhancement suggestions"""
        return ["Consider adding comparative analysis for better insights"]


# Global context completer instance
context_completer = ContextCompleter()
