"""
Query Analyzer for NLP Processor
Deep query analysis and preprocessing for banking domain queries
Provides complexity assessment, ambiguity detection, and analyst pattern recognition
"""

from collections import defaultdict
import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import spacy

from agent.nlp_processor.core.exceptions import QueryAmbiguityError
from agent.nlp_processor.core.metrics import ComponentType, track_processing_time, metrics_collector


logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"          # Single table, basic filters
    MODERATE = "moderate"      # Multiple tables, joins, aggregations
    COMPLEX = "complex"        # Multiple joins, subqueries, advanced analytics
    EXPERT = "expert"          # Cross-domain analysis, complex business logic


class QueryType(Enum):
    """Types of analyst queries"""
    DESCRIPTIVE = "descriptive"        # What happened?
    DIAGNOSTIC = "diagnostic"          # Why did it happen?
    PREDICTIVE = "predictive"          # What might happen?
    PRESCRIPTIVE = "prescriptive"      # What should we do?
    EXPLORATORY = "exploratory"        # General data exploration


class AnalystIntent(Enum):
    """Professional analyst intent categories"""
    RISK_ANALYSIS = "risk_analysis"
    CUSTOMER_ANALYSIS = "customer_analysis"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    COMPLIANCE_CHECK = "compliance_check"
    PERFORMANCE_REVIEW = "performance_review"
    TREND_ANALYSIS = "trend_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    OPERATIONAL_METRICS = "operational_metrics"


class AmbiguityType(Enum):
    """Types of query ambiguity"""
    TEMPORAL = "temporal"              # Unclear time references
    ENTITY = "entity"                  # Ambiguous entity references
    SCOPE = "scope"                    # Unclear analysis scope
    METRIC = "metric"                  # Ambiguous metrics/calculations
    RELATIONSHIP = "relationship"      # Unclear data relationships


@dataclass
class QueryFeatures:
    """Extracted features from query analysis"""
    # Linguistic features
    word_count: int = 0
    sentence_count: int = 0
    avg_word_length: float = 0.0
    
    # Business features
    banking_terms: List[str] = field(default_factory=list)
    financial_amounts: List[str] = field(default_factory=list)
    date_expressions: List[str] = field(default_factory=list)
    entity_mentions: List[str] = field(default_factory=list)
    
    # Query structure
    has_aggregation: bool = False
    has_comparison: bool = False
    has_filtering: bool = False
    has_sorting: bool = False
    has_grouping: bool = False
    
    # Professional indicators
    analyst_terminology: List[str] = field(default_factory=list)
    regional_references: List[str] = field(default_factory=list)
    compliance_indicators: List[str] = field(default_factory=list)


@dataclass
class AmbiguityDetection:
    """Results of ambiguity detection"""
    is_ambiguous: bool = False
    ambiguity_types: List[AmbiguityType] = field(default_factory=list)
    ambiguous_terms: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    suggestions: List[str] = field(default_factory=list)


@dataclass
class QueryAnalysisResult:
    """Complete query analysis result"""
    original_query: str
    normalized_query: str
    complexity: QueryComplexity
    query_type: QueryType
    analyst_intent: AnalystIntent
    features: QueryFeatures
    ambiguity: AmbiguityDetection
    confidence_score: float
    processing_hints: Dict[str, Any] = field(default_factory=dict)
    recommended_approach: str = ""


class QueryAnalyzer:
    """
    Deep query analysis for banking domain queries
    Analyzes complexity, detects ambiguity, and provides processing guidance
    """
    
    def __init__(self):
        """Initialize query analyzer with banking domain knowledge"""
        # Load spaCy model for linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Using simplified analysis.")
            self.nlp = None
        
        # Banking domain vocabulary
        self.banking_terms = {
            'accounts': ['account', 'accounts', 'acc', 'a/c'],
            'customers': ['customer', 'customers', 'client', 'clients', 'borrower', 'borrowers'],
            'loans': ['loan', 'loans', 'credit', 'lending', 'advance', 'advances'],
            'deposits': ['deposit', 'deposits', 'savings', 'fd', 'fixed deposit', 'rd', 'recurring deposit'],
            'transactions': ['transaction', 'transactions', 'txn', 'payment', 'payments', 'transfer', 'transfers'],
            'collateral': ['collateral', 'security', 'guarantee', 'pledge', 'mortgage'],
            'branches': ['branch', 'branches', 'location', 'office', 'center'],
            'products': ['product', 'products', 'scheme', 'schemes', 'plan', 'plans']
        }
        
        # Financial amount patterns
        self.amount_patterns = [
            r'\b\d+\s*(?:crore|crores|cr)\b',
            r'\b\d+\s*(?:lakh|lakhs|lac|lacs)\b',
            r'\b\d+\s*(?:thousand|k)\b',
            r'\b₹\s*\d+(?:,\d+)*(?:\.\d+)?\b',
            r'\b\d+(?:,\d+)*(?:\.\d+)?\s*(?:rupees?|rs\.?|inr)\b'
        ]
        
        # Date/time patterns
        self.temporal_patterns = [
            r'\blast\s+\d+\s+(?:days?|weeks?|months?|years?)\b',
            r'\b(?:yesterday|today|tomorrow)\b',
            r'\b(?:last|this|next)\s+(?:week|month|quarter|year)\b',
            r'\b(?:q[1-4]|quarter\s+[1-4])\s+(?:20\d{2}|fy\s*20\d{2})\b',
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+20\d{2}\b',
            r'\b\d{1,2}[-/]\d{1,2}[-/](?:20)?\d{2}\b'
        ]
        
        # Regional patterns
        self.regional_patterns = [
            r'\b(?:north|south|east|west|central)\s+(?:india|region|zone)\b',
            r'\b(?:mumbai|delhi|chennai|kolkata|bangalore|hyderabad|pune|ahmedabad)\b',
            r'\b(?:maharashtra|karnataka|tamil nadu|gujarat|rajasthan|punjab|haryana|kerala|bihar|odisha|west bengal|uttar pradesh|madhya pradesh|andhra pradesh|telangana)\b'
        ]
        
        # Aggregation indicators
        self.aggregation_terms = {
            'sum': ['sum', 'total', 'aggregate', 'combined'],
            'count': ['count', 'number', 'how many', 'quantity'],
            'average': ['average', 'avg', 'mean', 'typical'],
            'maximum': ['maximum', 'max', 'highest', 'peak', 'top'],
            'minimum': ['minimum', 'min', 'lowest', 'bottom'],
            'percentage': ['percentage', 'percent', '%', 'proportion', 'ratio']
        }
        
        # Comparison indicators
        self.comparison_terms = [
            'compare', 'comparison', 'versus', 'vs', 'against', 'between',
            'higher', 'lower', 'greater', 'less', 'more', 'fewer',
            'better', 'worse', 'increase', 'decrease', 'growth', 'decline'
        ]
        
        # Professional analyst patterns
        self.analyst_patterns = {
            'risk': ['risk', 'risky', 'default', 'npas', 'provisions', 'exposure', 'concentration'],
            'performance': ['performance', 'efficiency', 'productivity', 'roi', 'returns', 'profitability'],
            'compliance': ['compliance', 'regulatory', 'audit', 'violation', 'adherence', 'guidelines'],
            'trends': ['trend', 'trending', 'pattern', 'seasonality', 'cyclical', 'movement'],
            'segmentation': ['segment', 'category', 'group', 'cluster', 'classification', 'demographic']
        }
        
        logger.info("QueryAnalyzer initialized with banking domain knowledge")
    
    def analyze_query(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> QueryAnalysisResult:
        """
        Perform comprehensive query analysis
        
        Args:
            query_text: Natural language query from analyst
            context: Optional context information
            
        Returns:
            Complete query analysis result
        """
        with track_processing_time(ComponentType.QUERY_ANALYZER, "analyze_query"):
            try:
                # Normalize query
                normalized_query = self._normalize_query(query_text)
                
                # Extract features
                features = self._extract_features(normalized_query)
                
                # Assess complexity
                complexity = self._assess_complexity(features, normalized_query)
                
                # Determine query type
                query_type = self._determine_query_type(normalized_query, features)
                
                # Identify analyst intent
                analyst_intent = self._identify_analyst_intent(normalized_query, features)
                
                # Detect ambiguity
                ambiguity = self._detect_ambiguity(normalized_query, features)
                
                # Calculate confidence
                confidence_score = self._calculate_confidence(features, ambiguity)
                
                # Generate processing hints
                processing_hints = self._generate_processing_hints(complexity, query_type, features)
                
                # Recommend processing approach
                recommended_approach = self._recommend_approach(complexity, query_type, analyst_intent)
                
                result = QueryAnalysisResult(
                    original_query=query_text,
                    normalized_query=normalized_query,
                    complexity=complexity,
                    query_type=query_type,
                    analyst_intent=analyst_intent,
                    features=features,
                    ambiguity=ambiguity,
                    confidence_score=confidence_score,
                    processing_hints=processing_hints,
                    recommended_approach=recommended_approach
                )
                
                # Record metrics
                metrics_collector.record_accuracy(
                    ComponentType.QUERY_ANALYZER,
                    not ambiguity.is_ambiguous,
                    confidence_score,
                    "query_analysis"
                )
                
                # Check for critical ambiguity
                if ambiguity.is_ambiguous and ambiguity.confidence_score < 0.6:
                    raise QueryAmbiguityError(
                        query_text=query_text,
                        ambiguous_terms=ambiguity.ambiguous_terms
                    )
                
                return result
                
            except Exception as e:
                logger.error(f"Query analysis failed: {e}")
                raise
    
    def _normalize_query(self, query_text: str) -> str:
        """Normalize query text for analysis"""
        # Basic cleaning
        normalized = query_text.strip().lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Expand common abbreviations
        abbreviations = {
            'acc': 'account',
            'txn': 'transaction',
            'cr': 'crore',
            'lac': 'lakh',
            'k': 'thousand',
            'q1': 'quarter 1',
            'q2': 'quarter 2',
            'q3': 'quarter 3',
            'q4': 'quarter 4',
            'fy': 'financial year',
            'yoy': 'year over year',
            'mom': 'month over month'
        }
        
        for abbr, full in abbreviations.items():
            normalized = re.sub(rf'\b{abbr}\b', full, normalized)
        
        return normalized
    
    def _extract_features(self, query_text: str) -> QueryFeatures:
        """Extract comprehensive features from query"""
        features = QueryFeatures()
        
        # Basic linguistic features
        words = query_text.split()
        features.word_count = len(words)
        features.sentence_count = len([s for s in query_text.split('.') if s.strip()])
        features.avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Banking terms
        for category, terms in self.banking_terms.items():
            for term in terms:
                if term in query_text:
                    features.banking_terms.append(term)
        
        # Financial amounts
        for pattern in self.amount_patterns:
            matches = re.findall(pattern, query_text, re.IGNORECASE)
            features.financial_amounts.extend(matches)
        
        # Date expressions
        for pattern in self.temporal_patterns:
            matches = re.findall(pattern, query_text, re.IGNORECASE)
            features.date_expressions.extend(matches)
        
        # Regional references
        for pattern in self.regional_patterns:
            matches = re.findall(pattern, query_text, re.IGNORECASE)
            features.regional_references.extend(matches)
        
        # Query structure analysis
        features.has_aggregation = any(
            any(term in query_text for term in terms)
            for terms in self.aggregation_terms.values()
        )
        
        features.has_comparison = any(term in query_text for term in self.comparison_terms)
        
        features.has_filtering = any(word in query_text for word in [
            'where', 'filter', 'only', 'exclude', 'include', 'condition'
        ])
        
        features.has_sorting = any(word in query_text for word in [
            'sort', 'order', 'rank', 'top', 'bottom', 'highest', 'lowest'
        ])
        
        features.has_grouping = any(word in query_text for word in [
            'group', 'segment', 'category', 'by region', 'by branch', 'by product'
        ])
        
        # Professional analyst terminology
        for category, terms in self.analyst_patterns.items():
            for term in terms:
                if term in query_text:
                    features.analyst_terminology.append(term)
        
        # Compliance indicators
        compliance_terms = ['compliance', 'regulatory', 'audit', 'violation', 'policy', 'guideline']
        features.compliance_indicators = [term for term in compliance_terms if term in query_text]
        
        # Entity mentions (simplified without spaCy)
        if self.nlp:
            doc = self.nlp(query_text)
            features.entity_mentions = [ent.text for ent in doc.ents]
        
        return features
    
    def _assess_complexity(self, features: QueryFeatures, query_text: str) -> QueryComplexity:
        """Assess query complexity based on features"""
        complexity_score = 0
        
        # Basic factors
        if features.word_count > 20:
            complexity_score += 1
        if features.word_count > 40:
            complexity_score += 1
        
        # Banking domain complexity
        if len(features.banking_terms) > 3:
            complexity_score += 1
        
        # Structural complexity
        if features.has_aggregation:
            complexity_score += 1
        if features.has_comparison:
            complexity_score += 1
        if features.has_grouping:
            complexity_score += 1
        if features.has_filtering and features.has_sorting:
            complexity_score += 1
        
        # Advanced features
        if len(features.financial_amounts) > 2:
            complexity_score += 1
        if len(features.date_expressions) > 1:
            complexity_score += 1
        if len(features.regional_references) > 0:
            complexity_score += 1
        
        # Professional analysis complexity
        if len(features.analyst_terminology) > 2:
            complexity_score += 1
        if features.compliance_indicators:
            complexity_score += 1
        
        # Advanced query patterns
        advanced_patterns = [
            'correlation', 'trend analysis', 'forecasting', 'prediction',
            'machine learning', 'clustering', 'segmentation', 'cohort',
            'survival analysis', 'time series', 'regression'
        ]
        if any(pattern in query_text for pattern in advanced_patterns):
            complexity_score += 2
        
        # Determine complexity level
        if complexity_score <= 2:
            return QueryComplexity.SIMPLE
        elif complexity_score <= 5:
            return QueryComplexity.MODERATE
        elif complexity_score <= 8:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.EXPERT
    
    def _determine_query_type(self, query_text: str, features: QueryFeatures) -> QueryType:
        """Determine the type of query"""
        # Descriptive patterns
        descriptive_patterns = ['what', 'how many', 'show me', 'list', 'display', 'count']
        if any(pattern in query_text for pattern in descriptive_patterns):
            return QueryType.DESCRIPTIVE
        
        # Diagnostic patterns
        diagnostic_patterns = ['why', 'reason', 'cause', 'explain', 'analyze', 'investigate']
        if any(pattern in query_text for pattern in diagnostic_patterns):
            return QueryType.DIAGNOSTIC
        
        # Predictive patterns
        predictive_patterns = ['predict', 'forecast', 'estimate', 'project', 'expect', 'likely']
        if any(pattern in query_text for pattern in predictive_patterns):
            return QueryType.PREDICTIVE
        
        # Prescriptive patterns
        prescriptive_patterns = ['recommend', 'suggest', 'optimize', 'improve', 'should', 'strategy']
        if any(pattern in query_text for pattern in prescriptive_patterns):
            return QueryType.PRESCRIPTIVE
        
        # Default to exploratory if unclear
        return QueryType.EXPLORATORY
    
    def _identify_analyst_intent(self, query_text: str, features: QueryFeatures) -> AnalystIntent:
        """Identify professional analyst intent"""
        # Risk analysis
        if any(term in features.analyst_terminology for term in ['risk', 'default', 'npas', 'provisions']):
            return AnalystIntent.RISK_ANALYSIS
        
        # Customer analysis
        if 'customers' in features.banking_terms or 'client' in query_text:
            return AnalystIntent.CUSTOMER_ANALYSIS
        
        # Portfolio analysis
        if any(term in query_text for term in ['portfolio', 'asset', 'investment', 'allocation']):
            return AnalystIntent.PORTFOLIO_ANALYSIS
        
        # Compliance check
        if features.compliance_indicators:
            return AnalystIntent.COMPLIANCE_CHECK
        
        # Performance review
        if any(term in features.analyst_terminology for term in ['performance', 'efficiency', 'roi']):
            return AnalystIntent.PERFORMANCE_REVIEW
        
        # Trend analysis
        if any(term in features.analyst_terminology for term in ['trend', 'pattern', 'seasonality']):
            return AnalystIntent.TREND_ANALYSIS
        
        # Comparative analysis
        if features.has_comparison:
            return AnalystIntent.COMPARATIVE_ANALYSIS
        
        # Default to operational metrics
        return AnalystIntent.OPERATIONAL_METRICS
    
    def _detect_ambiguity(self, query_text: str, features: QueryFeatures) -> AmbiguityDetection:
        """Detect potential ambiguities in the query"""
        ambiguity = AmbiguityDetection()
        ambiguity_score = 0
        
        # Temporal ambiguity
        temporal_ambiguous = ['recent', 'current', 'latest', 'new', 'old', 'previous']
        if any(term in query_text for term in temporal_ambiguous) and not features.date_expressions:
            ambiguity.ambiguity_types.append(AmbiguityType.TEMPORAL)
            ambiguity.ambiguous_terms.extend([term for term in temporal_ambiguous if term in query_text])
            ambiguity_score += 1
        
        # Entity ambiguity
        entity_ambiguous = ['it', 'they', 'them', 'this', 'that', 'these', 'those']
        if any(term in query_text for term in entity_ambiguous):
            ambiguity.ambiguity_types.append(AmbiguityType.ENTITY)
            ambiguity.ambiguous_terms.extend([term for term in entity_ambiguous if term in query_text])
            ambiguity_score += 0.5
        
        # Scope ambiguity
        if not features.regional_references and not features.banking_terms:
            ambiguity.ambiguity_types.append(AmbiguityType.SCOPE)
            ambiguity_score += 1
        
        # Metric ambiguity
        vague_metrics = ['performance', 'efficiency', 'good', 'bad', 'high', 'low']
        if any(term in query_text for term in vague_metrics) and not features.has_aggregation:
            ambiguity.ambiguity_types.append(AmbiguityType.METRIC)
            ambiguity.ambiguous_terms.extend([term for term in vague_metrics if term in query_text])
            ambiguity_score += 0.5
        
        # Relationship ambiguity
        if features.word_count > 15 and not (features.has_filtering or features.has_grouping):
            ambiguity.ambiguity_types.append(AmbiguityType.RELATIONSHIP)
            ambiguity_score += 0.5
        
        # Determine if ambiguous
        ambiguity.is_ambiguous = ambiguity_score > 1.0
        ambiguity.confidence_score = max(0.0, 1.0 - (ambiguity_score / 3.0))
        
        # Generate suggestions
        if ambiguity.is_ambiguous:
            ambiguity.suggestions = self._generate_ambiguity_suggestions(ambiguity.ambiguity_types)
        
        return ambiguity
    
    def _generate_ambiguity_suggestions(self, ambiguity_types: List[AmbiguityType]) -> List[str]:
        """Generate suggestions to resolve ambiguity"""
        suggestions = []
        
        if AmbiguityType.TEMPORAL in ambiguity_types:
            suggestions.append("Specify exact time periods like 'last 30 days' or 'Q1 2024'")
        
        if AmbiguityType.ENTITY in ambiguity_types:
            suggestions.append("Use specific names or identifiers instead of pronouns")
        
        if AmbiguityType.SCOPE in ambiguity_types:
            suggestions.append("Specify the region, branch, or product scope for your analysis")
        
        if AmbiguityType.METRIC in ambiguity_types:
            suggestions.append("Define specific metrics like 'total amount', 'average balance', or 'count of accounts'")
        
        if AmbiguityType.RELATIONSHIP in ambiguity_types:
            suggestions.append("Clarify how different data elements should be related or grouped")
        
        return suggestions
    
    def _calculate_confidence(self, features: QueryFeatures, ambiguity: AmbiguityDetection) -> float:
        """Calculate overall confidence in query analysis"""
        base_confidence = 0.7
        
        # Boost confidence for clear banking terms
        if len(features.banking_terms) >= 2:
            base_confidence += 0.1
        
        # Boost for specific amounts and dates
        if features.financial_amounts:
            base_confidence += 0.05
        if features.date_expressions:
            base_confidence += 0.05
        
        # Boost for clear structure
        if features.has_aggregation or features.has_filtering:
            base_confidence += 0.05
        
        # Reduce for ambiguity
        base_confidence *= ambiguity.confidence_score
        
        return min(1.0, max(0.0, base_confidence))
    
    def _generate_processing_hints(self, complexity: QueryComplexity, query_type: QueryType, features: QueryFeatures) -> Dict[str, Any]:
        """Generate hints for downstream processing"""
        hints = {
            "complexity_level": complexity.value,
            "expected_processing_time": self._estimate_processing_time(complexity),
            "recommended_timeout": self._recommend_timeout(complexity),
            "parallelization_candidate": complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT],
            "cache_candidate": query_type == QueryType.DESCRIPTIVE and complexity == QueryComplexity.SIMPLE,
            "requires_human_review": complexity == QueryComplexity.EXPERT,
            "priority_level": self._determine_priority(features),
            "suggested_optimizations": self._suggest_optimizations(features, complexity)
        }
        
        return hints
    
    def _recommend_approach(self, complexity: QueryComplexity, query_type: QueryType, intent: AnalystIntent) -> str:
        """Recommend processing approach based on analysis"""
        if complexity == QueryComplexity.SIMPLE:
            return "direct_processing"
        elif complexity == QueryComplexity.MODERATE:
            if query_type == QueryType.DESCRIPTIVE:
                return "standard_pipeline"
            else:
                return "enhanced_pipeline"
        elif complexity == QueryComplexity.COMPLEX:
            if intent in [AnalystIntent.RISK_ANALYSIS, AnalystIntent.COMPLIANCE_CHECK]:
                return "specialist_pipeline"
            else:
                return "advanced_pipeline"
        else:  # EXPERT
            return "expert_review_required"
    
    def _estimate_processing_time(self, complexity: QueryComplexity) -> float:
        """Estimate processing time in seconds"""
        time_estimates = {
            QueryComplexity.SIMPLE: 2.0,
            QueryComplexity.MODERATE: 5.0,
            QueryComplexity.COMPLEX: 15.0,
            QueryComplexity.EXPERT: 30.0
        }
        return time_estimates[complexity]
    
    def _recommend_timeout(self, complexity: QueryComplexity) -> float:
        """Recommend timeout in seconds"""
        return self._estimate_processing_time(complexity) * 3
    
    def _determine_priority(self, features: QueryFeatures) -> str:
        """Determine processing priority"""
        if features.compliance_indicators:
            return "high"
        elif len(features.banking_terms) > 3:
            return "medium"
        else:
            return "normal"
    
    def _suggest_optimizations(self, features: QueryFeatures, complexity: QueryComplexity) -> List[str]:
        """Suggest query optimizations"""
        optimizations = []
        
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]:
            optimizations.append("Consider breaking into smaller sub-queries")
        
        if features.has_aggregation and features.has_filtering:
            optimizations.append("Apply filters before aggregation for better performance")
        
        if len(features.regional_references) > 1:
            optimizations.append("Consider regional partitioning for parallel processing")
        
        if len(features.date_expressions) > 1:
            optimizations.append("Optimize date range queries using indexed columns")
        
        return optimizations
    
    def get_query_patterns(self, queries: List[str]) -> Dict[str, Any]:
        """Analyze patterns across multiple queries"""
        patterns = {
            "complexity_distribution": {complexity.value: 0 for complexity in QueryComplexity},
            "query_type_distribution": {qtype.value: 0 for qtype in QueryType},
            "intent_distribution": {intent.value: 0 for intent in AnalystIntent},
            "common_terms": defaultdict(int),
            "ambiguity_frequency": 0,
            "avg_confidence": 0.0
        }
        
        total_confidence = 0
        
        for query in queries:
            try:
                result = self.analyze_query(query)
                
                patterns["complexity_distribution"][result.complexity.value] += 1
                patterns["query_type_distribution"][result.query_type.value] += 1
                patterns["intent_distribution"][result.analyst_intent.value] += 1
                
                for term in result.features.banking_terms:
                    patterns["common_terms"][term] += 1
                
                if result.ambiguity.is_ambiguous:
                    patterns["ambiguity_frequency"] += 1
                
                total_confidence += result.confidence_score
                
            except Exception as e:
                logger.warning(f"Pattern analysis failed for query: {e}")
        
        if queries:
            patterns["avg_confidence"] = total_confidence / len(queries)
            patterns["ambiguity_rate"] = patterns["ambiguity_frequency"] / len(queries)
        
        return patterns


# Global query analyzer instance
query_analyzer = QueryAnalyzer()
