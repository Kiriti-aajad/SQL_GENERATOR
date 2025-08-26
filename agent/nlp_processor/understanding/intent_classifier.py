"""
Professional Query Intent Classification
Classifies analyst queries into business intent categories
Handles temporal analysis, regional aggregation, defaulter analysis, etc.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from agent.nlp_processor.core.data_models import IntentResult, QueryType
from agent.nlp_processor.config_module import get_config
from agent.nlp_processor.utils.metadata_loader import get_metadata_loader

logger = logging.getLogger(__name__)

class IntentClassifier:
    """
    Classify professional analyst queries into business intent categories
    Optimized for banking/financial domain queries
    """
    
    def __init__(self):
        """Initialize intent classifier with domain knowledge"""
        self.config = get_config()
        self.metadata_loader = get_metadata_loader()
        
        # Load analyst patterns from configuration
        self.temporal_patterns = self.config.get_analyst_patterns('temporal_analysis')
        self.regional_patterns = self.config.get_analyst_patterns('regional_analysis') 
        self.aggregation_patterns = self.config.get_analyst_patterns('aggregation_analysis')
        
        # Confidence thresholds
        self.intent_threshold = self.config.get('understanding.intent_threshold', 0.7)
        
        # Domain-specific pattern definitions
        self._initialize_domain_patterns()
        
        logger.info("Intent classifier initialized with banking domain patterns")
    
    def _initialize_domain_patterns(self):
        """Initialize banking/financial domain-specific patterns"""
        
        # Temporal analysis patterns
        self.temporal_keywords = {
            'relative_time': [
                r'last\s+(\d+)\s+(days?|weeks?|months?)',
                r'recent',
                r'within\s+(\d+)\s+(days?|weeks?|months?)',
                r'past\s+(\d+)\s+(days?|weeks?|months?)',
                r'yesterday',
                r'today',
                r'this\s+(week|month|year)'
            ],
            'absolute_time': [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
                r'(january|february|march|april|may|june|july|august|september|october|november|december)',
                r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
            ]
        }
        
        # Regional analysis patterns
        self.regional_keywords = {
            'geographic_terms': [
                'region', 'regions', 'state', 'states', 'country', 'countries',
                'location', 'locations', 'geographic', 'geographical',
                'area', 'areas', 'zone', 'zones', 'territory', 'territories'
            ],
            'specific_locations': [
                'delhi', 'mumbai', 'bangalore', 'chennai', 'kolkata', 'hyderabad',
                'pune', 'ahmedabad', 'surat', 'jaipur', 'lucknow', 'kanpur',
                'north', 'south', 'east', 'west', 'central', 'northeast', 'northwest'
            ]
        }
        
        # Aggregation analysis patterns
        self.aggregation_keywords = {
            'sum_functions': ['sum', 'total', 'aggregate', 'combined', 'overall'],
            'avg_functions': ['average', 'avg', 'mean'],
            'max_functions': ['maximum', 'max', 'highest', 'largest', 'biggest', 'peak'],
            'min_functions': ['minimum', 'min', 'lowest', 'smallest', 'least'],
            'count_functions': ['count', 'number', 'how many', 'quantity']
        }
        
        # Business entity patterns
        self.business_entities = {
            'counterparty_terms': [
                'customer', 'customers', 'counterparty', 'counterparties',
                'client', 'clients', 'company', 'companies', 'corp', 'corporation',
                'business', 'businesses', 'entity', 'entities'
            ],
            'application_terms': [
                'application', 'applications', 'loan', 'loans', 'request', 'requests',
                'proposal', 'proposals', 'submission', 'submissions'
            ],
            'financial_terms': [
                'amount', 'value', 'collateral', 'security', 'guarantee',
                'disbursement', 'payment', 'transaction', 'money', 'funds'
            ],
            'workflow_terms': [
                'status', 'stage', 'phase', 'step', 'workflow', 'process',
                'tracking', 'progress', 'deviation', 'deviations', 'exception'
            ]
        }
        
        # Specific analysis patterns
        self.analysis_patterns = {
            'defaulter_analysis': [
                r'defaulter', r'default', r'non.?performing', r'npa', r'bad\s+debt',
                r'overdue', r'delinquent', r'late\s+payment'
            ],
            'deviation_analysis': [
                r'deviation', r'exception', r'anomaly', r'unusual', r'irregular',
                r'non.?standard', r'out\s+of\s+norm'
            ],
            'collateral_analysis': [
                r'collateral', r'security', r'guarantee', r'pledge', r'mortgage',
                r'asset', r'backing'
            ]
        }
    
    def classify(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> IntentResult:
        """
        Classify analyst query into business intent category
        
        Args:
            query_text: Natural language query from analyst
            context: Optional context from previous queries
            
        Returns:
            IntentResult with classified intent and confidence
        """
        query_lower = query_text.lower().strip()
        
        # Analyze different intent dimensions
        temporal_score, temporal_context = self._analyze_temporal_intent(query_lower)
        regional_score = self._analyze_regional_intent(query_lower)
        aggregation_score, aggregation_type = self._analyze_aggregation_intent(query_lower)
        defaulter_score = self._analyze_defaulter_intent(query_lower)
        deviation_score = self._analyze_deviation_intent(query_lower)
        collateral_score = self._analyze_collateral_intent(query_lower)
        
        # Determine primary intent based on highest confidence score
        intent_scores = [
            (QueryType.TEMPORAL_ANALYSIS, temporal_score, {"temporal_context": temporal_context}),
            (QueryType.REGIONAL_AGGREGATION, regional_score, {}),
            (QueryType.DEFAULTER_ANALYSIS, defaulter_score, {}),
            (QueryType.DEVIATION_ANALYSIS, deviation_score, {}),
            (QueryType.COLLATERAL_ANALYSIS, collateral_score, {}),
            (QueryType.CUSTOMER_ANALYSIS, 0.5, {})  # Base customer analysis
        ]
        
        # Sort by confidence score
        intent_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get the highest scoring intent
        primary_intent, confidence, intent_context = intent_scores[0]
        
        # Enhance with aggregation information if detected
        if aggregation_score > 0.6:
            intent_context["aggregation_type"] = aggregation_type
        
        # Suggest target tables based on intent
        target_tables = self._suggest_target_tables(primary_intent, query_lower)
        
        # Create intent result
        result = IntentResult(
            query_type=primary_intent,
            confidence=confidence,
            temporal_context=intent_context.get("temporal_context"),
            aggregation_type=intent_context.get("aggregation_type"),
            target_tables=target_tables
        )
        
        logger.info(f"Classified query intent: {primary_intent.value} (confidence: {confidence:.2f})")
        return result
    
    def _analyze_temporal_intent(self, query: str) -> Tuple[float, Optional[str]]:
        """Analyze temporal analysis intent in query"""
        score = 0.0
        temporal_context = None
        
        # Check for relative time patterns
        for pattern in self.temporal_keywords['relative_time']:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                score += 0.8
                temporal_context = match.group(0)
                break
        
        # Check for absolute time patterns
        for pattern in self.temporal_keywords['absolute_time']:
            if re.search(pattern, query, re.IGNORECASE):
                score += 0.6
                break
        
        # Check for temporal verbs and phrases
        temporal_phrases = [
            'created', 'generated', 'added', 'updated', 'modified',
            'processed', 'submitted', 'approved', 'rejected'
        ]
        
        for phrase in temporal_phrases:
            if phrase in query:
                score += 0.3
                break
        
        return min(score, 1.0), temporal_context
    
    def _analyze_regional_intent(self, query: str) -> float:
        """Analyze regional aggregation intent in query"""
        score = 0.0
        
        # Check for geographic terms
        for term in self.regional_keywords['geographic_terms']:
            if term in query:
                score += 0.7
        
        # Check for specific locations
        for location in self.regional_keywords['specific_locations']:
            if location in query:
                score += 0.8
        
        # Check for grouping words with geographic context
        grouping_words = ['by', 'per', 'across', 'within', 'among']
        has_grouping = any(word in query for word in grouping_words)
        has_geographic = score > 0
        
        if has_grouping and has_geographic:
            score += 0.5
        
        return min(score, 1.0)
    
    def _analyze_aggregation_intent(self, query: str) -> Tuple[float, Optional[str]]:
        """Analyze aggregation analysis intent in query"""
        score = 0.0
        aggregation_type = None
        
        # Check each aggregation type
        for agg_type, keywords in self.aggregation_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    score += 0.8
                    if agg_type.startswith('sum'):
                        aggregation_type = 'SUM'
                    elif agg_type.startswith('avg'):
                        aggregation_type = 'AVG'
                    elif agg_type.startswith('max'):
                        aggregation_type = 'MAX'
                    elif agg_type.startswith('min'):
                        aggregation_type = 'MIN'
                    elif agg_type.startswith('count'):
                        aggregation_type = 'COUNT'
                    break
            if aggregation_type:
                break
        
        return min(score, 1.0), aggregation_type
    
    def _analyze_defaulter_intent(self, query: str) -> float:
        """Analyze defaulter analysis intent in query"""
        score = 0.0
        
        for pattern in self.analysis_patterns['defaulter_analysis']:
            if re.search(pattern, query, re.IGNORECASE):
                score += 0.9
        
        return min(score, 1.0)
    
    def _analyze_deviation_intent(self, query: str) -> float:
        """Analyze deviation analysis intent in query"""
        score = 0.0
        
        for pattern in self.analysis_patterns['deviation_analysis']:
            if re.search(pattern, query, re.IGNORECASE):
                score += 0.9
        
        return min(score, 1.0)
    
    def _analyze_collateral_intent(self, query: str) -> float:
        """Analyze collateral analysis intent in query"""
        score = 0.0
        
        for pattern in self.analysis_patterns['collateral_analysis']:
            if re.search(pattern, query, re.IGNORECASE):
                score += 0.9
        
        return min(score, 1.0)
    
    def _suggest_target_tables(self, intent: QueryType, query: str) -> List[str]:
        """Suggest target database tables based on intent"""
        
        base_tables = ['tblCounterparty']  # Always include counterparty
        
        if intent == QueryType.TEMPORAL_ANALYSIS:
            base_tables.extend([
                'tblOApplicationMaster',
                'tblOSWFActionStatusApplicationTracker'
            ])
        
        elif intent == QueryType.REGIONAL_AGGREGATION:
            base_tables.extend([
                'tblCTPTAddress'
            ])
        
        elif intent == QueryType.DEFAULTER_ANALYSIS:
            base_tables.extend([
                'tblOApplicationMaster',
                'tblCTPTAddress'
            ])
        
        elif intent == QueryType.DEVIATION_ANALYSIS:
            base_tables.extend([
                'tblOApplicationMaster',
                'tblOSWFActionStatusDeviationsTracker'
            ])
        
        elif intent == QueryType.COLLATERAL_ANALYSIS:
            base_tables.extend([
                'tblOApplicationMaster',
                'tblOSWFActionStatusCollateralTracker'
            ])
        
        elif intent == QueryType.CUSTOMER_ANALYSIS:
            base_tables.extend([
                'tblCTPTContactDetails',
                'tblCTPTAddress',
                'tblOApplicationMaster'
            ])
        
        # Add contact and address tables if needed
        needs_contact_info = any(term in query for term in ['contact', 'email', 'phone'])
        if needs_contact_info and 'tblCTPTContactDetails' not in base_tables:
            base_tables.append('tblCTPTContactDetails')
        
        needs_address_info = any(term in query for term in ['address', 'location'])
        if needs_address_info and 'tblCTPTAddress' not in base_tables:
            base_tables.append('tblCTPTAddress')
        
        return base_tables
    
    def get_supported_intents(self) -> List[Dict[str, Any]]:
        """Get list of supported intent types with descriptions"""
        return [
            {
                "intent": QueryType.TEMPORAL_ANALYSIS.value,
                "description": "Time-based analysis queries",
                "examples": ["last 10 days created customers", "recent applications", "this month's data"],
                "confidence_threshold": self.intent_threshold
            },
            {
                "intent": QueryType.REGIONAL_AGGREGATION.value,
                "description": "Geographic and regional analysis",
                "examples": ["customers by region", "applications per state", "regional distribution"],
                "confidence_threshold": self.intent_threshold
            },
            {
                "intent": QueryType.DEFAULTER_ANALYSIS.value,
                "description": "Defaulter and non-performing analysis",
                "examples": ["maximum defaulters", "defaulted applications by region", "bad debt analysis"],
                "confidence_threshold": self.intent_threshold
            },
            {
                "intent": QueryType.DEVIATION_ANALYSIS.value,
                "description": "Deviation and exception analysis",
                "examples": ["deviations for customer", "exception tracking", "unusual patterns"],
                "confidence_threshold": self.intent_threshold
            },
            {
                "intent": QueryType.COLLATERAL_ANALYSIS.value,
                "description": "Collateral and security analysis",
                "examples": ["sum of collateral", "security by customer", "collateral distribution"],
                "confidence_threshold": self.intent_threshold
            },
            {
                "intent": QueryType.CUSTOMER_ANALYSIS.value,
                "description": "General customer and counterparty analysis",
                "examples": ["customer details", "counterparty information", "client overview"],
                "confidence_threshold": self.intent_threshold
            }
        ]

def main():
    """Test intent classifier functionality"""
    try:
        classifier = IntentClassifier()
        print("Intent classifier initialized successfully!")
        
        # Test with sample analyst queries
        test_queries = [
            "Give me last 10 days created customers",
            "Which regions have maximum defaulters", 
            "What is the sum of collateral for ABC Corporation",
            "Show me deviations for XYZ customer",
            "Average loan amount by state",
            "Recent applications in Delhi region"
        ]
        
        print(f"\nTesting with {len(test_queries)} analyst queries:")
        
        for query in test_queries:
            result = classifier.classify(query)
            print(f"\nQuery: '{query}'")
            print(f"Intent: {result.query_type.value}")
            print(f"Confidence: {result.confidence:.2f}")
            if result.temporal_context:
                print(f"Temporal context: {result.temporal_context}")
            if result.aggregation_type:
                print(f"Aggregation type: {result.aggregation_type}")
            print(f"Target tables: {result.target_tables}")
        
        print(f"\nSupported intents:")
        for intent_info in classifier.get_supported_intents():
            print(f"- {intent_info['intent']}: {intent_info['description']}")
        
    except Exception as e:
        print(f"Error testing intent classifier: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
