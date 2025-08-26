"""
Linguistic Processor for Professional Banking Queries
Advanced linguistic analysis and text processing for banking domain
Handles tokenization, POS tagging, dependency parsing, and domain-specific normalization
"""

import logging
import re
import string
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag

from core.data_models import LinguisticAnalysis, Token, SemanticRelation # type: ignore
from config_module import get_config
from utils.metadata_loader import get_metadata_loader
from utils.text_preprocessing import TextPreprocessor


logger = logging.getLogger(__name__)


@dataclass
class Token:
    """Represents a linguistic token with analysis"""
    text: str
    lemma: str
    pos_tag: str
    is_stopword: bool
    is_banking_term: bool
    is_entity: bool
    entity_type: Optional[str] = None
    confidence: float = 1.0
    semantic_role: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class SemanticRelation:
    """Represents semantic relationship between tokens"""
    relation_type: str
    head_token: str
    dependent_token: str
    confidence: float
    context: Optional[str] = None


@dataclass
class LinguisticAnalysis:
    """Complete linguistic analysis result"""
    original_query: str
    normalized_query: str
    tokens: List[Token]
    sentences: List[str]
    semantic_relations: List[SemanticRelation]
    banking_concepts: List[str]
    query_patterns: List[str]
    complexity_score: float
    readability_score: float
    ambiguity_indicators: List[str] = field(default_factory=list)


class BankingLinguisticProcessor:
    """
    Advanced linguistic processor specialized for banking domain queries
    Handles domain-specific normalization, tokenization, and semantic analysis
    """
    
    def __init__(self):
        """Initialize linguistic processor with banking domain knowledge"""
        self.config = get_config()
        self.metadata_loader = get_metadata_loader()
        
        # Initialize NLTK components
        self._initialize_nltk_components()
        
        # Load banking domain knowledge
        self._initialize_banking_vocabulary()
        
        # Initialize linguistic patterns
        self._initialize_linguistic_patterns()
        
        # Text preprocessor
        self.text_preprocessor = TextPreprocessor()
        
        # Stemmer and lemmatizer
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        logger.info("Linguistic processor initialized with banking domain knowledge")
    
    def _initialize_nltk_components(self):
        """Initialize required NLTK components"""
        try:
            # Download required NLTK data
            nltk_downloads = [
                'punkt', 'stopwords', 'averaged_perceptron_tagger',
                'wordnet', 'maxent_ne_chunker', 'words'
            ]
            
            for download in nltk_downloads:
                try:
                    nltk.data.find(f'tokenizers/{download}')
                except LookupError:
                    nltk.download(download, quiet=True)
            
            # Initialize stopwords
            self.stop_words = set(stopwords.words('english'))
            
            # Add banking-specific stopwords
            banking_stopwords = {
                'bank', 'banking', 'financial', 'finance', 'institution',
                'please', 'kindly', 'show', 'display', 'get', 'give'
            }
            self.stop_words.update(banking_stopwords)
            
        except Exception as e:
            logger.warning(f"NLTK initialization warning: {e}")
            # Fallback to basic stopwords
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def _initialize_banking_vocabulary(self):
        """Initialize banking domain vocabulary and terminology"""
        
        # Core banking terms
        self.banking_vocabulary = {
            'entities': {
                'customer', 'counterparty', 'client', 'borrower', 'depositor',
                'applicant', 'account holder', 'guarantor', 'co-borrower'
            },
            'products': {
                'loan', 'credit', 'advance', 'facility', 'mortgage', 'overdraft',
                'deposit', 'account', 'savings', 'current', 'fixed deposit',
                'recurring deposit', 'casa'
            },
            'processes': {
                'application', 'approval', 'disbursement', 'repayment',
                'collection', 'recovery', 'restructuring', 'settlement',
                'closure', 'renewal', 'enhancement'
            },
            'risk_terms': {
                'npa', 'default', 'delinquent', 'overdue', 'provision',
                'write-off', 'recovery', 'restructured', 'standard',
                'sub-standard', 'doubtful', 'loss'
            },
            'financial_metrics': {
                'amount', 'balance', 'limit', 'exposure', 'outstanding',
                'principal', 'interest', 'emi', 'installment', 'tenure',
                'maturity', 'yield', 'spread', 'margin'
            },
            'compliance_terms': {
                'rbi', 'regulatory', 'compliance', 'audit', 'basel',
                'crar', 'provisioning', 'guidelines', 'norms', 'circular'
            },
            'operational_terms': {
                'branch', 'region', 'zone', 'center', 'location',
                'workflow', 'process', 'tracking', 'monitoring',
                'exception', 'deviation', 'sla', 'turnaround'
            }
        }
        
        # Flatten vocabulary for quick lookup
        self.all_banking_terms = set()
        for category, terms in self.banking_vocabulary.items():
            self.all_banking_terms.update(terms)
        
        # Banking abbreviations and their expansions
        self.banking_abbreviations = {
            'npa': 'non performing asset',
            'crar': 'capital to risk weighted assets ratio',
            'casa': 'current account savings account',
            'ctpt': 'counterparty',
            'fac': 'facility',
            'app': 'application',
            'rbi': 'reserve bank of india',
            'emi': 'equated monthly installment',
            'fd': 'fixed deposit',
            'rd': 'recurring deposit',
            'kyc': 'know your customer',
            'aml': 'anti money laundering',
            'cibil': 'credit information bureau india limited',
            'ifrs': 'international financial reporting standards',
            'sarfaesi': 'securitisation and reconstruction of financial assets and enforcement of security interest'
        }
        
        # Indian currency and measurement terms
        self.indian_financial_terms = {
            'lakh', 'lakhs', 'crore', 'crores', 'thousand', 'thousands',
            'rupee', 'rupees', 'rs', 'inr', 'paisa', 'paise'
        }
        
        # Geographic terms specific to India
        self.indian_geographic_terms = {
            'state', 'states', 'district', 'districts', 'city', 'cities',
            'metro', 'urban', 'rural', 'tier1', 'tier2', 'tier3',
            'north', 'south', 'east', 'west', 'central', 'northeast'
        }
    
    def _initialize_linguistic_patterns(self):
        """Initialize linguistic patterns for banking queries"""
        
        # Query structure patterns
        self.query_patterns = {
            'information_seeking': [
                r'^(?:what|who|when|where|which|how)\s+',
                r'\b(?:show|display|get|find|retrieve)\s+',
                r'\b(?:tell|give)\s+me\s+',
                r'\b(?:i\s+(?:want|need))\s+'
            ],
            'analytical': [
                r'\b(?:analyze|analyse|compare|evaluate)\s+',
                r'\b(?:trend|pattern|distribution)\s+',
                r'\b(?:performance|efficiency)\s+',
                r'\b(?:correlation|relationship)\s+'
            ],
            'aggregational': [
                r'\b(?:sum|total|count|average|maximum|minimum)\s+',
                r'\b(?:group\s+by|aggregate)\s+',
                r'\b(?:how\s+(?:much|many))\s+',
                r'\b(?:calculate|compute)\s+'
            ],
            'temporal': [
                r'\b(?:last|past|recent|within)\s+\d+\s+(?:days?|weeks?|months?|years?)',
                r'\b(?:today|yesterday|tomorrow)\b',
                r'\b(?:this|current|previous|next)\s+(?:week|month|quarter|year)',
                r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'
            ],
            'conditional': [
                r'\b(?:where|if|when)\s+',
                r'\b(?:with|having)\s+',
                r'\b(?:greater|less|equal)\s+(?:than|to)\s+',
                r'\b(?:above|below|between)\s+'
            ]
        }
        
        # Semantic role patterns
        self.semantic_roles = {
            'agent': ['customer', 'counterparty', 'applicant', 'borrower'],
            'object': ['loan', 'application', 'facility', 'account'],
            'attribute': ['amount', 'balance', 'status', 'rating'],
            'location': ['branch', 'region', 'state', 'city'],
            'time': ['date', 'period', 'duration', 'time']
        }
        
        # Dependency patterns for banking queries
        self.dependency_patterns = {
            'possession': ['of', 'for', 'belonging to'],
            'location': ['in', 'at', 'from', 'within'],
            'comparison': ['than', 'compared to', 'versus', 'against'],
            'aggregation': ['by', 'per', 'across', 'throughout']
        }
    
    def process(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> LinguisticAnalysis:
        """
        Process query text with comprehensive linguistic analysis
        
        Args:
            query_text: Natural language query from analyst
            context: Optional context from previous processing
            
        Returns:
            Complete linguistic analysis result
        """
        try:
            # Step 1: Text normalization and preprocessing
            normalized_query = self._normalize_query(query_text)
            
            # Step 2: Sentence segmentation
            sentences = self._segment_sentences(normalized_query)
            
            # Step 3: Tokenization with banking context
            tokens = self._tokenize_with_context(normalized_query)
            
            # Step 4: Semantic relation extraction
            semantic_relations = self._extract_semantic_relations(tokens)
            
            # Step 5: Banking concept identification
            banking_concepts = self._identify_banking_concepts(tokens)
            
            # Step 6: Query pattern analysis
            query_patterns = self._analyze_query_patterns(normalized_query)
            
            # Step 7: Complexity assessment
            complexity_score = self._assess_complexity(tokens, semantic_relations)
            
            # Step 8: Readability assessment
            readability_score = self._assess_readability(normalized_query)
            
            # Step 9: Ambiguity detection
            ambiguity_indicators = self._detect_ambiguity(tokens, semantic_relations)
            
            result = LinguisticAnalysis(
                original_query=query_text,
                normalized_query=normalized_query,
                tokens=tokens,
                sentences=sentences,
                semantic_relations=semantic_relations,
                banking_concepts=banking_concepts,
                query_patterns=query_patterns,
                complexity_score=complexity_score,
                readability_score=readability_score,
                ambiguity_indicators=ambiguity_indicators
            )
            
            logger.info(f"Linguistic analysis completed for query with {len(tokens)} tokens")
            return result
            
        except Exception as e:
            logger.error(f"Linguistic processing failed: {e}")
            # Return minimal analysis on failure
            return LinguisticAnalysis(
                original_query=query_text,
                normalized_query=query_text,
                tokens=[],
                sentences=[query_text],
                semantic_relations=[],
                banking_concepts=[],
                query_patterns=[],
                complexity_score=0.5,
                readability_score=0.5,
                ambiguity_indicators=["processing_error"]
            )
    
    def _normalize_query(self, query_text: str) -> str:
        """Normalize query text with banking domain specifics"""
        
        # Basic text cleaning
        normalized = query_text.strip()
        
        # Expand banking abbreviations
        for abbr, expansion in self.banking_abbreviations.items():
            pattern = rf'\b{re.escape(abbr)}\b'
            normalized = re.sub(pattern, expansion, normalized, flags=re.IGNORECASE)
        
        # Normalize Indian currency terms
        currency_patterns = [
            (r'\brs\.?\s*(\d+)', r'rupees \1'),
            (r'₹\s*(\d+)', r'rupees \1'),
            (r'\binr\s*(\d+)', r'rupees \1'),
        ]
        
        for pattern, replacement in currency_patterns:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Normalize numeric scales
        scale_patterns = [
            (r'(\d+)\s*k\b', r'\1 thousand'),
            (r'(\d+)\s*m\b', r'\1 million'),
            (r'(\d+)\s*b\b', r'\1 billion'),
        ]
        
        for pattern, replacement in scale_patterns:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Handle contractions
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            normalized = normalized.replace(contraction, expansion)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences"""
        try:
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except:
            # Fallback: split by common sentence endings
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _tokenize_with_context(self, text: str) -> List[Token]:
        """Tokenize text with banking context and linguistic analysis"""
        
        tokens = []
        
        try:
            # Basic tokenization
            words = word_tokenize(text.lower())
            
            # POS tagging
            pos_tags = pos_tag(words)
            
            for word, pos in pos_tags:
                # Skip punctuation
                if word in string.punctuation:
                    continue
                
                # Lemmatization
                lemma = self.lemmatizer.lemmatize(word, self._get_wordnet_pos(pos))
                
                # Banking term detection
                is_banking_term = (
                    word in self.all_banking_terms or
                    lemma in self.all_banking_terms or
                    word in self.indian_financial_terms or
                    word in self.indian_geographic_terms
                )
                
                # Stopword detection (excluding banking terms)
                is_stopword = word in self.stop_words and not is_banking_term
                
                # Entity detection (basic pattern matching)
                is_entity, entity_type = self._detect_entity_type(word)
                
                # Semantic role assignment
                semantic_role = self._assign_semantic_role(word, pos)
                
                token = Token(
                    text=word,
                    lemma=lemma,
                    pos_tag=pos,
                    is_stopword=is_stopword,
                    is_banking_term=is_banking_term,
                    is_entity=is_entity,
                    entity_type=entity_type,
                    semantic_role=semantic_role
                )
                
                tokens.append(token)
                
        except Exception as e:
            logger.warning(f"Tokenization error: {e}")
            # Fallback tokenization
            words = text.lower().split()
            for word in words:
                if word not in string.punctuation:
                    tokens.append(Token(
                        text=word,
                        lemma=word,
                        pos_tag='NN',  # Default to noun
                        is_stopword=word in self.stop_words,
                        is_banking_term=word in self.all_banking_terms,
                        is_entity=False
                    ))
        
        return tokens
    
    def _get_wordnet_pos(self, pos_tag: str) -> str:
        """Convert POS tag to WordNet format"""
        if pos_tag.startswith('J'):
            return 'a'  # adjective
        elif pos_tag.startswith('V'):
            return 'v'  # verb
        elif pos_tag.startswith('N'):
            return 'n'  # noun
        elif pos_tag.startswith('R'):
            return 'r'  # adverb
        else:
            return 'n'  # default to noun
    
    def _detect_entity_type(self, word: str) -> Tuple[bool, Optional[str]]:
        """Detect if word is an entity and determine its type"""
        
        # ID patterns
        if re.match(r'^[A-Z]{2,}\d+$', word.upper()):
            if word.upper().startswith('CTPT'):
                return True, 'counterparty_id'
            elif word.upper().startswith('APP'):
                return True, 'application_id'
            elif word.upper().startswith('FAC'):
                return True, 'facility_id'
            else:
                return True, 'identifier'
        
        # Numeric patterns
        if re.match(r'^\d+(?:,\d{3})*(?:\.\d{2})?$', word):
            return True, 'amount'
        
        # Date patterns
        if re.match(r'^\d{4}-\d{2}-\d{2}$', word):
            return True, 'date'
        
        # Percentage patterns
        if re.match(r'^\d+(?:\.\d+)?%$', word):
            return True, 'percentage'
        
        return False, None
    
    def _assign_semantic_role(self, word: str, pos_tag: str) -> Optional[str]:
        """Assign semantic role based on word and POS tag"""
        
        for role, terms in self.semantic_roles.items():
            if word in terms:
                return role
        
        # Role assignment based on POS tag
        if pos_tag.startswith('NN'):  # Nouns
            if word in self.banking_vocabulary.get('entities', set()):
                return 'agent'
            elif word in self.banking_vocabulary.get('products', set()):
                return 'object'
            elif word in self.banking_vocabulary.get('financial_metrics', set()):
                return 'attribute'
        
        elif pos_tag.startswith('VB'):  # Verbs
            return 'action'
        
        elif pos_tag.startswith('JJ'):  # Adjectives
            return 'modifier'
        
        elif pos_tag in ['IN', 'TO']:  # Prepositions
            return 'relation'
        
        return None
    
    def _extract_semantic_relations(self, tokens: List[Token]) -> List[SemanticRelation]:
        """Extract semantic relations between tokens"""
        
        relations = []
        
        # Simple dependency-like relation extraction
        for i, token in enumerate(tokens):
            # Look for prepositions and their objects
            if token.pos_tag in ['IN', 'TO'] and i > 0 and i < len(tokens) - 1:
                head_token = tokens[i-1].text
                dependent_token = tokens[i+1].text
                
                relation_type = self._determine_relation_type(token.text)
                
                relations.append(SemanticRelation(
                    relation_type=relation_type,
                    head_token=head_token,
                    dependent_token=dependent_token,
                    confidence=0.8,
                    context=token.text
                ))
            
            # Look for possessive relations
            if "'s" in token.text or token.text == 'of':
                if i > 0 and i < len(tokens) - 1:
                    relations.append(SemanticRelation(
                        relation_type='possession',
                        head_token=tokens[i+1].text,
                        dependent_token=tokens[i-1].text,
                        confidence=0.9,
                        context='possessive'
                    ))
        
        return relations
    
    def _determine_relation_type(self, preposition: str) -> str:
        """Determine semantic relation type from preposition"""
        
        relation_map = {
            'in': 'location',
            'at': 'location',
            'from': 'source',
            'to': 'destination',
            'by': 'agent',
            'with': 'instrument',
            'for': 'beneficiary',
            'of': 'possession',
            'on': 'temporal',
            'during': 'temporal',
            'before': 'temporal',
            'after': 'temporal'
        }
        
        return relation_map.get(preposition.lower(), 'relation')
    
    def _identify_banking_concepts(self, tokens: List[Token]) -> List[str]:
        """Identify banking concepts from tokens"""
        
        concepts = []
        
        # Single token concepts
        for token in tokens:
            if token.is_banking_term:
                concepts.append(token.text)
        
        # Multi-token concepts (bigrams and trigrams)
        token_texts = [t.text for t in tokens]
        
        # Check for compound banking terms
        compound_terms = [
            'non performing asset', 'capital adequacy ratio', 'current account',
            'savings account', 'fixed deposit', 'recurring deposit',
            'loan application', 'credit facility', 'risk assessment',
            'collateral security', 'interest rate', 'defaulted loan'
        ]
        
        query_text = ' '.join(token_texts)
        for term in compound_terms:
            if term in query_text:
                concepts.append(term)
        
        return list(set(concepts))  # Remove duplicates
    
    def _analyze_query_patterns(self, query: str) -> List[str]:
        """Analyze linguistic patterns in query"""
        
        identified_patterns = []
        query_lower = query.lower()
        
        for pattern_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    identified_patterns.append(pattern_type)
                    break  # Only add pattern type once
        
        return identified_patterns
    
    def _assess_complexity(self, tokens: List[Token], relations: List[SemanticRelation]) -> float:
        """Assess linguistic complexity of query"""
        
        complexity_factors = {
            'token_count': len(tokens),
            'banking_terms': len([t for t in tokens if t.is_banking_term]),
            'entities': len([t for t in tokens if t.is_entity]),
            'relations': len(relations),
            'unique_pos_tags': len(set(t.pos_tag for t in tokens))
        }
        
        # Complexity scoring
        score = 0.0
        
        # Token count factor
        if complexity_factors['token_count'] > 20:
            score += 0.3
        elif complexity_factors['token_count'] > 10:
            score += 0.2
        else:
            score += 0.1
        
        # Banking terms factor
        banking_ratio = complexity_factors['banking_terms'] / max(complexity_factors['token_count'], 1)
        score += banking_ratio * 0.2
        
        # Entity factor
        entity_ratio = complexity_factors['entities'] / max(complexity_factors['token_count'], 1)
        score += entity_ratio * 0.2
        
        # Relations factor
        relations_ratio = complexity_factors['relations'] / max(complexity_factors['token_count'], 1)
        score += relations_ratio * 0.2
        
        # POS diversity factor
        pos_diversity = complexity_factors['unique_pos_tags'] / max(complexity_factors['token_count'], 1)
        score += pos_diversity * 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _assess_readability(self, query: str) -> float:
        """Assess readability of query"""
        
        # Simple readability metrics
        words = query.split()
        sentences = len(re.split(r'[.!?]+', query))
        
        if sentences == 0:
            sentences = 1
        
        avg_words_per_sentence = len(words) / sentences
        
        # Simple scoring based on average sentence length
        if avg_words_per_sentence <= 10:
            return 1.0  # High readability
        elif avg_words_per_sentence <= 20:
            return 0.8  # Good readability
        elif avg_words_per_sentence <= 30:
            return 0.6  # Moderate readability
        else:
            return 0.4  # Low readability
    
    def _detect_ambiguity(self, tokens: List[Token], relations: List[SemanticRelation]) -> List[str]:
        """Detect potential ambiguity indicators in query"""
        
        ambiguity_indicators = []
        
        # Check for ambiguous pronouns
        pronouns = ['it', 'this', 'that', 'these', 'those', 'they', 'them']
        for token in tokens:
            if token.text in pronouns:
                ambiguity_indicators.append(f"ambiguous_pronoun:{token.text}")
        
        # Check for vague terms
        vague_terms = ['some', 'any', 'several', 'various', 'different', 'many', 'few']
        for token in tokens:
            if token.text in vague_terms:
                ambiguity_indicators.append(f"vague_quantifier:{token.text}")
        
        # Check for missing entities
        action_tokens = [t for t in tokens if t.pos_tag.startswith('VB')]
        entity_tokens = [t for t in tokens if t.is_entity or t.semantic_role == 'agent']
        
        if action_tokens and not entity_tokens:
            ambiguity_indicators.append("missing_subject")
        
        # Check for multiple possible interpretations
        banking_entities = [t for t in tokens if t.entity_type in ['counterparty_id', 'application_id', 'facility_id']]
        if len(banking_entities) > 2:
            ambiguity_indicators.append("multiple_entities")
        
        return ambiguity_indicators
    
    def get_linguistic_features(self, analysis: LinguisticAnalysis) -> Dict[str, Any]:
        """Extract linguistic features for downstream processing"""
        
        return {
            'token_count': len(analysis.tokens),
            'banking_term_count': len([t for t in analysis.tokens if t.is_banking_term]),
            'entity_count': len([t for t in analysis.tokens if t.is_entity]),
            'relation_count': len(analysis.semantic_relations),
            'complexity_score': analysis.complexity_score,
            'readability_score': analysis.readability_score,
            'query_patterns': analysis.query_patterns,
            'banking_concepts': analysis.banking_concepts,
            'has_ambiguity': len(analysis.ambiguity_indicators) > 0,
            'ambiguity_count': len(analysis.ambiguity_indicators),
            'semantic_roles': list(set(t.semantic_role for t in analysis.tokens if t.semantic_role)),
            'pos_tags': list(set(t.pos_tag for t in analysis.tokens)),
            'entity_types': list(set(t.entity_type for t in analysis.tokens if t.entity_type))
        }


def main():
    """Test linguistic processor functionality"""
    try:
        processor = BankingLinguisticProcessor()
        print("Linguistic processor initialized successfully!")
        
        # Test with sample banking queries
        test_queries = [
            "Show me last 10 days created customers in Maharashtra",
            "What is the NPA ratio for ABC Corporation's loan facilities?",
            "Give me total collateral amount for defaulted applications in Delhi region",
            "Analyze customer acquisition trend over past 6 months",
            "Compare this quarter's disbursement with previous quarter by state",
            "Which counterparties have maximum exposure above 5 crores?"
        ]
        
        print(f"\nTesting linguistic analysis with {len(test_queries)} queries:")
        
        for query in test_queries:
            analysis = processor.process(query)
            features = processor.get_linguistic_features(analysis)
            
            print(f"\nQuery: '{query}'")
            print(f"Normalized: '{analysis.normalized_query}'")
            print(f"Tokens: {len(analysis.tokens)} | Banking terms: {features['banking_term_count']} | Entities: {features['entity_count']}")
            print(f"Complexity: {analysis.complexity_score:.2f} | Readability: {analysis.readability_score:.2f}")
            print(f"Patterns: {analysis.query_patterns}")
            print(f"Banking concepts: {analysis.banking_concepts}")
            if analysis.ambiguity_indicators:
                print(f"Ambiguity indicators: {analysis.ambiguity_indicators}")
            
            # Show sample tokens
            banking_tokens = [t for t in analysis.tokens if t.is_banking_term][:3]
            if banking_tokens:
                print(f"Sample banking tokens: {[f'{t.text}({t.pos_tag})' for t in banking_tokens]}")
    
    except Exception as e:
        print(f"Error testing linguistic processor: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
