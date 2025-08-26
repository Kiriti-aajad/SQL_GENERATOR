"""
Semantic Parser for Banking Domain Queries
Creates structured semantic representations from analyzed queries
Handles complex semantic relationships and provides query understanding at the meaning level
FIXED: Updated to handle serialized entity dictionaries instead of BusinessEntity objects
"""

import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from core.data_models import (
    DatabaseField, LinguisticAnalysis,
    BusinessEntity, IntentResult, QueryType
)
from config_module import get_config
from utils.metadata_loader import get_metadata_loader

# Import your existing understanding components
from .intent_classifier import IntentClassifier
from .entity_extractor import EntityExtractor  
from .linguistic_processor import BankingLinguisticProcessor
from .query_analyzer import QueryAnalyzer, AmbiguityType

logger = logging.getLogger(__name__)

class SemanticRelationType(Enum):
    """Types of semantic relationships"""
    SUBJECT_OBJECT = "subject_object"
    ATTRIBUTE = "attribute"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    COMPARATIVE = "comparative"
    AGGREGATION = "aggregation"
    FILTERING = "filtering"
    GROUPING = "grouping"
    ORDERING = "ordering"
    CAUSAL = "causal"
    CONDITIONAL = "conditional"

class ConceptType(Enum):
    """Types of semantic concepts"""
    ENTITY = "entity"
    ACTION = "action"
    ATTRIBUTE = "attribute"
    QUANTIFIER = "quantifier"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CONDITION = "condition"
    MEASURE = "measure"

@dataclass
class SemanticConcept:
    """Represents a semantic concept in the query"""
    concept_id: str
    concept_type: ConceptType
    surface_form: str  # Original text
    canonical_form: str  # Normalized form
    confidence: float
    properties: Dict[str, Any] = field(default_factory=dict)
    database_mapping: Optional[DatabaseField] = None
    constraints: List[str] = field(default_factory=list)

    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert SemanticConcept to dictionary for JSON serialization"""
        return {
            'concept_id': self.concept_id,
            'concept_type': self.concept_type.value if isinstance(self.concept_type, ConceptType) else str(self.concept_type),
            'surface_form': self.surface_form,
            'canonical_form': self.canonical_form,
            'confidence': self.confidence,
            'properties': self.properties,
            'database_mapping': self.database_mapping.to_dict() if self.database_mapping and hasattr(self.database_mapping, 'to_dict') else str(self.database_mapping) if self.database_mapping else None,
            'constraints': self.constraints
        }

@dataclass
class SemanticRelation:
    """Represents a semantic relationship between concepts"""
    relation_id: str
    relation_type: SemanticRelationType
    head_concept: str  # concept_id
    tail_concept: str  # concept_id
    confidence: float
    properties: Dict[str, Any] = field(default_factory=dict)
    linguistic_evidence: List[str] = field(default_factory=list)

    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert SemanticRelation to dictionary for JSON serialization"""
        return {
            'relation_id': self.relation_id,
            'relation_type': self.relation_type.value if isinstance(self.relation_type, SemanticRelationType) else str(self.relation_type),
            'head_concept': self.head_concept,
            'tail_concept': self.tail_concept,
            'confidence': self.confidence,
            'properties': self.properties,
            'linguistic_evidence': self.linguistic_evidence
        }

@dataclass
class SemanticFrame:
    """Represents a semantic frame (structured meaning representation)"""
    frame_id: str
    frame_type: str  # e.g., "query_intent", "aggregation", "comparison"
    core_concepts: List[str]  # concept_ids
    frame_elements: Dict[str, str] = field(default_factory=dict)  # role -> concept_id
    constraints: List[str] = field(default_factory=list)
    confidence: float = 1.0

    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert SemanticFrame to dictionary for JSON serialization"""
        return {
            'frame_id': self.frame_id,
            'frame_type': self.frame_type,
            'core_concepts': self.core_concepts,
            'frame_elements': self.frame_elements,
            'constraints': self.constraints,
            'confidence': self.confidence
        }

@dataclass
class SemanticParseTree:
    """Hierarchical semantic representation"""
    root_concept: str
    concepts: Dict[str, SemanticConcept]
    relations: List[SemanticRelation]
    frames: List[SemanticFrame]
    discourse_structure: Dict[str, Any] = field(default_factory=dict)
    pragmatic_context: Dict[str, Any] = field(default_factory=dict)

    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert SemanticParseTree to dictionary for JSON serialization"""
        return {
            'root_concept': self.root_concept,
            'concepts': {k: v.to_dict() for k, v in self.concepts.items()},
            'relations': [r.to_dict() for r in self.relations],
            'frames': [f.to_dict() for f in self.frames],
            'discourse_structure': self.discourse_structure,
            'pragmatic_context': self.pragmatic_context
        }

@dataclass 
class QuerySemantics:
    """Complete semantic representation of a banking query"""
    original_query: str
    parse_tree: SemanticParseTree
    query_intent_frame: SemanticFrame
    execution_semantics: Dict[str, Any]
    business_semantics: Dict[str, Any]
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    ambiguity_resolutions: List[str] = field(default_factory=list)
    semantic_validation: Dict[str, Any] = field(default_factory=dict)

    # ADDED: JSON Serialization Support
    def to_dict(self):
        """Convert QuerySemantics to dictionary for JSON serialization"""
        return {
            'original_query': self.original_query,
            'parse_tree': self.parse_tree.to_dict() if self.parse_tree else None,
            'query_intent_frame': self.query_intent_frame.to_dict() if self.query_intent_frame else None,
            'execution_semantics': self.execution_semantics,
            'business_semantics': self.business_semantics,
            'confidence_scores': self.confidence_scores,
            'ambiguity_resolutions': self.ambiguity_resolutions,
            'semantic_validation': self.semantic_validation
        }

class BankingSemanticGrammar:
    """
    Banking domain semantic grammar for parsing banking queries
    """
    
    def __init__(self):
        """Initialize banking semantic grammar"""
        
        # Banking semantic roles
        self.semantic_roles = {
            'CUSTOMER': ['customer', 'counterparty', 'client', 'borrower', 'depositor'],
            'LOAN': ['loan', 'credit', 'advance', 'facility', 'lending'],
            'AMOUNT': ['amount', 'balance', 'value', 'sum', 'total'],
            'TIME': ['date', 'period', 'duration', 'time', 'when'],
            'LOCATION': ['branch', 'region', 'state', 'city', 'location'],
            'STATUS': ['status', 'state', 'condition', 'stage'],
            'PRODUCT': ['product', 'scheme', 'plan', 'service'],
            'RISK': ['risk', 'rating', 'grade', 'score'],
            'COLLATERAL': ['collateral', 'security', 'guarantee', 'pledge']
        }
        
        # Semantic frames for banking operations
        self.banking_frames = {
            'CUSTOMER_INQUIRY': {
                'core_elements': ['CUSTOMER', 'ATTRIBUTE'],
                'optional_elements': ['TIME', 'LOCATION', 'CONDITION'],
                'constraints': ['CUSTOMER must be specified', 'ATTRIBUTE must be queryable']
            },
            'LOAN_ANALYSIS': {
                'core_elements': ['LOAN', 'MEASURE'],
                'optional_elements': ['CUSTOMER', 'TIME', 'LOCATION', 'CONDITION'],
                'constraints': ['MEASURE must be aggregatable']
            },
            'RISK_ASSESSMENT': {
                'core_elements': ['CUSTOMER', 'RISK'],
                'optional_elements': ['TIME', 'LOAN', 'AMOUNT'],
                'constraints': ['RISK must be evaluable']
            },
            'PORTFOLIO_AGGREGATION': {
                'core_elements': ['MEASURE', 'AGGREGATION_FUNCTION'],
                'optional_elements': ['GROUPING', 'TIME', 'LOCATION'],
                'constraints': ['AGGREGATION_FUNCTION must be valid']
            },
            'TEMPORAL_COMPARISON': {
                'core_elements': ['MEASURE', 'TIME_PERIOD_1', 'TIME_PERIOD_2'],
                'optional_elements': ['CUSTOMER', 'LOCATION'],
                'constraints': ['Time periods must not overlap']
            }
        }
        
        # Action-to-SQL mappings
        self.action_mappings = {
            'show': 'SELECT',
            'count': 'COUNT',
            'sum': 'SUM', 
            'average': 'AVG',
            'maximum': 'MAX',
            'minimum': 'MIN',
            'compare': 'COMPARE',
            'analyze': 'ANALYZE',
            'group': 'GROUP BY',
            'filter': 'WHERE',
            'order': 'ORDER BY'
        }
        
        # Semantic constraints
        self.semantic_constraints = {
            'temporal_consistency': [
                'start_date < end_date',
                'current_date >= query_date',
                'financial_year_alignment'
            ],
            'business_logic': [
                'loan_amount > 0',
                'interest_rate >= 0',
                'customer_exists',
                'valid_product_codes'
            ],
            'data_integrity': [
                'referential_integrity',
                'null_constraints',
                'valid_enumerations'
            ]
        }

class SemanticParser:
    """
    Advanced semantic parser for banking domain queries
    Creates structured semantic representations from analyzed queries
    """
    
    def __init__(self):
        """Initialize semantic parser with banking domain knowledge"""
        self.config = get_config()
        self.metadata_loader = get_metadata_loader()
        
        # Initialize understanding components
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.linguistic_processor = BankingLinguisticProcessor()
        self.query_analyzer = QueryAnalyzer()
        
        # Initialize semantic grammar
        self.grammar = BankingSemanticGrammar()
        
        # Semantic parsing configuration
        self.parsing_config = self._initialize_parsing_config()
        
        # Concept ID counter for unique IDs
        self._concept_counter = 0
        self._relation_counter = 0
        self._frame_counter = 0
        
        logger.info("Semantic parser initialized with banking domain knowledge")
    
    def _initialize_parsing_config(self) -> Dict[str, Any]:
        """Initialize semantic parsing configuration"""
        return {
            'enable_discourse_analysis': True,
            'enable_pragmatic_inference': True,
            'enable_semantic_validation': True,
            'confidence_threshold': 0.6,
            'max_concepts_per_query': 50,
            'enable_concept_disambiguation': True,
            'enable_frame_inference': True,
            'semantic_similarity_threshold': 0.8
        }
    
    def parse_query_semantics(self, query_text: str, 
                            context: Optional[Dict[str, Any]] = None) -> QuerySemantics:
        """
        Parse query into comprehensive semantic representation
        
        Args:
            query_text: Natural language banking query
            context: Optional context from previous queries
            
        Returns:
            Complete semantic representation
        """
        try:
            # Step 1: Basic NLP analysis using existing components
            intent_result = self.intent_classifier.classify(query_text, context)
            entities = self.entity_extractor.extract(query_text, context)  # Now returns dictionaries
            linguistic_analysis = self.linguistic_processor.process(query_text, context)
            query_analysis = self.query_analyzer.analyze_query(query_text, context)
            
            # Step 2: Extract semantic concepts
            concepts = self._extract_semantic_concepts(
                query_text, entities, linguistic_analysis, query_analysis # pyright: ignore[reportArgumentType]
            )
            
            # Step 3: Identify semantic relations
            relations = self._identify_semantic_relations(
                concepts, linguistic_analysis, query_analysis # pyright: ignore[reportArgumentType]
            )
            
            # Step 4: Build semantic frames
            frames = self._build_semantic_frames(
                concepts, relations, intent_result, query_analysis
            )
            
            # Step 5: Construct parse tree
            parse_tree = self._construct_parse_tree(concepts, relations, frames)
            
            # Step 6: Identify query intent frame
            query_intent_frame = self._identify_query_intent_frame(
                frames, intent_result, query_analysis
            )
            
            # Step 7: Generate execution semantics
            execution_semantics = self._generate_execution_semantics(
                parse_tree, query_intent_frame, entities
            )
            
            # Step 8: Extract business semantics
            business_semantics = self._extract_business_semantics(
                parse_tree, query_analysis, intent_result
            )
            
            # Step 9: Calculate confidence scores
            confidence_scores = self._calculate_semantic_confidence(
                concepts, relations, frames, linguistic_analysis # pyright: ignore[reportArgumentType]
            )
            
            # Step 10: Resolve ambiguities
            ambiguity_resolutions = self._resolve_semantic_ambiguities(
                parse_tree, query_analysis.ambiguity
            )
            
            # Step 11: Validate semantics
            semantic_validation = self._validate_semantic_consistency(
                parse_tree, execution_semantics, business_semantics
            )
            
            result = QuerySemantics(
                original_query=query_text,
                parse_tree=parse_tree,
                query_intent_frame=query_intent_frame,
                execution_semantics=execution_semantics,
                business_semantics=business_semantics,
                confidence_scores=confidence_scores,
                ambiguity_resolutions=ambiguity_resolutions,
                semantic_validation=semantic_validation
            )
            
            logger.info(f"Semantic parsing completed with {len(concepts)} concepts and {len(relations)} relations")
            return result
            
        except Exception as e:
            logger.error(f"Semantic parsing failed: {e}")
            return self._create_fallback_semantics(query_text, str(e))
    
    # CRITICAL FIX: Updated to handle dictionary entities instead of BusinessEntity objects
    def _extract_semantic_concepts(self, query_text: str, entities: List[Dict[str, Any]],
                                 linguistic_analysis: Optional[LinguisticAnalysis],
                                 query_analysis) -> Dict[str, SemanticConcept]:
        """Extract semantic concepts from query"""
        
        concepts = {}
        
        # Extract concepts from entities (now dictionaries)
        for entity_dict in entities:
            concept_id = self._generate_concept_id()
            concept_type = self._map_entity_to_concept_type(entity_dict.get('entity_type', ''))
            
            # Handle field_mapping which might be a dict or None
            field_mapping = None
            if entity_dict.get('field_mapping'):
                if isinstance(entity_dict['field_mapping'], dict):
                    # Convert dict back to DatabaseField if needed
                    field_mapping = None  # Simplified for now
                else:
                    field_mapping = entity_dict['field_mapping']
            
            concept = SemanticConcept(
                concept_id=concept_id,
                concept_type=concept_type,
                surface_form=entity_dict.get('entity_value', ''),
                canonical_form=self._canonicalize_entity_dict(entity_dict),
                confidence=entity_dict.get('confidence', 0.5),
                database_mapping=field_mapping,
                properties={
                    'entity_type': entity_dict.get('entity_type', ''),
                    'table_mapping': entity_dict.get('table_mapping', '')
                }
            )
            concepts[concept_id] = concept
        
        # Extract concepts from linguistic analysis
        if linguistic_analysis:
            for token in linguistic_analysis.tokens:
                if token.is_banking_term or token.is_entity:
                    concept_id = self._generate_concept_id()
                    concept_type = self._infer_concept_type_from_token(token)
                    
                    concept = SemanticConcept(
                        concept_id=concept_id,
                        concept_type=concept_type,
                        surface_form=token.text,
                        canonical_form=token.lemma,
                        confidence=token.confidence,
                        properties={
                            'pos_tag': token.pos_tag,
                            'semantic_role': token.semantic_role,
                            'is_banking_term': token.is_banking_term
                        }
                    )
                    concepts[concept_id] = concept
        
        # Extract action concepts from query analysis
        for term in query_analysis.features.analyst_terminology:
            if term not in [c.surface_form for c in concepts.values()]:
                concept_id = self._generate_concept_id()
                
                concept = SemanticConcept(
                    concept_id=concept_id,
                    concept_type=ConceptType.ACTION,
                    surface_form=term,
                    canonical_form=term,
                    confidence=0.8,
                    properties={
                        'analyst_term': True,
                        'domain': 'banking'
                    }
                )
                concepts[concept_id] = concept
        
        # Extract measure concepts from aggregation features
        if query_analysis.features.has_aggregation:
            aggregation_terms = ['sum', 'count', 'average', 'maximum', 'minimum']
            for term in aggregation_terms:
                if term in query_text.lower():
                    concept_id = self._generate_concept_id()
                    
                    concept = SemanticConcept(
                        concept_id=concept_id,
                        concept_type=ConceptType.MEASURE,
                        surface_form=term,
                        canonical_form=term.upper(),
                        confidence=0.9,
                        properties={
                            'aggregation_function': True,
                            'sql_mapping': self.grammar.action_mappings.get(term, term.upper())
                        }
                    )
                    concepts[concept_id] = concept
        
        return concepts
    
    def _identify_semantic_relations(self, concepts: Dict[str, SemanticConcept],
                                   linguistic_analysis: Optional[LinguisticAnalysis],
                                   query_analysis) -> List[SemanticRelation]:
        """Identify semantic relations between concepts"""
        
        relations = []
        
        # Extract relations from linguistic analysis
        if linguistic_analysis:
            for relation in linguistic_analysis.semantic_relations:
                head_concept = self._find_concept_by_surface_form(
                    concepts, relation.head_token
                )
                tail_concept = self._find_concept_by_surface_form(
                    concepts, relation.dependent_token
                )
                
                if head_concept and tail_concept:
                    relation_id = self._generate_relation_id()
                    relation_type = self._map_linguistic_to_semantic_relation(
                        relation.relation_type
                    )
                    
                    semantic_relation = SemanticRelation(
                        relation_id=relation_id,
                        relation_type=relation_type,
                        head_concept=head_concept,
                        tail_concept=tail_concept,
                        confidence=relation.confidence,
                        linguistic_evidence=[relation.context] if relation.context else []
                    )
                    relations.append(semantic_relation)
        
        # Infer relations from query structure
        if query_analysis.features.has_aggregation:
            # Find aggregation function and measure concepts
            agg_concepts = [cid for cid, c in concepts.items() 
                          if c.concept_type == ConceptType.MEASURE]
            measure_concepts = [cid for cid, c in concepts.items() 
                              if c.concept_type == ConceptType.ATTRIBUTE]
            
            for agg_concept in agg_concepts:
                for measure_concept in measure_concepts:
                    relation_id = self._generate_relation_id()
                    
                    relation = SemanticRelation(
                        relation_id=relation_id,
                        relation_type=SemanticRelationType.AGGREGATION,
                        head_concept=agg_concept,
                        tail_concept=measure_concept,
                        confidence=0.8,
                        properties={'inferred_from': 'aggregation_structure'}
                    )
                    relations.append(relation)
        
        # Infer temporal relations
        temporal_concepts = [cid for cid, c in concepts.items() 
                           if c.concept_type == ConceptType.TEMPORAL]
        entity_concepts = [cid for cid, c in concepts.items() 
                         if c.concept_type == ConceptType.ENTITY]
        
        for temporal_concept in temporal_concepts:
            for entity_concept in entity_concepts:
                relation_id = self._generate_relation_id()
                
                relation = SemanticRelation(
                    relation_id=relation_id,
                    relation_type=SemanticRelationType.TEMPORAL,
                    head_concept=entity_concept,
                    tail_concept=temporal_concept,
                    confidence=0.7,
                    properties={'inferred_from': 'temporal_structure'}
                )
                relations.append(relation)
        
        return relations
    
    def _build_semantic_frames(self, concepts: Dict[str, SemanticConcept],
                             relations: List[SemanticRelation],
                             intent_result: IntentResult,
                             query_analysis) -> List[SemanticFrame]:
        """Build semantic frames from concepts and relations"""
        
        frames = []
        
        # Build query intent frame
        intent_frame = self._build_intent_frame(
            concepts, relations, intent_result, query_analysis
        )
        frames.append(intent_frame)
        
        # Build aggregation frames
        if query_analysis.features.has_aggregation:
            agg_frame = self._build_aggregation_frame(concepts, relations)
            if agg_frame:
                frames.append(agg_frame)
        
        # Build comparison frames
        if query_analysis.features.has_comparison:
            comp_frame = self._build_comparison_frame(concepts, relations)
            if comp_frame:
                frames.append(comp_frame)
        
        # Build temporal frames
        temporal_concepts = [cid for cid, c in concepts.items() 
                           if c.concept_type == ConceptType.TEMPORAL]
        if temporal_concepts:
            temporal_frame = self._build_temporal_frame(concepts, relations, temporal_concepts)
            frames.append(temporal_frame)
        
        return frames
    
    def _build_intent_frame(self, concepts: Dict[str, SemanticConcept],
                          relations: List[SemanticRelation],
                          intent_result: IntentResult,
                          query_analysis) -> SemanticFrame:
        """Build the main query intent frame"""
        
        frame_id = self._generate_frame_id()
        
        # FIXED: Handle QueryType enum serialization
        query_type_value = intent_result.query_type.value if hasattr(intent_result.query_type, 'value') else str(intent_result.query_type)
        frame_type = f"query_intent_{query_type_value}"
        
        # Identify core concepts based on intent
        core_concepts = []
        frame_elements = {}
        
        if intent_result.query_type == QueryType.CUSTOMER_ANALYSIS:
            customer_concepts = [cid for cid, c in concepts.items() 
                               if 'customer' in c.canonical_form.lower() or 
                                  'counterparty' in c.canonical_form.lower()]
            core_concepts.extend(customer_concepts)
            if customer_concepts:
                frame_elements['TARGET'] = customer_concepts[0]
        
        elif intent_result.query_type == QueryType.TEMPORAL_ANALYSIS:
            temporal_concepts = [cid for cid, c in concepts.items() 
                               if c.concept_type == ConceptType.TEMPORAL]
            core_concepts.extend(temporal_concepts)
            if temporal_concepts:
                frame_elements['TIME_SCOPE'] = temporal_concepts[0]
        
        elif intent_result.query_type == QueryType.REGIONAL_AGGREGATION:
            spatial_concepts = [cid for cid, c in concepts.items() 
                              if c.concept_type == ConceptType.SPATIAL]
            core_concepts.extend(spatial_concepts)
            if spatial_concepts:
                frame_elements['REGION'] = spatial_concepts[0]
        
        # Add measure concepts for all query types
        measure_concepts = [cid for cid, c in concepts.items() 
                          if c.concept_type in [ConceptType.MEASURE, ConceptType.ATTRIBUTE]]
        core_concepts.extend(measure_concepts[:2])  # Limit to 2 main measures
        
        if measure_concepts:
            frame_elements['MEASURE'] = measure_concepts[0]
        
        return SemanticFrame(
            frame_id=frame_id,
            frame_type=frame_type,
            core_concepts=core_concepts,
            frame_elements=frame_elements,
            confidence=intent_result.confidence
        )
    
    def _build_aggregation_frame(self, concepts: Dict[str, SemanticConcept],
                               relations: List[SemanticRelation]) -> Optional[SemanticFrame]:
        """Build aggregation semantic frame"""
        
        agg_concepts = [cid for cid, c in concepts.items() 
                       if c.concept_type == ConceptType.MEASURE and 
                          c.properties.get('aggregation_function', False)]
        
        if not agg_concepts:
            return None
        
        frame_id = self._generate_frame_id()
        
        # Find what's being aggregated
        measure_concepts = []
        for relation in relations:
            if (relation.relation_type == SemanticRelationType.AGGREGATION and
                relation.head_concept in agg_concepts):
                measure_concepts.append(relation.tail_concept)
        
        frame_elements = {
            'FUNCTION': agg_concepts[0],
            'MEASURE': measure_concepts[0] if measure_concepts else '',
        }
        
        return SemanticFrame(
            frame_id=frame_id,
            frame_type="aggregation",
            core_concepts=agg_concepts + measure_concepts,
            frame_elements=frame_elements,
            confidence=0.8
        )
    
    def _build_comparison_frame(self, concepts: Dict[str, SemanticConcept],
                              relations: List[SemanticRelation]) -> Optional[SemanticFrame]:
        """Build comparison semantic frame"""
        
        comparison_indicators = ['compare', 'versus', 'vs', 'against', 'higher', 'lower']
        
        comparison_concepts = [cid for cid, c in concepts.items() 
                             if any(indicator in c.surface_form.lower() 
                                   for indicator in comparison_indicators)]
        
        if not comparison_concepts:
            return None
        
        frame_id = self._generate_frame_id()
        
        # Find entities being compared
        entity_concepts = [cid for cid, c in concepts.items() 
                         if c.concept_type == ConceptType.ENTITY]
        
        temporal_concepts = [cid for cid, c in concepts.items() 
                           if c.concept_type == ConceptType.TEMPORAL]
        
        frame_elements = {
            'COMPARISON_TYPE': comparison_concepts[0] if comparison_concepts else '',
            'ENTITY_1': entity_concepts[0] if len(entity_concepts) > 0 else '',
            'ENTITY_2': entity_concepts[1] if len(entity_concepts) > 1 else '',
            'TIME_1': temporal_concepts[0] if len(temporal_concepts) > 0 else '',
            'TIME_2': temporal_concepts[1] if len(temporal_concepts) > 1 else ''
        }
        
        return SemanticFrame(
            frame_id=frame_id,
            frame_type="comparison",
            core_concepts=comparison_concepts + entity_concepts + temporal_concepts,
            frame_elements=frame_elements,
            confidence=0.7
        )
    
    def _build_temporal_frame(self, concepts: Dict[str, SemanticConcept],
                            relations: List[SemanticRelation],
                            temporal_concepts: List[str]) -> SemanticFrame:
        """Build temporal semantic frame"""
        
        frame_id = self._generate_frame_id()
        
        # Identify temporal frame elements
        frame_elements = {}
        
        if temporal_concepts:
            # Classify temporal concepts
            for concept_id in temporal_concepts:
                concept = concepts[concept_id]
                if 'last' in concept.surface_form.lower() or 'past' in concept.surface_form.lower():
                    frame_elements['TIME_REFERENCE'] = concept_id
                elif any(period in concept.surface_form.lower() 
                        for period in ['days', 'weeks', 'months', 'years']):
                    frame_elements['TIME_DURATION'] = concept_id
                elif re.search(r'\d{4}', concept.surface_form):
                    frame_elements['ABSOLUTE_TIME'] = concept_id
        
        return SemanticFrame(
            frame_id=frame_id,
            frame_type="temporal",
            core_concepts=temporal_concepts,
            frame_elements=frame_elements,
            confidence=0.8
        )
    
    def _construct_parse_tree(self, concepts: Dict[str, SemanticConcept],
                            relations: List[SemanticRelation],
                            frames: List[SemanticFrame]) -> SemanticParseTree:
        """Construct hierarchical semantic parse tree"""
        
        # Find root concept (usually the main action or intent)
        root_concept = self._identify_root_concept(concepts, relations, frames)
        
        # Build discourse structure
        discourse_structure = {
            'query_focus': root_concept,
            'information_structure': self._analyze_information_structure(concepts, relations),
            'coherence_relations': self._identify_coherence_relations(relations)
        }
        
        # Build pragmatic context
        pragmatic_context = {
            'query_purpose': 'information_seeking',  # Could be inferred from analysis
            'user_assumptions': [],
            'domain_context': 'banking',
            'interaction_context': {}
        }
        
        return SemanticParseTree(
            root_concept=root_concept,
            concepts=concepts,
            relations=relations,
            frames=frames,
            discourse_structure=discourse_structure,
            pragmatic_context=pragmatic_context
        )
    
    def _identify_query_intent_frame(self, frames: List[SemanticFrame],
                                   intent_result: IntentResult,
                                   query_analysis) -> SemanticFrame:
        """Identify the main query intent frame"""
        
        # Look for query intent frame
        for frame in frames:
            if frame.frame_type.startswith('query_intent'):
                return frame
        
        # Fallback: create basic intent frame
        frame_id = self._generate_frame_id()
        query_type_value = intent_result.query_type.value if hasattr(intent_result.query_type, 'value') else str(intent_result.query_type)
        return SemanticFrame(
            frame_id=frame_id,
            frame_type=f"query_intent_{query_type_value}",
            core_concepts=[],
            confidence=intent_result.confidence
        )
    
    # FIXED: Updated to handle dictionary entities
    def _generate_execution_semantics(self, parse_tree: SemanticParseTree,
                                    query_intent_frame: SemanticFrame,
                                    entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate execution semantics for SQL generation"""
        
        execution_semantics = {
            'query_type': 'SELECT',  # Default
            'select_elements': [],
            'from_tables': set(),
            'join_conditions': [],
            'where_conditions': [],
            'group_by': [],
            'having_conditions': [],
            'order_by': [],
            'aggregations': [],
            'temporal_filters': [],
            'spatial_filters': []
        }
        
        # Analyze concepts for SQL elements
        for concept_id, concept in parse_tree.concepts.items():
            if concept.database_mapping:
                if concept.concept_type == ConceptType.MEASURE:
                    if concept.properties.get('aggregation_function'):
                        agg_func = concept.properties.get('sql_mapping', 'COUNT')
                        field_name = concept.database_mapping.name
                        execution_semantics['aggregations'].append(f"{agg_func}({field_name})")
                        execution_semantics['select_elements'].append(f"{agg_func}({field_name})")
                    else:
                        execution_semantics['select_elements'].append(concept.database_mapping.name)
                
                elif concept.concept_type == ConceptType.ENTITY:
                    # Add to WHERE conditions if it's a filter
                    if hasattr(concept, 'filter_value'):
                        condition = f"{concept.database_mapping.name} = '{concept.canonical_form}'"
                        execution_semantics['where_conditions'].append(condition)
                
                elif concept.concept_type == ConceptType.TEMPORAL:
                    # Add temporal filters
                    if concept.database_mapping.temporal:
                        temporal_filter = self._generate_temporal_filter(concept)
                        if temporal_filter:
                            execution_semantics['temporal_filters'].append(temporal_filter)
                
                # Add table to FROM clause
                if concept.database_mapping.table:
                    execution_semantics['from_tables'].add(concept.database_mapping.table)
        
        # Add tables from entities (now dictionaries)
        for entity_dict in entities:
            if entity_dict.get('table_mapping'):
                execution_semantics['from_tables'].add(entity_dict['table_mapping'])
            
            # Handle field_mapping
            field_mapping = entity_dict.get('field_mapping')
            if field_mapping:
                if isinstance(field_mapping, dict) and field_mapping.get('table'):
                    execution_semantics['from_tables'].add(field_mapping['table'])
                elif hasattr(field_mapping, 'table') and field_mapping.table:
                    execution_semantics['from_tables'].add(field_mapping.table)
        
        # Convert sets to lists for JSON serialization
        execution_semantics['from_tables'] = list(execution_semantics['from_tables'])
        
        return execution_semantics
    
    def _extract_business_semantics(self, parse_tree: SemanticParseTree,
                                  query_analysis,
                                  intent_result: IntentResult) -> Dict[str, Any]:
        """Extract business-level semantic information"""
        
        # FIXED: Handle QueryType enum serialization
        query_type_value = intent_result.query_type.value if hasattr(intent_result.query_type, 'value') else str(intent_result.query_type)
        analyst_intent_value = query_analysis.analyst_intent.value if hasattr(query_analysis.analyst_intent, 'value') else str(query_analysis.analyst_intent)
        complexity_value = query_analysis.complexity.value if hasattr(query_analysis.complexity, 'value') else str(query_analysis.complexity)
        
        business_semantics = {
            'business_domain': 'banking',
            'analysis_type': query_type_value,
            'analyst_intent': analyst_intent_value,
            'complexity_level': complexity_value,
            'business_entities': [],
            'key_metrics': [],
            'business_rules': [],
            'regulatory_context': [],
            'risk_indicators': [],
            'performance_indicators': []
        }
        
        # Extract business entities
        for concept_id, concept in parse_tree.concepts.items():
            if concept.concept_type == ConceptType.ENTITY:
                business_semantics['business_entities'].append({
                    'entity_type': concept.properties.get('entity_type', 'unknown'),
                    'canonical_form': concept.canonical_form,
                    'confidence': concept.confidence
                })
        
        # Extract key metrics
        for concept_id, concept in parse_tree.concepts.items():
            if concept.concept_type in [ConceptType.MEASURE, ConceptType.ATTRIBUTE]:
                business_semantics['key_metrics'].append({
                    'metric_name': concept.canonical_form,
                    'metric_type': concept.concept_type.value,
                    'confidence': concept.confidence
                })
        
        # Identify business rules from frames
        for frame in parse_tree.frames:
            if frame.frame_type == 'aggregation':
                business_semantics['business_rules'].append('aggregation_required')
            elif frame.frame_type == 'comparison':
                business_semantics['business_rules'].append('comparison_analysis')
            elif frame.frame_type == 'temporal':
                business_semantics['business_rules'].append('temporal_analysis')
        
        # Risk indicators
        risk_terms = ['risk', 'default', 'npa', 'provision', 'exposure']
        for concept_id, concept in parse_tree.concepts.items():
            if any(risk_term in concept.canonical_form.lower() for risk_term in risk_terms):
                business_semantics['risk_indicators'].append(concept.canonical_form)
        
        # Performance indicators
        performance_terms = ['performance', 'efficiency', 'roi', 'profit', 'growth']
        for concept_id, concept in parse_tree.concepts.items():
            if any(perf_term in concept.canonical_form.lower() for perf_term in performance_terms):
                business_semantics['performance_indicators'].append(concept.canonical_form)
        
        return business_semantics
    
    def _calculate_semantic_confidence(self, concepts: Dict[str, SemanticConcept],
                                     relations: List[SemanticRelation],
                                     frames: List[SemanticFrame],
                                     linguistic_analysis: Optional[LinguisticAnalysis]) -> Dict[str, float]:
        """Calculate confidence scores for semantic parsing"""
        
        # Concept confidence (average)
        concept_confidence = 0.0
        if concepts:
            concept_confidence = sum(c.confidence for c in concepts.values()) / len(concepts)
        
        # Relation confidence (average)
        relation_confidence = 0.0
        if relations:
            relation_confidence = sum(r.confidence for r in relations) / len(relations)
        
        # Frame confidence (average)
        frame_confidence = 0.0
        if frames:
            frame_confidence = sum(f.confidence for f in frames) / len(frames)
        
        # Linguistic confidence
        linguistic_confidence = 0.8  # Default
        if linguistic_analysis:
            linguistic_confidence = 1.0 - (linguistic_analysis.complexity_score * 0.2)
        
        # Overall confidence (weighted average)
        overall_confidence = (
            concept_confidence * 0.3 +
            relation_confidence * 0.2 +
            frame_confidence * 0.3 +
            linguistic_confidence * 0.2
        )
        
        return {
            'overall': overall_confidence,
            'concepts': concept_confidence,
            'relations': relation_confidence,
            'frames': frame_confidence,
            'linguistic': linguistic_confidence
        }
    
    def _resolve_semantic_ambiguities(self, parse_tree: SemanticParseTree,
                                    ambiguity_detection) -> List[str]:
        """Resolve semantic ambiguities using context and domain knowledge"""
        
        resolutions = []
        
        if ambiguity_detection.is_ambiguous:
            for ambiguity_type in ambiguity_detection.ambiguity_types:
                if ambiguity_type == AmbiguityType.TEMPORAL:
                    resolutions.append("Resolved temporal ambiguity using financial year context")
                elif ambiguity_type == AmbiguityType.ENTITY:
                    resolutions.append("Resolved entity ambiguity using banking domain knowledge")
                elif ambiguity_type == AmbiguityType.SCOPE:
                    resolutions.append("Inferred analysis scope from query structure")
                elif ambiguity_type == AmbiguityType.METRIC:
                    resolutions.append("Disambiguated metrics using aggregation context")
        
        return resolutions
    
    def _validate_semantic_consistency(self, parse_tree: SemanticParseTree,
                                     execution_semantics: Dict[str, Any],
                                     business_semantics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate semantic consistency and coherence"""
        
        validation = {
            'is_consistent': True,
            'consistency_score': 1.0,
            'violations': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Check temporal consistency
        temporal_concepts = [c for c in parse_tree.concepts.values() 
                           if c.concept_type == ConceptType.TEMPORAL]
        if len(temporal_concepts) > 1:
            # Check for temporal contradictions
            validation['warnings'].append("Multiple temporal references detected")
        
        # Check business rule consistency
        if 'aggregation_required' in business_semantics.get('business_rules', []):
            if not execution_semantics.get('aggregations'):
                validation['violations'].append("Aggregation required but not found in execution plan")
                validation['is_consistent'] = False
        
        # Check entity-attribute consistency
        entities = [c for c in parse_tree.concepts.values() 
                   if c.concept_type == ConceptType.ENTITY]
        attributes = [c for c in parse_tree.concepts.values() 
                     if c.concept_type == ConceptType.ATTRIBUTE]
        
        if attributes and not entities:
            validation['warnings'].append("Attributes specified without clear entity context")
        
        # Calculate consistency score
        violation_penalty = len(validation['violations']) * 0.2
        warning_penalty = len(validation['warnings']) * 0.1
        validation['consistency_score'] = max(0.0, 1.0 - violation_penalty - warning_penalty)
        
        return validation
    
    # Utility methods
    
    def _generate_concept_id(self) -> str:
        """Generate unique concept ID"""
        self._concept_counter += 1
        return f"concept_{self._concept_counter:04d}"
    
    def _generate_relation_id(self) -> str:
        """Generate unique relation ID"""
        self._relation_counter += 1
        return f"relation_{self._relation_counter:04d}"
    
    def _generate_frame_id(self) -> str:
        """Generate unique frame ID"""
        self._frame_counter += 1
        return f"frame_{self._frame_counter:04d}"
    
    def _map_entity_to_concept_type(self, entity_type: str) -> ConceptType:
        """Map entity type to concept type"""
        entity_concept_mapping = {
            'counterparty': ConceptType.ENTITY,
            'customer': ConceptType.ENTITY,
            'amount': ConceptType.QUANTIFIER,
            'percentage': ConceptType.QUANTIFIER,
            'date': ConceptType.TEMPORAL,
            'temporal': ConceptType.TEMPORAL,
            'state': ConceptType.SPATIAL,
            'city': ConceptType.SPATIAL,
            'region': ConceptType.SPATIAL
        }
        
        for key, concept_type in entity_concept_mapping.items():
            if key in entity_type.lower():
                return concept_type
        
        return ConceptType.ENTITY  # Default
    
    # FIXED: Updated to handle dictionary entities
    def _canonicalize_entity_dict(self, entity_dict: Dict[str, Any]) -> str:
        """Canonicalize entity value from dictionary"""
        entity_value = entity_dict.get('entity_value', '')
        entity_type = entity_dict.get('entity_type', '')
        
        canonical = str(entity_value).lower().strip()
        
        # Handle common canonicalizations
        if entity_type == 'amount':
            # Normalize amount representations
            canonical = re.sub(r'[₹,\s]', '', canonical)
        elif entity_type in ['state', 'city']:
            # Standardize geographic names
            canonical = canonical.title()
        
        return canonical
    
    def _infer_concept_type_from_token(self, token) -> ConceptType:
        """Infer concept type from linguistic token"""
        if token.pos_tag.startswith('VB'):
            return ConceptType.ACTION
        elif token.pos_tag.startswith('NN'):
            if token.semantic_role == 'agent':
                return ConceptType.ENTITY
            else:
                return ConceptType.ATTRIBUTE
        elif token.pos_tag.startswith('JJ'):
            return ConceptType.ATTRIBUTE
        elif token.pos_tag.startswith('CD'):
            return ConceptType.QUANTIFIER
        else:
            return ConceptType.ATTRIBUTE
    
    def _find_concept_by_surface_form(self, concepts: Dict[str, SemanticConcept], 
                                    surface_form: str) -> Optional[str]:
        """Find concept ID by surface form"""
        for concept_id, concept in concepts.items():
            if concept.surface_form.lower() == surface_form.lower():
                return concept_id
        return None
    
    def _map_linguistic_to_semantic_relation(self, linguistic_relation: str) -> SemanticRelationType:
        """Map linguistic relation to semantic relation type"""
        relation_mapping = {
            'in': SemanticRelationType.SPATIAL,
            'on': SemanticRelationType.TEMPORAL,
            'of': SemanticRelationType.ATTRIBUTE,
            'by': SemanticRelationType.GROUPING,
            'with': SemanticRelationType.CONDITIONAL,
            'for': SemanticRelationType.SUBJECT_OBJECT,
            'than': SemanticRelationType.COMPARATIVE
        }
        
        return relation_mapping.get(linguistic_relation.lower(), SemanticRelationType.SUBJECT_OBJECT)
    
    def _identify_root_concept(self, concepts: Dict[str, SemanticConcept],
                             relations: List[SemanticRelation],
                             frames: List[SemanticFrame]) -> str:
        """Identify root concept for parse tree"""
        
        # Look for action concepts first
        action_concepts = [cid for cid, c in concepts.items() 
                         if c.concept_type == ConceptType.ACTION]
        if action_concepts:
            return action_concepts[0]
        
        # Look for measure concepts (common in banking queries)
        measure_concepts = [cid for cid, c in concepts.items() 
                          if c.concept_type == ConceptType.MEASURE]
        if measure_concepts:
            return measure_concepts[0]
        
        # Default to first entity concept
        entity_concepts = [cid for cid, c in concepts.items() 
                         if c.concept_type == ConceptType.ENTITY]
        if entity_concepts:
            return entity_concepts[0]
        
        # Fallback to first concept
        if concepts:
            return list(concepts.keys())[0]
        
        return "root_concept_unknown"
    
    def _analyze_information_structure(self, concepts: Dict[str, SemanticConcept],
                                     relations: List[SemanticRelation]) -> Dict[str, Any]:
        """Analyze information structure of the query"""
        return {
            'focus_concepts': [cid for cid, c in concepts.items() 
                             if c.confidence > 0.8],
            'background_concepts': [cid for cid, c in concepts.items() 
                                  if c.confidence <= 0.8],
            'relation_strength': {r.relation_id: r.confidence for r in relations}
        }
    
    def _identify_coherence_relations(self, relations: List[SemanticRelation]) -> List[str]:
        """Identify coherence relations in the semantic structure"""
        coherence_relations = []
        
        # Group relations by type
        relation_types = defaultdict(int)
        for relation in relations:
            relation_types[relation.relation_type.value] += 1
        
        # Identify dominant relation patterns
        for rel_type, count in relation_types.items():
            if count > 1:
                coherence_relations.append(f"multiple_{rel_type}_relations")
        
        return coherence_relations
    
    def _generate_temporal_filter(self, temporal_concept: SemanticConcept) -> Optional[str]:
        """Generate temporal filter condition"""
        
        surface_form = temporal_concept.surface_form.lower()
        
        if 'last' in surface_form and 'days' in surface_form:
            # Extract number of days
            match = re.search(r'(\d+)\s*days?', surface_form)
            if match:
                days = match.group(1)
                return f"date_field >= DATEADD(day, -{days}, GETDATE())"
        
        elif 'last' in surface_form and 'months' in surface_form:
            match = re.search(r'(\d+)\s*months?', surface_form)
            if match:
                months = match.group(1)
                return f"date_field >= DATEADD(month, -{months}, GETDATE())"
        
        return None
    
    def _create_fallback_semantics(self, query_text: str, error_message: str) -> QuerySemantics:
        """Create fallback semantics when parsing fails"""
        
        # Create minimal concepts
        concepts = {
            "fallback_concept": SemanticConcept(
                concept_id="fallback_concept",
                concept_type=ConceptType.ENTITY,
                surface_form=query_text,
                canonical_form=query_text.lower(),
                confidence=0.1
            )
        }
        
        # Create minimal parse tree
        parse_tree = SemanticParseTree(
            root_concept="fallback_concept",
            concepts=concepts,
            relations=[],
            frames=[]
        )
        
        # Create fallback intent frame
        intent_frame = SemanticFrame(
            frame_id="fallback_frame",
            frame_type="fallback",
            core_concepts=["fallback_concept"],
            confidence=0.1
        )
        
        return QuerySemantics(
            original_query=query_text,
            parse_tree=parse_tree,
            query_intent_frame=intent_frame,
            execution_semantics={'error': error_message},
            business_semantics={'error': True, 'error_message': error_message},
            confidence_scores={'overall': 0.1},
            ambiguity_resolutions=[],
            semantic_validation={'is_consistent': False, 'error': error_message}
        )

def main():
    """Test semantic parser functionality"""
    try:
        parser = SemanticParser()
        print("Semantic parser initialized successfully!")
        
        # Test queries
        test_queries = [
            "Show me last 10 days created customers in Maharashtra",
            "What is the total loan amount for ABC Corporation?",
            "Compare this quarter's NPA ratio with last quarter",
            "Give me maximum exposure by region for defaulted loans",
            "Analyze customer acquisition trend over past 6 months"
        ]
        
        print(f"\nTesting semantic parsing with {len(test_queries)} queries:")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*70}")
            print(f"Query {i}: '{query}'")
            print('='*70)
            
            semantics = parser.parse_query_semantics(query)
            
            print(f"Concepts extracted: {len(semantics.parse_tree.concepts)}")
            print(f"Relations identified: {len(semantics.parse_tree.relations)}")
            print(f"Semantic frames: {len(semantics.parse_tree.frames)}")
            print(f"Overall confidence: {semantics.confidence_scores['overall']:.2f}")
            print(f"Intent frame: {semantics.query_intent_frame.frame_type}")
            
            # Show sample concepts
            sample_concepts = list(semantics.parse_tree.concepts.items())[:3]
            if sample_concepts:
                print("Sample concepts:")
                for concept_id, concept in sample_concepts:
                    print(f"  - {concept.concept_type.value}: '{concept.surface_form}' → '{concept.canonical_form}' (conf: {concept.confidence:.2f})")
            
            # Show execution semantics
            if 'select_elements' in semantics.execution_semantics:
                print(f"SQL elements: SELECT {semantics.execution_semantics['select_elements']}")
                print(f"FROM tables: {semantics.execution_semantics['from_tables']}")
            
            # Show business semantics
            if semantics.business_semantics.get('key_metrics'):
                print(f"Key metrics: {[m['metric_name'] for m in semantics.business_semantics['key_metrics']]}")
            
            # Show validation
            if semantics.semantic_validation.get('is_consistent'):
                print("Semantically consistent")
            else:
                print("Semantic inconsistencies detected")
                if semantics.semantic_validation.get('violations'):
                    print(f"   Violations: {semantics.semantic_validation['violations']}")
    
    except Exception as e:
        print(f"Error testing semantic parser: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
