"""
Entity Processor for Banking Domain Queries
Advanced entity extraction and processing with dynamic field discovery
Supports banking entities like CTPT_ID, FAC_ID, amounts, dates, and identifiers
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from decimal import Decimal
import datetime
from dateutil.parser import parse as date_parse

from ..core.exceptions import NLPProcessorBaseException, ValidationError
from ..core.metrics import ComponentType, track_processing_time, metrics_collector


logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of banking entities that can be extracted"""
    COUNTERPARTY_ID = "counterparty_id"
    FACILITY_ID = "facility_id"
    APPLICATION_ID = "application_id"
    USER_ID = "user_id"
    ACCOUNT_NUMBER = "account_number"
    AMOUNT = "amount"
    PERCENTAGE = "percentage"
    DATE = "date"
    STATUS = "status"
    RISK_RATING = "risk_rating"
    BRANCH_CODE = "branch_code"
    PHONE_NUMBER = "phone_number"
    EMAIL = "email"
    PAN_NUMBER = "pan_number"
    IDENTIFIER = "identifier"
    CODE = "code"


class EntityPatternType(Enum):  # RENAMED FROM EntityPattern
    """Patterns for entity recognition"""
    ALPHANUMERIC_ID = "alphanumeric_id"        # ABC123, CTPT001
    NUMERIC_ID = "numeric_id"                  # 123456, 999
    FORMATTED_ID = "formatted_id"              # CTPT-123, FAC/2024/001
    AMOUNT_WITH_CURRENCY = "amount_currency"   # Rs 50000, ₹ 1,00,000
    PERCENTAGE_VALUE = "percentage"            # 15%, 0.15
    DATE_VARIOUS = "date_various"              # 2024-01-15, 15-Jan-2024
    STATUS_KEYWORD = "status_keyword"          # Active, Pending, Approved
    EMAIL_FORMAT = "email_format"              # user@domain.com
    PHONE_FORMAT = "phone_format"              # +91-9876543210


@dataclass
class EntityPattern:
    """Represents a pattern for entity recognition"""
    pattern_type: EntityPatternType  # UPDATED TO USE THE RENAMED ENUM
    regex_pattern: str
    confidence_weight: float
    validation_rules: List[str] = field(default_factory=list)
    transformation_rules: List[str] = field(default_factory=list)


@dataclass
class ExtractedEntity:
    """Represents an extracted entity from text"""
    entity_type: EntityType
    raw_value: str
    normalized_value: str
    confidence_score: float
    position: Tuple[int, int]  # Start and end position in text
    context: str = ""
    field_mapping: Optional[str] = None
    table_mapping: Optional[str] = None
    validation_status: bool = True
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class EntityFieldMapping:
    """Maps entity types to discovered database fields"""
    entity_type: EntityType
    field_name: str
    table_name: str
    confidence_score: float
    field_type: str = "VARCHAR"
    validation_pattern: Optional[str] = None
    business_rules: List[str] = field(default_factory=list)


@dataclass
class EntityProcessingResult:
    """Result of entity processing"""
    original_query: str
    extracted_entities: List[ExtractedEntity]
    field_mappings: List[EntityFieldMapping]
    entity_filters: List[Dict[str, Any]]
    sql_conditions: List[str] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    business_insights: Dict[str, Any] = field(default_factory=dict)


class BankingEntityExtractor:
    """
    Extracts banking-specific entities from text using pattern recognition
    """
    
    def __init__(self):
        """Initialize with banking entity patterns"""
        self.entity_patterns = self._initialize_entity_patterns()
        self.business_validations = self._initialize_business_validations()
        self.normalization_rules = self._initialize_normalization_rules()
    
    def _initialize_entity_patterns(self) -> Dict[EntityType, List[Dict[str, Any]]]:
        """Initialize patterns for different entity types"""
        return {
            EntityType.COUNTERPARTY_ID: [
                {
                    "pattern": r'\bCTPT[_-]?(\d{3,10})\b',
                    "confidence": 0.95,
                    "description": "CTPT ID pattern"
                },
                {
                    "pattern": r'\b(counterparty|customer|client)\s+(?:id|number|code)[\s:=]+([A-Z0-9_-]{3,15})\b',
                    "confidence": 0.8,
                    "description": "Counterparty ID with keyword"
                },
                {
                    "pattern": r'\b([A-Z]{2,4}\d{3,8})\b',
                    "confidence": 0.6,
                    "description": "Generic alphanumeric ID"
                }
            ],
            
            EntityType.FACILITY_ID: [
                {
                    "pattern": r'\bFAC[_-]?(\d{3,10})\b',
                    "confidence": 0.95,
                    "description": "FAC ID pattern"
                },
                {
                    "pattern": r'\b(facility|loan|credit)\s+(?:id|number|code)[\s:=]+([A-Z0-9_/-]{3,20})\b',
                    "confidence": 0.8,
                    "description": "Facility ID with keyword"
                },
                {
                    "pattern": r'\bLN\d{6,12}\b',
                    "confidence": 0.7,
                    "description": "Loan number pattern"
                }
            ],
            
            EntityType.APPLICATION_ID: [
                {
                    "pattern": r'\bAPP[_-]?(\d{3,10})\b',
                    "confidence": 0.95,
                    "description": "Application ID pattern"
                },
                {
                    "pattern": r'\b(application|app)\s+(?:id|number|code)[\s:=]+([A-Z0-9_/-]{3,20})\b',
                    "confidence": 0.8,
                    "description": "Application ID with keyword"
                },
                {
                    "pattern": r'\b\d{10,15}\b',
                    "confidence": 0.4,
                    "description": "Long numeric ID (could be application)"
                }
            ],
            
            EntityType.AMOUNT: [
                {
                    "pattern": r'(?:Rs\.?|₹|INR)\s*([0-9,]+(?:\.[0-9]{1,2})?)\s*(?:crore|cr|lakh|lac|thousand|k)?',
                    "confidence": 0.9,
                    "description": "Indian currency amount"
                },
                {
                    "pattern": r'\b([0-9,]+(?:\.[0-9]{1,2})?)\s*(?:crore|cr|lakh|lac|thousand|k)\b',
                    "confidence": 0.85,
                    "description": "Amount with Indian scale"
                },
                {
                    "pattern": r'\b([0-9,]+(?:\.[0-9]{1,2})?)\s*(?:rupees?|rs\.?)\b',
                    "confidence": 0.8,
                    "description": "Amount with rupees"
                },
                {
                    "pattern": r'\bamount[\s:=]+([0-9,]+(?:\.[0-9]{1,2})?)',
                    "confidence": 0.7,
                    "description": "Amount with keyword"
                }
            ],
            
            EntityType.PERCENTAGE: [
                {
                    "pattern": r'\b([0-9]+(?:\.[0-9]{1,4})?)\s*%',
                    "confidence": 0.9,
                    "description": "Percentage with % symbol"
                },
                {
                    "pattern": r'\b([0-9]+(?:\.[0-9]{1,4})?)\s*percent',
                    "confidence": 0.85,
                    "description": "Percentage with word"
                },
                {
                    "pattern": r'\b(0\.[0-9]{1,4})\b',
                    "confidence": 0.6,
                    "description": "Decimal percentage (0.15 = 15%)"
                }
            ],
            
            EntityType.DATE: [
                {
                    "pattern": r'\b(\d{4}-\d{2}-\d{2})\b',
                    "confidence": 0.95,
                    "description": "ISO date format"
                },
                {
                    "pattern": r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
                    "confidence": 0.85,
                    "description": "Common date formats"
                },
                {
                    "pattern": r'\b(\d{1,2}[-\s](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-\s]\d{2,4})\b',
                    "confidence": 0.9,
                    "description": "Date with month name"
                },
                {
                    "pattern": r'\b(today|yesterday|tomorrow)\b',
                    "confidence": 0.8,
                    "description": "Relative date terms"
                }
            ],
            
            EntityType.STATUS: [
                {
                    "pattern": r'\b(active|inactive|pending|approved|rejected|closed|open|suspended|cancelled)\b',
                    "confidence": 0.85,
                    "description": "Common status values"
                },
                {
                    "pattern": r'\bstatus[\s:=]+(active|inactive|pending|approved|rejected|closed|open)',
                    "confidence": 0.9,
                    "description": "Status with keyword"
                }
            ],
            
            EntityType.EMAIL: [
                {
                    "pattern": r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
                    "confidence": 0.95,
                    "description": "Email address pattern"
                }
            ],
            
            EntityType.PHONE_NUMBER: [
                {
                    "pattern": r'\b(\+91[-\s]?[6-9]\d{9})\b',
                    "confidence": 0.9,
                    "description": "Indian mobile number with country code"
                },
                {
                    "pattern": r'\b([6-9]\d{9})\b',
                    "confidence": 0.8,
                    "description": "Indian mobile number"
                },
                {
                    "pattern": r'\b(\d{2,4}[-\s]?\d{6,8})\b',
                    "confidence": 0.6,
                    "description": "General phone pattern"
                }
            ],
            
            EntityType.PAN_NUMBER: [
                {
                    "pattern": r'\b([A-Z]{5}[0-9]{4}[A-Z])\b',
                    "confidence": 0.95,
                    "description": "PAN card pattern"
                }
            ],
            
            EntityType.BRANCH_CODE: [
                {
                    "pattern": r'\b(BR\d{3,6})\b',
                    "confidence": 0.8,
                    "description": "Branch code pattern"
                },
                {
                    "pattern": r'\bbranch[\s:=]+([A-Z0-9]{3,8})\b',
                    "confidence": 0.85,
                    "description": "Branch code with keyword"
                }
            ]
        }
    
    def _initialize_business_validations(self) -> Dict[EntityType, List[Dict[str, Any]]]:
        """Initialize business validation rules"""
        return {
            EntityType.COUNTERPARTY_ID: [
                {"rule": "length_check", "min_length": 3, "max_length": 20},
                {"rule": "format_check", "allowed_chars": "A-Z0-9_-"}
            ],
            
            EntityType.FACILITY_ID: [
                {"rule": "length_check", "min_length": 3, "max_length": 25},
                {"rule": "format_check", "allowed_chars": "A-Z0-9_/-"}
            ],
            
            EntityType.AMOUNT: [
                {"rule": "range_check", "min_value": 0, "max_value": 999999999999},
                {"rule": "decimal_places", "max_decimals": 2}
            ],
            
            EntityType.PERCENTAGE: [
                {"rule": "range_check", "min_value": 0, "max_value": 100},
                {"rule": "decimal_places", "max_decimals": 4}
            ],
            
            EntityType.EMAIL: [
                {"rule": "format_validation", "pattern": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'}
            ],
            
            EntityType.PHONE_NUMBER: [
                {"rule": "length_check", "min_length": 10, "max_length": 13},
                {"rule": "india_mobile_check", "pattern": r'^(\+91)?[6-9]\d{9}$'}
            ],
            
            EntityType.PAN_NUMBER: [
                {"rule": "format_validation", "pattern": r'^[A-Z]{5}[0-9]{4}[A-Z]$'},
                {"rule": "checksum_validation", "algorithm": "pan_checksum"}
            ]
        }
    
    def _initialize_normalization_rules(self) -> Dict[EntityType, List[Dict[str, Any]]]:
        """Initialize normalization rules for extracted entities"""
        return {
            EntityType.AMOUNT: [
                {"rule": "remove_commas", "pattern": r',', "replacement": ''},
                {"rule": "convert_scale", "crore": 10000000, "lakh": 100000, "thousand": 1000, "k": 1000},
                {"rule": "decimal_precision", "places": 2}
            ],
            
            EntityType.PERCENTAGE: [
                {"rule": "normalize_decimal", "convert_to_percentage": True},
                {"rule": "decimal_precision", "places": 4}
            ],
            
            EntityType.PHONE_NUMBER: [
                {"rule": "add_country_code", "default": "+91"},
                {"rule": "remove_spaces_hyphens", "pattern": r'[-\s]', "replacement": ''}
            ],
            
            EntityType.EMAIL: [
                {"rule": "lowercase", "apply": True},
                {"rule": "trim_whitespace", "apply": True}
            ],
            
            EntityType.DATE: [
                {"rule": "parse_to_iso", "target_format": "%Y-%m-%d"},
                {"rule": "validate_date_range", "min_year": 1900, "max_year": 2100}
            ]
        }
    
    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """
        Extract all banking entities from the given text
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of extracted entities with confidence scores
        """
        extracted_entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                confidence = pattern_info["confidence"]
                
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    # Get the matched value (use first group if available, otherwise full match)
                    if match.groups():
                        raw_value = match.group(1)
                    else:
                        raw_value = match.group(0)
                    
                    # Normalize the value
                    normalized_value = self._normalize_entity_value(entity_type, raw_value)
                    
                    # Validate the entity
                    is_valid, validation_errors = self._validate_entity(entity_type, normalized_value)
                    
                    # Calculate final confidence
                    final_confidence = confidence
                    if not is_valid:
                        final_confidence *= 0.5  # Reduce confidence for invalid entities
                    
                    # Extract context
                    context = self._extract_context(text, match.start(), match.end())
                    
                    entity = ExtractedEntity(
                        entity_type=entity_type,
                        raw_value=raw_value,
                        normalized_value=normalized_value,
                        confidence_score=final_confidence,
                        position=(match.start(), match.end()),
                        context=context,
                        validation_status=is_valid,
                        validation_errors=validation_errors
                    )
                    
                    extracted_entities.append(entity)
        
        # Remove duplicates and keep highest confidence entities
        return self._deduplicate_entities(extracted_entities)
    
    def _normalize_entity_value(self, entity_type: EntityType, raw_value: str) -> str:
        """Normalize entity value according to business rules"""
        
        if entity_type not in self.normalization_rules:
            return raw_value.strip()
        
        normalized = raw_value.strip()
        rules = self.normalization_rules[entity_type]
        
        for rule in rules:
            rule_type = rule["rule"]
            
            if rule_type == "remove_commas":
                normalized = re.sub(rule["pattern"], rule["replacement"], normalized)
            
            elif rule_type == "convert_scale" and entity_type == EntityType.AMOUNT:
                # Handle Indian number scales
                original_normalized = normalized
                if "crore" in raw_value.lower() or "cr" in raw_value.lower():
                    normalized = str(float(normalized.replace(",", "")) * rule["crore"])
                elif "lakh" in raw_value.lower() or "lac" in raw_value.lower():
                    normalized = str(float(normalized.replace(",", "")) * rule["lakh"])
                elif "thousand" in raw_value.lower() or raw_value.lower().endswith("k"):
                    normalized = str(float(normalized.replace(",", "")) * rule["thousand"])
            
            elif rule_type == "normalize_decimal" and entity_type == EntityType.PERCENTAGE:
                # Convert decimal to percentage if needed
                try:
                    value = float(normalized)
                    if value <= 1.0 and "convert_to_percentage" in rule:
                        normalized = str(value * 100)
                except ValueError:
                    pass
            
            elif rule_type == "add_country_code" and entity_type == EntityType.PHONE_NUMBER:
                if not normalized.startswith("+"):
                    normalized = rule["default"] + normalized
            
            elif rule_type == "remove_spaces_hyphens":
                normalized = re.sub(rule["pattern"], rule["replacement"], normalized)
            
            elif rule_type == "lowercase":
                normalized = normalized.lower()
            
            elif rule_type == "trim_whitespace":
                normalized = normalized.strip()
            
            elif rule_type == "parse_to_iso" and entity_type == EntityType.DATE:
                try:
                    # Handle relative dates
                    if normalized.lower() == "today":
                        normalized = datetime.datetime.now().strftime(rule["target_format"])
                    elif normalized.lower() == "yesterday":
                        normalized = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime(rule["target_format"])
                    elif normalized.lower() == "tomorrow":
                        normalized = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime(rule["target_format"])
                    else:
                        # Parse actual date
                        parsed_date = date_parse(normalized)
                        normalized = parsed_date.strftime(rule["target_format"])
                except:
                    # Keep original if parsing fails
                    pass
        
        return normalized
    
    def _validate_entity(self, entity_type: EntityType, value: str) -> Tuple[bool, List[str]]:
        """Validate entity according to business rules"""
        
        if entity_type not in self.business_validations:
            return True, []  # No validation rules = valid
        
        validation_errors = []
        rules = self.business_validations[entity_type]
        
        for rule in rules:
            rule_type = rule["rule"]
            
            if rule_type == "length_check":
                if len(value) < rule["min_length"] or len(value) > rule["max_length"]:
                    validation_errors.append(f"Length must be between {rule['min_length']} and {rule['max_length']}")
            
            elif rule_type == "format_check":
                pattern = f"^[{rule['allowed_chars']}]+$"
                if not re.match(pattern, value):
                    validation_errors.append(f"Invalid format. Allowed characters: {rule['allowed_chars']}")
            
            elif rule_type == "range_check":
                try:
                    num_value = float(value.replace(",", ""))
                    if num_value < rule["min_value"] or num_value > rule["max_value"]:
                        validation_errors.append(f"Value must be between {rule['min_value']} and {rule['max_value']}")
                except ValueError:
                    validation_errors.append("Must be a valid number")
            
            elif rule_type == "format_validation":
                if not re.match(rule["pattern"], value):
                    validation_errors.append("Invalid format")
            
            elif rule_type == "india_mobile_check":
                cleaned_number = re.sub(r'[+\-\s]', '', value)
                if not re.match(rule["pattern"], cleaned_number):
                    validation_errors.append("Invalid Indian mobile number format")
        
        is_valid = len(validation_errors) == 0
        return is_valid, validation_errors
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 30) -> str:
        """Extract context around the matched entity"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        
        context = text[context_start:context_end]
        
        # Mark the entity in context
        entity_start = start - context_start
        entity_end = end - context_start
        
        return context[:entity_start] + "[" + context[entity_start:entity_end] + "]" + context[entity_end:]
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities, keeping the one with highest confidence"""
        
        # Group by position and value
        entity_groups = defaultdict(list)
        
        for entity in entities:
            key = (entity.position, entity.normalized_value)
            entity_groups[key].append(entity)
        
        # Keep the highest confidence entity from each group
        deduplicated = []
        for group in entity_groups.values():
            best_entity = max(group, key=lambda e: e.confidence_score)
            deduplicated.append(best_entity)
        
        return sorted(deduplicated, key=lambda e: e.position[0])


class EntityProcessor:
    """
    Advanced entity processor with dynamic field mapping
    Processes banking entities and maps them to database fields
    """
    
    def __init__(self, schema_tables: Optional[List[str]] = None):
        """Initialize entity processor with schema knowledge"""
        
        # Schema tables
        self.schema_tables = schema_tables or [
            "tblCTPTAddress", "tblCTPTContactDetails", "tblCTPTIdentifiersDetails",
            "tblCTPTOwner", "tblCTPTRMDetails", "tblCounterparty",
            "tblOApplicationMaster", "tblOSWFActionStatusApplicationTracker",
            "tblOSWFActionStatusApplicationTrackerExtended", 
            "tblOSWFActionStatusApplicationTracker_History",
            "tblOSWFActionStatusAssesmentTracker", "tblOSWFActionStatusCollateralTracker",
            "tblOSWFActionStatusConditionTracker", "tblOSWFActionStatusDeviationsTracker",
            "tblOSWFActionStatusFacilityTracker", "tblOSWFActionStatusFinancialTracker",
            "tblOSWFActionStatusScoringTracker", "tblRoles", "tblScheduledItemsTracker",
            "tblScheduledItemsTracker_History", "tbluserroles", "tblusers"
        ]
        
        # Initialize entity extractor
        self.entity_extractor = BankingEntityExtractor()
        
        # Field discovery patterns for different entity types
        self.entity_field_patterns = self._initialize_entity_field_patterns()
        
        logger.info(f"EntityProcessor initialized with {len(self.schema_tables)} schema tables")
    
    def process_entities(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> EntityProcessingResult:
        """
        Process entities in query text with dynamic field mapping
        
        Args:
            query_text: Query text containing entities
            context: Additional context including available fields
            
        Returns:
            Complete entity processing result
        """
        with track_processing_time(ComponentType.NLP_ORCHESTRATOR, "process_entities"):
            try:
                # Extract entities from query text
                extracted_entities = self.entity_extractor.extract_entities(query_text)
                
                # Discover field mappings for entities
                field_mappings = self._discover_entity_field_mappings(extracted_entities, context)
                
                # Map entities to discovered fields
                self._map_entities_to_fields(extracted_entities, field_mappings)
                
                # Create entity filters
                entity_filters = self._create_entity_filters(extracted_entities, field_mappings)
                
                # Generate SQL conditions
                sql_conditions = self._generate_entity_sql_conditions(entity_filters)
                
                # Validate entity processing
                validation_results = self._validate_entity_processing(extracted_entities, field_mappings)
                
                # Extract business insights
                business_insights = self._extract_entity_insights(
                    extracted_entities, field_mappings, query_text
                )
                
                result = EntityProcessingResult(
                    original_query=query_text,
                    extracted_entities=extracted_entities,
                    field_mappings=field_mappings,
                    entity_filters=entity_filters,
                    sql_conditions=sql_conditions,
                    validation_results=validation_results,
                    business_insights=business_insights
                )
                
                # Record metrics
                metrics_collector.record_accuracy(
                    ComponentType.NLP_ORCHESTRATOR,
                    len(extracted_entities) > 0,
                    len([e for e in extracted_entities if e.validation_status]) / max(len(extracted_entities), 1),
                    "entity_processing"
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Entity processing failed: {e}")
                raise ValidationError(
                    validation_type="entity_processing",
                    failed_rules=[str(e)]
                )
    
    def _initialize_entity_field_patterns(self) -> Dict[EntityType, List[str]]:
        """Initialize patterns for discovering entity fields"""
        return {
            EntityType.COUNTERPARTY_ID: [
                r".*ctpt.*id.*",
                r".*counterparty.*id.*",
                r".*customer.*id.*",
                r".*client.*id.*"
            ],
            
            EntityType.FACILITY_ID: [
                r".*fac.*id.*",
                r".*facility.*id.*",
                r".*loan.*id.*",
                r".*credit.*id.*"
            ],
            
            EntityType.APPLICATION_ID: [
                r".*app.*id.*",
                r".*application.*id.*",
                r".*request.*id.*"
            ],
            
            EntityType.USER_ID: [
                r".*user.*id.*",
                r".*emp.*id.*",
                r".*employee.*id.*",
                r".*staff.*id.*"
            ],
            
            EntityType.AMOUNT: [
                r".*amount.*",
                r".*value.*",
                r".*balance.*",
                r".*sum.*",
                r".*total.*",
                r".*limit.*"
            ],
            
            EntityType.STATUS: [
                r".*status.*",
                r".*state.*",
                r".*condition.*",
                r".*flag.*"
            ],
            
            EntityType.DATE: [
                r".*date.*",
                r".*time.*",
                r".*created.*",
                r".*modified.*",
                r".*updated.*",
                r".*effective.*"
            ],
            
            EntityType.EMAIL: [
                r".*email.*",
                r".*mail.*",
                r".*e[-_]?mail.*"
            ],
            
            EntityType.PHONE_NUMBER: [
                r".*phone.*",
                r".*mobile.*",
                r".*contact.*",
                r".*number.*"
            ],
            
            EntityType.BRANCH_CODE: [
                r".*branch.*",
                r".*office.*",
                r".*location.*",
                r".*center.*"
            ]
        }
    
    def _discover_entity_field_mappings(self, entities: List[ExtractedEntity], context: Optional[Dict[str, Any]]) -> List[EntityFieldMapping]:
        """Discover field mappings for extracted entities"""
        
        field_mappings = []
        
        if not context or "available_fields" not in context:
            return field_mappings
        
        available_fields = context["available_fields"]
        
        # Get unique entity types from extracted entities
        entity_types = set(entity.entity_type for entity in entities)
        
        for entity_type in entity_types:
            if entity_type not in self.entity_field_patterns:
                continue
            
            patterns = self.entity_field_patterns[entity_type]
            
            # Search for matching fields across all tables
            for table_name, fields in available_fields.items():
                for field_name in fields:
                    field_lower = field_name.lower()
                    
                    for pattern in patterns:
                        if re.match(pattern, field_lower, re.IGNORECASE):
                            confidence = self._calculate_entity_field_confidence(
                                entity_type, field_name, table_name, pattern
                            )
                            
                            field_mapping = EntityFieldMapping(
                                entity_type=entity_type,
                                field_name=field_name,
                                table_name=table_name,
                                confidence_score=confidence,
                                field_type=self._infer_field_type_for_entity(entity_type),
                                validation_pattern=pattern
                            )
                            
                            field_mappings.append(field_mapping)
                            break  # Found a match, no need to check other patterns
        
        # Sort by confidence and remove duplicates
        field_mappings.sort(key=lambda x: x.confidence_score, reverse=True)
        return self._deduplicate_field_mappings(field_mappings)
    
    def _calculate_entity_field_confidence(self, entity_type: EntityType, field_name: str, table_name: str, pattern: str) -> float:
        """Calculate confidence for entity-field mapping"""
        
        confidence = 0.5  # Base confidence
        field_lower = field_name.lower()
        table_lower = table_name.lower()
        
        # Direct keyword matches
        entity_keywords = {
            EntityType.COUNTERPARTY_ID: ["ctpt", "counterparty", "customer", "client"],
            EntityType.FACILITY_ID: ["fac", "facility", "loan", "credit"],
            EntityType.APPLICATION_ID: ["app", "application", "request"],
            EntityType.USER_ID: ["user", "emp", "employee", "staff"],
            EntityType.AMOUNT: ["amount", "value", "balance", "sum", "total"],
            EntityType.STATUS: ["status", "state", "condition", "flag"],
            EntityType.DATE: ["date", "time", "created", "modified"],
            EntityType.EMAIL: ["email", "mail"],
            EntityType.PHONE_NUMBER: ["phone", "mobile", "contact"],
            EntityType.BRANCH_CODE: ["branch", "office", "location"]
        }
        
        if entity_type in entity_keywords:
            keywords = entity_keywords[entity_type]
            for keyword in keywords:
                if keyword in field_lower:
                    confidence += 0.3
                    break
        
        # Table context relevance
        table_relevance = {
            EntityType.COUNTERPARTY_ID: ["ctpt", "counterparty", "customer"],
            EntityType.FACILITY_ID: ["facility", "fac", "loan"],
            EntityType.APPLICATION_ID: ["application", "app", "request"],
            EntityType.USER_ID: ["user", "employee", "staff"],
            EntityType.EMAIL: ["contact", "details", "ctpt", "customer"],
            EntityType.PHONE_NUMBER: ["contact", "details", "ctpt", "customer"]
        }
        
        if entity_type in table_relevance:
            relevant_terms = table_relevance[entity_type]
            for term in relevant_terms:
                if term in table_lower:
                    confidence += 0.2
                    break
        
        # ID field specific patterns
        if entity_type.name.endswith("_ID"):
            if "id" in field_lower:
                confidence += 0.2
            if field_name.endswith("_ID") or field_name.endswith("ID"):
                confidence += 0.1
        
        return min(1.0, confidence)
    
    def _infer_field_type_for_entity(self, entity_type: EntityType) -> str:
        """Infer database field type for entity type"""
        
        type_mappings = {
            EntityType.COUNTERPARTY_ID: "VARCHAR",
            EntityType.FACILITY_ID: "VARCHAR",
            EntityType.APPLICATION_ID: "VARCHAR",
            EntityType.USER_ID: "VARCHAR",
            EntityType.ACCOUNT_NUMBER: "VARCHAR",
            EntityType.AMOUNT: "DECIMAL",
            EntityType.PERCENTAGE: "DECIMAL",
            EntityType.DATE: "DATETIME",
            EntityType.STATUS: "VARCHAR",
            EntityType.RISK_RATING: "VARCHAR",
            EntityType.BRANCH_CODE: "VARCHAR",
            EntityType.PHONE_NUMBER: "VARCHAR",
            EntityType.EMAIL: "VARCHAR",
            EntityType.PAN_NUMBER: "VARCHAR",
            EntityType.IDENTIFIER: "VARCHAR",
            EntityType.CODE: "VARCHAR"
        }
        
        return type_mappings.get(entity_type, "VARCHAR")
    
    def _deduplicate_field_mappings(self, field_mappings: List[EntityFieldMapping]) -> List[EntityFieldMapping]:
        """Remove duplicate field mappings, keeping highest confidence"""
        
        # Group by entity type and field name
        mapping_groups = defaultdict(list)
        
        for mapping in field_mappings:
            key = (mapping.entity_type, mapping.field_name, mapping.table_name)
            mapping_groups[key].append(mapping)
        
        # Keep highest confidence mapping from each group
        deduplicated = []
        for group in mapping_groups.values():
            best_mapping = max(group, key=lambda m: m.confidence_score)
            deduplicated.append(best_mapping)
        
        return deduplicated
    
    def _map_entities_to_fields(self, entities: List[ExtractedEntity], field_mappings: List[EntityFieldMapping]) -> None:
        """Map extracted entities to discovered database fields"""
        
        # Create mapping lookup
        mapping_lookup = {}
        for mapping in field_mappings:
            if mapping.entity_type not in mapping_lookup:
                mapping_lookup[mapping.entity_type] = []
            mapping_lookup[mapping.entity_type].append(mapping)
        
        # Assign field mappings to entities
        for entity in entities:
            if entity.entity_type in mapping_lookup:
                # Get the best mapping for this entity type
                best_mapping = max(mapping_lookup[entity.entity_type], key=lambda m: m.confidence_score)
                entity.field_mapping = best_mapping.field_name
                entity.table_mapping = best_mapping.table_name
    
    def _create_entity_filters(self, entities: List[ExtractedEntity], field_mappings: List[EntityFieldMapping]) -> List[Dict[str, Any]]:
        """Create filter conditions from extracted entities"""
        
        entity_filters = []
        
        for entity in entities:
            if not entity.field_mapping or not entity.table_mapping:
                continue  # Skip entities without field mappings
            
            # Determine appropriate operator based on entity type
            operator = "="
            if entity.entity_type == EntityType.AMOUNT:
                # For amounts, we might want range queries
                if "greater than" in entity.context.lower() or ">" in entity.context:
                    operator = ">"
                elif "less than" in entity.context.lower() or "<" in entity.context:
                    operator = "<"
                elif "between" in entity.context.lower():
                    operator = "BETWEEN"
            
            elif entity.entity_type in [EntityType.COUNTERPARTY_ID, EntityType.FACILITY_ID, EntityType.APPLICATION_ID]:
                # For IDs, use exact match or IN for multiple values
                operator = "="
            
            elif entity.entity_type == EntityType.STATUS:
                # For status, might use LIKE for partial matches
                if len(entity.normalized_value) < 3:
                    operator = "LIKE"
            
            entity_filter = {
                "entity_type": entity.entity_type.value,
                "field_name": entity.field_mapping,
                "table_name": entity.table_mapping,
                "operator": operator,
                "value": entity.normalized_value,
                "confidence": entity.confidence_score,
                "validation_status": entity.validation_status
            }
            
            entity_filters.append(entity_filter)
        
        return entity_filters
    
    def _generate_entity_sql_conditions(self, entity_filters: List[Dict[str, Any]]) -> List[str]:
        """Generate SQL WHERE conditions from entity filters"""
        
        sql_conditions = []
        
        for filter_condition in entity_filters:
            if not filter_condition["validation_status"]:
                continue  # Skip invalid entities
            
            table_name = filter_condition["table_name"]
            field_name = filter_condition["field_name"]
            operator = filter_condition["operator"]
            value = filter_condition["value"]
            
            # Create table alias
            table_alias = table_name.lower()[:3]
            field_ref = f"{table_alias}.{field_name}"
            
            # Generate condition based on operator
            if operator == "=":
                condition = f"{field_ref} = '{value}'"
            elif operator in [">", "<", ">=", "<="]:
                condition = f"{field_ref} {operator} {value}"
            elif operator == "LIKE":
                condition = f"{field_ref} LIKE '%{value}%'"
            elif operator == "IN":
                # Handle multiple values (if value is a list)
                if isinstance(value, list):
                    values_str = "', '".join(value)
                    condition = f"{field_ref} IN ('{values_str}')"
                else:
                    condition = f"{field_ref} = '{value}'"
            else:
                condition = f"{field_ref} = '{value}'"
            
            sql_conditions.append(condition)
            
            # Add comment for clarity
            entity_type = filter_condition["entity_type"]
            sql_conditions.append(f"-- Entity filter: {entity_type}")
        
        return sql_conditions
    
    def _validate_entity_processing(self, entities: List[ExtractedEntity], field_mappings: List[EntityFieldMapping]) -> Dict[str, Any]:
        """Validate entity processing results"""
        
        validation_results = {
            "total_entities_extracted": len(entities),
            "valid_entities": len([e for e in entities if e.validation_status]),
            "invalid_entities": len([e for e in entities if not e.validation_status]),
            "entities_with_field_mapping": len([e for e in entities if e.field_mapping]),
            "entities_without_field_mapping": len([e for e in entities if not e.field_mapping]),
            "field_mappings_discovered": len(field_mappings),
            "high_confidence_mappings": len([m for m in field_mappings if m.confidence_score > 0.8]),
            "validation_errors": []
        }
        
        # Collect validation errors
        for entity in entities:
            if entity.validation_errors:
                validation_results["validation_errors"].extend([
                    f"{entity.entity_type.value}: {error}" for error in entity.validation_errors
                ])
        
        # Calculate success rates
        if entities:
            validation_results["validation_success_rate"] = validation_results["valid_entities"] / len(entities) * 100
            validation_results["field_mapping_success_rate"] = validation_results["entities_with_field_mapping"] / len(entities) * 100
        
        return validation_results
    
    def _extract_entity_insights(self, entities: List[ExtractedEntity], field_mappings: List[EntityFieldMapping], query_text: str) -> Dict[str, Any]:
        """Extract business insights from entity processing"""
        
        insights = {
            "entity_distribution": {},
            "confidence_analysis": {},
            "field_mapping_coverage": {},
            "business_relevance": [],
            "query_complexity": "low"
        }
        
        # Entity type distribution
        entity_type_counts = defaultdict(int)
        for entity in entities:
            entity_type_counts[entity.entity_type.value] += 1
        
        insights["entity_distribution"] = dict(entity_type_counts)
        
        # Confidence analysis
        if entities:
            confidences = [e.confidence_score for e in entities]
            insights["confidence_analysis"] = {
                "average_confidence": sum(confidences) / len(confidences),
                "high_confidence_entities": len([c for c in confidences if c > 0.8]),
                "low_confidence_entities": len([c for c in confidences if c < 0.6])
            }
        
        # Field mapping coverage
        mapped_entity_types = set(m.entity_type for m in field_mappings)
        extracted_entity_types = set(e.entity_type for e in entities)
        
        insights["field_mapping_coverage"] = {
            "total_entity_types": len(extracted_entity_types),
            "mapped_entity_types": len(mapped_entity_types),
            "coverage_percentage": len(mapped_entity_types) / max(len(extracted_entity_types), 1) * 100
        }
        
        # Business relevance
        if EntityType.COUNTERPARTY_ID in extracted_entity_types:
            insights["business_relevance"].append("Customer-specific analysis - suitable for individual customer insights")
        
        if EntityType.FACILITY_ID in extracted_entity_types:
            insights["business_relevance"].append("Facility-specific analysis - suitable for loan/credit performance")
        
        if EntityType.AMOUNT in extracted_entity_types:
            insights["business_relevance"].append("Amount-based filtering - suitable for financial analysis")
        
        if EntityType.DATE in extracted_entity_types:
            insights["business_relevance"].append("Time-bound analysis - suitable for temporal reporting")
        
        # Query complexity
        if len(entities) > 5:
            insights["query_complexity"] = "high"
        elif len(entities) > 2:
            insights["query_complexity"] = "medium"
        
        return insights


# Global entity processor instance
entity_processor = EntityProcessor()
