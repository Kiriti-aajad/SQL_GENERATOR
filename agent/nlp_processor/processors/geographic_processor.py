"""
Geographic Processor for Banking Domain Queries
Advanced geographic processing with dynamic field discovery and Indian state mapping
Supports state abbreviations, city mapping, and regional analysis for banking operations
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import re
import difflib

from ..core.exceptions import NLPProcessorBaseException, ValidationError
from ..core.metrics import ComponentType, track_processing_time, metrics_collector


logger = logging.getLogger(__name__)


class GeographicLevel(Enum):
    """Levels of geographic granularity"""
    COUNTRY = "country"
    REGION = "region"
    STATE = "state"
    DISTRICT = "district"
    CITY = "city"
    AREA = "area"
    PINCODE = "pincode"


class GeographicScope(Enum):
    """Scope of geographic analysis"""
    NATIONAL = "national"
    REGIONAL = "regional"
    STATE_WISE = "state_wise"
    CITY_WISE = "city_wise"
    BRANCH_WISE = "branch_wise"
    LOCAL = "local"


@dataclass
class GeographicMapping:
    """Represents a geographic entity mapping"""
    user_input: str
    mapped_value: str
    confidence_score: float
    geographic_level: GeographicLevel
    aliases: List[str] = field(default_factory=list)
    parent_geography: Optional[str] = None
    child_geographies: List[str] = field(default_factory=list)


@dataclass
class GeographicFieldDiscovery:
    """Result of geographic field discovery"""
    field_name: str
    table_name: str
    geographic_level: GeographicLevel
    confidence_score: float
    sample_values: List[str] = field(default_factory=list)
    pattern_matched: str = ""


@dataclass
class GeographicFilter:
    """Represents a geographic filter condition"""
    field_name: str
    table_name: str
    operator: str = "="
    values: List[str] = field(default_factory=list)
    geographic_level: GeographicLevel = GeographicLevel.STATE
    include_children: bool = False  # Include child geographies


@dataclass
class GeographicProcessingResult:
    """Result of geographic processing"""
    original_query: str
    identified_locations: List[str]
    geographic_mappings: List[GeographicMapping]
    discovered_fields: List[GeographicFieldDiscovery]
    geographic_filters: List[GeographicFilter]
    sql_conditions: List[str] = field(default_factory=list)
    geographic_scope: GeographicScope = GeographicScope.NATIONAL
    business_insights: Dict[str, Any] = field(default_factory=dict)


class IndianGeographyMapper:
    """
    Maps Indian geographic entities to database codes and handles hierarchies
    """
    
    def __init__(self):
        """Initialize with comprehensive Indian geography mapping"""
        
        # Indian States and Union Territories mapping
        self.state_mappings = self._initialize_state_mappings()
        
        # Major cities to state mapping
        self.city_to_state = self._initialize_city_mappings()
        
        # Districts to state mapping (major districts)
        self.district_to_state = self._initialize_district_mappings()
        
        # Regional groupings
        self.regional_groupings = self._initialize_regional_groupings()
        
        # Banking zones (as per RBI zones)
        self.banking_zones = self._initialize_banking_zones()
        
        # Alternative names and aliases
        self.geographic_aliases = self._initialize_geographic_aliases()
    
    def _initialize_state_mappings(self) -> Dict[str, str]:
        """Initialize Indian state and UT mappings to standard codes"""
        return {
            # States
            "andhra pradesh": "AP",
            "arunachal pradesh": "AR", 
            "assam": "AS",
            "bihar": "BR",
            "chhattisgarh": "CG",
            "goa": "GA",
            "gujarat": "GJ",
            "haryana": "HR",
            "himachal pradesh": "HP",
            "jharkhand": "JH",
            "karnataka": "KA",
            "kerala": "KL",
            "madhya pradesh": "MP",
            "maharashtra": "MH",
            "manipur": "MN",
            "meghalaya": "ML",
            "mizoram": "MZ",
            "nagaland": "NL",
            "odisha": "OR",
            "punjab": "PB",
            "rajasthan": "RJ",
            "sikkim": "SK",
            "tamil nadu": "TN",
            "telangana": "TG",
            "tripura": "TR",
            "uttar pradesh": "UP",
            "uttarakhand": "UK",
            "west bengal": "WB",
            
            # Union Territories
            "andaman and nicobar islands": "AN",
            "chandigarh": "CH",
            "dadra and nagar haveli and daman and diu": "DN",
            "delhi": "DL",
            "jammu and kashmir": "JK",
            "ladakh": "LA",
            "lakshadweep": "LD",
            "puducherry": "PY",
            
            # Common alternative names
            "orissa": "OR",  # Old name for Odisha
            "pondicherry": "PY",  # Old name for Puducherry
            "bombay": "MH",  # Historical reference to Maharashtra
            "madras": "TN",   # Historical reference to Tamil Nadu
            "calcutta": "WB", # Historical reference to West Bengal
        }
    
    def _initialize_city_mappings(self) -> Dict[str, str]:
        """Initialize major Indian cities to state mapping"""
        return {
            # Major Metro Cities
            "mumbai": "MH", "pune": "MH", "nagpur": "MH", "nashik": "MH", "aurangabad": "MH",
            "delhi": "DL", "new delhi": "DL", "gurgaon": "HR", "noida": "UP", "faridabad": "HR",
            "bangalore": "KA", "mysore": "KA", "hubli": "KA", "mangalore": "KA",
            "chennai": "TN", "coimbatore": "TN", "madurai": "TN", "salem": "TN", "tiruchirappalli": "TN",
            "kolkata": "WB", "howrah": "WB", "durgapur": "WB", "siliguri": "WB",
            "hyderabad": "TG", "secunderabad": "TG", "warangal": "TG", "nizamabad": "TG",
            "ahmedabad": "GJ", "surat": "GJ", "vadodara": "GJ", "rajkot": "GJ", "gandhinagar": "GJ",
            "jaipur": "RJ", "jodhpur": "RJ", "kota": "RJ", "ajmer": "RJ", "udaipur": "RJ",
            "lucknow": "UP", "kanpur": "UP", "agra": "UP", "varanasi": "UP", "allahabad": "UP", "prayagraj": "UP",
            "bhopal": "MP", "indore": "MP", "gwalior": "MP", "jabalpur": "MP",
            "chandigarh": "CH", "ludhiana": "PB", "amritsar": "PB", "jalandhar": "PB",
            "patna": "BR", "gaya": "BR", "bhagalpur": "BR", "muzaffarpur": "BR",
            "bhubaneswar": "OR", "cuttack": "OR", "rourkela": "OR",
            "kochi": "KL", "thiruvananthapuram": "KL", "kozhikode": "KL", "thrissur": "KL",
            "vizag": "AP", "visakhapatnam": "AP", "vijayawada": "AP", "guntur": "AP", "tirupati": "AP",
            "raipur": "CG", "bilaspur": "CG", "durg": "CG",
            "ranchi": "JH", "jamshedpur": "JH", "dhanbad": "JH",
            "dehradun": "UK", "haridwar": "UK", "rishikesh": "UK",
            "shimla": "HP", "dharamshala": "HP", "manali": "HP",
            "panaji": "GA", "margao": "GA", "vasco": "GA",
            "guwahati": "AS", "dibrugarh": "AS", "jorhat": "AS",
            "agartala": "TR", "aizawl": "MZ", "imphal": "MN", "kohima": "NL", "shillong": "ML", "gangtok": "SK"
        }
    
    def _initialize_district_mappings(self) -> Dict[str, str]:
        """Initialize major districts to state mapping"""
        return {
            # Maharashtra districts
            "mumbai": "MH", "thane": "MH", "pune": "MH", "nagpur": "MH", "nashik": "MH",
            
            # Karnataka districts  
            "bangalore urban": "KA", "mysore": "KA", "hubli-dharwad": "KA", "belgaum": "KA",
            
            # Tamil Nadu districts
            "chennai": "TN", "coimbatore": "TN", "madurai": "TN", "salem": "TN", "tiruchirappalli": "TN",
            
            # Gujarat districts
            "ahmedabad": "GJ", "surat": "GJ", "vadodara": "GJ", "rajkot": "GJ",
            
            # Uttar Pradesh districts
            "lucknow": "UP", "kanpur nagar": "UP", "agra": "UP", "varanasi": "UP", "allahabad": "UP",
            
            # Add more as needed based on your data
        }
    
    def _initialize_regional_groupings(self) -> Dict[str, List[str]]:
        """Initialize regional groupings for analysis"""
        return {
            "northern_region": ["DL", "HR", "HP", "JK", "LA", "PB", "RJ", "UP", "UK", "CH"],
            "southern_region": ["AP", "KA", "KL", "TN", "TG", "PY", "AN", "LD"],
            "eastern_region": ["BR", "JH", "OR", "WB", "SK", "AS", "AR", "MN", "ML", "MZ", "NL", "TR"],
            "western_region": ["GJ", "MH", "MP", "CG", "GA", "DN"],
            "central_region": ["MP", "CG", "UP", "UK"],
            "north_east_region": ["AS", "AR", "MN", "ML", "MZ", "NL", "SK", "TR"]
        }
    # Needs Correction region should not be like that there is a specific regions table present in database 
    def _initialize_banking_zones(self) -> Dict[str, List[str]]:
        """Initialize RBI banking zones"""
        return {
            "north_zone": ["DL", "HR", "HP", "JK", "LA", "PB", "RJ", "UP", "UK", "CH"],
            "south_zone": ["AP", "KA", "KL", "TN", "TG", "PY"],
            "east_zone": ["BR", "JH", "OR", "WB", "SK"],
            "west_zone": ["GJ", "MH", "MP", "CG", "GA", "DN"],
            "north_east_zone": ["AS", "AR", "MN", "ML", "MZ", "NL", "TR"]
        }
    
    def _initialize_geographic_aliases(self) -> Dict[str, List[str]]:
        """Initialize common aliases and alternative names"""
        return {
            "delhi": ["delhi", "new delhi", "ncr", "national capital region"],
            "mumbai": ["mumbai", "bombay", "maximum city"],
            "bangalore": ["bangalore", "bengaluru", "silicon city"],
            "chennai": ["chennai", "madras"],
            "kolkata": ["kolkata", "calcutta"],
            "hyderabad": ["hyderabad", "cyberabad", "hitec city"],
            "pune": ["pune", "poona"],
            "ahmedabad": ["ahmedabad", "amdavad"],
            "up": ["uttar pradesh", "u.p.", "up state"],
            "mp": ["madhya pradesh", "m.p.", "mp state"],
            "tn": ["tamil nadu", "tamilnadu", "t.n."],
            "mh": ["maharashtra", "maharastra", "maha"],
            "ka": ["karnataka", "karnatak"],
            "ap": ["andhra pradesh", "andhra"],
            "tg": ["telangana", "telengana"],
            "gj": ["gujarat", "gujrat"],
            "wb": ["west bengal", "w.b.", "bengal"],
            "or": ["odisha", "orissa"],
            "rj": ["rajasthan", "raj"],
            "pb": ["punjab", "panjab"],
            "hr": ["haryana", "hariyana"],
            "br": ["bihar", "behar"],
            "jh": ["jharkhand", "jharkhnd"],
            "cg": ["chhattisgarh", "chattisgarh"],
            "kl": ["kerala", "keralam"],
            "as": ["assam", "asom"],
            "dl": ["delhi", "new delhi", "ncr"]
        }
    
    def map_location(self, location_text: str) -> Optional[GeographicMapping]:
        """
        Map a location text to database code with confidence scoring
        
        Args:
            location_text: Location mentioned by user (e.g., "Delhi", "Maharashtra", "Mumbai")
            
        Returns:
            GeographicMapping with mapped value and confidence
        """
        location_lower = location_text.lower().strip()
        
        # Direct state mapping
        if location_lower in self.state_mappings:
            return GeographicMapping(
                user_input=location_text,
                mapped_value=self.state_mappings[location_lower],
                confidence_score=1.0,
                geographic_level=GeographicLevel.STATE,
                aliases=self.geographic_aliases.get(self.state_mappings[location_lower], [])
            )
        
        # City to state mapping
        if location_lower in self.city_to_state:
            return GeographicMapping(
                user_input=location_text,
                mapped_value=self.city_to_state[location_lower],
                confidence_score=0.9,
                geographic_level=GeographicLevel.CITY,
                parent_geography=self.city_to_state[location_lower]
            )
        
        # District to state mapping
        if location_lower in self.district_to_state:
            return GeographicMapping(
                user_input=location_text,
                mapped_value=self.district_to_state[location_lower],
                confidence_score=0.8,
                geographic_level=GeographicLevel.DISTRICT,
                parent_geography=self.district_to_state[location_lower]
            )
        
        # Fuzzy matching for aliases
        for state_code, aliases in self.geographic_aliases.items():
            for alias in aliases:
                if location_lower in alias or alias in location_lower:
                    similarity = difflib.SequenceMatcher(None, location_lower, alias).ratio()
                    if similarity > 0.8:
                        return GeographicMapping(
                            user_input=location_text,
                            mapped_value=state_code.upper(),
                            confidence_score=similarity * 0.9,
                            geographic_level=self._determine_geographic_level(alias),
                            aliases=[alias]
                        )
        
        # Fuzzy matching for state names
        state_matches = difflib.get_close_matches(location_lower, self.state_mappings.keys(), n=1, cutoff=0.7)
        if state_matches:
            matched_state = state_matches[0]
            similarity = difflib.SequenceMatcher(None, location_lower, matched_state).ratio()
            return GeographicMapping(
                user_input=location_text,
                mapped_value=self.state_mappings[matched_state],
                confidence_score=similarity * 0.8,
                geographic_level=GeographicLevel.STATE
            )
        
        # Fuzzy matching for cities
        city_matches = difflib.get_close_matches(location_lower, self.city_to_state.keys(), n=1, cutoff=0.7)
        if city_matches:
            matched_city = city_matches[0]
            similarity = difflib.SequenceMatcher(None, location_lower, matched_city).ratio()
            return GeographicMapping(
                user_input=location_text,
                mapped_value=self.city_to_state[matched_city],
                confidence_score=similarity * 0.7,
                geographic_level=GeographicLevel.CITY,
                parent_geography=self.city_to_state[matched_city]
            )
        
        return None
    
    def _determine_geographic_level(self, location_name: str) -> GeographicLevel:
        """Determine geographic level based on location name"""
        location_lower = location_name.lower()
        
        if location_lower in self.state_mappings:
            return GeographicLevel.STATE
        elif location_lower in self.city_to_state:
            return GeographicLevel.CITY
        elif location_lower in self.district_to_state:
            return GeographicLevel.DISTRICT
        else:
            return GeographicLevel.AREA
    
    def get_regional_states(self, region_name: str) -> List[str]:
        """Get states belonging to a specific region"""
        region_lower = region_name.lower().replace(" ", "_")
        
        if region_lower in self.regional_groupings:
            return self.regional_groupings[region_lower]
        
        # Try banking zones
        zone_key = f"{region_lower}_zone"
        if zone_key in self.banking_zones:
            return self.banking_zones[zone_key]
        
        return []
    
    def expand_geographic_scope(self, location_code: str) -> List[str]:
        """Expand a geographic location to include related locations"""
        expanded_locations = [location_code]
        
        # If it's a state, include major cities
        major_cities = [city for city, state in self.city_to_state.items() if state == location_code]
        if major_cities:
            expanded_locations.extend(major_cities)
        
        return expanded_locations


class GeographicProcessor:
    """
    Advanced geographic processor with dynamic field discovery and Indian mapping
    """
    
    def __init__(self, schema_tables: Optional[List[str]] = None):
        """Initialize geographic processor with Indian geography knowledge"""
        
        # Schema tables
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
        
        # Initialize Indian geography mapper
        self.geography_mapper = IndianGeographyMapper()
        
        # Geographic field patterns for discovery
        self.geographic_field_patterns = self._initialize_geographic_patterns()
        
        # Location extraction patterns
        self.location_extraction_patterns = self._initialize_location_patterns()
        
        logger.info("GeographicProcessor initialized with Indian geography mapping")
    
    def process_geographic_query(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> GeographicProcessingResult:
        """
        Process geographic elements in natural language query
        
        Args:
            query_text: Query text containing geographic references
            context: Additional context including available fields
            
        Returns:
            Complete geographic processing result
        """
        with track_processing_time(ComponentType.NLP_ORCHESTRATOR, "process_geographic"):
            try:
                # Identify location mentions in query
                identified_locations = self._identify_locations_in_query(query_text)
                
                # Map locations to database codes
                geographic_mappings = []
                for location in identified_locations:
                    mapping = self.geography_mapper.map_location(location)
                    if mapping:
                        geographic_mappings.append(mapping)
                
                # Discover geographic fields in schema
                discovered_fields = self._discover_geographic_fields(context)
                
                # Create geographic filters
                geographic_filters = self._create_geographic_filters(
                    geographic_mappings, discovered_fields, query_text
                )
                
                # Generate SQL conditions
                sql_conditions = self._generate_sql_conditions(geographic_filters)
                
                # Determine geographic scope
                geographic_scope = self._determine_geographic_scope(geographic_mappings, query_text)
                
                # Extract business insights
                business_insights = self._extract_geographic_insights(
                    geographic_mappings, geographic_scope, query_text
                )
                
                result = GeographicProcessingResult(
                    original_query=query_text,
                    identified_locations=identified_locations,
                    geographic_mappings=geographic_mappings,
                    discovered_fields=discovered_fields,
                    geographic_filters=geographic_filters,
                    sql_conditions=sql_conditions,
                    geographic_scope=geographic_scope,
                    business_insights=business_insights
                )
                
                # Record metrics
                metrics_collector.record_accuracy(
                    ComponentType.NLP_ORCHESTRATOR,
                    len(geographic_mappings) > 0,
                    len(geographic_mappings) / max(len(identified_locations), 1),
                    "geographic_processing"
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Geographic processing failed: {e}")
                raise ValidationError(
                    validation_type="geographic_processing",
                    failed_rules=[str(e)]
                )
    
    def _initialize_geographic_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for discovering geographic fields"""
        return {
            "state_fields": [
                r".*state.*",
                r".*st.*",
                r".*province.*",
                r".*region.*"
            ],
            "city_fields": [
                r".*city.*",
                r".*town.*",
                r".*urban.*",
                r".*municipal.*"
            ],
            "district_fields": [
                r".*district.*",
                r".*dist.*",
                r".*area.*"
            ],
            "address_fields": [
                r".*address.*",
                r".*location.*",
                r".*place.*",
                r".*addr.*"
            ],
            "pincode_fields": [
                r".*pin.*code.*",
                r".*pincode.*",
                r".*postal.*code.*",
                r".*zip.*"
            ],
            "branch_fields": [
                r".*branch.*",
                r".*office.*",
                r".*center.*",
                r".*outlet.*"
            ]
        }
    
    def _initialize_location_patterns(self) -> List[str]:
        """Initialize patterns for extracting locations from text"""
        return [
            r'\bin\s+([A-Za-z\s]{2,20})\b',  # "in Delhi", "in Maharashtra"
            r'\bfrom\s+([A-Za-z\s]{2,20})\b',  # "from Mumbai"
            r'\bat\s+([A-Za-z\s]{2,20})\b',   # "at Bangalore"
            r'\bof\s+([A-Za-z\s]{2,20})\b',   # "of Gujarat"
            r'\b([A-Za-z\s]{2,20})\s+state\b',  # "Maharashtra state"
            r'\b([A-Za-z\s]{2,20})\s+city\b',   # "Mumbai city"
            r'\b([A-Za-z\s]{2,20})\s+region\b', # "North region"
        ]
    
    def _identify_locations_in_query(self, query_text: str) -> List[str]:
        """Identify location mentions in the query text"""
        locations = set()
        
        # Use patterns to extract potential locations
        for pattern in self.location_extraction_patterns:
            matches = re.finditer(pattern, query_text, re.IGNORECASE)
            for match in matches:
                location = match.group(1).strip()
                if len(location) > 2 and not location.lower() in ['the', 'and', 'for', 'with', 'are', 'all']:
                    locations.add(location)
        
        # Also check for direct mentions of known locations
        query_lower = query_text.lower()
        
        # Check for states
        for state_name in self.geography_mapper.state_mappings.keys():
            if state_name in query_lower:
                locations.add(state_name.title())
        
        # Check for cities
        for city_name in self.geography_mapper.city_to_state.keys():
            if city_name in query_lower:
                locations.add(city_name.title())
        
        # Check for regional terms
        regional_terms = ["north", "south", "east", "west", "central", "northeast", "northern", "southern", "eastern", "western"]
        for term in regional_terms:
            if f"{term} region" in query_lower or f"{term} zone" in query_lower or f"{term} india" in query_lower:
                locations.add(f"{term} region")
        
        return list(locations)
    
    def _discover_geographic_fields(self, context: Optional[Dict[str, Any]]) -> List[GeographicFieldDiscovery]:
        """Discover geographic fields in the schema dynamically"""
        discovered_fields = []
        
        if not context or "available_fields" not in context:
            return discovered_fields
        
        available_fields = context["available_fields"]
        
        for table_name, fields in available_fields.items():
            for field_name in fields:
                field_lower = field_name.lower()
                
                # Check against geographic patterns
                for geo_type, patterns in self.geographic_field_patterns.items():
                    for pattern in patterns:
                        if re.match(pattern, field_lower, re.IGNORECASE):
                            confidence = self._calculate_geographic_field_confidence(
                                field_name, table_name, geo_type
                            )
                            
                            geographic_level = self._infer_geographic_level_from_field(field_name, geo_type)
                            
                            discovered_fields.append(GeographicFieldDiscovery(
                                field_name=field_name,
                                table_name=table_name,
                                geographic_level=geographic_level,
                                confidence_score=confidence,
                                pattern_matched=pattern
                            ))
                            break  # Match found, no need to check other patterns
        
        # Sort by confidence
        discovered_fields.sort(key=lambda x: x.confidence_score, reverse=True)
        return discovered_fields
    
    def _calculate_geographic_field_confidence(self, field_name: str, table_name: str, geo_type: str) -> float:
        """Calculate confidence for geographic field mapping"""
        confidence = 0.5  # Base confidence
        
        field_lower = field_name.lower()
        table_lower = table_name.lower()
        
        # Direct keyword matches
        if geo_type == "state_fields" and "state" in field_lower:
            confidence += 0.4
        elif geo_type == "city_fields" and "city" in field_lower:
            confidence += 0.4
        elif geo_type == "address_fields" and "address" in field_lower:
            confidence += 0.3
        elif geo_type == "pincode_fields" and "pin" in field_lower:
            confidence += 0.4
        elif geo_type == "branch_fields" and "branch" in field_lower:
            confidence += 0.3
        
        # Table context relevance
        if "address" in table_lower:
            confidence += 0.2
        elif "location" in table_lower:
            confidence += 0.2
        elif "branch" in table_lower:
            confidence += 0.1
        
        # Field naming conventions
        if field_name.endswith("_CD") or field_name.endswith("_CODE"):
            confidence -= 0.1  # Codes might need different handling
        
        return min(1.0, confidence)
    
    def _infer_geographic_level_from_field(self, field_name: str, geo_type: str) -> GeographicLevel:
        """Infer geographic level from field name and type"""
        type_to_level = {
            "state_fields": GeographicLevel.STATE,
            "city_fields": GeographicLevel.CITY,
            "district_fields": GeographicLevel.DISTRICT,
            "address_fields": GeographicLevel.AREA,
            "pincode_fields": GeographicLevel.PINCODE,
            "branch_fields": GeographicLevel.AREA
        }
        
        return type_to_level.get(geo_type, GeographicLevel.AREA)
    
    def _create_geographic_filters(self, geographic_mappings: List[GeographicMapping], discovered_fields: List[GeographicFieldDiscovery], query_text: str) -> List[GeographicFilter]:
        """Create geographic filter conditions"""
        filters = []
        
        for mapping in geographic_mappings:
            # Find appropriate fields for this geographic level
            suitable_fields = [
                field for field in discovered_fields 
                if field.geographic_level == mapping.geographic_level or 
                   (mapping.geographic_level == GeographicLevel.CITY and field.geographic_level == GeographicLevel.STATE)
            ]
            
            if not suitable_fields:
                # Fallback to any geographic field
                suitable_fields = discovered_fields[:1]
            
            for field in suitable_fields[:2]:  # Use top 2 most relevant fields
                # Determine values to filter by
                filter_values = [mapping.mapped_value]
                
                # For cities, also include the parent state if field is state-level
                if (mapping.geographic_level == GeographicLevel.CITY and 
                    field.geographic_level == GeographicLevel.STATE and 
                    mapping.parent_geography):
                    filter_values = [mapping.parent_geography]
                
                # Check if query asks for inclusive scope
                include_children = any(term in query_text.lower() for term in ["including", "and surrounding", "region"])
                
                filters.append(GeographicFilter(
                    field_name=field.field_name,
                    table_name=field.table_name,
                    operator="IN" if len(filter_values) > 1 else "=",
                    values=filter_values,
                    geographic_level=mapping.geographic_level,
                    include_children=include_children
                ))
        
        return filters
    
    def _generate_sql_conditions(self, geographic_filters: List[GeographicFilter]) -> List[str]:
        """Generate SQL WHERE conditions for geographic filters"""
        sql_conditions = []
        
        for filter_condition in geographic_filters:
            table_alias = filter_condition.table_name.lower()[:3]  # Simple alias
            field_ref = f"{table_alias}.{filter_condition.field_name}"
            
            if filter_condition.operator == "IN":
                values_str = "', '".join(filter_condition.values)
                sql_condition = f"{field_ref} IN ('{values_str}')"
            else:
                sql_condition = f"{field_ref} = '{filter_condition.values[0]}'"
            
            sql_conditions.append(sql_condition)
            
            # Add comment for clarity
            sql_conditions.append(f"-- Geographic filter: {filter_condition.geographic_level.value}")
        
        return sql_conditions
    
    def _determine_geographic_scope(self, geographic_mappings: List[GeographicMapping], query_text: str) -> GeographicScope:
        """Determine the scope of geographic analysis"""
        
        if not geographic_mappings:
            return GeographicScope.NATIONAL
        
        # Check for regional indicators
        if any("region" in mapping.user_input.lower() for mapping in geographic_mappings):
            return GeographicScope.REGIONAL
        
        # Check levels of mappings
        levels = [mapping.geographic_level for mapping in geographic_mappings]
        
        if GeographicLevel.CITY in levels:
            return GeographicScope.CITY_WISE
        elif GeographicLevel.STATE in levels:
            return GeographicScope.STATE_WISE
        elif any("branch" in query_text.lower() for _ in [1]):
            return GeographicScope.BRANCH_WISE
        else:
            return GeographicScope.REGIONAL
    
    def _extract_geographic_insights(self, geographic_mappings: List[GeographicMapping], geographic_scope: GeographicScope, query_text: str) -> Dict[str, Any]:
        """Extract business insights from geographic processing"""
        
        insights = {
            "identified_locations": len(geographic_mappings),
            "geographic_scope": geographic_scope.value,
            "location_mappings": {},
            "confidence_analysis": {},
            "business_relevance": []
        }
        
        # Location mappings
        for mapping in geographic_mappings:
            insights["location_mappings"][mapping.user_input] = {
                "mapped_to": mapping.mapped_value,
                "confidence": mapping.confidence_score,
                "level": mapping.geographic_level.value
            }
        
        # Confidence analysis
        if geographic_mappings:
            confidences = [m.confidence_score for m in geographic_mappings]
            insights["confidence_analysis"] = {
                "average_confidence": sum(confidences) / len(confidences),
                "high_confidence_mappings": len([c for c in confidences if c > 0.8]),
                "low_confidence_mappings": len([c for c in confidences if c < 0.6])
            }
        
        # Business relevance
        if geographic_scope == GeographicScope.CITY_WISE:
            insights["business_relevance"].append("City-level analysis suitable for branch performance and local market insights")
        elif geographic_scope == GeographicScope.STATE_WISE:
            insights["business_relevance"].append("State-level analysis suitable for regulatory reporting and regional strategy")
        elif geographic_scope == GeographicScope.REGIONAL:
            insights["business_relevance"].append("Regional analysis suitable for zone-wise performance and market comparison")
        
        # Coverage analysis
        mapped_states = set()
        for mapping in geographic_mappings:
            if mapping.mapped_value:
                mapped_states.add(mapping.mapped_value)
            if mapping.parent_geography:
                mapped_states.add(mapping.parent_geography)
        
        insights["coverage_analysis"] = {
            "states_covered": list(mapped_states),
            "multi_state_analysis": len(mapped_states) > 1,
            "pan_india_scope": len(mapped_states) > 10
        }
        
        return insights
    
    def get_geographic_expansion(self, location_mappings: List[GeographicMapping], expansion_type: str = "related") -> Dict[str, List[str]]:
        """Get expanded geographic scope for broader analysis"""
        
        expanded_locations = {}
        
        for mapping in location_mappings:
            location_code = mapping.mapped_value
            
            if expansion_type == "related":
                # Include related cities/areas within the same state
                expanded = self.geography_mapper.expand_geographic_scope(location_code)
                expanded_locations[mapping.user_input] = expanded
                
            elif expansion_type == "regional":
                # Include neighboring states in the same region
                for region, states in self.geography_mapper.regional_groupings.items():
                    if location_code in states:
                        expanded_locations[mapping.user_input] = states
                        break
                        
            elif expansion_type == "banking_zone":
                # Include states in the same banking zone
                for zone, states in self.geography_mapper.banking_zones.items():
                    if location_code in states:
                        expanded_locations[mapping.user_input] = states
                        break
        
        return expanded_locations
    
    def validate_geographic_query(self, query_result: GeographicProcessingResult) -> Tuple[bool, List[str]]:
        """Validate geographic query processing results"""
        
        validation_errors = []
        
        # Check if locations were identified
        if not query_result.identified_locations:
            validation_errors.append("No geographic locations identified in query")
        
        # Check mapping success rate
        mapping_rate = len(query_result.geographic_mappings) / max(len(query_result.identified_locations), 1)
        if mapping_rate < 0.5:
            validation_errors.append(f"Low location mapping success rate: {mapping_rate:.2%}")
        
        # Check confidence levels
        if query_result.geographic_mappings:
            low_confidence_mappings = [m for m in query_result.geographic_mappings if m.confidence_score < 0.6]
            if len(low_confidence_mappings) > len(query_result.geographic_mappings) / 2:
                validation_errors.append("Many geographic mappings have low confidence")
        
        # Check if fields were discovered
        if not query_result.discovered_fields:
            validation_errors.append("No geographic fields discovered in schema")
        
        is_valid = len(validation_errors) == 0
        return is_valid, validation_errors


# Global geographic processor instance
geographic_processor = GeographicProcessor()
