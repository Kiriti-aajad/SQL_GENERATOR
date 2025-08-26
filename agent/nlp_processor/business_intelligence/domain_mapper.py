"""
Domain Mapper for Banking Business Intelligence
Maps business terms to database schema for banking domain queries
Provides intelligent mapping from analyst terminology to database fields
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import defaultdict
from pathlib import Path

from ..core.exceptions import DomainMappingError, NLPProcessorBaseException
from ..core.metrics import ComponentType, track_processing_time, metrics_collector


logger = logging.getLogger(__name__)


class MappingConfidence(Enum):
    """Confidence levels for domain mapping"""
    HIGH = "high"           # Direct, unambiguous mapping
    MEDIUM = "medium"       # Good mapping with minor ambiguity
    LOW = "low"            # Uncertain mapping, needs verification
    AMBIGUOUS = "ambiguous" # Multiple possible mappings


class BusinessDomain(Enum):
    """Business domains in banking"""
    CUSTOMER_MANAGEMENT = "customer_management"
    LOAN_PORTFOLIO = "loan_portfolio"
    DEPOSIT_MANAGEMENT = "deposit_management"
    RISK_MANAGEMENT = "risk_management"
    COMPLIANCE = "compliance"
    OPERATIONS = "operations"
    PERFORMANCE = "performance"
    COLLATERAL = "collateral"


@dataclass
class FieldMapping:
    """Mapping between business term and database field"""
    business_term: str
    table_name: str
    column_name: str
    confidence: MappingConfidence
    domain: BusinessDomain
    aliases: List[str] = field(default_factory=list)
    data_type: str = "VARCHAR"
    description: str = ""
    examples: List[str] = field(default_factory=list)
    context_hints: List[str] = field(default_factory=list)


@dataclass
class MappingResult:
    """Result of domain mapping operation"""
    business_term: str
    mappings: List[FieldMapping]
    best_mapping: Optional[FieldMapping] = None
    confidence_score: float = 0.0
    alternative_terms: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class DomainMapper:
    """
    Maps banking business terminology to database schema fields
    Provides intelligent mapping with context awareness and confidence scoring
    """
    
    def __init__(self, schema_path: Optional[str] = None):
        """Initialize domain mapper with banking schema knowledge"""
        self.schema_path = schema_path
        
        # Core banking schema mapping (22 verified tables)
        self.table_mappings = self._initialize_table_mappings()
        
        # Business vocabulary to field mappings
        self.business_mappings = self._initialize_business_mappings()
        
        # Synonym mappings for business terms
        self.synonym_mappings = self._initialize_synonym_mappings()
        
        # Context-based mapping rules
        self.context_rules = self._initialize_context_rules()
        
        # Ambiguity resolution patterns
        self.ambiguity_patterns = self._initialize_ambiguity_patterns()
        
        # Performance cache
        self.mapping_cache: Dict[str, MappingResult] = {}
        
        logger.info("DomainMapper initialized with banking schema knowledge")
    
    def map_business_term(self, business_term: str, context: Optional[Dict[str, Any]] = None) -> MappingResult:
        """
        Map a business term to database fields
        
        Args:
            business_term: Business terminology used by analyst
            context: Optional context for better mapping
            
        Returns:
            Mapping result with confidence scoring
        """
        with track_processing_time(ComponentType.DOMAIN_MAPPER, "map_business_term"):
            try:
                # Check cache first
                cache_key = f"{business_term.lower()}_{hash(str(context))}"
                if cache_key in self.mapping_cache:
                    return self.mapping_cache[cache_key]
                
                # Normalize business term
                normalized_term = self._normalize_term(business_term)
                
                # Find potential mappings
                potential_mappings = self._find_potential_mappings(normalized_term, context)
                
                # Apply context-based filtering
                if context:
                    potential_mappings = self._apply_context_filtering(potential_mappings, context)
                
                # Score and rank mappings
                scored_mappings = self._score_mappings(potential_mappings, normalized_term, context)
                
                # Determine best mapping
                best_mapping = scored_mappings[0] if scored_mappings else None
                confidence_score = self._calculate_confidence_score(scored_mappings)
                
                # Generate suggestions
                suggestions = self._generate_suggestions(business_term, scored_mappings)
                
                # Find alternative terms
                alternative_terms = self._find_alternative_terms(business_term)
                
                result = MappingResult(
                    business_term=business_term,
                    mappings=scored_mappings,
                    best_mapping=best_mapping,
                    confidence_score=confidence_score,
                    alternative_terms=alternative_terms,
                    suggestions=suggestions
                )
                
                # Cache result
                self.mapping_cache[cache_key] = result
                
                # Record metrics
                metrics_collector.record_accuracy(
                    ComponentType.DOMAIN_MAPPER,
                    confidence_score > 0.7,
                    confidence_score,
                    "domain_mapping"
                )
                
                # Check for mapping failure
                if not scored_mappings:
                    raise DomainMappingError(
                        business_term=business_term,
                        domain="banking"
                    )
                
                return result
                
            except Exception as e:
                logger.error(f"Domain mapping failed for '{business_term}': {e}")
                raise
    
    def map_multiple_terms(self, business_terms: List[str], context: Optional[Dict[str, Any]] = None) -> Dict[str, MappingResult]:
        """Map multiple business terms with cross-term optimization"""
        results = {}
        
        # First pass - individual mappings
        for term in business_terms:
            try:
                results[term] = self.map_business_term(term, context)
            except DomainMappingError:
                logger.warning(f"Failed to map term: {term}")
                continue
        
        # Second pass - optimize based on co-occurrence patterns
        self._optimize_cross_term_mappings(results, context)
        
        return results
    
    def suggest_related_fields(self, table_name: str, mapped_fields: List[str]) -> List[FieldMapping]:
        """Suggest related fields from the same table or related tables"""
        suggestions = []
        
        # Fields from same table
        if table_name in self.table_mappings:
            table_info = self.table_mappings[table_name]
            for field_name, field_info in table_info.get("columns", {}).items():
                if field_name not in mapped_fields:
                    suggestions.append(FieldMapping(
                        business_term=field_info.get("business_name", field_name),
                        table_name=table_name,
                        column_name=field_name,
                        confidence=MappingConfidence.MEDIUM,
                        domain=BusinessDomain(field_info.get("domain", "operations")),
                        description=field_info.get("description", ""),
                        data_type=field_info.get("data_type", "VARCHAR")
                    ))
        
        # Related table fields
        related_tables = self._find_related_tables(table_name)
        for related_table in related_tables[:3]:  # Limit to top 3 related tables
            if related_table in self.table_mappings:
                table_info = self.table_mappings[related_table]
                for field_name, field_info in list(table_info.get("columns", {}).items())[:5]:
                    suggestions.append(FieldMapping(
                        business_term=field_info.get("business_name", field_name),
                        table_name=related_table,
                        column_name=field_name,
                        confidence=MappingConfidence.LOW,
                        domain=BusinessDomain(field_info.get("domain", "operations")),
                        description=f"From related table: {field_info.get('description', '')}",
                        data_type=field_info.get("data_type", "VARCHAR")
                    ))
        
        return suggestions[:10]  # Return top 10 suggestions
    
    def validate_mapping(self, mapping: FieldMapping, query_context: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
        """Validate a field mapping against business rules"""
        validation_errors = []
        
        # Check if table exists
        if mapping.table_name not in self.table_mappings:
            validation_errors.append(f"Table '{mapping.table_name}' not found in schema")
        
        # Check if column exists
        elif mapping.column_name not in self.table_mappings[mapping.table_name].get("columns", {}):
            validation_errors.append(f"Column '{mapping.column_name}' not found in table '{mapping.table_name}'")
        
        # Context-specific validations
        if query_context:
            # Check domain consistency
            query_domain = query_context.get("domain")
            if query_domain and query_domain != mapping.domain.value:
                validation_errors.append(f"Domain mismatch: query expects {query_domain}, mapping is {mapping.domain.value}")
            
            # Check data type compatibility
            expected_type = query_context.get("expected_data_type")
            if expected_type and not self._is_type_compatible(mapping.data_type, expected_type):
                validation_errors.append(f"Data type mismatch: expected {expected_type}, got {mapping.data_type}")
        
        is_valid = len(validation_errors) == 0
        return is_valid, validation_errors
    
    def get_domain_vocabulary(self, domain: BusinessDomain) -> Dict[str, List[str]]:
        """Get comprehensive vocabulary for a specific business domain"""
        vocabulary = defaultdict(list)
        
        for term, mappings in self.business_mappings.items():
            for mapping in mappings:
                if mapping.domain == domain:
                    vocabulary[mapping.table_name].append(term)
                    vocabulary[mapping.table_name].extend(mapping.aliases)
        
        return dict(vocabulary)
    
    def _initialize_table_mappings(self) -> Dict[str, Any]:
        """Initialize core banking table mappings"""
        return {
            # Customer Management
            "customer_master": {
                "domain": BusinessDomain.CUSTOMER_MANAGEMENT,
                "description": "Master customer information",
                "columns": {
                    "customer_id": {"business_name": "customer id", "data_type": "VARCHAR", "domain": "customer_management"},
                    "customer_name": {"business_name": "customer name", "data_type": "VARCHAR", "domain": "customer_management"},
                    "customer_type": {"business_name": "customer type", "data_type": "VARCHAR", "domain": "customer_management"},
                    "date_of_birth": {"business_name": "date of birth", "data_type": "DATE", "domain": "customer_management"},
                    "pan_number": {"business_name": "pan number", "data_type": "VARCHAR", "domain": "customer_management"},
                    "mobile_number": {"business_name": "mobile number", "data_type": "VARCHAR", "domain": "customer_management"},
                    "email_id": {"business_name": "email id", "data_type": "VARCHAR", "domain": "customer_management"},
                    "address": {"business_name": "address", "data_type": "VARCHAR", "domain": "customer_management"},
                    "city": {"business_name": "city", "data_type": "VARCHAR", "domain": "customer_management"},
                    "state": {"business_name": "state", "data_type": "VARCHAR", "domain": "customer_management"},
                    "pincode": {"business_name": "pincode", "data_type": "VARCHAR", "domain": "customer_management"},
                    "occupation": {"business_name": "occupation", "data_type": "VARCHAR", "domain": "customer_management"},
                    "income": {"business_name": "income", "data_type": "DECIMAL", "domain": "customer_management"},
                    "branch_code": {"business_name": "branch code", "data_type": "VARCHAR", "domain": "operations"},
                    "relationship_manager": {"business_name": "relationship manager", "data_type": "VARCHAR", "domain": "customer_management"},
                    "customer_since": {"business_name": "customer since", "data_type": "DATE", "domain": "customer_management"},
                    "customer_status": {"business_name": "customer status", "data_type": "VARCHAR", "domain": "customer_management"}
                }
            },
            
            # Account Management
            "account_master": {
                "domain": BusinessDomain.DEPOSIT_MANAGEMENT,
                "description": "Master account information",
                "columns": {
                    "account_number": {"business_name": "account number", "data_type": "VARCHAR", "domain": "deposit_management"},
                    "customer_id": {"business_name": "customer id", "data_type": "VARCHAR", "domain": "customer_management"},
                    "account_type": {"business_name": "account type", "data_type": "VARCHAR", "domain": "deposit_management"},
                    "product_code": {"business_name": "product code", "data_type": "VARCHAR", "domain": "deposit_management"},
                    "opening_date": {"business_name": "opening date", "data_type": "DATE", "domain": "deposit_management"},
                    "closing_date": {"business_name": "closing date", "data_type": "DATE", "domain": "deposit_management"},
                    "account_balance": {"business_name": "account balance", "data_type": "DECIMAL", "domain": "deposit_management"},
                    "available_balance": {"business_name": "available balance", "data_type": "DECIMAL", "domain": "deposit_management"},
                    "branch_code": {"business_name": "branch code", "data_type": "VARCHAR", "domain": "operations"},
                    "currency": {"business_name": "currency", "data_type": "VARCHAR", "domain": "deposit_management"},
                    "interest_rate": {"business_name": "interest rate", "data_type": "DECIMAL", "domain": "deposit_management"},
                    "maturity_date": {"business_name": "maturity date", "data_type": "DATE", "domain": "deposit_management"},
                    "status": {"business_name": "account status", "data_type": "VARCHAR", "domain": "deposit_management"},
                    "freeze_amount": {"business_name": "freeze amount", "data_type": "DECIMAL", "domain": "deposit_management"},
                    "lien_amount": {"business_name": "lien amount", "data_type": "DECIMAL", "domain": "deposit_management"}
                }
            },
            
            # Loan Management
            "loan_master": {
                "domain": BusinessDomain.LOAN_PORTFOLIO,
                "description": "Master loan information",
                "columns": {
                    "loan_account_number": {"business_name": "loan account number", "data_type": "VARCHAR", "domain": "loan_portfolio"},
                    "customer_id": {"business_name": "customer id", "data_type": "VARCHAR", "domain": "customer_management"},
                    "loan_type": {"business_name": "loan type", "data_type": "VARCHAR", "domain": "loan_portfolio"},
                    "loan_product": {"business_name": "loan product", "data_type": "VARCHAR", "domain": "loan_portfolio"},
                    "sanctioned_amount": {"business_name": "sanctioned amount", "data_type": "DECIMAL", "domain": "loan_portfolio"},
                    "disbursed_amount": {"business_name": "disbursed amount", "data_type": "DECIMAL", "domain": "loan_portfolio"},
                    "outstanding_amount": {"business_name": "outstanding amount", "data_type": "DECIMAL", "domain": "loan_portfolio"},
                    "interest_rate": {"business_name": "interest rate", "data_type": "DECIMAL", "domain": "loan_portfolio"},
                    "tenure_months": {"business_name": "tenure months", "data_type": "INTEGER", "domain": "loan_portfolio"},
                    "emi_amount": {"business_name": "emi amount", "data_type": "DECIMAL", "domain": "loan_portfolio"},
                    "sanction_date": {"business_name": "sanction date", "data_type": "DATE", "domain": "loan_portfolio"},
                    "disbursement_date": {"business_name": "disbursement date", "data_type": "DATE", "domain": "loan_portfolio"},
                    "maturity_date": {"business_name": "maturity date", "data_type": "DATE", "domain": "loan_portfolio"},
                    "next_due_date": {"business_name": "next due date", "data_type": "DATE", "domain": "loan_portfolio"},
                    "overdue_amount": {"business_name": "overdue amount", "data_type": "DECIMAL", "domain": "risk_management"},
                    "dpd": {"business_name": "days past due", "data_type": "INTEGER", "domain": "risk_management"},
                    "npa_status": {"business_name": "npa status", "data_type": "VARCHAR", "domain": "risk_management"},
                    "branch_code": {"business_name": "branch code", "data_type": "VARCHAR", "domain": "operations"},
                    "loan_status": {"business_name": "loan status", "data_type": "VARCHAR", "domain": "loan_portfolio"}
                }
            },
            
            # Transaction Management
            "transaction_details": {
                "domain": BusinessDomain.OPERATIONS,
                "description": "Transaction details",
                "columns": {
                    "transaction_id": {"business_name": "transaction id", "data_type": "VARCHAR", "domain": "operations"},
                    "account_number": {"business_name": "account number", "data_type": "VARCHAR", "domain": "deposit_management"},
                    "transaction_date": {"business_name": "transaction date", "data_type": "DATE", "domain": "operations"},
                    "transaction_time": {"business_name": "transaction time", "data_type": "TIMESTAMP", "domain": "operations"},
                    "transaction_type": {"business_name": "transaction type", "data_type": "VARCHAR", "domain": "operations"},
                    "debit_amount": {"business_name": "debit amount", "data_type": "DECIMAL", "domain": "operations"},
                    "credit_amount": {"business_name": "credit amount", "data_type": "DECIMAL", "domain": "operations"},
                    "balance_after_transaction": {"business_name": "balance after transaction", "data_type": "DECIMAL", "domain": "operations"},
                    "transaction_channel": {"business_name": "transaction channel", "data_type": "VARCHAR", "domain": "operations"},
                    "reference_number": {"business_name": "reference number", "data_type": "VARCHAR", "domain": "operations"},
                    "description": {"business_name": "transaction description", "data_type": "VARCHAR", "domain": "operations"},
                    "branch_code": {"business_name": "branch code", "data_type": "VARCHAR", "domain": "operations"},
                    "user_id": {"business_name": "user id", "data_type": "VARCHAR", "domain": "operations"},
                    "authorization_status": {"business_name": "authorization status", "data_type": "VARCHAR", "domain": "operations"}
                }
            },
            
            # Collateral Management
            "collateral_master": {
                "domain": BusinessDomain.COLLATERAL,
                "description": "Collateral information",
                "columns": {
                    "collateral_id": {"business_name": "collateral id", "data_type": "VARCHAR", "domain": "collateral"},
                    "loan_account_number": {"business_name": "loan account number", "data_type": "VARCHAR", "domain": "loan_portfolio"},
                    "collateral_type": {"business_name": "collateral type", "data_type": "VARCHAR", "domain": "collateral"},
                    "collateral_value": {"business_name": "collateral value", "data_type": "DECIMAL", "domain": "collateral"},
                    "market_value": {"business_name": "market value", "data_type": "DECIMAL", "domain": "collateral"},
                    "forced_sale_value": {"business_name": "forced sale value", "data_type": "DECIMAL", "domain": "collateral"},
                    "valuation_date": {"business_name": "valuation date", "data_type": "DATE", "domain": "collateral"},
                    "next_valuation_date": {"business_name": "next valuation date", "data_type": "DATE", "domain": "collateral"},
                    "location": {"business_name": "collateral location", "data_type": "VARCHAR", "domain": "collateral"},
                    "ownership_type": {"business_name": "ownership type", "data_type": "VARCHAR", "domain": "collateral"},
                    "legal_status": {"business_name": "legal status", "data_type": "VARCHAR", "domain": "collateral"},
                    "insurance_status": {"business_name": "insurance status", "data_type": "VARCHAR", "domain": "collateral"},
                    "lien_status": {"business_name": "lien status", "data_type": "VARCHAR", "domain": "collateral"}
                }
            },
            
            # Branch Management
            "branch_master": {
                "domain": BusinessDomain.OPERATIONS,
                "description": "Branch information",
                "columns": {
                    "branch_code": {"business_name": "branch code", "data_type": "VARCHAR", "domain": "operations"},
                    "branch_name": {"business_name": "branch name", "data_type": "VARCHAR", "domain": "operations"},
                    "branch_type": {"business_name": "branch type", "data_type": "VARCHAR", "domain": "operations"},
                    "region": {"business_name": "region", "data_type": "VARCHAR", "domain": "operations"},
                    "zone": {"business_name": "zone", "data_type": "VARCHAR", "domain": "operations"},
                    "state": {"business_name": "state", "data_type": "VARCHAR", "domain": "operations"},
                    "city": {"business_name": "city", "data_type": "VARCHAR", "domain": "operations"},
                    "address": {"business_name": "branch address", "data_type": "VARCHAR", "domain": "operations"},
                    "pincode": {"business_name": "pincode", "data_type": "VARCHAR", "domain": "operations"},
                    "phone": {"business_name": "phone", "data_type": "VARCHAR", "domain": "operations"},
                    "email": {"business_name": "email", "data_type": "VARCHAR", "domain": "operations"},
                    "manager_name": {"business_name": "branch manager", "data_type": "VARCHAR", "domain": "operations"},
                    "opening_date": {"business_name": "opening date", "data_type": "DATE", "domain": "operations"},
                    "status": {"business_name": "branch status", "data_type": "VARCHAR", "domain": "operations"}
                }
            }
            # Additional 16 tables would be defined similarly...
        }
    
    def _initialize_business_mappings(self) -> Dict[str, List[FieldMapping]]:
        """Initialize business term to field mappings"""
        mappings = defaultdict(list)
        
        # Customer-related terms
        customer_terms = {
            "customer": [("customer_master", "customer_name"), ("customer_master", "customer_id")],
            "client": [("customer_master", "customer_name"), ("customer_master", "customer_id")],
            "borrower": [("loan_master", "customer_id"), ("customer_master", "customer_name")],
            "customer name": [("customer_master", "customer_name")],
            "customer id": [("customer_master", "customer_id")],
            "pan": [("customer_master", "pan_number")],
            "mobile": [("customer_master", "mobile_number")],
            "phone": [("customer_master", "mobile_number"), ("branch_master", "phone")],
            "email": [("customer_master", "email_id"), ("branch_master", "email")],
            "address": [("customer_master", "address"), ("branch_master", "address")],
            "income": [("customer_master", "income")],
            "occupation": [("customer_master", "occupation")]
        }
        
        # Account-related terms
        account_terms = {
            "account": [("account_master", "account_number")],
            "account number": [("account_master", "account_number")],
            "balance": [("account_master", "account_balance"), ("account_master", "available_balance")],
            "account balance": [("account_master", "account_balance")],
            "available balance": [("account_master", "available_balance")],
            "deposit": [("account_master", "account_balance")],
            "savings": [("account_master", "account_balance")],
            "current account": [("account_master", "account_number")],
            "fixed deposit": [("account_master", "account_number")],
            "fd": [("account_master", "account_number")],
            "interest rate": [("account_master", "interest_rate"), ("loan_master", "interest_rate")]
        }
        
        # Loan-related terms
        loan_terms = {
            "loan": [("loan_master", "loan_account_number")],
            "credit": [("loan_master", "loan_account_number")],
            "advance": [("loan_master", "loan_account_number")],
            "sanctioned amount": [("loan_master", "sanctioned_amount")],
            "disbursed amount": [("loan_master", "disbursed_amount")],
            "outstanding": [("loan_master", "outstanding_amount")],
            "outstanding amount": [("loan_master", "outstanding_amount")],
            "emi": [("loan_master", "emi_amount")],
            "installment": [("loan_master", "emi_amount")],
            "tenure": [("loan_master", "tenure_months")],
            "overdue": [("loan_master", "overdue_amount")],
            "dpd": [("loan_master", "dpd")],
            "days past due": [("loan_master", "dpd")],
            "npa": [("loan_master", "npa_status")],
            "non performing asset": [("loan_master", "npa_status")]
        }
        
        # Transaction-related terms
        transaction_terms = {
            "transaction": [("transaction_details", "transaction_id")],
            "payment": [("transaction_details", "transaction_id")],
            "transfer": [("transaction_details", "transaction_id")],
            "debit": [("transaction_details", "debit_amount")],
            "credit": [("transaction_details", "credit_amount")],
            "withdrawal": [("transaction_details", "debit_amount")],
            "deposit": [("transaction_details", "credit_amount")],
            "transaction amount": [("transaction_details", "debit_amount"), ("transaction_details", "credit_amount")],
            "reference number": [("transaction_details", "reference_number")]
        }
        
        # Collateral-related terms
        collateral_terms = {
            "collateral": [("collateral_master", "collateral_id")],
            "security": [("collateral_master", "collateral_id")],
            "guarantee": [("collateral_master", "collateral_id")],
            "pledge": [("collateral_master", "collateral_id")],
            "mortgage": [("collateral_master", "collateral_id")],
            "collateral value": [("collateral_master", "collateral_value")],
            "market value": [("collateral_master", "market_value")],
            "forced sale value": [("collateral_master", "forced_sale_value")]
        }
        
        # Branch/Location terms
        branch_terms = {
            "branch": [("branch_master", "branch_name"), ("branch_master", "branch_code")],
            "region": [("branch_master", "region")],
            "zone": [("branch_master", "zone")],
            "location": [("branch_master", "city"), ("branch_master", "state")],
            "state": [("branch_master", "state"), ("customer_master", "state")],
            "city": [("branch_master", "city"), ("customer_master", "city")],
            "branch manager": [("branch_master", "manager_name")]
        }
        
        # Convert to FieldMapping objects
        all_terms = {**customer_terms, **account_terms, **loan_terms, **transaction_terms, **collateral_terms, **branch_terms}
        
        for term, table_columns in all_terms.items():
            for table_name, column_name in table_columns:
                if table_name in self.table_mappings:
                    table_info = self.table_mappings[table_name]
                    column_info = table_info["columns"].get(column_name, {})
                    
                    mapping = FieldMapping(
                        business_term=term,
                        table_name=table_name,
                        column_name=column_name,
                        confidence=MappingConfidence.HIGH,
                        domain=BusinessDomain(column_info.get("domain", "operations")),
                        data_type=column_info.get("data_type", "VARCHAR"),
                        description=column_info.get("description", ""),
                        aliases=[]
                    )
                    mappings[term].append(mapping)
        
        return mappings
    
    def _initialize_synonym_mappings(self) -> Dict[str, List[str]]:
        """Initialize synonym mappings for business terms"""
        return {
            # Customer synonyms
            "customer": ["client", "borrower", "account holder", "customer"],
            "client": ["customer", "borrower", "account holder"],
            
            # Account synonyms
            "account": ["acc", "a/c", "account"],
            "balance": ["amount", "value", "sum"],
            "deposit": ["credit", "amount"],
            
            # Loan synonyms
            "loan": ["credit", "advance", "lending", "facility"],
            "outstanding": ["due", "balance", "pending"],
            "overdue": ["past due", "delinquent", "arrears"],
            "emi": ["installment", "repayment", "monthly payment"],
            
            # Transaction synonyms
            "transaction": ["txn", "payment", "transfer"],
            "debit": ["withdrawal", "outgoing", "payment"],
            "credit": ["deposit", "incoming", "receipt"],
            
            # Geographic synonyms
            "region": ["area", "territory", "zone"],
            "branch": ["location", "office", "center"],
            "state": ["province", "region"],
            
            # Time synonyms
            "date": ["time", "period", "when"],
            "amount": ["value", "sum", "total"],
            "status": ["condition", "state", "situation"]
        }
    
    def _initialize_context_rules(self) -> List[Dict[str, Any]]:
        """Initialize context-based mapping rules"""
        return [
            {
                "rule": "loan_context",
                "conditions": {"query_contains": ["loan", "credit", "advance"]},
                "boost_tables": ["loan_master", "collateral_master"],
                "boost_factor": 1.5
            },
            {
                "rule": "customer_context",
                "conditions": {"query_contains": ["customer", "client", "borrower"]},
                "boost_tables": ["customer_master"],
                "boost_factor": 1.3
            },
            {
                "rule": "transaction_context",
                "conditions": {"query_contains": ["payment", "transaction", "transfer"]},
                "boost_tables": ["transaction_details"],
                "boost_factor": 1.4
            },
            {
                "rule": "risk_context",
                "conditions": {"query_contains": ["risk", "npa", "default", "overdue"]},
                "boost_domain": BusinessDomain.RISK_MANAGEMENT,
                "boost_factor": 1.6
            },
            {
                "rule": "geographic_context",
                "conditions": {"query_contains": ["region", "state", "city", "branch"]},
                "boost_tables": ["branch_master"],
                "boost_factor": 1.2
            }
        ]
    
    def _initialize_ambiguity_patterns(self) -> List[Dict[str, Any]]:
        """Initialize ambiguity resolution patterns"""
        return [
            {
                "pattern": "amount_disambiguation",
                "ambiguous_terms": ["amount", "value", "sum"],
                "resolution_rules": [
                    {"context": ["loan"], "prefer": "sanctioned_amount"},
                    {"context": ["outstanding"], "prefer": "outstanding_amount"},
                    {"context": ["transaction"], "prefer": "debit_amount"},
                    {"context": ["balance"], "prefer": "account_balance"}
                ]
            },
            {
                "pattern": "date_disambiguation",
                "ambiguous_terms": ["date", "time"],
                "resolution_rules": [
                    {"context": ["transaction"], "prefer": "transaction_date"},
                    {"context": ["opening"], "prefer": "opening_date"},
                    {"context": ["maturity"], "prefer": "maturity_date"},
                    {"context": ["due"], "prefer": "next_due_date"}
                ]
            }
        ]
    
    def _normalize_term(self, term: str) -> str:
        """Normalize business term for mapping"""
        # Convert to lowercase
        normalized = term.lower().strip()
        
        # Remove special characters
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Remove extra spaces
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Handle common abbreviations
        abbreviations = {
            'acc': 'account',
            'txn': 'transaction',
            'cr': 'crore',
            'lac': 'lakh',
            'no': 'number',
            'amt': 'amount',
            'dt': 'date'
        }
        
        for abbr, full in abbreviations.items():
            normalized = normalized.replace(f' {abbr} ', f' {full} ')
            normalized = normalized.replace(f'{abbr} ', f'{full} ')
            normalized = normalized.replace(f' {abbr}', f' {full}')
        
        return normalized.strip()
    
    def _find_potential_mappings(self, normalized_term: str, context: Optional[Dict[str, Any]] = None) -> List[FieldMapping]:
        """Find potential field mappings for normalized term"""
        potential_mappings = []
        
        # Direct mapping lookup
        if normalized_term in self.business_mappings:
            potential_mappings.extend(self.business_mappings[normalized_term])
        
        # Synonym-based lookup
        for synonym_group in self.synonym_mappings.values():
            if normalized_term in synonym_group:
                for synonym in synonym_group:
                    if synonym in self.business_mappings:
                        for mapping in self.business_mappings[synonym]:
                            if mapping not in potential_mappings:
                                potential_mappings.append(mapping)
        
        # Partial matching
        words = normalized_term.split()
        if len(words) > 1:
            for word in words:
                if word in self.business_mappings:
                    for mapping in self.business_mappings[word]:
                        # Reduce confidence for partial matches
                        partial_mapping = FieldMapping(
                            business_term=mapping.business_term,
                            table_name=mapping.table_name,
                            column_name=mapping.column_name,
                            confidence=MappingConfidence.MEDIUM if mapping.confidence == MappingConfidence.HIGH else MappingConfidence.LOW,
                            domain=mapping.domain,
                            data_type=mapping.data_type,
                            description=mapping.description,
                            aliases=mapping.aliases
                        )
                        if partial_mapping not in potential_mappings:
                            potential_mappings.append(partial_mapping)
        
        return potential_mappings
    
    def _apply_context_filtering(self, mappings: List[FieldMapping], context: Dict[str, Any]) -> List[FieldMapping]:
        """Apply context-based filtering to mappings"""
        filtered_mappings = []
        
        for mapping in mappings:
            # Check domain context
            if "domain" in context and context["domain"] != mapping.domain.value:
                continue
            
            # Check table context
            if "preferred_tables" in context and mapping.table_name not in context["preferred_tables"]:
                continue
            
            # Check data type context
            if "data_type" in context and not self._is_type_compatible(mapping.data_type, context["data_type"]):
                continue
            
            filtered_mappings.append(mapping)
        
        return filtered_mappings or mappings  # Return original if no matches
    
    def _score_mappings(self, mappings: List[FieldMapping], normalized_term: str, context: Optional[Dict[str, Any]]) -> List[FieldMapping]:
        """Score and rank mappings based on relevance"""
        scored_mappings = []
        
        for mapping in mappings:
            score = self._calculate_mapping_score(mapping, normalized_term, context)
            mapping.confidence = self._score_to_confidence(score)
            scored_mappings.append((score, mapping))
        
        # Sort by score (descending)
        scored_mappings.sort(key=lambda x: x[0], reverse=True)
        
        return [mapping for score, mapping in scored_mappings]
    
    def _calculate_mapping_score(self, mapping: FieldMapping, normalized_term: str, context: Optional[Dict[str, Any]]) -> float:
        """Calculate relevance score for a mapping"""
        base_score = 1.0
        
        # Exact match bonus
        if mapping.business_term.lower() == normalized_term:
            base_score += 2.0
        
        # Partial match scoring
        term_words = set(normalized_term.split())
        mapping_words = set(mapping.business_term.lower().split())
        overlap = len(term_words & mapping_words)
        if overlap > 0:
            base_score += overlap * 0.5
        
        # Context-based boosting
        if context:
            query_text = context.get("query_text", "").lower()
            
            # Apply context rules
            for rule in self.context_rules:
                conditions = rule.get("conditions", {})
                query_contains = conditions.get("query_contains", [])
                
                if any(term in query_text for term in query_contains):
                    boost_factor = rule.get("boost_factor", 1.0)
                    
                    # Table-specific boost
                    if "boost_tables" in rule and mapping.table_name in rule["boost_tables"]:
                        base_score *= boost_factor
                    
                    # Domain-specific boost
                    if "boost_domain" in rule and mapping.domain == rule["boost_domain"]:
                        base_score *= boost_factor
        
        # Data type relevance
        if context and "expected_data_type" in context:
            if self._is_type_compatible(mapping.data_type, context["expected_data_type"]):
                base_score += 0.5
        
        return base_score
    
    def _score_to_confidence(self, score: float) -> MappingConfidence:
        """Convert numeric score to confidence level"""
        if score >= 3.0:
            return MappingConfidence.HIGH
        elif score >= 2.0:
            return MappingConfidence.MEDIUM
        elif score >= 1.0:
            return MappingConfidence.LOW
        else:
            return MappingConfidence.AMBIGUOUS
    
    def _calculate_confidence_score(self, mappings: List[FieldMapping]) -> float:
        """Calculate overall confidence score"""
        if not mappings:
            return 0.0
        
        best_mapping = mappings[0]
        confidence_values = {
            MappingConfidence.HIGH: 0.9,
            MappingConfidence.MEDIUM: 0.7,
            MappingConfidence.LOW: 0.5,
            MappingConfidence.AMBIGUOUS: 0.3
        }
        
        return confidence_values.get(best_mapping.confidence, 0.5)
    
    def _generate_suggestions(self, business_term: str, mappings: List[FieldMapping]) -> List[str]:
        """Generate helpful suggestions for mapping improvements"""
        suggestions = []
        
        if not mappings:
            suggestions.append(f"Try using more specific banking terminology instead of '{business_term}'")
            suggestions.append("Consider using terms like 'customer name', 'account balance', or 'loan amount'")
        
        elif len(mappings) > 1:
            suggestions.append(f"Multiple mappings found for '{business_term}'. Consider being more specific:")
            for mapping in mappings[:3]:
                suggestions.append(f"- Use '{mapping.business_term}' for {mapping.table_name}.{mapping.column_name}")
        
        elif mappings[0].confidence == MappingConfidence.LOW:
            suggestions.append(f"Low confidence mapping for '{business_term}'")
            suggestions.append(f"Consider using '{mappings[0].business_term}' instead")
        
        return suggestions
    
    def _find_alternative_terms(self, business_term: str) -> List[str]:
        """Find alternative terms for the given business term"""
        alternatives = []
        normalized_term = self._normalize_term(business_term)
        
        # Find synonyms
        for term, synonyms in self.synonym_mappings.items():
            if normalized_term == term or normalized_term in synonyms:
                alternatives.extend([s for s in synonyms if s != normalized_term])
                break
        
        # Find related business terms
        for term in self.business_mappings.keys():
            if term != normalized_term and any(word in term.split() for word in normalized_term.split()):
                alternatives.append(term)
        
        return list(set(alternatives))[:5]  # Return top 5 unique alternatives
    
    def _optimize_cross_term_mappings(self, results: Dict[str, MappingResult], context: Optional[Dict[str, Any]]) -> None:
        """Optimize mappings based on cross-term relationships"""
        # This could be enhanced with machine learning in the future
        # For now, implement basic consistency checks
        
        mapped_tables = set()
        for result in results.values():
            if result.best_mapping:
                mapped_tables.add(result.best_mapping.table_name)
        
        # Boost related table mappings
        for result in results.values():
            if result.best_mapping and len(result.mappings) > 1:
                for mapping in result.mappings[1:]:
                    if mapping.table_name in mapped_tables:
                        # Move related table mapping up if it's close in score
                        if mapping.confidence.value != MappingConfidence.AMBIGUOUS.value:
                            result.best_mapping = mapping
                            break
    
    def _find_related_tables(self, table_name: str) -> List[str]:
        """Find tables related to the given table"""
        # This would ideally use your 173 verified joins
        # For now, implement basic relationship detection
        
        related_tables = []
        
        # Customer-centric relationships
        if table_name == "customer_master":
            related_tables = ["account_master", "loan_master", "transaction_details"]
        elif table_name == "account_master":
            related_tables = ["customer_master", "transaction_details", "branch_master"]
        elif table_name == "loan_master":
            related_tables = ["customer_master", "collateral_master", "branch_master"]
        elif table_name == "transaction_details":
            related_tables = ["account_master", "customer_master", "branch_master"]
        elif table_name == "collateral_master":
            related_tables = ["loan_master", "customer_master"]
        elif table_name == "branch_master":
            related_tables = ["customer_master", "account_master", "loan_master"]
        
        return related_tables
    
    def _is_type_compatible(self, mapping_type: str, expected_type: str) -> bool:
        """Check if data types are compatible"""
        type_compatibility = {
            "VARCHAR": ["TEXT", "STRING", "CHAR"],
            "DECIMAL": ["NUMERIC", "FLOAT", "DOUBLE", "MONEY"],
            "INTEGER": ["INT", "BIGINT", "SMALLINT"],
            "DATE": ["DATETIME", "TIMESTAMP"],
            "TIMESTAMP": ["DATETIME", "DATE"]
        }
        
        mapping_type = mapping_type.upper()
        expected_type = expected_type.upper()
        
        if mapping_type == expected_type:
            return True
        
        return expected_type in type_compatibility.get(mapping_type, [])


# Global domain mapper instance
domain_mapper = DomainMapper()
