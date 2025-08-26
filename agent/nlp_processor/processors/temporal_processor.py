"""
Temporal Processor for Banking Domain Queries
Advanced temporal query processing with banking-specific time handling
Supports Indian financial calendar, business days, and analyst time patterns
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse as date_parse
import calendar
import holidays # pyright: ignore[reportMissingImports]

from ..core.exceptions import TemporalProcessingError, NLPProcessorBaseException
from ..core.metrics import ComponentType, track_processing_time, metrics_collector


logger = logging.getLogger(__name__)


class TemporalType(Enum):
    """Types of temporal expressions"""
    ABSOLUTE_DATE = "absolute_date"           # 2024-01-15, 15-Jan-2024
    RELATIVE_DATE = "relative_date"           # last 30 days, previous quarter
    RANGE = "range"                           # from 2024-01-01 to 2024-01-31
    PERIOD = "period"                         # Q1 2024, FY2024, January 2024
    BUSINESS_TIME = "business_time"           # last working day, next business quarter
    RECURRING = "recurring"                   # every month, quarterly, annually


class BusinessCalendar(Enum):
    """Business calendar types"""
    FINANCIAL_YEAR = "financial_year"         # April to March (Indian FY)
    CALENDAR_YEAR = "calendar_year"           # January to December
    BUSINESS_QUARTER = "business_quarter"     # Q1: Apr-Jun, Q2: Jul-Sep, etc.
    CALENDAR_QUARTER = "calendar_quarter"     # Q1: Jan-Mar, Q2: Apr-Jun, etc.


class TimeGranularity(Enum):
    """Time granularity levels"""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


@dataclass
class TemporalRange:
    """Represents a temporal range"""
    start_date: datetime
    end_date: datetime
    granularity: TimeGranularity
    calendar_type: BusinessCalendar
    is_business_days_only: bool = False
    timezone: str = "Asia/Kolkata"
    description: str = ""


@dataclass
class TemporalExpression:
    """Represents a parsed temporal expression"""
    original_text: str
    temporal_type: TemporalType
    parsed_range: TemporalRange
    confidence_score: float
    business_context: Dict[str, Any] = field(default_factory=dict)
    sql_conditions: List[str] = field(default_factory=list)
    optimization_hints: List[str] = field(default_factory=list)


@dataclass
class TemporalProcessingResult:
    """Result of temporal processing"""
    original_query: str
    identified_expressions: List[str]
    processed_expressions: List[TemporalExpression]
    recommended_range: Optional[TemporalRange]
    sql_where_clauses: List[str]
    performance_optimizations: List[str] = field(default_factory=list)
    business_insights: Dict[str, Any] = field(default_factory=dict)


class TemporalProcessor:
    """
    Advanced temporal processing for banking domain queries
    Handles Indian financial calendar and business-specific temporal logic
    """
    
    def __init__(self):
        """Initialize temporal processor with banking calendar knowledge"""
        
        # Indian holidays for business day calculations
        self.india_holidays = holidays.India()
        
        # Banking-specific temporal patterns
        self.temporal_patterns = self._initialize_temporal_patterns()
        
        # Financial year configuration (Indian: April-March)
        self.fy_config = self._initialize_fy_config()
        
        # Business day rules
        self.business_day_rules = self._initialize_business_day_rules()
        
        # Common banking time periods
        self.banking_periods = self._initialize_banking_periods()
        
        # Performance optimization rules
        self.optimization_rules = self._initialize_optimization_rules()
        
        logger.info("TemporalProcessor initialized with Indian banking calendar")
    
    def process_temporal_expressions(self, query_text: str, context: Optional[Dict[str, Any]] = None) -> TemporalProcessingResult:
        """
        Process all temporal expressions in a query
        
        Args:
            query_text: Query text containing temporal expressions
            context: Additional context (user timezone, default period, etc.)
            
        Returns:
            Complete temporal processing result
        """
        with track_processing_time(ComponentType.TEMPORAL_PROCESSOR, "process_temporal"):
            try:
                # Identify temporal expressions in text
                identified_expressions = self._identify_temporal_expressions(query_text)
                
                # Process each expression
                processed_expressions = []
                for expr_text in identified_expressions:
                    try:
                        processed_expr = self._process_single_expression(expr_text, context)
                        processed_expressions.append(processed_expr)
                    except TemporalProcessingError as e:
                        logger.warning(f"Failed to process temporal expression '{expr_text}': {e}")
                        continue
                
                # Select recommended range
                recommended_range = self._select_recommended_range(processed_expressions, context)
                
                # Generate SQL conditions
                sql_where_clauses = self._generate_sql_conditions(processed_expressions)
                
                # Generate performance optimizations
                performance_optimizations = self._generate_performance_optimizations(
                    processed_expressions, context
                )
                
                # Extract business insights
                business_insights = self._extract_business_insights(
                    processed_expressions, context
                )
                
                result = TemporalProcessingResult(
                    original_query=query_text,
                    identified_expressions=identified_expressions,
                    processed_expressions=processed_expressions,
                    recommended_range=recommended_range,
                    sql_where_clauses=sql_where_clauses,
                    performance_optimizations=performance_optimizations,
                    business_insights=business_insights
                )
                
                # Record metrics
                metrics_collector.record_accuracy(
                    ComponentType.TEMPORAL_PROCESSOR,
                    len(processed_expressions) == len(identified_expressions),
                    len(processed_expressions) / max(len(identified_expressions), 1),
                    "temporal_processing"
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Temporal processing failed: {e}")
                raise TemporalProcessingError(
                    temporal_expression=query_text[:50] + "..."
                )
    
    def calculate_business_days(self, start_date: datetime, end_date: datetime) -> int:
        """
        Calculate business days between two dates (excluding Indian holidays)
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Number of business days
        """
        if start_date > end_date:
            return 0
        
        business_days = 0
        current_date = start_date.date()
        end_date_only = end_date.date()
        
        while current_date <= end_date_only:
            # Skip weekends and Indian holidays
            if (current_date.weekday() < 5 and  # Monday=0, Sunday=6
                current_date not in self.india_holidays):
                business_days += 1
            current_date += timedelta(days=1)
        
        return business_days
    
    def get_financial_year_range(self, fy_year: int) -> TemporalRange:
        """
        Get date range for Indian financial year
        
        Args:
            fy_year: Financial year (e.g., 2024 for FY2023-24)
            
        Returns:
            TemporalRange for the financial year
        """
        start_date = datetime(fy_year - 1, 4, 1)  # April 1st of previous year
        end_date = datetime(fy_year, 3, 31, 23, 59, 59)  # March 31st of current year
        
        return TemporalRange(
            start_date=start_date,
            end_date=end_date,
            granularity=TimeGranularity.YEAR,
            calendar_type=BusinessCalendar.FINANCIAL_YEAR,
            description=f"Financial Year {fy_year-1}-{str(fy_year)[-2:]}"
        )
    
    def get_quarter_range(self, quarter: int, year: int, calendar_type: BusinessCalendar = BusinessCalendar.FINANCIAL_YEAR) -> TemporalRange:
        """
        Get date range for a specific quarter
        
        Args:
            quarter: Quarter number (1-4)
            year: Year
            calendar_type: Financial or calendar year
            
        Returns:
            TemporalRange for the quarter
        """
        if calendar_type == BusinessCalendar.FINANCIAL_YEAR:
            # Indian financial year quarters
            quarter_months = {
                1: (4, 6),   # Apr-Jun
                2: (7, 9),   # Jul-Sep
                3: (10, 12), # Oct-Dec
                4: (1, 3)    # Jan-Mar
            }
            
            start_month, end_month = quarter_months[quarter]
            
            if quarter == 4:  # Jan-Mar of next calendar year
                start_date = datetime(year, start_month, 1)
                end_date = datetime(year + 1, end_month, self._get_last_day_of_month(year + 1, end_month), 23, 59, 59)
            else:
                start_date = datetime(year - 1 if start_month >= 4 else year, start_month, 1)
                end_date = datetime(year - 1 if start_month >= 4 else year, end_month, self._get_last_day_of_month(year - 1 if start_month >= 4 else year, end_month), 23, 59, 59)
        
        else:  # Calendar year quarters
            quarter_months = {
                1: (1, 3),   # Jan-Mar
                2: (4, 6),   # Apr-Jun
                3: (7, 9),   # Jul-Sep
                4: (10, 12)  # Oct-Dec
            }
            
            start_month, end_month = quarter_months[quarter]
            start_date = datetime(year, start_month, 1)
            end_date = datetime(year, end_month, self._get_last_day_of_month(year, end_month), 23, 59, 59)
        
        return TemporalRange(
            start_date=start_date,
            end_date=end_date,
            granularity=TimeGranularity.QUARTER,
            calendar_type=calendar_type,
            description=f"Q{quarter} {year} ({'FY' if calendar_type == BusinessCalendar.FINANCIAL_YEAR else 'CY'})"
        )
    
    def parse_relative_date(self, relative_expression: str, reference_date: Optional[datetime] = None) -> TemporalRange:
        """
        Parse relative date expressions like 'last 30 days', 'previous quarter'
        
        Args:
            relative_expression: Relative date expression
            reference_date: Reference date (defaults to today)
            
        Returns:
            TemporalRange for the relative period
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        expr = relative_expression.lower().strip()
        
        # Last X days/weeks/months/years
        if match := re.search(r'last\s+(\d+)\s+(days?|weeks?|months?|years?)', expr):
            count = int(match.group(1))
            unit = match.group(2).rstrip('s')
            
            if unit == 'day':
                start_date = reference_date - timedelta(days=count)
                end_date = reference_date
                granularity = TimeGranularity.DAY
            elif unit == 'week':
                start_date = reference_date - timedelta(weeks=count)
                end_date = reference_date
                granularity = TimeGranularity.WEEK
            elif unit == 'month':
                start_date = reference_date - relativedelta(months=count)
                end_date = reference_date
                granularity = TimeGranularity.MONTH
            elif unit == 'year':
                start_date = reference_date - relativedelta(years=count)
                end_date = reference_date
                granularity = TimeGranularity.YEAR
            
            return TemporalRange(
                start_date=start_date, # pyright: ignore[reportPossiblyUnboundVariable]
                end_date=end_date, # pyright: ignore[reportPossiblyUnboundVariable]
                granularity=granularity, # pyright: ignore[reportPossiblyUnboundVariable]
                calendar_type=BusinessCalendar.CALENDAR_YEAR,
                description=f"Last {count} {unit}{'s' if count > 1 else ''}"
            )
        
        # Previous/last quarter
        elif 'previous quarter' in expr or 'last quarter' in expr:
            current_quarter = self._get_current_financial_quarter(reference_date)
            if current_quarter == 1:
                prev_quarter = 4
                prev_year = reference_date.year - 1
            else:
                prev_quarter = current_quarter - 1
                prev_year = reference_date.year
            
            return self.get_quarter_range(prev_quarter, prev_year, BusinessCalendar.FINANCIAL_YEAR)
        
        # This/current quarter
        elif 'this quarter' in expr or 'current quarter' in expr:
            current_quarter = self._get_current_financial_quarter(reference_date)
            return self.get_quarter_range(current_quarter, reference_date.year, BusinessCalendar.FINANCIAL_YEAR)
        
        # Previous/last financial year
        elif 'previous fy' in expr or 'last fy' in expr or 'previous financial year' in expr:
            return self.get_financial_year_range(reference_date.year - 1)
        
        # Current financial year
        elif 'current fy' in expr or 'this fy' in expr or 'current financial year' in expr:
            current_fy = self._get_current_financial_year(reference_date)
            return self.get_financial_year_range(current_fy)
        
        # Year to date (YTD) - from start of current FY to today
        elif 'ytd' in expr or 'year to date' in expr:
            current_fy = self._get_current_financial_year(reference_date)
            fy_range = self.get_financial_year_range(current_fy)
            
            return TemporalRange(
                start_date=fy_range.start_date,
                end_date=reference_date,
                granularity=TimeGranularity.DAY,
                calendar_type=BusinessCalendar.FINANCIAL_YEAR,
                description="Year to Date"
            )
        
        # Month to date (MTD)
        elif 'mtd' in expr or 'month to date' in expr:
            month_start = reference_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            return TemporalRange(
                start_date=month_start,
                end_date=reference_date,
                granularity=TimeGranularity.DAY,
                calendar_type=BusinessCalendar.CALENDAR_YEAR,
                description="Month to Date"
            )
        
        # Today/yesterday/tomorrow
        elif expr in ['today', 'today\'s']:
            start_date = reference_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = reference_date.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            return TemporalRange(
                start_date=start_date,
                end_date=end_date,
                granularity=TimeGranularity.DAY,
                calendar_type=BusinessCalendar.CALENDAR_YEAR,
                description="Today"
            )
        
        elif expr in ['yesterday', 'yesterday\'s']:
            yesterday = reference_date - timedelta(days=1)
            start_date = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            return TemporalRange(
                start_date=start_date,
                end_date=end_date,
                granularity=TimeGranularity.DAY,
                calendar_type=BusinessCalendar.CALENDAR_YEAR,
                description="Yesterday"
            )
        
        else:
            raise TemporalProcessingError(
                temporal_expression=relative_expression
            )
    
    def _initialize_temporal_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for temporal expression recognition"""
        return {
            "relative_days": [
                r'\blast\s+\d+\s+days?\b',
                r'\bprevious\s+\d+\s+days?\b',
                r'\bpast\s+\d+\s+days?\b'
            ],
            "relative_periods": [
                r'\blast\s+(?:week|month|quarter|year)\b',
                r'\bprevious\s+(?:week|month|quarter|year)\b',
                r'\bcurrent\s+(?:week|month|quarter|year)\b',
                r'\bthis\s+(?:week|month|quarter|year)\b'
            ],
            "financial_periods": [
                r'\bfy\s*\d{4}\b',
                r'\bfinancial\s+year\s+\d{4}\b',
                r'\bq[1-4]\s+\d{4}\b',
                r'\bquarter\s+[1-4]\s+\d{4}\b'
            ],
            "specific_dates": [
                r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
                r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
                r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
            ],
            "banking_periods": [
                r'\bytd\b|\byear\s+to\s+date\b',
                r'\bmtd\b|\bmonth\s+to\s+date\b',
                r'\bqtd\b|\bquarter\s+to\s+date\b',
                r'\beom\b|\bend\s+of\s+month\b',
                r'\beoq\b|\bend\s+of\s+quarter\b',
                r'\beoy\b|\bend\s+of\s+year\b'
            ],
            "business_days": [
                r'\bbusiness\s+days?\b',
                r'\bworking\s+days?\b',
                r'\bweekdays?\b'
            ]
        }
    
    def _initialize_fy_config(self) -> Dict[str, Any]:
        """Initialize financial year configuration"""
        return {
            "start_month": 4,  # April
            "start_day": 1,
            "end_month": 3,    # March
            "end_day": 31,
            "quarters": {
                1: {"start": (4, 1), "end": (6, 30), "name": "Q1"},
                2: {"start": (7, 1), "end": (9, 30), "name": "Q2"},
                3: {"start": (10, 1), "end": (12, 31), "name": "Q3"},
                4: {"start": (1, 1), "end": (3, 31), "name": "Q4"}
            }
        }
    
    def _initialize_business_day_rules(self) -> Dict[str, Any]:
        """Initialize business day calculation rules"""
        return {
            "weekend_days": [5, 6],  # Saturday, Sunday
            "holiday_calendar": "india",
            "banking_holidays": [
                # Additional banking-specific holidays
                "Second Saturday",
                "Fourth Saturday"
            ]
        }
    
    def _initialize_banking_periods(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common banking time periods"""
        return {
            "reporting_cycles": {
                "daily": {"frequency": "daily", "granularity": TimeGranularity.DAY},
                "weekly": {"frequency": "weekly", "granularity": TimeGranularity.WEEK},
                "fortnightly": {"frequency": "bi-weekly", "granularity": TimeGranularity.WEEK},
                "monthly": {"frequency": "monthly", "granularity": TimeGranularity.MONTH},
                "quarterly": {"frequency": "quarterly", "granularity": TimeGranularity.QUARTER},
                "annual": {"frequency": "annual", "granularity": TimeGranularity.YEAR}
            },
            "compliance_periods": {
                "crar_reporting": {"frequency": "quarterly", "due_days": 21},
                "alm_reporting": {"frequency": "monthly", "due_days": 15},
                "rbi_returns": {"frequency": "monthly", "due_days": 10}
            }
        }
    
    def _initialize_optimization_rules(self) -> Dict[str, Any]:
        """Initialize performance optimization rules"""
        return {
            "index_hints": {
                "date_columns": ["transaction_date", "created_date", "modified_date", "effective_date"],
                "partition_suggestions": {
                    "daily_partitions": ["transaction_details", "audit_trail"],
                    "monthly_partitions": ["historical_data", "archived_records"]
                }
            },
            "query_optimizations": {
                "range_queries": "Use date ranges instead of functions on date columns",
                "index_usage": "Ensure date filters use indexed columns",
                "partition_elimination": "Align queries with partition boundaries"
            }
        }
    
    def _identify_temporal_expressions(self, query_text: str) -> List[str]:
        """Identify temporal expressions in query text"""
        expressions = []
        query_lower = query_text.lower()
        
        for category, patterns in self.temporal_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query_lower, re.IGNORECASE)
                for match in matches:
                    expressions.append(match.group(0))
        
        return list(set(expressions))  # Remove duplicates
    
    def _process_single_expression(self, expression: str, context: Optional[Dict[str, Any]]) -> TemporalExpression:
        """Process a single temporal expression"""
        expr_lower = expression.lower().strip()
        
        # Determine temporal type
        temporal_type = self._determine_temporal_type(expr_lower)
        
        # Parse based on type
        if temporal_type == TemporalType.RELATIVE_DATE:
            parsed_range = self.parse_relative_date(expr_lower, context.get('reference_date') if context else None)
        elif temporal_type == TemporalType.ABSOLUTE_DATE:
            parsed_range = self._parse_absolute_date(expr_lower)
        elif temporal_type == TemporalType.PERIOD:
            parsed_range = self._parse_period_expression(expr_lower)
        elif temporal_type == TemporalType.RANGE:
            parsed_range = self._parse_range_expression(expr_lower)
        else:
            parsed_range = self._parse_generic_expression(expr_lower)
        
        # Calculate confidence
        confidence_score = self._calculate_confidence(expression, parsed_range)
        
        # Generate SQL conditions
        sql_conditions = self._generate_sql_conditions_for_range(parsed_range)
        
        # Generate optimization hints
        optimization_hints = self._generate_optimization_hints_for_range(parsed_range)
        
        # Extract business context
        business_context = self._extract_business_context_for_range(parsed_range, context)
        
        return TemporalExpression(
            original_text=expression,
            temporal_type=temporal_type,
            parsed_range=parsed_range,
            confidence_score=confidence_score,
            business_context=business_context,
            sql_conditions=sql_conditions,
            optimization_hints=optimization_hints
        )
    
    def _determine_temporal_type(self, expression: str) -> TemporalType:
        """Determine the type of temporal expression"""
        
        # Relative patterns
        relative_patterns = ['last', 'previous', 'past', 'current', 'this', 'ytd', 'mtd', 'qtd']
        if any(pattern in expression for pattern in relative_patterns):
            return TemporalType.RELATIVE_DATE
        
        # Period patterns
        period_patterns = ['fy', 'financial year', 'q1', 'q2', 'q3', 'q4', 'quarter']
        if any(pattern in expression for pattern in period_patterns):
            return TemporalType.PERIOD
        
        # Range patterns
        if 'from' in expression and 'to' in expression:
            return TemporalType.RANGE
        if 'between' in expression and 'and' in expression:
            return TemporalType.RANGE
        
        # Absolute date patterns
        date_patterns = [r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', r'\d{4}[-/]\d{1,2}[-/]\d{1,2}']
        if any(re.search(pattern, expression) for pattern in date_patterns):
            return TemporalType.ABSOLUTE_DATE
        
        # Default to relative
        return TemporalType.RELATIVE_DATE
    
    def _parse_absolute_date(self, expression: str) -> TemporalRange:
        """Parse absolute date expressions"""
        try:
            parsed_date = date_parse(expression, fuzzy=True)
            
            # Convert to full day range
            start_date = parsed_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = parsed_date.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            return TemporalRange(
                start_date=start_date,
                end_date=end_date,
                granularity=TimeGranularity.DAY,
                calendar_type=BusinessCalendar.CALENDAR_YEAR,
                description=f"Date: {parsed_date.strftime('%Y-%m-%d')}"
            )
        
        except Exception as e:
            raise TemporalProcessingError(temporal_expression=expression)
    
    def _parse_period_expression(self, expression: str) -> TemporalRange:
        """Parse period expressions like FY2024, Q1 2024"""
        
        # Financial Year
        if match := re.search(r'fy\s*(\d{4})', expression):
            year = int(match.group(1))
            return self.get_financial_year_range(year)
        
        # Quarter
        elif match := re.search(r'q(\d)\s+(\d{4})', expression):
            quarter = int(match.group(1))
            year = int(match.group(2))
            return self.get_quarter_range(quarter, year, BusinessCalendar.FINANCIAL_YEAR)
        
        # Month Year
        elif match := re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})', expression):
            month_name = match.group(1)
            year = int(match.group(2))
            
            month_num = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }[month_name[:3]]
            
            start_date = datetime(year, month_num, 1)
            end_date = datetime(year, month_num, self._get_last_day_of_month(year, month_num), 23, 59, 59)
            
            return TemporalRange(
                start_date=start_date,
                end_date=end_date,
                granularity=TimeGranularity.MONTH,
                calendar_type=BusinessCalendar.CALENDAR_YEAR,
                description=f"{month_name.title()} {year}"
            )
        
        else:
            raise TemporalProcessingError(temporal_expression=expression)
    
    def _parse_range_expression(self, expression: str) -> TemporalRange:
        """Parse range expressions like 'from 2024-01-01 to 2024-01-31'"""
        
        # Extract dates from range expression
        date_patterns = [r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}']
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, expression)
            dates.extend(matches)
        
        if len(dates) >= 2:
            try:
                start_date = date_parse(dates[0])
                end_date = date_parse(dates[1])
                
                # Ensure end date includes full day
                if end_date.hour == 0 and end_date.minute == 0:
                    end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                
                return TemporalRange(
                    start_date=start_date,
                    end_date=end_date,
                    granularity=TimeGranularity.DAY,
                    calendar_type=BusinessCalendar.CALENDAR_YEAR,
                    description=f"Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                )
            
            except Exception as e:
                raise TemporalProcessingError(temporal_expression=expression)
        
        else:
            raise TemporalProcessingError(temporal_expression=expression)
    
    def _parse_generic_expression(self, expression: str) -> TemporalRange:
        """Parse generic/fallback temporal expressions"""
        # Default to last 30 days if can't parse specifically
        reference_date = datetime.now()
        start_date = reference_date - timedelta(days=30)
        
        return TemporalRange(
            start_date=start_date,
            end_date=reference_date,
            granularity=TimeGranularity.DAY,
            calendar_type=BusinessCalendar.CALENDAR_YEAR,
            description=f"Default: Last 30 days (fallback for '{expression}')"
        )
    
    def _get_current_financial_year(self, reference_date: datetime) -> int:
        """Get current financial year based on reference date"""
        if reference_date.month >= 4:  # April onwards
            return reference_date.year + 1
        else:  # January to March
            return reference_date.year
    
    def _get_current_financial_quarter(self, reference_date: datetime) -> int:
        """Get current financial quarter"""
        month = reference_date.month
        
        if 4 <= month <= 6:      # Apr-Jun
            return 1
        elif 7 <= month <= 9:    # Jul-Sep
            return 2
        elif 10 <= month <= 12:  # Oct-Dec
            return 3
        else:                    # Jan-Mar
            return 4
    
    def _get_last_day_of_month(self, year: int, month: int) -> int:
        """Get the last day of a month"""
        return calendar.monthrange(year, month)[1]
    
    def _calculate_confidence(self, expression: str, parsed_range: TemporalRange) -> float:
        """Calculate confidence score for parsed temporal expression"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for well-known patterns
        well_known_patterns = [
            'last', 'previous', 'current', 'this', 'fy', 'q1', 'q2', 'q3', 'q4',
            'ytd', 'mtd', 'quarter', 'year', 'month', 'day'
        ]
        
        if any(pattern in expression.lower() for pattern in well_known_patterns):
            confidence += 0.3
        
        # Boost for specific dates
        if re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', expression):
            confidence += 0.2
        
        # Reduce confidence if range is too broad or narrow
        range_days = (parsed_range.end_date - parsed_range.start_date).days
        if range_days > 1095:  # More than 3 years
            confidence -= 0.1
        elif range_days < 1:  # Less than a day
            confidence -= 0.1
        
        return min(1.0, max(0.1, confidence))
    
    def _generate_sql_conditions_for_range(self, temporal_range: TemporalRange) -> List[str]:
        """Generate SQL conditions for a temporal range"""
        conditions = []
        
        # Standard date range condition
        start_str = temporal_range.start_date.strftime('%Y-%m-%d %H:%M:%S')
        end_str = temporal_range.end_date.strftime('%Y-%m-%d %H:%M:%S')
        
        conditions.append(f"date_column >= '{start_str}' AND date_column <= '{end_str}'")
        
        # Business days only condition
        if temporal_range.is_business_days_only:
            conditions.append("DATEPART(weekday, date_column) NOT IN (1, 7)")  # Exclude weekends
        
        # Quarter-specific conditions
        if temporal_range.granularity == TimeGranularity.QUARTER:
            quarter = self._get_quarter_from_date(temporal_range.start_date, temporal_range.calendar_type)
            if temporal_range.calendar_type == BusinessCalendar.FINANCIAL_YEAR:
                conditions.append(f"financial_quarter = {quarter}")
            else:
                conditions.append(f"calendar_quarter = {quarter}")
        
        return conditions
    
    def _generate_optimization_hints_for_range(self, temporal_range: TemporalRange) -> List[str]:
        """Generate optimization hints for a temporal range"""
        hints = []
        
        range_days = (temporal_range.end_date - temporal_range.start_date).days
        
        # Large date range hints
        if range_days > 365:
            hints.append("Consider using date partitioning for large date ranges")
            hints.append("Add appropriate indexes on date columns")
        
        # Small date range hints
        elif range_days <= 7:
            hints.append("Consider using cached results for recent small date ranges")
        
        # Granularity-specific hints
        if temporal_range.granularity == TimeGranularity.DAY:
            hints.append("Daily granularity - ensure date columns are indexed")
        elif temporal_range.granularity == TimeGranularity.MONTH:
            hints.append("Monthly aggregation - consider pre-computed monthly summaries")
        
        return hints
    
    def _extract_business_context_for_range(self, temporal_range: TemporalRange, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract business context for temporal range"""
        business_context = {
            "period_type": temporal_range.granularity.value,
            "calendar_type": temporal_range.calendar_type.value,
            "business_significance": "",
            "reporting_relevance": "",
            "compliance_context": ""
        }
        
        # Financial year context
        if temporal_range.calendar_type == BusinessCalendar.FINANCIAL_YEAR:
            business_context["business_significance"] = "Financial year analysis relevant for annual reporting"
            business_context["compliance_context"] = "Aligns with RBI reporting requirements"
        
        # Quarter context
        if temporal_range.granularity == TimeGranularity.QUARTER:
            business_context["reporting_relevance"] = "Quarterly analysis for regulatory reporting"
        
        # Recent data context
        range_days = (temporal_range.end_date - temporal_range.start_date).days
        if range_days <= 30:
            business_context["business_significance"] = "Recent data analysis for operational insights"
        
        return business_context
    
    def _select_recommended_range(self, expressions: List[TemporalExpression], context: Optional[Dict[str, Any]]) -> Optional[TemporalRange]:
        """Select the recommended temporal range from processed expressions"""
        if not expressions:
            return None
        
        # Sort by confidence score
        sorted_expressions = sorted(expressions, key=lambda e: e.confidence_score, reverse=True)
        
        return sorted_expressions[0].parsed_range
    
    def _generate_sql_conditions(self, expressions: List[TemporalExpression]) -> List[str]:
        """Generate SQL WHERE clauses from temporal expressions"""
        sql_conditions = []
        
        for expr in expressions:
            sql_conditions.extend(expr.sql_conditions)
        
        return sql_conditions
    
    def _generate_performance_optimizations(self, expressions: List[TemporalExpression], context: Optional[Dict[str, Any]]) -> List[str]:
        """Generate performance optimization recommendations"""
        optimizations = set()
        
        for expr in expressions:
            optimizations.update(expr.optimization_hints)
        
        # Add general optimizations
        if len(expressions) > 1:
            optimizations.add("Multiple temporal conditions - consider combining into single range")
        
        return list(optimizations)
    
    def _extract_business_insights(self, expressions: List[TemporalExpression], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract business insights from temporal processing"""
        insights = {
            "temporal_complexity": len(expressions),
            "dominant_calendar_type": None,
            "analysis_scope": "",
            "business_relevance": []
        }
        
        if expressions:
            # Determine dominant calendar type
            calendar_types = [expr.parsed_range.calendar_type for expr in expressions]
            dominant_calendar = max(set(calendar_types), key=calendar_types.count)
            insights["dominant_calendar_type"] = dominant_calendar.value
            
            # Determine analysis scope
            total_days = sum((expr.parsed_range.end_date - expr.parsed_range.start_date).days for expr in expressions)
            if total_days <= 30:
                insights["analysis_scope"] = "short_term"
            elif total_days <= 365:
                insights["analysis_scope"] = "medium_term"
            else:
                insights["analysis_scope"] = "long_term"
            
            # Business relevance
            for expr in expressions:
                insights["business_relevance"].extend(list(expr.business_context.keys()))
        
        return insights
    
    def _get_quarter_from_date(self, date_obj: datetime, calendar_type: BusinessCalendar) -> int:
        """Get quarter number from date"""
        if calendar_type == BusinessCalendar.FINANCIAL_YEAR:
            month = date_obj.month
            if 4 <= month <= 6:
                return 1
            elif 7 <= month <= 9:
                return 2
            elif 10 <= month <= 12:
                return 3
            else:
                return 4
        else:  # Calendar year
            return (date_obj.month - 1) // 3 + 1


# Global temporal processor instance
temporal_processor = TemporalProcessor()
