# agent/schema_searcher/utils/validators.py
"""
Shared validation utilities for query inputs, metadata schemas,
and retrieved column values.
"""

from typing import Any, Dict, Optional
import re


MAX_QUERY_LENGTH = 1000
MAX_TABLE_NAME_LENGTH = 128
MAX_COLUMN_NAME_LENGTH = 128
MAX_DESCRIPTION_LENGTH = 2000


class ValidationError(Exception):
    """Exception for input validation errors."""
    pass


def validate_query(query: str) -> None:
    """Validate that the input query is a non-empty, short-enough string."""
    if not query or not isinstance(query, str):
        raise ValidationError("Query must be a non-empty string.")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValidationError(f"Query exceeds max length of {MAX_QUERY_LENGTH} characters.")
    if not re.search(r"[a-zA-Z]", query):
        raise ValidationError("Query must contain alphabetical characters.")


def validate_table_name(name: str) -> None:
    """Validate table names only contain alphanumeric, underscore, or dot."""
    if not name or len(name) > MAX_TABLE_NAME_LENGTH:
        raise ValidationError("Invalid or oversized table name.")
    if not re.match(r"^[\w\.]+$", name):
        raise ValidationError("Table name contains invalid characters.")


def validate_column_name(name: str) -> None:
    """Validate column names are safe and reasonable in length."""
    if not name or len(name) > MAX_COLUMN_NAME_LENGTH:
        raise ValidationError("Invalid or oversized column name.")
    if not re.match(r"^[a-zA-Z0-9_]+$", name):
        raise ValidationError("Column name contains invalid characters.")


def validate_description(description: str) -> None:
    """Ensure description is a string and of valid length."""
    if not isinstance(description, str):
        raise ValidationError("Description must be a string.")
    if len(description) > MAX_DESCRIPTION_LENGTH:
        raise ValidationError(f"Description exceeds {MAX_DESCRIPTION_LENGTH} characters.")


def safe_get(d: Dict[str, Any], key: str, fallback: Optional[Any] = None) -> Optional[Any]:
    """Safe get from dict with fallback."""
    return d[key] if key in d else fallback
