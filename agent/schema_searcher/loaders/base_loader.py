# agent/schema_searcher/loaders/base_loader.py
"""
Abstract base class for all schema metadata loaders (schema, XML, joins, etc.).

Defines the base interface for loading data from supported sources
(JSON, pickle, database, etc.). Each concrete implementation must
extend this class and implement the `load()` method.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
import os
import json
import logging

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """
    Abstract class for loaders used to load schema metadata
    from various formats like JSON, pickle, database, etc.
    """

    def __init__(self, source_path: Path):
        self.source_path: Path = source_path
        self.logger = logger.getChild(self.__class__.__name__)

    def exists(self) -> bool:
        """Check if the target file exists"""
        return self.source_path.exists() and self.source_path.is_file()

    def validate(self) -> None:
        if not self.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.source_path}")
        self.logger.debug(f"Validated existence of: {self.source_path}")

    def read_json(self) -> List[Dict[str, Any]]:
        """
        Load and parse a JSON file into a list of dictionaries.
        Used by default for most static schema metadata.
        """
        self.validate()

        try:
            with open(self.source_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    # Convert to list of dicts if wrapped
                    return list(data.values())
                else:
                    raise ValueError("Invalid JSON format: Expected list or dict")

        except Exception as e:
            self.logger.error(f"Error reading JSON file: {self.source_path} → {e}")
            raise

    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """
        Load and return a list of metadata dictionaries
        (e.g., column definitions, xml fields, join configurations).
        """
        ...
