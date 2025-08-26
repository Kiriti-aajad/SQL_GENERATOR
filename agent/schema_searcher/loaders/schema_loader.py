# agent/schema_searcher/loaders/schema_loader.py
"""
Loader for relational schema metadata from a JSON file.

Schema input format:
[
  {
    "table": "table_name",
    "column": "column_name",
    "datatype": "nvarchar",
    "type": "relational" | "xml" | "computed",
    "description": "text"
  },
  ...
]
"""

from agent.schema_searcher.loaders.base_loader import BaseLoader
from agent.schema_searcher.core.config import SCHEMA_PATH
from agent.schema_searcher.core.data_models import (
    RetrievedColumn, ColumnType, SearchMethod, DataType
)
from typing import List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__) # type: ignore


class SchemaLoader(BaseLoader):
    """
    Loads customer relational schema fields from JSON.

    Only columns marked as "relational" type are returned.
    Use for semantic, keyword, fuzzy search inputs.
    """

    def __init__(self, source_path=SCHEMA_PATH):
        super().__init__(source_path)
        self.logger = logger.getChild("SchemaLoader")

    def load(self) -> List[Dict[str, Any]]:
        """Raw JSON list for custom processing (engines, prompts)"""
        raw_data = self.read_json()
        valid_rows = []

        for row in raw_data:
            row_type = str(row.get("type", "unknown")).lower()

            if row_type == "relational":
                valid_rows.append(row)

        self.logger.info(f"Loaded {len(valid_rows)} relational schema columns from {self.source_path}")
        return valid_rows

    def get_standardized(self) -> List[RetrievedColumn]:
        """Return structured column objects"""
        documents = self.load()
        results = []

        for row in documents:
            try:
                column = RetrievedColumn(
                    table=row.get("table", "").strip(),
                    column=row.get("column", "").strip(),
                    datatype=row.get("datatype", "unknown"),
                    type=ColumnType.RELATIONAL,
                    description=row.get("description", ""),
                    confidence_score=1.0,
                    retrieval_method=SearchMethod.SEMANTIC,
                    nullable=True,
                    primary_key=False,
                    foreign_key=False,
                    retrieved_at=datetime.now()
                )
                results.append(column)
            except Exception as e:
                self.logger.warning(f"Error converting row: {row} → {e}")

        return results

    def get_table_schema(self, table_name: str) -> List[RetrievedColumn]:
        """Get all columns for a specific table"""
        return [col for col in self.get_standardized() if col.table.lower() == table_name.lower()]
