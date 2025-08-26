# agent/schema_searcher/loaders/xml_loader.py
"""
Loads structured XML field metadata from a JSON file.

Expected format:
[
  {
    "table": "tblCounterparty",
    "xml_column": "CTPT_XML",
    "fields": [
      {
        "name": "CPArrLeadBBFamilyMember",
        "xpath": "/CTPT/CPArrLeadBBFamilyMember",
        "sql_expression": "CTPT_XML.value('(/CTPT/CPArrLeadBBFamilyMember)[1]', 'varchar(100)')"
      },
      ...
    ]
  },
  ...
]
"""

from agent.schema_searcher.loaders.base_loader import BaseLoader
from agent.schema_searcher.core.config import XML_SCHEMA_PATH
from agent.schema_searcher.core.data_models import (
    RetrievedColumn, ColumnType, SearchMethod, DataType
)
from typing import List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__) # type: ignore


class XMLLoader(BaseLoader):
    """
    Loads XML schema field metadata from disk
    and returns field definitions with SQL-access paths.
    """

    def __init__(self, source_path=XML_SCHEMA_PATH):
        super().__init__(source_path)
        self.logger = logger.getChild("XMLLoader")

    def load(self) -> List[Dict[str, Any]]:
        """Load full raw structure from file"""
        data = self.read_json()
        valid_rows = []

        for entry in data:
            table = entry.get("table", "").strip()
            xml_column = entry.get("xml_column", "").strip()
            fields = entry.get("fields", [])

            if table and xml_column and isinstance(fields, list):
                valid_rows.append({
                    "table": table,
                    "xml_column": xml_column,
                    "fields": fields
                })

        self.logger.info(f"Loaded XML definitions for {len(valid_rows)} tables.")
        return valid_rows

    def get_standardized(self) -> List[RetrievedColumn]:
        """Parse and return flattened XML fields as RetrievedColumns"""
        documents = self.load()
        results: List[RetrievedColumn] = []

        for doc in documents:
            table = doc["table"]
            xml_col = doc["xml_column"]

            for field in doc.get("fields", []):
                try:
                    column = RetrievedColumn(
                        table=table,
                        column=field.get("name", "").strip(),
                        datatype="xml_field",
                        type=ColumnType.XML,
                        description=f"XML field in {xml_col}: {field.get('xpath', '')}",
                        xml_column=xml_col,
                        xpath=field.get("xpath", ""),
                        sql_expression=field.get("sql_expression", ""),
                        confidence_score=1.0,
                        retrieval_method=SearchMethod.SEMANTIC,
                        nullable=True,
                        retrieved_at=datetime.now()
                    )

                    results.append(column)
                except Exception as e:
                    self.logger.warning(f"Skipping corrupted XML field: {field} → {e}")

        return results
