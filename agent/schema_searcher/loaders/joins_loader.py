# agent/schema_searcher/loaders/joins_loader.py
"""
Loader for join metadata used in SQL query planning.

Sources:
1. JSON-based join definitions from `joins_verified.json`
2. Pre-built graph object using NetworkX from `join_graph.gpickle`

This module returns structured join records and optionally
a graph object useful for topological join resolution.
"""

import json
import logging
import pickle
from typing import List, Dict, Any
from pathlib import Path
import networkx as nx

from agent.schema_searcher.loaders.base_loader import BaseLoader
from agent.schema_searcher.core.config import JOINS_PATH
from agent.schema_searcher.core.data_models import RetrievedJoin, JoinType, Priority
from agent.schema_searcher.core.config import PROJECT_ROOT

# Path to additional join graph file (optional)
GRAPH_PATH = PROJECT_ROOT / "data" / "join_graph" / "join_graph.gpickle"

logger = logging.getLogger(__name__) # type: ignore

class JoinsLoader(BaseLoader):
    """
    Loads structured join definitions from JSON
    and optionally parses a join graph from disk.
    """

    def __init__(self, source_path: Path = JOINS_PATH, graph_path: Path = GRAPH_PATH):
        super().__init__(source_path)
        self.graph_path = graph_path
        self.logger = logger.getChild("JoinsLoader")

    def load(self) -> List[Dict[str, Any]]:
        """
        Load raw join definitions from verified JSON file.
        """
        raw_joins = self.read_json()
        valid_joins = []

        for join in raw_joins:
            try:
                if all([
                    join.get("source"),
                    join.get("source_column"),
                    join.get("target"),
                    join.get("target_column")
                ]):
                    valid_joins.append(join)
            except Exception as ex:
                self.logger.warning(f"Skipping invalid join entry: {join} → {ex}")

        self.logger.info(f"Loaded {len(valid_joins)} joins from {self.source_path}")
        return valid_joins

    def get_standardized(self) -> List[RetrievedJoin]:
        """Return list of RetrievedJoin objects with proper typing"""
        joins = self.load()
        results: List[RetrievedJoin] = []

        for item in joins:
            try:
                results.append(RetrievedJoin(
                    source_table=item["source"],
                    source_column=item["source_column"],
                    target_table=item["target"],
                    target_column=item["target_column"],
                    join_type=JoinType(item.get("type", "inner")),
                    confidence=int(item.get("confidence", 100)),
                    verified=bool(item.get("verified", True)),
                    priority=Priority(item.get("priority", "medium")),
                    comment=item.get("comment", "")
                ))
            except Exception as e:
                self.logger.warning(f"Invalid join skipped: {item} → {e}")

        return results

    def load_graph(self) -> nx.DiGraph:
        """
        Load pre-built join graph from .gpickle.
        This graph can be used to compute join paths between tables.
        """
        if not self.graph_path.exists():
            raise FileNotFoundError(f"Join graph file not found at {self.graph_path}")

        try:
            # FIX: Use pickle.load instead of nx.read_gpickle
            with open(self.graph_path, 'rb') as f:
                graph = pickle.load(f)
            
            self.logger.info(f"Join graph loaded with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
            return graph
        except Exception as e:
            self.logger.error(f"Failed to load join graph: {e}")
            raise

    def save_graph(self, graph: nx.DiGraph, overwrite: bool = True) -> None:
        """
        Save a NetworkX graph to disk for future path finding.
        Used when building/rebuilding join graphs offline.
        """
        if self.graph_path.exists() and not overwrite:
            raise FileExistsError(f"Join graph already exists: {self.graph_path}")

        try:
            # FIX: Use pickle.dump instead of nx.write_gpickle
            with open(self.graph_path, 'wb') as f:
                pickle.dump(graph, f)
            
            self.logger.info(f"Join graph saved to {self.graph_path}")
        except Exception as e:
            self.logger.error(f"Failed to save join graph: {e}")
            raise

    def build_graph_from_json(self) -> nx.DiGraph:
        """
        Reconstructs a directed join graph from the raw JSON.
        Each edge represents a joinable relationship between tables.
        """
        joins = self.load()
        G = nx.DiGraph()

        for join in joins:
            try:
                source = join["source"]
                target = join["target"]
                attrs = {
                    "source_column": join["source_column"],
                    "target_column": join["target_column"],
                    "type": join.get("type", "inner"),
                    "confidence": int(join.get("confidence", 100)),
                    "weight": (100 - int(join.get("confidence", 100))) or 1  # smaller = stronger edge
                }
                G.add_edge(source, target, **attrs)
            except Exception as e:
                self.logger.warning(f"Failed to add join to graph: {join} → {e}")

        self.logger.info(f"Built join graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G
