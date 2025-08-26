"""
Schema Analyzer - Enhanced Implementation with E5-Base-V2 Consistency
Updated to use EmbeddingModelManager singleton pattern for memory optimization.
"""

from typing import List, Dict, Any
import logging
import numpy as np

from agent.schema_searcher.core.data_models import RetrievedColumn
from agent.schema_searcher.utils.mathstral_client import MathstralClient
from agent.schema_searcher.utils.embedding_manager import EmbeddingModelManager

class SchemaAnalyzer:
    """
    Mathstral-powered schema completeness analysis with proper separation of concerns:
    - Uses EmbeddingModelManager for shared embedding model instances
    - Uses Mathstral for high-level reasoning and gap analysis
    - Uses intfloat/e5-base-v2 for consistency with ChromaDB
    - Combines both for comprehensive schema analysis
    """

    def __init__(self, client: MathstralClient = None, embedding_model: str = "intfloat/e5-base-v2"):
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            self.logger.addHandler(h)
            self.logger.setLevel(logging.INFO)

        # Mathstral client for reasoning (NOT embeddings)
        self.mathstral_client = client or MathstralClient()
        
        # UPDATED: Use EmbeddingModelManager for shared model instance
        try:
            self.embedding_model = EmbeddingModelManager.get_model(embedding_model)
            self.logger.info(f"SchemaAnalyzer initialized with embedding model: {embedding_model}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model {embedding_model}: {e}")
            # Fallback to a simpler model
            self.embedding_model = EmbeddingModelManager.get_model("all-MiniLM-L6-v2")
            self.logger.warning("Fell back to all-MiniLM-L6-v2 embedding model")
        
        # Store the actual model name being used
        self.actual_embedding_model = embedding_model
        
        self.logger.info("SchemaAnalyzer initialized with proper architecture")

    def analyze_schema_completeness(
        self,
        query: str,
        schema_columns: List[RetrievedColumn]
    ) -> Dict[str, Any]:
        """
        Analyze schema completeness using e5-base-v2 embeddings + Mathstral reasoning.
        
        Returns:
          {
            'semantic_similarity': float,
            'coverage_ratio': float,
            'relationship_score': float,
            'mathstral_insights': dict,
            'completeness_score': float
          }
        """
        try:
            # 1. Use e5-base-v2 with proper prefixes for embeddings
            query_embedding = self._encode_with_e5_prefix([query], "query")[0]
            
            # 2. Embed each column with context
            col_texts = []
            for c in schema_columns:
                context = f"{c.table}.{c.column}"
                if hasattr(c, 'description') and c.description:
                    context += f" {c.description}"
                col_texts.append(context)
            
            if not col_texts:
                return self._empty_analysis()
            
            col_embeddings = self._encode_with_e5_prefix(col_texts, "passage")
            
            # 3. Calculate E5-optimized cosine similarities
            similarities = []
            for col_emb in col_embeddings:
                similarity = self._calculate_e5_similarity(query_embedding, col_emb)
                similarities.append(similarity)
            
            semantic_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            
            # 4. Coverage: fraction of query terms found in schema
            query_terms = set(query.lower().split())
            schema_terms = set()
            for col in schema_columns:
                schema_terms.update(col.table.lower().split())
                schema_terms.update(col.column.lower().split())
                if hasattr(col, 'description') and col.description:
                    schema_terms.update(col.description.lower().split())
            
            matched_terms = query_terms.intersection(schema_terms)
            coverage_ratio = len(matched_terms) / len(query_terms) if query_terms else 0.0
            
            # 5. Use Mathstral for high-level reasoning (NOT embeddings)
            mathstral_insights = self._get_mathstral_insights(query, schema_columns)
            
            # 6. Calculate relationship score
            relationship_score = self._calculate_relationship_score(schema_columns)
            
            # 7. Combine into overall completeness score with E5 adjustments
            completeness_score = self._calculate_completeness_score(
                semantic_similarity, coverage_ratio, relationship_score, mathstral_insights
            )
            
            results = {
                'semantic_similarity': semantic_similarity,
                'coverage_ratio': coverage_ratio,
                'relationship_score': relationship_score,
                'mathstral_insights': mathstral_insights,
                'completeness_score': completeness_score,
                'raw_similarities': similarities,
                'matched_terms': list(matched_terms),
                'total_columns': len(schema_columns),
                'embedding_model_used': self.actual_embedding_model
            }
            
            self.logger.debug(f"Schema completeness analysis: {completeness_score:.2%} complete using {self.actual_embedding_model}")
            return results
            
        except Exception as e:
            self.logger.error(f"Schema completeness analysis failed: {e}")
            return self._empty_analysis()

    def identify_gaps(
        self,
        query_intent: str,
        available_schema: List[RetrievedColumn]
    ) -> Dict[str, List[str]]:
        """
        Use Mathstral for gap analysis (its proper purpose).
        """
        try:
            # Prepare schema summary for Mathstral
            schema_summary = self._prepare_schema_summary(available_schema)
            
            # Use Mathstral to analyze gaps
            gap_analysis = self.mathstral_client.analyze_schema_gap(
                gap_description=f"Query intent: {query_intent}. Available schema: {schema_summary}",
                context="comprehensive gap analysis"
            )
            
            # Extract and categorize gaps
            gaps = {
                'missing_entities': self._extract_missing_entities(query_intent, available_schema, gap_analysis),
                'missing_relationships': self._extract_missing_relationships(available_schema),
                'missing_data_points': gap_analysis.get('search_terms', [])
            }
            
            self.logger.debug(f"Identified gaps: {sum(len(v) for v in gaps.values())} total")
            return gaps
            
        except Exception as e:
            self.logger.error(f"Gap identification failed: {e}")
            return {
                'missing_entities': [],
                'missing_relationships': [],
                'missing_data_points': []
            }

    def calculate_confidence_score(
        self,
        analysis_results: Dict[str, Any]
    ) -> float:
        """
        Combine multiple factors into confidence score [0–1].
        Enhanced for E5 model characteristics.
        """
        try:
            # Use the completeness score if available
            if 'completeness_score' in analysis_results:
                return analysis_results['completeness_score']
            
            # Fallback calculation with E5 adjustments
            coverage = analysis_results.get('coverage_ratio', 0.0)
            similarity = analysis_results.get('semantic_similarity', 0.0)
            relationship = analysis_results.get('relationship_score', 0.0)
            
            # E5 models tend to give higher similarity scores, adjust weights
            weights = {'coverage': 0.35, 'similarity': 0.45, 'relationship': 0.2}
            
            confidence = (
                weights['coverage'] * coverage +
                weights['similarity'] * similarity +
                weights['relationship'] * relationship
            )
            
            # E5-specific confidence boost for high semantic similarity
            if similarity > 0.8:
                confidence += 0.05  # Small boost for very high similarity
            
            self.logger.debug(f"Calculated confidence score: {confidence:.2%}")
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5  # Default moderate confidence

    def generate_gap_report(
        self,
        gaps: Dict[str, List[str]],
        analysis_results: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable gap analysis report with model information.
        """
        try:
            completeness = analysis_results.get('completeness_score', 0.0)
            coverage = analysis_results.get('coverage_ratio', 0.0)
            similarity = analysis_results.get('semantic_similarity', 0.0)
            model_used = analysis_results.get('embedding_model_used', 'unknown')
            
            lines = [
                "=== Schema Completeness Report ===",
                f"Overall Completeness: {completeness:.1%}",
                f"Query Coverage: {coverage:.1%}",
                f"Semantic Similarity: {similarity:.1%}",
                f"Total Columns Found: {analysis_results.get('total_columns', 0)}",
                f"Embedding Model: {model_used}",
                "",
                "Missing Entities:",
            ]
            
            entities = gaps.get('missing_entities', [])
            if entities:
                lines.extend([f"  • {entity}" for entity in entities])
            else:
                lines.append("  ✓ No missing entities detected")
            
            lines.extend(["", "Missing Relationships:"])
            relationships = gaps.get('missing_relationships', [])
            if relationships:
                lines.extend([f"  • {rel}" for rel in relationships])
            else:
                lines.append("  ✓ No missing relationships detected")
            
            lines.extend(["", "Suggested Additional Data Points:"])
            data_points = gaps.get('missing_data_points', [])
            if data_points:
                lines.extend([f"  • {point}" for point in data_points[:5]])  # Limit to top 5
            else:
                lines.append("  ✓ No additional data points suggested")
            
            # Add Mathstral insights if available
            insights = analysis_results.get('mathstral_insights', {})
            if insights.get('confidence_assessment'):
                lines.extend([
                    "",
                    "AI Assessment:",
                    f"  {insights['confidence_assessment']}"
                ])
            
            # Add E5-specific interpretation notes
            if 'e5' in model_used.lower() and similarity > 0.7:
                lines.extend([
                    "",
                    "E5 Model Notes:",
                    "  • High semantic similarity indicates strong conceptual match",
                    "  • E5 model excels at understanding business terminology"
                ])
            
            report = "\n".join(lines)
            self.logger.info("Generated comprehensive gap report")
            return report
            
        except Exception as e:
            self.logger.error(f"Gap report generation failed: {e}")
            return f"Gap report generation failed: {e}"

    # E5-SPECIFIC HELPER METHODS

    def _encode_with_e5_prefix(self, texts: List[str], text_type: str = "passage") -> np.ndarray:
        """
        Encode texts with appropriate E5 prefixes for better performance.
        
        Parameters:
        -----------
        texts : List[str]
            Texts to encode
        text_type : str
            Either "query" or "passage" for E5 prefixing
        """
        try:
            # E5 models perform better with prefixes
            if 'e5' in self.actual_embedding_model.lower():
                prefixed_texts = [f"{text_type}: {text}" for text in texts]
                return self.embedding_model.encode(prefixed_texts)
            else:
                # No prefixes for non-E5 models
                return self.embedding_model.encode(texts)
        except Exception as e:
            self.logger.warning(f"E5 encoding failed: {e}")
            # Fallback without prefixes
            return self.embedding_model.encode(texts)

    def _calculate_e5_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        E5-optimized similarity calculation with proper normalization.
        """
        try:
            # E5 models work better with normalized vectors
            vec_a_norm = vec_a / (np.linalg.norm(vec_a) + 1e-8)
            vec_b_norm = vec_b / (np.linalg.norm(vec_b) + 1e-8)
            
            # Calculate dot product (cosine similarity for normalized vectors)
            similarity = np.dot(vec_a_norm, vec_b_norm)
            
            # E5 models can return values in [-1, 1], normalize to [0, 1]
            if 'e5' in self.actual_embedding_model.lower():
                normalized_sim = (similarity + 1.0) / 2.0
            else:
                normalized_sim = similarity
            
            return max(0.0, min(1.0, normalized_sim))
            
        except Exception as e:
            self.logger.warning(f"E5 similarity calculation failed: {e}")
            # Fallback to simple dot product
            try:
                similarity = np.dot(vec_a, vec_b) / (
                    np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
                )
                return max(0.0, similarity)
            except Exception:
                return 0.0

    # PRIVATE HELPER METHODS

    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis results."""
        return {
            'semantic_similarity': 0.0,
            'coverage_ratio': 0.0,
            'relationship_score': 0.0,
            'mathstral_insights': {},
            'completeness_score': 0.0,
            'raw_similarities': [],
            'matched_terms': [],
            'total_columns': 0,
            'embedding_model_used': self.actual_embedding_model
        }

    def _get_mathstral_insights(self, query: str, schema_columns: List[RetrievedColumn]) -> Dict[str, Any]:
        """Use Mathstral for high-level reasoning insights."""
        try:
            # Prepare context for Mathstral
            schema_context = self._prepare_schema_summary(schema_columns)
            
            # Get Mathstral's assessment
            classification = self.mathstral_client.classify_business_entity(
                entity_text=query,
                context=f"Available schema: {schema_context}",
                domain="database schema analysis"
            )
            
            return {
                'entity_classification': classification.get('primary_type', 'unknown'),
                'confidence_assessment': f"Mathstral confidence: {classification.get('confidence', 0.0):.1%}",
                'domain_relevance': classification.get('domain', 'business'),
                'secondary_types': classification.get('secondary_types', [])
            }
            
        except Exception as e:
            self.logger.debug(f"Mathstral insights failed: {e}")
            return {}

    def _prepare_schema_summary(self, schema_columns: List[RetrievedColumn]) -> str:
        """Prepare concise schema summary for Mathstral."""
        if not schema_columns:
            return "No schema columns available"
        
        # Group by table
        tables = {}
        for col in schema_columns:
            if col.table not in tables:
                tables[col.table] = []
            tables[col.table].append(col.column)
        
        # Create summary
        summary_parts = []
        for table, columns in tables.items():
            summary_parts.append(f"{table}({', '.join(columns[:5])})")  # Limit columns per table
        
        return "; ".join(summary_parts[:10])  # Limit total tables

    def _calculate_relationship_score(self, schema_columns: List[RetrievedColumn]) -> float:
        """Calculate relationship completeness score."""
        try:
            # Simple heuristic: more tables = better relationships
            tables = set(col.table for col in schema_columns)
            table_count = len(tables)
            
            if table_count <= 1:
                return 0.0
            elif table_count <= 3:
                return 0.5
            elif table_count <= 6:
                return 0.7
            else:
                return 0.9
                
        except Exception:
            return 0.0

    def _calculate_completeness_score(
        self, 
        semantic_sim: float, 
        coverage: float, 
        relationship: float, 
        insights: Dict[str, Any]
    ) -> float:
        """Calculate overall completeness score with E5 adjustments."""
        # Adjusted weights for E5 model characteristics
        base_score = 0.35 * coverage + 0.45 * semantic_sim + 0.2 * relationship
        
        # Mathstral confidence boost
        mathstral_confidence = 0.0
        if insights and 'entity_classification' in insights:
            classification = insights['entity_classification']
            if classification != 'unknown':
                mathstral_confidence = 0.1  # 10% boost for successful classification
        
        # E5-specific boost for high semantic similarity
        e5_boost = 0.0
        if 'e5' in self.actual_embedding_model.lower() and semantic_sim > 0.85:
            e5_boost = 0.05  # E5 models are particularly good at semantic understanding
        
        final_score = min(1.0, base_score + mathstral_confidence + e5_boost)
        return final_score

    def _extract_missing_entities(
        self, 
        query: str, 
        schema: List[RetrievedColumn], 
        gap_analysis: Dict[str, Any]
    ) -> List[str]:
        """Extract missing entities from gap analysis."""
        try:
            # Get Mathstral suggestions
            suggestions = gap_analysis.get('search_terms', [])
            
            # Filter out entities that already exist in schema
            existing_terms = set()
            for col in schema:
                existing_terms.add(col.table.lower())
                existing_terms.add(col.column.lower())
            
            missing_entities = []
            for suggestion in suggestions:
                if suggestion.lower() not in existing_terms:
                    missing_entities.append(suggestion)
            
            return missing_entities[:5]  # Limit to top 5
            
        except Exception as e:
            self.logger.debug(f"Missing entity extraction failed: {e}")
            return []

    def _extract_missing_relationships(self, schema: List[RetrievedColumn]) -> List[str]:
        """Identify missing table relationships."""
        try:
            tables = list(set(col.table for col in schema))
            
            if len(tables) < 2:
                return ["Need additional related tables"]
            elif len(tables) == 2:
                return [f"Potential relationship between {tables[0]} and {tables[1]}"]
            else:
                return []  # Assume relationships exist with 3+ tables
                
        except Exception:
            return []

    # UTILITY METHODS

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model being used."""
        try:
            return {
                'model_name': self.actual_embedding_model,
                'model_type': 'e5-base-v2' if 'e5-base-v2' in self.actual_embedding_model else 'other',
                'supports_prefixes': 'e5' in self.actual_embedding_model.lower(),
                'embedding_dimension': self.embedding_model.get_sentence_embedding_dimension(),
                'max_sequence_length': getattr(self.embedding_model, 'max_seq_length', 'unknown'),
                'optimized_for_e5': 'e5' in self.actual_embedding_model.lower(),
                'using_shared_model': True  # Indicates using EmbeddingModelManager
            }
        except Exception as e:
            return {
                'model_name': self.actual_embedding_model,
                'using_shared_model': True,
                'error': str(e)
            }

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get statistics about analysis performance."""
        return {
            'embedding_model': self.actual_embedding_model,
            'mathstral_client_available': self.mathstral_client is not None,
            'supports_e5_prefixes': 'e5' in self.actual_embedding_model.lower(),
            'architecture': 'e5_optimized' if 'e5' in self.actual_embedding_model.lower() else 'standard',
            'memory_optimized': True  # Indicates using shared model
        }
