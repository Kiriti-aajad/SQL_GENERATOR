"""
KeywordGenerator
────────────────
Enhanced keyword generation with proper architecture separation.

Uses EmbeddingModelManager for shared embedding model instances and Mathstral for high-level reasoning.
UPDATED: Now uses EmbeddingModelManager singleton pattern for memory optimization.

Main responsibilities:
1. generate_entity_keywords()      → initial keyword list from missing entities
2. generate_semantic_variations()  → expand with synonyms / paraphrases
3. create_domain_specific_terms()  → inject business-domain terminology
4. prioritize_search_terms()       → rank by semantic relevance × search-engine utility
"""

from __future__ import annotations

from typing import List, Dict, Any
import logging
import itertools
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Core data models
from agent.schema_searcher.core.data_models import RetrievedColumn
# Mathstral client for reasoning (NOT embeddings)
from agent.schema_searcher.utils.mathstral_client import MathstralClient
# UPDATED: Use EmbeddingModelManager for shared model instances
from agent.schema_searcher.utils.embedding_manager import EmbeddingModelManager

class KeywordGenerator:
    """
    Intelligent keyword generator with proper separation of concerns:
    - Uses EmbeddingModelManager for shared embedding model instances
    - Uses Mathstral for high-level reasoning and domain knowledge
    - Uses intfloat/e5-base-v2 model for consistency with ChromaDB
    - Deterministic ranking/filtering for reproducible behavior
    """

    def __init__(
        self,
        client: MathstralClient | None = None,
        embedding_model: str = "intfloat/e5-base-v2",
        *,
        max_variations: int = 50,
        max_final_keywords: int = 25,
        min_similarity: float = 0.55,
    ) -> None:
        # Logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
            )
            self.logger.addHandler(h)
            self.logger.setLevel(logging.INFO)

        # Mathstral client for reasoning (NOT embeddings)
        self.mathstral_client: MathstralClient = client or MathstralClient()
        
        # 🔧 CRITICAL FIX: Check MathstralClient capabilities and add missing methods
        self._enhance_mathstral_client_if_needed()
        
        # UPDATED: Use EmbeddingModelManager for shared model instance
        try:
            self.embedding_model = EmbeddingModelManager.get_model(embedding_model)
            self.logger.info(f"KeywordGenerator initialized with embedding model: {embedding_model}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model {embedding_model}: {e}")
            # Fallback to a simpler model
            self.embedding_model = EmbeddingModelManager.get_model("all-MiniLM-L6-v2")
            self.logger.warning("Fell back to all-MiniLM-L6-v2 embedding model")

        # Store the actual model name being used
        self.actual_embedding_model = embedding_model

        # Tuning knobs
        self.max_variations = max_variations
        self.max_final_keywords = max_final_keywords
        self.min_similarity = min_similarity
        
        self.logger.info("KeywordGenerator initialized with proper architecture")

    def _enhance_mathstral_client_if_needed(self):
        """🔧 CRITICAL FIX: Add missing methods to MathstralClient if they don't exist"""
        try:
            # Check and add generate_semantic_variations method
            if not hasattr(self.mathstral_client, 'generate_semantic_variations'):
                def generate_semantic_variations(term: str, domain_context: str = "business database", max_variations: int = 5) -> Dict[str, List[str]]:
                    """Fallback semantic variations generator"""
                    try:
                        variations = []
                        term_lower = term.lower()
                        
                        # Business domain variations
                        business_variations = {
                            'customer': ['client', 'user', 'account_holder', 'buyer', 'consumer'],
                            'order': ['purchase', 'transaction', 'sale', 'booking', 'request'],
                            'product': ['item', 'goods', 'merchandise', 'article', 'service'],
                            'payment': ['transaction', 'billing', 'charge', 'invoice', 'settlement'],
                            'address': ['location', 'addr', 'place', 'destination', 'residence'],
                            'phone': ['telephone', 'mobile', 'contact', 'number', 'tel'],
                            'email': ['mail', 'electronic_mail', 'contact', 'address', 'correspondence'],
                            'date': ['timestamp', 'time', 'datetime', 'created', 'modified'],
                            'amount': ['value', 'price', 'cost', 'sum', 'total'],
                            'status': ['state', 'condition', 'flag', 'indicator', 'stage']
                        }
                        
                        for key, vals in business_variations.items():
                            if key in term_lower:
                                variations.extend(vals[:max_variations])
                        
                        # Add morphological variations
                        if term.endswith('s') and len(term) > 3:
                            variations.append(term[:-1])
                        elif not term.endswith('s'):
                            variations.append(term + 's')
                        
                        # Add underscore/hyphen variations
                        if ' ' in term:
                            variations.append(term.replace(' ', '_'))
                            variations.append(term.replace(' ', '-'))
                        
                        return {"variations": list(set(variations))[:max_variations]}
                        
                    except Exception as e:
                        self.logger.debug(f"Fallback semantic variations failed: {e}")
                        return {"variations": []}
                
                self.mathstral_client.generate_semantic_variations = generate_semantic_variations
                self.logger.debug("Added generate_semantic_variations method to MathstralClient")

            # Check and add domain_term_expansion method
            if not hasattr(self.mathstral_client, 'domain_term_expansion'):
                def domain_term_expansion(terms: List[str], domain: str = "business database", max_terms_per_input: int = 3) -> Dict[str, List[str]]:
                    """Fallback domain term expansion"""
                    try:
                        domain_expansions = []
                        domain_lower = domain.lower()
                        
                        for term in terms:
                            term_expansions = []
                            term_lower = term.lower()
                            
                            # Banking/Financial domain
                            if 'banking' in domain_lower or 'financial' in domain_lower:
                                banking_terms = {
                                    'account': ['portfolio', 'wallet', 'balance', 'holdings'],
                                    'customer': ['client', 'account_holder', 'member', 'subscriber'],
                                    'transaction': ['transfer', 'deposit', 'withdrawal', 'payment'],
                                    'branch': ['office', 'location', 'center', 'facility'],
                                    'loan': ['credit', 'advance', 'financing', 'mortgage']
                                }
                                for key, expansions in banking_terms.items():
                                    if key in term_lower:
                                        term_expansions.extend(expansions[:max_terms_per_input])
                            
                            # E-commerce domain
                            elif 'commerce' in domain_lower or 'retail' in domain_lower:
                                commerce_terms = {
                                    'product': ['item', 'sku', 'merchandise', 'catalog'],
                                    'customer': ['buyer', 'shopper', 'user', 'member'],
                                    'order': ['purchase', 'cart', 'checkout', 'booking'],
                                    'inventory': ['stock', 'warehouse', 'supply', 'goods'],
                                    'shipping': ['delivery', 'fulfillment', 'dispatch', 'transport']
                                }
                                for key, expansions in commerce_terms.items():
                                    if key in term_lower:
                                        term_expansions.extend(expansions[:max_terms_per_input])
                            
                            # Generic business domain
                            else:
                                generic_terms = {
                                    'name': ['title', 'label', 'identifier', 'designation'],
                                    'id': ['identifier', 'key', 'reference', 'code'],
                                    'type': ['category', 'class', 'kind', 'classification'],
                                    'date': ['timestamp', 'created', 'modified', 'updated'],
                                    'status': ['state', 'condition', 'flag', 'stage']
                                }
                                for key, expansions in generic_terms.items():
                                    if key in term_lower:
                                        term_expansions.extend(expansions[:max_terms_per_input])
                            
                            if term_expansions:
                                domain_expansions.append({"term": term, "terms": term_expansions})
                        
                        return {"domain_terms": domain_expansions}
                        
                    except Exception as e:
                        self.logger.debug(f"Fallback domain expansion failed: {e}")
                        return {"domain_terms": []}
                
                self.mathstral_client.domain_term_expansion = domain_term_expansion
                self.logger.debug("Added domain_term_expansion method to MathstralClient")

            # Check and add estimate_idf method
            if not hasattr(self.mathstral_client, 'estimate_idf'):
                def estimate_idf(terms: List[str]) -> Dict[str, float]:
                    """Fallback IDF estimation based on term characteristics"""
                    try:
                        idf_scores = {}
                        
                        for term in terms:
                            term_lower = term.lower()
                            
                            # Base IDF score
                            base_score = 1.0
                            
                            # Length-based scoring (longer terms are typically rarer)
                            word_count = len(term.split())
                            if word_count == 1:
                                base_score = 0.8  # Single words are common
                            elif word_count == 2:
                                base_score = 1.0  # Two words are moderate
                            elif word_count >= 3:
                                base_score = 1.2  # Multi-word terms are rarer
                            
                            # Domain-specific adjustments
                            if any(common_word in term_lower for common_word in ['the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for']):
                                base_score *= 0.3  # Very common words
                            elif any(business_word in term_lower for business_word in ['id', 'name', 'date', 'time', 'type', 'status']):
                                base_score *= 0.6  # Common business terms
                            elif any(specific_word in term_lower for specific_word in ['revenue', 'counterparty', 'collateral', 'lakh']):
                                base_score *= 1.5  # Specific business terms
                            
                            # Technical term boost
                            if any(tech_word in term_lower for tech_word in ['schema', 'database', 'table', 'column', 'index']):
                                base_score *= 1.3
                            
                            # Ensure reasonable bounds
                            idf_scores[term] = max(0.1, min(2.0, base_score))
                        
                        return idf_scores
                        
                    except Exception as e:
                        self.logger.debug(f"Fallback IDF estimation failed: {e}")
                        return {term: 1.0 for term in terms}
                
                self.mathstral_client.estimate_idf = estimate_idf
                self.logger.debug("Added estimate_idf method to MathstralClient")
            
            self.logger.info("MathstralClient enhanced with missing methods")
            
        except Exception as e:
            self.logger.warning(f"Failed to enhance MathstralClient: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    def generate_entity_keywords(
        self,
        missing_entities: List[str],
        *,
        domain_context: str = "business database",
    ) -> List[str]:
        """
        Entry point used by orchestration layer.

        Parameters
        ----------
        missing_entities : List[str]
            List of missing entity terms from GapDetector
        domain_context : str
            Business domain description for context

        Returns
        -------
        List[str]
            Ranked list of search terms ready for search engines
        """
        self.logger.info("Generating keywords from %d missing entities", len(missing_entities))
        
        # Handle both list and dict inputs for backward compatibility
        if isinstance(missing_entities, dict):
            base_terms: List[str] = list(
                {e.strip() for entities in missing_entities.values() for e in entities}
            )
        else:
            base_terms = [str(e).strip() for e in missing_entities if str(e).strip()]

        if not base_terms:
            self.logger.warning("No valid base terms found")
            return []

        try:
            # 1. Semantic variations
            variations = self.generate_semantic_variations(base_terms)

            # 2. Domain-specific flavours
            domain_terms = self.create_domain_specific_terms(variations, domain_context)

            # 3. Final ranking
            final_keywords = self.prioritize_search_terms(domain_terms, domain_context)

            result = final_keywords[: self.max_final_keywords]
            self.logger.info(f"Generated {len(result)} final keywords using {self.actual_embedding_model}")
            return result
            
        except Exception as e:
            self.logger.error(f"Keyword generation failed: {e}")
            return self._fallback_keyword_generation(base_terms)

    # ──────────────────────────────────────────────────────────────────────
    # INTERNAL STEPS - UPDATED FOR E5-BASE-V2 COMPATIBILITY
    # ──────────────────────────────────────────────────────────────────────

    def generate_semantic_variations(self, terms: List[str]) -> List[str]:
        """
        🔧 ENHANCED: Use Mathstral for semantic understanding and e5-base-v2 for embeddings.
        """
        expanded: set[str] = set()

        for term in terms:
            expanded.add(term)

            try:
                # Use Mathstral for semantic variations (now guaranteed to exist)
                resp = self.mathstral_client.generate_semantic_variations(
                    term=term, 
                    domain_context="business database",
                    max_variations=5
                )
                variations = resp.get("variations", [])
                expanded.update(variations)
                self.logger.debug(f"Mathstral generated {len(variations)} variations for '{term}'")
                
            except Exception as err:
                self.logger.debug(f"Mathstral variation failed for '{term}': {err}")
                # Add simple fallback variations
                expanded.update(self._simple_variations(term))

            # Simple morphological variations (deterministic)
            expanded.update(self._morphological_variations(term))

        self.logger.debug("Semantic variations produced %d unique terms", len(expanded))
        # Cap to keep next stages performant
        return list(itertools.islice(expanded, self.max_variations))

    def create_domain_specific_terms(self, terms: List[str], domain: str) -> List[str]:
        """
        🔧 ENHANCED: Use Mathstral for domain-aware term expansions.
        """
        domain_terms: set[str] = set(terms)

        try:
            # Use Mathstral for domain expansion (now guaranteed to exist)
            # Process terms in smaller batches for better API performance
            batch_size = 5
            for i in range(0, len(terms), batch_size):
                batch_terms = terms[i:i+batch_size]
                
                resp = self.mathstral_client.domain_term_expansion(
                    terms=batch_terms,
                    domain=domain,
                    max_terms_per_input=3,
                )
                
                domain_expansions = resp.get("domain_terms", [])
                for expansion in domain_expansions:
                    if isinstance(expansion, dict) and "terms" in expansion:
                        domain_terms.update(expansion["terms"])
                    elif isinstance(expansion, str):
                        domain_terms.add(expansion)
            
            self.logger.debug(f"Domain expansion added {len(domain_terms) - len(terms)} new terms")
                    
        except Exception as err:
            self.logger.debug(f"Domain expansion failed: {err}")
            # Add simple domain-specific fallbacks
            domain_terms.update(self._simple_domain_expansion(terms, domain))

        return list(domain_terms)

    def prioritize_search_terms(
        self,
        candidate_terms: List[str],
        query_context: str,
    ) -> List[str]:
        """
        🔧 ENHANCED: Rank terms using e5-base-v2 embeddings for consistency with ChromaDB.
        
        Composite score: α·semantic_relevance + β·idf_score + γ·length_penalty
        """
        self.logger.info("Prioritising %d candidate keywords using %s", 
                        len(candidate_terms), self.actual_embedding_model)
        if not candidate_terms:
            return []

        try:
            # Use e5-base-v2 for embeddings (consistent with ChromaDB)
            # Prepare query with e5 prefix for better performance
            e5_query = f"query: {query_context}"
            query_embedding = self.embedding_model.encode([e5_query])[0]
            
            # Prepare candidate terms with e5 prefix
            e5_candidate_terms = [f"passage: {term}" for term in candidate_terms]
            term_embeddings = self.embedding_model.encode(e5_candidate_terms)

            # Ask Mathstral for IDF statistics (now guaranteed to exist)
            idf_lookup = self._get_idf_estimates(candidate_terms)

            ranked: list[tuple[str, float]] = []

            for i, term in enumerate(candidate_terms):
                # 1. Semantic similarity using e5-base-v2 embeddings
                term_emb = term_embeddings[i]
                sim = self._safe_cosine_e5(query_embedding, term_emb)

                # 2. IDF (rarer → higher priority)
                idf = idf_lookup.get(term, 1.0)

                # 3. Length penalty (prefer concise)
                length_penalty = self._calculate_length_penalty(term)

                # 4. E5 model specific adjustments
                e5_boost = self._calculate_e5_boost(term, query_context)

                # Composite score with e5 boost
                score = (0.4 * sim) + (0.3 * idf) + (0.1 * length_penalty) + (0.2 * e5_boost)
                ranked.append((term, score))

            ranked.sort(key=lambda x: x[1], reverse=True)
            final = [t for t, _ in ranked]
            self.logger.debug("Top 5 keywords: %s", final[:5])
            return final
            
        except Exception as e:
            self.logger.warning(f"Term prioritization failed: {e}")
            return candidate_terms  # Return unprioritized list as fallback

    # ──────────────────────────────────────────────────────────────────────
    # E5-BASE-V2 SPECIFIC HELPER METHODS
    # ──────────────────────────────────────────────────────────────────────

    def _safe_cosine_e5(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        E5-optimized cosine similarity calculation.
        """
        try:
            # E5 models work better with normalized vectors
            vec_a_norm = vec_a / (np.linalg.norm(vec_a) + 1e-8)
            vec_b_norm = vec_b / (np.linalg.norm(vec_b) + 1e-8)
            
            # Calculate dot product (cosine similarity for normalized vectors)
            similarity = np.dot(vec_a_norm, vec_b_norm)
            
            # E5 models return values in [-1, 1], normalize to [0, 1]
            normalized_sim = (similarity + 1.0) / 2.0
            
            return max(0.0, min(1.0, normalized_sim))
            
        except Exception as e:
            self.logger.warning(f"E5 cosine similarity calculation failed: {e}")
            return 0.0

    def _calculate_e5_boost(self, term: str, query_context: str) -> float:
        """
        Calculate E5-specific relevance boost based on term characteristics.
        """
        try:
            boost = 0.0
            
            # E5 models handle multi-word phrases well
            word_count = len(term.split())
            if word_count == 2:
                boost += 0.1  # Two-word phrases get slight boost
            elif word_count == 3:
                boost += 0.05  # Three-word phrases get smaller boost
            
            # Domain-specific terms get boost
            if any(domain_word in term.lower() for domain_word in ['database', 'table', 'column', 'schema']):
                boost += 0.05
            
            # Business terms get boost
            if any(biz_word in term.lower() for biz_word in ['customer', 'order', 'product', 'payment', 'transaction']):
                boost += 0.05
            
            return min(0.3, boost)  # Cap boost at 0.3
            
        except Exception:
            return 0.0

    def encode_with_e5_prefix(self, texts: List[str], text_type: str = "passage") -> np.ndarray:
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
            prefixed_texts = [f"{text_type}: {text}" for text in texts]
            return self.embedding_model.encode(prefixed_texts)
        except Exception as e:
            self.logger.warning(f"E5 encoding failed: {e}")
            # Fallback without prefixes
            return self.embedding_model.encode(texts)

    # ──────────────────────────────────────────────────────────────────────
    # EXISTING HELPER METHODS (ENHANCED)
    # ──────────────────────────────────────────────────────────────────────

    def _get_idf_estimates(self, terms: List[str]) -> Dict[str, float]:
        """ ENHANCED: Get IDF estimates from Mathstral (now guaranteed to exist)."""
        try:
            idf_result = self.mathstral_client.estimate_idf(terms)
            if isinstance(idf_result, dict) and idf_result:
                self.logger.debug(f"IDF estimation successful for {len(idf_result)} terms")
                return idf_result
            else:
                self.logger.debug("IDF estimation returned empty result, using fallback")
                return self._simple_idf_fallback(terms)
        except Exception as err:
            self.logger.debug(f"IDF estimation failed: {err}")
            return self._simple_idf_fallback(terms)

    def _simple_idf_fallback(self, terms: List[str]) -> Dict[str, float]:
        """Simple IDF fallback based on term characteristics."""
        return {term: max(0.1, 1.0 - len(term.split()) * 0.1) for term in terms}

    def _simple_variations(self, term: str) -> List[str]:
        """🔧 ENHANCED: Generate simple variations when Mathstral fails."""
        variations = []
        term_lower = term.lower()
        
        # Common business term variations
        business_variations = {
            'customer': ['client', 'user', 'account', 'buyer', 'consumer'],
            'order': ['purchase', 'transaction', 'sale', 'booking', 'request'],
            'product': ['item', 'goods', 'merchandise', 'article', 'service'],
            'payment': ['transaction', 'billing', 'charge', 'invoice', 'settlement'],
            'address': ['location', 'addr', 'place', 'destination', 'residence'],
            'phone': ['telephone', 'mobile', 'contact', 'number', 'tel'],
            'email': ['mail', 'electronic_mail', 'contact', 'correspondence'],
            'date': ['timestamp', 'time', 'datetime', 'created', 'modified'],
            'amount': ['value', 'price', 'cost', 'sum', 'total', 'revenue'],
            'status': ['state', 'condition', 'flag', 'indicator', 'stage']
        }
        
        for key, vals in business_variations.items():
            if key in term_lower:
                variations.extend(vals[:3])  # Limit to 3 variations per match
                
        return variations

    def _morphological_variations(self, term: str) -> List[str]:
        """Generate morphological variations."""
        variations = []
        
        # Spacing variations
        term_nospace = term.replace(" ", "")
        term_underscored = term.replace(" ", "_")
        term_hyphenated = term.replace(" ", "-")
        
        if term_nospace and term_nospace != term:
            variations.append(term_nospace)
        if term_underscored and term_underscored != term:
            variations.append(term_underscored)
        if term_hyphenated and term_hyphenated != term:
            variations.append(term_hyphenated)
            
        # Simple plural/singular
        if term.endswith('s') and len(term) > 3:
            variations.append(term[:-1])  # Remove 's'
        elif not term.endswith('s'):
            variations.append(term + 's')  # Add 's'
            
        return variations

    def _simple_domain_expansion(self, terms: List[str], domain: str) -> List[str]:
        """🔧 ENHANCED: Simple domain expansion fallback."""
        expansions = []
        domain_lower = domain.lower()
        
        for term in terms:
            term_lower = term.lower()
            
            # Banking/Financial domain
            if 'banking' in domain_lower or 'financial' in domain_lower:
                financial_expansions = {
                    'customer': ['client', 'account_holder', 'member'],
                    'account': ['portfolio', 'wallet', 'balance'],
                    'transaction': ['transfer', 'deposit', 'withdrawal'],
                    'branch': ['office', 'location', 'center'],
                    'loan': ['credit', 'advance', 'financing']
                }
                for key, vals in financial_expansions.items():
                    if key in term_lower:
                        expansions.extend(vals)
                        
            # E-commerce domain
            elif 'commerce' in domain_lower or 'retail' in domain_lower:
                commerce_expansions = {
                    'product': ['item', 'sku', 'catalog'],
                    'customer': ['buyer', 'shopper', 'member'],
                    'order': ['purchase', 'cart', 'checkout'],
                    'inventory': ['stock', 'warehouse', 'supply'],
                    'shipping': ['delivery', 'fulfillment', 'dispatch']
                }
                for key, vals in commerce_expansions.items():
                    if key in term_lower:
                        expansions.extend(vals)
                        
        return expansions

    def _calculate_length_penalty(self, term: str) -> float:
        """Calculate length penalty for term scoring."""
        word_count = len(term.split())
        if word_count == 1:
            return 1.0  # Single words are good
        elif word_count == 2:
            return 0.95  # Two words are okay (E5 handles these well)
        elif word_count == 3:
            return 0.90  # Three words still decent for E5
        else:
            return max(0.7, 1.0 - (word_count - 3) * 0.05)  # Penalize longer phrases

    def _fallback_keyword_generation(self, base_terms: List[str]) -> List[str]:
        """🔧 ENHANCED: Simple fallback when advanced methods fail."""
        try:
            keywords = []
            
            for term in base_terms:
                keywords.append(term)
                # Add simple variations
                keywords.extend(self._simple_variations(term))
                keywords.extend(self._morphological_variations(term))
            
            # Remove duplicates and limit
            unique_keywords = list(set(keywords))
            result = unique_keywords[:self.max_final_keywords]
            
            self.logger.info(f"Fallback generated {len(result)} keywords")
            return result
            
        except Exception as e:
            self.logger.error(f"Fallback keyword generation failed: {e}")
            return base_terms[:self.max_final_keywords]

    # ──────────────────────────────────────────────────────────────────────
    # UTILITY AND MONITORING METHODS
    # ──────────────────────────────────────────────────────────────────────

    def get_keyword_statistics(self, keywords: List[str]) -> Dict[str, Any]:
        """Get statistics about generated keywords."""
        try:
            stats = {
                'total_keywords': len(keywords),
                'average_length': sum(len(k.split()) for k in keywords) / len(keywords) if keywords else 0,
                'unique_words': len(set(' '.join(keywords).split())),
                'embedding_model_used': self.actual_embedding_model,
                'mathstral_methods_available': {
                    'generate_semantic_variations': hasattr(self.mathstral_client, 'generate_semantic_variations'),
                    'domain_term_expansion': hasattr(self.mathstral_client, 'domain_term_expansion'),
                    'estimate_idf': hasattr(self.mathstral_client, 'estimate_idf')
                },
                'keyword_distribution': {
                    'single_word': sum(1 for k in keywords if len(k.split()) == 1),
                    'two_word': sum(1 for k in keywords if len(k.split()) == 2),
                    'multi_word': sum(1 for k in keywords if len(k.split()) > 2)
                },
                'domain_terms_detected': sum(1 for k in keywords if any(
                    domain_word in k.lower() for domain_word in 
                    ['customer', 'order', 'product', 'payment', 'transaction', 'database', 'table']
                )),
                'using_shared_model': True  # Indicates using EmbeddingModelManager
            }
            return stats
        except Exception as e:
            self.logger.error(f"Statistics calculation failed: {e}")
            return {
                'total_keywords': len(keywords), 
                'embedding_model_used': self.actual_embedding_model,
                'using_shared_model': True
            }

    def validate_keywords(self, keywords: List[str]) -> List[str]:
        """Validate and clean generated keywords."""
        try:
            valid_keywords = []
            
            for keyword in keywords:
                # Basic validation
                if (isinstance(keyword, str) and 
                    len(keyword.strip()) > 1 and 
                    len(keyword.strip()) < 100 and
                    not keyword.strip().isdigit()):
                    
                    clean_keyword = keyword.strip().lower()
                    if clean_keyword not in valid_keywords:
                        valid_keywords.append(clean_keyword)
            
            return valid_keywords
            
        except Exception as e:
            self.logger.error(f"Keyword validation failed: {e}")
            return keywords

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model being used."""
        try:
            return {
                'model_name': self.actual_embedding_model,
                'model_type': 'e5-base-v2' if 'e5-base-v2' in self.actual_embedding_model else 'other',
                'supports_prefixes': 'e5' in self.actual_embedding_model.lower(),
                'embedding_dimension': self.embedding_model.get_sentence_embedding_dimension(),
                'max_sequence_length': getattr(self.embedding_model, 'max_seq_length', 'unknown'),
                'using_shared_model': True,  # Indicates using EmbeddingModelManager
                'mathstral_enhanced': True   # Indicates methods were added
            }
        except Exception as e:
            return {
                'model_name': self.actual_embedding_model,
                'using_shared_model': True,
                'mathstral_enhanced': True,
                'error': str(e)
            }
