"""
Dynamic Intent Router Module
AI-Powered Intent Classification and Query Routing System
Replaces static keyword matching with adaptive AI classification
Version: 1.0.1
Date: 2025-08-14
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

@dataclass
class RoutingDecision:
    """Structured routing decision with full transparency"""
    selected_route: str
    confidence: float
    classification_time_ms: float
    intent_analysis: Dict[str, Any]
    routing_method: str
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass 
class RoutingStatistics:
    """Comprehensive routing performance statistics"""
    total_classifications: int = 0
    successful_routes: int = 0
    ai_failures: int = 0
    route_distribution: Dict[str, int] = field(default_factory=dict)
    average_classification_time_ms: float = 0.0
    ai_success_rate: float = 0.0

class DynamicIntentRouter:
    """
    AI-Powered Dynamic Intent Classification System
    Replaces static keyword matching with adaptive AI routing
    """
    
    def __init__(self, async_client_manager, logger: logging.Logger):
        self.async_client_manager = async_client_manager
        self.logger = logger
        
        # Routing statistics and learning
        self.stats = RoutingStatistics()
        self.intent_history: Dict[str, Dict[str, Any]] = {}
        self.successful_patterns: Dict[str, str] = {}
        
        # Dynamic intent to route mapping
        self.intent_routes = {
            "banking_operations": "nlp_schema_enhanced",
            "data_exploration": "nlp_schema_enhanced", 
            "analytical_queries": "enhanced_sql_strict",
            "direct_sql": "sql_generation",
            "schema_discovery": "nlp_schema_enhanced",
            "customer_inquiry": "nlp_schema_enhanced",
            "financial_analysis": "enhanced_sql_strict"
        }
        
        # Route priorities for tie-breaking
        self.route_priorities = {
            "nlp_schema_enhanced": 3,  # Highest priority - full AI pipeline
            "enhanced_sql_strict": 2,   # Medium priority - analytical queries
            "sql_generation": 1,        # Lower priority - direct SQL
            "fallback": 0              # Last resort
        }
        
        self.logger.info("Dynamic Intent Router initialized with AI classification")

    async def classify_and_route(self, query: str, context: Dict[str, Any]) -> RoutingDecision:
        """
        Main dynamic classification and routing method
        Returns comprehensive routing decision with full transparency
        """
        start_time = datetime.now()
        self.stats.total_classifications += 1
        
        try:
            self.logger.info(f"AI Classification starting for: '{query[:60]}...'")
            
            # Multi-method classification approach
            classification_results = await self._multi_method_classification(query, context)
            
            # Select optimal route using ensemble decision
            routing_decision = self._ensemble_route_selection(classification_results, query)
            
            # Update statistics
            self._update_routing_statistics(routing_decision, start_time)
            
            # Store for adaptive learning
            self._store_routing_pattern(query, routing_decision)
            
            # Log comprehensive decision
            self._log_routing_decision(query, routing_decision)
            
            return routing_decision
            
        except Exception as e:
            self.stats.ai_failures += 1
            self.logger.error(f"Dynamic classification failed for '{query[:50]}...': {e}")
            return self._create_fallback_decision(query, str(e))

    async def _multi_method_classification(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Multi-method classification using various AI techniques"""
        classification_methods = {}
        
        # Method 1: Primary AI Intent Classification
        classification_methods["ai_primary"] = await self._ai_intent_classification(query, context)
        
        # Method 2: Semantic Similarity Analysis
        classification_methods["semantic"] = await self._semantic_similarity_analysis(query)
        
        # Method 3: Contextual Pattern Recognition
        classification_methods["contextual"] = self._contextual_pattern_analysis(query, context)
        
        # Method 4: Historical Pattern Matching
        classification_methods["historical"] = self._historical_pattern_matching(query)
        
        return classification_methods

    async def _ai_intent_classification(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Primary AI-powered intent classification using AsyncClientManager"""
        if not self.async_client_manager:
            self.logger.warning("AsyncClientManager not available for AI classification")
            return self._rule_based_fallback_classification(query)
        
        classification_prompt = f"""
        Analyze this database query and classify the user's intent with detailed reasoning.
        
        Query: "{query}"
        Context: {json.dumps(context, indent=2)}
        
        Available Intent Categories:
        1. BANKING_OPERATIONS - Customer accounts, loans, transactions, counterparty information
        2. DATA_EXPLORATION - General data discovery, searching, listing records
        3. ANALYTICAL_QUERIES - Complex analysis, aggregations, reporting, calculations
        4. DIRECT_SQL - Explicit SQL commands or database operations
        5. SCHEMA_DISCOVERY - Finding table structures, column information, metadata
        6. CUSTOMER_INQUIRY - Specific customer information requests
        7. FINANCIAL_ANALYSIS - Financial metrics, performance analysis, risk assessment
        
        Consider these factors:
        - Banking domain terminology (customer, counterparty, account, loan, etc.)
        - Action words (show, find, get, calculate, analyze, etc.)
        - Data scope (single record, multiple records, aggregated data)
        - Query complexity (simple lookup vs complex analysis)
        
        Respond with this exact JSON format:
        {{
            "primary_intent": "BANKING_OPERATIONS",
            "confidence": 0.85,
            "detected_intents": [
                {{"intent": "BANKING_OPERATIONS", "confidence": 0.85, "evidence": ["counterparty", "details", "customer"]}},
                {{"intent": "DATA_EXPLORATION", "confidence": 0.3, "evidence": ["show", "all"]}}
            ],
            "reasoning": "Query requests specific banking entity information with data retrieval actions. Contains banking terminology and indicates customer data access.",
            "domain_context": "banking",
            "complexity_level": "medium",
            "action_type": "retrieve",
            "data_scope": "multiple_records"
        }}
        
        Respond with only the JSON object.
        """
        
        try:
            ai_result = await self.async_client_manager.generate_sql_async(classification_prompt)
            
            # CRITICAL FIX: Handle different response formats
            response_text = self._safe_extract_response(ai_result)
            
            # Clean and parse response
            cleaned_response = self._clean_ai_response(response_text)
            
            try:
                parsed_result = json.loads(cleaned_response)
                
                # Validate and enhance result
                enhanced_result = self._validate_and_enhance_ai_result(parsed_result, query)
                
                self.logger.debug(f"AI Classification successful: {enhanced_result.get('primary_intent')} (confidence: {enhanced_result.get('confidence', 0.0):.2f})")
                return enhanced_result
                
            except json.JSONDecodeError as json_err:
                self.logger.warning(f"AI returned malformed JSON: {json_err}")
                return self._parse_malformed_ai_response(response_text, query)
                
        except Exception as e:
            self.logger.error(f"AI classification request failed: {e}")
            return self._rule_based_fallback_classification(query)

    def _safe_extract_response(self, ai_result: Any) -> str:
        """Safely extract response text from various formats"""
        try:
            if isinstance(ai_result, dict):
                # Try different possible keys
                for key in ['sql', 'response', 'text', 'content', 'result']:
                    if key in ai_result:
                        value = ai_result[key]
                        return str(value) if value is not None else '{}'
                return str(ai_result)
            
            elif isinstance(ai_result, list):
                if ai_result and len(ai_result) > 0:
                    return str(ai_result[0])
                return '{}'
            
            elif isinstance(ai_result, str):
                return ai_result
            
            else:
                return str(ai_result)
                
        except Exception as e:
            self.logger.warning(f"Failed to extract response: {e}")
            return '{}'

    async def _semantic_similarity_analysis(self, query: str) -> Dict[str, Any]:
        """Semantic similarity analysis using example-based matching"""
        # Define semantic examples for each intent category
        intent_examples = {
            "banking_operations": [
                "show customer account details", "get counterparty information", 
                "find loan applications", "display account balances", "list client records",
                "customer account status", "banking relationship details", "give me counterparty details"
            ],
            "data_exploration": [
                "show all records", "find information about", "list all entries",
                "discover data", "explore database", "search for records",
                "display available data", "what data do we have"
            ],
            "analytical_queries": [
                "calculate total amounts", "analyze trends by region", "aggregate loan data",
                "summarize performance", "compare metrics", "statistical analysis",
                "generate reports", "performance indicators"
            ],
            "customer_inquiry": [
                "specific customer details", "individual client information",
                "personal account data", "customer profile", "client history"
            ]
        }
        
        similarity_scores = {}
        
        for intent, examples in intent_examples.items():
            max_similarity = 0.0
            best_match = ""
            
            for example in examples:
                similarity = self._calculate_text_similarity(query.lower(), example.lower())
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = example
            
            similarity_scores[intent] = {
                "similarity": max_similarity,
                "best_match": best_match
            }
        
        # Find best matching intent
        best_intent = max(similarity_scores.keys(), key=lambda k: similarity_scores[k]["similarity"])
        best_score = similarity_scores[best_intent]["similarity"]
        
        return {
            "method": "semantic_similarity",
            "best_intent": best_intent,
            "confidence": min(best_score * 1.2, 1.0),  # Boost similarity scores
            "similarity_scores": similarity_scores,
            "reasoning": f"Best semantic match with '{similarity_scores[best_intent]['best_match']}'"
        }

    def _contextual_pattern_analysis(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Contextual pattern recognition based on query structure and context"""
        query_lower = query.lower()
        
        # Enhanced banking domain indicators with weights
        banking_indicators = {
            'counterparty': 0.9, 'customer': 0.8, 'client': 0.7, 'account': 0.8,
            'loan': 0.8, 'balance': 0.7, 'transaction': 0.8, 'banking': 0.9,
            'financial': 0.7, 'credit': 0.7, 'debit': 0.6, 'payment': 0.7,
            'address': 0.6, 'contact': 0.6, 'relationship': 0.7
        }
        
        # Enhanced action indicators with weights
        action_indicators = {
            'show': 0.6, 'display': 0.6, 'get': 0.7, 'find': 0.7, 'list': 0.6,
            'give': 0.7, 'provide': 0.7, 'fetch': 0.7, 'retrieve': 0.8,
            'details': 0.8, 'information': 0.7, 'data': 0.6, 'records': 0.7,
            'all': 0.5, 'every': 0.5, 'each': 0.5
        }
        
        # Calculate pattern scores
        banking_score = sum(weight for term, weight in banking_indicators.items() if term in query_lower)
        action_score = sum(weight for term, weight in action_indicators.items() if term in query_lower)
        
        # Determine intent based on pattern analysis
        if banking_score >= 0.8 and action_score >= 0.6:
            primary_intent = "banking_operations"
            confidence = min((banking_score + action_score) / 2, 1.0)
        elif banking_score >= 0.5:
            primary_intent = "banking_operations"
            confidence = banking_score * 0.8
        elif action_score >= 0.8:
            primary_intent = "data_exploration"
            confidence = action_score * 0.7
        else:
            primary_intent = "data_exploration"
            confidence = 0.5
        
        return {
            "method": "contextual_pattern",
            "primary_intent": primary_intent,
            "confidence": confidence,
            "banking_score": banking_score,
            "action_score": action_score,
            "reasoning": f"Pattern analysis: banking_score={banking_score:.2f}, action_score={action_score:.2f}"
        }

    def _historical_pattern_matching(self, query: str) -> Dict[str, Any]:
        """Historical pattern matching for adaptive learning"""
        # Find similar historical queries
        similar_queries = []
        
        for hist_query, hist_data in self.intent_history.items():
            similarity = self._calculate_text_similarity(query.lower(), hist_query.lower())
            if similarity > 0.6:  # Similarity threshold
                similar_queries.append({
                    "query": hist_query,
                    "similarity": similarity,
                    "route": hist_data.get("route"),
                    "success_rate": hist_data.get("success_rate", 0.5),
                    "confidence": hist_data.get("confidence", 0.5)
                })
        
        if similar_queries:
            # Sort by similarity * success_rate
            similar_queries.sort(key=lambda x: x["similarity"] * x["success_rate"], reverse=True)
            best_match = similar_queries[0]
            
            return {
                "method": "historical_pattern",
                "recommended_route": best_match["route"],
                "confidence": best_match["similarity"] * best_match["success_rate"],
                "historical_matches": len(similar_queries),
                "best_match": best_match,
                "reasoning": f"Historical pattern match with {best_match['similarity']:.2f} similarity"
            }
        
        return {
            "method": "historical_pattern",
            "recommended_route": None,
            "confidence": 0.0,
            "historical_matches": 0,
            "reasoning": "No similar historical patterns found"
        }

    def _ensemble_route_selection(self, classification_results: Dict[str, Any], query: str) -> RoutingDecision:
        """Ensemble decision making from multiple classification methods"""
        # Weight different classification methods
        method_weights = {
            "ai_primary": 0.4,      # Highest weight for AI classification
            "semantic": 0.25,       # Good weight for semantic similarity
            "contextual": 0.25,     # Pattern analysis weight
            "historical": 0.1       # Learning from history
        }
        
        # Calculate weighted scores for each route
        route_scores = {}
        decision_evidence = []
        
        for method_name, method_result in classification_results.items():
            method_weight = method_weights.get(method_name, 0.1)
            
            if method_name == "ai_primary":
                primary_intent = method_result.get("primary_intent", "").lower()
                confidence = method_result.get("confidence", 0.0)
                
                # Map AI intent to route
                for intent_key, route in self.intent_routes.items():
                    if intent_key.replace("_", "") in primary_intent.replace("_", ""):
                        route_scores[route] = route_scores.get(route, 0) + (confidence * method_weight)
                        decision_evidence.append(f"AI: {primary_intent} → {route} (weight: {confidence * method_weight:.2f})")
                        break
            
            elif method_name == "semantic":
                best_intent = method_result.get("best_intent", "")
                confidence = method_result.get("confidence", 0.0)
                
                if best_intent in self.intent_routes:
                    route = self.intent_routes[best_intent]
                    route_scores[route] = route_scores.get(route, 0) + (confidence * method_weight)
                    decision_evidence.append(f"Semantic: {best_intent} → {route} (weight: {confidence * method_weight:.2f})")
            
            elif method_name == "contextual":
                intent = method_result.get("primary_intent", "")
                confidence = method_result.get("confidence", 0.0)
                
                intent_key = intent.lower().replace(" ", "_")
                if intent_key in self.intent_routes:
                    route = self.intent_routes[intent_key]
                    route_scores[route] = route_scores.get(route, 0) + (confidence * method_weight)
                    decision_evidence.append(f"Pattern: {intent} → {route} (weight: {confidence * method_weight:.2f})")
            
            elif method_name == "historical":
                recommended_route = method_result.get("recommended_route")
                confidence = method_result.get("confidence", 0.0)
                
                if recommended_route and confidence > 0:
                    route_scores[recommended_route] = route_scores.get(recommended_route, 0) + (confidence * method_weight)
                    decision_evidence.append(f"Historical: → {recommended_route} (weight: {confidence * method_weight:.2f})")
        
        # Select best route
        if route_scores:
            # Sort by score, then by priority for tie-breaking
            best_route = max(route_scores.keys(), 
                           key=lambda r: (route_scores[r], self.route_priorities.get(r, 0)))
            final_confidence = min(route_scores[best_route], 1.0)
        else:
            # Fallback to default route
            best_route = "nlp_schema_enhanced"
            final_confidence = 0.5
            decision_evidence.append("Fallback: No clear classification → nlp_schema_enhanced")
        
        # Create alternatives list
        alternatives = []
        for route, score in sorted(route_scores.items(), key=lambda x: x[1], reverse=True):
            alternatives.append({
                "route": route,
                "score": score,
                "priority": self.route_priorities.get(route, 0)
            })
        
        return RoutingDecision(
            selected_route=best_route,
            confidence=final_confidence,
            classification_time_ms=0.0,  # Will be updated by caller
            intent_analysis={
                "classification_methods": classification_results,
                "decision_evidence": decision_evidence,
                "route_scores": route_scores,
                "ensemble_reasoning": f"Selected {best_route} with confidence {final_confidence:.2f}"
            },
            routing_method="ensemble_ai_classification",
            alternatives=alternatives
        )

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity using word overlap"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    def _clean_ai_response(self, response_text: str) -> str:
        """Clean AI response to extract JSON - COMPLETELY FIXED"""
        # CRITICAL FIX: Handle case where response_text might be a list
        if isinstance(response_text, list):
            response_text = str(response_text) if response_text else "{}"
        
        # Convert to string if not already
        if not isinstance(response_text, str):
            response_text = str(response_text)
        
        # Remove markdown formatting
        if "```":
            parts = response_text.split("```json")
            if len(parts) > 1:
                # Get the content after ``````
                content = parts[1]
                if "```":
                    response_text = content.split("```")[0]
                else:
                    response_text = content
        elif "```":
            parts = response_text.split("```")
            if len(parts) > 1:
                # Get the content between the first set of backticks
                response_text = parts[1]
        
        # Remove extra whitespace and common prefixes
        response_text = response_text.strip()
        
        return response_text

    def _validate_and_enhance_ai_result(self, parsed_result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Validate and enhance AI classification result"""
        # Ensure required fields exist with defaults
        enhanced_result = {
            "primary_intent": parsed_result.get("primary_intent", "DATA_EXPLORATION"),
            "confidence": max(0.0, min(1.0, parsed_result.get("confidence", 0.5))),
            "detected_intents": parsed_result.get("detected_intents", []),
            "reasoning": parsed_result.get("reasoning", "AI classification completed"),
            "domain_context": parsed_result.get("domain_context", "general"),
            "complexity_level": parsed_result.get("complexity_level", "medium"),
            "query_length": len(query),
            "query_word_count": len(query.split())
        }
        
        return enhanced_result

    def _parse_malformed_ai_response(self, response_text: str, query: str) -> Dict[str, Any]:
        """Parse malformed AI response using pattern matching"""
        response_lower = response_text.lower()
        
        if "banking" in response_lower or "customer" in response_lower or "counterparty" in response_lower:
            return {
                "primary_intent": "BANKING_OPERATIONS",
                "confidence": 0.7,
                "reasoning": "Parsed from malformed AI response: banking context detected",
                "parsing_method": "pattern_fallback"
            }
        elif "analytical" in response_lower or "calculate" in response_lower:
            return {
                "primary_intent": "ANALYTICAL_QUERIES",
                "confidence": 0.6,
                "reasoning": "Parsed from malformed AI response: analytical context detected",
                "parsing_method": "pattern_fallback"
            }
        elif "sql" in response_lower or "select" in response_lower:
            return {
                "primary_intent": "DIRECT_SQL",
                "confidence": 0.8,
                "reasoning": "Parsed from malformed AI response: SQL context detected",
                "parsing_method": "pattern_fallback"
            }
        else:
            return self._rule_based_fallback_classification(query)

    def _rule_based_fallback_classification(self, query: str) -> Dict[str, Any]:
        """Rule-based fallback classification when AI fails"""
        query_lower = query.lower()
        
        # Enhanced banking terms detection
        banking_terms = ['customer', 'counterparty', 'client', 'account', 'loan', 'balance', 'transaction', 'address', 'details']
        banking_detected = any(term in query_lower for term in banking_terms)
        
        # SQL terms detection
        sql_terms = ['select', 'from', 'where', 'join', 'insert', 'update', 'delete']
        sql_detected = any(term in query_lower for term in sql_terms)
        
        # Analysis terms detection
        analysis_terms = ['calculate', 'analyze', 'aggregate', 'sum', 'count', 'average', 'total']
        analysis_detected = any(term in query_lower for term in analysis_terms)
        
        if sql_detected:
            return {
                "primary_intent": "DIRECT_SQL",
                "confidence": 0.8,
                "reasoning": "Rule-based fallback: SQL keywords detected",
                "fallback_method": "rule_based"
            }
        elif analysis_detected and banking_detected:
            return {
                "primary_intent": "ANALYTICAL_QUERIES",
                "confidence": 0.7,
                "reasoning": "Rule-based fallback: analysis + banking terms detected",
                "fallback_method": "rule_based"
            }
        elif banking_detected:
            return {
                "primary_intent": "BANKING_OPERATIONS",
                "confidence": 0.6,
                "reasoning": "Rule-based fallback: banking terms detected",
                "fallback_method": "rule_based"
            }
        else:
            return {
                "primary_intent": "DATA_EXPLORATION",
                "confidence": 0.5,
                "reasoning": "Rule-based fallback: default classification",
                "fallback_method": "rule_based"
            }

    def _create_fallback_decision(self, query: str, error_message: str) -> RoutingDecision:
        """Create fallback routing decision when classification fails"""
        return RoutingDecision(
            selected_route="nlp_schema_enhanced",  # Safe default
            confidence=0.5,
            classification_time_ms=0.0,
            intent_analysis={
                "primary_intent": "FALLBACK_CLASSIFICATION",
                "ai_reasoning": f"Classification failed: {error_message}",
                "confidence_score": 0.5,
                "fallback_used": True
            },
            routing_method="error_fallback"
        )

    def _update_routing_statistics(self, decision: RoutingDecision, start_time: datetime):
        """Update comprehensive routing statistics"""
        # Calculate classification time
        classification_time = (datetime.now() - start_time).total_seconds() * 1000
        decision.classification_time_ms = classification_time
        
        # Update statistics
        self.stats.successful_routes += 1
        route = decision.selected_route
        self.stats.route_distribution[route] = self.stats.route_distribution.get(route, 0) + 1
        
        # Update average classification time
        total_time = self.stats.average_classification_time_ms * (self.stats.total_classifications - 1)
        self.stats.average_classification_time_ms = (total_time + classification_time) / self.stats.total_classifications
        
        # Update AI success rate
        if self.stats.total_classifications > 0:
            self.stats.ai_success_rate = ((self.stats.total_classifications - self.stats.ai_failures) / 
                                        self.stats.total_classifications) * 100

    def _store_routing_pattern(self, query: str, decision: RoutingDecision):
        """Store routing pattern for adaptive learning"""
        self.intent_history[query] = {
            "route": decision.selected_route,
            "confidence": decision.confidence,
            "timestamp": datetime.now(),
            "success_rate": 0.8,  # Will be updated based on actual query success
            "intent": decision.intent_analysis.get("primary_intent", "unknown")
        }

    def _log_routing_decision(self, query: str, decision: RoutingDecision):
        """Log comprehensive routing decision"""
        self.logger.info(f"AI Routing Decision:")
        self.logger.info(f"   Query: '{query[:80]}...'")
        self.logger.info(f"   Selected Route: {decision.selected_route}")
        self.logger.info(f"   Confidence: {decision.confidence:.2f}")
        self.logger.info(f"   Classification Time: {decision.classification_time_ms:.1f}ms")
        self.logger.info(f"   Primary Intent: {decision.intent_analysis.get('primary_intent', 'unknown')}")
        self.logger.info(f"   Method: {decision.routing_method}")
        
        if decision.alternatives:
            alt_routes = [f"{alt['route']}({alt['score']:.2f})" for alt in decision.alternatives[:3]]
            self.logger.info(f"   Alternatives: {', '.join(alt_routes)}")

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing performance statistics"""
        total_routes = sum(self.stats.route_distribution.values())
        most_used_route: str = max(self.stats.route_distribution.keys(), 
                      key=lambda route: self.stats.route_distribution[route]
                            ) if self.stats.route_distribution else "none"

        
        return {
            "total_classifications": self.stats.total_classifications,
            "successful_routes": self.stats.successful_routes,
            "ai_failures": self.stats.ai_failures,
            "ai_success_rate": round(self.stats.ai_success_rate, 2),
            "average_classification_time_ms": round(self.stats.average_classification_time_ms, 2),
            "route_distribution": dict(self.stats.route_distribution),
            "route_distribution_percentages": {
                route: round((count / max(total_routes, 1)) * 100, 1)
                for route, count in self.stats.route_distribution.items()
            },
            "most_used_route": most_used_route,
            "adaptive_learning": {
                "historical_patterns_stored": len(self.intent_history),
                "successful_patterns": len(self.successful_patterns)
            }
        }

    def update_routing_success(self, query: str, route: str, success: bool):
        """Update routing success for adaptive learning"""
        if query in self.intent_history:
            current_success_rate = self.intent_history[query].get("success_rate", 0.5)
            # Update success rate with exponential moving average
            new_success_rate = 0.7 * current_success_rate + 0.3 * (1.0 if success else 0.0)
            self.intent_history[query]["success_rate"] = new_success_rate
            
            if success:
                self.successful_patterns[query] = route

    def get_intent_analysis_for_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Get stored intent analysis for a specific query"""
        return self.intent_history.get(query)
