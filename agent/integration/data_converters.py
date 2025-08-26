"""
Data Format Converters for NLP-Schema Integration
Handles precise data type conversions between NLP Processor and Schema Retrieval Agent
Based on actual component interfaces and data structures
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Import our standardized data models
from .data_models import (
    SchemaSearchRequest, SchemaSearchResponse, NLPEnhancedQuery, 
    QueryIntent, ComponentStatus, NLPInsights
)

logger = logging.getLogger(__name__)

class NLPToSchemaConverter:
    """
    Converts NLP Processor output to Schema Retrieval Agent compatible format
    Based on actual EnhancedQueryResult -> SchemaSearchRequest conversion
    """
    
    def __init__(self):
        self.logger = logger
        
        # Intent mapping from NLP processor format to our standardized format
        self.intent_mapping = {
            'temporal_analysis': QueryIntent.TEMPORAL_FILTER,
            'regional_aggregation': QueryIntent.AGGREGATION,
            'customer_analysis': QueryIntent.SIMPLE_LOOKUP,
            'deviation_analysis': QueryIntent.COMPLEX_ANALYSIS,
            'defaulter_analysis': QueryIntent.COMPLEX_ANALYSIS,
            'collateral_analysis': QueryIntent.AGGREGATION,
            'lookup': QueryIntent.SIMPLE_LOOKUP,
            'aggregation': QueryIntent.AGGREGATION,
            'analysis': QueryIntent.COMPLEX_ANALYSIS,
            'temporal': QueryIntent.TEMPORAL_FILTER,
            'join': QueryIntent.JOIN_ANALYSIS,
            'reporting': QueryIntent.REPORTING
        }
    
    def convert_nlp_result_to_schema_request(
        self,
        nlp_result: Any,  # EnhancedQueryResult from NLP Processor
        original_query: str,
        request_id: Optional[str] = None
    ) -> SchemaSearchRequest:
        """
        Convert NLP Processor EnhancedQueryResult to SchemaSearchRequest
        
        Args:
            nlp_result: EnhancedQueryResult from NLP processor
            original_query: Original user query
            request_id: Optional request ID
            
        Returns:
            SchemaSearchRequest ready for Schema Retrieval Agent
        """
        try:
            request_id = request_id or str(uuid.uuid4())
            self.logger.debug(f"[{request_id}] Converting NLP result to schema request")
            
            # Extract structured_query from NLP result
            structured_query = self._extract_structured_query(nlp_result)
            
            # Extract intent classification
            intent_data = structured_query.get('intent_classification', {})
            primary_intent = intent_data.get('primary_intent', 'unknown')
            confidence = intent_data.get('confidence', 0.8)
            
            # Map to standardized intent
            mapped_intent = self._map_intent_to_standard(primary_intent)
            
            # Extract search scope (target tables/columns)
            search_scope = structured_query.get('search_scope', {})
            target_tables = search_scope.get('target_tables', [])
            target_columns = search_scope.get('target_columns', [])
            
            # Extract enhanced query
            enhanced_query = structured_query.get('refined_query', original_query)
            
            # Determine processing parameters based on intent and complexity
            max_results = self._determine_max_results(mapped_intent, confidence)
            include_joins = self._should_include_joins(intent_data, target_tables)
            include_xml = self._should_include_xml(intent_data)
            min_confidence = self._determine_min_confidence(confidence)
            
            # Extract business context
            business_context = getattr(nlp_result, 'business_context', {})
            
            # Create comprehensive NLP context for schema agent
            nlp_context = {
                'original_nlp_result': self._serialize_nlp_result(nlp_result),
                'intent_classification': intent_data,
                'search_scope': search_scope,
                'business_context': business_context,
                'field_mappings': self._extract_field_mappings(nlp_result),
                'temporal_expressions': structured_query.get('temporal_expressions', []),
                'geographic_entities': structured_query.get('geographic_entities', []),
                'numerical_entities': structured_query.get('numerical_entities', []),
                'confidence_scores': getattr(nlp_result, 'confidence_scores', {}),
                'processing_hints': {
                    'nlp_confidence': confidence,
                    'complexity_score': intent_data.get('complexity', 0.5),
                    'requires_aggregation': 'aggregation' in primary_intent.lower(),
                    'requires_temporal_filter': 'temporal' in primary_intent.lower()
                }
            }
            
            # Create schema search request
            schema_request = SchemaSearchRequest(
                request_id=request_id,
                query=original_query,
                enhanced_query=enhanced_query,
                target_tables=target_tables,
                target_columns=target_columns,
                intent_classification={
                    'primary_intent': mapped_intent.value,
                    'confidence': confidence,
                    'secondary_intents': intent_data.get('secondary_intents', []),
                    'requires_joins': include_joins,
                    'requires_xml': include_xml,
                    'original_intent': primary_intent
                },
                include_xml=include_xml,
                include_joins=include_joins,
                max_results=max_results,
                min_confidence=min_confidence,
                nlp_context=nlp_context,
                processing_hints={
                    'nlp_enhanced': True,
                    'complexity_score': intent_data.get('complexity', 0.5),
                    'confidence_level': 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low',
                    'processing_priority': 'high' if mapped_intent in [QueryIntent.AGGREGATION, QueryIntent.COMPLEX_ANALYSIS] else 'normal'
                }
            )
            
            self.logger.debug(f"[{request_id}] NLP->Schema conversion successful")
            return schema_request
            
        except Exception as e:
            self.logger.error(f"[{request_id}] NLP->Schema conversion failed: {e}")
            # Return fallback request
            return self._create_fallback_request(original_query, request_id, str(e)) # type: ignore
    
    def _extract_structured_query(self, nlp_result: Any) -> Dict[str, Any]:
        """Extract structured_query from NLP result with multiple fallback approaches"""
        try:
            # Method 1: Direct attribute access
            if hasattr(nlp_result, 'structured_query'):
                return nlp_result.structured_query
            
            # Method 2: Dictionary access
            if isinstance(nlp_result, dict) and 'structured_query' in nlp_result:
                return nlp_result['structured_query']
            
            # Method 3: to_dict() method
            if hasattr(nlp_result, 'to_dict'):
                result_dict = nlp_result.to_dict() # type: ignore
                return result_dict.get('structured_query', {})
            
            # Method 4: Check for common NLP result patterns
            if isinstance(nlp_result, dict):
                # Look for intent classification directly
                if 'intent_classification' in nlp_result:
                    return nlp_result
                
                # Look for nested structure
                for key in ['result', 'data', 'analysis']:
                    if key in nlp_result and isinstance(nlp_result[key], dict):
                        nested = nlp_result[key]
                        if 'structured_query' in nested:
                            return nested['structured_query']
                        if 'intent_classification' in nested:
                            return nested
            
            self.logger.warning("Could not extract structured_query, using empty dict")
            return {}
            
        except Exception as e:
            self.logger.error(f"Error extracting structured_query: {e}")
            return {}
    
    def _extract_field_mappings(self, nlp_result: Any) -> List[Dict[str, Any]]:
        """Extract field mappings from NLP result"""
        try:
            field_mappings = []
            
            # Check for field_mappings attribute
            if hasattr(nlp_result, 'field_mappings'):
                mappings = nlp_result.field_mappings
                for mapping in mappings:
                    if hasattr(mapping, 'to_dict'):
                        field_mappings.append(mapping.to_dict())
                    else:
                        field_mappings.append({
                            'table': getattr(mapping, 'table', 'unknown'),
                            'column': getattr(mapping, 'name', getattr(mapping, 'column', 'unknown')),
                            'field_type': getattr(mapping, 'field_type', 'unknown'),
                            'confidence': getattr(mapping, 'confidence', 0.8)
                        })
            
            return field_mappings
            
        except Exception as e:
            self.logger.warning(f"Error extracting field mappings: {e}")
            return []
    
    def _serialize_nlp_result(self, nlp_result: Any) -> Dict[str, Any]:
        """Safely serialize NLP result for context preservation"""
        try:
            if hasattr(nlp_result, 'to_dict'):
                return nlp_result.to_dict()
            elif isinstance(nlp_result, dict):
                return nlp_result
            else:
                # Extract key attributes manually
                result = {}
                for attr in ['structured_query', 'field_mappings', 'business_context', 'confidence_scores']:
                    if hasattr(nlp_result, attr):
                        value = getattr(nlp_result, attr)
                        if hasattr(value, 'to_dict'):
                            result[attr] = value.to_dict()
                        elif isinstance(value, (list, dict, str, int, float, bool)):
                            result[attr] = value
                        else:
                            result[attr] = str(value)
                return result
        except Exception as e:
            self.logger.warning(f"Error serializing NLP result: {e}")
            return {'serialization_error': str(e)}
    
    def _map_intent_to_standard(self, primary_intent: str) -> QueryIntent:
        """Map NLP intent to standardized QueryIntent enum"""
        intent_lower = primary_intent.lower()
        
        # Direct mapping
        if intent_lower in self.intent_mapping:
            return self.intent_mapping[intent_lower]
        
        # Fuzzy mapping
        for key, value in self.intent_mapping.items():
            if key in intent_lower or intent_lower in key:
                return value
        
        # Keyword-based mapping
        if any(keyword in intent_lower for keyword in ['aggregate', 'sum', 'count', 'total', 'group']):
            return QueryIntent.AGGREGATION
        elif any(keyword in intent_lower for keyword in ['time', 'date', 'temporal', 'recent', 'last', 'period']):
            return QueryIntent.TEMPORAL_FILTER
        elif any(keyword in intent_lower for keyword in ['join', 'relationship', 'connect', 'link']):
            return QueryIntent.JOIN_ANALYSIS
        elif any(keyword in intent_lower for keyword in ['complex', 'analysis', 'analyze', 'detailed']):
            return QueryIntent.COMPLEX_ANALYSIS
        elif any(keyword in intent_lower for keyword in ['lookup', 'find', 'get', 'show', 'simple']):
            return QueryIntent.SIMPLE_LOOKUP
        elif any(keyword in intent_lower for keyword in ['report', 'reporting', 'summary']):
            return QueryIntent.REPORTING
        
        return QueryIntent.UNKNOWN
    
    def _determine_max_results(self, intent: QueryIntent, confidence: float) -> int:
        """Determine optimal max_results based on intent and confidence"""
        base_results = {
            QueryIntent.SIMPLE_LOOKUP: 15,
            QueryIntent.AGGREGATION: 40,
            QueryIntent.COMPLEX_ANALYSIS: 60,
            QueryIntent.TEMPORAL_FILTER: 25,
            QueryIntent.JOIN_ANALYSIS: 50,
            QueryIntent.REPORTING: 35,
            QueryIntent.UNKNOWN: 30
        }
        
        base = base_results.get(intent, 30)
        
        # Adjust based on confidence
        if confidence > 0.9:
            return base  # High confidence, use base
        elif confidence > 0.7:
            return min(base + 10, 70)  # Medium confidence, slight increase
        else:
            return min(base + 20, 80)  # Low confidence, more results
    
    def _should_include_joins(self, intent_data: Dict[str, Any], target_tables: List[str]) -> bool:
        """Determine if joins should be included"""
        # Always include joins if multiple tables are targeted
        if len(target_tables) > 1:
            return True
        
        # Check intent indicators
        primary_intent = intent_data.get('primary_intent', '').lower()
        
        # These intents typically require joins
        join_intents = ['aggregation', 'complex_analysis', 'join_analysis', 'reporting']
        if any(intent in primary_intent for intent in join_intents):
            return True
        
        # Check for explicit join requirements
        if intent_data.get('requires_joins', False):
            return True
        
        # Default based on complexity
        complexity = intent_data.get('complexity', 0.5)
        return complexity > 0.6
    
    def _should_include_xml(self, intent_data: Dict[str, Any]) -> bool:
        """Determine if XML processing should be included"""
        # Check for explicit XML requirements
        if intent_data.get('requires_xml') is not None:
            return intent_data['requires_xml']
        
        # Default to True for comprehensive results
        return True
    
    def _determine_min_confidence(self, nlp_confidence: float) -> float:
        """Determine minimum confidence threshold for schema results"""
        # Scale min_confidence inversely with NLP confidence
        if nlp_confidence > 0.9:
            return 0.3  # High NLP confidence, accept lower schema confidence
        elif nlp_confidence > 0.7:
            return 0.2  # Medium NLP confidence
        else:
            return 0.1  # Low NLP confidence, accept very low schema confidence
    
    def _create_fallback_request(self, query: str, request_id: str, error: str) -> SchemaSearchRequest:
        """Create fallback request when conversion fails"""
        return SchemaSearchRequest(
            request_id=request_id,
            query=query,
            enhanced_query=query,
            intent_classification={
                'primary_intent': QueryIntent.UNKNOWN.value,
                'confidence': 0.5,
                'conversion_error': error
            },
            include_xml=True,
            include_joins=True,
            max_results=30,
            processing_hints={
                'fallback_conversion': True,
                'error': error
            }
        )

class SchemaToNLPConverter:
    """
    Enhances Schema Retrieval Agent output with NLP insights
    Based on actual SchemaSearchResponse -> Enhanced Response conversion
    """
    
    def __init__(self):
        self.logger = logger
    
    def enhance_schema_response_with_nlp(
        self,
        schema_response: Dict[str, Any],
        nlp_enhanced_query: NLPEnhancedQuery,
        original_nlp_result: Any
    ) -> SchemaSearchResponse:
        """
        Enhance Schema Retrieval Agent response with NLP insights
        
        Args:
            schema_response: Response from Schema Retrieval Agent
            nlp_enhanced_query: Processed NLP query data
            original_nlp_result: Original NLP processor result
            
        Returns:
            Enhanced SchemaSearchResponse with NLP insights
        """
        try:
            request_id = schema_response.get('request_id', str(uuid.uuid4()))
            self.logger.debug(f"[{request_id}] Enhancing schema response with NLP insights")
            
            # Create base enhanced response
            enhanced_response = SchemaSearchResponse(
                request_id=request_id,
                query=schema_response.get('query', ''),
                status=schema_response.get('status', 'success'),
                data=schema_response.get('data', {}),
                error=schema_response.get('error'),
                execution_time_ms=schema_response.get('execution_time_ms', 0.0),
                metadata=schema_response.get('metadata', {})
            )
            
            # Create comprehensive NLP insights
            nlp_insights = self._create_nlp_insights(
                nlp_enhanced_query, original_nlp_result, enhanced_response.data
            )
            
            # Add NLP insights to response
            enhanced_response.nlp_insights = nlp_insights # type: ignore
            enhanced_response.processing_chain = ['nlp_processor', 'schema_retrieval_agent', 'nlp_enhancement']
            
            # Enhance metadata
            enhanced_response.metadata.update({
                'nlp_processing_applied': True,
                'nlp_enhancement_quality': nlp_enhanced_query.confidence_score,
                'processing_method': 'nlp_enhanced_schema_retrieval',
                'integration_version': '1.0.0'
            })
            
            self.logger.debug(f"[{request_id}] Schema response enhanced successfully")
            return enhanced_response
            
        except Exception as e:
            self.logger.error(f"Schema->NLP enhancement failed: {e}")
            return self._create_basic_response(schema_response, str(e))
    
    def _create_nlp_insights(
        self,
        nlp_enhanced_query: NLPEnhancedQuery,
        original_nlp_result: Any,
        schema_data: Dict[str, Any]
    ) -> NLPInsights:
        """Create comprehensive NLP insights"""
        try:
            # Extract actual tables and columns from schema results
            actual_tables = schema_data.get('tables', [])
            actual_columns_by_table = schema_data.get('columns_by_table', {})
            
            # Calculate prediction accuracy
            prediction_accuracy = self._calculate_prediction_accuracy(
                nlp_enhanced_query.target_tables,
                nlp_enhanced_query.target_columns,
                actual_tables,
                actual_columns_by_table
            )
            
            # Create insights object
            insights = NLPInsights(
                detected_intent={
                    'primary': nlp_enhanced_query.primary_intent.value,
                    'confidence': nlp_enhanced_query.confidence_score,
                    'complexity_score': nlp_enhanced_query.complexity_score
                },
                target_tables_predicted=nlp_enhanced_query.target_tables,
                target_columns_predicted=nlp_enhanced_query.target_columns,
                semantic_entities=nlp_enhanced_query.semantic_entities,
                processing_hints={
                    'requires_joins': nlp_enhanced_query.requires_joins,
                    'requires_xml': nlp_enhanced_query.requires_xml,
                    'temporal_context_detected': nlp_enhanced_query.temporal_context is not None,
                    'original_query': nlp_enhanced_query.original_query,
                    'enhanced_query': nlp_enhanced_query.enhanced_query
                },
                temporal_context=nlp_enhanced_query.temporal_context,
                prediction_accuracy=prediction_accuracy
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error creating NLP insights: {e}")
            return NLPInsights()  # Return empty insights
    
    def _calculate_prediction_accuracy(
        self,
        predicted_tables: List[str],
        predicted_columns: List[str],
        actual_tables: List[str],
        actual_columns_by_table: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Calculate accuracy of NLP predictions against actual schema results"""
        try:
            # Table prediction accuracy
            table_accuracy = self._calculate_list_accuracy(predicted_tables, actual_tables)
            
            # Column prediction accuracy
            actual_columns = []
            for table_columns in actual_columns_by_table.values():
                actual_columns.extend([col.get('column', '') for col in table_columns])
            
            column_accuracy = self._calculate_list_accuracy(predicted_columns, actual_columns)
            
            # Overall prediction quality
            overall_accuracy = (table_accuracy + column_accuracy) / 2
            
            return {
                'table_prediction_accuracy': round(table_accuracy, 3),
                'column_prediction_accuracy': round(column_accuracy, 3),
                'overall_accuracy': round(overall_accuracy, 3),
                'predicted_tables_found': len(set(predicted_tables).intersection(set(actual_tables))),
                'predicted_columns_found': len(set(predicted_columns).intersection(set(actual_columns))),
                'prediction_quality': 'high' if overall_accuracy > 0.7 else 'medium' if overall_accuracy > 0.4 else 'low'
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating prediction accuracy: {e}")
            return {'error': str(e)}
    
    def _calculate_list_accuracy(self, predicted: List[str], actual: List[str]) -> float:
        """Calculate accuracy between predicted and actual lists"""
        if not predicted and not actual:
            return 1.0
        if not predicted or not actual:
            return 0.0
        
        predicted_set = set(p.lower().strip() for p in predicted if p)
        actual_set = set(a.lower().strip() for a in actual if a)
        
        if not predicted_set and not actual_set:
            return 1.0
        if not predicted_set or not actual_set:
            return 0.0
        
        intersection = predicted_set.intersection(actual_set)
        union = predicted_set.union(actual_set)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _create_basic_response(self, schema_response: Dict[str, Any], error: str) -> SchemaSearchResponse:
        """Create basic response when enhancement fails"""
        return SchemaSearchResponse(
            request_id=schema_response.get('request_id', str(uuid.uuid4())),
            query=schema_response.get('query', ''),
            status=schema_response.get('status', 'success'),
            data=schema_response.get('data', {}),
            error=schema_response.get('error'),
            execution_time_ms=schema_response.get('execution_time_ms', 0.0),
            metadata={
                **schema_response.get('metadata', {}),
                'nlp_enhancement_error': error,
                'nlp_processing_applied': False
            }
        )

# Utility Functions
def validate_nlp_to_schema_conversion(
    nlp_result: Any, 
    schema_request: SchemaSearchRequest
) -> Dict[str, Any]:
    """Validate the conversion from NLP to Schema format"""
    validation_result = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    try:
        # Check required fields
        if not schema_request.query:
            validation_result['errors'].append("Query is empty")
            validation_result['valid'] = False
        
        # Check intent mapping
        if schema_request.intent_classification.get('primary_intent') == QueryIntent.UNKNOWN.value:
            validation_result['warnings'].append("Intent could not be determined")
        
        # Check target tables/columns
        if not schema_request.target_tables and not schema_request.target_columns:
            validation_result['warnings'].append("No target tables or columns predicted")
        
        # Check NLP context preservation
        if not schema_request.nlp_context:
            validation_result['warnings'].append("NLP context not preserved")
        
    except Exception as e:
        validation_result['errors'].append(f"Validation error: {str(e)}")
        validation_result['valid'] = False
    
    return validation_result

def create_nlp_enhanced_query_from_result(nlp_result: Any, original_query: str) -> NLPEnhancedQuery:
    """Create NLPEnhancedQuery from NLP processor result"""
    try:
        converter = NLPToSchemaConverter()
        structured_query = converter._extract_structured_query(nlp_result)
        
        intent_data = structured_query.get('intent_classification', {})
        search_scope = structured_query.get('search_scope', {})
        
        return NLPEnhancedQuery(
            original_query=original_query,
            enhanced_query=structured_query.get('refined_query', original_query),
            primary_intent=converter._map_intent_to_standard(intent_data.get('primary_intent', 'unknown')),
            confidence_score=intent_data.get('confidence', 0.8),
            target_tables=search_scope.get('target_tables', []),
            target_columns=search_scope.get('target_columns', []),
            semantic_entities=structured_query.get('semantic_entities', []),
            temporal_context=structured_query.get('temporal_context'),
            requires_joins=intent_data.get('requires_joins', True),
            requires_xml=intent_data.get('requires_xml', True),
            complexity_score=intent_data.get('complexity', 0.5),
            processing_metadata=getattr(nlp_result, 'processing_metadata', {})
        )
    except Exception as e:
        logger.error(f"Error creating NLPEnhancedQuery: {e}")
        return NLPEnhancedQuery(
            original_query=original_query,
            enhanced_query=original_query,
            primary_intent=QueryIntent.UNKNOWN,
            confidence_score=0.5
        )
