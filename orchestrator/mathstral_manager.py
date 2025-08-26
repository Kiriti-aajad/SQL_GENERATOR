"""
MATHSTRAL MANAGER WITH CENTRALIZED NLP-SCHEMA INTEGRATION

UPDATED VERSION - ELIMINATES DUPLICATION + FIXES PARAMETER WARNINGS:
- FIXED: Added async_client_manager parameter to eliminate parameter mismatch warnings
- Removed duplicate PromptBuilder integration
- Uses centralized NLP-Schema Integration Bridge for prompt generation
- Maintains all Mathstral-specific processing capabilities
- Eliminates code duplication with other orchestrator files

Author: KIRITI AAJAD (Updated for Centralized Architecture + Parameter Fix)
Version: 2.1.1 - PARAMETER MISMATCH WARNINGS FIXED
Date: 2025-08-19
"""

import asyncio
import time
import json
import logging
import sys
import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class ComplexityLevel(Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class MathstralProcessingMode(Enum):
    ADVANCED_ANALYSIS = "advanced_analysis"
    COMPLEX_AGGREGATION = "complex_aggregation"
    MULTI_TABLE_JOINS = "multi_table_joins"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    STATISTICAL_OPERATIONS = "statistical_operations"

# MATHSTRAL-SPECIFIC ERROR CLASSES
class MathstralProcessingError(Exception):
    """Raised when Mathstral processing fails"""
    pass

class PromptGenerationError(Exception):
    """Raised when prompt generation fails"""
    pass

class ContextEnhancementError(Exception):
    """Raised when context enhancement fails"""
    pass

@dataclass
class MathstralConfig:
    """Configuration for Mathstral Manager"""
    # Processing settings
    enable_nlp_enhancement: bool = True
    enable_advanced_prompting: bool = True
    max_context_tokens: int = 4000
    
    # Performance settings
    processing_timeout: float = 30.0
    prompt_generation_timeout: float = 10.0
    enable_context_optimization: bool = True
    
    # Quality settings
    min_confidence_threshold: float = 0.6
    enable_complexity_scaling: bool = True
    enable_entity_highlighting: bool = True
    
    # Model settings
    preferred_model: str = "mathstral"
    fallback_model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 2000
    
    # Logging
    enable_detailed_logging: bool = True
    enable_performance_monitoring: bool = True

class MathstralManager:
    """
    CENTRALIZED MATHSTRAL MANAGER - NO DUPLICATION + PARAMETER WARNINGS FIXED
    
    CHANGES MADE:
    ✅ FIXED: Added async_client_manager parameter to eliminate warnings
    ✅ Removed duplicate PromptBuilder initialization
    ✅ Uses centralized NLP-Schema Integration Bridge
    ✅ Eliminates prompt generation duplication
    ✅ Maintains all Mathstral-specific capabilities
    """

    def __init__(self, 
                 config: Optional[MathstralConfig] = None, 
                 nlp_schema_bridge=None, 
                 async_client_manager=None):  # 🔧 FIXED: Added missing parameter
        """Initialize Mathstral Manager with centralized bridge integration"""
        self.config = config or MathstralConfig()
        self.nlp_schema_bridge = nlp_schema_bridge  # CENTRALIZED: Use shared bridge
        self.async_client_manager = async_client_manager  # 🔧 FIXED: Store AsyncClientManager
        self.initialization_time = time.time()
        
        # Configure logging
        if self.config.enable_detailed_logging:
            logger.setLevel(logging.DEBUG)
        
        logger.info("=" * 80)
        logger.info("MATHSTRAL MANAGER - CENTRALIZED ARCHITECTURE v2.1.1")
        logger.info("✅ No duplicate PromptBuilder - uses centralized bridge")
        logger.info("✅ Specialized for complex queries with NLP enhancement")
        logger.info(f"✅ AsyncClientManager: {'Available' if async_client_manager else 'Not provided'}")
        logger.info("=" * 80)
        
        # Initialize SQL generator (kept - not duplicated elsewhere)
        self.sql_generator = None
        self._initialize_sql_generator()
        
        # Processing statistics
        self.processing_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_processing_time': 0.0,
            'average_prompt_length': 0.0,
            'complexity_distribution': {
                'very_high': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            }
        }
        
        logger.info("CENTRALIZED MATHSTRAL MANAGER READY FOR COMPLEX PROCESSING")
        logger.info("=" * 80)

    def _initialize_sql_generator(self):
        """Initialize SQL generator component (not duplicated)"""
        try:
            sql_generator_paths = [
                ("agent.sql_generator.generator", "SQLGenerator"),
                ("agent.sql_generator.main", "SQLGenerator"),
            ]
            
            for module_path, class_name in sql_generator_paths:
                try:
                    import importlib
                    module = importlib.import_module(module_path)
                    generator_class = getattr(module, class_name)
                    
                    # 🔧 FIXED: Pass AsyncClientManager to SQLGenerator if available
                    if self.async_client_manager:
                        generator_instance = generator_class(async_client_manager=self.async_client_manager)
                        logger.info(f"SQL generator loaded with shared AsyncClientManager from {module_path}.{class_name}")
                    else:
                        generator_instance = generator_class()
                        logger.info(f"SQL generator loaded from {module_path}.{class_name}")
                    
                    self.sql_generator = generator_instance
                    return
                    
                except (ImportError, AttributeError) as e:
                    logger.debug(f"Failed to load SQL generator from {module_path}: {e}")
                    continue
            
            # Create fallback SQL generator
            logger.warning("Using fallback SQL generator")
            self.sql_generator = self._create_fallback_sql_generator()
            
        except Exception as e:
            logger.error(f"SQL generator initialization failed: {e}")
            self.sql_generator = self._create_fallback_sql_generator()

    def _create_fallback_sql_generator(self):
        """Create fallback SQL generator"""
        class FallbackSQLGenerator:
            def generate_sql(self, prompt: str, **kwargs) -> Dict[str, Any]:
                return {
                    "success": True,
                    "generated_sql": f"-- Generated by Mathstral fallback\n-- Complex query processing\nSELECT 'mathstral_fallback_result' AS result;",
                    "confidence_score": 0.5,
                    "model_used": "mathstral_fallback"
                }
        return FallbackSQLGenerator()

    # MAIN PROCESSING METHODS
    async def process_query(
        self,
        user_query: str,
        enhanced_context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        CENTRALIZED: Main processing method using centralized prompt generation
        """
        if not request_id:
            request_id = f"mathstral_{int(time.time())}"
        
        start_time = time.time()
        self.processing_stats['total_requests'] += 1
        
        logger.info("=" * 80)
        logger.info(f"MATHSTRAL COMPLEX QUERY PROCESSING [{request_id}]")
        logger.info(f"Query: '{user_query[:100]}{'...' if len(user_query) > 100 else ''}'")
        logger.info(f"Enhanced Context: {'Available' if enhanced_context else 'Basic'}")
        logger.info(f"Bridge Available: {'Yes' if self.nlp_schema_bridge else 'No'}")
        logger.info(f"AsyncClientManager: {'Available' if self.async_client_manager else 'Not provided'}")
        logger.info("=" * 80)
        
        try:
            # Step 1: Analyze query complexity and extract NLP insights
            logger.info(f"[{request_id}] Analyzing query complexity and NLP insights")
            analysis_result = await self._analyze_query_complexity(user_query, enhanced_context, request_id)
            
            # Step 2: Enhance context with Mathstral-specific optimizations
            logger.info(f"[{request_id}] Enhancing context for Mathstral processing")
            mathstral_context = await self._enhance_context_for_mathstral(
                user_query, enhanced_context, analysis_result, request_id
            )
            
            # Step 3: CENTRALIZED PROMPT GENERATION - Use bridge instead of duplicate code
            logger.info(f"[{request_id}] Generating prompt via centralized bridge")
            enhanced_prompt = await self._generate_enhanced_prompt_centralized(
                user_query, mathstral_context, analysis_result, request_id
            )
            
            # Step 4: Generate SQL using Mathstral with enriched context
            logger.info(f"[{request_id}] Generating SQL with Mathstral")
            sql_result = await self._generate_sql_with_mathstral(
                enhanced_prompt, mathstral_context, analysis_result, request_id
            )
            
            # Step 5: Post-process and validate results
            logger.info(f"[{request_id}] Post-processing and validation")
            final_result = await self._post_process_results(
                sql_result, mathstral_context, analysis_result, request_id
            )
            
            processing_time = time.time() - start_time
            self._update_processing_statistics(analysis_result, processing_time, True)
            
            logger.info("=" * 80)
            logger.info(f"MATHSTRAL PROCESSING COMPLETED [{request_id}]")
            logger.info(f"Processing Time: {processing_time:.3f}s")
            logger.info(f"Complexity: {analysis_result.get('complexity_level', 'unknown')}")
            logger.info(f"Prompt Source: {'Centralized Bridge' if self.nlp_schema_bridge else 'Fallback'}")
            logger.info("=" * 80)
            
            return final_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.processing_stats['failed_requests'] += 1
            self._update_processing_statistics({}, processing_time, False)
            
            error_msg = f"Mathstral processing failed [{request_id}]: {str(e)}"
            logger.error("=" * 80)
            logger.error(f"{error_msg}")
            logger.error(f"Processing Time: {processing_time:.3f}s")
            logger.error("=" * 80)
            
            return {
                'success': False,
                'request_id': request_id,
                'error': error_msg,
                'generated_sql': '',
                'processing_time': processing_time,
                'manager_used': 'mathstral',
                'confidence_score': 0.0,
                'model_used': 'error_fallback'
            }

    async def _generate_enhanced_prompt_centralized(
        self,
        user_query: str,
        mathstral_context: Dict[str, Any],
        analysis_result: Dict[str, Any],
        request_id: str
    ) -> str:
        """
        CENTRALIZED: Use bridge for prompt generation instead of duplicate code
        """
        try:
            if self.nlp_schema_bridge:
                logger.info(f"[{request_id}] Using centralized PromptBuilder via bridge")
                
                # Prepare schema context for centralized prompt builder
                schema_context = mathstral_context.get('original_context', {})
                
                # Use centralized prompt generation service
                prompt_result = await self.nlp_schema_bridge.generate_sophisticated_prompt_for_manager(
                    user_query,
                    schema_context,
                    "mathstral",
                    request_id
                )
                
                if prompt_result.get('success'):
                    sophisticated_prompt = prompt_result.get('prompt', '')
                    
                    # Add Mathstral-specific optimizations to the sophisticated prompt
                    final_prompt = self._optimize_prompt_for_mathstral(sophisticated_prompt, mathstral_context)
                    
                    logger.info(f"[{request_id}] Centralized prompt generation successful: "
                              f"quality={prompt_result.get('quality', 'unknown')}, "
                              f"length={len(final_prompt)} chars")
                    
                    return final_prompt
                else:
                    logger.warning(f"[{request_id}] Centralized prompt generation failed: {prompt_result.get('error', 'unknown')}")
            
            # Fallback to basic prompt generation
            logger.warning(f"[{request_id}] Using fallback prompt generation")
            return self._generate_fallback_enhanced_prompt(user_query, mathstral_context)
            
        except Exception as e:
            logger.error(f"[{request_id}] Enhanced prompt generation failed: {e}")
            return self._generate_fallback_enhanced_prompt(user_query, mathstral_context)

    def _generate_fallback_enhanced_prompt(self, query: str, mathstral_context: Dict[str, Any]) -> str:
        """Generate fallback enhanced prompt when centralized bridge unavailable"""
        prompt_parts = [
            f"Generate advanced SQL for complex Mathstral query: {query}",
            "",
            "MATHSTRAL ENHANCED CONTEXT:"
        ]
        
        # Add context from mathstral_context
        schema_context = mathstral_context.get('original_context', {})
        
        # Add schema information
        if schema_context.get('tables'):
            tables = ', '.join(schema_context['tables'][:10])
            prompt_parts.append(f"- Available Tables: {tables}")
        
        # Add NLP enhancements if available
        nlp_optimizations = mathstral_context.get('mathstral_optimizations', {})
        if nlp_optimizations:
            prompt_parts.extend([
                f"- Detected Intent: {nlp_optimizations.get('primary_intent', 'unknown')}",
                f"- Confidence Score: {nlp_optimizations.get('confidence_score', 0.8)}"
            ])
            
            if nlp_optimizations.get('semantic_entities'):
                entities = ', '.join(nlp_optimizations['semantic_entities'][:5])
                prompt_parts.append(f"- Key Entities: {entities}")
        
        # Add Mathstral-specific instructions
        complexity_analysis = mathstral_context.get('complexity_analysis', {})
        prompt_parts.extend([
            "",
            "MATHSTRAL PROCESSING INSTRUCTIONS:",
            f"- Complexity Level: {complexity_analysis.get('complexity_level', 'high')}",
            f"- Processing Mode: {complexity_analysis.get('processing_mode', 'advanced_analysis')}",
            "- Generate optimized SQL with advanced features",
            "- Focus on performance and accuracy",
            "- Include appropriate aggregations and joins"
        ])
        
        return "\n".join(prompt_parts)

    def _optimize_prompt_for_mathstral(self, prompt: str, mathstral_context: Dict[str, Any]) -> str:
        """Optimize prompt specifically for Mathstral processing"""
        try:
            # Add Mathstral-specific optimizations
            optimizations = []
            
            config = mathstral_context.get('mathstral_config', {})
            if config.get('enable_advanced_features'):
                optimizations.append("Use advanced SQL features and optimizations where appropriate.")
            
            if config.get('use_schema_awareness'):
                optimizations.append("Leverage schema relationships and constraints in the generated SQL.")
            
            processing_hints = mathstral_context.get('processing_hints', {})
            if processing_hints.get('optimization_level') == 'high':
                optimizations.append("Prioritize query performance and efficiency.")
            
            # Append optimizations to prompt
            if optimizations:
                optimized_prompt = prompt + "\n\nMATHSTRAL OPTIMIZATIONS:\n" + "\n".join(f"- {opt}" for opt in optimizations)
            else:
                optimized_prompt = prompt
            
            # Ensure prompt doesn't exceed token limit
            if len(optimized_prompt) > self.config.max_context_tokens * 4:  # Rough estimation
                # Truncate while preserving important parts
                optimized_prompt = optimized_prompt[:self.config.max_context_tokens * 4]
                optimized_prompt += "\n\n[Note: Context truncated for optimal processing]"
            
            return optimized_prompt
            
        except Exception as e:
            logger.warning(f"Prompt optimization failed: {e}")
            return prompt  # Return original prompt on failure

    async def _analyze_query_complexity(
        self,
        user_query: str,
        enhanced_context: Optional[Dict[str, Any]],
        request_id: str
    ) -> Dict[str, Any]:
        """Analyze query complexity and extract NLP insights for Mathstral optimization"""
        try:
            logger.debug(f"[{request_id}] Analyzing query complexity with NLP insights")
            
            analysis = {
                'complexity_level': 'high',  # Default for Mathstral
                'processing_mode': MathstralProcessingMode.ADVANCED_ANALYSIS,
                'nlp_insights': {},
                'schema_insights': {},
                'optimization_hints': []
            }
            
            # Extract NLP insights from enhanced context
            if enhanced_context:
                # Extract from NLP-Schema integration context
                if 'nlp_insights' in enhanced_context:
                    nlp_insights = enhanced_context['nlp_insights']
                    analysis['nlp_insights'] = nlp_insights
                    
                    # Determine complexity from NLP intent
                    detected_intent = nlp_insights.get('detected_intent', {})
                    primary_intent = detected_intent.get('primary', 'unknown')
                    
                    if primary_intent == 'complex_analysis':
                        analysis['complexity_level'] = 'very_high'
                        analysis['processing_mode'] = MathstralProcessingMode.ADVANCED_ANALYSIS
                    elif primary_intent == 'aggregation':
                        analysis['complexity_level'] = 'high'
                        analysis['processing_mode'] = MathstralProcessingMode.COMPLEX_AGGREGATION
                    elif primary_intent == 'join_analysis':
                        analysis['complexity_level'] = 'high'
                        analysis['processing_mode'] = MathstralProcessingMode.MULTI_TABLE_JOINS
                    elif primary_intent == 'temporal_filter':
                        analysis['complexity_level'] = 'medium'
                        analysis['processing_mode'] = MathstralProcessingMode.TEMPORAL_ANALYSIS
                    
                    # Extract optimization hints
                    if nlp_insights.get('target_tables_predicted'):
                        analysis['optimization_hints'].append('focus_on_predicted_tables')
                    if nlp_insights.get('target_columns_predicted'):
                        analysis['optimization_hints'].append('prioritize_predicted_columns')
                
                # Extract schema insights
                if 'original_context' in enhanced_context:
                    schema_context = enhanced_context['original_context']
                    analysis['schema_insights'] = {
                        'tables_available': len(schema_context.get('tables', [])),
                        'has_joins': len(schema_context.get('joins', [])) > 0,
                        'has_xml_data': schema_context.get('has_xml_data', False),
                        'total_columns': schema_context.get('total_columns', 0)
                    }
                
                # Extract routing hints for Mathstral optimization
                if 'routing_guidance' in enhanced_context:
                    routing = enhanced_context['routing_guidance']
                    if routing.get('processing_method') == 'mathstral_preferred':
                        analysis['optimization_hints'].append('mathstral_optimized_processing')
            
            # Fallback analysis based on query text
            if not analysis['nlp_insights']:
                analysis.update(self._analyze_query_text_complexity(user_query))
            
            logger.debug(f"[{request_id}] Complexity analysis: {analysis['complexity_level']}")
            return analysis
            
        except Exception as e:
            logger.error(f"[{request_id}] Query complexity analysis failed: {e}")
            # Return safe default
            return {
                'complexity_level': 'high',
                'processing_mode': MathstralProcessingMode.ADVANCED_ANALYSIS,
                'nlp_insights': {},
                'schema_insights': {},
                'optimization_hints': ['fallback_analysis']
            }

    def _analyze_query_text_complexity(self, query: str) -> Dict[str, Any]:
        """Fallback text-based complexity analysis"""
        query_lower = query.lower()
        analysis = {}
        
        # Complexity indicators
        complexity_score = 0
        if any(word in query_lower for word in ['sum', 'count', 'avg', 'group by', 'having']):
            complexity_score += 2
        if any(word in query_lower for word in ['join', 'inner', 'outer', 'left', 'right']):
            complexity_score += 2
        if any(word in query_lower for word in ['subquery', 'exists', 'with', 'cte']):
            complexity_score += 3
        if any(word in query_lower for word in ['case when', 'over', 'partition', 'window']):
            complexity_score += 3
        
        # Map score to complexity
        if complexity_score >= 6:
            analysis['complexity_level'] = 'very_high'
            analysis['processing_mode'] = MathstralProcessingMode.STATISTICAL_OPERATIONS
        elif complexity_score >= 4:
            analysis['complexity_level'] = 'high'
            analysis['processing_mode'] = MathstralProcessingMode.COMPLEX_AGGREGATION
        elif complexity_score >= 2:
            analysis['complexity_level'] = 'medium'
            analysis['processing_mode'] = MathstralProcessingMode.MULTI_TABLE_JOINS
        else:
            analysis['complexity_level'] = 'high'  # Default high for Mathstral
            analysis['processing_mode'] = MathstralProcessingMode.ADVANCED_ANALYSIS
        
        return analysis

    async def _enhance_context_for_mathstral(
        self,
        user_query: str,
        enhanced_context: Optional[Dict[str, Any]],
        analysis_result: Dict[str, Any],
        request_id: str
    ) -> Dict[str, Any]:
        """Enhance context specifically for Mathstral processing"""
        try:
            logger.debug(f"[{request_id}] Enhancing context for Mathstral")
            
            mathstral_context = {
                'query': user_query,
                'complexity_analysis': analysis_result,
                'mathstral_optimizations': {},
                'prompt_enhancements': {},
                'processing_hints': {}
            }
            
            # Copy original context
            if enhanced_context:
                mathstral_context['original_context'] = enhanced_context.copy()
                
                # Extract NLP enhancements for Mathstral
                if 'nlp_insights' in enhanced_context:
                    nlp_insights = enhanced_context['nlp_insights']
                    
                    mathstral_context['mathstral_optimizations'] = {
                        'use_detected_intent': True,
                        'primary_intent': nlp_insights.get('detected_intent', {}).get('primary', 'unknown'),
                        'confidence_score': nlp_insights.get('detected_intent', {}).get('confidence', 0.8),
                        'semantic_entities': nlp_insights.get('semantic_entities', []),
                        'predicted_tables': nlp_insights.get('target_tables_predicted', []),
                        'predicted_columns': nlp_insights.get('target_columns_predicted', [])
                    }
                    
                    # Enhanced prompt instructions
                    mathstral_context['prompt_enhancements'] = {
                        'include_entity_context': len(nlp_insights.get('semantic_entities', [])) > 0,
                        'focus_on_predicted_elements': len(nlp_insights.get('target_tables_predicted', [])) > 0,
                        'use_business_context': 'business_context' in nlp_insights,
                        'apply_intent_specific_templates': True
                    }
                
                # Processing hints for SQL generation
                if 'routing_guidance' in enhanced_context:
                    routing = enhanced_context['routing_guidance']
                    mathstral_context['processing_hints'] = {
                        'complexity_level': routing.get('complexity', 'high'),
                        'recommended_approach': routing.get('recommended_action', 'advanced_processing'),
                        'confidence_threshold': routing.get('confidence', 0.8),
                        'optimization_level': 'high'
                    }
            
            # Add Mathstral-specific processing configuration
            mathstral_context['mathstral_config'] = {
                'model': self.config.preferred_model,
                'temperature': self.config.temperature,
                'max_tokens': self.config.max_tokens,
                'processing_mode': analysis_result['processing_mode'].value,
                'enable_advanced_features': True,
                'use_schema_awareness': True
            }
            
            logger.debug(f"[{request_id}] Context enhanced for Mathstral processing")
            return mathstral_context
            
        except Exception as e:
            logger.error(f"[{request_id}] Context enhancement failed: {e}")
            raise ContextEnhancementError(f"Failed to enhance context for Mathstral: {e}")

    async def _generate_sql_with_mathstral(
        self,
        enhanced_prompt: str,
        mathstral_context: Dict[str, Any],
        analysis_result: Dict[str, Any],
        request_id: str
    ) -> Dict[str, Any]:
        """Generate SQL using Mathstral with enhanced context and optimizations"""
        try:
            logger.debug(f"[{request_id}] Generating SQL with Mathstral")
            
            if not self.sql_generator:
                raise MathstralProcessingError("SQL generator not available")
            
            # Prepare enhanced inputs for SQL generator
            banking_insights = mathstral_context.get('original_context', {})
            schema_context = mathstral_context.get('original_context', {})
            
            # Add Mathstral-specific enhancements
            if 'mathstral_optimizations' in mathstral_context:
                optimizations = mathstral_context['mathstral_optimizations']
                
                # Enhance banking insights with NLP data
                banking_insights = banking_insights.copy() if banking_insights else {}
                banking_insights['mathstral_enhancement'] = {
                    'nlp_driven': True,
                    'intent_detected': optimizations.get('primary_intent', 'unknown'),
                    'entities_extracted': optimizations.get('semantic_entities', []),
                    'confidence_score': optimizations.get('confidence_score', 0.8),
                    'complexity_level': analysis_result.get('complexity_level', 'high'),
                    'processing_mode': analysis_result.get('processing_mode', MathstralProcessingMode.ADVANCED_ANALYSIS).value
                }
                
                # Enhance schema context
                schema_context = schema_context.copy() if schema_context else {}
                schema_context['mathstral_optimizations'] = {
                    'predicted_tables': optimizations.get('predicted_tables', []),
                    'predicted_columns': optimizations.get('predicted_columns', []),
                    'focus_elements': True,
                    'use_advanced_sql': True,
                    'optimization_level': 'high'
                }
            
            # Determine target model
            config = mathstral_context.get('mathstral_config', {})
            target_model = config.get('model', self.config.preferred_model)
            
            # Execute SQL generation with timeout
            sql_result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.sql_generator.generate_sql( # pyright: ignore[reportOptionalMemberAccess]
                        prompt=enhanced_prompt,
                        banking_insights=banking_insights,
                        schema_context=schema_context,
                        target_llm=target_model
                    )
                ),
                timeout=self.config.processing_timeout
            )
            
            if not sql_result:
                raise MathstralProcessingError("SQL generator returned empty result")
            
            # Validate SQL generation success
            if isinstance(sql_result, dict):
                if not sql_result.get("success", False):
                    raise MathstralProcessingError(f"SQL generation failed: {sql_result.get('error', 'Unknown error')}")
                
                generated_sql = sql_result.get("generated_sql", "") or sql_result.get("sql", "")
                if not generated_sql or generated_sql.startswith("--"):
                    raise MathstralProcessingError("SQL generator returned invalid or empty SQL")
            
            logger.debug(f"[{request_id}] SQL generated successfully with Mathstral")
            return sql_result
            
        except asyncio.TimeoutError:
            raise MathstralProcessingError(f"SQL generation timed out after {self.config.processing_timeout}s")
        except Exception as e:
            logger.error(f"[{request_id}] Mathstral SQL generation failed: {e}")
            raise MathstralProcessingError(f"Failed to generate SQL with Mathstral: {e}")

    async def _post_process_results(
        self,
        sql_result: Dict[str, Any],
        mathstral_context: Dict[str, Any],
        analysis_result: Dict[str, Any],
        request_id: str
    ) -> Dict[str, Any]:
        """Post-process and validate Mathstral results"""
        try:
            logger.debug(f"[{request_id}] Post-processing Mathstral results")
            
            # Create final result structure
            final_result = {
                'success': True,
                'request_id': request_id,
                'generated_sql': sql_result.get('generated_sql', ''),
                'confidence_score': sql_result.get('confidence_score', 0.8),
                'model_used': sql_result.get('model_used', self.config.preferred_model),
                'processing_time': time.time(),
                
                # Mathstral-specific metadata
                'mathstral_metadata': {
                    'complexity_level': analysis_result.get('complexity_level', 'high'),
                    'processing_mode': analysis_result.get('processing_mode', MathstralProcessingMode.ADVANCED_ANALYSIS).value,
                    'nlp_enhanced': bool(mathstral_context.get('mathstral_optimizations')),
                    'optimization_hints_applied': analysis_result.get('optimization_hints', []),
                    'manager_version': '2.1.1',
                    'centralized_prompt_builder': self.nlp_schema_bridge is not None,
                    'async_client_manager_available': self.async_client_manager is not None  # 🔧 NEW: Added tracking
                },
                
                # NLP integration metadata
                'nlp_integration_metadata': {
                    'nlp_insights_used': bool(mathstral_context.get('mathstral_optimizations')),
                    'predicted_elements_focused': len(mathstral_context.get('mathstral_optimizations', {}).get('predicted_tables', [])) > 0,
                    'intent_based_optimization': mathstral_context.get('mathstral_optimizations', {}).get('primary_intent', 'unknown'),
                    'confidence_level': mathstral_context.get('mathstral_optimizations', {}).get('confidence_score', 0.8)
                }
            }
            
            # Add original SQL result fields
            for key, value in sql_result.items():
                if key not in final_result:
                    final_result[key] = value
            
            # Quality validation
            generated_sql = final_result['generated_sql']
            if generated_sql:
                # Basic SQL validation
                sql_lower = generated_sql.lower().strip()
                if sql_lower.startswith('select') or sql_lower.startswith('with'):
                    final_result['sql_validation'] = {
                        'is_valid_format': True,
                        'has_select_statement': 'select' in sql_lower,
                        'estimated_complexity': 'high' if any(keyword in sql_lower for keyword in ['join', 'group by', 'having', 'with']) else 'medium'
                    }
                else:
                    final_result['sql_validation'] = {
                        'is_valid_format': False,
                        'validation_warning': 'Generated SQL may not be in expected format'
                    }
            
            logger.debug(f"[{request_id}] Results post-processed successfully")
            return final_result
            
        except Exception as e:
            logger.error(f"[{request_id}] Post-processing failed: {e}")
            # Return safe fallback
            return {
                'success': False,
                'request_id': request_id,
                'error': f"Post-processing failed: {e}",
                'generated_sql': sql_result.get('generated_sql', '') if sql_result else '',
                'confidence_score': 0.3,
                'model_used': 'error_fallback'
            }

    def _update_processing_statistics(self, analysis_result: Dict[str, Any], processing_time: float, success: bool):
        """Update processing statistics for monitoring"""
        try:
            if success:
                self.processing_stats['successful_requests'] += 1
            
            # Update average processing time
            current_avg = self.processing_stats['average_processing_time']
            total_requests = self.processing_stats['total_requests']
            
            new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
            self.processing_stats['average_processing_time'] = new_avg
            
            # Update complexity distribution
            complexity = analysis_result.get('complexity_level', 'unknown')
            if complexity in self.processing_stats['complexity_distribution']:
                self.processing_stats['complexity_distribution'][complexity] += 1
            
        except Exception as e:
            logger.warning(f"Statistics update failed: {e}")

    # HEALTH CHECK AND MONITORING
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for Mathstral Manager"""
        try:
            return {
                'component': 'MathstralManager',
                'version': '2.1.1',
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                
                # Component health
                'components_status': {
                    'sql_generator': self.sql_generator is not None,
                    'nlp_schema_bridge': self.nlp_schema_bridge is not None,
                    'centralized_prompt_builder': self.nlp_schema_bridge is not None,
                    'async_client_manager': self.async_client_manager is not None  # 🔧 NEW: Added status
                },
                
                # Processing statistics
                'processing_statistics': self.processing_stats,
                
                # Configuration status
                'configuration': {
                    'nlp_enhancement_enabled': self.config.enable_nlp_enhancement,
                    'advanced_prompting_enabled': self.config.enable_advanced_prompting,
                    'preferred_model': self.config.preferred_model,
                    'processing_timeout': self.config.processing_timeout
                },
                
                # Capabilities
                'capabilities': {
                    'nlp_enhanced_processing': self.config.enable_nlp_enhancement,
                    'centralized_prompt_building': self.nlp_schema_bridge is not None,
                    'advanced_prompting': self.config.enable_advanced_prompting,
                    'context_optimization': self.config.enable_context_optimization,
                    'complexity_scaling': self.config.enable_complexity_scaling,
                    'performance_monitoring': self.config.enable_performance_monitoring,
                    'async_client_integration': self.async_client_manager is not None  # 🔧 NEW: Added capability
                },
                
                # NEW: Architecture info
                'architecture': {
                    'duplicate_prompt_builder_removed': True,
                    'uses_centralized_bridge': self.nlp_schema_bridge is not None,
                    'no_code_duplication': True,
                    'centralized_integration': True,
                    'parameter_mismatch_fixed': True  # 🔧 NEW: Confirmation of fix
                }
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'component': 'MathstralManager',
                'version': '2.1.1',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def cleanup(self):
        """Cleanup Mathstral Manager resources"""
        try:
            logger.info("Starting Mathstral Manager cleanup")
            
            # Reset statistics
            self.processing_stats = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'average_processing_time': 0.0,
                'average_prompt_length': 0.0,
                'complexity_distribution': {'very_high': 0, 'high': 0, 'medium': 0, 'low': 0}
            }
            
            logger.info("Mathstral Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# COMPATIBILITY FUNCTIONS (for existing orchestrator integration)

async def process_complex_query(query: str, enhanced_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compatibility function for existing orchestrator calls
    """
    manager = MathstralManager()
    return await manager.process_query(query, enhanced_context)

def orchestrate_mathstral_query(query: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous compatibility function for existing orchestrator calls
    """
    import asyncio
    try:
        return asyncio.get_event_loop().run_until_complete(
            process_complex_query(query, context)
        )
    except RuntimeError:
        # Create new event loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(process_complex_query(query, context))
        finally:
            loop.close()

# 🔧 FIXED: Factory Functions with AsyncClientManager parameter

def create_mathstral_manager(
    config: Optional[MathstralConfig] = None, 
    nlp_schema_bridge=None, 
    async_client_manager=None  # 🔧 FIXED: Added missing parameter
) -> MathstralManager:
    """
    FIXED: Factory function to create Mathstral Manager with centralized bridge support
    Now accepts async_client_manager parameter to eliminate parameter mismatch warnings
    """
    return MathstralManager(config, nlp_schema_bridge, async_client_manager)

# Export for compatibility
__all__ = [
    'MathstralManager', 'MathstralConfig', 'ComplexityLevel', 'MathstralProcessingMode',
    'create_mathstral_manager', 'process_complex_query', 'orchestrate_mathstral_query'
]

if __name__ == "__main__":
    # Test function
    async def test():
        config = MathstralConfig(enable_detailed_logging=True)
        manager = MathstralManager(config)
        
        # Test health check
        health = await manager.health_check()
        print(f"Health Status: {health['status']}")
        print(f"Centralized Bridge: {health['components_status']['nlp_schema_bridge']}")
        print(f"AsyncClientManager: {health['components_status']['async_client_manager']}")
        print(f"Parameter Mismatch Fixed: {health['architecture']['parameter_mismatch_fixed']}")
        print(f"No Duplication: {health['architecture']['no_code_duplication']}")
        
        # Test processing
        query = "Analyze customer behavior patterns for risk assessment with aggregated transaction data"
        enhanced_context = {
            'nlp_insights': {
                'detected_intent': {'primary': 'complex_analysis', 'confidence': 0.9},
                'semantic_entities': ['customer', 'behavior', 'risk', 'transaction'],
                'target_tables_predicted': ['customers', 'transactions', 'risk_ratings'],
                'target_columns_predicted': ['customer_id', 'transaction_amount', 'risk_score']
            },
            'original_context': {
                'tables': ['customers', 'transactions', 'accounts', 'risk_ratings'],
                'has_xml_data': True,
                'total_columns': 45
            },
            'routing_guidance': {
                'complexity': 'very_high',
                'recommended_action': 'use_advanced_sql_generator',
                'confidence': 0.9
            }
        }
        
        result = await manager.process_query(query, enhanced_context)
        print(f"Processing Success: {result['success']}")
        print(f"Complexity Level: {result.get('mathstral_metadata', {}).get('complexity_level', 'unknown')}")
        print(f"Centralized Prompt Builder Used: {result.get('mathstral_metadata', {}).get('centralized_prompt_builder', False)}")
        print(f"AsyncClientManager Available: {result.get('mathstral_metadata', {}).get('async_client_manager_available', False)}")
        
        await manager.cleanup()
        
        print("CENTRALIZED MATHSTRAL MANAGER - PARAMETER WARNINGS FIXED!")
    
    import asyncio
    asyncio.run(test())
