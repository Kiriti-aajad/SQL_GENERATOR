"""
Model Configuration Management for Multi-Model SQL Generator - CORRECTED VERSION
FIXED: Model name consistency with simplified clients
FIXED: Proper NGROK endpoint configuration  
FIXED: Simplified configuration without BaseModelClient dependencies
FIXED: Complete helper functions and validation
"""

import os
import requests
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

class ModelType(Enum):
    """Supported model types - FIXED to match client expectations"""
    MATHSTRAL = "mathstral-7b-v0.1"  # FIXED: Matches MathstralClient
    DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # ✅ Correct

class QueryComplexity(Enum):
    """Query complexity levels for routing"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ANALYTICAL = "analytical"

class GenerationStrategy(Enum):
    """SQL generation strategies"""
    SINGLE_MODEL = "single_model"
    ENSEMBLE = "ensemble"
    ADAPTIVE = "adaptive"

@dataclass
class ModelEndpoint:
    """Model endpoint configuration for NGROK setup"""
    name: str
    base_url: str
    api_key: Optional[str] = None
    model_id: str = ""
    max_tokens: int = 1000
    temperature: float = 0.1
    timeout: int = 30
    retry_attempts: int = 3
    headers: Dict[str, str] = field(default_factory=dict)
    is_ngrok: bool = True

@dataclass
class ModelCapabilities:
    """Model capability specifications"""
    supports_function_calling: bool = False
    supports_reasoning_mode: bool = False
    supports_thinking_mode: bool = False
    max_context_length: int = 4096
    specializations: List[str] = field(default_factory=list)
    performance_score: float = 0.0

@dataclass
class RoutingRules:
    """Model routing configuration"""
    primary_models: List[ModelType] = field(default_factory=list)
    fallback_models: List[ModelType] = field(default_factory=list)
    confidence_threshold: float = 0.7
    generation_strategy: GenerationStrategy = GenerationStrategy.SINGLE_MODEL
    query_patterns: List[str] = field(default_factory=list)

@dataclass
class EnsembleConfig:
    """Ensemble generation configuration"""
    enabled: bool = True
    use_ensemble_for: List[str] = field(default_factory=list)
    single_model_for: List[str] = field(default_factory=list)
    combination_strategy: str = "confidence_weighted"
    consensus_threshold: float = 0.6

@dataclass
class CorrectionConfig:
    """SQL correction configuration"""
    enabled: bool = True
    max_correction_attempts: int = 3
    syntax_correction: bool = True
    semantic_correction: bool = False
    auto_fix_common_errors: bool = True

class ModelConfigs:
    """Central model configuration management for NGROK setup - CORRECTED VERSION"""
    
    def __init__(self):
        self.models: Dict[ModelType, ModelEndpoint] = {}
        self.capabilities: Dict[ModelType, ModelCapabilities] = {}
        self.routing: Dict[QueryComplexity, RoutingRules] = {}
        self.ensemble_config = EnsembleConfig()
        self.correction_config = CorrectionConfig()
        self._initialize_configurations()
    
    def _initialize_configurations(self):
        """Initialize all model configurations"""
        self._setup_mathstral_config()
        self._setup_deepseek_config()
        self._setup_model_capabilities()
        self._setup_routing_rules()
        self._setup_ensemble_config()
        self._setup_correction_config()
    
    def _setup_mathstral_config(self):
        """FIXED: Configure Mathstral with correct model name and endpoint"""
        self.models[ModelType.MATHSTRAL] = ModelEndpoint(
            name="Mathstral-7B-v0.1",
            base_url=os.getenv("MATHSTRAL_BASE_URL", "https://7ae8c3e93d6f.ngrok-free.app/v1/chat/completions"),  # FIXED: Full endpoint
            api_key=None,
            model_id="mathstral-7b-v0.1",  # FIXED: Matches client expectation
            max_tokens=1200,
            temperature=0.1,  # FIXED: Conservative for consistency
            timeout=30,  # FIXED: Reduced timeout
            retry_attempts=3,
            headers={
                "Content-Type": "application/json"
            },
            is_ngrok=True
        )
    
    def _setup_deepseek_config(self):
        """FIXED: Configure DeepSeek with proper endpoint"""
        self.models[ModelType.DEEPSEEK] = ModelEndpoint(
            name="DeepSeek-R1-Distill-Qwen-7B",
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://7ae8c3e93d6f.ngrok-free.app/v1/chat/completions"),  # FIXED: Full endpoint
            api_key=None,
            model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  # ✅ Correct
            max_tokens=1500,
            temperature=0.6,  # FIXED: DeepSeek optimal range
            timeout=30,
            retry_attempts=3,
            headers={
                "Content-Type": "application/json"
            },
            is_ngrok=True
        )
    
    def _setup_model_capabilities(self):
        """Define model-specific capabilities - SIMPLIFIED"""
        self.capabilities[ModelType.MATHSTRAL] = ModelCapabilities(
            supports_function_calling=False,  # FIXED: Simplified
            supports_reasoning_mode=False,    # FIXED: Simplified
            supports_thinking_mode=False,
            max_context_length=4096,          # FIXED: More realistic
            specializations=[
                "mathematical_analysis",
                "statistical_operations", 
                "analytical_queries",
                "financial_calculations",
                "scientific_calculations"
            ],
            performance_score=8.5
        )
        
        self.capabilities[ModelType.DEEPSEEK] = ModelCapabilities(
            supports_function_calling=False,  # FIXED: Simplified
            supports_reasoning_mode=False,    # FIXED: Simplified
            supports_thinking_mode=False,
            max_context_length=4096,          # FIXED: More realistic
            specializations=[
                "code_generation",
                "general_sql",
                "crud_operations",
                "xml_operations",             # FIXED: Added XML specialization
                "query_optimization",
                "multi_table_joins"
            ],
            performance_score=9.0
        )
    
    def _setup_routing_rules(self):
        """FIXED: Simplified routing rules"""
        self.routing[QueryComplexity.SIMPLE] = RoutingRules(
            primary_models=[ModelType.DEEPSEEK],
            fallback_models=[ModelType.MATHSTRAL],
            confidence_threshold=0.6,
            generation_strategy=GenerationStrategy.SINGLE_MODEL,
            query_patterns=[
                "SELECT * FROM",
                "INSERT INTO",
                "UPDATE.*SET",
                "DELETE FROM",
                "simple joins",
                "basic filtering"
            ]
        )
        
        self.routing[QueryComplexity.MODERATE] = RoutingRules(
            primary_models=[ModelType.DEEPSEEK],
            fallback_models=[ModelType.MATHSTRAL],
            confidence_threshold=0.7,
            generation_strategy=GenerationStrategy.SINGLE_MODEL,
            query_patterns=[
                "multi-table joins",
                "subqueries",
                "GROUP BY",
                "ORDER BY",
                "basic aggregations",
                "xml operations"
            ]
        )
        
        self.routing[QueryComplexity.COMPLEX] = RoutingRules(
            primary_models=[ModelType.DEEPSEEK, ModelType.MATHSTRAL],
            fallback_models=[ModelType.DEEPSEEK],
            confidence_threshold=0.8,
            generation_strategy=GenerationStrategy.ENSEMBLE,
            query_patterns=[
                "window functions",
                "complex CTEs",
                "recursive queries",
                "advanced joins",
                "performance optimization"
            ]
        )
        
        self.routing[QueryComplexity.ANALYTICAL] = RoutingRules(
            primary_models=[ModelType.MATHSTRAL, ModelType.DEEPSEEK],
            fallback_models=[ModelType.DEEPSEEK],
            confidence_threshold=0.8,
            generation_strategy=GenerationStrategy.ENSEMBLE,
            query_patterns=[
                "statistical functions",
                "mathematical calculations",
                "correlation analysis",
                "trend analysis",
                "financial calculations",
                "percentile operations",
                "variance calculations"
            ]
        )
    
    def _setup_ensemble_config(self):
        """Configure ensemble generation settings"""
        self.ensemble_config = EnsembleConfig(
            enabled=True,
            use_ensemble_for=["complex", "analytical", "high_stakes"],
            single_model_for=["simple", "crud", "basic_select"],
            combination_strategy="confidence_weighted",
            consensus_threshold=0.6
        )
    
    def _setup_correction_config(self):
        """Configure SQL correction settings"""
        self.correction_config = CorrectionConfig(
            enabled=True,
            max_correction_attempts=3,
            syntax_correction=True,
            semantic_correction=False,
            auto_fix_common_errors=True
        )
    
    def get_model_config(self, model_type: ModelType) -> ModelEndpoint:
        """Get configuration for specific model"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type.value} not configured")
        return self.models[model_type]
    
    def get_model_capabilities(self, model_type: ModelType) -> ModelCapabilities:
        """Get capabilities for specific model"""
        if model_type not in self.capabilities:
            raise ValueError(f"Capabilities for {model_type.value} not defined")
        return self.capabilities[model_type]
    
    def get_routing_rules(self, complexity: QueryComplexity) -> RoutingRules:
        """Get routing rules for query complexity"""
        return self.routing.get(complexity, self.routing[QueryComplexity.MODERATE])
    
    def get_available_models(self) -> List[ModelType]:
        """Get list of all available models"""
        return list(self.models.keys())
    
    def should_use_ensemble(self, complexity: QueryComplexity) -> bool:
        """Determine if ensemble should be used for given complexity"""
        routing_rules = self.get_routing_rules(complexity)
        return (self.ensemble_config.enabled and 
                routing_rules.generation_strategy == GenerationStrategy.ENSEMBLE)
    
    def validate_configuration(self) -> Dict[str, Any]:
        """SIMPLIFIED: Basic configuration validation"""
        validation_results = {
            "valid": True,
            "issues": [],
            "model_status": {}
        }
        
        for model_type, config in self.models.items():
            model_issues = []
            
            if not config.base_url:
                model_issues.append(f"Missing base URL for {model_type.value}")
            elif not config.base_url.startswith(('http://', 'https://')):
                model_issues.append(f"Invalid URL format for {model_type.value}")
            
            if not config.model_id:
                model_issues.append(f"Missing model ID for {model_type.value}")
            
            validation_results["model_status"][model_type.value] = {
                "configured": len(model_issues) == 0,
                "issues": model_issues,
                "base_url": config.base_url,
                "model_id": config.model_id
            }
            
            if model_issues:
                validation_results["valid"] = False
                validation_results["issues"].extend(model_issues)
        
        return validation_results
    
    def get_generation_strategy(self, query_text: str) -> Tuple[GenerationStrategy, List[ModelType]]:
        """Get generation strategy and models for a query"""
        complexity = self._analyze_query_complexity(query_text)
        routing_rules = self.get_routing_rules(complexity)
        return routing_rules.generation_strategy, routing_rules.primary_models
    
    def _analyze_query_complexity(self, query_text: str) -> QueryComplexity:
        """SIMPLIFIED: Analyze query complexity"""
        query_lower = query_text.lower()
        
        # FIXED: Simplified patterns that work reliably
        analytical_patterns = [
            'correlation', 'statistical', 'variance', 'stddev', 'percentile', 
            'regression', 'trend', 'calculate', 'mathematical', 'financial',
            'scientific', 'analytical', 'mean', 'median', 'deviation'
        ]
        
        complex_patterns = [
            'window', 'cte', 'recursive', 'pivot', 'unpivot', 'case when', 
            'exists', 'complex', 'optimization', 'performance'
        ]
        
        moderate_patterns = [
            'join', 'group by', 'order by', 'having', 'subquery', 'union',
            'xml', '.value(', '.query(', '.exist(', '.nodes('
        ]
        
        simple_patterns = [
            'select * from', 'insert into', 'update', 'delete from',
            'simple', 'basic', 'show', 'list', 'display'
        ]
        
        if any(pattern in query_lower for pattern in analytical_patterns):
            return QueryComplexity.ANALYTICAL
        elif any(pattern in query_lower for pattern in complex_patterns):
            return QueryComplexity.COMPLEX
        elif any(pattern in query_lower for pattern in moderate_patterns):
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE

class NGROKConfigHelper:
    """Helper class for NGROK configuration management"""
    
    @staticmethod
    def create_env_file():
        """Create a sample .env file for NGROK configuration"""
        env_content = """# NGROK Configuration for Multi-Model SQL Generator
# Update these URLs with your actual NGROK tunnel URLs

# Mathstral Model NGROK URL (full endpoint)
MATHSTRAL_BASE_URL=https://7ae8c3e93d6f.ngrok-free.app/v1/chat/completions

# DeepSeek Model NGROK URL (full endpoint)  
DEEPSEEK_BASE_URL=https://7ae8c3e93d6f.ngrok-free.app/v1/chat/completions

# Optional: Model-specific timeouts
# SQL_GEN_TIMEOUT=30
# SQL_GEN_CACHE_SIZE=1000
# SQL_GEN_CACHE_TTL=3600
"""
        
        with open('.env.example', 'w') as f:
            f.write(env_content)
        
        print("✅ Created .env.example file")
        print("📝 Copy this to .env and update with your NGROK URLs")
    
    @staticmethod
    def validate_ngrok_urls():
        """Validate NGROK URL format"""
        urls = {
            "MATHSTRAL_BASE_URL": os.getenv("MATHSTRAL_BASE_URL"),
            "DEEPSEEK_BASE_URL": os.getenv("DEEPSEEK_BASE_URL")
        }
        
        issues = []
        for name, url in urls.items():
            if not url:
                issues.append(f"❌ Missing {name} environment variable")
            elif not url.startswith(('http://', 'https://')):
                issues.append(f"❌ {name} must start with http:// or https://")
            elif '/v1/chat/completions' not in url:
                issues.append(f"⚠️ {name} should include full endpoint path")
        
        return issues

class OrchestratorIntegration:
    """Integration bridge for the orchestrator"""
    
    def __init__(self):
        self.model_configs = model_configs
    
    def get_request_config_for_model(self, model_type: ModelType) -> Dict[str, Any]:
        """Get request configuration for SQL generator requests"""
        config = self.model_configs.get_model_config(model_type)
        capabilities = self.model_configs.get_model_capabilities(model_type)
        
        return {
            "model_id": config.model_id,
            "base_url": config.base_url,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "timeout": config.timeout,
            "headers": config.headers,
            "supports_reasoning": capabilities.supports_reasoning_mode,
            "supports_thinking": capabilities.supports_thinking_mode,
            "max_context": capabilities.max_context_length,
            "specializations": capabilities.specializations
        }
    
    def select_models_for_query(self, query_text: str) -> Dict[str, Any]:
        """Select appropriate models for a query"""
        complexity = self.model_configs._analyze_query_complexity(query_text)
        routing_rules = self.model_configs.get_routing_rules(complexity)
        
        return {
            "complexity": complexity.value,
            "strategy": routing_rules.generation_strategy.value,
            "primary_models": [model.value for model in routing_rules.primary_models],
            "fallback_models": [model.value for model in routing_rules.fallback_models],
            "use_ensemble": routing_rules.generation_strategy == GenerationStrategy.ENSEMBLE,
            "confidence_threshold": routing_rules.confidence_threshold
        }
    
    def get_ensemble_config(self) -> Dict[str, Any]:
        """Get ensemble configuration for orchestrator"""
        return {
            "enabled": self.model_configs.ensemble_config.enabled,
            "combination_strategy": self.model_configs.ensemble_config.combination_strategy,
            "consensus_threshold": self.model_configs.ensemble_config.consensus_threshold,
            "use_ensemble_for": self.model_configs.ensemble_config.use_ensemble_for
        }

def test_model_connectivity():
    """Simple connectivity test for both models"""
    print("🔍 Testing Model Connectivity...")
    print("=" * 40)
    
    for model_type in ModelType:
        config = get_model_config(model_type)
        print(f"\n📡 Testing {model_type.value}:")
        print(f"   URL: {config.base_url}")
        print(f"   Model ID: {config.model_id}")
        print(f"   Temperature: {config.temperature}")
        print(f"   Max Tokens: {config.max_tokens}")
        
        try:
            # Test basic connectivity
            test_url = config.base_url.replace('/v1/chat/completions', '/health') if '/v1/chat/completions' in config.base_url else config.base_url
            response = requests.get(test_url, timeout=5, headers=config.headers)
            
            if response.status_code < 500:
                print(f"   Status: ✅ REACHABLE (HTTP {response.status_code})")
            else:
                print(f"   Status: ⚠️ ISSUES (HTTP {response.status_code})")
                
        except requests.exceptions.RequestException as e:
            error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
            print(f"   Status: ❌ UNREACHABLE ({error_msg})")

def run_comprehensive_test():
    """Run comprehensive test of the configuration system"""
    print("🚀 Comprehensive NGROK Model Configuration Test")
    print("=" * 60)
    
    print("\n1️⃣ Environment Configuration")
    issues = NGROKConfigHelper.validate_ngrok_urls()
    if issues:
        print("   Issues found:")
        for issue in issues:
            print(f"      {issue}")
        print("\n📝 Creating example configuration...")
        NGROKConfigHelper.create_env_file()
    else:
        print("   ✅ All environment variables configured correctly")
    
    print("\n2️⃣ Model Configuration Validation")
    validation = validate_all_models()
    print(f"   Status: {'✅ VALID' if validation['valid'] else '❌ ISSUES FOUND'}")
    
    if not validation['valid']:
        print("   Issues:")
        for issue in validation['issues']:
            print(f"      - {issue}")
    
    for model_name, status in validation['model_status'].items():
        print(f"   {model_name}: {'✅ CONFIGURED' if status['configured'] else '❌ ISSUES'}")
    
    print("\n3️⃣ Query Routing Test")
    test_queries = [
        "SELECT * FROM customers",
        "Calculate statistical correlation between sales and marketing spend",
        "Complex analytical query with multiple CTEs and window functions",
        "Extract XML data using .value() function from customer_info"
    ]
    
    for query in test_queries:
        selection = orchestrator_integration.select_models_for_query(query)
        query_short = query[:40] + "..." if len(query) > 40 else query
        print(f"   Query: '{query_short}'")
        print(f"      → Complexity: {selection['complexity']}")
        print(f"      → Strategy: {selection['strategy']}")
        print(f"      → Models: {selection['primary_models']}")
    
    print("\n4️⃣ Model Connectivity Test")
    test_model_connectivity()
    
    print("\n5️⃣ Integration Configuration")
    ensemble_config = orchestrator_integration.get_ensemble_config()
    print(f"   Ensemble enabled: {'✅ YES' if ensemble_config['enabled'] else '❌ NO'}")
    print(f"   Combination strategy: {ensemble_config['combination_strategy']}")
    print(f"   Consensus threshold: {ensemble_config['consensus_threshold']}")
    
    print(f"\n🎯 Test Summary:")
    print(f"   Models configured: {len(ModelType)}")
    print(f"   Configuration valid: {'✅ YES' if validation['valid'] else '❌ NO'}")
    print(f"   Environment issues: {len(issues)} found")
    print(f"   Ready for production: {'✅ YES' if validation['valid'] and len(issues) == 0 else '❌ NO'}")

# FIXED: Create global configuration instance
model_configs = ModelConfigs()

# FIXED: Simplified helper functions
def get_model_config(model_type: ModelType) -> ModelEndpoint:
    """Get model configuration"""
    return model_configs.get_model_config(model_type)

def get_generation_strategy(query_text: str) -> Tuple[GenerationStrategy, List[ModelType]]:
    """Get generation strategy and models for a query"""
    return model_configs.get_generation_strategy(query_text)

def should_use_ensemble(query_text: str) -> bool:
    """Determine if ensemble should be used for this query"""
    complexity = model_configs._analyze_query_complexity(query_text)
    return model_configs.should_use_ensemble(complexity)

def validate_all_models() -> Dict[str, Any]:
    """Validate all model configurations"""
    return model_configs.validate_configuration()

# FIXED: Create global integration instance
orchestrator_integration = OrchestratorIntegration()

# Export main classes and functions
__all__ = [
    'ModelType', 'QueryComplexity', 'GenerationStrategy', 
    'ModelEndpoint', 'ModelCapabilities', 'RoutingRules',
    'ModelConfigs', 'NGROKConfigHelper', 'OrchestratorIntegration',
    'model_configs', 'get_model_config', 'validate_all_models',
    'orchestrator_integration'
]

if __name__ == "__main__":
    run_comprehensive_test()
