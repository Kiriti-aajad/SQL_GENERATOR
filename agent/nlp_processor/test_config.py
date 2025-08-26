"""
Simple test for NLP Processor configuration
"""
import sys
import os
from pathlib import Path

# Add the project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from agent.nlp_processor.config import get_config
    
    print("Testing NLP Processor Configuration...")
    
    # Test configuration loading
    config = get_config()
    print("Configuration loaded successfully!")
    
    # Test configuration statistics
    stats = config.get_statistics()
    print(f"Configuration statistics: {stats}")
    
    # Test specific configurations
    print(f"High confidence joins: {len(config.get_high_confidence_joins())}")
    print(f"XML enabled tables: {config.get_xml_enabled_tables()}")
    print(f"Processing timeout: {config.get('processing.timeout_seconds', 'default')}")
    
    print("Configuration test completed successfully!")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure config.py is in the nlp_processor directory")
    print("Check if agent/nlp_processor/config.py exists")
except Exception as e:
    print(f"Configuration test failed: {e}")
    import traceback
    traceback.print_exc()
