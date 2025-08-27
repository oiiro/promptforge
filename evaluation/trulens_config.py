"""
TruLens Configuration and Setup
Comprehensive evaluation and monitoring backbone for PromptForge
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

# TruLens imports - Updated for v2.x
from trulens.core import TruSession as Tru, Select
from trulens.core.feedback import Feedback
from trulens.feedback import GroundTruthAgreement
from trulens.feedback.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

# Custom feedback functions (defined as module-level functions for serialization)
def check_financial_compliance(response: str) -> float:
    """Custom feedback function for financial services compliance"""
    prohibited_terms = [
        'investment advice', 'stock recommendation', 'buy stocks',
        'financial advice', 'portfolio', 'trading', 'invest in',
        'insider trading', 'market manipulation', 'guaranteed returns'
    ]
    
    response_lower = response.lower()
    violations = sum(1 for term in prohibited_terms if term in response_lower)
    
    # Return compliance score (1.0 = fully compliant, 0.0 = major violations)
    if violations == 0:
        return 1.0
    elif violations <= 2:
        return 0.5
    else:
        return 0.0

def check_schema_compliance(response: str) -> float:
    """Custom feedback function for JSON schema compliance"""
    import json
    import jsonschema
    
    try:
        # Try to parse as JSON
        parsed = json.loads(response)
        
        # Expected schema for capital finder
        expected_schema = {
            "type": "object",
            "properties": {
                "capital": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "metadata": {"type": "object"}
            },
            "required": ["capital", "confidence"]
        }
        
        # Validate against schema
        jsonschema.validate(parsed, expected_schema)
        return 1.0  # Perfect compliance
        
    except json.JSONDecodeError:
        return 0.0  # Not valid JSON
    except jsonschema.exceptions.ValidationError:
        return 0.5  # Valid JSON but doesn't match schema
    except Exception:
        return 0.0  # Other errors

class MockProvider:
    """Mock provider for testing when no API keys are available"""
    
    def relevance_with_cot_reasons(self, prompt: str, response: str) -> float:
        """Mock relevance evaluation"""
        return 0.8
    
    def toxicity(self, text: str) -> float:
        """Mock toxicity evaluation"""
        return 0.1
    
    def conciseness(self, text: str) -> float:
        """Mock conciseness evaluation"""
        return 0.7
    
    def language_match(self, prompt: str, response: str) -> float:
        """Mock language match evaluation"""
        return 0.9

class TruLensConfig:
    """TruLens configuration and initialization"""
    
    def __init__(self, 
                 database_url: str = "sqlite:///trulens.db",
                 reset_database: bool = False):
        """Initialize TruLens configuration"""
        self.database_url = database_url
        self.reset_database = reset_database
        self.tru = None
        self.feedback_functions = {}
        self.providers = {}
        
        self._setup_providers()
        self._initialize_tru()
        
    def _setup_providers(self):
        """Setup feedback providers based on available API keys"""
        
        # Try to setup OpenAI LLMProvider
        if os.getenv('OPENAI_API_KEY'):
            try:
                self.providers['openai'] = LLMProvider(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    model_engine="gpt-4-turbo-preview"
                )
                logger.info("OpenAI LLM provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI provider: {e}")
        
        # Use mock provider as fallback
        if not self.providers:
            logger.warning("No API keys configured. Using mock provider for testing.")
            self.providers['mock'] = MockProvider()
    
    def _initialize_tru(self):
        """Initialize TruLens with database connection"""
        try:
            self.tru = Tru(database_url=self.database_url)
            
            if self.reset_database:
                self.tru.reset_database()
                logger.info("TruLens database reset")
            
            logger.info(f"TruLens initialized with database: {self.database_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize TruLens: {e}")
            raise
    
    def create_feedback_functions(self) -> Dict[str, Feedback]:
        """Create comprehensive feedback functions for evaluation"""
        
        # Get primary provider
        primary_provider = (self.providers.get('openai') or 
                          self.providers.get('mock'))
        if not primary_provider:
            raise ValueError("No primary provider available for feedback functions")
        
        feedback_functions = {}
        
        # 1. Answer Relevance
        try:
            feedback_functions['answer_relevance'] = Feedback(
                primary_provider.relevance_with_cot_reasons,
                name="Answer Relevance"
            )
            logger.info("Answer relevance feedback function created")
        except Exception as e:
            logger.warning(f"Failed to create answer relevance feedback: {e}")
        
        # 2. Toxicity Detection
        try:
            feedback_functions['toxicity'] = Feedback(
                primary_provider.toxicity,
                name="Toxicity"
            )
            logger.info("Toxicity feedback function created")
        except Exception as e:
            logger.warning(f"Failed to create toxicity feedback: {e}")
        
        # 3. Response Conciseness
        try:
            feedback_functions['conciseness'] = Feedback(
                primary_provider.conciseness,
                name="Response Conciseness"
            )
            logger.info("Conciseness feedback function created")
        except Exception as e:
            logger.warning(f"Failed to create conciseness feedback: {e}")
        
        # 4. Language Match (for international financial services)
        try:
            feedback_functions['language_match'] = Feedback(
                primary_provider.language_match,
                name="Language Match"
            )
            logger.info("Language match feedback function created")
        except Exception as e:
            logger.warning(f"Failed to create language match feedback: {e}")
        
        # 5. Financial Services Compliance (Custom)
        try:
            feedback_functions['financial_compliance'] = Feedback(
                check_financial_compliance,
                name="Financial Services Compliance"
            )
            logger.info("Financial compliance feedback function created")
        except Exception as e:
            logger.warning(f"Failed to create financial compliance feedback: {e}")
        
        # 6. JSON Schema Compliance (Custom)
        try:
            feedback_functions['schema_compliance'] = Feedback(
                check_schema_compliance,
                name="JSON Schema Compliance"
            )
            logger.info("Schema compliance feedback function created")
        except Exception as e:
            logger.warning(f"Failed to create schema compliance feedback: {e}")
        
        self.feedback_functions = feedback_functions
        logger.info(f"Created {len(feedback_functions)} feedback functions")
        
        return feedback_functions
    
    def create_golden_dataset_feedback(self, golden_dataset_path: str) -> Feedback:
        """Create ground truth agreement feedback using golden dataset"""
        
        try:
            # Load golden dataset
            golden_df = pd.read_csv(golden_dataset_path)
            
            # Create ground truth mapping
            ground_truth_dict = dict(zip(golden_df['country'], golden_df['capital']))
            
            # Create ground truth agreement feedback
            ground_truth_agreement = GroundTruthAgreement(
                ground_truth=ground_truth_dict,
                provider=self.providers.get('openai') or self.providers.get('anthropic')
            )
            
            feedback = Feedback(
                ground_truth_agreement.agreement_measure,
                name="Golden Dataset Agreement"
            ).on(Select.RecordCalls.llm.args.inputs.country) \
             .on_output()
            
            logger.info(f"Golden dataset feedback created with {len(ground_truth_dict)} examples")
            return feedback
            
        except Exception as e:
            logger.error(f"Failed to create golden dataset feedback: {e}")
            raise
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.providers.keys())
    
    def get_tru_session(self):
        """Get the TruLens session"""
        return self.tru
    
    def get_feedback_functions(self) -> Dict[str, Feedback]:
        """Get created feedback functions"""
        if not self.feedback_functions:
            self.create_feedback_functions()
        return self.feedback_functions
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of available feedback functions"""
        return {
            'total_functions': len(self.feedback_functions),
            'function_names': list(self.feedback_functions.keys()),
            'providers': list(self.providers.keys()),
            'database_url': self.database_url
        }

# Global TruLens configuration instance
trulens_config = None

def get_trulens_config(reset_database: bool = False) -> TruLensConfig:
    """Get or create global TruLens configuration"""
    global trulens_config
    
    if trulens_config is None:
        database_url = os.getenv('TRULENS_DATABASE_URL', 'sqlite:///trulens_promptforge.db')
        trulens_config = TruLensConfig(database_url=database_url, reset_database=reset_database)
    
    return trulens_config

# Example usage
if __name__ == "__main__":
    # Initialize TruLens configuration
    config = TruLensConfig()
    
    # Create feedback functions
    feedback_functions = config.create_feedback_functions()
    
    # Print summary
    summary = config.get_feedback_summary()
    print(f"TruLens Configuration Summary:")
    print(f"- Total feedback functions: {summary['total_functions']}")
    print(f"- Function names: {', '.join(summary['function_names'])}")
    print(f"- Available providers: {', '.join(summary['providers'])}")
    print(f"- Database: {summary['database_url']}")