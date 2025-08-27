"""
Vendor-neutral LLM Client with pluggable providers
Supports OpenAI, Anthropic, HuggingFace, and Airia.ai
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path

import openai
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if provider is available"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider implementation"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4-turbo-preview"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
        
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response using OpenAI API"""
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0),
                max_tokens=kwargs.get("max_tokens", 200),
                response_format={"type": "json_object"}
            )
            
            elapsed_time = (time.time() - start_time) * 1000
            
            return {
                "response": response.choices[0].message.content,
                "model": self.model,
                "provider": "openai",
                "latency_ms": elapsed_time,
                "tokens": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check OpenAI API availability"""
        try:
            self.client.models.list()
            return True
        except:
            return False

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation"""
    
    def __init__(self, api_key: str = None, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response using Anthropic API"""
        try:
            start_time = time.time()
            
            # Add JSON instruction for Claude
            json_prompt = f"{prompt}\n\nRespond with valid JSON only."
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 200),
                temperature=kwargs.get("temperature", 0),
                messages=[{"role": "user", "content": json_prompt}]
            )
            
            elapsed_time = (time.time() - start_time) * 1000
            
            return {
                "response": response.content[0].text,
                "model": self.model,
                "provider": "anthropic",
                "latency_ms": elapsed_time,
                "tokens": {
                    "prompt": response.usage.input_tokens,
                    "completion": response.usage.output_tokens,
                    "total": response.usage.input_tokens + response.usage.output_tokens
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check Anthropic API availability"""
        try:
            # Simple check with minimal tokens
            self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except:
            return False

class HuggingFaceProvider(LLMProvider):
    """HuggingFace provider implementation"""
    
    def __init__(self, api_key: str = None, model: str = "meta-llama/Llama-2-70b-chat-hf"):
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.model = model
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"
        
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response using HuggingFace API"""
        import requests
        
        try:
            start_time = time.time()
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": kwargs.get("max_tokens", 200),
                    "temperature": kwargs.get("temperature", 0),
                    "return_full_text": False
                }
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            elapsed_time = (time.time() - start_time) * 1000
            result = response.json()
            
            # Extract generated text
            generated_text = result[0]["generated_text"] if isinstance(result, list) else result
            
            return {
                "response": generated_text,
                "model": self.model,
                "provider": "huggingface",
                "latency_ms": elapsed_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check HuggingFace API availability"""
        import requests
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                "https://api-inference.huggingface.co/api/status",
                headers=headers
            )
            return response.status_code == 200
        except:
            return False

class AiriaProvider(LLMProvider):
    """Airia.ai provider implementation"""
    
    def __init__(self, api_key: str = None, endpoint: str = None):
        self.api_key = api_key or os.getenv("AIRIA_API_KEY")
        self.endpoint = endpoint or os.getenv("AIRIA_ENDPOINT", "https://api.airia.ai/v1/completions")
        
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response using Airia.ai API"""
        import requests
        
        try:
            start_time = time.time()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 200),
                "temperature": kwargs.get("temperature", 0),
                "format": "json"
            }
            
            response = requests.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            
            elapsed_time = (time.time() - start_time) * 1000
            result = response.json()
            
            return {
                "response": result.get("completion", result),
                "model": "airia",
                "provider": "airia",
                "latency_ms": elapsed_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Airia generation error: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check Airia API availability"""
        import requests
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                self.endpoint.replace("/completions", "/health"),
                headers=headers
            )
            return response.status_code == 200
        except:
            return False

class LLMClient:
    """
    Unified LLM client with provider abstraction
    Manages prompt templates, provider selection, and response handling
    """
    
    def __init__(self, provider: str = None, model: str = None):
        """Initialize LLM client with specified provider"""
        self.provider_name = provider or os.getenv("DEFAULT_LLM_PROVIDER", "openai")
        self.model = model or os.getenv("DEFAULT_MODEL")
        self.provider = self._initialize_provider()
        self.prompt_template = self._load_prompt_template()
        
    def _initialize_provider(self) -> LLMProvider:
        """Initialize the appropriate provider"""
        providers = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "huggingface": HuggingFaceProvider,
            "airia": AiriaProvider
        }
        
        provider_class = providers.get(self.provider_name.lower())
        if not provider_class:
            raise ValueError(f"Unknown provider: {self.provider_name}")
        
        if self.model:
            return provider_class(model=self.model)
        else:
            return provider_class()
    
    def _load_prompt_template(self) -> str:
        """Load prompt template from file"""
        template_path = Path("prompts/find_capital/template.txt")
        if template_path.exists():
            with open(template_path, "r") as f:
                return f.read()
        else:
            # Fallback template
            return """Find the capital of: {country}
            
            Respond with JSON: {"capital": "city_name", "confidence": 0-1}"""
    
    def switch_provider(self, provider: str, model: str = None):
        """Switch to a different LLM provider"""
        self.provider_name = provider
        self.model = model
        self.provider = self._initialize_provider()
        logger.info(f"Switched to provider: {provider}")
    
    def generate(self, country: str, **kwargs) -> str:
        """
        Generate capital information for given country
        Returns JSON string response
        """
        # Format prompt
        prompt = self.prompt_template.format(country=country)
        
        # Add metadata
        metadata = {
            "request_id": os.urandom(16).hex(),
            "timestamp": datetime.utcnow().isoformat(),
            "input": country
        }
        
        try:
            # Generate response
            result = self.provider.generate(prompt, **kwargs)
            
            # Parse and validate response
            response_text = result["response"]
            
            # Try to parse as JSON
            try:
                response_json = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback response
                response_json = {
                    "capital": "Unknown",
                    "confidence": 0.0
                }
            
            # Add metadata
            response_json["metadata"] = {
                "source": "geographical_database",
                "timestamp": datetime.utcnow().isoformat(),
                "model": result.get("model"),
                "latency_ms": result.get("latency_ms"),
                "token_usage": result.get("tokens")
            }
            
            # Log for audit
            logger.info(f"Generated response for {country}: {response_json.get('capital')}")
            
            return json.dumps(response_json)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            
            # Return safe fallback
            return json.dumps({
                "capital": "Unknown",
                "confidence": 0.0,
                "error": str(e),
                "metadata": metadata
            })
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of current provider"""
        is_healthy = self.provider.health_check()
        
        return {
            "provider": self.provider_name,
            "model": self.model,
            "healthy": is_healthy,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def benchmark_providers(self, test_input: str = "France") -> Dict[str, Any]:
        """Benchmark all available providers"""
        results = {}
        
        providers = ["openai", "anthropic", "huggingface", "airia"]
        
        for provider_name in providers:
            try:
                # Switch provider
                self.switch_provider(provider_name)
                
                # Check health
                if not self.provider.health_check():
                    results[provider_name] = {"status": "unavailable"}
                    continue
                
                # Generate response
                start_time = time.time()
                response = self.generate(test_input)
                elapsed_time = (time.time() - start_time) * 1000
                
                # Parse response
                response_json = json.loads(response)
                
                results[provider_name] = {
                    "status": "success",
                    "response": response_json.get("capital"),
                    "confidence": response_json.get("confidence"),
                    "latency_ms": elapsed_time,
                    "model": self.model
                }
                
            except Exception as e:
                results[provider_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize client
    client = LLMClient()
    
    # Test generation
    print("Testing LLM client...")
    response = client.generate("France")
    print(f"Response: {response}")
    
    # Health check
    health = client.health_check()
    print(f"Health: {health}")
    
    # Benchmark providers (if API keys are available)
    # print("\nBenchmarking providers...")
    # benchmark = client.benchmark_providers()
    # print(json.dumps(benchmark, indent=2))