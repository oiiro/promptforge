"""
Vendor-neutral LLM Client with pluggable providers and Presidio PII Integration
Supports OpenAI, Anthropic, HuggingFace, and Airia.ai with comprehensive PII anonymization
"""

import os
import json
import time
import logging
import uuid
import asyncio
from typing import Dict, Any, Optional, Union, Tuple
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path

import openai
import anthropic
import redis
from dotenv import load_dotenv

# Import Presidio middleware (graceful fallback if not available)
try:
    from presidio.middleware import PresidioMiddleware
    from presidio.policies import PIIPolicyEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    logger.warning("Presidio middleware not available. PII features disabled.")
    PRESIDIO_AVAILABLE = False

# Import guardrails (graceful fallback if not available)
try:
    from guardrails.validators import GuardrailOrchestrator
    GUARDRAILS_AVAILABLE = True
except ImportError:
    logger.warning("Guardrails not available. Security features disabled.")
    GUARDRAILS_AVAILABLE = False

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


class PIIAwareLLMClient:
    """
    Enhanced LLM Client with Presidio PII Anonymization Integration
    
    Provides comprehensive PII protection workflow:
    1. Pre-processing: PII detection and anonymization
    2. LLM Generation: Safe prompt processing
    3. Post-processing: PII restoration (if authorized)
    4. Audit logging: Complete PII handling trail
    """
    
    def __init__(
        self,
        provider: str = None,
        model: str = None,
        pii_policy: str = "financial_services_standard",
        enable_pii_processing: bool = True,
        enable_guardrails: bool = True,
        redis_host: str = "localhost",
        redis_port: int = 6379
    ):
        """Initialize PII-aware LLM client"""
        
        # Initialize base LLM client
        self.llm_client = LLMClient(provider=provider, model=model)
        
        # Initialize Presidio middleware if available
        self.presidio = None
        self.pii_policy = pii_policy
        self.enable_pii_processing = enable_pii_processing and PRESIDIO_AVAILABLE
        
        if self.enable_pii_processing:
            try:
                self.presidio = PresidioMiddleware(
                    redis_host=redis_host,
                    redis_port=redis_port
                )
                logger.info("Presidio middleware initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Presidio middleware: {e}")
                self.enable_pii_processing = False
        
        # Initialize guardrails if available
        self.guardrails = None
        self.enable_guardrails = enable_guardrails and GUARDRAILS_AVAILABLE
        
        if self.enable_guardrails:
            try:
                self.guardrails = GuardrailOrchestrator()
                logger.info("Guardrails initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize guardrails: {e}")
                self.enable_guardrails = False
    
    async def generate_with_pii_protection(
        self,
        prompt: str,
        session_id: str = None,
        restore_pii: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response with comprehensive PII protection
        
        Args:
            prompt: Input text that may contain PII
            session_id: Unique session identifier for PII mapping
            restore_pii: Whether to restore PII in final response
            **kwargs: Additional generation parameters
            
        Returns:
            Dict containing response, PII metadata, and processing details
        """
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        processing_metadata = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "pii_processing_enabled": self.enable_pii_processing,
            "guardrails_enabled": self.enable_guardrails,
            "policy_used": self.pii_policy,
            "restore_pii_requested": restore_pii,
            "processing_steps": []
        }
        
        try:
            # Step 1: Pre-execution guardrails
            processed_prompt = prompt
            guardrails_violations = []
            
            if self.enable_guardrails:
                start_time = time.time()
                is_valid, processed_prompt, violations = self.guardrails.validate_request(prompt)
                guardrails_time = (time.time() - start_time) * 1000
                
                processing_metadata["processing_steps"].append({
                    "step": "pre_execution_guardrails",
                    "duration_ms": round(guardrails_time, 2),
                    "violations_found": len(violations),
                    "prompt_modified": processed_prompt != prompt
                })
                
                if not is_valid:
                    return {
                        "response": "Request blocked by security guardrails",
                        "error": "GUARDRAIL_VIOLATION",
                        "violations": violations,
                        "metadata": processing_metadata,
                        "session_id": session_id
                    }
                
                guardrails_violations = violations
            
            # Step 2: PII Detection and Anonymization
            anonymized_prompt = processed_prompt
            pii_metadata = {}
            
            if self.enable_pii_processing and self.presidio:
                start_time = time.time()
                anonymized_prompt, pii_metadata = await self.presidio.anonymize(
                    text=processed_prompt,
                    session_id=session_id,
                    policy_name=self.pii_policy
                )
                pii_time = (time.time() - start_time) * 1000
                
                processing_metadata["processing_steps"].append({
                    "step": "pii_anonymization",
                    "duration_ms": round(pii_time, 2),
                    "entities_found": pii_metadata.get("entities_found", 0),
                    "entities_processed": pii_metadata.get("entities_processed", []),
                    "reversible_entities": pii_metadata.get("reversible_entities", 0)
                })
                
                logger.info(
                    f"Session {session_id}: Anonymized {pii_metadata.get('entities_found', 0)} PII entities"
                )
            
            # Step 3: LLM Generation
            start_time = time.time()
            
            # Extract country from prompt for capital finder use case
            country = self._extract_country_from_prompt(anonymized_prompt)
            if not country:
                country = anonymized_prompt.split()[-1]  # Fallback: use last word
            
            llm_response = self.llm_client.generate(country, **kwargs)
            llm_time = (time.time() - start_time) * 1000
            
            processing_metadata["processing_steps"].append({
                "step": "llm_generation",
                "duration_ms": round(llm_time, 2),
                "provider": self.llm_client.provider_name,
                "model": self.llm_client.model
            })
            
            # Parse LLM response
            try:
                llm_response_json = json.loads(llm_response)
            except json.JSONDecodeError:
                llm_response_json = {"response": llm_response}
            
            # Step 4: Post-execution guardrails
            validated_response = llm_response_json
            
            if self.enable_guardrails:
                start_time = time.time()
                response_text = json.dumps(llm_response_json)
                is_valid, sanitized_response, post_violations = self.guardrails.validate_response(response_text)
                post_guardrails_time = (time.time() - start_time) * 1000
                
                processing_metadata["processing_steps"].append({
                    "step": "post_execution_guardrails",
                    "duration_ms": round(post_guardrails_time, 2),
                    "violations_found": len(post_violations),
                    "response_modified": sanitized_response != response_text
                })
                
                if sanitized_response != response_text:
                    try:
                        validated_response = json.loads(sanitized_response)
                    except json.JSONDecodeError:
                        validated_response = {"response": sanitized_response}
                
                guardrails_violations.extend(post_violations)
            
            # Step 5: PII Restoration (if requested and authorized)
            final_response = validated_response
            restoration_metadata = {}
            
            if restore_pii and self.enable_pii_processing and self.presidio:
                start_time = time.time()
                
                # Convert response back to string for restoration
                response_text = json.dumps(validated_response)
                
                restored_text, restoration_metadata = await self.presidio.deanonymize(
                    anonymized_text=response_text,
                    session_id=session_id,
                    policy_name=self.pii_policy
                )
                restoration_time = (time.time() - start_time) * 1000
                
                processing_metadata["processing_steps"].append({
                    "step": "pii_restoration",
                    "duration_ms": round(restoration_time, 2),
                    "entities_restored": restoration_metadata.get("restored_entities", 0),
                    "restoration_rate": restoration_metadata.get("restoration_rate", 0.0)
                })
                
                # Parse restored response
                try:
                    final_response = json.loads(restored_text)
                except json.JSONDecodeError:
                    final_response = {"response": restored_text}
                
                logger.info(
                    f"Session {session_id}: Restored {restoration_metadata.get('restored_entities', 0)} PII entities"
                )
            
            # Step 6: Compile final response with comprehensive metadata
            total_processing_time = sum(
                step.get("duration_ms", 0) 
                for step in processing_metadata["processing_steps"]
            )
            
            processing_metadata.update({
                "total_processing_time_ms": round(total_processing_time, 2),
                "pii_entities_detected": pii_metadata.get("entities_found", 0),
                "pii_entities_restored": restoration_metadata.get("restored_entities", 0),
                "guardrail_violations": len(guardrails_violations),
                "processing_successful": True
            })
            
            # Add PII protection status to response
            if isinstance(final_response, dict):
                final_response["pii_protection"] = {
                    "session_id": session_id,
                    "pii_detected": pii_metadata.get("entities_found", 0) > 0,
                    "pii_anonymized": self.enable_pii_processing,
                    "pii_restored": restore_pii and restoration_metadata.get("restored_entities", 0) > 0,
                    "policy_applied": self.pii_policy,
                    "guardrails_applied": self.enable_guardrails
                }
            
            return {
                "response": final_response,
                "session_id": session_id,
                "metadata": processing_metadata,
                "pii_metadata": pii_metadata,
                "restoration_metadata": restoration_metadata,
                "guardrails_violations": guardrails_violations,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Generation with PII protection failed for session {session_id}: {e}")
            
            # Clean up session on error
            if self.enable_pii_processing and self.presidio:
                try:
                    await self.presidio.cleanup_session(session_id)
                except:
                    pass
            
            processing_metadata["processing_successful"] = False
            processing_metadata["error"] = str(e)
            
            return {
                "response": {
                    "capital": "Unknown",
                    "confidence": 0.0,
                    "error": "Processing failed with PII protection"
                },
                "session_id": session_id,
                "metadata": processing_metadata,
                "error": str(e),
                "status": "error"
            }
    
    def _extract_country_from_prompt(self, prompt: str) -> Optional[str]:
        """Extract country name from prompt for capital finder use case"""
        
        # Common patterns in capital finder prompts
        patterns = [
            "capital of ",
            "capital city of ",
            "what is the capital of ",
            "find the capital of ",
            "tell me the capital of "
        ]
        
        prompt_lower = prompt.lower()
        
        for pattern in patterns:
            if pattern in prompt_lower:
                # Extract text after the pattern
                start_idx = prompt_lower.find(pattern) + len(pattern)
                remaining = prompt[start_idx:].strip()
                
                # Take first word/phrase (until punctuation or newline)
                import re
                match = re.match(r'^([A-Za-z\s]+)', remaining)
                if match:
                    return match.group(1).strip()
        
        # Fallback: look for country-like words
        words = prompt.split()
        for word in words:
            word = word.strip('.,!?:"()[]{}')
            if len(word) > 2 and word.isalpha() and word[0].isupper():
                return word
        
        return None
    
    async def generate(
        self,
        country: str,
        session_id: str = None,
        restore_pii: bool = False,
        **kwargs
    ) -> str:
        """
        Simplified generate method for backward compatibility
        
        Returns JSON string response (same as original LLMClient)
        """
        
        # Create prompt for capital finder
        prompt = f"What is the capital of {country}?"
        
        # Call comprehensive PII-aware generation
        result = await self.generate_with_pii_protection(
            prompt=prompt,
            session_id=session_id,
            restore_pii=restore_pii,
            **kwargs
        )
        
        # Return JSON response for compatibility
        response_data = result.get("response", {})
        if isinstance(response_data, dict):
            return json.dumps(response_data)
        else:
            return json.dumps({"response": str(response_data)})
    
    def health_check(self) -> Dict[str, Any]:
        """Enhanced health check including PII protection components"""
        
        base_health = self.llm_client.health_check()
        
        # Check Presidio middleware health
        presidio_health = False
        if self.enable_pii_processing and self.presidio:
            try:
                # Test Redis connection
                test_info = self.presidio.redis_client.ping()
                presidio_health = test_info
            except:
                presidio_health = False
        
        # Check guardrails health
        guardrails_health = False
        if self.enable_guardrails and self.guardrails:
            try:
                # Test guardrails validation
                self.guardrails.validate_request("test")
                guardrails_health = True
            except:
                guardrails_health = False
        
        return {
            **base_health,
            "pii_protection": {
                "presidio_enabled": self.enable_pii_processing,
                "presidio_healthy": presidio_health,
                "guardrails_enabled": self.enable_guardrails,
                "guardrails_healthy": guardrails_health,
                "policy": self.pii_policy,
                "redis_connected": presidio_health
            },
            "features": {
                "pii_anonymization": self.enable_pii_processing,
                "pii_restoration": self.enable_pii_processing,
                "security_guardrails": self.enable_guardrails,
                "audit_logging": True,
                "session_management": self.enable_pii_processing
            }
        }
    
    async def cleanup_session(self, session_id: str) -> Dict[str, Any]:
        """Clean up PII mappings for a session"""
        
        if self.enable_pii_processing and self.presidio:
            try:
                return await self.presidio.cleanup_session(session_id)
            except Exception as e:
                logger.error(f"Failed to cleanup session {session_id}: {e}")
                return {"error": str(e)}
        
        return {"message": "No cleanup needed - PII processing disabled"}
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a PII processing session"""
        
        if self.enable_pii_processing and self.presidio:
            try:
                return self.presidio.get_session_info(session_id)
            except Exception as e:
                logger.error(f"Failed to get session info for {session_id}: {e}")
                return None
        
        return None


# Mock provider for testing without API keys
class MockProvider(LLMProvider):
    """Mock provider for testing and development"""
    
    def __init__(self, model: str = "mock-model"):
        self.model = model
        
        # Predefined responses for capital cities
        self.responses = {
            "france": {"capital": "Paris", "confidence": 1.0},
            "germany": {"capital": "Berlin", "confidence": 1.0},
            "japan": {"capital": "Tokyo", "confidence": 1.0},
            "uk": {"capital": "London", "confidence": 1.0},
            "united kingdom": {"capital": "London", "confidence": 1.0},
            "usa": {"capital": "Washington D.C.", "confidence": 1.0},
            "united states": {"capital": "Washington D.C.", "confidence": 1.0},
            "canada": {"capital": "Ottawa", "confidence": 1.0},
            "australia": {"capital": "Canberra", "confidence": 1.0},
            "italy": {"capital": "Rome", "confidence": 1.0},
            "spain": {"capital": "Madrid", "confidence": 1.0},
        }
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate mock response for testing"""
        
        # Extract country from prompt
        prompt_lower = prompt.lower()
        found_country = None
        
        for country in self.responses.keys():
            if country in prompt_lower:
                found_country = country
                break
        
        # Generate response
        if found_country:
            response_data = self.responses[found_country]
        else:
            response_data = {"capital": "Unknown", "confidence": 0.0}
        
        return {
            "response": json.dumps(response_data),
            "model": self.model,
            "provider": "mock",
            "latency_ms": 50,  # Simulated latency
            "tokens": {
                "prompt": len(prompt.split()),
                "completion": 5,
                "total": len(prompt.split()) + 5
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def health_check(self) -> bool:
        """Mock provider is always healthy"""
        return True


# Enhanced example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_pii_aware_client():
        """Test the PII-aware LLM client with comprehensive features"""
        
        print("🧪 Testing PII-Aware LLM Client...")
        print("=" * 50)
        
        # Initialize with mock provider (no API keys needed)
        client = PIIAwareLLMClient(
            provider="mock",
            enable_pii_processing=True,
            enable_guardrails=True
        )
        
        # Health check
        print("\n1. Health Check:")
        health = client.health_check()
        print(json.dumps(health, indent=2))
        
        # Test with PII in prompt
        print("\n2. Testing with PII in prompt:")
        pii_prompt = "My name is John Smith and my SSN is 123-45-6789. What's the capital of France?"
        
        try:
            result = await client.generate_with_pii_protection(
                prompt=pii_prompt,
                session_id="test_session_001",
                restore_pii=False
            )
            
            print(f"Response: {json.dumps(result['response'], indent=2)}")
            print(f"PII Metadata: {result['pii_metadata']}")
            print(f"Processing Steps: {len(result['metadata']['processing_steps'])}")
            
        except Exception as e:
            print(f"Test failed: {e}")
        
        # Test simplified interface
        print("\n3. Testing simplified interface:")
        try:
            simple_response = await client.generate("Germany", session_id="test_session_002")
            print(f"Simple response: {simple_response}")
            
        except Exception as e:
            print(f"Simple test failed: {e}")
        
        # Cleanup
        print("\n4. Session cleanup:")
        try:
            cleanup_result = await client.cleanup_session("test_session_001")
            print(f"Cleanup result: {cleanup_result}")
            
        except Exception as e:
            print(f"Cleanup failed: {e}")
        
        print("\n✅ PII-Aware LLM Client testing completed!")
    
    # For backward compatibility testing
    print("🔄 Testing backward compatibility...")
    basic_client = LLMClient(provider="mock")
    
    # Add mock provider to basic client
    basic_client.provider = MockProvider()
    
    print("Basic client response:", basic_client.generate("France"))
    print("Basic client health:", basic_client.health_check())
    
    # Run async tests
    print("\n🚀 Running comprehensive PII tests...")
    asyncio.run(test_pii_aware_client())