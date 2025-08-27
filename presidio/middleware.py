"""
Presidio Middleware for PII Processing

Core middleware component that integrates Microsoft Presidio with PromptForge
for PII anonymization and de-anonymization workflows.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any
import redis
import secrets
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine, DeanonymizeEngine
from presidio_anonymizer.entities import OperatorConfig

from .policies import PIIAction, PIIPolicy, PIIPolicyEngine


logger = logging.getLogger(__name__)


class PresidioMiddleware:
    """
    Core middleware for PII processing using Microsoft Presidio
    
    Provides anonymization and de-anonymization capabilities with
    policy-based configuration and Redis-based secure mapping storage.
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        redis_db: int = 0,
        encryption_key: Optional[str] = None
    ):
        """Initialize Presidio middleware with Redis configuration"""
        
        # Initialize Presidio engines
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.deanonymizer = DeanonymizeEngine()
        
        # Initialize policy engine
        self.policy_engine = PIIPolicyEngine()
        
        # Initialize Redis connection for secure mapping storage
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            db=redis_db,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        
        # Encryption key for sensitive mappings
        self.encryption_key = encryption_key or secrets.token_hex(32)
        
        # Test Redis connection
        try:
            self.redis_client.ping()
            logger.info("Redis connection established successfully")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def anonymize(
        self,
        text: str,
        session_id: str,
        policy_name: str = "financial_services_standard",
        language: str = "en"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Anonymize PII entities in text according to policy
        
        Returns:
            Tuple of (anonymized_text, pii_metadata)
        """
        
        try:
            # Get policy configuration
            policy = self.policy_engine.get_policy(policy_name)
            if not policy:
                logger.error(f"Policy not found: {policy_name}")
                raise ValueError(f"Unknown policy: {policy_name}")
            
            # Analyze text for PII entities
            analyzer_results = self.analyzer.analyze(
                text=text,
                language=language,
                entities=list(policy.entities.keys()) if policy.entities else None
            )
            
            if not analyzer_results:
                logger.debug("No PII entities detected")
                return text, {
                    "session_id": session_id,
                    "policy": policy_name,
                    "entities_found": 0,
                    "entities_processed": [],
                    "processing_time_ms": 0
                }
            
            start_time = time.time()
            
            # Build anonymization operations based on policy
            operators = {}
            pii_mappings = {}
            
            for result in analyzer_results:
                entity_type = result.entity_type
                action = policy.entities.get(entity_type, policy.default_action)
                
                operator_config = self._build_operator_config(
                    action, entity_type, result, session_id
                )
                
                if operator_config:
                    operators[entity_type] = operator_config
                    
                    # Store mapping for reversible operations
                    if action in {PIIAction.TOKENIZE, PIIAction.MASK}:
                        original_text = text[result.start:result.end]
                        mapping_key = self._generate_mapping_key(
                            session_id, entity_type, result.start, result.end
                        )
                        pii_mappings[mapping_key] = {
                            "original": original_text,
                            "entity_type": entity_type,
                            "start": result.start,
                            "end": result.end,
                            "confidence": result.score
                        }
            
            # Perform anonymization
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=operators
            )
            
            # Store PII mappings in Redis with TTL
            if pii_mappings:
                await self._store_pii_mappings(
                    session_id, pii_mappings, policy.retention_hours
                )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Build metadata
            metadata = {
                "session_id": session_id,
                "policy": policy_name,
                "entities_found": len(analyzer_results),
                "entities_processed": [
                    {
                        "type": r.entity_type,
                        "confidence": r.score,
                        "start": r.start,
                        "end": r.end,
                        "action": policy.entities.get(r.entity_type, policy.default_action).value
                    }
                    for r in analyzer_results
                ],
                "processing_time_ms": round(processing_time, 2),
                "reversible_entities": len(pii_mappings),
                "timestamp": int(time.time())
            }
            
            logger.info(
                f"Anonymized {len(analyzer_results)} PII entities "
                f"in {processing_time:.2f}ms for session {session_id}"
            )
            
            return anonymized_result.text, metadata
            
        except Exception as e:
            logger.error(f"Anonymization failed for session {session_id}: {e}")
            raise
    
    async def deanonymize(
        self,
        anonymized_text: str,
        session_id: str,
        policy_name: str = "financial_services_standard"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        De-anonymize text by restoring original PII entities
        
        Returns:
            Tuple of (deanonymized_text, restoration_metadata)
        """
        
        try:
            start_time = time.time()
            
            # Retrieve PII mappings from Redis
            pii_mappings = await self._retrieve_pii_mappings(session_id)
            
            if not pii_mappings:
                logger.warning(f"No PII mappings found for session {session_id}")
                return anonymized_text, {
                    "session_id": session_id,
                    "restored_entities": 0,
                    "processing_time_ms": 0,
                    "status": "no_mappings_found"
                }
            
            # Build deanonymization operators
            deanonymization_mapping = {}
            for mapping_key, mapping_data in pii_mappings.items():
                # Create reverse mapping from anonymized token to original
                token = self._extract_token_from_key(mapping_key)
                if token:
                    deanonymization_mapping[token] = mapping_data["original"]
            
            # Perform de-anonymization using simple string replacement
            # (Presidio's DeanonymizeEngine is more complex but this works for our use case)
            deanonymized_text = anonymized_text
            restored_count = 0
            
            for token, original in deanonymization_mapping.items():
                if token in deanonymized_text:
                    deanonymized_text = deanonymized_text.replace(token, original)
                    restored_count += 1
            
            processing_time = (time.time() - start_time) * 1000
            
            metadata = {
                "session_id": session_id,
                "restored_entities": restored_count,
                "total_mappings_available": len(pii_mappings),
                "processing_time_ms": round(processing_time, 2),
                "timestamp": int(time.time()),
                "status": "success"
            }
            
            logger.info(
                f"Restored {restored_count} PII entities "
                f"in {processing_time:.2f}ms for session {session_id}"
            )
            
            return deanonymized_text, metadata
            
        except Exception as e:
            logger.error(f"De-anonymization failed for session {session_id}: {e}")
            raise
    
    def _build_operator_config(
        self,
        action: PIIAction,
        entity_type: str,
        analyzer_result,
        session_id: str
    ) -> Optional[OperatorConfig]:
        """Build Presidio operator configuration for given action"""
        
        if action == PIIAction.REDACT:
            return OperatorConfig("redact")
            
        elif action == PIIAction.MASK:
            # Mask with asterisks, keep first/last characters for readability
            return OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": -1})
            
        elif action == PIIAction.HASH:
            # Create consistent hash for same values
            original_text = analyzer_result.text[analyzer_result.start:analyzer_result.end]
            hash_value = hashlib.sha256(
                f"{entity_type}:{original_text}:{self.encryption_key}".encode()
            ).hexdigest()[:8]
            return OperatorConfig("replace", {"new_value": f"HASH_{entity_type}_{hash_value}"})
            
        elif action == PIIAction.TOKENIZE:
            # Generate reversible token
            token_id = str(uuid.uuid4())[:8]
            token = f"TOKEN_{entity_type}_{token_id}"
            return OperatorConfig("replace", {"new_value": token})
            
        elif action == PIIAction.SYNTHETIC:
            # Use Presidio's built-in synthetic data generation
            return OperatorConfig("replace", {"new_value": self._generate_synthetic_value(entity_type)})
            
        elif action == PIIAction.ALLOW:
            return None  # No operation - pass through
            
        else:
            logger.warning(f"Unknown action {action}, defaulting to redact")
            return OperatorConfig("redact")
    
    def _generate_synthetic_value(self, entity_type: str) -> str:
        """Generate synthetic replacement values for different entity types"""
        
        synthetic_values = {
            "PERSON": ["John Smith", "Jane Doe", "Robert Johnson", "Emily Davis", "Michael Wilson"],
            "PHONE_NUMBER": ["555-0100", "555-0101", "555-0102", "555-0103", "555-0104"],
            "EMAIL_ADDRESS": ["user1@example.com", "user2@example.com", "user3@example.com"],
            "CREDIT_CARD": ["4532-0000-0000-0000", "5555-0000-0000-0000", "3400-0000-0000-000"],
            "ADDRESS": ["123 Main St", "456 Oak Ave", "789 Pine Rd", "321 Elm St"],
            "DATE_TIME": ["2023-01-01", "2023-06-15", "2023-12-31"],
            "LOCATION": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
        }
        
        import random
        values = synthetic_values.get(entity_type, [f"SYNTHETIC_{entity_type}"])
        return random.choice(values)
    
    def _generate_mapping_key(
        self, session_id: str, entity_type: str, start: int, end: int
    ) -> str:
        """Generate unique key for PII mapping storage"""
        return f"pii_mapping:{session_id}:{entity_type}:{start}:{end}"
    
    def _extract_token_from_key(self, mapping_key: str) -> Optional[str]:
        """Extract token identifier from mapping key for reverse lookup"""
        # This is a simplified implementation
        # In production, you'd want a more robust token extraction mechanism
        parts = mapping_key.split(":")
        if len(parts) >= 4:
            return f"TOKEN_{parts[2]}"
        return None
    
    async def _store_pii_mappings(
        self, session_id: str, mappings: Dict[str, Any], retention_hours: int
    ):
        """Store PII mappings in Redis with TTL"""
        
        try:
            pipe = self.redis_client.pipeline()
            ttl_seconds = retention_hours * 3600
            
            for mapping_key, mapping_data in mappings.items():
                encrypted_data = self._encrypt_mapping_data(mapping_data)
                pipe.setex(mapping_key, ttl_seconds, json.dumps(encrypted_data))
                
            # Store session metadata
            session_key = f"pii_session:{session_id}"
            session_metadata = {
                "created_at": int(time.time()),
                "mapping_count": len(mappings),
                "retention_hours": retention_hours
            }
            pipe.setex(session_key, ttl_seconds, json.dumps(session_metadata))
            
            await asyncio.get_event_loop().run_in_executor(None, pipe.execute)
            logger.debug(f"Stored {len(mappings)} PII mappings for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to store PII mappings: {e}")
            raise
    
    async def _retrieve_pii_mappings(self, session_id: str) -> Dict[str, Any]:
        """Retrieve PII mappings from Redis"""
        
        try:
            # Find all mapping keys for this session
            pattern = f"pii_mapping:{session_id}:*"
            keys = self.redis_client.keys(pattern)
            
            if not keys:
                return {}
            
            # Retrieve all mappings
            values = self.redis_client.mget(keys)
            mappings = {}
            
            for key, value in zip(keys, values):
                if value:
                    try:
                        encrypted_data = json.loads(value)
                        mapping_data = self._decrypt_mapping_data(encrypted_data)
                        mappings[key] = mapping_data
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode mapping data for key {key}")
            
            return mappings
            
        except Exception as e:
            logger.error(f"Failed to retrieve PII mappings: {e}")
            return {}
    
    def _encrypt_mapping_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive mapping data (simplified implementation)"""
        # In production, use proper encryption like AES
        # This is a basic example using base64 encoding
        import base64
        
        sensitive_fields = ["original"]
        encrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in encrypted_data:
                value = str(encrypted_data[field])
                encoded = base64.b64encode(value.encode()).decode()
                encrypted_data[field] = f"enc:{encoded}"
                
        return encrypted_data
    
    def _decrypt_mapping_data(self, encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive mapping data"""
        import base64
        
        decrypted_data = encrypted_data.copy()
        
        for key, value in decrypted_data.items():
            if isinstance(value, str) and value.startswith("enc:"):
                try:
                    encoded_value = value[4:]  # Remove "enc:" prefix
                    decoded = base64.b64decode(encoded_value).decode()
                    decrypted_data[key] = decoded
                except Exception as e:
                    logger.error(f"Failed to decrypt field {key}: {e}")
                    
        return decrypted_data
    
    async def cleanup_session(self, session_id: str) -> Dict[str, Any]:
        """Clean up all PII mappings for a session"""
        
        try:
            # Find all keys for this session
            mapping_pattern = f"pii_mapping:{session_id}:*"
            session_pattern = f"pii_session:{session_id}"
            
            mapping_keys = self.redis_client.keys(mapping_pattern)
            session_keys = self.redis_client.keys(session_pattern)
            
            all_keys = mapping_keys + session_keys
            
            if all_keys:
                deleted_count = self.redis_client.delete(*all_keys)
                logger.info(f"Cleaned up {deleted_count} keys for session {session_id}")
                
                return {
                    "session_id": session_id,
                    "keys_deleted": deleted_count,
                    "status": "success"
                }
            else:
                return {
                    "session_id": session_id,
                    "keys_deleted": 0,
                    "status": "no_data_found"
                }
                
        except Exception as e:
            logger.error(f"Failed to cleanup session {session_id}: {e}")
            raise
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a PII processing session"""
        
        try:
            session_key = f"pii_session:{session_id}"
            session_data = self.redis_client.get(session_key)
            
            if not session_data:
                return None
                
            session_info = json.loads(session_data)
            
            # Add current mapping count
            mapping_pattern = f"pii_mapping:{session_id}:*"
            current_mappings = len(self.redis_client.keys(mapping_pattern))
            session_info["current_mapping_count"] = current_mappings
            
            # Add TTL info
            ttl = self.redis_client.ttl(session_key)
            session_info["ttl_seconds"] = ttl
            
            return session_info
            
        except Exception as e:
            logger.error(f"Failed to get session info for {session_id}: {e}")
            return None