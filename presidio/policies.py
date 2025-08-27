"""
PII Policy Management for Presidio Integration

Defines configurable policies for PII handling actions across different
compliance requirements and use cases.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Any
import json


class PIIAction(Enum):
    """Available actions for PII entities"""
    REDACT = "redact"           # Complete removal: "John Smith" -> "[REDACTED]"
    MASK = "mask"               # Partial masking: "555-123-4567" -> "555-***-****"
    HASH = "hash"               # One-way hash: "john@example.com" -> "hash_abc123"
    TOKENIZE = "tokenize"       # Reversible token: "4532-1234-5678-9012" -> "TOKEN_cc_001"
    SYNTHETIC = "synthetic"     # Synthetic replacement: "John Smith" -> "David Johnson"
    ALLOW = "allow"             # No action taken - pass through


@dataclass
class PIIPolicy:
    """Configuration for PII handling policy"""
    name: str
    version: str
    description: Optional[str] = None
    entities: Dict[str, PIIAction] = None
    default_action: PIIAction = PIIAction.REDACT
    retention_hours: int = 24
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = {}
        if self.metadata is None:
            self.metadata = {}


class PIIPolicyEngine:
    """Manages PII policies and provides policy lookup"""
    
    def __init__(self):
        self.policies = {}
        self._load_default_policies()
    
    def _load_default_policies(self):
        """Load predefined policies for common use cases"""
        
        # Financial services compliance policy
        self.policies["financial_services_standard"] = PIIPolicy(
            name="financial_services_standard",
            version="1.0.0",
            description="Standard financial services PII policy for regulatory compliance",
            entities={
                # Financial identifiers - high security
                "CREDIT_CARD": PIIAction.TOKENIZE,
                "US_SSN": PIIAction.TOKENIZE,
                "US_BANK_NUMBER": PIIAction.TOKENIZE,
                "IBAN_CODE": PIIAction.TOKENIZE,
                
                # Contact information - moderate security
                "PHONE_NUMBER": PIIAction.MASK,
                "EMAIL_ADDRESS": PIIAction.HASH,
                "IP_ADDRESS": PIIAction.HASH,
                
                # Personal identifiers - synthetic replacement
                "PERSON": PIIAction.SYNTHETIC,
                "US_DRIVER_LICENSE": PIIAction.TOKENIZE,
                "US_PASSPORT": PIIAction.TOKENIZE,
                
                # Location data - masking
                "LOCATION": PIIAction.MASK,
                "ADDRESS": PIIAction.MASK,
                
                # Medical - high security
                "MEDICAL_LICENSE": PIIAction.TOKENIZE,
                "US_DEA_NUMBER": PIIAction.TOKENIZE,
                
                # Dates - contextual handling
                "DATE_TIME": PIIAction.MASK,
            },
            default_action=PIIAction.REDACT,
            retention_hours=72,  # Extended for audit trail
            metadata={
                "compliance_framework": "SOX, PCI-DSS, GDPR",
                "risk_level": "high",
                "audit_required": True
            }
        )
        
        # Development/testing policy - more permissive
        self.policies["development"] = PIIPolicy(
            name="development",
            version="1.0.0", 
            description="Development environment policy with synthetic data",
            entities={
                "CREDIT_CARD": PIIAction.SYNTHETIC,
                "US_SSN": PIIAction.SYNTHETIC,
                "PHONE_NUMBER": PIIAction.SYNTHETIC,
                "EMAIL_ADDRESS": PIIAction.SYNTHETIC,
                "PERSON": PIIAction.SYNTHETIC,
                "ADDRESS": PIIAction.SYNTHETIC,
                "DATE_TIME": PIIAction.ALLOW,  # Allow for testing
            },
            default_action=PIIAction.SYNTHETIC,
            retention_hours=1,  # Short retention for dev
            metadata={
                "environment": "development",
                "risk_level": "low"
            }
        )
        
        # Strict compliance policy - maximum protection
        self.policies["strict"] = PIIPolicy(
            name="strict",
            version="1.0.0",
            description="Maximum protection policy for highly sensitive data",
            entities={
                # Everything gets redacted or tokenized
                "CREDIT_CARD": PIIAction.TOKENIZE,
                "US_SSN": PIIAction.REDACT,
                "PHONE_NUMBER": PIIAction.REDACT,
                "EMAIL_ADDRESS": PIIAction.REDACT,
                "PERSON": PIIAction.REDACT,
                "ADDRESS": PIIAction.REDACT,
                "DATE_TIME": PIIAction.REDACT,
                "IP_ADDRESS": PIIAction.REDACT,
                "MEDICAL_LICENSE": PIIAction.REDACT,
            },
            default_action=PIIAction.REDACT,
            retention_hours=0,  # No retention
            metadata={
                "compliance_framework": "HIPAA, SOX, GDPR",
                "risk_level": "maximum",
                "audit_required": True,
                "reversible": False
            }
        )
    
    def get_policy(self, policy_name: str) -> Optional[PIIPolicy]:
        """Retrieve policy by name"""
        return self.policies.get(policy_name)
    
    def add_policy(self, policy: PIIPolicy):
        """Add or update a policy"""
        self.policies[policy.name] = policy
    
    def list_policies(self) -> Dict[str, PIIPolicy]:
        """Get all available policies"""
        return self.policies.copy()
    
    def get_action_for_entity(self, entity_type: str, policy_name: str) -> PIIAction:
        """Get the action for a specific entity type under a policy"""
        policy = self.get_policy(policy_name)
        if not policy:
            return PIIAction.REDACT  # Safe default
            
        return policy.entities.get(entity_type, policy.default_action)
    
    def validate_policy(self, policy: PIIPolicy) -> Dict[str, Any]:
        """Validate policy configuration"""
        issues = []
        warnings = []
        
        # Check for required fields
        if not policy.name:
            issues.append("Policy name is required")
        if not policy.version:
            issues.append("Policy version is required")
            
        # Validate entity actions
        for entity, action in policy.entities.items():
            if not isinstance(action, PIIAction):
                issues.append(f"Invalid action for entity {entity}: {action}")
                
        # Check retention policy
        if policy.retention_hours < 0:
            issues.append("Retention hours cannot be negative")
        elif policy.retention_hours > 168:  # 7 days
            warnings.append(f"Long retention period: {policy.retention_hours} hours")
            
        # Validate reversibility requirements
        reversible_actions = {PIIAction.TOKENIZE, PIIAction.MASK}
        non_reversible_actions = {PIIAction.REDACT, PIIAction.HASH}
        
        has_reversible = any(action in reversible_actions for action in policy.entities.values())
        has_non_reversible = any(action in non_reversible_actions for action in policy.entities.values())
        
        if has_reversible and policy.retention_hours == 0:
            warnings.append("Reversible actions with zero retention may cause issues")
            
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "has_reversible_actions": has_reversible,
            "has_non_reversible_actions": has_non_reversible
        }
    
    def export_policy(self, policy_name: str) -> Optional[str]:
        """Export policy as JSON string"""
        policy = self.get_policy(policy_name)
        if not policy:
            return None
            
        policy_dict = {
            "name": policy.name,
            "version": policy.version,
            "description": policy.description,
            "entities": {k: v.value for k, v in policy.entities.items()},
            "default_action": policy.default_action.value,
            "retention_hours": policy.retention_hours,
            "metadata": policy.metadata
        }
        
        return json.dumps(policy_dict, indent=2)
    
    def import_policy(self, policy_json: str) -> PIIPolicy:
        """Import policy from JSON string"""
        policy_dict = json.loads(policy_json)
        
        # Convert action strings back to enums
        entities = {}
        for entity, action_str in policy_dict.get("entities", {}).items():
            entities[entity] = PIIAction(action_str)
            
        default_action = PIIAction(policy_dict.get("default_action", "redact"))
        
        return PIIPolicy(
            name=policy_dict["name"],
            version=policy_dict["version"], 
            description=policy_dict.get("description"),
            entities=entities,
            default_action=default_action,
            retention_hours=policy_dict.get("retention_hours", 24),
            metadata=policy_dict.get("metadata", {})
        )