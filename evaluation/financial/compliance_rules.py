"""
Financial Services Compliance Evaluation Rules
Custom metrics for regulatory compliance in financial prompts
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime
from langfuse import observe


class ComplianceLevel(Enum):
    """Compliance requirement levels"""
    CRITICAL = "critical"  # Must pass or deployment blocked
    HIGH = "high"         # Should pass, requires approval to override
    MEDIUM = "medium"     # Should pass, warning if failed
    LOW = "low"          # Informational only


@dataclass
class ComplianceViolation:
    """Represents a compliance violation"""
    rule: str
    level: ComplianceLevel
    message: str
    evidence: str
    suggestion: Optional[str] = None


class FinancialComplianceEvaluator:
    """
    Evaluates prompts for financial services compliance requirements
    including SEC, FINRA, and internal policies
    """
    
    def __init__(self):
        """Initialize compliance evaluator with rule sets"""
        self.required_disclosures = {
            "past_performance": {
                "pattern": r"past\s+performance.*not.*indicative.*future",
                "message": "Missing past performance disclaimer",
                "level": ComplianceLevel.CRITICAL
            },
            "risk_warning": {
                "pattern": r"(investment|trading|financial).*risk|risk.*loss",
                "message": "Missing risk warning",
                "level": ComplianceLevel.CRITICAL
            },
            "not_advice": {
                "pattern": r"not.*financial.*advice|informational.*purposes",
                "message": "Missing 'not financial advice' disclaimer",
                "level": ComplianceLevel.HIGH
            },
            "regulatory_notice": {
                "pattern": r"SEC|FINRA|regulatory|compliance",
                "message": "Missing regulatory notice",
                "level": ComplianceLevel.MEDIUM
            }
        }
        
        self.prohibited_terms = {
            "guaranteed_returns": {
                "patterns": [
                    r"guaranteed.*return",
                    r"risk[\s-]?free.*investment",
                    r"cannot.*lose",
                    r"sure.*thing"
                ],
                "message": "Contains prohibited guarantee language",
                "level": ComplianceLevel.CRITICAL
            },
            "insider_info": {
                "patterns": [
                    r"insider.*information",
                    r"confidential.*tip",
                    r"not.*public"
                ],
                "message": "References potential insider information",
                "level": ComplianceLevel.CRITICAL
            },
            "misleading_claims": {
                "patterns": [
                    r"always.*profit",
                    r"never.*lose",
                    r"beat.*market.*every",
                    r"outperform.*guaranteed"
                ],
                "message": "Contains misleading performance claims",
                "level": ComplianceLevel.CRITICAL
            },
            "unregistered_advice": {
                "patterns": [
                    r"should.*buy",
                    r"recommend.*purchase",
                    r"advise.*invest",
                    r"must.*sell"
                ],
                "message": "Provides specific investment advice without registration",
                "level": ComplianceLevel.HIGH
            }
        }
        
        self.bias_patterns = {
            "demographic_bias": {
                "patterns": [
                    r"(young|old|elderly|millennial).*investor",
                    r"(male|female|men|women).*portfolio",
                    r"(race|ethnic|cultural).*investment"
                ],
                "message": "Contains potential demographic bias",
                "level": ComplianceLevel.HIGH
            },
            "socioeconomic_bias": {
                "patterns": [
                    r"(rich|wealthy|poor).*client",
                    r"high[\s-]?net[\s-]?worth.*only",
                    r"sophisticated.*investor.*only"
                ],
                "message": "Contains socioeconomic bias",
                "level": ComplianceLevel.MEDIUM
            }
        }
    
    @observe(name="evaluate_compliance")
    def evaluate(
        self,
        prompt_output: str,
        prompt_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate prompt output for compliance
        
        Args:
            prompt_output: The generated prompt output
            prompt_context: Optional context about the prompt
        
        Returns:
            Compliance evaluation results
        """
        violations = []
        scores = {}
        
        # Check required disclosures
        disclosure_results = self._check_disclosures(prompt_output)
        violations.extend(disclosure_results["violations"])
        scores["disclosure_compliance"] = disclosure_results["score"]
        
        # Check prohibited terms
        prohibition_results = self._check_prohibitions(prompt_output)
        violations.extend(prohibition_results["violations"])
        scores["prohibition_compliance"] = prohibition_results["score"]
        
        # Check for bias
        bias_results = self._check_bias(prompt_output)
        violations.extend(bias_results["violations"])
        scores["bias_score"] = bias_results["score"]
        
        # Check data handling
        data_results = self._check_data_handling(prompt_output)
        violations.extend(data_results["violations"])
        scores["data_compliance"] = data_results["score"]
        
        # Check regulatory requirements
        regulatory_results = self._check_regulatory_requirements(
            prompt_output,
            prompt_context
        )
        violations.extend(regulatory_results["violations"])
        scores["regulatory_compliance"] = regulatory_results["score"]
        
        # Calculate overall compliance score
        critical_violations = [v for v in violations if v.level == ComplianceLevel.CRITICAL]
        high_violations = [v for v in violations if v.level == ComplianceLevel.HIGH]
        
        if critical_violations:
            overall_score = 0.0
        elif high_violations:
            overall_score = max(0.5, 1.0 - (0.1 * len(high_violations)))
        else:
            overall_score = min(scores.values()) if scores else 1.0
        
        return {
            "compliant": len(critical_violations) == 0,
            "overall_score": overall_score,
            "scores": scores,
            "violations": violations,
            "critical_count": len(critical_violations),
            "high_count": len(high_violations),
            "total_violations": len(violations),
            "timestamp": datetime.utcnow().isoformat(),
            "evaluator_version": "1.0.0"
        }
    
    def _check_disclosures(self, text: str) -> Dict[str, Any]:
        """Check for required regulatory disclosures"""
        violations = []
        found_count = 0
        
        text_lower = text.lower()
        
        for disclosure_key, disclosure_config in self.required_disclosures.items():
            if not re.search(disclosure_config["pattern"], text_lower):
                violations.append(ComplianceViolation(
                    rule=f"disclosure_{disclosure_key}",
                    level=disclosure_config["level"],
                    message=disclosure_config["message"],
                    evidence="Pattern not found in output",
                    suggestion=f"Add appropriate {disclosure_key.replace('_', ' ')} disclaimer"
                ))
            else:
                found_count += 1
        
        score = found_count / len(self.required_disclosures) if self.required_disclosures else 1.0
        
        return {
            "score": score,
            "violations": violations
        }
    
    def _check_prohibitions(self, text: str) -> Dict[str, Any]:
        """Check for prohibited terms and claims"""
        violations = []
        text_lower = text.lower()
        
        for prohibition_key, prohibition_config in self.prohibited_terms.items():
            for pattern in prohibition_config["patterns"]:
                match = re.search(pattern, text_lower)
                if match:
                    violations.append(ComplianceViolation(
                        rule=f"prohibition_{prohibition_key}",
                        level=prohibition_config["level"],
                        message=prohibition_config["message"],
                        evidence=f"Found: '{match.group()}'",
                        suggestion="Remove or rephrase prohibited language"
                    ))
                    break
        
        score = 1.0 if not violations else 0.0
        
        return {
            "score": score,
            "violations": violations
        }
    
    def _check_bias(self, text: str) -> Dict[str, Any]:
        """Check for demographic and other biases"""
        violations = []
        text_lower = text.lower()
        bias_found = False
        
        for bias_key, bias_config in self.bias_patterns.items():
            for pattern in bias_config["patterns"]:
                match = re.search(pattern, text_lower)
                if match:
                    violations.append(ComplianceViolation(
                        rule=f"bias_{bias_key}",
                        level=bias_config["level"],
                        message=bias_config["message"],
                        evidence=f"Found: '{match.group()}'",
                        suggestion="Use neutral, inclusive language"
                    ))
                    bias_found = True
                    break
        
        score = 0.95 if not bias_found else max(0.5, 1.0 - (0.1 * len(violations)))
        
        return {
            "score": score,
            "violations": violations
        }
    
    def _check_data_handling(self, text: str) -> Dict[str, Any]:
        """Check for proper data handling and privacy"""
        violations = []
        text_lower = text.lower()
        
        # Check for PII exposure
        pii_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{9}\b",              # SSN without dashes
            r"\b[A-Z]{2}\d{6}\b",      # Account numbers
            r"\b\d{16}\b",             # Credit card
        ]
        
        for pattern in pii_patterns:
            if re.search(pattern, text):
                violations.append(ComplianceViolation(
                    rule="data_pii_exposure",
                    level=ComplianceLevel.CRITICAL,
                    message="Potential PII exposure detected",
                    evidence="Pattern matching PII format found",
                    suggestion="Redact or tokenize sensitive information"
                ))
                break
        
        # Check for data retention claims
        if re.search(r"(store|retain|keep).*permanently|forever", text_lower):
            violations.append(ComplianceViolation(
                rule="data_retention_policy",
                level=ComplianceLevel.MEDIUM,
                message="Improper data retention claim",
                evidence="Claims permanent data retention",
                suggestion="Align with company data retention policy"
            ))
        
        score = 1.0 if not violations else 0.0
        
        return {
            "score": score,
            "violations": violations
        }
    
    def _check_regulatory_requirements(
        self,
        text: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check specific regulatory requirements based on context"""
        violations = []
        text_lower = text.lower()
        
        # Determine applicable regulations from context
        if context:
            product_type = context.get("product_type", "").lower()
            
            # SEC requirements for securities
            if "security" in product_type or "stock" in product_type:
                if not re.search(r"SEC|securities.*exchange.*commission", text_lower):
                    violations.append(ComplianceViolation(
                        rule="regulatory_sec_disclosure",
                        level=ComplianceLevel.HIGH,
                        message="Missing SEC disclosure for securities discussion",
                        evidence="No SEC reference found",
                        suggestion="Add appropriate SEC disclosures"
                    ))
            
            # CFTC requirements for derivatives
            if "derivative" in product_type or "futures" in product_type:
                if not re.search(r"CFTC|commodity.*futures", text_lower):
                    violations.append(ComplianceViolation(
                        rule="regulatory_cftc_disclosure",
                        level=ComplianceLevel.HIGH,
                        message="Missing CFTC disclosure for derivatives",
                        evidence="No CFTC reference found",
                        suggestion="Add appropriate CFTC disclosures"
                    ))
        
        score = 1.0 if not violations else max(0.7, 1.0 - (0.15 * len(violations)))
        
        return {
            "score": score,
            "violations": violations
        }


class RiskDisclosureEvaluator:
    """Evaluates risk disclosure completeness and accuracy"""
    
    def __init__(self):
        """Initialize risk disclosure evaluator"""
        self.required_risks = [
            "market_risk",
            "credit_risk",
            "liquidity_risk",
            "operational_risk",
            "regulatory_risk"
        ]
    
    @observe(name="evaluate_risk_disclosure")
    def evaluate(self, prompt_output: str) -> Dict[str, Any]:
        """
        Evaluate risk disclosure quality
        
        Args:
            prompt_output: The generated prompt output
        
        Returns:
            Risk disclosure evaluation results
        """
        scores = {}
        missing_risks = []
        
        text_lower = prompt_output.lower()
        
        # Check each required risk type
        risk_patterns = {
            "market_risk": r"market.*risk|price.*fluctuat|volatility",
            "credit_risk": r"credit.*risk|default.*risk|counterparty",
            "liquidity_risk": r"liquidity.*risk|unable.*sell|market.*depth",
            "operational_risk": r"operational.*risk|system.*failure|human.*error",
            "regulatory_risk": r"regulatory.*risk|compliance.*risk|legal.*risk"
        }
        
        for risk_type, pattern in risk_patterns.items():
            if re.search(pattern, text_lower):
                scores[risk_type] = 1.0
            else:
                scores[risk_type] = 0.0
                missing_risks.append(risk_type)
        
        # Calculate overall score
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        
        # Check for clear risk presentation
        clarity_score = 1.0
        if not re.search(r"risk.*include|following.*risk|risk.*factor", text_lower):
            clarity_score = 0.7
        
        return {
            "complete": len(missing_risks) == 0,
            "overall_score": overall_score * clarity_score,
            "risk_scores": scores,
            "missing_risks": missing_risks,
            "clarity_score": clarity_score,
            "recommendation": "Add missing risk disclosures" if missing_risks else "Risk disclosure complete"
        }


class BiasDetector:
    """Detects various forms of bias in financial prompts"""
    
    @observe(name="detect_bias")
    def detect(self, prompt_output: str) -> Dict[str, Any]:
        """
        Detect bias in prompt output
        
        Args:
            prompt_output: The generated prompt output
        
        Returns:
            Bias detection results
        """
        biases = []
        
        # Demographic bias
        demographic_terms = {
            "age": ["young", "old", "elderly", "millennial", "boomer"],
            "gender": ["male", "female", "men", "women", "he", "she"],
            "ethnicity": ["white", "black", "asian", "hispanic", "ethnic"]
        }
        
        for bias_type, terms in demographic_terms.items():
            for term in terms:
                if re.search(rf"\b{term}\b", prompt_output, re.IGNORECASE):
                    biases.append({
                        "type": f"demographic_{bias_type}",
                        "term": term,
                        "severity": "high"
                    })
        
        # Investment bias
        if re.search(r"conservative.*investor|aggressive.*trader", prompt_output, re.IGNORECASE):
            biases.append({
                "type": "investment_style",
                "term": "style labeling",
                "severity": "medium"
            })
        
        # Calculate bias score (1.0 = no bias, 0.0 = severe bias)
        if not biases:
            bias_score = 1.0
        else:
            high_bias = len([b for b in biases if b["severity"] == "high"])
            medium_bias = len([b for b in biases if b["severity"] == "medium"])
            bias_score = max(0.0, 1.0 - (0.2 * high_bias) - (0.1 * medium_bias))
        
        return {
            "bias_free": len(biases) == 0,
            "bias_score": bias_score,
            "biases_found": biases,
            "bias_count": len(biases)
        }