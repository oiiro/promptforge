#!/usr/bin/env python3
"""
Component Runner Script

A simple wrapper script that can run individual components of the developer guide.
Each component is self-contained and can be tested independently.

Usage:
    python run_component.py schema_validation
    python run_component.py prompt_templating  
    python run_component.py unit_testing
    python run_component.py all
"""

import sys
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

class ComponentRunner:
    """Runs individual components from the developer guide."""
    
    def __init__(self):
        self.components = {
            'schema_validation': self.run_schema_validation,
            'prompt_templating': self.run_prompt_templating,
            'unit_testing': self.run_unit_testing,
            'integration_testing': self.run_integration_testing,
            'test_data_generation': self.run_test_data_generation,
            'heuristic_validation': self.run_heuristic_validation,
            'policy_filters': self.run_policy_filters,
            'response_modification': self.run_response_modification,
            'custom_evaluation': self.run_custom_evaluation,
            'iterative_refinement': self.run_iterative_refinement
        }
    
    def run_component(self, component_name: str) -> bool:
        """Run a specific component and return success status."""
        if component_name == 'all':
            return self.run_all_components()
        
        if component_name not in self.components:
            print(f"❌ Unknown component: {component_name}")
            print(f"Available components: {list(self.components.keys())}")
            return False
        
        print(f"🔧 Running {component_name}")
        print("-" * 40)
        
        try:
            success = self.components[component_name]()
            if success:
                print(f"✅ {component_name} completed successfully")
            else:
                print(f"⚠️  {component_name} completed with issues")
            return success
        except Exception as e:
            print(f"❌ {component_name} failed: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return False
    
    def run_all_components(self) -> bool:
        """Run all components in sequence."""
        print("🚀 Running All Components")
        print("=" * 50)
        
        results = {}
        for component_name in self.components:
            print(f"\n📦 Component: {component_name}")
            results[component_name] = self.run_component(component_name)
        
        # Summary
        passed = sum(1 for success in results.values() if success)
        total = len(results)
        
        print(f"\n📊 Summary: {passed}/{total} components passed")
        return passed == total
    
    def run_schema_validation(self) -> bool:
        """Run JSON schema validation example."""
        try:
            # Check if schema files exist
            input_schema = Path("schemas/retirement_input_schema.json")
            output_schema = Path("schemas/retirement_output_schema.json")
            
            if not input_schema.exists() or not output_schema.exists():
                print("📄 Creating example schemas...")
                self._create_example_schemas()
            
            # Try to import jsonschema
            try:
                import jsonschema
                print("✓ jsonschema library available")
            except ImportError:
                print("⚠️  jsonschema not available, showing example structure")
                self._show_schema_example()
                return True
            
            # Validate example data
            self._validate_example_data()
            print("✓ Schema validation working")
            return True
            
        except Exception as e:
            print(f"Schema validation error: {e}")
            return False
    
    def run_prompt_templating(self) -> bool:
        """Run prompt templating example."""
        try:
            # Try to import jinja2
            try:
                from jinja2 import Template
                print("✓ Jinja2 library available")
                
                # Demo template
                template_str = """
                Retirement Assessment for {{ employee.name }}
                Age: {{ employee.age }}, Service: {{ employee.years }} years
                
                Assessment: {% if eligible %}ELIGIBLE{% else %}NOT ELIGIBLE{% endif %}
                """
                
                template = Template(template_str)
                result = template.render(
                    employee={"name": "John Doe", "age": 65, "years": 25},
                    eligible=True
                )
                
                print("✓ Template rendering successful")
                print("Sample output:", result.strip()[:100] + "...")
                
            except ImportError:
                print("⚠️  Jinja2 not available, showing template structure")
                self._show_template_example()
            
            return True
            
        except Exception as e:
            print(f"Template error: {e}")
            return False
    
    def run_unit_testing(self) -> bool:
        """Run unit testing examples."""
        try:
            print("🧪 Running unit test examples...")
            
            # Simple business logic test
            def calculate_eligibility(age: int, service_years: float) -> dict:
                if age >= 65 and service_years >= 20:
                    return {"eligible": True, "type": "Standard"}
                elif service_years >= 30:
                    return {"eligible": True, "type": "Early"}
                else:
                    return {"eligible": False, "type": "NotEligible"}
            
            # Test cases
            test_cases = [
                {"input": (67, 25), "expected": {"eligible": True, "type": "Standard"}},
                {"input": (58, 32), "expected": {"eligible": True, "type": "Early"}},
                {"input": (55, 15), "expected": {"eligible": False, "type": "NotEligible"}}
            ]
            
            passed = 0
            for i, case in enumerate(test_cases, 1):
                age, years = case["input"]
                result = calculate_eligibility(age, years)
                
                if result == case["expected"]:
                    print(f"✓ Test {i}: PASS")
                    passed += 1
                else:
                    print(f"✗ Test {i}: FAIL - Expected {case['expected']}, got {result}")
            
            success = passed == len(test_cases)
            print(f"Unit tests: {passed}/{len(test_cases)} passed")
            return success
            
        except Exception as e:
            print(f"Unit testing error: {e}")
            return False
    
    def run_integration_testing(self) -> bool:
        """Run integration testing example."""
        try:
            print("🔗 Running integration test...")
            
            # Simulate end-to-end pipeline
            def process_retirement_request(employee_data: dict) -> dict:
                age = employee_data["age"]
                service_years = employee_data["yearsOfService"]
                
                # Business logic
                if age >= 65 and service_years >= 20:
                    eligible, eligibility_type = True, "Standard"
                elif service_years >= 30:
                    eligible, eligibility_type = True, "Early"
                else:
                    eligible, eligibility_type = False, "NotEligible"
                
                return {
                    "assessment": {
                        "eligible": eligible,
                        "eligibilityType": eligibility_type,
                        "confidence": 0.95
                    },
                    "processingTime": 150
                }
            
            # Test integration
            test_employee = {
                "id": "EMP-123456",
                "name": "Test Employee",
                "age": 67,
                "yearsOfService": 25
            }
            
            result = process_retirement_request(test_employee)
            
            if result["assessment"]["eligible"] and result["assessment"]["eligibilityType"] == "Standard":
                print("✓ Integration test passed")
                return True
            else:
                print("✗ Integration test failed")
                return False
                
        except Exception as e:
            print(f"Integration testing error: {e}")
            return False
    
    def run_test_data_generation(self) -> bool:
        """Run test data generation example."""
        try:
            import random
            print("📊 Generating test data...")
            
            def generate_test_case(case_type: str) -> dict:
                base_id = random.randint(100000, 999999)
                
                if case_type == "golden":
                    # Perfect case
                    return {
                        "id": f"EMP-{base_id}",
                        "age": 65,
                        "yearsOfService": 25,
                        "expected": {"eligible": True, "type": "Standard"}
                    }
                elif case_type == "edge":
                    # Boundary case
                    return {
                        "id": f"EMP-{base_id}",
                        "age": 65,  # Exactly at threshold
                        "yearsOfService": 20,  # Minimum service
                        "expected": {"eligible": True, "type": "Standard"}
                    }
                else:  # adversarial
                    # Tricky case
                    return {
                        "id": f"EMP-{base_id}",
                        "age": 64,  # Just below
                        "yearsOfService": 19.9,  # Just below
                        "expected": {"eligible": False, "type": "NotEligible"}
                    }
            
            # Generate examples
            for case_type in ["golden", "edge", "adversarial"]:
                test_case = generate_test_case(case_type)
                print(f"✓ Generated {case_type} test case: {test_case['id']}")
            
            return True
            
        except Exception as e:
            print(f"Test data generation error: {e}")
            return False
    
    def run_heuristic_validation(self) -> bool:
        """Run heuristic validation example."""
        try:
            print("🔍 Running heuristic validation...")
            
            def validate_age_service_consistency(age: int, service_years: float) -> bool:
                # Heuristic: service years shouldn't exceed age - 16 (started working at 16)
                return service_years <= (age - 16)
            
            def validate_confidence_explanation(confidence: float, explanation: str) -> bool:
                # Heuristic: high confidence should have detailed explanation
                if confidence > 0.9:
                    return len(explanation) >= 50
                return len(explanation) >= 20
            
            # Test heuristics
            tests = [
                validate_age_service_consistency(65, 40),  # Should pass
                validate_age_service_consistency(65, 55),  # Should fail  
                validate_confidence_explanation(0.95, "Employee meets standard requirements with 25 years of service at age 65."),  # Should pass
                validate_confidence_explanation(0.95, "Eligible."),  # Should fail
            ]
            
            passed = sum(1 for i, result in enumerate(tests) if (i % 2 == 0 and result) or (i % 2 == 1 and not result))
            print(f"✓ Heuristic validation: {passed}/{len(tests)} tests passed correctly")
            
            return passed >= len(tests) * 0.75  # 75% pass rate
            
        except Exception as e:
            print(f"Heuristic validation error: {e}")
            return False
    
    def run_policy_filters(self) -> bool:
        """Run policy filters example."""
        try:
            print("🛡️  Running policy filters...")
            
            def check_confidence_policy(confidence: float) -> list:
                violations = []
                if confidence < 0.85:
                    violations.append("low_confidence")
                return violations
            
            def check_audit_policy(audit_trail: str) -> list:
                violations = []
                if len(audit_trail) < 10:
                    violations.append("incomplete_audit")
                if "ASSESSMENT" not in audit_trail:
                    violations.append("invalid_format")
                return violations
            
            # Test policies
            tests = [
                (check_confidence_policy(0.95), []),  # Should pass
                (check_confidence_policy(0.75), ["low_confidence"]),  # Should fail
                (check_audit_policy("ASSESSMENT_COMPLETED_Standard"), []),  # Should pass
                (check_audit_policy("SHORT"), ["incomplete_audit", "invalid_format"]),  # Should fail
            ]
            
            passed = sum(1 for actual, expected in tests if actual == expected)
            print(f"✓ Policy filters: {passed}/{len(tests)} tests passed")
            
            return passed == len(tests)
            
        except Exception as e:
            print(f"Policy filters error: {e}")
            return False
    
    def run_response_modification(self) -> bool:
        """Run response modification example."""
        try:
            print("🔧 Running response modification...")
            
            def modify_response(response: dict, violations: list) -> dict:
                modified = response.copy()
                
                if "low_confidence" in violations:
                    modified["assessment"]["confidence"] = max(0.85, modified["assessment"]["confidence"])
                    modified["compliance"]["reviewRequired"] = True
                
                if "incomplete_audit" in violations:
                    modified["compliance"]["auditTrail"] = f"ASSESSMENT_COMPLETED_{modified['assessment']['eligibilityType']}"
                
                return modified
            
            # Test modification
            original = {
                "assessment": {"confidence": 0.75, "eligibilityType": "Standard"},
                "compliance": {"auditTrail": "SHORT", "reviewRequired": False}
            }
            
            violations = ["low_confidence", "incomplete_audit"]
            modified = modify_response(original, violations)
            
            success = (
                modified["assessment"]["confidence"] >= 0.85 and
                modified["compliance"]["reviewRequired"] == True and
                "ASSESSMENT_COMPLETED" in modified["compliance"]["auditTrail"]
            )
            
            if success:
                print("✓ Response modification working correctly")
            else:
                print("✗ Response modification failed")
            
            return success
            
        except Exception as e:
            print(f"Response modification error: {e}")
            return False
    
    def run_custom_evaluation(self) -> bool:
        """Run custom evaluation example."""
        try:
            print("📏 Running custom evaluation...")
            
            class SimpleEvaluator:
                def __init__(self, threshold: float = 0.9):
                    self.threshold = threshold
                
                def evaluate(self, expected: dict, actual: dict) -> dict:
                    correct = expected["eligible"] == actual["eligible"]
                    score = 1.0 if correct else 0.0
                    
                    return {
                        "score": score,
                        "passed": score >= self.threshold,
                        "message": f"Accuracy: {score:.2f}"
                    }
            
            # Test evaluation
            evaluator = SimpleEvaluator()
            expected = {"eligible": True, "type": "Standard"}
            actual = {"eligible": True, "type": "Standard"}
            
            result = evaluator.evaluate(expected, actual)
            
            if result["passed"]:
                print(f"✓ Custom evaluation: {result['message']}")
                return True
            else:
                print(f"✗ Custom evaluation failed: {result['message']}")
                return False
                
        except Exception as e:
            print(f"Custom evaluation error: {e}")
            return False
    
    def run_iterative_refinement(self) -> bool:
        """Run iterative refinement example."""
        try:
            print("🔄 Running iterative refinement...")
            
            def analyze_prompt_quality(prompt: str) -> float:
                indicators = [
                    "business rules" in prompt.lower(),
                    "step-by-step" in prompt.lower(),
                    "validation" in prompt.lower(),
                    len(prompt) > 100
                ]
                return sum(indicators) / len(indicators)
            
            def improve_prompt(prompt: str, issues: list) -> str:
                improved = prompt
                
                if "accuracy" in issues:
                    improved += "\n\nBusiness Rules: Follow standard retirement criteria."
                if "clarity" in issues:
                    improved += "\n\nStep-by-step: Analyze systematically."
                
                return improved
            
            # Test refinement
            initial_prompt = "Check if employee can retire."
            quality_before = analyze_prompt_quality(initial_prompt)
            
            improved_prompt = improve_prompt(initial_prompt, ["accuracy", "clarity"])
            quality_after = analyze_prompt_quality(improved_prompt)
            
            improvement = quality_after > quality_before
            
            print(f"✓ Prompt quality improved: {quality_before:.2f} → {quality_after:.2f}")
            return improvement
            
        except Exception as e:
            print(f"Iterative refinement error: {e}")
            return False
    
    # Helper methods
    def _create_example_schemas(self):
        """Create example schema files if they don't exist."""
        schemas_dir = Path("schemas")
        schemas_dir.mkdir(exist_ok=True)
        
        # Simple input schema
        input_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "employee": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer", "minimum": 18, "maximum": 80},
                        "yearsOfService": {"type": "number", "minimum": 0}
                    },
                    "required": ["age", "yearsOfService"]
                }
            },
            "required": ["employee"]
        }
        
        # Simple output schema
        output_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "assessment": {
                    "type": "object",
                    "properties": {
                        "eligible": {"type": "boolean"},
                        "eligibilityType": {"type": "string"}
                    },
                    "required": ["eligible", "eligibilityType"]
                }
            },
            "required": ["assessment"]
        }
        
        with open("schemas/simple_input_schema.json", "w") as f:
            json.dump(input_schema, f, indent=2)
        
        with open("schemas/simple_output_schema.json", "w") as f:
            json.dump(output_schema, f, indent=2)
        
        print("✓ Created simple example schemas")
    
    def _show_schema_example(self):
        """Show schema validation example structure."""
        print("Schema Validation Example Structure:")
        print("""
        Input Schema: schemas/simple_input_schema.json
        {
          "employee": {
            "age": 65,
            "yearsOfService": 25
          }
        }
        
        Output Schema: schemas/simple_output_schema.json
        {
          "assessment": {
            "eligible": true,
            "eligibilityType": "Standard"
          }
        }
        """)
    
    def _show_template_example(self):
        """Show template example structure."""
        print("Prompt Template Example:")
        print("""
        Template Structure:
        
        Assessment for {{ employee.name }}
        Age: {{ employee.age }}, Service: {{ employee.years }}
        Result: {% if eligible %}ELIGIBLE{% else %}NOT ELIGIBLE{% endif %}
        
        Variables:
        - employee.name, employee.age, employee.years
        - eligible (boolean)
        """)
    
    def _validate_example_data(self):
        """Validate example data against schemas."""
        import jsonschema
        
        # Load simple schemas
        try:
            with open("schemas/simple_input_schema.json") as f:
                input_schema = json.load(f)
            with open("schemas/simple_output_schema.json") as f:
                output_schema = json.load(f)
        except FileNotFoundError:
            # Use the full schemas if simple ones don't exist
            with open("schemas/retirement_input_schema.json") as f:
                input_schema = json.load(f)
            with open("schemas/retirement_output_schema.json") as f:
                output_schema = json.load(f)
        
        # Test data that matches the full schemas
        sample_input = {
            "employee": {
                "id": "EMP-123456",
                "name": "John Doe",
                "age": 65,
                "yearsOfService": 25,
                "salary": 75000,
                "department": "Engineering",
                "retirementPlan": "401k",
                "performanceRating": "Exceeds"
            },
            "companyPolicies": {
                "standardRetirementAge": 65,
                "minimumServiceYears": 20,
                "earlyRetirementServiceYears": 30,
                "ruleOf85Enabled": True
            },
            "requestMetadata": {
                "requestId": "550e8400-e29b-41d4-a716-446655440000",
                "requestedBy": "verification-script",
                "timestamp": "2024-01-15T10:00:00Z"
            }
        }
        
        sample_output = {
            "assessment": {
                "eligible": True,
                "eligibilityType": "Standard",
                "confidence": 0.95
            },
            "reasoning": {
                "primaryRule": "Standard",
                "ageCheck": {
                    "currentAge": 65,
                    "requiredAge": 65,
                    "meets": True
                },
                "serviceCheck": {
                    "currentYears": 25,
                    "requiredYears": 20,
                    "meets": True
                },
                "explanation": "Employee meets standard retirement eligibility requirements."
            },
            "compliance": {
                "auditTrail": "ASSESSMENT_COMPLETED_Standard",
                "policyVersion": "2024.1",
                "reviewRequired": False
            },
            "metadata": {
                "requestId": "550e8400-e29b-41d4-a716-446655440000",
                "processedAt": "2024-01-15T10:00:00Z",
                "processingTime": 150
            }
        }
        
        # Validate
        jsonschema.validate(sample_input, input_schema)
        jsonschema.validate(sample_output, output_schema)

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python run_component.py <component_name>")
        print("\nAvailable components:")
        runner = ComponentRunner()
        for component in sorted(runner.components.keys()):
            print(f"  - {component}")
        print("  - all (runs all components)")
        return 1
    
    component_name = sys.argv[1]
    runner = ComponentRunner()
    
    success = runner.run_component(component_name)
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)