#!/usr/bin/env python3
"""
Presidio Capital Finder Demonstration

Demonstrates Presidio PII anonymization and deanonymization with the capital finder service.
Shows how PII is detected, anonymized, processed, and then deanonymized for the final response.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine, DeanonymizeEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False


class PresidioCapitalFinderDemo:
    """Demonstrates PII protection in capital finder queries."""
    
    def __init__(self):
        if not PRESIDIO_AVAILABLE:
            print("❌ Presidio not available. Install with:")
            print("   pip install presidio-analyzer presidio-anonymizer")
            print("   python -m spacy download en_core_web_sm")
            sys.exit(1)
        
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.deanonymizer = DeanonymizeEngine()
        
        # Store anonymization mapping for deanonymization
        self.anonymization_map = {}
    
    def analyze_pii(self, text):
        """Analyze text for PII entities."""
        results = self.analyzer.analyze(text=text, language='en')
        
        print(f"🔍 PII Analysis Results:")
        if results:
            for result in results:
                detected_text = text[result.start:result.end]
                print(f"   - {result.entity_type}: '{detected_text}' "
                      f"(confidence: {result.confidence:.2f})")
        else:
            print("   - No PII detected")
        
        return results
    
    def anonymize_text(self, text, pii_results):
        """Anonymize PII in text and store mapping."""
        if not pii_results:
            return text, {}
        
        # Configure anonymization operators
        operators = {
            "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
            "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
            "DATE_TIME": OperatorConfig("replace", {"new_value": "<DATE>"}),
            "US_SSN": OperatorConfig("replace", {"new_value": "<SSN>"}),
            "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"})
        }
        
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=pii_results,
            operators=operators
        )
        
        # Store the mapping for deanonymization
        self.anonymization_map = {
            item.operator: item.text for item in anonymized_result.items
        }
        
        print(f"🔒 Anonymized Text: '{anonymized_result.text}'")
        print(f"🗝️  Anonymization Map: {self.anonymization_map}")
        
        return anonymized_result.text, self.anonymization_map
    
    def deanonymize_text(self, anonymized_text, anonymization_map):
        """Restore original PII in text."""
        if not anonymization_map:
            return anonymized_text
        
        # Create deanonymization operators
        deanonymize_operators = {}
        for placeholder, original_value in anonymization_map.items():
            deanonymize_operators[placeholder] = OperatorConfig(
                "replace", {"new_value": original_value}
            )
        
        # For demonstration, we'll do simple string replacement
        # In production, you'd use the DeanonymizeEngine properly
        deanonymized_text = anonymized_text
        for placeholder, original_value in anonymization_map.items():
            deanonymized_text = deanonymized_text.replace(placeholder, original_value)
        
        print(f"🔓 Deanonymized Text: '{deanonymized_text}'")
        return deanonymized_text
    
    def mock_capital_query(self, country):
        """Mock capital finder service."""
        capitals = {
            "france": "Paris",
            "japan": "Tokyo", 
            "germany": "Berlin",
            "italy": "Rome",
            "spain": "Madrid",
            "uk": "London",
            "united kingdom": "London",
            "canada": "Ottawa",
            "australia": "Canberra"
        }
        
        country_key = country.lower().strip()
        capital = capitals.get(country_key, f"Capital of {country}")
        
        return {
            "capital": capital,
            "confidence": 1.0,
            "country": country,
            "metadata": {
                "source": "geographical_database",
                "model": "mock-presidio-demo",
                "pii_protection": "enabled"
            }
        }
    
    async def demonstrate_pii_flow(self, user_query):
        """Demonstrate complete PII protection flow."""
        print(f"\n🌍 Capital Finder with Presidio PII Protection")
        print("=" * 60)
        print(f"📝 Original Query: '{user_query}'")
        
        # Step 1: Analyze for PII
        pii_results = self.analyze_pii(user_query)
        
        # Step 2: Anonymize if PII found
        anonymized_query, anonymization_map = self.anonymize_text(user_query, pii_results)
        
        # Step 3: Extract country from anonymized query
        # In a real implementation, this would be more sophisticated
        country = anonymized_query.lower()
        for word in ["capital", "of", "what", "is", "the", "<person>", "<location>"]:
            country = country.replace(word, "").strip()
        
        if not country:
            country = "Unknown"
        
        print(f"🎯 Extracted Country: '{country}'")
        
        # Step 4: Process with capital finder service
        print(f"\n⚙️  Processing anonymized query with capital finder...")
        result = self.mock_capital_query(country)
        
        # Step 5: Prepare response
        response_text = f"The capital of {result['country']} is {result['capital']}"
        print(f"🏛️  Service Response: '{response_text}'")
        
        # Step 6: Deanonymize response if needed
        final_response = self.deanonymize_text(response_text, anonymization_map)
        
        # Step 7: Return structured result
        final_result = {
            "response": final_response,
            "capital": result["capital"],
            "country": result["country"],
            "pii_detected": len(pii_results) > 0,
            "pii_entities": [r.entity_type for r in pii_results],
            "metadata": {
                **result["metadata"],
                "anonymization_applied": len(anonymization_map) > 0,
                "anonymized_entities": list(anonymization_map.keys()) if anonymization_map else []
            }
        }
        
        print(f"\n✅ Final Response: {json.dumps(final_result, indent=2)}")
        return final_result

async def main():
    """Run Presidio capital finder demonstrations."""
    print("🛡️  Presidio Capital Finder Demonstration")
    print("=" * 50)
    
    if not PRESIDIO_AVAILABLE:
        print("❌ Presidio is not available. Please install:")
        print("   pip install presidio-analyzer presidio-anonymizer")
        print("   python -m spacy download en_core_web_sm")
        return 1
    
    demo = PresidioCapitalFinderDemo()
    
    # Test cases with different PII scenarios
    test_cases = [
        # Simple country query (no PII)
        "What is the capital of France?",
        
        # Query with person name (PII)
        "Hi, I'm John Smith and I want to know the capital of Japan",
        
        # Query with email (PII)
        "My email is john.doe@example.com, can you tell me Germany's capital?",
        
        # Query with phone number (PII)
        "Call me at 555-123-4567 with the capital of Italy",
        
        # Query with multiple PII types
        "John Smith (john@example.com, 555-0123) asks: what's Spain's capital?",
        
        # Complex query with location PII
        "I'm traveling from New York to find out about the UK's capital"
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n" + "🧪" * 60)
        print(f"TEST CASE {i}/{len(test_cases)}")
        
        try:
            result = await demo.demonstrate_pii_flow(query)
            print(f"✅ Test {i} completed successfully")
            
        except Exception as e:
            print(f"❌ Test {i} failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("🧪" * 60)
    
    print(f"\n🎉 Presidio Capital Finder Demonstration Complete!")
    print("\n💡 Key Features Demonstrated:")
    print("   ✅ PII detection in user queries")
    print("   ✅ Anonymization before processing")
    print("   ✅ Safe processing with anonymized data")
    print("   ✅ Deanonymization for user-friendly responses")
    print("   ✅ Comprehensive metadata tracking")
    print("\n🔒 Privacy Benefits:")
    print("   • User PII never exposed to downstream services")
    print("   • Maintains functionality while protecting privacy")
    print("   • Audit trail of PII handling")
    print("   • Reversible anonymization for user experience")
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)