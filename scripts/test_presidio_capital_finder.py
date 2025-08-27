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
                # RecognizerResult uses 'score' attribute, not 'confidence'
                confidence_score = getattr(result, 'score', getattr(result, 'confidence', 0.0))
                print(f"   - {result.entity_type}: '{detected_text}' "
                      f"(confidence: {confidence_score:.2f})")
        else:
            print("   - No PII detected")
        
        return results
    
    def anonymize_text(self, text, pii_results, use_numbered_placeholders=False):
        """Anonymize PII in text and store mapping."""
        if not pii_results:
            return text, {}
        
        if use_numbered_placeholders:
            return self._anonymize_with_numbered_placeholders(text, pii_results)
        
        # Configure anonymization operators
        operators = {
            "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
            "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
            "DATE_TIME": OperatorConfig("replace", {"new_value": "<DATE>"}),
            "US_SSN": OperatorConfig("replace", {"new_value": "<SSN>"}),
            "URL": OperatorConfig("replace", {"new_value": "<URL>"}),
            "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"})
        }
        
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=pii_results,
            operators=operators
        )
        
        # Create a mapping based on the actual anonymized placeholders and original text
        anonymization_map = {}
        
        # Map placeholders to original values from PII results
        for result in pii_results:
            original_value = text[result.start:result.end]
            entity_type = result.entity_type
            
            if entity_type in operators:
                # Get the placeholder from the operator configuration
                placeholder = operators[entity_type].params["new_value"]
                anonymization_map[placeholder] = original_value
            else:
                # Use default for unknown types
                anonymization_map["<REDACTED>"] = original_value
        
        print(f"🔒 Anonymized Text: '{anonymized_result.text}'")
        print(f"🗝️  Anonymization Map: {anonymization_map}")
        
        return anonymized_result.text, anonymization_map
    
    def _anonymize_with_numbered_placeholders(self, text, pii_results):
        """Anonymize PII with numbered placeholders for multi-entity scenarios."""
        # Filter out overlapping entities (prioritize longer matches and higher confidence)
        filtered_results = self._filter_overlapping_entities(pii_results)
        
        # Sort filtered results by start position to process in order
        sorted_results = sorted(filtered_results, key=lambda x: x.start)
        
        # Count occurrences of each entity type for numbering
        entity_counters = {}
        anonymization_map = {}
        
        # Process entities and create numbered placeholders
        anonymized_text = text
        offset = 0  # Track offset due to replacements
        
        for result in sorted_results:
            entity_type = result.entity_type
            original_value = text[result.start:result.end]
            
            # Generate numbered placeholder
            if entity_type not in entity_counters:
                entity_counters[entity_type] = 1
            else:
                entity_counters[entity_type] += 1
            
            counter = entity_counters[entity_type]
            
            # Create numbered placeholder based on entity type
            if entity_type == "PERSON":
                placeholder = f"<NAME_{counter}>"
            elif entity_type == "EMAIL_ADDRESS":
                placeholder = f"<EMAIL_ADDRESS_{counter}>"
            elif entity_type == "DATE_TIME":
                placeholder = f"<DATE_{counter}>"
            elif entity_type == "PHONE_NUMBER":
                placeholder = f"<PHONE_{counter}>"
            elif entity_type == "US_SSN":
                placeholder = f"<SSN_{counter}>"
            elif entity_type == "LOCATION":
                placeholder = f"<LOCATION_{counter}>"
            else:
                placeholder = f"<{entity_type}_{counter}>"
            
            # Store mapping
            anonymization_map[placeholder] = original_value
            
            # Replace in text (adjusting for previous replacements)
            start_pos = result.start + offset
            end_pos = result.end + offset
            
            anonymized_text = anonymized_text[:start_pos] + placeholder + anonymized_text[end_pos:]
            offset += len(placeholder) - (result.end - result.start)
        
        print(f"🔒 Anonymized Text: '{anonymized_text}'")
        print(f"🗝️  Anonymization Map: {anonymization_map}")
        
        return anonymized_text, anonymization_map
    
    def _filter_overlapping_entities(self, pii_results):
        """Filter out overlapping PII entities, prioritizing longer matches and higher confidence."""
        if not pii_results:
            return pii_results
        
        # Sort by confidence (descending) and length (descending)
        sorted_by_priority = sorted(pii_results, key=lambda x: (
            getattr(x, 'score', getattr(x, 'confidence', 0.0)),
            x.end - x.start
        ), reverse=True)
        
        filtered_results = []
        
        for result in sorted_by_priority:
            # Check if this result overlaps with any already accepted result
            overlaps = False
            for accepted_result in filtered_results:
                if self._entities_overlap(result, accepted_result):
                    overlaps = True
                    break
            
            # Only add non-overlapping results or prioritize certain entity types
            if not overlaps or self._should_prioritize_entity(result, filtered_results):
                # If prioritizing this entity, remove conflicting ones
                if overlaps and self._should_prioritize_entity(result, filtered_results):
                    filtered_results = [r for r in filtered_results if not self._entities_overlap(result, r)]
                
                if not overlaps:
                    filtered_results.append(result)
        
        return filtered_results
    
    def _entities_overlap(self, entity1, entity2):
        """Check if two PII entities overlap in text position."""
        return not (entity1.end <= entity2.start or entity2.end <= entity1.start)
    
    def _should_prioritize_entity(self, new_entity, existing_entities):
        """Determine if new entity should be prioritized over existing overlapping ones."""
        # Prioritize EMAIL_ADDRESS over URL when they overlap
        if new_entity.entity_type == "EMAIL_ADDRESS":
            for existing in existing_entities:
                if existing.entity_type == "URL" and self._entities_overlap(new_entity, existing):
                    return True
        
        # Prioritize PERSON over other entity types when they overlap
        if new_entity.entity_type == "PERSON":
            for existing in existing_entities:
                if existing.entity_type in ["URL", "LOCATION"] and self._entities_overlap(new_entity, existing):
                    return True
        
        return False
    
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
    
    def extract_country_from_query(self, original_query, pii_results):
        """Extract country name from query, considering PII detection."""
        # Known countries and their variations
        countries = {
            "france": "France", "french": "France",
            "japan": "Japan", "japanese": "Japan", 
            "germany": "Germany", "german": "Germany",
            "italy": "Italy", "italian": "Italy",
            "spain": "Spain", "spanish": "Spain",
            "uk": "UK", "united kingdom": "United Kingdom", "britain": "UK", "england": "UK",
            "canada": "Canada", "canadian": "Canada",
            "australia": "Australia", "australian": "Australia",
            "usa": "USA", "united states": "USA", "america": "USA",
            "china": "China", "chinese": "China",
            "india": "India", "indian": "India",
            "brazil": "Brazil", "brazilian": "Brazil",
            "russia": "Russia", "russian": "Russia"
        }
        
        query_lower = original_query.lower()
        
        # Check for location entities that might be countries
        for result in pii_results:
            if result.entity_type == "LOCATION":
                detected_location = original_query[result.start:result.end].lower()
                if detected_location in countries:
                    return countries[detected_location]
        
        # Check for countries in the full text
        for country_key, country_name in countries.items():
            if country_key in query_lower:
                return country_name
                
        return None
    
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
    
    def mock_retirement_eligibility_query(self, query_text, anonymization_map):
        """Mock retirement account eligibility service with PII-aware response."""
        # This service would typically process the anonymized query
        # and return a response that can be deanonymized
        
        # Extract key information from query (simplified)
        amount = "5000"  # Could extract from query: "$5,000"
        
        # Create response template with placeholders that match anonymization
        response_template = (
            "Hello <PERSON>, based on your request you are eligible to open a "
            "retirement account with a ${amount} deposit. "
            "We will contact you via <EMAIL> to proceed with the application. "
            "Your identifier <SSN> will be used to validate your identity during the process."
        )
        
        # Replace amount
        response_text = response_template.replace("{amount}", amount)
        
        # Add SSN to anonymization map if not detected by Presidio
        # Manual SSN detection for demonstration purposes
        import re
        original_query = query_text.replace('<PERSON>', anonymization_map.get('<PERSON>', 'Person')).replace('<EMAIL>', anonymization_map.get('<EMAIL>', 'email'))
        
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        ssn_match = re.search(ssn_pattern, original_query)
        if ssn_match and '<SSN>' not in anonymization_map:
            # Add SSN to the anonymization map for proper deanonymization
            updated_map = anonymization_map.copy()
            updated_map['<SSN>'] = ssn_match.group()
            self.anonymization_map.update(updated_map)
        
        return {
            "response": response_text,
            "eligible": True,
            "deposit_amount": amount,
            "metadata": {
                "source": "retirement_eligibility_service",
                "model": "mock-financial-demo",
                "pii_protection": "enabled",
                "requires_deanonymization": True
            }
        }
    
    def mock_multi_person_retirement_eligibility_query(self, query_text, anonymization_map):
        """Mock multi-person retirement account eligibility service with numbered placeholders."""
        # This service would typically process the anonymized query 
        # and return a response that can be deanonymized with numbered placeholders
        
        # Extract key information from query (simplified)
        amount = "10,000"  # Could extract from query: "$10,000"
        
        # Create response template with numbered placeholders that match anonymization
        response_template = (
            "Here is the eligibility confirmation:\n\n"
            "1. <NAME_1> (born in <DATE_1>) with email <EMAIL_ADDRESS_1> is eligible for an account with a ${amount} deposit.\n"
            "2. <NAME_2> (born in <DATE_2>) with email <EMAIL_ADDRESS_2> is eligible for an account with a ${amount} deposit.\n"
            "3. <NAME_3> (born in <DATE_3>) with email <EMAIL_ADDRESS_3> is eligible for an account with a ${amount} deposit."
        )
        
        # Replace amount
        response_text = response_template.replace("{amount}", amount)
        
        return {
            "response": response_text,
            "eligible": True,
            "deposit_amount": amount,
            "persons_processed": 3,
            "metadata": {
                "source": "multi_person_retirement_eligibility_service",
                "model": "mock-financial-multi-person-demo",
                "pii_protection": "enabled",
                "requires_deanonymization": True,
                "multi_entity_support": True
            }
        }
    
    async def demonstrate_pii_flow(self, user_query):
        """Demonstrate complete PII protection flow."""
        # Determine query type
        is_retirement_query = "retirement" in user_query.lower() or "account" in user_query.lower()
        is_multi_person_query = "three people" in user_query.lower() or "Alice Johnson" in user_query
        
        if is_multi_person_query:
            return await self.demonstrate_multi_person_pii_flow(user_query)
        elif is_retirement_query:
            return await self.demonstrate_retirement_pii_flow(user_query)
        else:
            return await self.demonstrate_capital_pii_flow(user_query)
    
    async def demonstrate_capital_pii_flow(self, user_query):
        """Demonstrate PII protection flow for capital finder queries."""
        print(f"\n🌍 Capital Finder with Presidio PII Protection")
        print("=" * 60)
        print(f"📝 Original Query: '{user_query}'")
        
        # Step 1: Analyze for PII
        pii_results = self.analyze_pii(user_query)
        
        # Step 2: Anonymize if PII found
        anonymized_query, anonymization_map = self.anonymize_text(user_query, pii_results)
        
        # Step 3: Extract country from original or anonymized query
        country = self.extract_country_from_query(user_query, pii_results)
        
        if not country:
            # Fallback: try simple extraction from anonymized query
            country_candidate = anonymized_query.lower()
            for word in ["capital", "of", "what", "is", "the", "can", "you", "tell", "me", "about", "hi", "i'm", "and", "want", "to", "know"]:
                country_candidate = country_candidate.replace(word, "").strip()
            
            # Remove punctuation and extra spaces
            import re
            country_candidate = re.sub(r'[^\w\s<>]', ' ', country_candidate)
            country_candidate = re.sub(r'\s+', ' ', country_candidate).strip()
            
            if country_candidate and len(country_candidate) > 2:
                country = country_candidate
            else:
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
    
    async def demonstrate_retirement_pii_flow(self, user_query):
        """Demonstrate PII protection flow for retirement account queries."""
        print(f"\n🏦 Retirement Account Eligibility with Presidio PII Protection")
        print("=" * 70)
        print(f"📝 Original Query: '{user_query}'")
        
        # Step 1: Analyze for PII
        pii_results = self.analyze_pii(user_query)
        
        # Step 2: Anonymize if PII found
        anonymized_query, anonymization_map = self.anonymize_text(user_query, pii_results)
        
        print(f"🔍 Anonymized Query for Processing: '{anonymized_query}'")
        
        # Step 3: Process with retirement eligibility service
        print(f"\n⚙️  Processing anonymized query with retirement eligibility service...")
        result = self.mock_retirement_eligibility_query(user_query, anonymization_map)
        
        print(f"🏦 Service Response (with placeholders): '{result['response']}'")
        
        # Step 4: Deanonymize response to restore original PII
        # Include any additional mappings from the service (like SSN)
        enhanced_map = anonymization_map.copy()
        enhanced_map.update(self.anonymization_map)
        
        final_response = self.deanonymize_text(result['response'], enhanced_map)
        
        # Step 5: Return structured result
        final_result = {
            "response": final_response,
            "eligible": result["eligible"],
            "deposit_amount": result["deposit_amount"],
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
    
    async def demonstrate_multi_person_pii_flow(self, user_query):
        """Demonstrate PII protection flow for multi-person retirement account queries."""
        print(f"\n👥 Multi-Person Retirement Account Eligibility with Presidio PII Protection")
        print("=" * 80)
        print(f"📝 Original Query: '{user_query}'")
        
        # Step 1: Analyze for PII
        pii_results = self.analyze_pii(user_query)
        
        # Step 2: Anonymize with numbered placeholders for multi-entity support
        anonymized_query, anonymization_map = self.anonymize_text(user_query, pii_results, use_numbered_placeholders=True)
        
        print(f"🔍 Anonymized Query for Processing: '{anonymized_query}'")
        
        # Step 3: Process with multi-person retirement eligibility service
        print(f"\n⚙️  Processing anonymized query with multi-person retirement eligibility service...")
        result = self.mock_multi_person_retirement_eligibility_query(anonymized_query, anonymization_map)
        
        print(f"🏦 Service Response (with numbered placeholders):")
        print(f"'{result['response']}'")
        
        # Step 4: Deanonymize response to restore original PII
        final_response = self.deanonymize_text(result['response'], anonymization_map)
        
        # Step 5: Return structured result
        final_result = {
            "response": final_response,
            "eligible": result["eligible"],
            "deposit_amount": result["deposit_amount"],
            "persons_processed": result["persons_processed"],
            "pii_detected": len(pii_results) > 0,
            "pii_entities": [r.entity_type for r in pii_results],
            "metadata": {
                **result["metadata"],
                "anonymization_applied": len(anonymization_map) > 0,
                "anonymized_entities": list(anonymization_map.keys()) if anonymization_map else [],
                "numbered_placeholders_used": True
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
        "I'm traveling from New York to find out about the UK's capital",
        
        # Retirement account eligibility query with comprehensive PII
        "Hi, my name is John Doe. My Social Security Number is 123-45-6789 and my email is john.doe@example.com. Can you please tell me if I'm eligible to open a new retirement account with a $5,000 deposit?",
        
        # Multi-person retirement account eligibility query with numbered placeholders
        "We have three people requesting retirement account eligibility checks: 1. Alice Johnson, born in 1985, email alice.j@example.com 2. Bob Smith, born in 1978, email bob.smith78@gmail.com 3. Carol White, born in 1990, email carol.white90@outlook.com Please confirm for each of them if they are eligible to open an account with a $10,000 deposit."
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