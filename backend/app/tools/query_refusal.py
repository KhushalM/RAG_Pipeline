from ..mistral import MistralLLM

class QueryRefusal:
    def __init__(self):
        self.mistral = MistralLLM(model="mistral-small-2503")

    def should_refuse_query(self, query: str) -> tuple[str, str]:
        """
        Determine if the query should be refused or needs a disclaimer due to:
        1. PII (Personally Identifiable Information) - HARD REFUSAL
        2. Legal advice requests - DISCLAIMER
        3. Medical advice requests - DISCLAIMER
        
        Returns:
            tuple[str, str]: (action, message)
            action: "REFUSE" | "DISCLAIMER" | "ALLOW"
            message: refusal/disclaimer text or empty string
        """
        refusal_check_prompt = f"""You are a query safety classifier. Analyze the following query and determine if it should be refused.

A query should be REFUSED if it:
1. Contains or requests PII (Personally Identifiable Information) such as:
   - Social Security Numbers (SSN)
   - Credit card numbers
   - Bank account numbers
   - Passport numbers
   - Driver's license numbers
   - Any sensitive personal identification data
   
2. Requests legal advice or legal opinions such as:
   - Contract interpretation
   - Legal rights and obligations
   - Lawsuit guidance
   - Legal strategy or representation
   - Interpretation of laws or regulations
   
3. Requests medical advice or diagnosis such as:
   - Symptom diagnosis
   - Treatment recommendations
   - Medication advice
   - Medical procedure guidance
   - Health condition assessment

Query: "{query}"

Respond ONLY with one of these exact formats:
- If query contains PII: "REFUSE: PII"
- If query requests legal advice/opinion: "DISCLAIMER: LEGAL"
- If query requests medical advice/diagnosis: "DISCLAIMER: MEDICAL"
- If query is acceptable: "ALLOW"

Examples:
- "What is my SSN 123-45-6789?" -> "REFUSE: PII"
- "My credit card is 4532-1234-5678-9012" -> "REFUSE: PII"
- "Can I sue my landlord?" -> "DISCLAIMER: LEGAL"
- "Should I sign this contract?" -> "DISCLAIMER: LEGAL"
- "What is contract law?" -> "ALLOW"
- "I have a headache, what medication should I take?" -> "DISCLAIMER: MEDICAL"
- "Should I get this surgery?" -> "DISCLAIMER: MEDICAL"
- "What are the main causes of headaches?" -> "ALLOW"
"""

        response = self.mistral.client.chat.complete(
            model=self.mistral.model,
            messages=[
                {
                    "role": "user",
                    "content": refusal_check_prompt
                }
            ]
        )
        
        result = str(response.choices[0].message.content).strip()
        print(f"Refusal check result: {result}")
        
        
        if "refuse" in result.lower():
            category = "PII"
            refusal_message = self._get_refusal_message(category)
            return "REFUSE", refusal_message
        elif "disclaimer" in result.lower():
            category = "LEGAL" if "legal" in result.lower() else "MEDICAL"
            disclaimer_message = self._get_disclaimer_message(category)
            return "DISCLAIMER", disclaimer_message
        
        return "ALLOW", ""

    def _get_refusal_message(self, category: str) -> str:
        """Generate appropriate refusal message for hard refusals (PII only)"""
        messages = {
            "PII": "I cannot process queries containing personally identifiable information (PII) such as social security numbers, credit card numbers, or other sensitive personal data. Please remove any PII from your query and try again."
        }
        
        return messages.get(category, "I cannot process queries containing personally identifiable information (PII) such as social security numbers, credit card numbers, or other sensitive personal data. Please remove any PII from your query and try again.")
    
    def _get_disclaimer_message(self, category: str) -> str:
        """Generate appropriate disclaimer message for legal/medical queries"""
        disclaimers = {
            "LEGAL": "LEGAL DISCLAIMER: This information is for general educational purposes only and does not constitute legal advice. Laws vary by jurisdiction and individual circumstances differ. For specific legal guidance, please consult with a qualified attorney licensed in your area.",
            "MEDICAL": "MEDICAL DISCLAIMER: This information is for general educational purposes only and does not constitute medical advice, diagnosis, or treatment recommendations. For medical concerns, symptoms, or health-related questions, please consult with a qualified healthcare professional or your doctor."
        }
        
        return disclaimers.get(category, "DISCLAIMER: The following information is for general educational purposes only.")
