from ..mistral import MistralLLM
from typing import Tuple, Optional, List, Dict
import json

class QueryRouter:
    def __init__(self):
        self.mistral = MistralLLM(model="mistral-small-2503")

    def analyze_and_transform(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Tuple[bool, str]:
        """
        Determine if retrieval is needed AND transform the query in one call
        
        Returns:
            Tuple[bool, str]: (needs_retrieval, transformed_query)
        """
        # Build conversation context if available
        context_section = ""
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-6:]
            context_lines = []
            for msg in recent_history:
                role = msg["role"].capitalize()
                context_lines.append(f"{role}: {msg['content']}")
            context_section = f"""
                Previous Conversation:
                {chr(10).join(context_lines)}
                """
                        
        router_prompt = f"""You are a query analysis assistant for a RAG pipeline.

                        {context_section}
                        User Query: "{query}"

                        Analyze this query and provide two pieces of information:

                        1. **Needs Retrieval**: Does this query require searching through a knowledge base?
                        - Return TRUE if the query asks about specific facts, documents, or information that would be in a knowledge base
                        - Return FALSE for general conversation, greetings, or questions that can be answered with general knowledge
                        
                        2. **Transformed Query**: If retrieval is needed, optimize the query for semantic search:
                        - Add relevant keywords and synonyms
                        - Make implicit concepts explicit
                        - Expand abbreviations and acronyms
                        - If there's conversation history, resolve pronouns and add context
                        - **Detect output format intent and append formatting instructions:**
                          * If asking for multiple items/benefits/reasons → append "Provide as a numbered list"
                          * If asking for steps/process/how-to → append "Provide as step-by-step instructions"
                          * If asking for definition/meaning → append "Provide definition with key details"
                          * If asking to compare/contrast → append "Provide structured comparison"
                          * Otherwise keep general format
                        - Keep it concise but comprehensive
                        - If retrieval is NOT needed, just return the original query

                        Return your response in this EXACT JSON format:
                        {{
                            "needs_retrieval": true/false,
                            "transformed_query": "your optimized query here"
                        }}

                        Examples:

                        Query: "Hello, how are you?"
                        {{
                            "needs_retrieval": false,
                            "transformed_query": "Hello, how are you?"
                        }}

                        Query: "What did Steve Jobs say about connecting the dots?"
                        {{
                            "needs_retrieval": true,
                            "transformed_query": "What did Steve Jobs say about connecting the dots? Steve Jobs philosophy on connecting dots in retrospect"
                        }}

                        Query: "Tell me more about it" (with context about Apple in conversation history)
                        {{
                            "needs_retrieval": true,
                            "transformed_query": "Tell me more about Apple Inc company history and products"
                        }}

                        Query: "What are the main benefits of machine learning?"
                        {{
                            "needs_retrieval": true,
                            "transformed_query": "What are the main benefits advantages of machine learning ML artificial intelligence? Provide as a numbered list"
                        }}

                        Query: "How do I train a neural network?"
                        {{
                            "needs_retrieval": true,
                            "transformed_query": "How do I train a neural network deep learning model? Provide as step-by-step instructions"
                        }}

                        Now analyze the user query above and return ONLY the JSON, no other text."""

        response = self.mistral.client.chat.complete(
            model=self.mistral.model,
            messages=[
                {
                    "role": "user",
                    "content": router_prompt
                }
            ]
        )
        
        response_text = str(response.choices[0].message.content).strip()
        
        # Parse JSON response
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        result = json.loads(response_text)
        needs_retrieval = result.get("needs_retrieval", True)
        transformed_query = result.get("transformed_query", query)
        
        print(f"Needs retrieval: {needs_retrieval}")
        print(f"Transformed query: {transformed_query}")
        
        return needs_retrieval, transformed_query
