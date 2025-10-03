from ..mistral import MistralLLM
from typing import List, Dict, Optional

class QueryTransformation:
    def __init__(self):
        self.mistral = MistralLLM(model="mistral-small-2503")

    def transform_query(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Transform the query to a more specific query, using conversation history for context
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
        
        transformation_prompt = f"""You are a query/prompt optimization assistant to help with the RAG pipeline.
{context_section}
        Transform this query to make it more effective for searching through documents:
        - Add relevant keywords and synonyms
        - Make implicit concepts explicit
        - Expand abbreviations and acronyms
        - Rephrase for better semantic matching
        - Keep the core intent intact
        - Keep the query as concise as possible
        - Also mention if the query intends a list, comparison, definition, or steps
        - If there is conversation history above, use it to resolve pronouns (like "it", "that", "this") and add missing context

        Original Query: "{query}"

        Examples:
        - "How does it work?" -> "How does this system work? What is the purpose of this system?"
        - "What are ML models?" -> "What are machine learning models? What is the purpose of machine learning models?"
        - With context about "Python" previously discussed: "What about its performance?" -> "What about Python's performance? How does Python perform in terms of speed and efficiency?"

        Provide only the transformed query, no other text.
        """

        response = self.mistral.client.chat.complete(
            model=self.mistral.model,
            messages=[
                {
                    "role": "user",
                    "content": transformation_prompt
                }
            ]
        )
        return str(response.choices[0].message.content)