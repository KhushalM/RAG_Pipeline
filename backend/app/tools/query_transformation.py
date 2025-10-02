from ..mistral import MistralLLM

class QueryTransformation:
    def __init__(self):
        self.mistral = MistralLLM()

    def transform_query(self, query: str) -> str:
        """
        Transform the query to a more specific query
        """
        transformation_prompt = f"""You are a query/prompt optimization assistant to help with the RAG pipeline.

        Transform this query to make it more effective for searching through documents:
        - Add relevant keywords and synonyms
        - Make implicit concepts explicit
        - Expand abbreviations and acronyms
        - Rephrase for better semantic matching
        - Keep the core intent intact

        Original Query: "{query}"

        Examples:
        - "How does it work?" -> "How does this system work? What is the purpose of this system?"
        - "What are ML models?" -> "What are machine learning models? What is the purpose of machine learning models?"

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
        return response.choices[0].message.content