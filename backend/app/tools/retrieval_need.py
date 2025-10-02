from ..mistral import MistralLLM

class RetrievalNeed:
    def __init__(self):
        self.mistral = MistralLLM()

    def need_to_retrieve(self, prompt: str) -> bool:
        """
        Determine if the prompt requires retrieval
        """
        prompt = f"""
        Determine if the prompt requires retrieval from an external knowledge base.
        If the questions is basic and straightforward, return "false".
        Return only "true" or "false".

        Prompt: {prompt}
        """

        message = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = self.mistral.client.chat.complete(
            model=self.mistral.model,
            messages=message
        )
        return response.choices[0].message.content == "true" or response.choices[0].message.content == "True" or response.choices[0].message.content == "TRUE"
        