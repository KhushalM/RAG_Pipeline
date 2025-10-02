import os
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

mistral = Mistral(api_key=os.getenv("mistral_api_key"))

class MistralEmbeddings:
    def __init__(self):
        self.api_key = os.getenv("mistral_api_key")
        self.client = Mistral(api_key=self.api_key)
        self.model = 'mistral-embed'

    def embed_documents(self, documents):
        """
        Embedding all chunked documents into a list of embeddings using Mistral Embeddings API
        """
        embeddings = []
        batch_size = 32
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            embeddings.extend(self.client.embeddings.create(model=self.model, inputs=batch).data)

        return embeddings
    
    def embed_query(self, query):
        """
        Embedding a single query into an embedding using Mistral Embeddings API
        """
        response = self.client.embeddings.create(model=self.model, inputs=[query])
        return response.data[0]

class MistralLLM:
    def __init__(self):
        self.api_key = os.getenv("mistral_api_key")
        self.client = Mistral(api_key=self.api_key)
        self.model = 'mistral-large-latest'

    # async def generate_response(self, prompt) -> str:
    #     """
    #     Generating a response using Mistral LLM API
    #     """
    #     message = [
    #         {
    #             "role": "user",
    #             "content": prompt
    #         }
    #     ]
    #     response = await self.client.chat.stream_async(model=self.model, messages=message[ChatMessage(role="user", content=prompt)])
    #     return response.choices[0].message.content