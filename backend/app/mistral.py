import os
from typing import List, Dict
from dotenv import load_dotenv
from mistralai import Mistral
import time

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
        print(f"mistral_api_key: {os.getenv('mistral_api_key')}")
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
    def __init__(self, model: str = 'mistral-large-2411'):
        self.api_key = os.getenv("mistral_api_key")
        self.client = Mistral(api_key=self.api_key)
        self.model = model        

    def generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """
        Generate a response using Mistral LLM API
        """
        messages = [
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        response = self.client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def create_rag_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Create a well-structured RAG prompt
        """
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            source_info = f"[Source {i}: {chunk['metadata']['filename']}]"
            context_text += f"{source_info}\n{chunk['text']}\n\n"
        
        prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context documents. 

Context Documents:
{context_text}

User Question: {query}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so clearly
- Cite specific sources when making claims
- Be concise but comprehensive
- If asked about something not in the context, acknowledge the limitation

Answer:"""
        
        return prompt