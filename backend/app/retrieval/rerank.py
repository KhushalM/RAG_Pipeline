from typing import List, Dict, Any
from ..mistral import MistralLLM


class LLMReranker:
    """
    LLM-based reranking for retrieved documents.
    Uses an LLM to score and reorder documents based on relevance to the query.
    """
    
    def __init__(self):
        self.llm = MistralLLM()
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:

        if not documents:
            return []
        
        # Limit documents to avoid too many API calls
        max_docs_to_rerank = 10
        if len(documents) > max_docs_to_rerank:
            documents = documents[:max_docs_to_rerank]
        
        #Step 1: Score documents in batch by LLM between 0.0 and 1.0
        scores = self._batch_score_documents(query, documents)
        
        #Step 2: Attach scores to documents
        scored_documents = []
        for i, doc in enumerate(documents):
            doc_copy = doc.copy()
            doc_copy['llm_score'] = scores[i] if i < len(scores) else 0.5
            scored_documents.append(doc_copy)
        
        #Step 3: Sort by LLM score (higher is better)
        reranked_docs = sorted(scored_documents, key=lambda x: x['llm_score'], reverse=True)
        
        return reranked_docs[:top_k]
    
    def _batch_score_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        if not documents:
            return []
        
        # Create a batch prompt for all documents
        prompt = f"""Rate how relevant each document is to the query on a scale from 0.0 to 1.0.
Return only the scores separated by commas, no other text.

Query: {query}

Documents:"""
        
        for i, doc in enumerate(documents):
            # Truncate document text to avoid token limits
            doc_text = doc['text'][:500] + "..." if len(doc['text']) > 500 else doc['text']
            prompt += f"\n\nDocument {i+1}: {doc_text}"
        
        prompt += f"\n\nScores (comma-separated, {len(documents)} numbers between 0.0 and 1.0):"
        
        response = self.llm.generate_response(
            prompt=prompt,
            temperature=0.1,
            max_tokens=50
        )
        
        return self._extract_batch_scores(response, len(documents))
    
    
    def _extract_batch_scores(self, response: str, expected_count: int) -> List[float]:
        """Extract multiple numerical scores from LLM response."""
        import re
        
        response = response.strip()
        # Find all numbers in the response
        score_matches = re.findall(r'(\d*\.?\d+)', response)
        
        scores = []
        for match in score_matches:
            score = float(match)
            scores.append(max(0.0, min(1.0, score)))
        
        # Pad with default scores if we don't have enough
        while len(scores) < expected_count:
            scores.append(0.5)
        
        # Truncate if we have too many
        return scores[:expected_count]
    
