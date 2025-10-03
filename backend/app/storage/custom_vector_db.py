import numpy as np
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any
import math


class HybridVectorDB:
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []
        self.ids = []
        self.word_freq = defaultdict(dict)  
        self.doc_lengths = []
        self.avg_doc_length = 0.0
        self._token_cache = {}
        self._minimum_similarity_threshold = 0.5
    
    def add(self, vectors: List[np.ndarray], texts: List[str], metadatas: List[Dict], ids: List[str]):
        """Add documents with vectors, texts, and metadata"""
        # Normalize vectors before adding
        normalized_vectors = [v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v for v in vectors]
        
        self.vectors.extend(normalized_vectors)
        self.texts.extend(texts)
        self.metadata.extend(metadatas)
        self.ids.extend(ids)
        
        # Build BM25 index
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index for lexical search"""
        self.word_freq = {}
        self.doc_lengths = []
        
        for i, text in enumerate(self.texts):
            words = self._tokenize(text)
            self.doc_lengths.append(len(words))
            
            word_count = Counter(words)
            for word, count in word_count.items():
                if word not in self.word_freq:
                    self.word_freq[word] = {}
                self.word_freq[word][i] = count
        
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        if text in self._token_cache and len(self._token_cache) < 5000:
            self._token_cache[text] = re.findall(r'\b\w+\b', text.lower())
        return self._token_cache.get(text, re.findall(r'\b\w+\b', text.lower()))
    
    def _bm25_score(self, query_words: List[str], doc_idx: int, k1=1.5, b=0.75) -> float:
        """Calculate BM25 score for a document"""
        if doc_idx >= len(self.doc_lengths):
            return 0.0
        
        score = 0.0
        doc_length = self.doc_lengths[doc_idx]
        
        for word in query_words:
            if word in self.word_freq and doc_idx in self.word_freq[word]:
                tf = self.word_freq[word][doc_idx]
                df = len(self.word_freq[word])
                idf = math.log((len(self.texts) - df + 0.5) / (df + 0.5))
                
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / self.avg_doc_length)))
        
        return score
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot_product / norm_product if norm_product > 0 else 0.0
    
    def semantic_search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Semantic search using vector similarity"""
        results = []
        if not self.vectors:
            return results
        
        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm
        
        # Since stored vectors are already normalized, cosine similarity = dot product
        for i, vector in enumerate(self.vectors):
            similarity = float(np.dot(query_vector, vector))
            results.append({
                'id': self.ids[i],
                'score': similarity,
                'text': self.texts[i],
                'metadata': self.metadata[i],
                'type': 'semantic'
            })
        
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        
        return sorted_results
    
    def lexical_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Lexical search using BM25"""
        query_words = self._tokenize(query)
        results = []
        
        for i in range(len(self.texts)):
            score = self._bm25_score(query_words, i)
            if score > 0:
                results.append({
                    'id': self.ids[i],
                    'score': score,
                    'text': self.texts[i],
                    'metadata': self.metadata[i],
                    'type': 'lexical'
                })
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        if sorted_results:
            max_score = sorted_results[0]['score']
            for result in sorted_results:
                result['norm_score'] = result['score'] / max_score if max_score > 0 else 0
            
        return sorted_results
    
    def hybrid_search(self, query: str, query_vector: np.ndarray, 
                     top_k: int = 5, semantic_weight: float = 0.65) -> List[Dict]:
        """Hybrid search combining semantic and lexical results"""
        # Step 1: Get results from both searches
        semantic_results = self.semantic_search(query_vector, top_k)
        lexical_results = self.lexical_search(query, top_k)
        
        for r in semantic_results:
            r['norm_score'] = r['score']
        
        if lexical_results:
            max_lex = max(r['score'] for r in lexical_results)
            for r in lexical_results:
                r['norm_score'] = r['score'] / max_lex if max_lex > 0 else 0
        
        # Step 3: Combine results
        combined = {}
        lexical_weight = 1 - semantic_weight
        
        # Step 4: Add semantic results
        for result in semantic_results:
            doc_id = result['id']
            combined[doc_id] = {
                'id': doc_id,
                'text': result['text'],
                'metadata': result['metadata'],
                'semantic_score': result['norm_score'],
                'lexical_score': 0,
                'combined_score': result['norm_score'] * semantic_weight
            }
        
        # Step 5: Add lexical results
        for result in lexical_results:
            doc_id = result['id']
            if doc_id in combined:
                combined[doc_id]['lexical_score'] = result['norm_score']
                combined[doc_id]['combined_score'] = (
                    combined[doc_id]['semantic_score'] * semantic_weight +
                    result['norm_score'] * lexical_weight
                )
            else:
                combined[doc_id] = {
                    'id': doc_id,
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'semantic_score': 0,
                    'lexical_score': result['norm_score'],
                    'combined_score': result['norm_score'] * lexical_weight
                }
        
        # Step 6: Sort by combined score and return top_k
        final_results = sorted(combined.values(), key=lambda x: x['combined_score'], reverse=True)
        
        # Filter out results below threshold
        final_results = [r for r in final_results if r['combined_score'] >= self._minimum_similarity_threshold]
        
        # If no results meet threshold, return empty list
        if not final_results:
            return []
        
        return final_results[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            'total_documents': len(self.vectors),
            'vocabulary_size': len(self.word_freq),
            'avg_doc_length': round(self.avg_doc_length, 2)
        }
