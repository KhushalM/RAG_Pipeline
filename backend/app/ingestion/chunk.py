from ..mistral import MistralEmbeddings
import numpy as np
import re

#Could implement a dynamic threshold based on the text length 
class SemanticChunks:
    def __init__(self, similarity_threshold=0.7, min_chunk_size=100, max_chunk_size=2000):
        self.embeddings = MistralEmbeddings()
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def sentence_chunks(self, text) -> list[str]:
        """
        Splitting text into sentences and returning them as a list.
        """
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
        
        print(f"Found {len(sentences)} sentences")

        if not sentences:
            return []
        elif len(sentences) == 1:
            return sentences
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed max size
            test_chunk = current_chunk + (" " + sentence if current_chunk else sentence)
            
            if len(test_chunk) <= self.max_chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        print(f"Grouped into {len(chunks)} size-based chunks")
        return chunks

    def cosine_similarity(self, vector1, vector2) -> float:
        """
        Calculating cosine similarity between vectors to find semantic similarity.
        Returns similarity score (higher = more similar)
        """
        dot_product = np.dot(vector1, vector2)
        norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        
        if norm_product == 0:
            return 0.0
        
        return dot_product / norm_product

    def chunking(self, text) -> list[str]:
        """
        Semantically chunking pdf paragraphs returning chunks.
        """
        if not text or not text.strip():
            return []
        
        #Step 1: We split into paragraphs
        sentences = self.sentence_chunks(text)
        print(f"Number of sentences: {len(sentences)}")
        if not sentences:
            return []
        elif len(sentences) == 1:
            paras = self.filter_chunks(sentences)
            print(f"Number of paragraphs after filtering: {len(paras)}")
            return paras
        
        #Step 2: We embed each paragraph
        embeddings = self.embeddings.embed_documents(sentences)
        vectors = [np.array(emb.embedding) for emb in embeddings]
        
        #Step 3: We calculate cosine similarity between paragraphs
        similarity_scores = []
        for i in range(len(vectors)-1):
           similarity = self.cosine_similarity(vectors[i], vectors[i+1])
           similarity_scores.append(similarity)
        
        #Step4: We create chunks based on the similarity scores (Using sliding window approach)
        chunks = []
        current_paragraph = sentences[0]
        for i in range(1, len(sentences)):
            next_paragraph = sentences[i]
            similarity = similarity_scores[i-1]
            combined_size = len(current_paragraph) + len(next_paragraph) + 2
            if similarity > self.similarity_threshold and combined_size <= self.max_chunk_size:
                current_paragraph += "\n\n" + next_paragraph
            else:
                chunks.append(current_paragraph)
                current_paragraph = next_paragraph
        
        chunks.append(current_paragraph)

        #Step 5: We filter out chunks that are too small or too large
        final_chunks = self.filter_chunks(chunks)
        return final_chunks

    def filter_chunks(self, chunks) -> list[str]:
        """
        Filtering out and adjusting chunks that are too small or too large
        """
        merged_chunks = []
        i = 0

        while i < len(chunks):
            chunk = chunks[i]
            while len(chunk) < self.min_chunk_size and i + 1 < len(chunks) and len(chunk+ "\n\n" + chunks[i+1]) <= self.max_chunk_size:
                i += 1
                chunk += "\n\n" + chunks[i]
            merged_chunks.append(chunk)
            i += 1
        
        #Split larger paragraphs
        final_chunks = []
        for chunk in merged_chunks:
            if len(chunk) <= self.max_chunk_size:
                final_chunks.append(chunk)
            else:
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 2 <= self.max_chunk_size:
                        current_chunk += " " + sentence
                    else:
                        if current_chunk:
                            final_chunks.append(current_chunk.strip())
                        current_chunk = " " + sentence 
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
        return final_chunks
    
    def get_chunk_info(self, chunks) -> dict:
        """Get information about the chunks"""
        if not chunks:
            return {
                "total_chunks": 0,
                "average_size": 0,
                "min_size": 0,
                "max_size": 0
            }
        
        sizes = [len(chunk) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "average_size": sum(sizes) // len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "total_characters": sum(sizes)
        }