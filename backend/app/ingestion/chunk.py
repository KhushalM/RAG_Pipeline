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

    def paragraph_chunks(self, text) -> list[str]:
        """
        Splitting text into paragraphs and returning them as a list.
        """
        # Split by double newlines and filter out empty paragraphs
        paragraphs = re.split(r'\n\n+', text.strip())
        return [p.strip() for p in paragraphs if p.strip()]

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

    def should_merge_paragraphs(self, para1, para2) -> bool:
        """
        Determine if two paragraphs should be merged based on semantic similarity
        """
        # Skip embedding for very short paragraphs
        if len(para1) < 50 or len(para2) < 50:
            return True
        
        # Get embeddings for both paragraphs
        embeddings = self.embeddings.embed_documents([para1, para2])
        
        if len(embeddings) < 2:
            return False
            
        # Extract embedding vectors
        vec1 = np.array(embeddings[0].embedding)
        vec2 = np.array(embeddings[1].embedding)
        
        # Calculate similarity
        similarity = self.cosine_similarity(vec1, vec2)
        
        return similarity >= self.similarity_threshold

    def merge_small_chunks(self, chunks) -> list[str]:
        """
        Merge chunks that are too small with adjacent chunks
        """
        if not chunks:
            return []
        
        merged_chunks = []
        current_chunk = chunks[0]
        
        for i in range(1, len(chunks)):
            next_chunk = chunks[i]
            
            # If current chunk is too small, try to merge with next
            if len(current_chunk) < self.min_chunk_size:
                combined = current_chunk + "\n\n" + next_chunk
                if len(combined) <= self.max_chunk_size:
                    current_chunk = combined
                    continue
            
            # If next chunk is too small, merge it with current
            elif len(next_chunk) < self.min_chunk_size:
                combined = current_chunk + "\n\n" + next_chunk
                if len(combined) <= self.max_chunk_size:
                    current_chunk = combined
                    continue
            
            # Add current chunk and move to next
            merged_chunks.append(current_chunk)
            current_chunk = next_chunk
        
        # Add the last chunk
        merged_chunks.append(current_chunk)
        
        return merged_chunks

    def split_large_chunks(self, chunks) -> list[str]:
        """
        Split chunks that are too large
        """
        final_chunks = []
        
        for chunk in chunks:
            if len(chunk) <= self.max_chunk_size:
                final_chunks.append(chunk)
            else:
                # Split large chunk by sentences
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk + sentence) <= self.max_chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            final_chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
        
        return final_chunks
    
    def chunking(self, text) -> list[str]:
        """
        Semantic paragraph-based chunking with similarity merging
        """
        if not text or not text.strip():
            return []
        
        # Step 1: Split into paragraphs
        paragraphs = self.paragraph_chunks(text)
        
        if not paragraphs:
            return []
        
        # Step 2: Merge semantically similar adjacent paragraphs
        semantic_chunks = []
        current_chunk = paragraphs[0]
        
        for i in range(1, len(paragraphs)):
            next_paragraph = paragraphs[i]
            
            # Check if we should merge based on semantic similarity
            should_merge = self.should_merge_paragraphs(current_chunk, next_paragraph)
            
            # Also check size constraints
            combined_size = len(current_chunk) + len(next_paragraph) + 2  # +2 for \n\n
            
            if should_merge and combined_size <= self.max_chunk_size:
                current_chunk += "\n\n" + next_paragraph
            else:
                semantic_chunks.append(current_chunk)
                current_chunk = next_paragraph
        
        # Add the last chunk
        semantic_chunks.append(current_chunk)
        
        # Step 3: Handle size constraints
        merged_chunks = self.merge_small_chunks(semantic_chunks)
        final_chunks = self.split_large_chunks(merged_chunks)
        
        # Filter out empty chunks
        return [chunk.strip() for chunk in final_chunks if chunk.strip()]

    def get_chunk_info(self, chunks) -> dict:
        """
        Get information about the chunks
        """
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