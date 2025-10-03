import re
from typing import List, Dict, Any
from ..mistral import MistralLLM

class HallucinationCheck:
    def __init__(self):
        self.mistral = MistralLLM(model="mistral-medium")
        self.confidence_threshold = 0.5
    
    def _sentence_chunks(self, text) -> list[str]:
        """
        Splitting text into sentences and returning them as a list.
        """
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
        return sentences
    
    def _combine_context_chunks(self, context_chunks: List[Dict]) -> str:
        """
        Compare the sentence with the context chunks and return the most similar context chunk
        """
        return "\n\n".join([chunk.get('text', '') for chunk in context_chunks])
    
    def _verify_sentence(self, claimed_sentence: str, combined_context_chunks: str, query: str) -> tuple[bool, float]:
        """
        Verify the sentence against the combined context chunks using a LLM
        """
        prompt = f"""You are an evidence verification assistant. Your task is to determine if a claim is supported by the provided context.

                Context:
                {combined_context_chunks}

                Claim to verify: "{claimed_sentence}"

                Instructions:
                - Answer "SUPPORTED" if the claim is directly stated or can be reasonably inferred from the context
                - Answer "UNSUPPORTED" if the claim contradicts the context or has no evidence in the context
                - Then provide a confidence score from 0.0 to 1.0

                Format your response as:
                VERDICT: [SUPPORTED or UNSUPPORTED]
                CONFIDENCE: [0.0-1.0]

                Response:"""

        response = self.mistral.client.chat.complete(
            model=self.mistral.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=100
        )
        try:
            response = response.choices[0].message.content.strip()
        except:
            return True, -1.0
        is_supported = "SUPPORTED" in response.upper() and "UNSUPPORTED" not in response.upper()
        
        confidence_match = re.search(r'CONFIDENCE:\s*(0?\.\d+|1\.0|0|1)', response, re.IGNORECASE)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        
        return is_supported and confidence >= self.confidence_threshold, confidence
    
    def check_hallucination(self, query: str, answer: str, context_chunks: List[Dict]) -> tuple[str, Dict]:
        """
        Split the answer into sentences and check if the answer is hallucinated against the knowledge base
        """
        sentences = self._sentence_chunks(answer)
        combined_context_chunks = self._combine_context_chunks(context_chunks)
        
        unverified_sentences = []
        claimed_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            is_supported, confidence = self._verify_sentence(sentence, combined_context_chunks, query)
            claimed_sentences.append({
                "claimed_sentence": sentence,
                "supported": is_supported,
                "confidence": confidence
            })

            if not is_supported:
                unverified_sentences.append(sentence)
        
        unverified_answer_list = ""
        for i, unverified_sentence in enumerate(unverified_sentences):
            unverified_answer_list += f"{i+1}. {unverified_sentence}\n"

        report = {
            "original_claimed_sentences": len(sentences),
            "unverified_sentences": len(unverified_sentences),
            "claimed_sentences": claimed_sentences,
            "unverified_answer_list": unverified_answer_list
        }
        return unverified_answer_list, report
        