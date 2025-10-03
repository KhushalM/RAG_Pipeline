import re
from typing import Dict, Any

class AnswerShaping:
    def detect_intent(self, query: str) -> str:
        """Simple pattern matching for intent detection"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['list', 'what are', 'name', 'identify', 'enumerate']):
            return 'list'
        elif any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus', 'better']):
            return 'comparison'
        elif any(word in query_lower for word in ['what is', 'define', 'meaning of']):
            return 'definition'
        elif any(word in query_lower for word in ['how to', 'how do', 'how can', 'steps']):
            return 'steps'
        else:
            return 'general'
    
    def shape_answer(self, query: str, answer: str) -> Dict[str, Any]:
        """Shape answer based on detected intent using simple text processing"""
        intent = self.detect_intent(query)
        
        items = re.findall(r'(?:^|\n)[-•*]\s*(.+?)(?=\n|$)', answer)
        numbered_items = re.findall(r'(?:^|\n)(\d+)[.)]\s*(.+?)(?=\n\d+[.)]|\n\n|$)', answer, re.MULTILINE)
        
        paragraphs = [p.strip() for p in answer.split('\n\n') if p.strip()]
        all_items = items + [item[1] for item in numbered_items]

        
        if intent == 'list' and items:
            return {
                "format_type": "list",
                "items": all_items[:10],
                "raw_answer": answer
            }
        
        elif intent == 'steps' and steps:
            return {
                "format_type": "steps",
                "steps": [{"num": num, "text": text.strip()} for num, text in steps],
                "raw_answer": answer
            }
        
        elif intent == 'definition':
            # First paragraph is definition, rest are details
            return {
                "format_type": "definition",
                "definition": paragraphs[0] if paragraphs else answer,
                "details": paragraphs[1:] if len(paragraphs) > 1 else [],
                "raw_answer": answer
            }
        
        else:
            # General format - just paragraphs
            return {
                "format_type": "general",
                "paragraphs": paragraphs if paragraphs else [answer],
                "raw_answer": answer
            }