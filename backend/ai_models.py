from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import re
import numpy as np
import torch

class AIModels:
    def __init__(self):
        # Initialize models with default parameters
        self.summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn"
        )
        self.qa_model = pipeline(
            "question-answering", 
            model="deepset/roberta-base-squad2"
        )
        
        # Question generation setup
        self.qg_tokenizer = AutoTokenizer.from_pretrained(
            "mrm8488/t5-base-finetuned-question-generation-ap"
        )
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(
            "mrm8488/t5-base-finetuned-question-generation-ap"
        )
        
        # Answer evaluation model
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Configuration
        self.MAX_SUMMARY_LENGTH = 150
        self.MAX_QA_CONTEXT = 20000  # Characters

    # ai_models.py - Update generate_summary method
# Update the generate_summary function
def generate_summary(self, text: str) -> str:
    """Generate concise summary (≤150 words) with repetition handling"""
    # Remove duplicate sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    unique_sentences = []
    seen = set()
    
    for sentence in sentences:
        clean_sentence = re.sub(r'\W+', '', sentence.lower())
        if clean_sentence not in seen and len(clean_sentence) > 20:
            seen.add(clean_sentence)
            unique_sentences.append(sentence)
    
    # Use first 10 unique sentences or 1000 characters
    input_text = " ".join(unique_sentences[:10])[:1000]
    
    # Generate summary
    summary = self.summarizer(
        input_text,
        max_length=self.MAX_SUMMARY_LENGTH,
        min_length=30,
        do_sample=False
    )[0]['summary_text']
    
    # Ensure word count limit
    words = summary.split()
    return ' '.join(words[:self.MAX_SUMMARY_LENGTH])            

def answer_question(self, question: str, context: str) -> dict:
        """
        Answer a question based on document context
        Returns answer with confidence score and position
        """
        # Truncate context to model limits while preserving paragraphs
        if len(context) > self.MAX_QA_CONTEXT:
            paragraphs = context.split('\n\n')
            context = ""
            for para in paragraphs:
                if len(context) + len(para) < self.MAX_QA_CONTEXT:
                    context += para + "\n\n"
                else:
                    break
        
        result = self.qa_model(question=question, context=context)
        
        # Find exact position in original text
        start_idx = context.find(result['answer'])
        if start_idx == -1:
            # Try with normalization for better matching
            clean_answer = re.sub(r'[^\w\s]', '', result['answer'])
            start_idx = context.find(clean_answer)
            if start_idx != -1:
                end_idx = start_idx + len(clean_answer)
            else:
                start_idx, end_idx = -1, -1
        else:
            end_idx = start_idx + len(result['answer'])
        
        return {
            "answer": result['answer'],
            "confidence": result['score'],
            "start_index": start_idx,
            "end_index": end_idx
        }

def generate_questions(self, text: str, num_questions: int = 3) -> list:
        """
        Generate comprehension questions from document text
        Returns list of dictionaries with question, answer, and reference
        """
        # Split document into logical chunks
        chunks = [chunk for chunk in text.split('\n\n') if len(chunk) > 100]
        questions = []
        
        for chunk in chunks[:5]:  # Process first 5 chunks max
            # Prepare input for question generation
            inputs = self.qg_tokenizer.encode(
                "generate questions: " + chunk, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )
            
            # Generate questions
            outputs = self.qg_model.generate(
                inputs,
                max_length=64,
                num_beams=5,
                num_return_sequences=min(3, num_questions),
                early_stopping=True
            )
            
            for output in outputs:
                question = self.qg_tokenizer.decode(
                    output, 
                    skip_special_tokens=True
                )
                
                # Filter invalid questions
                if not question or '?' not in question:
                    continue
                
                # Get expected answer from QA model
                answer_result = self.answer_question(question, chunk)
                
                # Only keep questions with found answers
                if answer_result["start_index"] != -1:
                    questions.append({
                        "question": question,
                        "expected_answer": answer_result["answer"],
                        "reference": chunk[:200] + "..."  # Store reference snippet
                    })
                    
                    if len(questions) >= num_questions:
                        return questions
        
        return questions

def evaluate_answer(self, user_answer: str, expected_answer: str) -> dict:
        """
        Evaluate user's answer against expected answer using semantic similarity
        Returns verdict (Correct/Partially Correct/Incorrect) and similarity score
        """
        # Handle empty answers
        if not user_answer.strip():
            return {"verdict": "Incorrect", "score": 0.0}
        
        # Compute embeddings
        embeddings = self.similarity_model.encode(
            [user_answer, expected_answer],
            convert_to_tensor=True
        )
        
        # Calculate cosine similarity
        cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        
        # Convert to evaluation
        if cos_sim > 0.8:
            verdict = "Correct"
        elif cos_sim > 0.5:
            verdict = "Partially Correct"
        else:
            verdict = "Incorrect"
        
        return {"verdict": verdict, "score": cos_sim}

def find_reference(self, full_text: str, answer: str, context_window: int = 100) -> str:
        """
        Find the source reference for an answer with surrounding context
        """
        # First try exact match
        start_idx = full_text.find(answer)
        if start_idx != -1:
            end_idx = start_idx + len(answer)
        else:
            # Try normalized match (ignore punctuation)
            clean_answer = re.sub(r'[^\w\s]', '', answer)
            start_idx = full_text.find(clean_answer)
            if start_idx != -1:
                end_idx = start_idx + len(clean_answer)
            else:
                # Fallback to returning the answer itself
                return answer
        
        # Extract context around answer
        context_start = max(0, start_idx - context_window)
        context_end = min(len(full_text), end_idx + context_window)
        
        # Create highlighted reference
        before = full_text[context_start:start_idx]
        highlighted = full_text[start_idx:end_idx]
        after = full_text[end_idx:context_end]
        
        return f"...{before}<b>{highlighted}</b>{after}..."

# Singleton instance for shared model usage
ai_models = AIModels()