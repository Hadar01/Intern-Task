from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import uuid
import os
import re
import textwrap
from PyPDF2 import PdfReader
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
try:
    from huggingface_hub import cached_download
except ImportError:
    from huggingface_hub import hf_hub_download as cached_download
    

# Initialize FastAPI app
app = FastAPI(title="Research Assistant API")

# CORS configuration (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI models (load once at startup)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
qg_model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# In-memory storage (replace with database in production)
document_store: Dict[str, dict] = {}
MAX_SUMMARY_LENGTH = 150

# Pydantic models for request/response
class AskRequest(BaseModel):
    upload_id: str
    question: str

class ChallengeResponse(BaseModel):
    question_id: str
    user_answer: str

class QuestionItem(BaseModel):
    id: str
    question: str
    expected_answer: str
    reference: str

# Helper functions
def process_document(file: UploadFile) -> str:
    """Extract text from PDF or TXT files"""
    content = file.file.read()
    
    if file.filename.endswith('.pdf'):
        try:
            text = ""
            reader = PdfReader(file.file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise HTTPException(400, f"PDF processing error: {str(e)}")
    
    elif file.filename.endswith('.txt'):
        return content.decode('utf-8')
    
    raise HTTPException(400, "Unsupported file format")

def generate_summary(text: str) -> str:
    """Generate concise summary (≤150 words)"""
    # Truncate to model's max token limit while preserving paragraphs
    paragraphs = text.split('\n\n')
    input_text = ""
    for para in paragraphs:
        if len(input_text + para) < 1000:  # Model token limit
            input_text += para + "\n\n"
        else:
            break
    
    summary = summarizer(
        input_text,
        max_length=MAX_SUMMARY_LENGTH,
        min_length=30,
        do_sample=False
    )[0]['summary_text']
    
    # Ensure word count limit
    words = summary.split()
    return ' '.join(words[:MAX_SUMMARY_LENGTH])

# main.py - Replace the generate_questions function
def generate_questions(text: str, num_questions: int = 3) -> List[QuestionItem]:
    """Generate comprehension questions from document text"""
    # Split document into logical chunks
    chunks = [chunk for chunk in text.split('\n\n') if len(chunk) > 100]
    questions = []
    
    for chunk in chunks[:5]:  # Process first 5 chunks
        inputs = tokenizer.encode("generate questions: " + chunk, return_tensors="pt", max_length=512, truncation=True)
        outputs = qg_model.generate(
            inputs,
            max_length=64,
            num_beams=5,
            num_return_sequences=1,
            early_stopping=True
        )
        
        for output in outputs:
            question = tokenizer.decode(output, skip_special_tokens=True)
            if question and '?' in question:
                # Get expected answer from QA model
                answer_result = qa_model(question=question, context=chunk)
                questions.append(QuestionItem(
                    id=str(uuid.uuid4()),
                    question=question,
                    expected_answer=answer_result['answer'],
                    reference=chunk[:200] + "..."  # Store reference snippet
                ))
                if len(questions) >= num_questions:
                    return questions
    return questions

def evaluate_answer(user_answer: str, expected_answer: str) -> dict:
    """Evaluate user answer against expected answer using semantic similarity"""
    embeddings = similarity_model.encode([user_answer, expected_answer])
    cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    
    # Convert similarity to evaluation
    if cos_sim > 0.8:
        return {"verdict": "Correct", "score": cos_sim}
    elif cos_sim > 0.5:
        return {"verdict": "Partially Correct", "score": cos_sim}
    else:
        return {"verdict": "Incorrect", "score": cos_sim}

def find_reference(text: str, answer: str) -> dict:
    """Find the source reference for an answer"""
    # Simple pattern matching (improve with NLP techniques)
    start_idx = text.find(answer)
    if start_idx == -1:
        # Try with normalization
        clean_answer = re.sub(r'[^\w\s]', '', answer)
        start_idx = text.find(clean_answer)
    
    if start_idx != -1:
        end_idx = start_idx + len(answer)
        context_start = max(0, start_idx - 50)
        context_end = min(len(text), end_idx + 50)
        return {
            "text": text[context_start:context_end],
            "start": start_idx,
            "end": end_idx
        }
    return {"text": "Reference not found", "start": -1, "end": -1}

# API Endpoints
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Process uploaded document and generate summary"""
    try:
        # Extract text
        text = process_document(file)
        if not text.strip():
            raise HTTPException(400, "Document appears to be empty")
        
        # Generate summary
        summary = generate_summary(text)
        
        # Create document record
        upload_id = str(uuid.uuid4())
        document_store[upload_id] = {
            "text": text,
            "summary": summary,
            "questions": []
        }
        
        return {"upload_id": upload_id, "summary": summary}
    
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")

@app.post("/ask")
async def ask_question(request: AskRequest):
    """Answer user questions about the document"""
    if request.upload_id not in document_store:
        raise HTTPException(404, "Document not found")
    
    doc = document_store[request.upload_id]
    try:
        # Get answer from QA model
        result = qa_model(
            question=request.question,
            context=doc["text"][:20000]  # Use first 20k chars for demo
        )
        
        # Find source reference
        reference = find_reference(doc["text"], result['answer'])
        
        return {
            "answer": result['answer'],
            "confidence": result['score'],
            "reference": reference["text"],
            "start_index": reference["start"],
            "end_index": reference["end"]
        }
    except Exception as e:
        raise HTTPException(500, f"QA failed: {str(e)}")

# Update the /generate_questions endpoint
# Update the endpoint to use query parameters
@app.post("/generate_questions")
async def get_challenge_questions(upload_id: str = Query(..., description="Document ID")):
    """Generate challenge questions for document"""
    if upload_id not in document_store:
        raise HTTPException(404, "Document not found")
    
    doc = document_store[upload_id]
    if not doc["questions"]:
        # Generate and store questions
        doc["questions"] = [q.dict() for q in generate_questions(doc["text"])]
    
    # Return without answers
    return [
        {"id": q["id"], "question": q["question"]} 
        for q in doc["questions"]
    ]

@app.post("/evaluate_answer")
async def evaluate_challenge_response(response: ChallengeResponse):
    """Evaluate user's answer to a challenge question"""
    # Find question in all documents
    question_data = None
    for doc in document_store.values():
        for q in doc["questions"]:
            if q["id"] == response.question_id:
                question_data = q
                break
        if question_data:
            break
    
    if not question_data:
        raise HTTPException(404, "Question not found")
    
    # Evaluate answer
    evaluation = evaluate_answer(response.user_answer, question_data["expected_answer"])
    
    return {
        "question": question_data["question"],
        "expected_answer": question_data["expected_answer"],
        "user_answer": response.user_answer,
        "verdict": evaluation["verdict"],
        "score": evaluation["score"],
        "reference": question_data["reference"]
    }

@app.get("/health")
def health_check():
    return {"status": "active", "models_loaded": True}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)