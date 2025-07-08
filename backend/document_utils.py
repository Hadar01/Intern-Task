import re
import os
import uuid
from PyPDF2 import PdfReader
from typing import Union, Tuple, Dict

class DocumentUtils:
    @staticmethod
    def process_uploaded_file(file: Union[str, bytes], filename: str) -> Tuple[str, str]:
        """
        Process uploaded file and extract text content
        Returns: (text_content, error_message)
        """
        try:
            # Handle PDF files
            if filename.lower().endswith('.pdf'):
                return DocumentUtils.extract_pdf_text(file), ""
            
            # Handle TXT files
            elif filename.lower().endswith('.txt'):
                if isinstance(file, bytes):
                    return file.decode('utf-8'), ""
                return file, ""
            
            return "", "Unsupported file format. Please upload PDF or TXT."
        
        except Exception as e:
            return "", f"Error processing file: {str(e)}"

   # Update extract_pdf_text method
# Update extract_pdf_text method
@staticmethod
def extract_pdf_text(file: Union[str, bytes]) -> str:
    """Extract text from PDF using PyMuPDF for better accuracy"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n\n"
        return DocumentUtils.clean_text(text)
    except ImportError:
        # Fallback to PyPDF2
        from PyPDF2 import PdfReader
        from io import BytesIO
        reader = PdfReader(BytesIO(file))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        return DocumentUtils.clean_text(text)
    

@staticmethod
def clean_text(text: str) -> str:
        """
        Normalize and clean extracted text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Restore paragraph breaks around common section markers
        text = re.sub(r'(\n\s*)([A-Z][A-Za-z ]+:)', r'\n\n\2', text)
        
        # Fix hyphenated words across lines
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        
        # Remove header/footer artifacts
        text = re.sub(r'\d+\s+\|\s+', '', text)  # Page numbers
        text = re.sub(r'www\.\S+', '', text)     # URLs
        text = re.sub(r'\b[A-Z]{3,}\b', '', text)  # ALL CAPS headers
        
        # Normalize unicode characters
        text = text.encode('ascii', 'ignore').decode()
        
        return text.strip()

@staticmethod
def split_into_chunks(text: str, min_chunk_size: int = 200, max_chunk_size: int = 1000) -> list:
        """
        Split document into logical chunks (paragraphs or sections)
        """
        # First try splitting by double newlines (paragraphs)
        chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
        
        # If chunks are too large, split by sentence boundaries
        refined_chunks = []
        for chunk in chunks:
            if len(chunk) > max_chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < max_chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            refined_chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                if current_chunk:
                    refined_chunks.append(current_chunk.strip())
            else:
                refined_chunks.append(chunk)
        
        # Filter small chunks
        return [chunk for chunk in refined_chunks if len(chunk) >= min_chunk_size]

@staticmethod
def create_reference(text: str, answer: str, context_chars: int = 100) -> Dict[str, Union[str, int]]:
        """
        Find the source reference for an answer in the text
        Returns dictionary with:
          - reference: text snippet with context
          - start: start index in full text
          - end: end index in full text
        """
        # First try exact match
        start_idx = text.find(answer)
        if start_idx != -1:
            end_idx = start_idx + len(answer)
        else:
            # Try case-insensitive search
            start_idx = text.lower().find(answer.lower())
            if start_idx != -1:
                end_idx = start_idx + len(answer)
            else:
                # Try normalized match (ignore punctuation and whitespace)
                clean_answer = re.sub(r'[^\w\s]', '', answer).strip()
                clean_text = re.sub(r'[^\w\s]', '', text)
                start_idx = clean_text.find(clean_answer)
                if start_idx != -1:
                    # Map back to original text positions
                    orig_start = text.find(clean_text[start_idx:start_idx+20])
                    if orig_start != -1:
                        start_idx = orig_start
                        end_idx = start_idx + len(clean_answer)
                else:
                    return {
                        "text": f"[Reference not found for: '{answer}']",
                        "start": -1,
                        "end": -1
                    }
        
        # Extract context around the answer
        context_start = max(0, start_idx - context_chars)
        context_end = min(len(text), end_idx + context_chars)
        
        # Create context snippet
        before_context = text[context_start:start_idx]
        answer_text = text[start_idx:end_idx]
        after_context = text[end_idx:context_end]
        
        # Highlight answer in context
        highlighted = f"...{before_context}<b>{answer_text}</b>{after_context}..."
        
        return {
            "text": highlighted,
            "start": start_idx,
            "end": end_idx
        }

@staticmethod
def generate_upload_id() -> str:
        """Generate unique document ID"""
        return f"doc_{uuid.uuid4().hex[:10]}"

@staticmethod
def validate_text_content(text: str, min_length: int = 50) -> bool:
        """Check if text has sufficient content"""
        return len(text.strip()) >= min_length

@staticmethod
def extract_metadata(text: str) -> dict:
        """Extract basic metadata from document text"""
        # Try to find title in first few lines
        first_lines = text.split('\n')[:5]
        potential_title = ""
        for line in first_lines:
            if len(line.split()) > 3 and len(line) < 120:
                potential_title = line
                break
        
        # Estimate word count
        word_count = len(re.findall(r'\w+', text))
        
        return {
            "title": potential_title[:100] if potential_title else "Untitled Document",
            "word_count": word_count,
            "paragraph_count": len(text.split('\n\n')),
            "estimated_pages": max(1, word_count // 500)
        }