import streamlit as st
import requests
import uuid
import time
from typing import List, Dict, Tuple
# Add this to your main() function before any other content

# Configuration
BACKEND_URL = "http://localhost:8000"
MAX_SUMMARY_LENGTH = 150

def initialize_session():
    """Initialize session state variables"""
    if "document" not in st.session_state:
        st.session_state.document = {
            "upload_id": None,
            "text": "",
            "summary": "",
            "metadata": {}
        }
    
    if "challenge_questions" not in st.session_state:
        st.session_state.challenge_questions = []
    
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

def upload_document(file) -> bool:
    """Upload document to backend and process it"""
    if file is None:
        return False
    
    try:
        # Send file to backend
        files = {"file": file}
        response = requests.post(f"{BACKEND_URL}/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.document["upload_id"] = data["upload_id"]
            st.session_state.document["summary"] = data["summary"]
            
            # Get document metadata
            st.session_state.document["metadata"] = {
                "name": file.name,
                "size": f"{len(file.getvalue()) / 1024:.1f} KB",
                "type": file.type
            }
            
            # Initialize QA history
            st.session_state.qa_history = []
            return True
        else:
            st.error(f"Backend error: {response.text}")
            return False
    except Exception as e:
        st.error(f"Connection failed: {str(e)}")
        return False

def ask_question(question: str) -> Dict:
    """Send question to backend QA endpoint"""
    if not st.session_state.document["upload_id"]:
        return {"error": "No document uploaded"}
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/ask",
            json={
                "upload_id": st.session_state.document["upload_id"],
                "question": question
            }
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# app.py - Update generate_challenge_questions function
# Update the generate_challenge_questions function
def generate_challenge_questions() -> List[Dict]:
    """Generate challenge questions from backend"""
    if not st.session_state.document["upload_id"]:
        return []
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/generate_questions",
            params={"upload_id": st.session_state.document["upload_id"]}  # Send as query parameter
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Backend error: {response.text}")
            return []
    except Exception as e:
        st.error(f"Failed to generate questions: {str(e)}")
        return []
    
def evaluate_answer(question_id: str, user_answer: str) -> Dict:
    """Evaluate user's answer to a challenge question"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/evaluate_answer",
            json={
                "question_id": question_id,
                "user_answer": user_answer
            }
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def display_sidebar():
    """Display sidebar with document info and controls"""
    with st.sidebar:
        st.header("Research Assistant")
        st.subheader("Document Explorer")
        
        # Document upload section
        uploaded_file = st.file_uploader(
            "Upload Research Document",
            type=["pdf", "txt"],
            accept_multiple_files=False
        )
        
        if uploaded_file:
            if st.button("Process Document"):
                with st.spinner("Analyzing document..."):
                    if upload_document(uploaded_file):
                        st.success("Document processed successfully!")
        
        # Show document info if uploaded
        if st.session_state.document["upload_id"]:
            st.divider()
            st.subheader("Document Info")
            st.write(f"**Name**: {st.session_state.document['metadata']['name']}")
            st.write(f"**Size**: {st.session_state.document['metadata']['size']}")
            
            # Summary section
            st.subheader("Document Summary")
            summary = st.session_state.document["summary"]
            if len(summary.split()) > MAX_SUMMARY_LENGTH:
                summary = ' '.join(summary.split()[:MAX_SUMMARY_LENGTH]) + "..."
            st.info(summary)
            
            # Conversation memory
            if st.session_state.qa_history:
                st.divider()
                st.subheader("Conversation History")
                for i, qa in enumerate(st.session_state.qa_history):
                    with st.expander(f"Q{i+1}: {qa['question']}"):
                        st.markdown(f"**Answer**: {qa['answer']}")
                        st.markdown(f"**Reference**: {qa['reference']}", unsafe_allow_html=True)
        
        st.divider()
        st.markdown("### About")
        st.info("""
        This research assistant helps you:
        - Summarize documents
        - Answer questions about content
        - Test your comprehension
        """)

def display_ask_anything():
    """Display QA interface"""
    st.subheader("Ask Anything")
    
    # Question input
    question = st.text_input(
        "Ask a question about the document:",
        placeholder="What is the main finding?",
        key="question_input"
    )
    
    # Submit button
    if st.button("Get Answer") and question:
        with st.spinner("Analyzing document..."):
            response = ask_question(question)
            
            if "error" in response:
                st.error(f"Error: {response['error']}")
            else:
                # Add to conversation history
                qa_record = {
                    "question": question,
                    "answer": response["answer"],
                    "reference": response["reference"],
                    "confidence": response["confidence"]
                }
                st.session_state.qa_history.append(qa_record)
                
                # Display results
                st.success(f"**Answer**: {response['answer']}")
                st.markdown("**Reference**:")
                st.markdown(response["reference"], unsafe_allow_html=True)
                st.caption(f"Confidence: {response['confidence']:.2f}")
                
                # Follow-up suggestion
                if len(st.session_state.qa_history) == 1:
                    st.info("💡 You can now ask follow-up questions about this topic")

def display_challenge_me():
    """Display challenge mode interface with improved UI"""
    st.subheader("🧠 Challenge Me")
    
    # Generate questions button with better styling
    if not st.session_state.challenge_questions:
        generate_btn = st.button(
            "✨ Generate Comprehension Questions",
            key="generate_questions_btn",
            use_container_width=True,
            type="primary"
        )
        
        if generate_btn:
            with st.spinner("Creating challenging questions..."):
                questions = generate_challenge_questions()
                if questions:
                    st.session_state.challenge_questions = [
                        {"id": q["id"], "question": q["question"], "user_answer": "", "feedback": None}
                        for q in questions
                    ]
                    st.success(f"Generated {len(questions)} questions! Scroll down to answer them.")
                else:
                    st.error("Failed to generate questions. Try processing the document again.")
    
    # Display questions and answer forms
    if st.session_state.challenge_questions:
        st.divider()
        
        # Progress tracker
        answered_count = sum(1 for q in st.session_state.challenge_questions if q["user_answer"].strip())
        st.caption(f"Progress: {answered_count}/{len(st.session_state.challenge_questions)} questions answered")
        
        # Question cards
        for i, q in enumerate(st.session_state.challenge_questions):
            with st.container(border=True):
                col1, col2 = st.columns([1, 0.1])
                with col1:
                    st.markdown(f"### ❓ Question {i+1}")
                    st.markdown(f"**{q['question']}**")
                    
                    # Answer input
                    answer = st.text_area(
                        "Your answer:",
                        value=q["user_answer"],
                        key=f"answer_{q['id']}",
                        height=100,
                        label_visibility="collapsed",
                        placeholder="Type your answer here..."
                    )
                    st.session_state.challenge_questions[i]["user_answer"] = answer
                    
                    # Submit button
                    submit_btn = st.button(
                        f"Submit Answer {i+1}",
                        key=f"submit_{i}",
                        type="primary" if not q["feedback"] else "secondary"
                    )
                    
                # Status indicator
                with col2:
                    if q["feedback"]:
                        if q["feedback"]["verdict"] == "Correct":
                            st.success("✓", help="Correct answer")
                        elif q["feedback"]["verdict"] == "Partially Correct":
                            st.warning("~", help="Partially correct")
                        else:
                            st.error("✗", help="Incorrect answer")
                
                # Handle submission
                if submit_btn:
                    if not answer.strip():
                        st.warning("Please enter an answer before submitting")
                    else:
                        with st.spinner("Evaluating your answer..."):
                            evaluation = evaluate_answer(q["id"], answer)
                            if "error" in evaluation:
                                st.error(evaluation["error"])
                            else:
                                st.session_state.challenge_questions[i]["feedback"] = evaluation
                                st.rerun()
                
                # Display feedback if available
                # Replace the feedback display section in display_challenge_me()
                if q["feedback"]:
                    feedback = q["feedback"]
                    if feedback["verdict"] == "Correct":
                        st.success(f"✅ {feedback['verdict']} (Score: {feedback['score']:.2f})")
                    elif feedback["verdict"] == "Partially Correct":
                        st.warning(f"⚠️ {feedback['verdict']} (Score: {feedback['score']:.2f})")
                    else:
                        # REMOVE THE HELP PARAMETER HERE
                        st.error(f"❌ {feedback['verdict']} (Score: {feedback['score']:.2f})")
                    
                    st.markdown(f"**Expected Answer**: {feedback['expected_answer']}")
                    st.markdown(f"**Reference**: {feedback['reference']}")
        
        # Summary report
        if all(q["feedback"] for q in st.session_state.challenge_questions):
            st.divider()
            st.subheader("📊 Comprehension Report")
            
            scores = [q["feedback"]["score"] for q in st.session_state.challenge_questions]
            avg_score = sum(scores) / len(scores)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Average Score", f"{avg_score:.2f}/1.0")
                
            with col2:
                st.progress(avg_score, text=f"Overall Comprehension: {avg_score*100:.0f}%")
            
            if avg_score > 0.75:
                st.success("🎉 Excellent understanding! You've mastered this document.")
            elif avg_score > 0.5:
                st.info("👍 Good understanding! You've grasped the main concepts.")
            else:
                st.warning("📚 Review recommended! Consider reading the document again.")
            
            if st.button("🔄 Generate New Questions", key="new_questions"):
                st.session_state.challenge_questions = []
                st.rerun()

def main():
    """Main application layout"""
    # Initialize session state
    initialize_session()
    
    # Configure page
    st.set_page_config(
        page_title="Research Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    # Custom CSS for styling
    st.markdown("""
<style>
    /* Improved container styling */
    .stContainer {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Better question cards */
    [data-testid="stExpander"] {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    /* Button improvements */
    .stButton>button {
        border-radius: 8px;
        padding: 8px 16px;
    }
    
    /* Text area improvements */
    .stTextArea>textarea {
        border-radius: 8px;
        padding: 12px;
    }
    
    /* Status indicators */
    .stSuccess, .stWarning, .stError {
        border-radius: 50%;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)
   
    
    # Main layout
    st.title("📚 Research Assistant")
    st.caption("An AI-powered tool for document comprehension and analysis")
    
    # Sidebar
    display_sidebar()
    
    # Main content area
    if not st.session_state.document["upload_id"]:
        # Welcome screen
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Get Started")
            st.info("""
            1. Upload a research paper, report, or technical document (PDF/TXT)
            2. Get an automatic summary
            3. Choose an interaction mode:
               - **Ask Anything**: Get answers to your questions
               - **Challenge Me**: Test your comprehension
            """)
            st.image("https://images.unsplash.com/photo-1507842217343-583bb7270b66?auto=format&fit=crop&w=600", 
                     caption="Research Assistant in Action")
        
        with col2:
            st.subheader("Example Documents")
            st.markdown("""
            - Research papers (PDF)
            - Technical documentation (PDF/TXT)
            - Legal contracts (PDF)
            - Business reports (PDF/TXT)
            - Academic articles (PDF)
            """)
            st.divider()
            st.markdown("""
            **Sample questions you can ask:**
            - What is the main contribution of this paper?
            - List the key findings from section 3
            - How was the experiment conducted?
            - What future work is suggested?
            """)
        return
    
    # Document is uploaded - show interaction modes
    tab1, tab2 = st.tabs(["📝 Ask Anything", "🧠 Challenge Me"])
    
    with tab1:
        display_ask_anything()
    
    with tab2:
        display_challenge_me()

if __name__ == "__main__":
    main()