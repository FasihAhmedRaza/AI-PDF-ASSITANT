import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split extracted text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a FAISS vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Local embeddings model
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Add a powerful QA model (BERT) for specific question-answering
qa_pipeline_bert = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Add another model (DistilBERT) for variation in performance
qa_pipeline_distilbert = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Add a model for general text generation (GPT-2)
text_generation_pipeline = pipeline("text-generation", model="gpt2")

# Function to get answers from different NLP models
def get_answer_from_documents(context, question):
    answer_bert = qa_pipeline_bert(question=question, context=context)['answer']
    answer_distilbert = qa_pipeline_distilbert(question=question, context=context)['answer']
    # Add more logic to integrate different models and combine or choose the answer
    combined_answer = f"BERT: {answer_bert}\nDistilBERT: {answer_distilbert}"
    return combined_answer

# Function to generate a more comprehensive response (text generation model)
def generate_extended_response(context, question):
    generated_text = text_generation_pipeline(question, max_length=50, num_return_sequences=1)
    return generated_text[0]['generated_text']

# Function to handle user input and query FAISS index
def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Local embeddings model

    # Load FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=5)  # Retrieve more documents (e.g., 5 instead of 3)

    # Combine the text of retrieved documents for better context
    combined_context = " ".join([doc.page_content for doc in docs])

    # Get the answer from different models
    answer = get_answer_from_documents(combined_context, user_question)

    # Optionally generate a longer response using the text generation model
    extended_response = generate_extended_response(combined_context, user_question)

    return answer + "\n\nExtended Response: " + extended_response

# Custom CSS for enhanced UI
def add_custom_css():
    st.markdown(
        """
    <style>
    /* General styling */
    body {
        background-color: #0E1117;
    }
    .stApp {
        background-color: #0E1117;
    }

    /* Header styling */
    .main-header {
        font-family: 'Trebuchet MS', sans-serif;
        font-size: 3em;
        color: #4f93ce ;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2em;
        color: #888;
        margin-bottom: 2em;
    }

    /* Input field styling */
    .stTextInput > div > input {
        border-radius: 10px;
        padding: 10px;
        border: 2px solid #7aa6fc;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        color: white;
        background-color: #1f1f1f;
    }

    /* Sidebar styling */
    .stSidebar {
        background-color: #1f4f78;
    }
    .stSidebar h1 {
        color: #fff;
    }
    .stSidebar button {
        background-color: #4f93ce;
        color: #fff;
        border-radius: 10px;
        padding: 0.6em;
    }

    /* Button and spinner styling */
    .stButton button {
        background-color: #007bff;
        border: none;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 16px;
        cursor: pointer;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        transition: background-color 0.2s ease-in;
    }
    .stButton button:hover {
        background-color: #0056b3;
    }

    /* Chat history styling */
    .chat-bubble {
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 10px;
        width: fit-content;
        max-width: 80%;
    }
    .user-bubble {
        background-color: #4f93ce;
        color: white;
        align-self: flex-end;
    }
    .assistant-bubble {
        background-color: #333;
        color: white;
        align-self: flex-start;
    }
    .chat-container {
        display: flex;
        flex-direction: column-reverse; /* Display most recent messages at the bottom */
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

# Main function for Streamlit app
def main():
    st.set_page_config(page_title="AI PDF ASSISTANT", page_icon="📄", layout="wide")

    # Add custom CSS for enhanced UI
    add_custom_css()

    # Initialize session state for chat history if not already initialized
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # App header
    st.markdown('<h1 class="main-header">AI CHAT ASSISTANT</h1>', unsafe_allow_html=True)

    # Input field for user's question
    user_question = st.text_input("🔍 Ask a Question from the PDF Files")

    if user_question:
        response = user_input(user_question)
        # Update the chat history
        st.session_state.chat_history.append(
            {"question": user_question, "response": response}
        )

    # Display chat history
    if st.session_state.chat_history:
        chat_container = st.container()
        with chat_container:
            # Display the chat history in reverse order
            for chat in reversed(st.session_state.chat_history):
                st.markdown(
                    f"""
    <div class="chat-container">
        <div style="display: flex; align-items: flex-end; justify-content: flex-end;">
            <img src="https://img.icons8.com/color/48/000000/person-male.png" width="30" style="margin-right: 10px;">
            <div class="chat-bubble user-bubble">
                <strong></strong> {chat["question"]}
            </div>
        </div>
    </div>
    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="chat-container">
                         <div style="display: flex; align-items: flex-start;">
            <img src="https://img.icons8.com/fluency/48/000000/chatbot.png" width="30" style="margin-right: 10px;">
            <div class="chat-bubble assistant-bubble">
                <strong></strong> {chat["response"]}
            </div>
        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("<hr>", unsafe_allow_html=True)

    # Sidebar for file upload and processing
    with st.sidebar:
        st.title("📁 PDF Uploader")
        st.markdown("Upload your PDF files and ask questions.")

        # File uploader for PDF documents
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)

        if st.button("📥 Submit & Process"):
            with st.spinner("⏳ Processing..."):
                # Process uploaded PDF files
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Processing Complete")

# Run the app
if __name__ == "__main__":
    main()
