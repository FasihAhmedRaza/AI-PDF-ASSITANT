import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import spacy  
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Preprocess text using spaCy
def preprocess_text_with_spacy(text):
    doc = nlp(text)
    processed_text = " ".join([token.lemma_ for token in doc if not token.is_stop])  # Lemmatization & Stopword removal
    return processed_text

# Function to split extracted text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a FAISS vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain using a custom prompt and Gemini model
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, 
    just say, "The requested information is not available within the provided PDF content.Please Ask Another Question " Do not provide incorrect information.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to handle user input and query FAISS index
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load FAISS index with dangerous deserialization allowed (if you trust the source)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Get the conversational chain and generate the response
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # Return the response text
    return response["output_text"]

# Define custom CSS for enhanced UI and chat styling
def add_custom_css():
    st.markdown("""
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
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="AI PDF ASSISTANT", page_icon="📄", layout="wide")

    # Add custom CSS for enhanced UI
    add_custom_css()

    # Initialize session state for chat history if not already initialized
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # App header
    st.markdown('<h1 class="main-header">Chat with PDF </h1>', unsafe_allow_html=True)
    # st.markdown('<p class="sub-header">Easily chat with your uploaded PDF files. Powered by Google Generative AI and LLM.</p>', unsafe_allow_html=True)

    # Input field for user's question
    user_question = st.text_input("🔍 Ask a Question from the PDF Files")

    if user_question:
        response = user_input(user_question)
        # Update the chat history
        st.session_state.chat_history.append({"question": user_question, "response": response})
 # Display chat history
    if st.session_state.chat_history:
        chat_container = st.container()
        with chat_container:
            # Display the chat history in reverse order
            for chat in reversed(st.session_state.chat_history):
                st.markdown(
                    f'''
                    <div class="chat-container">
                        <div class="chat-bubble user-bubble">
                            <img src="https://img.icons8.com/color/48/000000/person-male.png" width="20" style="vertical-align: middle; margin-right: 10px;">
                            <strong></strong> {chat["question"]}
                        </div>
                    </div>
                    ''', 
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'''
                    <div class="chat-container">
                        <div class="chat-bubble assistant-bubble">
                            <img src="https://img.icons8.com/fluency/48/000000/chatbot.png" width="20" style="vertical-align: middle; margin-right: 10px;">
                            <strong></strong> {chat["response"]}
                        </div>
                    </div>
                    ''', 
                    unsafe_allow_html=True
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
                processed_text = preprocess_text_with_spacy(raw_text)  # Preprocess text with spaCy
                text_chunks = get_text_chunks(processed_text)
                get_vector_store(text_chunks)
                st.success("✅ Processing Complete")

# Run the app
if __name__ == "__main__":
    main()
