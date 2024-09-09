import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import arabic_reshaper
from bidi.algorithm import get_display
from camel_tools.tokenizers.word import simple_word_tokenize


# Load environment variables
load_dotenv()

# Configure Google API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Download stopwords if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Load stopwords list
stop_words_en = set(stopwords.words('english'))
stop_words_ar = set(stopwords.words('arabic'))


# Function to extract text from PDF files with better handling
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF file {pdf.name}: {str(e)}")
    if not text:
        st.warning("No text extracted from PDF files.")
    return text

# Function to split extracted text into chunks with dynamic chunk size
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
def get_conversational_chain(language='english'):
    prompt_template = """
    Answer the question in detail using the provided context. If the answer is not available, respond with:
    "The requested information is not available within the provided PDF content. Please ask another question." 
    Ensure the response is accurate based on the provided context.

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    
    # Load a multilingual or Arabic-specific model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Preprocess text based on language
# Preprocess text based on language
def preprocess_text(text, language='english'):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    if language == 'arabic':
        stop_words = stop_words_ar
        tokens = simple_word_tokenize(text)  # Tokenize using camel-tools
    else:
        stop_words = stop_words_en
        tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Modify user_input function to preprocess the question
def user_input(user_question, language):
    try:
        # Initialize embeddings with a valid model name
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        new_db = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
        
        docs = new_db.similarity_search(user_question)
        
        if not docs:
            st.warning("No relevant documents found in the index for your query.")
            return ""

        chain = get_conversational_chain(language)
        response = chain(
            {"input_documents": docs, "question": user_question}, return_only_outputs=True
        )
        return response["output_text"]
        
    except FileNotFoundError:
        st.error("FAISS index not found. Please upload and process PDF files first.")
        return ""
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return ""

# Custom CSS for enhanced UI
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

    # Initialize session state for chat history and PDF text if not already initialized
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = ""

    # App header
    st.markdown('<h1 class="main-header">AI CHAT ASSISTANT </h1>', unsafe_allow_html=True)

    # File uploader for PDF files
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        # Save uploaded files temporarily and extract text
        temp_files = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_files.append(temp_file.name)

        # Extract text from the uploaded PDFs
        pdf_text = get_pdf_text(temp_files)
        st.session_state.pdf_text = pdf_text

        # Split and index the text
        if pdf_text:
            text_chunks = get_text_chunks(pdf_text)
            get_vector_store(text_chunks)
            st.success("PDF files processed and indexed successfully.")

    # Language selection
    language = st.selectbox("Select Language", ["English", "Arabic"])

    # Input field for user's question
    user_question = st.text_input("🔍 Ask a Question from the PDF Files")

    if user_question:
        if language == 'Arabic':
            user_question = preprocess_text(user_question, language='arabic')
        response = user_input(user_question, language)
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
            <img src="https://img.icons8.com/color/48/000000/person-male.png" width="30" style="margin-right: 10px;"/>
            <div class="chat-bubble user-bubble">
                {chat['question']}
            </div>
        </div>
        <div style="display: flex; align-items: flex-start; justify-content: flex-start;">
            <div class="chat-bubble assistant-bubble">
                {chat['response']}
            </div>
        </div>
    </div>
    """,
                    unsafe_allow_html=True,
                )

# Run the main function
if __name__ == "__main__":
    main()
