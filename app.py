import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
import re
from nltk.corpus import stopwords
import nltk

# Set the page config as the first Streamlit command
st.set_page_config(page_title="AI PDF ASSISTANT", page_icon="📄", layout="wide")

# Directly configure Google API Key
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"  # Replace with your actual API key
genai.api_key = GOOGLE_API_KEY

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

# Function to create a conversational chain using a custom prompt and Gemini model
def get_conversational_chain():
    prompt_template = """
    Answer the question in detail using the provided context. If the answer is not available, respond with:
    "The requested information is not available within the provided PDF content. Please ask another question." 
    Ensure the response is accurate based on the provided context.

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Load stopwords list
stop_words = set(stopwords.words('english'))

def preprocess_question(question):
    # Convert to lowercase and remove common stopwords
    question = question.lower()
    question = ' '.join([word for word in question.split() if word not in stop_words])
    
    # Remove special characters
    question = re.sub(r'[^\w\s]', '', question)
    
    return question

# Modify user_input function to preprocess the question and use text chunks directly
def user_input(user_question, text_chunks):
    try:
        if not text_chunks:
            st.warning("No text chunks are available for your query.")
            return ""
        
        # Convert text chunks to Document objects
        docs = [Document(page_content=chunk) for chunk in text_chunks]

        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question}, return_only_outputs=True
        )
        return response["output_text"]
        
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
    # Add custom CSS for enhanced UI
    add_custom_css()

    # Initialize session state for chat history and text chunks if not already initialized
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = []

    # App header
    st.markdown('<h1 class="main-header">AI CHAT ASSISTANT </h1>', unsafe_allow_html=True)

    # Input field for user's question
    user_question = st.text_input("🔍 Ask a Question from the PDF Files")

    if user_question:
        response = user_input(user_question, st.session_state.text_chunks)
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
        st.subheader("Upload PDF Files")

        pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)

        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Extracting text..."):
                    raw_text = get_pdf_text(pdf_docs)

                    with st.spinner("Splitting text into chunks..."):
                        st.session_state.text_chunks = get_text_chunks(raw_text)

                    st.success("PDFs processed successfully!")
            else:
                st.warning("Please upload one or more PDF files to process.")

# Run the app
if __name__ == "__main__":
    main()
