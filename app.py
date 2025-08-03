import streamlit as st
import os
import pandas as pd
from docx import Document
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# Set up your OpenAI API key as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"

def extract_text_from_file(file):
    """Extracts text from an uploaded file object."""
    file_extension = os.path.splitext(file.name)[1].lower()
    text = ""
    
    if file_extension == '.docx':
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_extension == '.pdf':
        try:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            text = ""
    elif file_extension in ['.xlsx', '.xls', '.csv']:
        try:
            df = pd.read_excel(file) if file_extension in ['.xlsx', '.xls'] else pd.read_csv(file)
            text = df.to_string(index=False)
        except Exception as e:
            st.error(f"Error reading Excel/CSV: {e}")
            text = ""
    else:
        st.warning(f"Unsupported file type: {file_extension}")
        text = ""

    return text

def get_vector_store(text):
    """Creates a vector store from text chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    """Creates a conversational Q&A chain."""
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    return conversation_chain

def handle_user_input(user_question):
    """Handles user questions and displays the chatbot's response."""
    if "conversation" in st.session_state and "chat_history" in st.session_state:
        response = st.session_state.conversation({'question': user_question, 'chat_history': st.session_state.chat_history})
        st.session_state.chat_history = response['chat_history']
        st.write("You:", user_question)
        st.write("Bot:", response['answer'])

def main():
    st.set_page_config(page_title="Document Chatbot", page_icon=":books:")
    st.title("Chat with Your Documents")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.sidebar:
        st.subheader("Upload your documents")
        uploaded_file = st.file_uploader(
            "Upload files and click 'Process'",
            type=["pdf", "docx", "xlsx", "csv"],
            accept_multiple_files=False
        )
        if st.button("Process Document"):
            if uploaded_file:
                with st.spinner("Processing..."):
                    raw_text = extract_text_from_file(uploaded_file)
                    if raw_text:
                        vector_store = get_vector_store(raw_text)
                        st.session_state.conversation = get_conversation_chain(vector_store)
                        st.success("Document processed successfully!")
                    else:
                        st.error("Failed to process the document.")
            else:
                st.error("Please upload a document first.")
    
    st.subheader("Ask a question about your document:")
    user_question = st.text_input("Your question:")
    if user_question:
        handle_user_input(user_question)

if __name__ == '__main__':
    main()
