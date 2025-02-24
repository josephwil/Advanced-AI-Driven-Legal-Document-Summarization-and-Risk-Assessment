import os
import streamlit as st
from PyPDF2 import PdfReader
import docx
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()   

# Get the API key from the .env file
GROQ_API_KEY = os.getenv("api_key")

# Initialize Groq model
model = ChatGroq(temperature=0.6, groq_api_key=GROQ_API_KEY, model_name="llama-3.1-70b-versatile")

# Inline CSS and templates
css = """
<style>
body {
    font-family: Arial, sans-serif;
}
</style>
"""

bot_template = """
<div style="background-color: #f0f0f0; border-radius: 10px; padding: 10px; margin: 5px 0;">
    <strong>Bot:</strong> {{MSG}}
</div>
"""

user_template = """
<div style="background-color: #d4e0ff; border-radius: 10px; padding: 10px; margin: 5px 0;">
    <strong>You:</strong> {{MSG}}
</div>
"""

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_word_text(word_docs):
    text = ""
    for doc in word_docs:
        doc_reader = docx.Document(doc)
        for para in doc_reader.paragraphs:
            text += para.text + "\n"
    return text

def get_text(text_files):
    text = ""
    for text_file in text_files:
        with open(text_file, "r", encoding="utf-8") as file:
            text += file.read() + "\n"
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

class CustomEmbeddings:
    def embed_documents(self, texts):
        return [[ord(char) for char in text] for text in texts]

def get_vectorstore(text_chunks):
    embedding_model = CustomEmbeddings()
    embeddings = embedding_model.embed_documents(text_chunks)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embedding_model)
    return vectorstore

def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def summarize_text(text, reduction_percentage=80):
    text_chunks = get_text_chunks(text)
    summary_chunks = []
    for chunk in text_chunks:
        response = model.summarize({"input_text": chunk, "reduction_percentage": reduction_percentage})
        summary_chunks.append(response["summary"])
    return "\n".join(summary_chunks)

def handle_userinput(user_question, convert_only):
    if convert_only:
        st.session_state.result_text = user_question
    else:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.result_text = response['summary']

def main():
    st.set_page_config(page_title="Document Converter and Summarizer", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "result_text" not in st.session_state:
        st.session_state.result_text = ""

    st.header("Document Converter and Summarizer :books:")
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type=["pdf"])
        word_docs = st.file_uploader("Upload your Word documents here", accept_multiple_files=True, type=["docx"])
        text_files = st.file_uploader("Upload your text files here", accept_multiple_files=True, type=["txt"])

        if st.button("Process"):
            with st.spinner("Processing"):
                # Get text from different document types
                raw_text = get_pdf_text(pdf_docs) + get_word_text(word_docs) + get_text(text_files)
                
                # Option to convert to text or summarize
                option = st.radio("Choose an option:", ("Convert to Text", "Generate Summary"))
                convert_only = option == "Convert to Text"
                
                # Create text chunks and conversation chain
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
                # Handle user input for conversion or summarization
                handle_userinput(raw_text, convert_only)

                st.success("Processing completed!")

    user_question = st.text_input("Ask a question about your documents:")
    chat = st.button('Chat')
    if chat:
        handle_userinput(user_question, False)
    
    st.subheader("Output")
    st.write(st.session_state.result_text)
    st.download_button(label="Download Result", data=st.session_state.result_text, file_name="result.txt")

if __name__ == '__main__':
    main()
