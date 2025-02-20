import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Function to convert PDF to text
def pdf_to_text(pdf_file):
    text = ''
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to convert image to text using Tesseract OCR
def image_to_text(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

# Streamlit web interface
st.title("File Upload and Text Extraction")

# File uploader widget
uploaded_file = st.file_uploader("Choose a PDF, FIR Image or Document file", type=["pdf", "png", "jpg", "jpeg", "doc", "docx"])

# Submit button
if st.button("Submit"):
    if uploaded_file is not None:
        file_type = uploaded_file.type
        if file_type == 'application/pdf':
            # Convert PDF to text
            extracted_text = pdf_to_text(uploaded_file)
        elif file_type in ['image/png', 'image/jpg', 'image/jpeg']:
            # Convert image to text
            extracted_text = image_to_text(uploaded_file)
        elif file_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            # Handle document files (FIR doc)
            # This part can be implemented using libraries like python-docx or similar
            extracted_text = 'Document to text conversion not implemented in this example'
        else:
            extracted_text = "Unsupported file type"

        # Display extracted text
        st.text_area("Extracted Text", extracted_text, height=200)

        # Use Hugging Face summarizer to summarize the text
        summary = summarizer(extracted_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        st.text_area("Summary", summary, height=100)

        # Save extracted text and summary to a new text file
        with open("extracted_text.txt", "w") as text_file:
            text_file.write(extracted_text)
        with open("summary_text.txt", "w") as summary_file:
            summary_file.write(summary)

        st.success("Text extraction and summarization completed. Files saved as extracted_text.txt and summary_text.txt.")
    else:
        st.error("Please upload a file.")

# Comments explaining what each section does are included inline in the code
