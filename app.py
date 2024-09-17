import streamlit as st
import requests
print("Importing main")
from main import query, is_valid_pdf, create_chroma_db, BytesIO
# Assuming that query function and other dependencies have been correctly imported and set up

print("Main Imported")

# Initialize your Streamlit app
st.title("UN SDGs Chatbot")

# Sidebar for file upload
st.sidebar.title("Chatbot Configuration")
st.sidebar.info("Upload PDFs to expand the RAG knowledge base.")

uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

# Check if a PDF file is uploaded and process it
if uploaded_file is not None:
    pdf_stream = BytesIO(uploaded_file.read())
    if is_valid_pdf(pdf_stream):
        create_chroma_db(pdf_stream)
        st.sidebar.success("PDF successfully added to the knowledge base!")
    else:
        st.sidebar.error("The uploaded PDF has no readable content.")

# Input for user's question
st.subheader("Ask your question")
question = st.text_input("Enter your question below:")

# A button to submit the question
if st.button("Submit Question"):
    if question.strip():  # Check if the question is not empty
        with st.spinner("Fetching response..."):
            # Call the query function with the user's question
            try:
                response = query(question=question)
                if response:
                    st.success("Response received:")
                    st.write(response)
                else:
                    st.warning("No relevant information found in the RAG database.")
            except Exception as e:
                st.error(f"An error occurred while processing your query: {e}")
    else:
        st.error("Please enter a question before submitting.")
