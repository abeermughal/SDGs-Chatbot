from io import BytesIO
import fitz  # PyMuPDF
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

from chromadb import PersistentClient

CHROMA_BASE_PATH = "./db"

GROQ_API_KEY = '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'

print("In Main")


embedding = HuggingFaceEmbeddings(
    model_name="./models/all-mpnet-base-v2",  # Path to the local directory
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)


print("Embedding Loaded")

model = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    api_key=GROQ_API_KEY
)

print("Model Loaded")
template = """
You are a Sustainable Development Professional. Strictly use the tone of a professional.
Answer the question based on the context below. If you can't answer the question
or the context is not relevant to the question, reply "I don't have that information".

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)

#directory_path = os.fsencode("./pdfs/")

def query(question: str) -> None:
        try:
            chroma_path = os.path.join(CHROMA_BASE_PATH)

            # Check if the Chroma database for this server exists
            if not os.path.exists(chroma_path):
                return

            db = Chroma(persist_directory=chroma_path, embedding_function=embedding)
            results = db.similarity_search_with_relevance_scores(question, k=3)

            if len(results) == 0 or results[0][1] < 0.3:
                return

            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

            # Generate the response using the model
            chain = prompt | model | StrOutputParser()
            response = chain.invoke({
                "context": context_text,
                "question": question
            })

        except Exception as e:
            print(f"Error processing query: {e}")
        return response

# Define the directory path as a string, not bytes
directory_path = "./pdfs/"

def upload_file() -> None:
    # Iterate over files in the specified directory
    for file in os.listdir(directory_path):
        filename = os.fsdecode(file)  # Decode to string if encoded, or simply use file if it's a string already

        # Ensure that the file is a valid PDF by checking the extension
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory_path, filename)  # Ensure all path components are strings

            try:
                # Load the file content into memory as bytes
                with open(filepath, 'rb') as f:
                    file_data = f.read()  # Read the entire file as bytes

                # Create a BytesIO stream from the byte data
                pdf_stream = BytesIO(file_data)

                # Check if the PDF is valid and contains readable text
                if is_valid_pdf(pdf_stream):
                    create_chroma_db(pdf_stream)  # Process the file with your existing function
                    print(f"{filename} successfully uploaded ✅")
                else:
                    print(f"Error: No readable content found in {filename}")

            except Exception as e:
                print(f"Error uploading file {filename}: {e}")

        else:
            print(f"{filename} is not a valid PDF file")

def is_valid_pdf(pdf_stream: BytesIO) -> bool:
    """
    Checks if the provided PDF stream contains readable text.
    """
    try:
        pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            text = page.get_text("text")
            if text.strip():  # Check if there's any readable text
                return True
        return False  # No readable text found
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return False


def upload_file_error(error):
        print(f"Error uploading file: {error}")

def create_chroma_db(pdf_stream: BytesIO):
    def generate_data_store():
        documents = load_documents()
        chunks = split_text(documents)
        save_to_chroma(chunks)

    def load_documents():
        documents = []
        pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            text = page.get_text("text")
            metadata = {
                "page_number": page_number + 1,
            }
            documents.append(Document(page_content=text, metadata=metadata))
        print("Documents loaded")
        return documents

    def split_text(documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks


    def save_to_chroma(chunks: list[Document]):
        chroma_path = os.path.join(CHROMA_BASE_PATH)


        print(f"Creating new Chroma database in directory {chroma_path}")
        try:
            # Create a new DB from the documents
            db = Chroma.from_documents(
                chunks, embedding, persist_directory=chroma_path
            )

        except Exception as e:
            raise  # Re-raise the exception to be caught in the `upload_file` method

    generate_data_store()

#upload_file()
