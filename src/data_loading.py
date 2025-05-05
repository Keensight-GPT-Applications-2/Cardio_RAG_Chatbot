# src/data_loading.py
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from src.helper import clean_text

# ========== 1. Load PDFs from folder ==========

def load_pdf_documents(data_folder):
    """Load PDF documents from the specified folder."""
    loader = DirectoryLoader(
        data_folder,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# ========== 2. Clean extracted documents ==========

def clean_documents(documents):
    """Apply text cleaning to all documents and filter pages."""
    cleaned = []
    for doc in documents:
        content = clean_text(doc.page_content)
        if len(content.strip()) > 50:  # Only keeping meaningful content
            cleaned.append(Document(page_content=content, metadata=doc.metadata))
    return cleaned

# ========== 3. Split documents into chunks ==========

def split_documents(documents, chunk_size=1000, chunk_overlap=150):
    """Split documents into smaller overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks
