# src/embedding.py

from langchain.embeddings import HuggingFaceEmbeddings

def load_biobert_embedding(model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"):
    """
    Loads BioBERT embedding model for clinical and biomedical text.
    Returns a LangChain-compatible embedding object.
    """
    return HuggingFaceEmbeddings(model_name=model_name)
