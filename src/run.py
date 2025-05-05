# run.py

from src.helper import load_env_variables
from src.vectorstore import get_existing_vectorstore
from src.prompt import build_rag_chain
from src.evaluator import compare_k_results_with_judge

from langchain_core.documents import Document

def main():
    # Load environment variables
    pinecone_api_key, openai_api_key = load_env_variables()

    # Load vectorstore (assumes pre-uploaded)
    vectorstore = get_existing_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # Build RAG chain (prompt + LLM chain)
    rag_chain, question_answer_chain = build_rag_chain()

    # Run an evaluation query
    query = "What are the treatment goals in heart failure with reduced ejection fraction?"
    result = compare_k_results_with_judge(query, retriever, question_answer_chain, rag_chain)

    # Final output summary
    print("\n🔍 Final Output Summary")
    print("Query:", query)
    print("Best k:", result["k"])
    print("\n📄 Refined Answer:\n", result["refined_answer"])


if __name__ == "__main__":
    main()
