# src/prompt.py

from langchain.prompts import PromptTemplate

# ========= System prompt for RAG =========
RAG_SYSTEM_PROMPT = (
    "You are a helpful and knowledgeable assistant specialized in cardiology and cardiovascular medicine. "
    "Use only the provided context from medical guidelines and textbooks to answer the question. "
    "When relevant, mention the document source name and page number in parentheses. "
    "If the answer is not contained in the context, reply with: 'I’m not sure based on the provided information.' "
    "Keep your answer medically accurate, concise (min 5 sentences and max 10 sentences), and avoid speculation."
)

rag_prompt = PromptTemplate.from_template(
    """
    System: {system_prompt}

    Context:
    {context}

    Human: {input}
    """
)

# ========= Evaluation prompt - LLM Judge =========
EVAL_PROMPT_TEMPLATE = """
You are a medical expert evaluating an assistant's answer to a clinical query. Use only the provided context.

Rate the answer from 1 to 5 (5 = excellent) using following task-specific metrics:

- Relevance
- Factual Accuracy
- Completeness
- Source Attribution
- Clarity

Then provide a two-line explanation and the average score.

---
Query: {query}

Context:
{context}

Answer:
{answer}

Evaluation Format:
Relevance: X  
Factual Accuracy: X  
Completeness: X  
Source Attribution: X  
Clarity: X  
Explanation: <your explanation>  
Average Score: X
"""

eval_prompt = PromptTemplate.from_template(EVAL_PROMPT_TEMPLATE)

# ========= Refinement prompt =========
refinement_prompt = PromptTemplate.from_template("""
You are a senior medical editor. Improve the assistant's answer to the medical query below.

Focus on improving the following area(s) based on expert LLM feedback: {weak_dimensions}.

Ensure your answer is accurate, concise (≤5 sentences), context-based, and includes inline citations where appropriate (e.g., Hurst's the Heart, p.495). Do not add hallucinated content.

Query: {query}
Original Answer: {answer}

Sources:
{sources}

Refined Answer:
""")
