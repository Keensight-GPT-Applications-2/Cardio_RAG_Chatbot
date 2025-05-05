# rag_pipeline.py

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

from src.prompt import rag_prompt, refinement_prompt, evaluation_prompt
from src.helper import clean_text

# Initialize LLMs
retrieval_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
judge_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
refine_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)

# Create document QA chain
question_answer_chain = create_stuff_documents_chain(retrieval_llm, rag_prompt)

# Define the RAG pipeline

def build_rag_pipeline(retriever):
    return create_retrieval_chain(retriever, question_answer_chain)

# Define refinement chain
refinement_chain = LLMChain(llm=refine_llm, prompt=refinement_prompt)

# Define evaluation function
from langchain.prompts import PromptTemplate
import re

def evaluate_answer(query, context_docs, answer):
    context_text = "\n\n".join([doc.page_content[:300] for doc in context_docs])
    eval_prompt = PromptTemplate.from_template(evaluation_prompt)
    prompt_input = eval_prompt.format(query=query, context=context_text, answer=answer)
    return judge_llm.invoke(prompt_input).content

def extract_score(text):
    scores = {}
    pattern = r"(Relevance|Factual Accuracy|Completeness|Source Attribution|Clarity):\\s*([0-5](\\.\\d+)?)"
    for match in re.findall(pattern, text, re.IGNORECASE):
        dimension = match[0].strip()
        score = float(match[1])
        scores[dimension] = score

    avg_match = re.search(r"Average Score:\\s*([0-5](\\.\\d+)?)", text, re.IGNORECASE)
    avg = float(avg_match.group(1)) if avg_match else None
    return scores, avg
