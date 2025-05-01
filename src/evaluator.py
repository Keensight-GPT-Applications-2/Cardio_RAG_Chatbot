import os
import json
from datetime import datetime
from src.helper import save_json, load_json
from src.prompt import refinement_chain
from src.rag_pipeline import build_rag_pipeline, evaluate_answer, extract_score

# Optional refinement toggle
ENABLE_REFINEMENT = True

ALL_LOGS_FILE = "research/comparison_results.json"

def refine_answer_smart(query, answer, sources, weak_dimensions):
    weak_string = ", ".join(weak_dimensions)
    source_text = "\n".join([f"- {s['source']}, page {s['page']}" for s in sources])
    return refinement_chain.invoke({
        "query": query,
        "answer": answer,
        "sources": source_text,
        "weak_dimensions": weak_string
    })["text"]

def compare_k_results_with_judge(query, retriever, k_values=[4, 6, 8, 10]):
    rag_chain, judge_llm = build_rag_pipeline(retriever)
    local_log = []
    best_result = None
    highest_score = (-1, -1)  # (average, completeness)

    for k in k_values:
        print(f"\n====================\nRunning with k = {k}\n====================")
        retrieved_docs = retriever.get_relevant_documents(query, k=k)

        # Deduplicate by (source, page)
        seen = set()
        unique_docs = []
        for doc in retrieved_docs:
            key = (doc.metadata.get("source"), doc.metadata.get("page"))
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        # Run RAG chain
        input_payload = {"input": query, "context": unique_docs}
        result = rag_chain.invoke(input_payload)

        # Evaluate answer
        evaluation = evaluate_answer(query, unique_docs, result, judge_llm)
        scores_dict, avg_score = extract_score(evaluation)
        completeness_score = scores_dict.get("Completeness", 0)
        score_tuple = (avg_score, completeness_score)

        sources = [
            {
                "source": doc.metadata.get("source", "Unknown").split("\\")[-1],
                "page": int(doc.metadata.get("page", 0))
            }
            for doc in unique_docs
        ]

        print(f"\n📊 Evaluation for k={k}")
        print("Answer:\n", result)
        print("LLM Judge Feedback:\n", evaluation)
        print(f"📈 Scores: {scores_dict}")
        print(f"🏁 Avg Score: {avg_score}")

        refined_answer = None

        if score_tuple > highest_score:
            highest_score = score_tuple
            best_result = {
                "k": k,
                "answer": result,
                "score": avg_score,
                "dimension_scores": scores_dict,
                "sources": sources
            }

        local_log.append({
            "query": query,
            "k": k,
            "answer": result,
            "llm_judge_feedback": evaluation,
            "llm_judge_score": {
                "dimension_scores": scores_dict,
                "average": avg_score
            },
            "sources": sources,
            "refined_answer": None
        })

    print(f"\n🏆 Best k based on judge score: {best_result['k']} (Score: {best_result['score']})")
    print("\n📋 Best Answer Before Refinement:\n", best_result["answer"])

    # Apply refinement if needed
    weak_areas = [k for k, v in best_result["dimension_scores"].items() if v < 5.0]
    refined_answer = None
    if ENABLE_REFINEMENT and weak_areas:
        print(f"\n🔧 Refining answer based on weaknesses: {weak_areas}...")
        refined_answer = refine_answer_smart(
            query=query,
            answer=best_result["answer"],
            sources=best_result["sources"],
            weak_dimensions=weak_areas
        )
        print("\n🧪 Refined Answer:\n", refined_answer)
    elif ENABLE_REFINEMENT:
        print("\n✅ No refinement needed. All scores are 5.0.")

    # Attach refined answer to the best result
    best_result["refined_answer"] = refined_answer

    for entry in local_log:
        if entry["k"] == best_result["k"]:
            entry["refined_answer"] = refined_answer
            break

    all_logs = load_json(ALL_LOGS_FILE)
    all_logs.extend(local_log)
    save_json(ALL_LOGS_FILE, all_logs)

    print(f"\n✅ All results for '{query}' saved in: {ALL_LOGS_FILE}")
    return best_result
