"""
Module d'évaluation quantitative et qualitative du système RAG Puls-Events.
Calcule le Hit Rate @ K (Retrieval), le rappel des mots-clés, la similarité cosinus
sémantique (Mistral Embeddings) et le respect des garde-fous hors périmètre.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from langchain_mistralai import MistralAIEmbeddings

import src.config as config
from src.rag_engine import PulsEventsRAG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MISTRAL_API_KEY = getattr(config, "MISTRAL_API_KEY", None)
MISTRAL_EMBED_MODEL = getattr(config, "MISTRAL_EMBED_MODEL", "mistral-embed")

EVAL_DIR = Path("data/evaluation")
TEST_DATASET_PATH = EVAL_DIR / "test_dataset.json"
RESULTS_JSON_PATH = EVAL_DIR / "evaluation_results.json"
RESULTS_CSV_PATH = EVAL_DIR / "evaluation_summary.csv"

REFUSAL_INDICATORS = [
    "aucun",
    "ne dispose",
    "pas d'événement",
    "ne figure",
    "ne correspond",
    "uniquement",
    "paris",
    "désolé",
    "hors périmètre",
    "ne peux pas",
]


def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calcule la similarité cosinus entre deux vecteurs d'embedding."""
    a = np.array(vec1, dtype=np.float64)
    b = np.array(vec2, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def check_refusal(answer: str) -> bool:
    """Vérifie si la réponse contient une formule de refus pour les questions hors périmètre."""
    answer_lower = answer.lower()
    return any(indicator in answer_lower for indicator in REFUSAL_INDICATORS)


def compute_keyword_recall(answer: str, expected_keywords: List[str]) -> float:
    """Calcule la proportion de mots-clés attendus présents dans la réponse générée."""
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return round(matches / len(expected_keywords), 4)


def run_evaluation(
    dataset_path: Path = TEST_DATASET_PATH,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Exécute le benchmark complet sur le jeu de données test annoté."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Jeu de test introuvable : {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8-sig") as f:
        test_cases = json.load(f)

    rag = PulsEventsRAG(top_k=top_k)
    embeddings_model = MistralAIEmbeddings(
        model=MISTRAL_EMBED_MODEL,
        api_key=MISTRAL_API_KEY,
    )

    raw_runs = []
    logger.info(f"Démarrage de l'évaluation RAG sur {len(test_cases)} questions (top_k={top_k})...")

    for idx, item in enumerate(test_cases, start=1):
        q_id = item["id"]
        category = item["category"]
        question = item["question"]

        logger.info(f"[{idx}/{len(test_cases)}] Évaluation de {q_id} ({category}) : {question}")
        rag_output = rag.ask(question, k=top_k)

        raw_runs.append(
            {
                "item": item,
                "generated_answer": rag_output["answer"],
                "retrieved_uids": [str(src["uid"]) for src in rag_output["sources"]],
            }
        )

        if idx < len(test_cases):
            time.sleep(4.0)

    logger.info("Calcul groupé (batch) des embeddings pour la similarité sémantique...")
    time.sleep(2.0)
    ground_truths = [r["item"]["ground_truth"] for r in raw_runs]
    generated_answers = [r["generated_answer"] for r in raw_runs]
    all_vectors = embeddings_model.embed_documents(ground_truths + generated_answers)

    n = len(raw_runs)
    gt_vectors = all_vectors[:n]
    gen_vectors = all_vectors[n:]

    detailed_results = []
    for idx, run in enumerate(raw_runs):
        item = run["item"]
        category = item["category"]
        generated_answer = run["generated_answer"]
        retrieved_uids = run["retrieved_uids"]
        expected_uids = [str(u) for u in item.get("expected_uids", [])]
        expected_keywords = item.get("expected_keywords", [])

        semantic_sim = round(
            compute_cosine_similarity(gt_vectors[idx], gen_vectors[idx]), 4
        )

        if category == "in_scope":
            hit_rate = 1.0 if any(uid in retrieved_uids for uid in expected_uids) else 0.0
            kw_recall = compute_keyword_recall(generated_answer, expected_keywords)
            is_correct = (hit_rate == 1.0) and (semantic_sim >= 0.70)
        else:
            hit_rate = None
            kw_recall = None
            is_correct = check_refusal(generated_answer)

        detailed_results.append(
            {
                "id": item["id"],
                "category": category,
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "generated_answer": generated_answer,
                "expected_uids": expected_uids,
                "retrieved_uids": retrieved_uids,
                "hit_rate": hit_rate,
                "keyword_recall": kw_recall,
                "semantic_similarity": semantic_sim,
                "is_correct": is_correct,
            }
        )

    in_scope_items = [r for r in detailed_results if r["category"] == "in_scope"]
    out_of_scope_items = [r for r in detailed_results if r["category"] == "out_of_scope"]

    summary_metrics = {
        "total_questions": len(detailed_results),
        "in_scope_count": len(in_scope_items),
        "out_of_scope_count": len(out_of_scope_items),
        "retrieval_hit_rate_at_k": round(
            float(np.mean([r["hit_rate"] for r in in_scope_items])), 4
        ),
        "mean_keyword_recall": round(
            float(np.mean([r["keyword_recall"] for r in in_scope_items])), 4
        ),
        "mean_semantic_similarity_in_scope": round(
            float(np.mean([r["semantic_similarity"] for r in in_scope_items])), 4
        ),
        "mean_semantic_similarity_overall": round(
            float(np.mean([r["semantic_similarity"] for r in detailed_results])), 4
        ),
        "out_of_scope_guardrail_accuracy": round(
            float(np.mean([1.0 if r["is_correct"] else 0.0 for r in out_of_scope_items])), 4
        ),
        "overall_accuracy": round(
            float(np.mean([1.0 if r["is_correct"] else 0.0 for r in detailed_results])), 4
        ),
    }

    report = {
        "metrics": summary_metrics,
        "results": detailed_results,
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    df_summary = pd.DataFrame(
        [
            {
                "id": r["id"],
                "category": r["category"],
                "hit_rate": r["hit_rate"],
                "keyword_recall": r["keyword_recall"],
                "semantic_similarity": r["semantic_similarity"],
                "is_correct": r["is_correct"],
                "question": r["question"],
            }
            for r in detailed_results
        ]
    )
    df_summary.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8")

    return report


if __name__ == "__main__":
    eval_report = run_evaluation(top_k=5)
    m = eval_report["metrics"]

    print("\n" + "=" * 80)
    print("SYNTHÈSE DE L'ÉVALUATION DU SYSTÈME RAG PULS-EVENTS")
    print("=" * 80)
    print(f" - Questions évaluées              : {m['total_questions']} ({m['in_scope_count']} in-scope, {m['out_of_scope_count']} out-of-scope)")
    print(f" - Hit Rate @ 5 (Retrieval FAISS)  : {m['retrieval_hit_rate_at_k'] * 100:.1f}%")
    print(f" - Rappel moyen des mots-clés      : {m['mean_keyword_recall'] * 100:.1f}%")
    print(f" - Similarité sémantique (in-scope): {m['mean_semantic_similarity_in_scope']:.4f}")
    print(f" - Garde-fou hors périmètre        : {m['out_of_scope_guardrail_accuracy'] * 100:.1f}%")
    print(f" - Taux de réponses correctes      : {m['overall_accuracy'] * 100:.1f}%")
    print("=" * 80)
