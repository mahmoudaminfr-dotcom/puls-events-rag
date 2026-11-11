"""
Module d'évaluation quantitative et qualitative du système RAG Puls-Events.
Calcule le Hit Rate @ K (Retrieval), l'exactitude des 5 faits clés (titre, date, heure, lieu, URL),
le rappel des mots-clés, la similarité cosinus sémantique et la rigueur des refus.
"""

import json
import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from langchain_mistralai import MistralAIEmbeddings

import src.config as config
from src.rag_engine import PulsEventsRAG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MISTRAL_EMBED_MODEL = getattr(config, "EMBEDDING_MODEL", "mistral-embed")

EVAL_DIR = Path("data/evaluation")
TEST_DATASET_PATH = EVAL_DIR / "test_dataset.json"
RESULTS_JSON_PATH = EVAL_DIR / "evaluation_results.json"
RESULTS_CSV_PATH = EVAL_DIR / "evaluation_summary.csv"

# Formules explicites de refus (sans mots ambigus tels que "paris" ou "uniquement")
REFUSAL_PHRASES = [
    "ne dispose d'aucun",
    "ne dispose pas",
    "aucun evenement",
    "pas d'evenement",
    "ne figure pas",
    "aucun resultat",
    "hors perimetre",
    "hors de mon perimetre",
    "ne peux pas repondre",
    "ne peux pas vous fournir",
    "ne suis pas en mesure",
    "impossible de repondre",
]


def normalize_text(text: str) -> str:
    """Normalise un texte (minuscules, sans accents, ligatures ni ponctuation) pour comparaison fiable."""
    if not text:
        return ""
    text = text.lower().replace("œ", "oe").replace("æ", "ae")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"['’‘`\-–—_|,:;!?()\[\]*\"«»@]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_hours_minutes(text: str) -> List[str]:
    """
    Extrait et normalise tous les horaires présents dans un texte au format canonique 'HHhMM'
    (ex: '20h', '20 h', '20 h 00', '20:00', '20 heures' -> '20h00' ; '8 h 30', '08:30' -> '08h30').
    """
    if not text:
        return []

    cleaned = re.sub(r"https?://\S+", " ", text)
    found: List[str] = []

    patterns = [
        r"(?<!\d)([01]?\d|2[0-3])\s*[hH](?:\s*([0-5]\d))?(?![0-9a-zA-ZÀ-ÿ])",
        r"(?<!\d)([01]?\d|2[0-3])\s+heures?(?:\s*([0-5]\d))?(?!\d)",
        r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, cleaned, flags=re.IGNORECASE):
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) is not None else 0
            canonical = f"{hour:02d}h{minute:02d}"
            if canonical not in found:
                found.append(canonical)

    return found


def check_time_match(
    answer: str,
    expected_time: Any,
    forbidden_times: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Vérifie la conformité des horaires mentionnés dans la réponse :
    - présence de tous les horaires attendus en heure de Paris (si définis) ;
    - absence totale d'horaires UTC non convertis (forbidden_times).
    """
    extracted_times = extract_hours_minutes(answer)
    forbidden_canonical: List[str] = []
    for ft in (forbidden_times or []):
        forbidden_canonical.extend(extract_hours_minutes(str(ft)))

    no_forbidden_times = not any(t in extracted_times for t in forbidden_canonical)

    if expected_time is None or expected_time == "" or expected_time == []:
        expected_canonical: List[str] = []
        expected_present = True
    else:
        raw_expected = (
            expected_time if isinstance(expected_time, list) else [expected_time]
        )
        expected_canonical = []
        for et in raw_expected:
            expected_canonical.extend(extract_hours_minutes(str(et)))
        expected_present = bool(expected_canonical) and all(
            t in extracted_times for t in expected_canonical
        )

    return {
        "extracted_times": extracted_times,
        "expected_times": expected_canonical,
        "expected_time_present": expected_present,
        "no_forbidden_times": no_forbidden_times,
    }


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
    """
    Vérifie si la réponse constitue un vrai refus hors périmètre :
    - présence d'une expression explicite de refus (sans faux positif sur 'Paris' ou 'uniquement') ;
    - absence de lien d'événement Open Agenda recommandé en substitution.
    """
    if not answer or not answer.strip():
        return False
    answer_lower = answer.lower()
    if "openagenda.com/" in answer_lower:
        return False

    norm_refusal = answer_lower.replace("œ", "oe").replace("’", "'").replace("‘", "'")
    norm_refusal = unicodedata.normalize("NFD", norm_refusal)
    norm_refusal = "".join(ch for ch in norm_refusal if unicodedata.category(ch) != "Mn")
    norm_refusal = re.sub(r"\s+", " ", norm_refusal).strip()

    return any(phrase in norm_refusal for phrase in REFUSAL_PHRASES)


def compute_keyword_recall(answer: str, expected_keywords: List[str]) -> float:
    """Calcule la proportion de mots-clés attendus présents dans la réponse normalisée."""
    if not expected_keywords:
        return 1.0
    norm_answer = normalize_text(answer)
    matches = sum(1 for kw in expected_keywords if normalize_text(kw) in norm_answer)
    return round(matches / len(expected_keywords), 4)


def evaluate_key_facts(
    answer: str,
    expected_facts: Dict[str, Any],
    forbidden_claims: Optional[List[str]] = None,
    forbidden_times: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Vérifie la présence des 5 faits clés obligatoires (titre, date, heure, lieu, lien URL)
    ainsi que l'absence d'amalgames (forbidden_claims) et d'horaires UTC décalés (forbidden_times).
    """
    norm_answer = normalize_text(answer)
    answer_lower = (answer or "").lower()
    forbidden_claims = forbidden_claims or []

    title_val = expected_facts.get("title", "")
    date_val = expected_facts.get("date", "")
    time_val = expected_facts.get("time")
    location_val = expected_facts.get("location", "")
    url_val = expected_facts.get("url", "")

    title_ok = bool(title_val) and (normalize_text(title_val) in norm_answer)
    date_ok = bool(date_val) and (normalize_text(date_val) in norm_answer)
    location_ok = bool(location_val) and (normalize_text(location_val) in norm_answer)
    url_ok = bool(url_val) and (url_val.strip().lower() in answer_lower)

    time_info = check_time_match(answer, time_val, forbidden_times)
    no_forbidden_times = time_info["no_forbidden_times"]
    if time_val is None or time_val == "" or time_val == []:
        time_ok = date_ok and no_forbidden_times
    else:
        time_ok = time_info["expected_time_present"] and no_forbidden_times

    no_forbidden = not any(
        normalize_text(claim) in norm_answer for claim in forbidden_claims if claim
    )

    fact_score = round(
        (
            int(title_ok)
            + int(date_ok)
            + int(time_ok)
            + int(location_ok)
            + int(url_ok)
        )
        / 5.0,
        4,
    )
    all_valid = (
        title_ok
        and date_ok
        and time_ok
        and location_ok
        and url_ok
        and no_forbidden
        and no_forbidden_times
    )

    return {
        "title_ok": title_ok,
        "date_ok": date_ok,
        "time_ok": time_ok,
        "location_ok": location_ok,
        "url_ok": url_ok,
        "extracted_times": time_info["extracted_times"],
        "no_forbidden_claims": no_forbidden,
        "no_forbidden_times": no_forbidden_times,
        "fact_score": fact_score,
        "all_valid": all_valid,
    }


def run_evaluation(
    dataset_path: Path = TEST_DATASET_PATH,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Exécute le benchmark complet sur le jeu de données test annoté."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Jeu de test introuvable : {dataset_path}")

    api_key = getattr(config, "MISTRAL_API_KEY", None)
    if not api_key or api_key == "your_mistral_api_key_here":
        raise ValueError("MISTRAL_API_KEY est manquante dans la configuration.")

    with open(dataset_path, "r", encoding="utf-8-sig") as f:
        test_cases = json.load(f)

    rag = PulsEventsRAG(top_k=top_k)
    embeddings_model = MistralAIEmbeddings(
        model=MISTRAL_EMBED_MODEL,
        api_key=api_key,
    )

    raw_runs = []
    logger.info(
        f"Démarrage de l'évaluation RAG sur {len(test_cases)} questions (top_k={top_k})..."
    )

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
        expected_facts = item.get("expected_facts", {})
        forbidden_claims = item.get("forbidden_claims", [])
        forbidden_times = item.get("forbidden_times", [])

        semantic_sim = round(
            compute_cosine_similarity(gt_vectors[idx], gen_vectors[idx]), 4
        )

        if category == "in_scope":
            hit_rate = 1.0 if any(uid in retrieved_uids for uid in expected_uids) else 0.0
            kw_recall = compute_keyword_recall(generated_answer, expected_keywords)
            fact_checks = evaluate_key_facts(
                generated_answer,
                expected_facts,
                forbidden_claims,
                forbidden_times,
            )
            fact_score = fact_checks["fact_score"]
            is_correct = (
                hit_rate == 1.0
                and fact_checks["all_valid"]
                and not check_refusal(generated_answer)
                and semantic_sim >= 0.70
            )
        else:
            hit_rate = None
            kw_recall = None
            fact_checks = None
            fact_score = None
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
                "fact_score": fact_score,
                "fact_checks": fact_checks,
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
        "mean_fact_score": round(
            float(np.mean([r["fact_score"] for r in in_scope_items])), 4
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
                "fact_score": r["fact_score"],
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
    print(f" - Exactitude des faits clés (5/5) : {m['mean_fact_score'] * 100:.1f}%")
    print(f" - Rappel moyen des mots-clés      : {m['mean_keyword_recall'] * 100:.1f}%")
    print(f" - Similarité sémantique (in-scope): {m['mean_semantic_similarity_in_scope']:.4f}")
    print(f" - Garde-fou hors périmètre        : {m['out_of_scope_guardrail_accuracy'] * 100:.1f}%")
    print(f" - Taux de réponses correctes      : {m['overall_accuracy'] * 100:.1f}%")
    print("=" * 80)
