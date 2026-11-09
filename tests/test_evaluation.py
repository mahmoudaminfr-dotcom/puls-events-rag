"""
Tests unitaires pour le module d'évaluation du système RAG (src/evaluation.py).
Vérifie les fonctions de calcul de métriques et la conformité du rapport d'évaluation.
"""

import json
from src.evaluation import (
    compute_cosine_similarity,
    check_refusal,
    compute_keyword_recall,
    TEST_DATASET_PATH,
    RESULTS_JSON_PATH,
    RESULTS_CSV_PATH,
)


def test_compute_cosine_similarity():
    """Vérifie le calcul mathématique de la similarité cosinus."""
    assert abs(compute_cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-6
    assert abs(compute_cosine_similarity([1.0, 0.0], [0.0, 1.0]) - 0.0) < 1e-6
    assert compute_cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_check_refusal_and_keyword_recall():
    """Vérifie la détection du garde-fou et le calcul du rappel de mots-clés."""
    assert check_refusal("Je ne dispose d'aucun événement à Lyon.") is True
    assert check_refusal("Voici un super concert ce soir à 20h !") is False

    recall = compute_keyword_recall(
        "Concert de Jazzy Bazz au Zénith le 15 avril 2027.",
        ["JAZZY BAZZ", "Zénith", "Gratuit"],
    )
    assert abs(recall - 0.6667) < 1e-3
    assert compute_keyword_recall("Réponse libre", []) == 1.0


def test_evaluation_artifacts_and_quality_thresholds():
    """Vérifie l'existence des rapports d'évaluation et l'atteinte des seuils du POC."""
    assert TEST_DATASET_PATH.exists(), "Le fichier test_dataset.json est manquant."
    assert RESULTS_JSON_PATH.exists(), "Le rapport evaluation_results.json est manquant."
    assert RESULTS_CSV_PATH.exists(), "Le fichier evaluation_summary.csv est manquant."

    with open(RESULTS_JSON_PATH, "r", encoding="utf-8-sig") as f:
        report = json.load(f)

    metrics = report["metrics"]
    assert metrics["total_questions"] >= 10
    assert metrics["retrieval_hit_rate_at_k"] >= 0.80
    assert metrics["mean_semantic_similarity_in_scope"] >= 0.80
    assert metrics["out_of_scope_guardrail_accuracy"] == 1.0
    assert metrics["overall_accuracy"] >= 0.80
