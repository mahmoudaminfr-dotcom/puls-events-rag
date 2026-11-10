"""
Tests unitaires pour le module d'évaluation du système RAG (src/evaluation.py).
Vérifie les métriques, les contre-exemples de refus, le contrôle des faits clés
(titre, date, lieu, lien) et la conformité du rapport d'évaluation.
"""

import json
from src.evaluation import (
    RESULTS_CSV_PATH,
    RESULTS_JSON_PATH,
    TEST_DATASET_PATH,
    check_refusal,
    compute_cosine_similarity,
    compute_keyword_recall,
    evaluate_key_facts,
)


def test_compute_cosine_similarity():
    """Vérifie le calcul mathématique de la similarité cosinus."""
    assert abs(compute_cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-6
    assert abs(compute_cosine_similarity([1.0, 0.0], [0.0, 1.0]) - 0.0) < 1e-6
    assert compute_cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_check_refusal_positive_and_counter_examples():
    """Vérifie les vrais refus et rejette les faux positifs contenant 'Paris' ou un lien."""
    assert check_refusal("Je ne dispose d'aucun événement à Lyon dans la base Puls-Events.") is True
    assert check_refusal("Cette demande est hors périmètre : Puls-Events couvre uniquement Paris.") is True

    # Contre-exemples : une recommandation valide citant "Paris" et "uniquement" n'est PAS un refus
    assert check_refusal("Voici un concert de jazz à Paris accessible uniquement sur réservation.") is False
    assert check_refusal("Paris accueille une exposition gratuite ce samedi à 20h.") is False

    # Contre-exemple : un faux refus recommandant malgré tout une fiche Open Agenda est rejeté
    assert check_refusal(
        "Je ne dispose pas de recette de crêpes, mais voici un atelier : https://openagenda.com/econature/events/test"
    ) is False


def test_evaluate_key_facts_and_misattribution():
    """Vérifie le contrôle des 4 faits clés (titre, date, lieu, URL) et la détection d'amalgames."""
    expected_schumann = {
        "title": "LE POINT DU JOUR | SCHUMANN | MAYER",
        "date": "22 avril 2027",
        "location": "Théâtre des Champs-Elysées",
        "url": "https://openagenda.com/shotgun/events/le-point-du-jour-or-schumann-or-mayer",
    }
    forbidden = ["faust de felix mendelssohn", "faust de mendelssohn", "mendelssohn (mayer)"]

    valid_answer = (
        "Titre : LE POINT DU JOUR | SCHUMANN | MAYER\n"
        "Dates : Jeudi 22 avril 2027 à 20h00\n"
        "Lieu : Théâtre des Champs-Élysées\n"
        "Programme : Faust-Ouverture d'Emilie Mayer et Concerto de Schumann.\n"
        "Lien : https://openagenda.com/shotgun/events/le-point-du-jour-or-schumann-or-mayer"
    )
    res_ok = evaluate_key_facts(valid_answer, expected_schumann, forbidden)
    assert res_ok["fact_score"] == 1.0
    assert res_ok["all_valid"] is True

    # Cas d'amalgame factuel (attribution erronée de Faust à Mendelssohn) -> doit échouer
    misattributed_answer = valid_answer.replace(
        "Faust-Ouverture d'Emilie Mayer",
        "Ouverture Faust de Felix Mendelssohn (Mayer)",
    )
    res_bad = evaluate_key_facts(misattributed_answer, expected_schumann, forbidden)
    assert res_bad["no_forbidden_claims"] is False
    assert res_bad["all_valid"] is False

    # Cas où le lien URL est omis -> fact_score = 0.75 et all_valid = False
    missing_url_answer = valid_answer.split("Lien :")[0]
    res_no_url = evaluate_key_facts(missing_url_answer, expected_schumann, forbidden)
    assert res_no_url["url_ok"] is False
    assert res_no_url["fact_score"] == 0.75
    assert res_no_url["all_valid"] is False


def test_keyword_recall_normalization():
    """Vérifie que le rappel de mots-clés gère les tirets et accents."""
    recall = compute_keyword_recall(
        "Exposition au 182 rue Saint-Honoré et au Théâtre des Champs-Élysées.",
        ["Saint Honoré", "Champs-Elysées", "Gratuit"],
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
    assert metrics["mean_fact_score"] >= 0.80
    assert metrics["mean_semantic_similarity_in_scope"] >= 0.80
    assert metrics["out_of_scope_guardrail_accuracy"] == 1.0
    assert metrics["overall_accuracy"] >= 0.80