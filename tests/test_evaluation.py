"""
Tests unitaires pour le module d'évaluation du système RAG (src/evaluation.py).
Vérifie les métriques, les contre-exemples de refus, le contrôle des 5 faits clés
(titre, date, heure de Paris, lieu, lien) et la conformité du rapport d'évaluation.
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
    extract_hours_minutes,
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
    """Vérifie le contrôle des 5 faits clés (titre, date, heure, lieu, URL) et la détection d'amalgames."""
    expected_schumann = {
        "title": "LE POINT DU JOUR | SCHUMANN | MAYER",
        "date": "22 avril 2027",
        "time": "20h00",
        "location": "Théâtre des Champs-Elysées",
        "url": "https://openagenda.com/shotgun/events/le-point-du-jour-or-schumann-or-mayer",
    }
    forbidden = ["faust de felix mendelssohn", "faust de mendelssohn", "mendelssohn (mayer)"]

    valid_answer = (
        "Titre : LE POINT DU JOUR | SCHUMANN | MAYER\n"
        "Horaires : Jeudi 22 avril 2027 à 20h00\n"
        "Lieu : Théâtre des Champs-Élysées\n"
        "Programme : Faust-Ouverture d'Emilie Mayer et Concerto de Schumann.\n"
        "Lien : https://openagenda.com/shotgun/events/le-point-du-jour-or-schumann-or-mayer"
    )
    res_ok = evaluate_key_facts(valid_answer, expected_schumann, forbidden, ["18h00"])
    assert res_ok["fact_score"] == 1.0
    assert res_ok["all_valid"] is True

    # Cas d'amalgame factuel (attribution erronée de Faust à Mendelssohn) -> doit échouer
    misattributed_answer = valid_answer.replace(
        "Faust-Ouverture d'Emilie Mayer",
        "Ouverture Faust de Felix Mendelssohn (Mayer)",
    )
    res_bad = evaluate_key_facts(misattributed_answer, expected_schumann, forbidden, ["18h00"])
    assert res_bad["no_forbidden_claims"] is False
    assert res_bad["all_valid"] is False

    # Cas où le lien URL est omis -> fact_score = 0.8 (4/5) et all_valid = False
    missing_url_answer = valid_answer.split("Lien :")[0]
    res_no_url = evaluate_key_facts(missing_url_answer, expected_schumann, forbidden, ["18h00"])
    assert res_no_url["url_ok"] is False
    assert res_no_url["fact_score"] == 0.8
    assert res_no_url["all_valid"] is False


def test_time_extraction_and_paris_hour_validation():
    """Vérifie le rejet des heures UTC décalées (19h au lieu de 20h sur Q05, 8h30 au lieu de 9h30 sur Q06)."""
    assert extract_hours_minutes("Concert à 20 h, portes à 19h30 ou 21:30 (19 habitants).") == [
        "20h00",
        "19h30",
        "21h30",
    ]

    # Cas Q05 : The Clash (20h00 heure de Paris vs 19h00 UTC)
    q05_facts = {
        "title": "The Last Band In Town : Hommage à The Clash",
        "date": "20 mars 2027",
        "time": "20h00",
        "location": "Théâtre Traversière",
        "url": "https://openagenda.com/shotgun/events/the-last-band-in-town-hommage-a-the-clash",
    }
    q05_utc_wrong = (
        "The Last Band In Town : Hommage à The Clash a lieu au Théâtre Traversière "
        "le samedi 20 mars 2027 à 19 h. "
        "Lien : https://openagenda.com/shotgun/events/the-last-band-in-town-hommage-a-the-clash"
    )
    res_q05_bad = evaluate_key_facts(q05_utc_wrong, q05_facts, [], ["19h00"])
    assert res_q05_bad["time_ok"] is False
    assert res_q05_bad["no_forbidden_times"] is False
    assert res_q05_bad["fact_score"] == 0.8
    assert res_q05_bad["all_valid"] is False

    q05_paris_ok = q05_utc_wrong.replace("19 h", "20 h")
    res_q05_ok = evaluate_key_facts(q05_paris_ok, q05_facts, [], ["19h00"])
    assert res_q05_ok["time_ok"] is True
    assert res_q05_ok["fact_score"] == 1.0
    assert res_q05_ok["all_valid"] is True

    # Cas Q06 : Musée Postal (09h30 heure de Paris vs 08h30 UTC)
    q06_facts = {
        "title": "Collecte de photographies argentiques",
        "date": "15 mars 2027",
        "time": "09h30",
        "location": "Musée Postal",
        "url": "https://openagenda.com/bicentenaire-photographie-fr/events/collecte-de-photographies-argentiques",
    }
    q06_utc_wrong = (
        "La Collecte de photographies argentiques est organisée au Musée Postal "
        "le lundi 15 mars 2027 à 8 h 30. "
        "Lien : https://openagenda.com/bicentenaire-photographie-fr/events/collecte-de-photographies-argentiques"
    )
    res_q06_bad = evaluate_key_facts(q06_utc_wrong, q06_facts, [], ["08h30"])
    assert res_q06_bad["time_ok"] is False
    assert res_q06_bad["no_forbidden_times"] is False
    assert res_q06_bad["fact_score"] == 0.8
    assert res_q06_bad["all_valid"] is False

    q06_paris_ok = q06_utc_wrong.replace("8 h 30", "9 h 30")
    res_q06_ok = evaluate_key_facts(q06_paris_ok, q06_facts, [], ["08h30"])
    assert res_q06_ok["time_ok"] is True
    assert res_q06_ok["fact_score"] == 1.0
    assert res_q06_ok["all_valid"] is True


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

    in_scope_results = [r for r in report["results"] if r["category"] == "in_scope"]
    assert all("time_ok" in r["fact_checks"] for r in in_scope_results)
    assert all("no_forbidden_times" in r["fact_checks"] for r in in_scope_results)
