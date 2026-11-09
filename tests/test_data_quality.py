import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.config import (
    FUTURE_DAYS,
    HISTORY_DAYS,
    PROCESSED_EVENTS_FILE,
    REFERENCE_DATE,
    TARGET_CITY,
)
from src.preprocessing import clean_html_and_whitespace, preprocess_events


def test_clean_html_and_whitespace():
    """Vérifie la suppression des balises HTML et la normalisation des listes."""
    raw_html = "<p>Concert &amp; <b>exposition</b>   à Paris</p>"
    assert clean_html_and_whitespace(raw_html) == "Concert & exposition à Paris"
    assert clean_html_and_whitespace(["jazz", "concert", None]) == "jazz, concert"
    assert clean_html_and_whitespace(None) == ""


def test_preprocess_events_filters_out_invalid_records():
    """Vérifie que les événements hors périmètre (ville ou > 1 an) sont exclus."""
    sample_records = [
        {
            "uid": "101",
            "title_fr": "Événement valide à Paris",
            "description_fr": "Description valide",
            "location_city": "Paris",
            "firstdate_begin": "2026-10-15T18:00:00+00:00",
            "lastdate_end": "2026-10-15T21:00:00+00:00",
        },
        {
            "uid": "101",  # Doublon d'UID
            "title_fr": "Événement valide à Paris (doublon)",
            "description_fr": "Description valide",
            "location_city": "Paris",
            "firstdate_begin": "2026-10-15T18:00:00+00:00",
        },
        {
            "uid": "102",  # Mauvaise ville
            "title_fr": "Concert à Lyon",
            "description_fr": "Hors périmètre géographique",
            "location_city": "Lyon",
            "firstdate_begin": "2026-10-15T18:00:00+00:00",
        },
        {
            "uid": "103",  # Événement trop ancien (> 1 an avant 2026-11-05)
            "title_fr": "Exposition ancienne",
            "description_fr": "Hors périmètre temporel",
            "location_city": "Paris",
            "firstdate_begin": "2024-05-10T10:00:00+00:00",
        },
    ]

    df_result = preprocess_events(
        raw_records=sample_records,
        target_city="Paris",
        reference_date="2026-11-05",
        history_days=365,
        future_days=180,
    )

    assert len(df_result) == 1
    assert df_result.iloc[0]["uid"] == "101"
    assert df_result.iloc[0]["location_city"] == "Paris"


def test_processed_dataset_exists_and_not_empty():
    """Vérifie que le fichier de données nettoyées existe et contient des événements."""
    assert PROCESSED_EVENTS_FILE.exists(), (
        f"Le fichier {PROCESSED_EVENTS_FILE} est introuvable. "
        "Exécutez python -m src.preprocessing."
    )
    df = pd.read_json(PROCESSED_EVENTS_FILE)
    assert not df.empty, "Le dataset nettoyé ne doit pas être vide."
    assert len(df) >= 100, "Le volume d'événements nettoyés est insuffisant."


def test_processed_dataset_geographic_scope():
    """Vérifie que 100% des événements nettoyés sont situés dans la ville cible."""
    df = pd.read_json(PROCESSED_EVENTS_FILE)
    cities = df["location_city"].astype(str).str.lower().unique()
    assert list(cities) == [TARGET_CITY.lower()], (
        f"Des villes hors périmètre ont été détectées : {cities}"
    )


def test_processed_dataset_temporal_scope_less_than_one_year():
    """Vérifie que tous les événements ont moins d'un an par rapport à la date de référence."""
    df = pd.read_json(PROCESSED_EVENTS_FILE)
    dates = pd.to_datetime(df["firstdate_begin"], utc=True, errors="coerce")

    assert dates.isna().sum() == 0, "Certaines dates de début sont invalides (NaT)."

    ref_dt = datetime.strptime(REFERENCE_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    min_allowed_dt = ref_dt - timedelta(days=HISTORY_DAYS)
    max_allowed_dt = ref_dt + timedelta(days=FUTURE_DAYS + 1)

    oldest_event = dates.min()
    newest_event = dates.max()

    assert oldest_event >= min_allowed_dt, (
        f"Un événement date de plus d'un an ({oldest_event} < {min_allowed_dt})."
    )
    assert newest_event <= max_allowed_dt, (
        f"Un événement dépasse la fenêtre future autorisée ({newest_event} > {max_allowed_dt})."
    )


def test_processed_dataset_uniqueness_and_text_integrity():
    """Vérifie l'absence de doublons et la présence du texte combiné pour FAISS."""
    df = pd.read_json(PROCESSED_EVENTS_FILE)
    assert df["uid"].is_unique, "Le dataset contient des identifiants (uid) en doublon."
    assert (df["combined_text"].astype(str).str.strip() != "").all(), (
        "Certains événements ont un champ combined_text vide."
    )
