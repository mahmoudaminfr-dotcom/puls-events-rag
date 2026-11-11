import html
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.config import (
    FUTURE_DAYS,
    HISTORY_DAYS,
    PROCESSED_EVENTS_CSV,
    PROCESSED_EVENTS_FILE,
    RAW_EVENTS_FILE,
    REFERENCE_DATE,
    TARGET_CITY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_HAS_EXPLICIT_HOUR_RE = re.compile(r"(?<!\d)(?:[01]?\d|2[0-3])\s*[hH:]")


def clean_html_and_whitespace(text: Any) -> str:
    """Supprime les balises HTML, décode les entités et normalise les espaces."""
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        text = ", ".join(str(item) for item in text if item is not None and str(item).strip())
    elif pd.isna(text):
        return ""

    text_str = html.unescape(str(text))
    if not text_str.strip():
        return ""
    text_str = re.sub(r"<[^>]+>", " ", text_str)
    text_str = re.sub(r"\s+", " ", text_str)
    return text_str.strip()


def to_paris_iso(value: Any) -> str:
    """Convertit un horodatage ISO (UTC ou avec offset) vers le fuseau Europe/Paris."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(ts):
        return raw
    return ts.tz_convert("Europe/Paris").isoformat()


def format_paris_schedule(daterange_fr: str, first_paris_iso: str) -> str:
    """Construit le libellé d'horaires en heure locale de Paris sans fuite UTC."""
    dr = (daterange_fr or "").strip()
    iso_str = to_paris_iso(first_paris_iso) if first_paris_iso else ""

    paris_hour = ""
    if iso_str and "T" in iso_str:
        try:
            dt = datetime.fromisoformat(iso_str)
            if dt.hour >= 6:
                paris_hour = f"{dt.hour:02d}h{dt.minute:02d}"
        except ValueError:
            paris_hour = ""

    if dr:
        if _HAS_EXPLICIT_HOUR_RE.search(dr) or not paris_hour:
            return dr
        return f"{dr} (à {paris_hour}, heure de Paris)"

    if iso_str:
        if paris_hour:
            return f"{iso_str[:10]} à {paris_hour}"
        return iso_str[:10]
    return ""


def build_combined_text(row: pd.Series) -> str:
    """Construit le texte complet structuré d'un événement pour la vectorisation."""
    parts = [
        f"Titre : {row['title_fr']}",
        f"Ville : {row['location_city']}",
    ]
    if row.get("location_name"):
        parts.append(f"Lieu : {row['location_name']}")
    if row.get("location_address"):
        parts.append(
            f"Adresse : {row['location_address']} {row.get('location_postalcode', '')}".strip()
        )
    schedule_paris = format_paris_schedule(
        str(row.get("daterange_fr") or ""),
        str(row.get("firstdate_begin") or ""),
    )
    if schedule_paris:
        parts.append(f"Horaires (heure de Paris) : {schedule_paris}")
    if row.get("category"):
        parts.append(f"Catégorie : {row['category']}")
    if row.get("conditions_fr"):
        parts.append(f"Conditions / Tarifs : {row['conditions_fr']}")
    if row.get("keywords_fr"):
        parts.append(f"Mots-clés : {row['keywords_fr']}")
    if row.get("description_fr"):
        parts.append(f"Résumé : {row['description_fr']}")
    if row.get("longdescription_fr"):
        parts.append(f"Description détaillée : {row['longdescription_fr']}")

    return "\n".join(parts)


def preprocess_events(
    raw_records: List[Dict[str, Any]],
    target_city: str = TARGET_CITY,
    reference_date: str = REFERENCE_DATE,
    history_days: int = HISTORY_DAYS,
    future_days: int = FUTURE_DAYS,
) -> pd.DataFrame:
    """Nettoie, filtre et structure les données brutes d'événements."""
    if not raw_records:
        raise ValueError("La liste d'événements bruts est vide.")

    df = pd.DataFrame(raw_records)
    initial_count = len(df)

    # 1. Sélection et initialisation des colonnes utiles
    selected_cols = [
        "uid",
        "title_fr",
        "description_fr",
        "longdescription_fr",
        "conditions_fr",
        "keywords_fr",
        "daterange_fr",
        "firstdate_begin",
        "lastdate_end",
        "location_name",
        "location_address",
        "location_postalcode",
        "location_city",
        "canonicalurl",
        "category",
    ]
    for col in selected_cols:
        if col not in df.columns:
            df[col] = ""
    df = df[selected_cols].copy()

    # 2. Suppression des doublons sur l'identifiant unique
    df["uid"] = df["uid"].astype(str)
    df = df.drop_duplicates(subset=["uid"], keep="first")

    # 3. Nettoyage des champs textuels
    text_columns = [
        "title_fr",
        "description_fr",
        "longdescription_fr",
        "conditions_fr",
        "keywords_fr",
        "daterange_fr",
        "location_name",
        "location_address",
        "location_postalcode",
        "location_city",
        "canonicalurl",
        "category",
    ]
    for col in text_columns:
        df[col] = df[col].apply(clean_html_and_whitespace)

    # 4. Suppression des événements sans titre ou sans description
    df = df[(df["title_fr"] != "") & (df["description_fr"] != "")].copy()

    # 5. Filtrage géographique strict (insensible à la casse)
    df = df[df["location_city"].str.lower() == target_city.lower()].copy()

    # 6. Conversion et filtrage temporel strict (< 1 an d'historique)
    df["firstdate_begin_dt"] = pd.to_datetime(
        df["firstdate_begin"], utc=True, errors="coerce"
    )
    df["lastdate_end_dt"] = pd.to_datetime(
        df["lastdate_end"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=["firstdate_begin_dt"]).copy()

    ref_dt = datetime.strptime(reference_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    min_dt = ref_dt - timedelta(days=history_days)
    max_dt = ref_dt + timedelta(days=future_days + 1)

    df = df[
        (df["firstdate_begin_dt"] >= min_dt) & (df["firstdate_begin_dt"] <= max_dt)
    ].copy()

    # Conversion explicite des horodatages ISO vers le fuseau Europe/Paris (+01:00 / +02:00)
    paris_begin = df["firstdate_begin_dt"].dt.tz_convert("Europe/Paris")
    paris_end = df["lastdate_end_dt"].dt.tz_convert("Europe/Paris")
    df["firstdate_begin"] = paris_begin.apply(
        lambda dt: dt.isoformat() if pd.notna(dt) else ""
    )
    df["lastdate_end"] = paris_end.apply(
        lambda dt: dt.isoformat() if pd.notna(dt) else ""
    )
    df = df.drop(columns=["firstdate_begin_dt", "lastdate_end_dt"])

    # 7. Création du texte enrichi pour le chunking et la vectorisation
    df["combined_text"] = df.apply(build_combined_text, axis=1)

    df = df.reset_index(drop=True)
    logger.info(
        f"Pré-processing terminé : {len(df)} événements valides conservés "
        f"(sur {initial_count} bruts)."
    )
    return df


def run_preprocessing(
    input_path: Path = RAW_EVENTS_FILE,
    output_json: Path = PROCESSED_EVENTS_FILE,
    output_csv: Path = PROCESSED_EVENTS_CSV,
) -> pd.DataFrame:
    """Charge le JSON brut, exécute le nettoyage et sauvegarde en JSON et CSV."""
    with open(input_path, "r", encoding="utf-8") as file:
        raw_records = json.load(file)

    df_cleaned = preprocess_events(raw_records)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_json(output_json, orient="records", force_ascii=False, indent=2)
    df_cleaned.to_csv(output_csv, index=False, encoding="utf-8")

    logger.info(f"Données nettoyées sauvegardées dans {output_json} et {output_csv}")
    return df_cleaned


if __name__ == "__main__":
    run_preprocessing()
