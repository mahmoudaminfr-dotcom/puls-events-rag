import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import requests

from src.config import (
    FUTURE_DAYS,
    HISTORY_DAYS,
    MAX_EVENTS,
    OPENAGENDA_API_URL,
    RAW_EVENTS_FILE,
    REFERENCE_DATE,
    TARGET_CITY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_odsql_where_clause(
    city: str,
    reference_date: str = REFERENCE_DATE,
    history_days: int = HISTORY_DAYS,
    future_days: int = FUTURE_DAYS,
) -> str:
    """Construit la clause WHERE ODSQL figée autour de la date de référence."""
    ref_dt = datetime.strptime(reference_date, "%Y-%m-%d")
    min_date = (ref_dt - timedelta(days=history_days)).strftime("%Y-%m-%d")
    max_date = (ref_dt + timedelta(days=future_days)).strftime("%Y-%m-%d")

    return (
        f'location_city="{city}" '
        f'AND firstdate_begin >= "{min_date}" '
        f'AND firstdate_begin <= "{max_date}" '
        f"AND description_fr IS NOT NULL"
    )


def fetch_openagenda_events(
    city: str = TARGET_CITY,
    reference_date: str = REFERENCE_DATE,
    history_days: int = HISTORY_DAYS,
    future_days: int = FUTURE_DAYS,
    max_events: int = MAX_EVENTS,
) -> List[Dict[str, Any]]:
    """Récupère les événements depuis l'API Open Agenda par pagination déterministe."""
    t0 = time.perf_counter()
    where_clause = build_odsql_where_clause(
        city=city,
        reference_date=reference_date,
        history_days=history_days,
        future_days=future_days,
    )
    all_records: List[Dict[str, Any]] = []
    batch_size = 100
    offset = 0

    logger.info(
        f"Extraction Open Agenda (Ville: {city}, Date réf: {reference_date}, "
        f"Fenêtre: -{history_days}j / +{future_days}j, Max: {max_events})"
    )

    while len(all_records) < max_events:
        limit = min(batch_size, max_events - len(all_records))
        params = {
            "where": where_clause,
            "order_by": "firstdate_begin DESC, uid ASC",
            "limit": limit,
            "offset": offset,
        }

        try:
            response = requests.get(OPENAGENDA_API_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.error(f"Erreur lors de l'appel API Open Agenda (offset={offset}): {exc}")
            raise

        results = data.get("results", [])
        if not results:
            break

        all_records.extend(results)
        offset += len(results)
        logger.info(f"Événements récupérés : {len(all_records)} / {max_events}")

        if len(results) < limit:
            break

    elapsed = time.perf_counter() - t0
    logger.info(f"Extraction terminée : {len(all_records)} événements en {elapsed:.2f} s.")
    return all_records


def save_raw_events(
    records: List[Dict[str, Any]], output_path: Path = RAW_EVENTS_FILE
) -> Path:
    """Sauvegarde les événements bruts au format JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
    logger.info(f"{len(records)} événements bruts sauvegardés dans {output_path}")
    return output_path


if __name__ == "__main__":
    events = fetch_openagenda_events()
    save_raw_events(events)