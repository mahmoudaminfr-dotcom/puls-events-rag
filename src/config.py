import os
from pathlib import Path
from dotenv import load_dotenv

# Désactivation de l'avertissement de cache Hugging Face sous Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Chargement des variables d'environnement depuis le fichier .env
load_dotenv()

# Chemins principaux du projet
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
EVALUATION_DATA_DIR = DATA_DIR / "evaluation"

# Fichiers de données par défaut
RAW_EVENTS_FILE = RAW_DATA_DIR / "openagenda_events_raw.json"
PROCESSED_EVENTS_FILE = PROCESSED_DATA_DIR / "events_cleaned.parquet"
PROCESSED_EVENTS_CSV = PROCESSED_DATA_DIR / "events_cleaned.csv"

# Configuration API Open Agenda (Opendatasoft v2.1)
OPENAGENDA_API_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "evenements-publics-openagenda/records"
)
TARGET_CITY = os.getenv("TARGET_CITY", "Paris")
HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "365"))
MAX_EVENTS = int(os.getenv("MAX_EVENTS", "1000"))

# Configuration Mistral AI & RAG
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
EMBEDDING_MODEL = "mistral-embed"
LLM_MODEL = "mistral-small-latest"

# Paramètres de découpage (Chunking) et de recherche (Retrieval)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
RETRIEVER_K = 5


def validate_config() -> None:
    """Vérifie que les variables d'environnement critiques sont configurées."""
    if not MISTRAL_API_KEY or MISTRAL_API_KEY == "your_mistral_api_key_here":
        raise ValueError(
            "La variable MISTRAL_API_KEY est manquante ou invalide dans le fichier .env."
        )
