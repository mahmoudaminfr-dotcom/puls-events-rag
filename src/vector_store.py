"""
Module de découpage (chunking), vectorisation et gestion de l'index FAISS.
Transforme les événements nettoyés en vecteurs sémantiques via Mistral AI.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import src.config as config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MISTRAL_EMBED_MODEL = getattr(config, "EMBEDDING_MODEL", "mistral-embed")
PROCESSED_DATA_DIR = getattr(config, "PROCESSED_DATA_DIR", Path("data/processed"))
FAISS_INDEX_DIR = getattr(config, "FAISS_INDEX_DIR", Path("data/faiss_index"))
CHUNK_SIZE = getattr(config, "CHUNK_SIZE", 800)
CHUNK_OVERLAP = getattr(config, "CHUNK_OVERLAP", 120)

CLEANED_JSON_PATH = PROCESSED_DATA_DIR / "events_cleaned.json"
BATCH_SIZE = 100
BATCH_SLEEP_SECONDS = 1.5


def load_cleaned_events(filepath: Path = CLEANED_JSON_PATH) -> List[Dict[str, Any]]:
    """Charge les événements nettoyés depuis le fichier JSON."""
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    with open(filepath, "r", encoding="utf-8-sig") as f:
        events = json.load(f)

    logger.info(f"{len(events)} événements chargés depuis {filepath.name}.")
    return events


def create_chunks(
    events: List[Dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Découpe le texte combiné de chaque événement en objets Document LangChain
    tout en conservant les métadonnées critiques dans chaque chunk.
    """
    t0 = time.perf_counter()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    documents: List[Document] = []

    for event in events:
        combined_text = (event.get("combined_text") or "").strip()
        if not combined_text:
            continue

        base_metadata = {
            "uid": str(event.get("uid") or ""),
            "title": str(event.get("title_fr") or event.get("title") or ""),
            "city": str(event.get("location_city") or event.get("city") or ""),
            "location_name": str(event.get("location_name") or ""),
            "address": str(event.get("location_address") or event.get("address") or ""),
            "postal_code": str(event.get("location_postalcode") or ""),
            "daterange": str(event.get("daterange_fr") or ""),
            "first_date": str(event.get("firstdate_begin") or event.get("first_date") or ""),
            "last_date": str(event.get("lastdate_end") or event.get("last_date") or ""),
            "url": str(event.get("canonicalurl") or event.get("url") or ""),
            "category": str(event.get("category") or ""),
        }

        splits = text_splitter.split_text(combined_text)

        for idx, split_text in enumerate(splits):
            if idx > 0:
                header = (
                    f"Événement (suite) : {base_metadata['title']} | "
                    f"Ville : {base_metadata['city']} | "
                    f"Dates : {base_metadata['first_date']} au {base_metadata['last_date']}\n"
                )
                page_content = header + split_text
            else:
                page_content = split_text

            chunk_metadata = {
                **base_metadata,
                "chunk_index": idx,
                "total_chunks": len(splits),
            }

            documents.append(
                Document(page_content=page_content, metadata=chunk_metadata)
            )

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Découpage terminé en {elapsed:.3f} s : {len(events)} événements -> {len(documents)} chunks "
        f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})."
    )
    return documents


def build_and_save_faiss_index(
    documents: List[Document],
    index_dir: Path = FAISS_INDEX_DIR,
    batch_size: int = BATCH_SIZE,
) -> FAISS:
    """
    Vectorise les documents par lots via MistralAIEmbeddings et sauvegarde l'index FAISS.
    """
    api_key = getattr(config, "MISTRAL_API_KEY", None)
    if not api_key or api_key == "your_mistral_api_key_here":
        raise ValueError("MISTRAL_API_KEY est manquante dans la configuration.")

    if not documents:
        raise ValueError("La liste de documents à indexer est vide.")

    t0 = time.perf_counter()
    embeddings = MistralAIEmbeddings(
        model=MISTRAL_EMBED_MODEL,
        api_key=api_key,
    )

    total_docs = len(documents)
    vector_store = None

    logger.info(
        f"Démarrage de la vectorisation ({total_docs} chunks) par lots de {batch_size}..."
    )

    for start_idx in range(0, total_docs, batch_size):
        end_idx = min(start_idx + batch_size, total_docs)
        batch = documents[start_idx:end_idx]

        logger.info(f"Vectorisation du lot {start_idx + 1} à {end_idx} / {total_docs}...")

        for attempt in range(1, 5):
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    vector_store.add_documents(batch)
                break
            except Exception as exc:
                if "429" in str(exc) and attempt < 4:
                    wait_s = 15.0 * attempt
                    logger.warning(f"Quota Mistral Embeddings (429). Pause de {wait_s:.0f}s...")
                    time.sleep(wait_s)
                else:
                    raise

        if end_idx < total_docs:
            time.sleep(BATCH_SLEEP_SECONDS)

    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_dir))
    elapsed = time.perf_counter() - t0
    logger.info(f"Index FAISS sauvegardé dans {index_dir} en {elapsed:.2f} s.")

    return vector_store


def load_faiss_index(index_dir: Path = FAISS_INDEX_DIR) -> FAISS:
    """Charge un index FAISS existant depuis le disque."""
    index_file = index_dir / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"Aucun index FAISS trouvé dans {index_dir}. Exécutez d'abord l'indexation."
        )

    api_key = getattr(config, "MISTRAL_API_KEY", None)
    embeddings = MistralAIEmbeddings(
        model=MISTRAL_EMBED_MODEL,
        api_key=api_key or "offline_placeholder",
    )

    vector_store = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info(
        f"Index FAISS chargé depuis {index_dir} ({vector_store.index.ntotal} vecteurs)."
    )
    return vector_store


if __name__ == "__main__":
    events_data = load_cleaned_events()
    chunked_docs = create_chunks(events_data)
    build_and_save_faiss_index(chunked_docs)