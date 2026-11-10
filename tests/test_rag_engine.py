"""
Tests pour le moteur RAG (src/rag_engine.py).
Sépare le test unitaire hors-ligne (formatage du contexte) des tests d'intégration API Mistral.
"""

import time
import pytest
from langchain_core.documents import Document

import src.config as config
from src.evaluation import check_refusal
from src.rag_engine import PulsEventsRAG, format_retrieved_documents
from src.vector_store import FAISS_INDEX_DIR

HAS_FAISS_INDEX = (FAISS_INDEX_DIR / "index.faiss").exists()
HAS_MISTRAL_KEY = bool(config.MISTRAL_API_KEY) and config.MISTRAL_API_KEY != "your_mistral_api_key_here"


@pytest.fixture(scope="module")
def rag_engine():
    """Initialise le moteur RAG une seule fois pour les tests d'intégration."""
    if not HAS_FAISS_INDEX or not HAS_MISTRAL_KEY:
        pytest.skip("Index FAISS et MISTRAL_API_KEY requis pour les tests d'intégration RAG.")
    return PulsEventsRAG(top_k=3)


def test_format_retrieved_documents():
    """Vérifie hors-ligne que les documents extraits sont correctement formatés pour le prompt."""
    sample_docs = [
        Document(
            page_content="Concert exceptionnel de jazz brésilien.",
            metadata={
                "uid": "123",
                "title": "Soirée Jazz Test",
                "city": "Paris",
                "location_name": "Le Sunset",
                "address": "60 Rue des Lombards",
                "daterange": "Samedi 28 novembre à 20h",
                "url": "https://openagenda.com/test-jazz",
            },
        )
    ]

    formatted = format_retrieved_documents(sample_docs)

    assert "--- DOCUMENT 1 ---" in formatted
    assert "Soirée Jazz Test" in formatted
    assert "Le Sunset" in formatted
    assert "https://openagenda.com/test-jazz" in formatted
    assert format_retrieved_documents([]) == "Aucun événement pertinent trouvé dans la base."


@pytest.mark.integration
def test_rag_ask_in_scope_query(rag_engine):
    """Vérifie qu'une question dans le périmètre retourne une réponse structurée et ses sources."""
    time.sleep(1.2)
    question = "Quels concerts de jazz ou de musique brésilienne me conseilles-tu à Paris ?"
    result = rag_engine.ask(question, k=3)

    assert "answer" in result and len(result["answer"]) > 50
    assert "sources" in result and len(result["sources"]) >= 1
    assert "raw_chunks" in result and len(result["raw_chunks"]) == 3

    uids = [src["uid"] for src in result["sources"]]
    assert len(uids) == len(set(uids)), "Les sources retournées contiennent des doublons d'UID."


@pytest.mark.integration
def test_rag_ask_out_of_scope_guardrail(rag_engine):
    """Vérifie que le moteur RAG refuse proprement une demande hors périmètre."""
    time.sleep(1.5)
    question = "Quels sont les horaires de la compétition de surf à Honolulu à Hawaï demain ?"
    result = rag_engine.ask(question, k=3)

    assert check_refusal(result["answer"]) is True, (
        f"Le garde-fou anti-hallucination a échoué. Réponse obtenue : {result['answer']}"
    )