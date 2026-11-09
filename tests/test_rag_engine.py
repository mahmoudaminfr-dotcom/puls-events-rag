"""
Tests unitaires pour le moteur RAG (src/rag_engine.py).
Vérifie le formatage du contexte, la génération augmentée et le garde-fou anti-hallucination.
"""

import time
import pytest
from langchain_core.documents import Document
from src.rag_engine import PulsEventsRAG, format_retrieved_documents


@pytest.fixture(scope="module")
def rag_engine():
    """Initialise le moteur RAG une seule fois pour le module de test."""
    return PulsEventsRAG(top_k=3)


def test_format_retrieved_documents():
    """Vérifie que les documents extraits sont correctement formatés pour le prompt."""
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


def test_rag_ask_in_scope_query(rag_engine):
    """Vérifie qu'une question dans le périmètre retourne une réponse structurée et ses sources."""
    time.sleep(1.2)
    question = "Quels concerts de jazz ou de musique brésilienne me conseilles-tu à Paris ?"
    result = rag_engine.ask(question, k=3)

    assert "answer" in result and len(result["answer"]) > 50
    assert "sources" in result and len(result["sources"]) >= 1
    assert "raw_chunks" in result and len(result["raw_chunks"]) == 3

    # Vérifie que les sources sont bien dédoublonnées par uid
    uids = [src["uid"] for src in result["sources"]]
    assert len(uids) == len(set(uids)), "Les sources retournées contiennent des doublons d'UID."


def test_rag_ask_out_of_scope_guardrail(rag_engine):
    """Vérifie que le moteur RAG n'hallucine pas sur une demande hors périmètre."""
    time.sleep(1.5)
    question = "Quels sont les horaires de la compétition de surf à Honolulu à Hawaï demain ?"
    result = rag_engine.ask(question, k=3)

    answer_lower = result["answer"].lower()
    refusal_indicators = [
        "aucun",
        "ne dispose",
        "pas d'événement",
        "ne figure",
        "ne correspond",
        "uniquement",
        "paris",
        "désolé",
        "mentionne pas",
    ]
    assert any(word in answer_lower for word in refusal_indicators), (
        f"Le garde-fou anti-hallucination a échoué. Réponse obtenue : {result['answer']}"
    )
