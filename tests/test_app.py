"""Tests unitaires de non-régression pour l'interface Streamlit app.py."""

import src.config as config
import app


def test_app_imports_and_config_alignment():
    """Vérifie que app.py s'importe proprement et reflète la configuration centrale."""
    assert app.EMBEDDING_MODEL == config.EMBEDDING_MODEL
    assert app.LLM_MODEL == config.LLM_MODEL
    assert app.RETRIEVER_K == config.RETRIEVER_K
    assert app.TARGET_CITY == config.TARGET_CITY


def test_app_demo_questions_non_empty():
    """Vérifie que les questions de démonstration sont définies et non vides."""
    assert isinstance(app.DEMO_QUESTIONS, list)
    assert len(app.DEMO_QUESTIONS) >= 3
    assert all(isinstance(q, str) and q.strip() for q in app.DEMO_QUESTIONS)


def test_app_entrypoints_callable():
    """Vérifie la présence des fonctions principales de l'application."""
    assert callable(app.main)
    assert callable(app.get_vector_store)
    assert callable(app.get_rag_engine)
    assert callable(app.render_sources)