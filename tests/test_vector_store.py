"""
Tests unitaires pour le module de vectorisation et l'index FAISS.
Vérifie le découpage en chunks, la dimension des vecteurs et la recherche sémantique.
"""

import pytest
from src.vector_store import FAISS_INDEX_DIR, create_chunks, load_faiss_index


@pytest.fixture(scope="module")
def faiss_store():
    """Charge l'index FAISS une seule fois pour tous les tests du module."""
    return load_faiss_index(FAISS_INDEX_DIR)


def test_chunking_with_context_header():
    """Vérifie qu'un texte long est découpé et que le 2e chunk reçoit l'en-tête de rappel."""
    long_text = "Titre : Festival Test\nVille : Paris\nDescription : " + ("Concert de musique. " * 80)
    fake_events = [
        {
            "uid": "999999",
            "title": "Festival Test",
            "city": "Paris",
            "location_name": "Salle Pleyel",
            "address": "252 Rue du Faubourg Saint-Honoré",
            "first_date": "2026-11-10",
            "last_date": "2026-11-12",
            "url": "https://openagenda.com/test",
            "combined_text": long_text,
        }
    ]

    chunks = create_chunks(fake_events, chunk_size=800, chunk_overlap=120)

    assert len(chunks) >= 2, "Le texte long aurait dû être découpé en au moins 2 chunks."
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[1].metadata["chunk_index"] == 1
    assert "Événement (suite) : Festival Test" in chunks[1].page_content
    assert chunks[1].metadata["city"] == "Paris"


def test_faiss_files_exist():
    """Vérifie que les fichiers binaires de l'index FAISS existent sur le disque."""
    assert (FAISS_INDEX_DIR / "index.faiss").exists(), "Le fichier index.faiss est manquant."
    assert (FAISS_INDEX_DIR / "index.pkl").exists(), "Le fichier index.pkl est manquant."


def test_faiss_index_dimensions_and_count(faiss_store):
    """Vérifie le nombre de vecteurs indexés (>= 1000) et la dimension Mistral (1024)."""
    assert faiss_store.index.ntotal >= 1000, (
        f"Nombre de vecteurs insuffisant dans FAISS : {faiss_store.index.ntotal}"
    )
    assert faiss_store.index.d == 1024, (
        f"Dimension vectorielle incorrecte : {faiss_store.index.d} (attendu : 1024)"
    )


def test_semantic_similarity_search(faiss_store):
    """Teste une recherche sémantique réelle et vérifie la structure des résultats."""
    query = "Concert de musique jazz ou exposition culturelle à Paris"
    results = faiss_store.similarity_search_with_score(query, k=3)

    assert len(results) == 3, "La recherche doit retourner exactement k=3 résultats."

    for doc, score in results:
        assert len(doc.page_content) > 20, "Le contenu du chunk retourné est vide ou trop court."
        assert "title" in doc.metadata and doc.metadata["title"], "Métadonnée 'title' manquante."
        assert doc.metadata.get("city", "").lower() == "paris", "La ville doit être Paris."
        assert isinstance(float(score), float), "Le score de distance L2 doit être numérique."
