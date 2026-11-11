"""Interface web Streamlit pour la démonstration du POC RAG Puls-Events."""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    MISTRAL_API_KEY,
    REFERENCE_DATE,
    RETRIEVER_K,
    TARGET_CITY,
)
from src.rag_engine import PulsEventsRAG
from src.vector_store import FAISS_INDEX_DIR, load_faiss_index

DEMO_QUESTIONS: List[str] = [
    "Quelles expositions de photographie sont programmées à Paris ?",
    "Quelles activités ou ateliers culturels faire en famille avec des enfants à Paris ?",
    "Quels concerts ou événements musicaux gratuits découvrir à Paris ?",
    "Quels événements culturels sont prévus à Lyon ce week-end ?",
    "Peux-tu me donner la recette traditionnelle de la pâte à crêpes ?",
]


@st.cache_resource(show_spinner="Chargement de l'index vectoriel FAISS...")
def get_vector_store():
    """Charge l'index FAISS local une seule fois en mémoire."""
    return load_faiss_index()


@st.cache_resource(show_spinner=False)
def get_rag_engine(temperature: float) -> PulsEventsRAG:
    """Initialise le moteur RAG en réutilisant l'index FAISS mis en cache."""
    vector_store = get_vector_store()
    return PulsEventsRAG(
        vector_store=vector_store,
        top_k=RETRIEVER_K,
        temperature=temperature,
    )


def render_sources(sources: List[Dict[str, Any]], raw_chunks: List[Any]) -> None:
    """Affiche les événements sources dédoublonnés et les passages FAISS associés."""
    if sources:
        with st.expander(f"Sources Open Agenda consultées ({len(sources)} événement(s))", expanded=False):
            for idx, src in enumerate(sources, start=1):
                title = src.get("title") or "Événement sans titre"
                location = src.get("location_name") or "Lieu non précisé"
                city = src.get("city") or TARGET_CITY
                daterange = src.get("daterange") or src.get("first_date") or "Dates non précisées"
                score = src.get("score")
                url = src.get("url") or ""

                score_label = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
                header = f"**{idx}. {title}** — `{location}` ({city})"
                details = f"Horaires (heure de Paris) : {daterange} | Distance L2 FAISS : `{score_label}`"
                if url:
                    details += f" | [Fiche Open Agenda]({url})"

                st.markdown(f"{header}  \n{details}")

    if raw_chunks:
        with st.expander(f"Détail technique : passages bruts extraits ({len(raw_chunks)} chunk(s))", expanded=False):
            for idx, chunk in enumerate(raw_chunks, start=1):
                meta = getattr(chunk, "metadata", {}) or {}
                content = getattr(chunk, "page_content", "")
                uid = meta.get("uid", "N/A")
                chunk_title = meta.get("title") or "Sans titre"
                chunk_city = meta.get("city") or TARGET_CITY
                chunk_location = meta.get("location_name") or "Lieu non précisé"
                chunk_address = meta.get("address") or "Adresse non précisée"
                chunk_dates = meta.get("daterange") or meta.get("first_date") or "Dates non précisées"

                st.markdown(f"**Chunk #{idx} — UID `{uid}` — {chunk_title}**")
                st.caption(
                    f"Métadonnées FAISS — Ville : {chunk_city} | "
                    f"Lieu : {chunk_location} | "
                    f"Adresse : {chunk_address} | "
                    f"Horaires (heure de Paris) : {chunk_dates}"
                )
                st.code(content, language="text", wrap_lines=True)
                if idx < len(raw_chunks):
                    st.divider()


def main() -> None:
    """Point d'entrée principal de l'application Streamlit."""
    st.set_page_config(
        page_title=f"Puls-Events — Assistant Culturel {TARGET_CITY}",
        page_icon="🎭",
        layout="wide",
    )

    st.title("Puls-Events — Assistant de recommandation culturelle")
    st.caption(
        f"POC RAG basé sur les événements publics Open Agenda (périmètre : {TARGET_CITY}) "
        "orchestré avec LangChain, FAISS et Mistral AI."
    )

    if not MISTRAL_API_KEY or MISTRAL_API_KEY == "your_mistral_api_key_here":
        st.error(
            "Clé API Mistral manquante ou invalide. Renseignez la variable `MISTRAL_API_KEY` "
            "dans le fichier `.env` à la racine du projet."
        )
        st.stop()

    index_file = Path(FAISS_INDEX_DIR) / "index.faiss"
    if not index_file.exists():
        st.error(
            f"Index FAISS introuvable dans `{FAISS_INDEX_DIR}`. "
            "Exécutez d'abord `python -m src.vector_store` pour générer la base vectorielle."
        )
        st.stop()

    try:
        vector_store = get_vector_store()
        total_vectors = vector_store.index.ntotal
    except Exception as exc:
        st.error(f"Erreur lors du chargement de l'index FAISS : {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    with st.sidebar:
        st.header("Paramètres du RAG")
        top_k = st.slider(
            "Passages extraits (Top-K)",
            min_value=1,
            max_value=10,
            value=int(RETRIEVER_K),
            step=1,
            help="Nombre de chunks récupérés dans FAISS avant synthèse par le LLM.",
        )
        temperature = st.slider(
            "Température du LLM",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
            help="Une valeur basse favorise des réponses factuelles ancrées dans le contexte.",
        )

        st.divider()
        st.subheader("Informations système")
        st.markdown(
            f"- **Périmètre :** {TARGET_CITY} (réf. `{REFERENCE_DATE}`)\n"
            f"- **Vecteurs indexés :** `{total_vectors}` chunks\n"
            f"- **Embeddings :** `{EMBEDDING_MODEL}`\n"
            f"- **Modèle LLM :** `{LLM_MODEL}`"
        )

        st.divider()
        st.subheader("Questions de démonstration")
        for idx, sample_q in enumerate(DEMO_QUESTIONS):
            if st.button(sample_q, key=f"demo_q_{idx}", use_container_width=True):
                st.session_state.pending_question = sample_q

        st.divider()
        if st.button("Réinitialiser l'affichage", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()

    if not st.session_state.messages:
        st.info(
            f"Posez une question sur les sorties culturelles à {TARGET_CITY} dans la barre ci-dessous "
            "ou sélectionnez un exemple dans le panneau latéral. Chaque requête est évaluée "
            "de manière indépendante sur la base vectorielle."
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("elapsed") is not None:
                    st.caption(f"Temps de réponse : {msg['elapsed']:.2f} s | Top-K = {msg.get('top_k', top_k)}")
                render_sources(msg.get("sources", []), msg.get("raw_chunks", []))

    user_input = st.chat_input(f"Votre question sur les événements culturels à {TARGET_CITY}...")
    active_question = user_input or st.session_state.pending_question

    if active_question:
        st.session_state.pending_question = None
        st.session_state.messages.append({"role": "user", "content": active_question})

        with st.chat_message("user"):
            st.markdown(active_question)

        with st.chat_message("assistant"):
            with st.spinner("Recherche dans FAISS et génération de la réponse..."):
                start_ts = time.perf_counter()
                try:
                    rag_engine = get_rag_engine(temperature=temperature)
                    result = rag_engine.ask(active_question, k=top_k)
                    elapsed = time.perf_counter() - start_ts

                    answer = result.get("answer", "")
                    sources = result.get("sources", [])
                    raw_chunks = result.get("raw_chunks", [])

                    st.markdown(answer)
                    st.caption(f"Temps de réponse : {elapsed:.2f} s | Top-K = {top_k}")
                    render_sources(sources, raw_chunks)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "raw_chunks": raw_chunks,
                            "elapsed": elapsed,
                            "top_k": top_k,
                        }
                    )
                except Exception as exc:
                    st.error(f"Erreur lors du traitement de la requête : {exc}")


if __name__ == "__main__":
    main()