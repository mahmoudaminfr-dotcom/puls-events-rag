"""
Moteur RAG (Retrieval-Augmented Generation) pour la recommandation d'événements Puls-Events.
Combine la recherche vectorielle FAISS et la génération via Mistral AI.
"""

import logging
import time
from typing import Dict, Any, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import FAISS

import src.config as config
from src.vector_store import load_faiss_index, FAISS_INDEX_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MISTRAL_API_KEY = getattr(config, "MISTRAL_API_KEY", None)
MISTRAL_CHAT_MODEL = getattr(config, "MISTRAL_CHAT_MODEL", "open-mistral-nemo")
REFERENCE_DATE = getattr(config, "REFERENCE_DATE", "2026-11-05")
DEFAULT_TOP_K = getattr(config, "TOP_K", 5)

SYSTEM_PROMPT = """Tu es l'assistant virtuel officiel de Puls-Events, spécialisé dans la recommandation d'événements culturels à Paris.
La date de référence actuelle est le {reference_date}.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT à partir des événements fournis dans le bloc [CONTEXTE] ci-dessous.
2. N'invente JAMAIS d'événement, de date, d'adresse, de tarif ou de lien qui ne figure pas explicitement dans le [CONTEXTE].
3. Si aucun événement du [CONTEXTE] ne correspond à la demande de l'utilisateur (ou si la demande concerne une autre ville que Paris ou un sujet hors périmètre), indique poliment que tu ne disposes d'aucun événement correspondant dans la base Puls-Events.
4. Pour chaque événement recommandé, mentionne clairement :
   - Le titre exact de l'événement
   - Les dates / horaires
   - Le nom du lieu et l'adresse
   - Un bref résumé pertinent par rapport à la question
   - Le lien URL s'il est disponible

[CONTEXTE]
{context}
[/CONTEXTE]"""


def format_retrieved_documents(docs: List[Document]) -> str:
    """Formate les documents extraits de FAISS en un bloc de texte structuré pour le LLM."""
    if not docs:
        return "Aucun événement pertinent trouvé dans la base."

    formatted_blocks = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata
        block = (
            f"--- DOCUMENT {idx} ---\n"
            f"Titre : {meta.get('title', 'Non spécifié')}\n"
            f"Ville : {meta.get('city', 'Paris')}\n"
            f"Lieu : {meta.get('location_name', 'Non spécifié')} ({meta.get('address', '')})\n"
            f"Dates : {meta.get('daterange') or (meta.get('first_date', '') + ' au ' + meta.get('last_date', ''))}\n"
            f"URL : {meta.get('url', 'Non disponible')}\n"
            f"Contenu : {doc.page_content}\n"
        )
        formatted_blocks.append(block)

    return "\n".join(formatted_blocks)


class PulsEventsRAG:
    """Classe encapsulant le pipeline RAG complet (FAISS + Mistral LLM)."""

    def __init__(
        self,
        vector_store: Optional[FAISS] = None,
        top_k: int = DEFAULT_TOP_K,
        temperature: float = 0.1,
    ):
        if not MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY est manquante dans la configuration.")

        self.vector_store = vector_store or load_faiss_index(FAISS_INDEX_DIR)
        self.top_k = top_k

        self.llm = ChatMistralAI(
            model=MISTRAL_CHAT_MODEL,
            api_key=MISTRAL_API_KEY,
            temperature=temperature,
            max_retries=5,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )

        self.chain = self.prompt | self.llm | StrOutputParser()

    def retrieve(self, question: str, k: Optional[int] = None) -> List[Document]:
        """Récupère les k chunks les plus pertinents avec leur score de distance L2."""
        search_k = k or self.top_k
        results_with_scores = self.vector_store.similarity_search_with_score(
            question, k=search_k
        )

        docs = []
        for doc, score in results_with_scores:
            doc.metadata["similarity_score"] = float(score)
            docs.append(doc)

        return docs

    def _invoke_with_backoff(self, payload: Dict[str, Any], max_attempts: int = 4) -> str:
        """Exécute l'appel LLM avec gestion explicite du Rate Limit HTTP 429 de Mistral."""
        for attempt in range(1, max_attempts + 1):
            try:
                return self.chain.invoke(payload)
            except Exception as exc:
                if "429" in str(exc) and attempt < max_attempts:
                    wait_time = 2.5 * attempt
                    logger.warning(
                        f"Rate limit Mistral (429) détecté. Nouvelle tentative {attempt}/{max_attempts} dans {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise

    def ask(self, question: str, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Exécute le pipeline RAG complet pour une question utilisateur.
        Retourne la réponse générée ainsi que les documents sources mobilisés.
        """
        if not question or not question.strip():
            raise ValueError("La question ne peut pas être vide.")

        logger.info(f"Requête RAG : '{question}'")
        retrieved_docs = self.retrieve(question, k=k)
        context_str = format_retrieved_documents(retrieved_docs)

        # Pause de 1.2s pour respecter le quota de 1 requête/seconde entre Embeddings et Chat
        time.sleep(1.2)

        answer = self._invoke_with_backoff(
            {
                "reference_date": REFERENCE_DATE,
                "context": context_str,
                "question": question,
            }
        )

        sources = []
        seen_uids = set()
        for doc in retrieved_docs:
            uid = doc.metadata.get("uid", "")
            if uid not in seen_uids:
                seen_uids.add(uid)
                sources.append(
                    {
                        "uid": uid,
                        "title": doc.metadata.get("title", ""),
                        "city": doc.metadata.get("city", ""),
                        "location_name": doc.metadata.get("location_name", ""),
                        "daterange": doc.metadata.get("daterange", ""),
                        "first_date": doc.metadata.get("first_date", ""),
                        "url": doc.metadata.get("url", ""),
                        "score": doc.metadata.get("similarity_score", 0.0),
                    }
                )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "raw_chunks": retrieved_docs,
        }


if __name__ == "__main__":
    rag = PulsEventsRAG(top_k=4)
    sample_question = "Quels concerts de jazz ou de musique brésilienne me conseilles-tu à Paris ?"
    result = rag.ask(sample_question)

    print("\n" + "=" * 80)
    print(f"QUESTION : {result['question']}")
    print("=" * 80)
    print(f"\nRÉPONSE DU CHATBOT :\n{result['answer']}\n")
    print("-" * 80)
    print("SOURCES UTILISÉES :")
    for src in result["sources"]:
        print(f" - [{src['title']}] ({src['location_name']}) | Score L2: {src['score']:.4f} | URL: {src['url']}")
    print("=" * 80)

