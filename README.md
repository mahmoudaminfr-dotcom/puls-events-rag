# Puls-Events — POC Assistant RAG d'Événements Culturels

Proof of Concept (POC) d'un système de génération augmentée de récupération (**RAG**) conçu pour **Puls-Events** afin de recommander des événements culturels à partir des données publiques **Open Agenda**, orchestré avec **LangChain**, **FAISS** et **Mistral AI**.

---

## 1. Contexte et Objectifs

Puls-Events développe une plateforme de découverte d'événements culturels et souhaite évaluer l'intégration d'un assistant conversationnel capable de répondre en langage naturel aux recherches des utilisateurs en s'appuyant exclusivement sur une base d'événements vérifiés.

Ce POC répond aux objectifs suivants :
- **Collecter et qualifier** les événements publics issus de l'API Open Agenda (Opendatasoft v2.1) sur un périmètre géographique ciblé (**Ville de Paris**) et une fenêtre temporelle glissante stricte (**moins d'un an d'historique** et programmation à venir sur 6 mois).
- **Découper et vectoriser** les fiches descriptives avec conservation des métadonnées structurées dans un index local **FAISS** reconstructible à la demande.
- **Générer des réponses factuelles et sourcées** via **LangChain** et **Mistral AI**, dotées de garde-fous explicites contre les requêtes hors périmètre géographique ou thématique.
- **Mesurer la fiabilité** du pipeline de bout en bout à l'aide d'une suite de tests unitaires automatisés (`pytest`) et d'un jeu d'évaluation annoté.

---

## 2. Architecture et Choix Techniques

| Composant | Technologie / Paramètre | Rôle dans le pipeline |
| :--- | :--- | :--- |
| **Source de données** | API Open Agenda (Opendatasoft v2.1) | Extraction paginée des événements publics (`Paris`, historique < 365 j, futur <= 180 j, plafond `1 000` événements). |
| **Pré-processing** | Python, Pandas (`2.2.3`) | Suppression des balises HTML, normalisation Unicode, dédoublonnage par `uid`, filtrage temporel/géographique et assemblage du champ `combined_text`. |
| **Découpage (Chunking)** | `RecursiveCharacterTextSplitter` | `chunk_size = 800`, `chunk_overlap = 120`, injection d'un en-tête de contexte sur les segments secondaires et rattachement systématique des métadonnées (`uid`, `title`, `city`, `location_name`, `address`, `daterange`, `url`). |
| **Embeddings** | `mistral-embed` (dim. `1024`) | Vectorisation sémantique par lots avec gestion des quotas API. |
| **Base vectorielle** | `faiss-cpu` (`IndexFlatL2`) | Recherche exacte par distance euclidienne L2 sur **2 699 vecteurs** persistés localement. |
| **Orchestration & LLM** | LangChain + `open-mistral-nemo` | Récupération `Top-K = 5`, dédoublonnage des sources par événement, génération à basse température (`0.1`) avec mécanisme de *retry/backoff* sur les limites de requêtes API. |
| **Interface de démonstration** | Streamlit (`app.py`) | Interface web interactive affichant la réponse synthétisée, les fiches Open Agenda consultées et l'inspection des chunks extraits. |

---

## 3. Organisation du Dépôt

```text
puls-events-rag/
├── app.py                              # Interface web Streamlit de démonstration live
├── requirements.txt                    # Dépendances Python du projet
├── .env.example                        # Modèle de configuration des variables d'environnement
├── data/
│   ├── raw/                            # Extraction brute JSON Open Agenda (ignorée par Git)
│   ├── processed/
│   │   └── events_cleaned.json         # Jeu d'événements nettoyés et filtrés prêt à indexer
│   ├── faiss_index/                    # Index vectoriel FAISS persisté sur disque (généré localement)
│   └── evaluation/
│       ├── test_dataset.json           # Jeu de 10 questions/réponses annotées (vérité terrain)
│       ├── evaluation_results.json     # Résultats détaillés du benchmark RAG
│       └── evaluation_summary.csv      # Synthèse tabulaire des métriques par question
├── src/
│   ├── config.py                       # Chargement de .env, chemins et constantes globales
│   ├── ingestion.py                    # Collecte paginée depuis l'API Open Agenda v2.1
│   ├── preprocessing.py                # Nettoyage HTML, filtrage géographique/temporel et structuration
│   ├── vector_store.py                 # Découpage en chunks, embeddings Mistral et gestion FAISS
│   ├── rag_engine.py                   # Chaîne RAG LangChain/Mistral et formatage des sources
│   └── evaluation.py                   # Évaluation quantitative et qualitative sur le jeu annoté
└── tests/
    ├── test_data_quality.py            # Contrôles du nettoyage, périmètre Paris et ancienneté < 1 an
    ├── test_vector_store.py            # Contrôles du chunking, dimensions et recherche FAISS
    ├── test_rag_engine.py              # Contrôles de bout en bout du moteur RAG et des garde-fous
    ├── test_evaluation.py              # Contrôles des métriques et seuils de qualité du benchmark
    └── test_app.py                     # Contrôles de non-régression de l'interface Streamlit
```

---

## 4. Installation et Configuration

### Prérequis
- **Python 3.11 ou 3.12**
- Une clé API active sur [Mistral AI La Plateforme](https://console.mistral.ai/)

### 1. Cloner le dépôt
```bash
git clone [https://github.com/mahmoudaminfr-dotcom/puls-events-rag.git](https://github.com/mahmoudaminfr-dotcom/puls-events-rag.git)
cd puls-events-rag
```

### 2. Créer l'environnement virtuel et installer les dépendances

**Avec `pip` standard (Windows PowerShell) :**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Avec `pip` standard (Linux / macOS) :**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Alternative rapide avec `uv` :**
```powershell
uv venv .venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

### 3. Configurer les variables d'environnement
Copier le fichier `.env.example` vers `.env` à la racine du projet et renseigner la clé `MISTRAL_API_KEY` :

```powershell
Copy-Item .env.example .env
```

Contenu attendu dans `.env` :
```ini
MISTRAL_API_KEY=votre_cle_api_mistral
TARGET_CITY=Paris
REFERENCE_DATE=2026-11-05
HISTORY_DAYS=365
FUTURE_DAYS=180
MAX_EVENTS=1000
```

---

## 5. Reproduction du Pipeline et Utilisation

L'index binaire `data/faiss_index/` n'est pas versionné dans Git. Il peut être reconstruit directement à partir du jeu nettoyé fourni (`data/processed/events_cleaned.json`), ou régénéré intégralement depuis l'API Open Agenda.

### Option A — Reconstruction rapide de l'index FAISS (recommandée après clonage)
```bash
python -m src.vector_store
```

### Option B — Exécution complète du pipeline depuis l'API Open Agenda
```bash
# 1. Extraction brute depuis l'API Open Agenda -> data/raw/openagenda_events_raw.json
python -m src.ingestion

# 2. Nettoyage, filtrage (< 1 an, Paris) et dédoublonnage -> data/processed/events_cleaned.json
python -m src.preprocessing

# 3. Découpage en chunks, vectorisation Mistral et sauvegarde -> data/faiss_index/
python -m src.vector_store
```

### Tester le moteur RAG en ligne de commande
```bash
python -m src.rag_engine
```

### Lancer l'interface web de démonstration (Streamlit)
```bash
streamlit run app.py
```
L'application s'ouvre sur `http://localhost:8501` et permet d'ajuster dynamiquement `Top-K` et la température du LLM, de tester des requêtes préconfigurées et d'inspecter les sources Open Agenda ainsi que les chunks FAISS associés.

---

## 6. Tests Automatisés et Évaluation du Système

### Exécuter la suite de tests unitaires (19 tests)
*Note : l'index local `data/faiss_index/` doit avoir été généré au préalable.*
```bash
pytest -v
```
La suite vérifie notamment :
- la suppression du HTML, l'unicité des identifiants `uid` et le respect strict du périmètre **Paris** et **moins d'un an** ;
- les dimensions (`1024`) et la cohérence documentaire de l'index FAISS ;
- la pertinence des réponses générées et le refus des requêtes hors périmètre ;
- le respect des seuils de qualité du benchmark et l'intégrité de `app.py`.

### Relancer le benchmark d'évaluation sur le jeu annoté
```bash
python -m src.evaluation
```

### Synthèse des résultats obtenus sur le jeu test (10 questions : 8 dans le périmètre, 2 hors périmètre)

| Indicateur | Résultat POC | Détail |
| :--- | :---: | :--- |
| **Hit Rate @ 5 (Retrieval FAISS)** | **87,5 %** | 7 événements cibles sur 8 retrouvés dans le Top-5 vectoriel. |
| **Similarité sémantique cosinus** | **0,8986** | Proximité moyenne (`mistral-embed`) entre réponses générées et réponses de référence. |
| **Rappel moyen des mots-clés** | **83,3 %** | Couverture factuelle des entités attendues (lieux, dates, intitulés). |
| **Garde-fous hors périmètre** | **100,0 %** | Refus explicite et sans hallucination sur une autre ville (Lyon) ou un sujet hors domaine. |
| **Taux global de réponses valides** | **90,0 %** | 9 scénarios sur 10 intégralement validés de bout en bout. |