# Puls-Events — POC Assistant RAG d'Événements Culturels

Proof of Concept (POC) d'un système de génération augmentée de récupération (**RAG**) conçu pour **Puls-Events** afin de recommander des événements culturels à partir des données publiques **Open Agenda**, orchestré avec **LangChain**, **FAISS** et **Mistral AI**.

---

## 1. Contexte et Objectifs

Puls-Events développe une plateforme de découverte d'événements culturels et souhaite évaluer l'intégration d'un assistant conversationnel capable de répondre en langage naturel aux recherches des utilisateurs en s'appuyant exclusivement sur une base d'événements vérifiés.

Ce POC répond aux objectifs suivants :
- **Collecter et qualifier** les événements publics issus de l'API Open Agenda (Opendatasoft v2.1) sur un périmètre géographique ciblé (**Ville de Paris**), une fenêtre temporelle glissante stricte (**moins d'un an d'historique** et programmation à venir sur 6 mois) et une conversion systématique des horaires en **heure de Paris (`Europe/Paris`)**.
- **Découper et vectoriser** les fiches descriptives avec conservation des métadonnées structurées dans un index local **FAISS** reconstructible à la demande.
- **Générer des réponses factuelles et sourcées** via **LangChain** et **Mistral AI**, dotées de garde-fous explicites contre les requêtes hors périmètre géographique ou thématique.
- **Mesurer la fiabilité** du pipeline à l'aide de tests unitaires automatisés hors-ligne (exécutés dans GitHub Actions sans clé API), de tests d'intégration et d'un jeu d'évaluation annoté vérifiant **5 faits clés** (`titre`, `date`, `heure`, `lieu`, `lien URL`).

---

## 2. Architecture et Choix Techniques

| Composant | Technologie / Paramètre | Rôle dans le pipeline |
| :--- | :--- | :--- |
| **Source de données** | API Open Agenda (Opendatasoft v2.1) | Extraction paginée des événements publics (`Paris`, historique < 365 j, futur <= 180 j, plafond `1 000` événements). |
| **Pré-processing** | Python, Pandas (`2.2.3`) | Suppression des balises HTML, normalisation Unicode, dédoublonnage par `uid`, filtrage temporel/géographique, conversion des horodatages vers `Europe/Paris` (`+01:00` / `+02:00`) et assemblage du champ `combined_text`. |
| **Découpage (Chunking)** | `RecursiveCharacterTextSplitter` | `chunk_size = 800`, `chunk_overlap = 120`, injection d'un en-tête de contexte en heure de Paris sur les segments secondaires et rattachement systématique des métadonnées (`uid`, `title`, `city`, `location_name`, `address`, `daterange`, `url`). |
| **Embeddings** | `mistral-embed` (dim. `1024`) | Vectorisation sémantique par lots avec gestion des quotas API. |
| **Base vectorielle** | `faiss-cpu` (`IndexFlatL2`) | Recherche exacte par distance euclidienne L2 sur **2 703 vecteurs** persistés localement. |
| **Orchestration & LLM** | LangChain + `open-mistral-nemo` | Récupération `Top-K = 5`, restitution des horaires en heure de Paris, dédoublonnage des sources par événement, génération à basse température (`0.1`) avec isolation stricte entre fiches et *retry/backoff* sur les limites API. |
| **Interface de démonstration** | Streamlit (`app.py`) | Interface web interactive affichant la réponse synthétisée, les fiches Open Agenda consultées (horaires de Paris) et l'inspection des chunks extraits. |

---

## 3. Organisation du Dépôt

```text
puls-events-rag/
├── .github/workflows/ci.yml            # Pipeline CI automatisé exécutant les tests unitaires sans clé API
├── app.py                              # Interface web Streamlit de démonstration live
├── pytest.ini                          # Configuration Pytest séparant tests unitaires et intégration
├── requirements.txt                    # Dépendances Python du projet
├── .env.example                        # Modèle de configuration des variables d'environnement
├── data/
│   ├── raw/                            # Extraction brute JSON Open Agenda (ignorée par Git)
│   ├── processed/
│   │   └── events_cleaned.json         # Jeu d'événements nettoyés et convertis en heure de Paris
│   ├── faiss_index/                    # Index vectoriel FAISS persisté sur disque (généré localement)
│   └── evaluation/
│       ├── test_dataset.json           # Jeu de 11 questions/réponses annotées et 5 faits clés attendus
│       ├── evaluation_results.json     # Résultats détaillés du benchmark RAG
│       └── evaluation_summary.csv      # Synthèse tabulaire des métriques par question
├── src/
│   ├── config.py                       # Chargement de .env, chemins et constantes globales
│   ├── ingestion.py                    # Collecte paginée depuis l'API Open Agenda v2.1
│   ├── preprocessing.py                # Nettoyage HTML, filtrage Paris/<1 an et horaires Europe/Paris
│   ├── vector_store.py                 # Découpage en chunks, embeddings Mistral et gestion FAISS
│   ├── rag_engine.py                   # Chaîne RAG LangChain/Mistral et formatage des sources
│   └── evaluation.py                   # Évaluation quantitative (5 faits clés + horaires) et garde-fous
└── tests/
    ├── test_data_quality.py            # Contrôles du nettoyage, périmètre Paris, < 1 an et fuseau Europe/Paris
    ├── test_vector_store.py            # Contrôles du chunking (unitaire) et de l'index FAISS (intégration)
    ├── test_rag_engine.py              # Contrôles du formatage (unitaire) et du moteur RAG (intégration)
    ├── test_evaluation.py              # Contrôles des métriques, 5 faits clés, horaires et refus
    └── test_app.py                     # Contrôles de non-régression de l'interface Streamlit
```

---

## 4. Installation et Configuration

### Prérequis
- **Python 3.11 ou 3.12**
- Une clé API active sur [Mistral AI La Plateforme](https://console.mistral.ai/) *(uniquement requise pour reconstruire l'index FAISS et interroger le chatbot)*

### 1. Cloner le dépôt
```bash
git clone https://github.com/mahmoudaminfr-dotcom/puls-events-rag.git
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

# 2. Nettoyage, filtrage (< 1 an, Paris) et conversion Europe/Paris -> data/processed/events_cleaned.json
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

---

## 6. Tests Automatisés et Évaluation du Système

Les tests sont séparés en deux niveaux via `pytest.ini` afin de permettre une validation immédiate sur un clone propre sans clé API (et dans le pipeline CI `.github/workflows/ci.yml`).

### 1. Tests unitaires hors-ligne (18 tests — sans clé API ni index FAISS)
Exécutés par défaut :
```bash
python -m pytest -v
```

### 2. Suite complète incluant les tests d'intégration FAISS & Mistral (23 tests)
*Nécessite `.env` configuré et `data/faiss_index/` généré :*
```bash
python -m pytest -m "integration or not integration" -v
```

### Relancer le benchmark d'évaluation sur le jeu annoté
```bash
python -m src.evaluation
```

### Synthèse des résultats recalculés (11 questions : 8 dans le périmètre Paris, 3 hors périmètre)

| Indicateur | Résultat POC | Détail |
| :--- | :---: | :--- |
| **Hit Rate @ 5 (Retrieval FAISS)** | **87,5 %** | 7 événements cibles sur 8 retrouvés dans le Top-5 vectoriel (`Q01` hors Top-5). |
| **Exactitude des faits clés (5/5)** | **87,5 %** | Présence vérifiée du `titre`, de la `date`, de l'`heure` (en heure de Paris, avec rejet des horaires UTC décalés), du `lieu` et du `lien URL` (100 % sur les 7 événements retrouvés). |
| **Similarité sémantique cosinus** | **0,9235** | Proximité moyenne (`mistral-embed`) entre réponses générées et réponses de référence sur Paris (`0,9242` sur l'ensemble des 11 questions). |
| **Rappel moyen des mots-clés** | **91,7 %** | Couverture des entités attendues après normalisation typographique. |
| **Garde-fous hors périmètre** | **100,0 %** | 3/3 refus stricts sans lien parasite (Lyon, recette de cuisine et hôtel à Paris). |
| **Taux global de réponses valides** | **90,9 %** | 10 scénarios sur 11 intégralement validés (`Q01` non remonté dans le Top-5 FAISS). |

---

## 7. Temps d'Exécution Mesurés (Environnement Local Python 3.12 / CPU)

| Étape du pipeline | Volume traité | Temps mesuré | Observation |
| :--- | :---: | :---: | :--- |
| **1. Extraction API Open Agenda** | 1 000 événements | **11,88 s** | Pagination Opendatasoft v2.1 par lots de 100 enregistrements. |
| **2. Nettoyage & Pré-processing** | 1 000 événements | **0,09 s** | Suppression HTML, filtrage Paris / < 1 an, conversion `Europe/Paris` et dédoublonnage. |
| **3. Découpage en extraits (Chunking)** | 2 703 segments | **1,01 s** | `chunk_size = 800`, `overlap = 120` + en-tête et métadonnées en heure de Paris. |
| **4. Vectorisation Mistral & Index FAISS** | 2 703 vecteurs (dim. 1024) | **157,36 s** (~2 min 37 s) | Par lots de 100 via `mistral-embed` (inclut les pauses anti-quota API). |
| **5. Recherche vectorielle FAISS pure** | Top-5 sur 2 703 vecteurs | **0,42 ms** | Comparaison exacte `IndexFlatL2` en mémoire vive (+ ~0,45 s d'encodage API de la question). |
