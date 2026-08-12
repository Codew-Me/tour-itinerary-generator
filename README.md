# Sri Lanka Travel Recommendation AI Agent

An evidence-grounded travel discovery agent for Sri Lanka. Uses **LangGraph** for genuine tool selection, **ChromaDB** for semantic review search, and **PostgreSQL/SQLite** for structured attraction data.

## Architecture

```
User → Streamlit UI → FastAPI → LangGraph Agent → Tool Selection
                                      ├── search_reviews → ChromaDB
                                      ├── search_attractions → PostgreSQL
                                      ├── get_destination_info → PostgreSQL + ChromaDB
                                      ├── list_by_district → PostgreSQL
                                      ├── compare_destinations → Both
                                      └── recommend_destinations → Both
                              → Evidence Evaluation → Recommendation
```

Unlike fixed RAG pipelines, the agent **chooses tools based on intent** — heritage filters use structured search; experience queries use review search; comparisons use the compare tool.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and set your LLM provider:

```bash
# OpenAI (default)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Or Ollama (local)
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
```

### 3. Prepare data

```bash
set PYTHONPATH=d:\proj3
python scripts/clean_data.py
python scripts/load_postgres.py
python scripts/build_vector_db.py   # ~10-15 min for 33K reviews
```

### 4. Run

```bash
# Terminal 1 — API
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — UI
streamlit run frontend/app.py --server.port 8501
```

Open **http://localhost:8501**

### Docker (PostgreSQL)

```bash
docker compose up -d postgres
# Set DATABASE_URL=postgresql+psycopg2://travel:travel@localhost:5432/sri_lanka_travel
```

## Evidence Rules

- **✓ Review-supported** — traveler review evidence exists in ChromaDB
- **ℹ Structured-data only** — category/mood/details only; no reviews in dataset
- Agent never fabricates reviews, prices, hours, or weather

## Ranking Methodology

Explainable scores (not raw review count):

| Factor | Weight |
|--------|--------|
| Preference match | 25% |
| Review relevance | 25% |
| Category match | 15% |
| Mood match | 15% |
| Location match | 10% |
| Evidence availability | 10% |

## Project Structure

See spec in project docs. Key paths: `src/agent/`, `src/tools/`, `api/main.py`, `frontend/app.py`.
