# TelecomRAGent

TelecomRAGent is a local RAG application for investigating Indian mobile voice-quality data. It combines a FastAPI backend, ChromaDB semantic retrieval, LangChain tools, Ollama reasoning, and a Streamlit operations console.

## Architecture

```text
TRAI MyCall CSV
      |
      v
cleaning -> MiniLM embeddings -> ChromaDB
                                      |
Streamlit -> FastAPI -> retrieve_logs -> analyze_root_cause
                                      -> generate_resolution
                                      -> generate_report -> Ollama
```

The selected MyCall data is released under India's National Data Sharing and Accessibility Policy. The repository mirror is documented in [scripts/download_data.py](scripts/download_data.py). Raw and processed data are ignored by Git.

## Requirements

- Linux, WSL, or macOS
- Python 3.12
- `uv` package manager
- Ollama
- Approximately 6-8 GB free disk
- 4 GB VRAM is supported; Ollama may offload layers to CPU

## Run After Cloning

Clone the repository and enter the project directory:

```bash
git clone git@github.com:sohansourab/TelecomRAGent.git
cd TelecomRAGent
```

Create the Python environment. Install CPU-only Torch first; otherwise the dependency resolver may download large CUDA packages:

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install --python .venv/bin/python \
      --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install --python .venv/bin/python -r requirements.txt
```

Create local configuration:

```bash
cp .env.example .env
```

The default local configuration uses:

```text
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:3b-instruct-q4_K_M
BACKEND_URL=http://127.0.0.1:8000
```

Install Ollama from <https://ollama.com/download>, start it if it is not already running, and download the model:

```bash
ollama serve
ollama pull llama3.2:3b-instruct-q4_K_M
```

If `ollama serve` reports `address already in use`, Ollama is already running. Verify it with:

```bash
curl http://127.0.0.1:11434/api/tags
```

Build the local dataset and vector index once:

```bash
python scripts/download_data.py
python scripts/validate_data.py
python scripts/preprocess_data.py
python scripts/index_chroma.py --rebuild --batch-size 32
```

This creates ignored local artifacts under `data/raw`, `data/processed`, and `data/chroma`.

## Start Locally

Use three terminals from the project directory.

Terminal 1, Ollama:

```bash
source .venv/bin/activate
ollama serve
```

Terminal 2, FastAPI:

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Terminal 3, Streamlit:

```bash
source .venv/bin/activate
streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

Open <http://localhost:8501>. API documentation is available at <http://localhost:8000/docs>.

## Verify

Check the backend before opening the frontend:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/stats
```

Run the tests:

```bash
python -m unittest discover -s tests -v
python -m compileall -q backend frontend scripts
```

Test a real question:

```bash
curl -X POST http://127.0.0.1:8000/query \
      -H "Content-Type: application/json" \
      -d '{"question":"Why are call drops high in Uttar Pradesh?"}'
```

The response contains an answer, retrieved sources, and this ordered tool trace:

```text
retrieve_logs -> analyze_root_cause -> generate_resolution -> generate_report
```

## Troubleshooting

### `ModuleNotFoundError: No module named pandas`

The virtual environment is active but dependencies were not installed. Run:

```bash
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install --python .venv/bin/python -r requirements.txt
```

### `Backend unavailable` in Streamlit

FastAPI is not running. Start Terminal 2 and verify:

```bash
curl http://127.0.0.1:8000/health
```

### Ollama connection refused

Start Ollama and confirm the model exists:

```bash
ollama serve
ollama list
```

### Port already in use

Stop stale local application processes, then restart:

```bash
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 8501/tcp 2>/dev/null || true
```

Ask questions about regions, operators, network types, ratings, call drops, and remediation priorities. Raw data, model files, ChromaDB, `.env`, and `.venv` are intentionally excluded from Git.