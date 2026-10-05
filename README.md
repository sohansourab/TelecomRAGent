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

- Python 3.12
- Ollama
- Approximately 8 GB free disk for the Mistral Q4 model and dependencies
- 4 GB VRAM is supported; Ollama may offload layers to CPU if required

## Setup

```bash
cd CallDropAI
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install --python .venv/bin/python -r requirements.txt
```

Install CPU-only Torch first on machines where the GPU is tight. If the full requirements command runs first, the resolver may select several hundred megabytes of CUDA libraries.

Install and start Ollama, then pull the configured model:

```bash
ollama serve
ollama pull mistral:7b-instruct-q4_K_M
```

## Build the index

```bash
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/preprocess_data.py
.venv/bin/python scripts/index_chroma.py --rebuild --batch-size 32
```

## Run

Start the backend in one terminal:

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Start the frontend in another:

```bash
source .venv/bin/activate
streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

Open <http://localhost:8501>. API documentation is available at <http://localhost:8000/docs>.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q backend frontend scripts
curl http://127.0.0.1:8000/health
```

Ask questions about regions, operators, network types, ratings, call drops, and remediation priorities. The response includes retrieved sources and the ordered tool trace.