# 🚗 Multimodal Car Dashboard Assistant (RAG + Vision)

An AI-powered automotive assistant combining Computer Vision (**YOLOv8**) and Retrieval-Augmented Generation (**RAG via ChromaDB & Llama 3**) to help drivers identify car dashboard warning lights and retrieve exact handling instructions from the vehicle owner's manual.

---

## 📽️ Demo & Video Walkthrough

> 🎬 **[Click Here to Watch the Full Video Walkthrough & Live Demo](YOUR_VIDEO_LINK_HERE)**

*(Optionally embed a GIF or screenshot of the running app here)*

---

## 📌 Project Overview

When a car dashboard warning light illuminates, drivers often struggle to locate the precise section in a 400+ page owner's manual. 

This project solves that by delivering an end-to-end multimodal pipeline:
1. **Vision Identification:** Detects dashboard warning symbols from uploaded images using fine-tuned YOLOv8.
2. **Context Retrieval:** Searches a local **ChromaDB** vector database containing embedded manual chunks (`all-MiniLM-L6-v2`).
3. **Grounded Generation:** Uses **Ollama (Llama 3)** to generate clear, safety-focused advice grounded in cited manual pages.
4. **Intelligent Fallback:** If an unlearned symbol or edge-case image is uploaded, the system automatically falls back to text-query RAG, ensuring helpful guidance is always provided.

---

## 🏗 System Architecture

```text
                               ┌──────────────────────────┐
                               │     Streamlit UI         │
                               └────────────┬─────────────┘
                                            │ (POST /api/v1/analyze)
                               ┌────────────▼─────────────┐
                               │     FastAPI Backend      │
                               └──────┬─────────────┬─────┘
                                      │             │
                    ┌─────────────────▼──┐       ┌──▼──────────────────┐
                    │  YOLOv8 Vision     │       │  User Text Query    │
                    │  (Symbol Detection)│       │                     │
                    └────────┬───────────┘       └──┬──────────────────┘
                             │                      │
                             └──────────┬───────────┘
                                        │ (Fused Context Query / Fallback)
                               ┌────────▼───────────┐
                               │  ChromaDB Vector   │
                               │  Retrieval Engine  │
                               └────────┬───────────┘
                                        │ (Retrieved Manual Chunks)
                               ┌────────▼───────────┐
                               │ Ollama (Llama 3)   │
                               │  Grounded Response │
                               └────────────────────┘
```

**Tech Stack:**
- **Computer Vision:** YOLOv8 (Ultralytics) fine-tuned on custom dashboard datasets.
- **Embeddings & Vector DB:** sentence-transformers/all-MiniLM-L6-v2, ChromaDB.
- **LLM Engine:** Ollama (Llama 3).
- **Backend:** FastAPI, Uvicorn, Pydantic, Pytest.
- **Frontend:** Streamlit, python-dotenv.
- **Language & Runtime:** Python 3.10+.

**Project Structure:**
```text
├── backend/
│ ├── app/
│ │ ├── api/ # Route handlers (/health, /analyze, /query)
│ │ ├── core/ # Config & environment settings
│ │ ├── schemas/ # Pydantic request/response schemas
│ │ ├── services/ # Vision detection & RAG retrieval services
│ │ └── utils/ # Helper utilities
│ ├── tests/
│ │ └── test_query.py # Pytest suite (health check & validation handling)
│ ├── main.py # FastAPI entrypoint
│ ├── .env.example
│ └── requirements.txt
├── frontend/
│ ├── app.py # Streamlit dashboard interface
│ ├── .env.example
│ └── requirements.txt
├── notebooks/
│ └── rag_pipeline.ipynb # Notebook: data loading, chunking, YOLO tuning, & RAG eval
├── data/
│ └── vector_store/ # Persisted ChromaDB vector database
├── README.md
└── .gitignore
```

**Environment & Setup Instructions:**
**Prerequisites:**
- Python 3.10+
- Ollama installed and running locally (`ollama pull llama3`)
- Git

**1. Clone the Repository**
```bash
git clone https://github.com/samagamal996/Car-Dashboard-Detection-Project.git
cd Car-Dashboard-Detection-Project
```

**2. Set Up Virtual Environment**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

**3. Backend Setup**
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
```

Start the FastAPI backend:
```bash
uvicorn main:app --reload --port 8000
```
API docs available at: http://localhost:8000/docs

**4. Frontend Setup**

Open a new terminal tab, activate `.venv`, and navigate to frontend:
```bash
cd frontend
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```
App will run at: http://localhost:8501

**Backend (`backend/.env`):**

| Variable | Default Value | Description |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | Backend API root URL |
| `CHROMA_DB_PATH` | `./data/vector_store` | Path to persisted vector store |
| `MODEL_WEIGHTS_PATH` | `./weights/best.pt` | Path to fine-tuned YOLOv8 weights |

**Frontend (`frontend/.env`):**

| Variable | Default Value | Description |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | Base URL for backend calls |

**Running Automated Tests:**
```bash
pytest backend/test_query.py
```

**Application Screenshots:**

<img width="1919" height="965" alt="image" src="https://github.com/user-attachments/assets/095352bf-c862-4ffa-8ff1-763a62710a06" />
<img width="1919" height="972" alt="image" src="https://github.com/user-attachments/assets/c3729d68-5ed3-48d4-9aaf-9d9d0ad7e697" />
<img width="1907" height="960" alt="image" src="https://github.com/user-attachments/assets/3582d0b3-944e-469e-8ff2-a8055ee645b6" />



