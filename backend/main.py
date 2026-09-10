import os
import shutil
from typing import List, Optional
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import ollama
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI(
    title="Tell-Tale Dashboard Assistant API",
    description="Backend API combining YOLOv8 vision detection with ChromaDB + Ollama RAG",
    version="1.0.0",
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL PATHS & INITIALIZATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "backend", "data", "vector_store")
# Pointing to your latest trained YOLO weights directory
YOLO_WEIGHTS_PATH = os.path.join(
    BASE_DIR, "runs", "detect", "train-3", "weights", "best.pt"
)

# Initialize ChromaDB Client
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = chroma_client.get_or_create_collection(
    name="car_manual", embedding_function=embedding_fn
)

# Load Fine-Tuned YOLO Model
if os.path.exists(YOLO_WEIGHTS_PATH):
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)
else:
    # Fallback to base model if custom weights aren't found at exact path
    yolo_model = YOLO("yolov8n.pt")


# --- CORE LOGIC FUNCTIONS ---
def retrieve_context(query: str, top_k: int = 3):
    results = collection.query(query_texts=[query], n_results=top_k)
    retrieved_texts = results["documents"][0]
    retrieved_metas = results["metadatas"][0]

    context = ""
    sources = []
    for text, meta in zip(retrieved_texts, retrieved_metas):
        context += f"\n--- Source: {meta['source']} ---\n{text}\n"
        sources.append(meta["source"])

    return context, list(set(sources))


def detect_symbols_from_bytes(
    image_bytes: bytes, conf_threshold: float = 0.15
) -> List[str]:
    temp_img_path = os.path.join(BASE_DIR, "temp_upload.jpg")
    with open(temp_img_path, "wb") as f:
        f.write(image_bytes)

    try:
        results = yolo_model(temp_img_path, conf=conf_threshold)
        detected_classes = []
        for box in results[0].boxes:
            cls_id = int(box.cls)
            detected_classes.append(yolo_model.names[cls_id])
        return list(set(detected_classes))
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)


# --- DATA MODELS ---
class QueryRequest(BaseModel):
    user_query: str
    detected_symbol: Optional[str] = None


class RAGResponse(BaseModel):
    detected_symbols: List[str]
    answer: str
    sources: List[str]


# --- API ENDPOINTS ---
@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Dashboard Assistant API is running.",
    }


@app.post("/api/v1/detect", response_model=List[str])
async def detect_symbol(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Uploaded file must be an image."
        )

    contents = await file.read()
    detected = detect_symbols_from_bytes(contents)
    return detected


@app.post("/api/v1/analyze", response_model=RAGResponse)
async def analyze_dashboard(
    user_query: str = Form(
        "What action should I take right now for this warning light?"
    ),
    file: Optional[UploadFile] = File(None),
):
    detected_symbols = []

    # 1. Vision Detection (if image uploaded)
    if file:
        contents = await file.read()
        detected_symbols = detect_symbols_from_bytes(contents)

    active_symbol = detected_symbols[0] if detected_symbols else None

    # 2. Build Query with Context
    if active_symbol:
        query_with_vision = (
            f"[Warning Light Detected: {active_symbol}] {user_query}"
        )
    else:
        query_with_vision = user_query

    # 3. Vector DB Retrieval
    context, sources = retrieve_context(query_with_vision, top_k=3)

    # 4. LLM Generation via Ollama
    prompt = f"""You are a helpful car owner assistant. Answer the question using ONLY the provided car manual context below. 
If the information is not in the context, say "I cannot find this in the owner manual." Always cite the manual page numbers provided.

Context:
{context}

User Question: {query_with_vision}

Answer:"""

    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
        )
        answer_text = response["message"]["content"]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ollama generation failed: {str(e)}"
        )

    return RAGResponse(
        detected_symbols=detected_symbols,
        answer=answer_text,
        sources=sources,
    )