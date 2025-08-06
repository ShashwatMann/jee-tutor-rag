from fastapi import FastAPI, Request, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from pathlib import Path
import numpy as np
import faiss
import requests
import json
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

embedder = None
index = None
metadata = None

chat_sessions: List[Dict] = []
correction_log: List[Dict] = []

correction_path = Path("corrections.json")
session_path = Path("chat_logs.json")

@app.on_event("startup")
def load_resources():
    global embedder, index, metadata

    # Load correction memory
    if correction_path.exists():
        with open(correction_path, "r") as f:
            correction_log.extend(json.load(f))
        print(f"✅ Loaded {len(correction_log)} corrections.")
    else:
        print("🟡 No previous corrections found.")

    # Load previous chat sessions
    if session_path.exists():
        with open(session_path, "r") as f:
            chat_sessions.extend(json.load(f))
        print(f"✅ Loaded {len(chat_sessions)} chat sessions.")
    else:
        print("🟡 No previous chat sessions found.")

    # Load embedder, FAISS index, and metadata
    print("Loading embedding model...")
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

    print("Loading FAISS index...")
    index_path = "index/jee.faiss"
    index = faiss.read_index(index_path)

    print("Loading metadata...")
    metadata = np.load("index/meta.npy", allow_pickle=True)
    print(f"✅ Loaded {len(metadata)} chunks.")


def detect_subject(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["velocity", "force", "motion", "charge", "energy", "current", "magnetic", "circuit", "resistance"]):
        return "Physics"
    elif any(w in q for w in ["mole", "reaction", "acid", "base", "organic", "bond", "atom", "compound"]):
        return "Chemistry"
    elif any(w in q for w in ["integration", "differentiation", "matrix", "vector", "equation", "probability", "limit", "function", "graph", "number"]):
        return "Maths"
    return "Unknown"

def search(question: str, k: int = 5):
    subject = detect_subject(question)
    print(f"Detected subject: {subject}")

    matching_indices = [
        i for i, meta in enumerate(metadata)
        if meta.get("subject") == subject or subject == "Unknown"
    ]

    if not matching_indices:
        matching_indices = list(range(len(metadata)))

    all_vectors = index.reconstruct_n(0, index.ntotal)
    selected_vectors = np.array([all_vectors[i] for i in matching_indices])
    query_vec = embedder.encode([question], normalize_embeddings=True).astype(np.float32)

    scores = np.dot(selected_vectors, query_vec.T).squeeze()
    top_k_idx = np.argsort(scores)[-k:][::-1]
    top_metadata_idx = [matching_indices[i] for i in top_k_idx]

    return [metadata[i] for i in top_metadata_idx]

def build_prompt(question: str, chunks: List[Dict]) -> str:
    context = "\n\n".join(f"{i+1}. {chunk['text']}" for i, chunk in enumerate(chunks))
    return (
        "You are an expert JEE tutor. Use the provided NCERT context to answer this Physics, Chemistry, or Math question accurately.\n\n"
        "Respond step-by-step using formulas and logic wherever required. Avoid assumptions not grounded in the context. "
        "Do not hallucinate facts. Show working wherever applicable, and make sure the final answer is clearly explained.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

def query_ollama(prompt: str) -> str:
    try:
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3:instruct",
            "prompt": prompt,
            "stream": False
        })
        return res.json().get("response", "Sorry, I couldn't find an answer.").strip()
    except Exception as e:
        print("Ollama error:", e)
        return "Ollama API error. Please check if it's running."

@app.post("/correct", response_class=HTMLResponse)
def correct_answer(request: Request, question: str = Form(...), correction: str = Form(...), chat_id: int = Form(...)):
    correction_entry = {
        "question": question.strip(),
        "correction": correction.strip()
    }
    correction_log.append(correction_entry)

    with open(correction_path, "w") as f:
        json.dump(correction_log, f, indent=2)

    if chat_id < len(chat_sessions):
        chat_sessions[chat_id]["messages"].append({
            "question": question.strip(),
            "answer": correction.strip()
        })

    with open(session_path, "w") as f:
        json.dump(chat_sessions, f, indent=2)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "question": "",
        "answer": f"✅ Correction saved and memory updated for: \"{question}\"",
        "sources": [],
        "chat_history": chat_sessions[chat_id]["messages"] if chat_id < len(chat_sessions) else [],
        "saved_chats": chat_sessions
    })

@app.post("/delete_chat/{chat_id}")
def delete_chat(chat_id: int):
    if 0 <= chat_id < len(chat_sessions):
        del chat_sessions[chat_id]
        # Save updated sessions
        with open(session_path, "w") as f:
            json.dump(chat_sessions, f, indent=2)
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

@app.post("/", response_class=HTMLResponse)
def handle_query(request: Request, question: str = Form(...), k: int = Form(5), chat_id: int = Form(0)):
    q_strip = question.strip()

    for c in correction_log:
        if c["question"].strip() == q_strip:
            print("🔁 Using corrected response")
            answer = c["correction"]
            break
    else:
        chunks = search(q_strip, k)
        prompt = build_prompt(q_strip, chunks)
        answer = query_ollama(prompt)

    if chat_id >= len(chat_sessions):
        chat_sessions.append({
            "title": question[:40] + "..." if len(question) > 40 else question,
            "messages": []
        })
        chat_id = len(chat_sessions) - 1

    chat_sessions[chat_id]["messages"].append({
        "question": q_strip,
        "answer": answer
    })

    if chat_sessions[chat_id]["title"] == "New Chat":
        chat_sessions[chat_id]["title"] = question[:40] + "..." if len(question) > 40 else question

    with open(session_path, "w") as f:
        json.dump(chat_sessions, f, indent=2)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "question": q_strip,
        "answer": answer,
        "sources": [c.get("source", "") for c in search(q_strip)],
        "chat_history": chat_sessions[chat_id]["messages"],
        "saved_chats": chat_sessions,
        "chat_id": chat_id
    })

@app.get("/", response_class=HTMLResponse)
def homepage(request: Request, chat_id: int = 0, new_chat: bool = False):
    if new_chat or chat_id >= len(chat_sessions):
        chat_sessions.append({
            "title": "New Chat",
            "messages": []
        })
        chat_id = len(chat_sessions) - 1

    return templates.TemplateResponse("index.html", {
        "request": request,
        "question": "",
        "answer": "",
        "sources": [],
        "chat_history": chat_sessions[chat_id]["messages"],
        "saved_chats": chat_sessions,
        "chat_id": chat_id
    })

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
