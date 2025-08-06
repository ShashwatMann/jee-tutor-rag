import os
import re
import uuid
from pathlib import Path
import fitz  # pip install pymupdf
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss



def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        page_text = page.get_text("text")
        page_text = re.sub(r"Page\s*\d+", "", page_text)
        page_text = re.sub(r"\s{2,}", " ", page_text)
        text += page_text.strip() + "\n"
    return text

def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

def infer_subject(filename):
    name = filename.lower()
    if "physics" in name:
        return "Physics"
    elif "chemistry" in name:
        return "Chemistry"
    elif "math" in name:
        return "Maths"
    else:
        return "Unknown"

def build_index(data_dir="data_raw", index_dir="index"):
    print(f"Looking for PDFs in: {data_dir}")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    all_texts, metadata = [], []

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                path = os.path.join(root, file)
                raw_text = extract_text(path)
                chunks = chunk_text(raw_text)
                subject = infer_subject(file)

                for chunk in chunks:
                    if len(chunk.strip()) < 40 or len(re.findall(r"[a-zA-Z]", chunk)) < 5:
                        continue
                    all_texts.append(chunk)
                    metadata.append({
                        "id": str(uuid.uuid4()),
                        "source": file,
                        "subject": subject,
                        "text": chunk
                    })

    vectors = model.encode(all_texts, normalize_embeddings=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.array(vectors, dtype="float32"))

    Path(index_dir).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, f"{index_dir}/jee.faiss")
    np.save(f"{index_dir}/meta.npy", np.array(metadata, dtype=object))
    print(f"\u2705 Indexed {len(metadata)} chunks from {len(files)} PDFs.")

if __name__ == "__main__":
    build_index()