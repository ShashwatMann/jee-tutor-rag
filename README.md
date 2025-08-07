# JEE Tutor – RAG-based Intelligent NCERT & JEE QA Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) application designed to help students prepare for the **JEE Main and Advanced** exams using **NCERT** textbooks and previous year papers.

This project supports question-answering, correction feedback, and chat history — all accessible through a user-friendly web interface.

---

## 🔧 Features

- 📘 RAG-based QA on NCERT content (Classes 11 & 12)
- ✍️ Correction submission system to improve model accuracy
- 💬 Chat history with saved conversations
- 🌙 Dark/light mode toggle
- 🚀 FastAPI backend + FAISS vector store + SentenceTransformers
- 🧠 Uses local model (Ollama/Mistral or compatible)

---

## 🗂️ Project Structure

jee-tutor-rag/
├── app.py # Main FastAPI app
├── rag.py # RAG logic (retrieval + generation)
├── templates/index.html # Frontend UI (Jinja2 template)
├── static/styles.css # Styling
├── index/ # FAISS vector store + metadata
│ ├── jee.faiss
│ └── meta.npy
├── data_raw/ # Raw NCERT PDFs
├── chat_logs.json # Stores chat history
├── corrections.json # Stores user corrections
├── .gitignore
├── requirements.txt
└── README.md



---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ShashwatMann/jee-tutor-rag.git
cd jee-tutor-rag
