# DocuMind AI: Offline-First Hybrid RAG Question Answering System 🤖📄

A high-performance, **Retrieval-Augmented Generation (RAG)** document question-answering system designed for offline-first, privacy-preserving operations on custom PDF databases. This project was developed as a Week 7 assignment/internship project, focusing on architectural logic, exact lexical retrieval, and high-accuracy offline synthesis using Streamlit.

---

## 🌟 Key Features

1. **Streamlit Python Application**:
   * A clean, interactive UI designed for document upload, index management, query input, and real-time response rendering.
2. **Robust Offline QA Engine (No Key Needed)**:
   * Falls back to a deterministic, high-accuracy extractive question answering pipeline when no external API key is provided.
   * Synthesizes readable responses with precise source citations (Document Name & Page Number).
3. **Hybrid Keyword & Dense Search (BM25 + TF-IDF)**:
   * Combines **TF-IDF vector cosine similarity** (for semantic relevance) with **BM25 Lexical Scoring** (to prioritize rare terms like names, commands, or tech terms).
4. **Targeted Resume/Section Extractor**:
   * Uses semantic synonyms to identify user intent (asking for *projects*, *experience*, *education*, *skills*, or *contact*).
   * Implements a forward-decaying flow algorithm to extract consecutive list elements and bullet points while preventing section overflow.
5. **Document-Level Context Boosting**:
   * When rare query keywords uniquely match a specific document (e.g. searching for your name on a resume), the system automatically boosts and expands context to retrieve all relevant pages of that target document, preventing unrelated documents from overriding results.
6. **Advanced PDF Processing**:
   * Fast text parsing with phrase-level deduplication to clean up page wrap fragments, ligatures, and scanning artifacts.

---

## 📁 Project Structure

```directory
week7/
├── data/                   # Directory where uploaded PDFs are stored
│   ├── CompTIA+Security++(SY0-701)Study+Plan.pdf
│   ├── Linux commands (1).pdf
│   ├── Resume.pdf
│   └── WEB HACKING AND PENETRATION TESTING.pdf
├── app.py                  # Streamlit Interface app
├── chatbot.py              # Extractive logic & answer synthesis helper
├── vectorstore.py          # Local Vector DB, BM25 logic & PDF chunking helper
├── requirements.txt        # Project python dependencies
├── run.bat                 # Single-click launcher batch script
├── vector_store.json       # Persisted vector database (highly optimized ~1.6 MB)
└── README.md               # Project documentation
```

---

## 🛠️ How it Works Under the Hood

### 1. Custom Embedding Vectorizer
Because standard local vectorizers require heavy deep-learning dependencies (like PyTorch or SentenceTransformers) that take gigabytes of space, DocuMind implements a custom **Sublinear TF-IDF Hashing Vectorizer** in NumPy. It filters common stop words, extracts unigrams and bigrams, and normalizes vectors to the unit circle for quick cosine calculations.

### 2. Hybrid Lexical Scorer (BM25)
Standard embeddings fail on queries targeting unique terms (like searching for `"what is the grep command"` in a folder containing thousands of general computer science pages). DocuMind runs a local **BM25 scorer**:
$$\text{Score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
This assigns extremely high score weights to terms that only appear in a single document (like your name in a resume).

### 3. Forward-Decaying Flow Parser
To isolate sections without LLMs, the system splits merged texts into units and scans them. When a section heading corresponding to the query intent is hit (e.g., `Projects`), a tracking flow starts at maximum score. As it moves down consecutive lines, the score decays forward (`flow_score *= 0.85`), keeping the bullets intact. If it encounters a heading for a different category (e.g., `Skills`), the flow instantly resets to `0`, ensuring clean isolation.

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
Make sure you have python (version 3.9 to 3.12 recommended) installed on your system.

### 2. Run the Automatic Launcher
Simply run the batch script from your terminal:
```powershell
.\run.bat
```
This script will automatically:
1. Create a Python Virtual Environment (`venv/`).
2. Upgrade `pip` and install all required libraries from `requirements.txt`.
3. Launch the Streamlit application interface.

### 3. Adding and Indexing Custom Files
1. Open the application, go to **Manage Documents** tab.
2. Upload any PDF/TXT file using the drag-and-drop box.
3. Click the **Index** button next to the file to split and vectorize it.
4. Switch back to the **Chat Assistant** tab and ask questions!

---

## 💻 Technical Stack

* **Core Language**: Python 3.12
* **Application Framework**: Streamlit
* **Libraries**: NumPy (Vector calculations), PyPDF (Document Parsing)
