# BioSynth-AI
# 🧠 BioSynth-AI: Academic PDF Analysis Assistant

BioSynth-AI is a Python-based web application built with Streamlit that empowers researchers, students, and professionals to analyze academic PDF files efficiently. The tool offers two primary modes:  
- **Single Document Analysis**: Extracts and summarizes structured content from one academic paper.  
- **Multi-Document Conceptual Analysis & Chat**: Enables upload of multiple papers to build conceptual maps and interactively explore relationships using RAG (Retrieval-Augmented Generation).

---

## 🚀 Features

### 📄 Single Document Mode
- Upload and parse academic PDFs using PyMuPDF.
- Automatically extract key sections (Abstract, Methods, Results, Discussion, Limitations).
- Summarize each section using OpenAI (or local fallback with NLTK).
- Export a professionally formatted PDF report.

### 📚 Multi-Document Mode
- Upload two or more PDFs for comparative analysis.
- Generate a **concept map** of relationships between papers based on content similarity.
- Use a chatbot to ask questions about uploaded papers via **FAISS** + **RAG** (Retrieval-Augmented Generation).
- Identify shared sections, publication years, and conceptual overlaps.

---

## 🧰 Technologies Used

| Tool/Library | Purpose |
|--------------|---------|
| `Streamlit` | Interactive web application framework |
| `PyMuPDF (fitz)` | PDF parsing and text extraction |
| `OpenAI API` | GPT-based summarization and chat |
| `NLTK` | Tokenization and local fallback summarization |
| `LangChain` | Retrieval and document chaining logic |
| `FAISS` | Vector similarity indexing for document chunks |
| `Graphviz` / `NetworkX` / `Plotly` | Concept map generation and visualization |
| `FPDF` | PDF export of analysis results |


🔧 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/BioSynthAI.git
cd BioSynthAI

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
cp .env.example .env
# Add your API key to .env as: OPENAI_API_KEY=your-key-here

Crete a virtual environment in VS (.venv/bin/activate
Run the app with this command
streamlit run app.py
Deploy the app
