# EntropyBasedChunking

EntropyBasedChunking is a PDF-focused RAG (Retrieval-Augmented Generation) demo that combines:
- Entropy-aware sentence chunking
- Hybrid dense + sparse retrieval (Qdrant + BM25)
- Reciprocal Rank Fusion merging
- Cross-encoder reranking
- LLM answer generation

It ships with both a CLI workflow and a Streamlit UI for exploring PDFs interactively.

---

## What it does
1. Extracts text from a PDF
2. Splits text into semantically coherent chunks using entropy signals
3. Embeds and stores chunks in an in-memory Qdrant vector store
4. Builds a BM25 index over the same chunks
5. Retrieves relevant chunks using hybrid search + reranking
6. Generates a concise answer using a local LLM (Qwen2.5-3B-Instruct)

---

## Repository structure
- `main.py` — CLI entry point for PDF processing + Q&A
- `app.py` — Streamlit UI for interactive browsing and querying
- `chunker.py` — entropy-based chunking logic
- `retriever.py` — BM25 index, hybrid search, reranking
- `pdf_utils.py` — PDF text extraction + Qdrant storage
- `llm.py` — LLM initialization and answer generation
- `requirements.txt` — Python dependencies

---

## Setup

### Prerequisites
- Python and pip installed
- Internet access on first run to download models

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## Run the CLI
```bash
python main.py
```
You will be prompted to:
1. Provide a PDF file path
2. Ask questions about the PDF
3. Optionally choose `top_k` results per query

---

## Trial run example
Example CLI session using the "Attention Is All You Need" PDF (condensed to three representative questions):

### 1) Scaled dot-product attention formula
**Top retrieved chunks (abridged)**
- `Chunk #45` — “We compute the dot products … divide each by √dₖ, and apply a softmax function …”
- `Chunk #47` — “Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V”
- `Chunk #48` — “Dot-product attention is identical … except for the scaling factor 1/√dₖ.”

**LLM answer**
> Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V

### 2) Learning rate schedule
**Top retrieved chunks (abridged)**
- `Chunk #98` — “We varied the learning rate … lrate = d_model^-0.5 · min(step_num^-0.5, step_num · warmup_steps^-1.5) … warmup_steps = 4000.”
- `Chunk #99` — “Increase linearly for warmup_steps, then decrease proportional to the inverse square root of step_num.”

**LLM answer**
> lrate = d_model^-0.5 · min(step_num^-0.5, step_num · warmup_steps^-1.5), with warmup_steps = 4000.

### 3) Base model attention heads
**Top retrieved chunks (abridged)**
- `Chunk #57` — “We employ h = 8 parallel attention layers (heads) …”

**LLM answer**
> The base Transformer uses 8 attention heads.

---

## Run the Streamlit app
```bash
streamlit run app.py
```
Steps:
1. Upload a PDF
2. Enter a question
3. Review retrieved chunks and the generated answer

---

## Notes
- The vector database is in-memory and resets each run.
- First run downloads models:
  - SentenceTransformer (`all-MiniLM-L6-v2`)
  - Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`)
  - LLM (`Qwen2.5-3B-Instruct`)
- GPU is optional but recommended for faster inference.

---

## Troubleshooting
- If PDF extraction fails, verify the file is not scanned-only (image-only) content.
- NLTK will download `punkt_tab` automatically on first use if missing.
