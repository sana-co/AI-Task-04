import os
import sys
import re
import json
import hashlib
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI

# ----------------------------
# Config
# ----------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 900       # characters
CHUNK_OVERLAP = 150    # characters
TOP_K = 5              # retrieved chunks per question

INDEX_DIR = ".rag_cache"
# ----------------------------


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        pages_text.append(t)

    text = "\n".join(pages_text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def embed_texts(client: OpenAI, texts: List[str], model: str = EMBED_MODEL, batch_size: int = 64) -> np.ndarray:
    """Embed texts in batches and return float32 array (N, D)."""
    all_vecs: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        all_vecs.extend([item.embedding for item in resp.data])

    return np.array(all_vecs, dtype=np.float32)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    """L2 normalize each row (for cosine similarity via inner product)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-10
    return x / norms


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """
    Cosine similarity = inner product on L2-normalized vectors.
    Use IndexFlatIP for exact search (simple and reliable).
    """
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def cache_paths(pdf_hash: str) -> Tuple[str, str]:
    os.makedirs(INDEX_DIR, exist_ok=True)
    index_path = os.path.join(INDEX_DIR, f"{pdf_hash}.faiss")
    meta_path = os.path.join(INDEX_DIR, f"{pdf_hash}.json")
    return index_path, meta_path


def load_cached_index(pdf_hash: str) -> Tuple[faiss.Index, List[str]]:
    index_path, meta_path = cache_paths(pdf_hash)
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        raise FileNotFoundError("No cached index found for this PDF.")

    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    chunks = meta["chunks"]
    return index, chunks


def save_cached_index(pdf_hash: str, index: faiss.Index, chunks: List[str]) -> None:
    index_path, meta_path = cache_paths(pdf_hash)
    faiss.write_index(index, index_path)

    meta: Dict[str, Any] = {
        "pdf_hash": pdf_hash,
        "chunks": chunks,
        "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def retrieve(index: faiss.Index, chunks: List[str], query_vec: np.ndarray, top_k: int = TOP_K) -> List[Tuple[int, float, str]]:
    """
    query_vec: shape (D,) float32 normalized
    Returns list of (chunk_id, score, chunk_text).
    """
    q = query_vec.reshape(1, -1).astype(np.float32)
    scores, ids = index.search(q, top_k)

    results = []
    for chunk_id, score in zip(ids[0].tolist(), scores[0].tolist()):
        if chunk_id == -1:
            continue
        results.append((chunk_id, float(score), chunks[chunk_id]))
    return results


def answer_question(client: OpenAI, question: str, retrieved: List[Tuple[int, float, str]]) -> str:
    context_blocks = []
    for cid, score, text in retrieved:
        context_blocks.append(f"[Chunk {cid} | score={score:.3f}]\n{text}")

    context = "\n\n---\n\n".join(context_blocks)

    system = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context from the PDF. "
        "If the answer is not in the context, say exactly: 'I don't know based on the provided PDF.' "
        "Be concise and precise."
    )

    user = f"""CONTEXT (retrieved from PDF):
{context}

QUESTION:
{question}
"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def main():
    if len(sys.argv) < 2:
        print('Usage: python main.py "C:\\path\\to\\file.pdf"')
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    if os.path.isdir(pdf_path) or not pdf_path.lower().endswith(".pdf"):
        print("Error: Please provide a PDF file path (ending with .pdf), not a folder.")
        sys.exit(1)

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found. Put it in a .env file.")
        sys.exit(1)

    # Safety check against the earlier issue you hit (quotes in key)
    if '"' in api_key or "'" in api_key:
        print("Error: OPENAI_API_KEY looks malformed (contains quotes). Remove quotes from .env.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"Loading PDF from: {pdf_path}")
    pdf_hash = sha256_file(pdf_path)

    # Try load cached FAISS index + chunks
    try:
        index, chunks = load_cached_index(pdf_hash)
        print("Loaded cached FAISS index (no re-embedding needed).")
    except FileNotFoundError:
        # Build from scratch
        text = read_pdf_text(pdf_path)
        if not text:
            print("Error: No text could be extracted from this PDF.")
            sys.exit(1)

        print(f"PDF loaded. Characters: {len(text)}")
        print("Chunking text...")
        chunks = chunk_text(text)
        if not chunks:
            print("Error: Chunking produced no chunks.")
            sys.exit(1)

        print(f"Created {len(chunks)} chunks.")
        print("Embedding chunks (this may take a moment)...")
        vecs = embed_texts(client, chunks)          # (N, D)
        vecs = normalize_rows(vecs)                # for cosine via inner product
        index = build_faiss_index(vecs)
        save_cached_index(pdf_hash, index, chunks)
        print("FAISS index built and cached.")

    print("\nRAG is ready. Ask questions about the PDF.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        try:
            # Embed query, normalize, retrieve, answer
            q_vec = embed_texts(client, [q])[0]
            q_vec = (q_vec / (np.linalg.norm(q_vec) + 1e-10)).astype(np.float32)

            retrieved = retrieve(index, chunks, q_vec, TOP_K)
            ans = answer_question(client, q, retrieved)
            print("\nAnswer:\n" + ans + "\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
