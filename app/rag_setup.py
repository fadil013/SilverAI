"""
RAG layer using LightRAG (knowledge graph).
Falls back to simple in-memory cosine-similarity search if LightRAG is unavailable.
"""
import os
import asyncio
import threading
import numpy as np
import llm as llm_module

_rag = None
_fallback_chunks: list[dict] = []  # [{text, embedding}]
_use_fallback = False


def _run_in_thread(coro):
    """Run an async coroutine in a fresh thread with its own event loop."""
    result = {}

    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result["value"] = loop.run_until_complete(coro)
        except Exception as e:
            result["error"] = e
        finally:
            loop.close()

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def init(working_dir: str = "./lightrag_data"):
    global _rag, _use_fallback
    os.makedirs(working_dir, exist_ok=True)
    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.utils import EmbeddingFunc

        async def _embed(texts: list) -> np.ndarray:
            import google.generativeai as genai
            embeddings = []
            for text in texts:
                r = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document",
                )
                embeddings.append(r["embedding"])
            return np.array(embeddings)

        _rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=llm_module.chat_async,
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=_embed,
            ),
        )
        _use_fallback = False
        print("[RAG] LightRAG initialized.")
    except Exception as e:
        _use_fallback = True
        print(f"[RAG] LightRAG unavailable ({e}), using in-memory fallback.")


def _embed_text(text: str) -> list[float]:
    import google.generativeai as genai
    r = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document",
    )
    return r["embedding"]


def _cosine_top_k(query_emb: list, k: int = 5) -> list[str]:
    if not _fallback_chunks:
        return []
    q = np.array(query_emb)
    scores = []
    for chunk in _fallback_chunks:
        v = np.array(chunk["embedding"])
        score = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9))
        scores.append((score, chunk["text"]))
    scores.sort(reverse=True)
    return [t for _, t in scores[:k]]


def insert(text: str):
    if _use_fallback:
        from pdf_parser import chunk_text
        for chunk in chunk_text(text, chunk_size=500, overlap=50):
            emb = _embed_text(chunk)
            _fallback_chunks.append({"text": chunk, "embedding": emb})
        return
    _run_in_thread(_rag.ainsert(text))


def query(question: str, mode: str = "hybrid") -> str:
    if _use_fallback:
        emb = _embed_text(question)
        chunks = _cosine_top_k(emb, k=5)
        return "\n\n".join(chunks) if chunks else ""
    result = _run_in_thread(_rag.aquery(question, param=__import__("lightrag").QueryParam(mode=mode)))
    return result or ""
