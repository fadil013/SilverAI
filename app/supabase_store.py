import os
from supabase import create_client, Client
import llm as llm_module

_sb: Client = None

def init(url: str, key: str):
    global _sb
    if not url or not key:
        return
    try:
        _sb = create_client(url, key)
    except Exception:
        _sb = None

def _embed(text: str) -> list[float]:
    return llm_module.embed(text)

def store_chunks(chunks: list[str], source: str):
    if not _sb:
        return
    rows = []
    for i, chunk in enumerate(chunks):
        emb = _embed(chunk)
        rows.append({
            "content": chunk,
            "source": source,
            "chunk_index": i,
            "embedding": emb,
        })
    _sb.table("documents").insert(rows).execute()

def search(query: str, top_k: int = 5) -> list[str]:
    if not _sb:
        return []
    emb = _embed(query)
    result = _sb.rpc("match_documents", {
        "query_embedding": emb,
        "match_count": top_k,
    }).execute()
    return [r["content"] for r in result.data]
