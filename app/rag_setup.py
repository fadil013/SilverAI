import os
import asyncio
import threading
import nest_asyncio
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np
import llm as llm_module

nest_asyncio.apply()

_rag: LightRAG = None

def _run_in_thread(coro):
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
    t = threading.Thread(target=runner)
    t.start()
    t.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")

def init(working_dir: str = "./lightrag_data"):
    global _rag
    os.makedirs(working_dir, exist_ok=True)

    async def _embed(texts: list[str]) -> np.ndarray:
        import google.generativeai as genai
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(result["embedding"])
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

def insert(text: str):
    _run_in_thread(_rag.ainsert(text))

def query(question: str, mode: str = "hybrid") -> str:
    return _run_in_thread(_rag.aquery(question, param=QueryParam(mode=mode)))
