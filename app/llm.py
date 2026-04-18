import os
import google.generativeai as genai

_model = None

def init(api_key: str):
    global _model
    genai.configure(api_key=api_key)
    _model = genai.GenerativeModel("gemini-1.5-flash")

def chat(prompt: str, system: str = "", max_tokens: int = 8192) -> str:
    full = f"{system}\n\n{prompt}" if system else prompt
    resp = _model.generate_content(
        full,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.7,
        )
    )
    return resp.text

async def chat_async(prompt: str, system_prompt: str = None, history_messages: list = [], **kwargs) -> str:
    parts = []
    if system_prompt:
        parts.append(f"System: {system_prompt}")
    for m in history_messages:
        parts.append(f"{m['role']}: {m['content']}")
    parts.append(f"User: {prompt}")
    return chat("\n".join(parts))

async def embed_async(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document",
        )
        embeddings.append(result["embedding"])
    return embeddings
