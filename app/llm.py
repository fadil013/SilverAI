import asyncio
import google.generativeai as genai

_model = None

def init(api_key: str):
    global _model
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Add it to app/.env")
    genai.configure(api_key=api_key)
    _model = genai.GenerativeModel("gemini-1.5-flash")

def chat(prompt: str, system: str = "", max_tokens: int = 8192) -> str:
    full = f"{system}\n\n{prompt}" if system else prompt
    resp = _model.generate_content(
        full,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.7,
        ),
    )
    return resp.text

async def chat_async(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = None,
    **kwargs,
) -> str:
    """Async wrapper used by LightRAG's llm_model_func."""
    parts = []
    if system_prompt:
        parts.append(f"System: {system_prompt}")
    for m in (history_messages or []):
        parts.append(f"{m['role']}: {m['content']}")
    parts.append(f"User: {prompt}")
    full_prompt = "\n".join(parts)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: chat(full_prompt))

async def embed_async(texts: list) -> list:
    """Async embedding — iterates one-by-one (Google API is per-text)."""
    loop = asyncio.get_event_loop()
    results = []
    for text in texts:
        result = await loop.run_in_executor(
            None,
            lambda t=text: genai.embed_content(
                model="models/text-embedding-004",
                content=t,
                task_type="retrieval_document",
            ),
        )
        results.append(result["embedding"])
    return results
