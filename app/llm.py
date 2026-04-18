"""
LLM wrapper — Gemini (google-genai) with multi-model fallback, or OpenRouter.
"""
import os
import time
import asyncio
from google import genai
from google.genai import types
from openai import OpenAI

_gemini_client: genai.Client = None
_openrouter_client: OpenAI = None
_provider = None
_chat_model = None

EMBED_MODEL = "gemini-embedding-001"
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash",
]
OPENROUTER_MODEL = "deepseek/deepseek-r1-0528:free"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _try_chat(client, model, prompt, max_retries=2) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=100,
                    temperature=0.7,
                ),
            )
            return resp.text
        except Exception as e:
            msg = str(e)
            if ("503" in msg or "UNAVAILABLE" in msg) and attempt < max_retries - 1:
                time.sleep(3)
                continue
            raise
    raise RuntimeError("Max retries exceeded")


def init(gemini_key: str = "", openrouter_key: str = ""):
    global _gemini_client, _openrouter_client, _provider, _chat_model

    if gemini_key:
        client = genai.Client(api_key=gemini_key)
        # Find a working chat model
        working_model = None
        for model in GEMINI_MODELS:
            try:
                _try_chat(client, model, "Hi")
                working_model = model
                break
            except Exception as e:
                print(f"[LLM] {model} failed: {str(e)[:80]}")
                continue

        if working_model:
            _gemini_client = client
            _chat_model = working_model
            _provider = "gemini"
            print(f"[LLM] Using Gemini model: {working_model}")
            return

        print("[LLM] All Gemini models failed, trying OpenRouter...")

    if openrouter_key:
        _openrouter_client = OpenAI(api_key=openrouter_key, base_url=OPENROUTER_BASE)
        _provider = "openrouter"
        _chat_model = OPENROUTER_MODEL
        print(f"[LLM] Using OpenRouter: {OPENROUTER_MODEL}")
        return

    raise RuntimeError(
        "No working LLM. Set GEMINI_API_KEY or OPENROUTER_API_KEY in app/.env"
    )


def chat(prompt: str, system: str = "", max_tokens: int = 8192) -> str:
    if _provider == "gemini":
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=0.7,
            system_instruction=system or None,
        )
        for attempt in range(3):
            try:
                resp = _gemini_client.models.generate_content(
                    model=_chat_model,
                    contents=prompt,
                    config=config,
                )
                return resp.text
            except Exception as e:
                if "503" in str(e) and attempt < 2:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise

    # OpenRouter
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = _openrouter_client.chat.completions.create(
        model=_chat_model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return resp.choices[0].message.content


def embed(text: str) -> list[float]:
    if _provider == "gemini":
        result = _gemini_client.models.embed_content(
            model=EMBED_MODEL, contents=text
        )
        return result.embeddings[0].values

    # Simple deterministic fallback for OpenRouter (no embedding API)
    import hashlib, math
    h = hashlib.sha256(text.encode()).digest()
    vec = []
    for i in range(0, len(h) - 2, 3):
        val = (h[i] + h[i+1]*256 + h[i+2]*65536) / 16777215.0
        vec += [math.sin(val * math.pi * 2), math.cos(val * math.pi * 2), val * 2 - 1]
    while len(vec) < 768:
        vec.append(0.0)
    return vec[:768]


async def chat_async(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = None,
    **kwargs,
) -> str:
    parts = []
    if system_prompt:
        parts.append(f"System: {system_prompt}")
    for m in (history_messages or []):
        parts.append(f"{m['role']}: {m['content']}")
    parts.append(f"User: {prompt}")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: chat("\n".join(parts)))


async def embed_async(texts: list) -> list:
    loop = asyncio.get_event_loop()
    results = []
    for text in texts:
        emb = await loop.run_in_executor(None, lambda t=text: embed(t))
        results.append(emb)
    return results
