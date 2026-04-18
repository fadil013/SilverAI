# SilverAI — AI Handbook Generator

An AI-powered application that ingests PDF documents, enables document Q&A via RAG, and generates structured **20,000+ word handbooks** using the LongWriter technique.

---

## What It Does

```
PDF Upload → pdfplumber (text extraction)
           → LightRAG  (knowledge graph indexing)
           → Supabase  (optional vector persistence)

Chat       → LightRAG  (hybrid retrieval)
           → Gemini 2.5 Flash (contextual answers)

Handbook   → LightRAG  (context retrieval)
           → LongWriter plan-then-write pipeline
               1. LLM generates a full section-level outline
               2. Each section written sequentially with accumulated context
           → 20,000+ word Markdown output + download
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Gradio |
| LLM | Gemini 2.5 Flash (free tier) |
| Embeddings | Google `gemini-embedding-001` |
| RAG | LightRAG (knowledge graph) |
| Storage | Supabase pgvector (optional) |
| PDF Parsing | pdfplumber |

---

## Setup

### 1. Install dependencies

```bash
cd app
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp app/.env.example app/.env
```

Edit `app/.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here

# Optional — app runs without these
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key_here
```

Get a free Gemini API key at https://aistudio.google.com/app/apikey

### 3. Run

```bash
cd app
python app.py
```

Open http://localhost:7860

---

## How to Use

1. **Upload Documents tab** — Upload one or more PDFs, click "Process & Index". The content is extracted and stored in a LightRAG knowledge graph.

2. **Chat with Documents tab** — Ask questions about your uploaded content. Answers are retrieved from the knowledge graph and answered by Gemini.

3. **Generate Handbook tab** — Enter a topic such as:
   > *"Write a comprehensive handbook on Retrieval-Augmented Generation for practitioners"*

   Click Generate. The app uses the LongWriter plan-then-write pipeline to produce a 20,000+ word structured document. When done, a Download button appears to save it as Markdown.

---

## LongWriter Implementation

The handbook generator in `app/handbook_gen.py` implements the LongWriter technique from the paper *"Unleashing 10,000+ Word Generation from Long Context LLMs"*:

1. **Plan** — The LLM is prompted to produce a paragraph-by-paragraph outline with target word counts, ensuring the total reaches 20,000+ words.
2. **Write** — Each section is written iteratively. The model receives the full plan and the last 3,000 characters of already-written text as context, so each section flows naturally from the previous.

This overcomes the standard LLM output length limit by decomposing the document into coordinated, context-aware chunks.

---

## Supabase Setup (Optional)

If you want to enable vector persistence, run this SQL in your Supabase SQL editor:

```sql
create extension if not exists vector;

create table documents (
  id bigserial primary key,
  content text,
  source text,
  chunk_index int,
  embedding vector(768)
);

create or replace function match_documents(
  query_embedding vector(768),
  match_count int
)
returns table(content text, source text)
language sql stable
as $$
  select content, source
  from documents
  order by embedding <=> query_embedding
  limit match_count;
$$;
```

---

## Project Structure

```
app/
  app.py           — Gradio UI, orchestration
  llm.py           — Gemini LLM wrapper (sync + async)
  rag_setup.py     — LightRAG init, insert, query
  handbook_gen.py  — LongWriter plan-then-write pipeline
  pdf_parser.py    — PDF extraction + chunking
  supabase_store.py— Optional Supabase vector store
  requirements.txt
  .env.example

LongWriter-main/   — Reference implementation (provided)
Documentation/     — Research paper (provided)
```

---

## Write-up

**What I built:** A full-stack AI application combining LightRAG's knowledge graph RAG with the LongWriter iterative generation technique to produce 20,000+ word handbooks from uploaded PDF content.

**Approach:** I used Gemini 2.5 Flash as the free LLM backbone for both RAG-grounded Q&A and long-form generation. LightRAG handles entity-level knowledge graph construction from PDFs, providing richer retrieval than simple vector search. The LongWriter pipeline first generates a structured plan, then writes each section with the full plan and preceding text as context — this is what enables coherent 20k+ word output without repetition.

**Challenges:** Gradio runs its own asyncio event loop, which conflicts with `asyncio.get_event_loop().run_until_complete()` used by LightRAG. Solved this by running LightRAG async calls in isolated threads with their own event loops via `nest_asyncio`.
