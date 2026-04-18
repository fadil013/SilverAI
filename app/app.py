import os
import re
import tempfile
import gradio as gr
from dotenv import load_dotenv
import pdf_parser
import rag_setup
import supabase_store
import handbook_gen
import llm as llm_module

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

llm_module.init(GEMINI_KEY)
rag_setup.init()
supabase_store.init(SUPABASE_URL, SUPABASE_KEY)

_HANDBOOK_TRIGGERS = re.compile(
    r"\b(create|write|generate|make|produce|build)\b.{0,40}\bhandbook\b",
    re.IGNORECASE,
)

# ── PDF Upload ────────────────────────────────────────────────────────────────

def upload_pdf(files):
    if not files:
        return "No files uploaded."
    msgs = []
    for f in files:
        name = os.path.basename(f.name)
        try:
            text = pdf_parser.extract_text(f.name)
            chunks = pdf_parser.chunk_text(text)
            rag_setup.insert(text)
            supabase_store.store_chunks(chunks, source=name)
            msgs.append(f"✅ {name} — {len(chunks)} chunks indexed")
        except Exception as e:
            msgs.append(f"❌ {name} — {e}")
    return "\n".join(msgs)


# ── Chat (with handbook detection) ───────────────────────────────────────────

def chat(message: str, history: list) -> tuple[str, list]:
    if _HANDBOOK_TRIGGERS.search(message):
        history.append((message, "⏳ Generating your 20,000-word handbook... this takes a few minutes."))
        yield "", history

        try:
            rag_context = rag_setup.query(message)
        except Exception:
            rag_context = ""

        sections_written = []

        def progress_cb(i, total, step):
            label = step[:80]
            sections_written.append(f"  ✍️ [{i+1}/{total}] {label}")
            status = "\n".join(sections_written[-5:])
            history[-1] = (message, f"⏳ Writing handbook...\n\n{status}")

        try:
            result = handbook_gen.generate_handbook(message, rag_context, progress_cb)
            word_count = len(result.split())
            header = f"# Handbook\n\n*{word_count:,} words generated*\n\n---\n\n"
            full = header + result
        except Exception as e:
            full = f"❌ Handbook generation failed: {e}"

        history[-1] = (message, full)
        yield "", history
        return

    # Normal RAG chat
    try:
        context = rag_setup.query(message)
        system = (
            "You are a knowledgeable assistant. Answer the user's question using the "
            "provided document context. If the context is not relevant, say so and answer "
            "from your general knowledge."
        )
        response = llm_module.chat(
            f"Document context:\n{context}\n\nUser question: {message}",
            system=system,
        )
    except Exception as e:
        response = f"Error: {e}"

    history.append((message, response))
    yield "", history


# ── Handbook tab (explicit, with progress bar) ────────────────────────────────

def generate_handbook_tab(topic: str, progress=gr.Progress()):
    if not topic.strip():
        return "Please enter a topic.", None

    try:
        rag_context = rag_setup.query(topic)
    except Exception:
        rag_context = ""

    def progress_cb(i, total, step):
        progress((i + 1) / total, desc=f"Section {i+1}/{total}: {step[:60]}")

    try:
        result = handbook_gen.generate_handbook(topic, rag_context, progress_cb)
    except Exception as e:
        return f"Generation failed: {e}", None

    word_count = len(result.split())
    header = f"# Handbook: {topic}\n\n*{word_count:,} words generated*\n\n---\n\n"
    full_text = header + result

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8",
        prefix="handbook_"
    )
    tmp.write(full_text)
    tmp.close()

    return full_text, tmp.name


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
.gr-button-primary { background: #2563eb !important; }
.gr-chatbot { font-size: 14px; }
"""

with gr.Blocks(title="SilverAI Handbook Generator", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown(
        "# SilverAI — AI Handbook Generator\n"
        "Upload PDFs · Chat with your documents · Generate 20,000+ word handbooks"
    )

    with gr.Tab("Upload Documents"):
        gr.Markdown(
            "Upload one or more PDF files. The content will be extracted and indexed "
            "into a LightRAG knowledge graph so you can chat with it and generate handbooks."
        )
        upload_input = gr.File(
            label="Upload PDF(s)",
            file_types=[".pdf"],
            file_count="multiple",
        )
        upload_btn = gr.Button("Process & Index", variant="primary")
        upload_output = gr.Textbox(label="Status", lines=6, interactive=False)
        upload_btn.click(upload_pdf, inputs=upload_input, outputs=upload_output)

    with gr.Tab("Chat"):
        gr.Markdown(
            "Ask questions about your documents, or say **'create a handbook on X'** "
            "to generate a 20,000-word handbook from the chat."
        )
        chatbot = gr.Chatbot(height=500, label="Document Q&A")
        with gr.Row():
            chat_input = gr.Textbox(
                placeholder="Ask a question or say 'create a handbook on...'",
                label="Message",
                scale=5,
                show_label=False,
            )
            chat_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.Button("Clear conversation", size="sm")

        chat_btn.click(chat, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot])
        chat_input.submit(chat, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, chat_input])

    with gr.Tab("Generate Handbook"):
        gr.Markdown(
            "Enter any topic. The app will build a structured 20,000+ word handbook using "
            "the **LongWriter** plan-then-write technique, grounded in your uploaded documents."
        )
        topic_input = gr.Textbox(
            label="Handbook topic",
            placeholder="e.g. 'Comprehensive handbook on Retrieval-Augmented Generation for AI practitioners'",
            lines=3,
        )
        gen_btn = gr.Button("Generate Handbook (20,000+ words)", variant="primary")
        handbook_output = gr.Markdown(label="Generated Handbook")
        download_btn = gr.DownloadButton("Download as Markdown", visible=False)

        def on_generate(topic, progress=gr.Progress()):
            text, filepath = generate_handbook_tab(topic, progress)
            return text, gr.update(value=filepath, visible=filepath is not None)

        gen_btn.click(
            on_generate,
            inputs=topic_input,
            outputs=[handbook_output, download_btn],
        )

if __name__ == "__main__":
    demo.launch(share=False)
