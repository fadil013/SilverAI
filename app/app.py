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
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

llm_module.init(gemini_key=GEMINI_KEY, openrouter_key=OPENROUTER_KEY)
rag_setup.init()
supabase_store.init(SUPABASE_URL, SUPABASE_KEY)

_HANDBOOK_TRIGGERS = re.compile(
    r"\b(create|write|generate|make|produce|build)\b.{0,40}\bhandbook\b",
    re.IGNORECASE,
)

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
            msgs.append(f"[OK] {name} — {len(chunks)} chunks indexed into knowledge graph")
        except Exception as e:
            msgs.append(f"[ERROR] {name} — {e}")
    return "\n".join(msgs)


def chat(message: str, history: list):
    if _HANDBOOK_TRIGGERS.search(message):
        history.append((message, "Generating your 20,000-word handbook — this takes a few minutes..."))
        yield "", history

        try:
            rag_context = rag_setup.query(message)
        except Exception:
            rag_context = ""

        sections_written = []

        def progress_cb(i, total, step):
            sections_written.append(f"[{i+1}/{total}] {step[:80]}")
            status = "\n".join(sections_written[-5:])
            history[-1] = (message, f"Writing handbook...\n\n{status}")

        try:
            result = handbook_gen.generate_handbook(message, rag_context, progress_cb)
            word_count = len(result.split())
            header = f"# Handbook\n\n*{word_count:,} words generated*\n\n---\n\n"
            full = header + result
        except Exception as e:
            full = f"Handbook generation failed: {e}"

        history[-1] = (message, full)
        yield "", history
        return

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


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base reset ─────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #060c18 !important;
    font-family: 'Inter', sans-serif !important;
    color: #e2e8f0 !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 0 24px 60px !important;
}

/* ── Header ─────────────────────────────────────────── */
#header-block {
    padding: 48px 0 32px !important;
    border-bottom: 1px solid rgba(148,163,184,0.08);
    margin-bottom: 32px;
}

#header-block h1 {
    font-size: 28px !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
    color: #f8fafc !important;
    margin: 0 0 8px !important;
    line-height: 1.2 !important;
}

#header-block p, #header-block .prose p {
    font-size: 14px !important;
    color: #64748b !important;
    margin: 0 !important;
    font-weight: 400 !important;
}

/* Accent bar on header */
#header-block::before {
    content: '';
    display: block;
    width: 40px;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #0ea5e9);
    border-radius: 2px;
    margin-bottom: 20px;
}

/* ── Tabs ────────────────────────────────────────────── */
.tabs { background: transparent !important; border: none !important; }

.tab-nav {
    background: transparent !important;
    border-bottom: 1px solid rgba(148,163,184,0.1) !important;
    padding: 0 !important;
    gap: 0 !important;
    margin-bottom: 28px !important;
}

.tab-nav button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #475569 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 12px 20px !important;
    margin-bottom: -1px !important;
    border-radius: 0 !important;
    transition: color 0.2s, border-color 0.2s !important;
    letter-spacing: 0.01em !important;
}

.tab-nav button:hover {
    color: #94a3b8 !important;
}

.tab-nav button.selected {
    color: #f8fafc !important;
    border-bottom-color: #3b82f6 !important;
    background: transparent !important;
}

/* ── Tab description text ────────────────────────────── */
.tab-desc p, .tab-desc .prose p {
    font-size: 13px !important;
    color: #475569 !important;
    margin-bottom: 20px !important;
    line-height: 1.6 !important;
}

/* ── Labels ──────────────────────────────────────────── */
label span, .gr-form label span {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #64748b !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── Inputs & Textareas ──────────────────────────────── */
textarea, input[type="text"], .gr-textbox textarea {
    background: #0d1525 !important;
    border: 1px solid rgba(148,163,184,0.12) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 12px 14px !important;
    transition: border-color 0.2s !important;
    resize: none !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: rgba(59,130,246,0.4) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.08) !important;
}

textarea::placeholder, input::placeholder {
    color: #334155 !important;
}

/* ── File upload ─────────────────────────────────────── */
.gr-file, .upload-container, [data-testid="file-upload"] {
    background: #0d1525 !important;
    border: 1px dashed rgba(148,163,184,0.15) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s !important;
}

.gr-file:hover, [data-testid="file-upload"]:hover {
    border-color: rgba(59,130,246,0.3) !important;
}

/* ── Buttons ─────────────────────────────────────────── */
button.primary, .gr-button-primary, button[variant="primary"] {
    background: #1d4ed8 !important;
    border: 1px solid rgba(59,130,246,0.3) !important;
    border-radius: 8px !important;
    color: #f8fafc !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    letter-spacing: 0.02em !important;
    transition: background 0.2s, transform 0.1s, box-shadow 0.2s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06) !important;
}

button.primary:hover, .gr-button-primary:hover {
    background: #2563eb !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.25) !important;
}

button.primary:active {
    transform: translateY(1px) !important;
}

button.secondary, .gr-button-secondary {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(148,163,184,0.12) !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}

button.secondary:hover {
    border-color: rgba(148,163,184,0.25) !important;
    color: #cbd5e1 !important;
}

/* ── Status / Output textbox ─────────────────────────── */
.output-textbox textarea, [data-testid="textbox"] textarea {
    background: #0a1020 !important;
    border: 1px solid rgba(148,163,184,0.08) !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    line-height: 1.7 !important;
}

/* ── Chatbot ─────────────────────────────────────────── */
.gr-chatbot, [data-testid="chatbot"] {
    background: #080e1c !important;
    border: 1px solid rgba(148,163,184,0.08) !important;
    border-radius: 10px !important;
}

.gr-chatbot .message.user, .message.user div {
    background: #1e3a5f !important;
    border: 1px solid rgba(59,130,246,0.15) !important;
    border-radius: 10px 10px 2px 10px !important;
    color: #e2e8f0 !important;
    font-size: 14px !important;
}

.gr-chatbot .message.bot, .message.bot div {
    background: #0f1e35 !important;
    border: 1px solid rgba(148,163,184,0.08) !important;
    border-radius: 10px 10px 10px 2px !important;
    color: #cbd5e1 !important;
    font-size: 14px !important;
}

/* ── Markdown output ─────────────────────────────────── */
.gr-markdown, .prose {
    color: #cbd5e1 !important;
    font-size: 14px !important;
    line-height: 1.75 !important;
}

.gr-markdown h1, .prose h1 {
    color: #f8fafc !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    border-bottom: 1px solid rgba(148,163,184,0.1) !important;
    padding-bottom: 10px !important;
    margin-bottom: 20px !important;
}

.gr-markdown h2, .prose h2 {
    color: #e2e8f0 !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    margin-top: 28px !important;
}

.gr-markdown h3, .prose h3 {
    color: #94a3b8 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

.gr-markdown code, .prose code {
    background: #0d1525 !important;
    border: 1px solid rgba(148,163,184,0.1) !important;
    border-radius: 4px !important;
    color: #7dd3fc !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    padding: 1px 6px !important;
}

/* ── Download button ─────────────────────────────────── */
.gr-download-button, [data-testid="download-btn"] {
    background: rgba(15,23,42,0.9) !important;
    border: 1px solid rgba(148,163,184,0.15) !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    margin-top: 16px !important;
}

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #060c18; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }

/* ── Footer ──────────────────────────────────────────── */
footer, .footer { display: none !important; }

/* ── Progress bar ────────────────────────────────────── */
.progress-bar { background: #1d4ed8 !important; }
.progress-bar-wrap { background: #0d1525 !important; border-radius: 4px !important; }
"""

with gr.Blocks(
    title="SilverAI by LunarTech",
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
) as demo:

    with gr.Column(elem_id="header-block"):
        gr.Markdown(
            "# SilverAI — Handbook Generator\n"
            "Upload any PDF · Ask questions · Generate 20,000-word structured handbooks"
        )

    with gr.Tab("Upload Documents"):
        gr.Markdown(
            "Drop one or more PDF files below. Text is extracted and indexed into a "
            "LightRAG knowledge graph — enabling semantic search across your documents.",
            elem_classes=["tab-desc"],
        )
        upload_input = gr.File(
            label="PDF Files",
            file_types=[".pdf"],
            file_count="multiple",
        )
        upload_btn = gr.Button("Process & Index", variant="primary")
        upload_output = gr.Textbox(
            label="Status",
            lines=6,
            interactive=False,
            elem_classes=["output-textbox"],
            placeholder="Upload results will appear here...",
        )
        upload_btn.click(upload_pdf, inputs=upload_input, outputs=upload_output)

    with gr.Tab("Chat"):
        gr.Markdown(
            "Ask questions about your uploaded documents. "
            "To generate a handbook, type: **create a handbook on [topic]**",
            elem_classes=["tab-desc"],
        )
        chatbot = gr.Chatbot(height=460, label="", show_label=False, bubble_full_width=False)
        with gr.Row():
            chat_input = gr.Textbox(
                placeholder="Ask a question or type 'create a handbook on...'",
                show_label=False,
                scale=5,
                container=False,
            )
            chat_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)
        clear_btn = gr.Button("Clear", size="sm", variant="secondary")

        chat_btn.click(chat, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot])
        chat_input.submit(chat, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, chat_input])

    with gr.Tab("Generate Handbook"):
        gr.Markdown(
            "Enter a topic and the LongWriter pipeline will produce a structured 20,000+ word handbook, "
            "grounded in your uploaded documents.",
            elem_classes=["tab-desc"],
        )
        topic_input = gr.Textbox(
            label="Handbook Topic",
            placeholder="e.g. Comprehensive handbook on Retrieval-Augmented Generation for AI practitioners",
            lines=3,
        )
        gen_btn = gr.Button("Generate Handbook  —  20,000+ words", variant="primary")
        download_btn = gr.DownloadButton("Download Markdown", visible=False)
        handbook_output = gr.Markdown(label="")

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
