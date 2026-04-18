import os
import re
import tempfile
import gradio as gr
from fastapi.responses import HTMLResponse
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
            msgs.append(f"[OK] {name} — {len(chunks)} chunks indexed")
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
            full = f"Generation failed: {e}"
        history[-1] = (message, full)
        yield "", history
        return

    try:
        context = rag_setup.query(message)
        system = (
            "You are a knowledgeable assistant. Answer based on document context. "
            "If context is irrelevant, use general knowledge."
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
        mode="w", suffix=".md", delete=False, encoding="utf-8", prefix="handbook_"
    )
    tmp.write(full_text)
    tmp.close()
    return full_text, tmp.name


# ── Custom theme ──────────────────────────────────────────────────────────────
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#050b17",
    body_text_color="#cbd5e1",
    body_text_color_subdued="#475569",
    block_background_fill="#0b1528",
    block_border_color="#1e293b",
    block_label_background_fill="#0b1528",
    block_label_text_color="#475569",
    block_label_text_size="11px",
    block_title_text_color="#94a3b8",
    block_radius="10px",
    block_shadow="0 1px 3px rgba(0,0,0,0.5)",
    input_background_fill="#0d1525",
    input_border_color="#1e293b",
    input_border_color_focus="#3b82f6",
    input_shadow_focus="0 0 0 3px rgba(59,130,246,0.12)",
    button_primary_background_fill="#1d4ed8",
    button_primary_background_fill_hover="#2563eb",
    button_primary_text_color="#f8fafc",
    button_primary_border_color="#1d4ed8",
    button_secondary_background_fill="#0f172a",
    button_secondary_background_fill_hover="#1e293b",
    button_secondary_text_color="#94a3b8",
    button_secondary_border_color="#1e293b",
    button_large_radius="8px",
    button_small_radius="6px",
    button_large_padding="10px 20px",
    chatbot_text_size="14px",
    border_color_accent="#3b82f6",
    color_accent="#3b82f6",
    color_accent_soft="#1d4ed8",
    slider_color="#3b82f6",
    table_even_background_fill="#0b1528",
    table_odd_background_fill="#080f1e",
)

# ── Extra CSS (only what the theme system can't do) ───────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

footer { display: none !important; }
.tabs { background: transparent !important; }

/* Header accent bar — single line above title */
#header-block > .gap > div:first-child > .prose > p:first-child::before {
    content: '';
    display: block;
    width: 36px;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #38bdf8);
    border-radius: 2px;
    margin-bottom: 20px;
}

#header-block h1 {
    font-size: 26px !important;
    font-weight: 700 !important;
    letter-spacing: -0.4px !important;
    color: #f1f5f9 !important;
}

#header-block p { color: #475569 !important; font-size: 14px !important; }

/* Tab nav */
.tab-nav { border-bottom: 1px solid #1e293b !important; margin-bottom: 24px !important; }
.tab-nav button {
    color: #475569 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 10px 18px !important;
    background: transparent !important;
    transition: color 0.15s !important;
}
.tab-nav button:hover { color: #94a3b8 !important; }
.tab-nav button.selected {
    color: #f1f5f9 !important;
    border-bottom-color: #3b82f6 !important;
    background: transparent !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #050b17; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }

/* Primary button glow */
button.primary:hover {
    box-shadow: 0 0 20px rgba(59,130,246,0.2) !important;
}

/* Chat bubbles */
.message { font-size: 14px !important; line-height: 1.65 !important; }
.user .message { color: #e2e8f0 !important; }
.bot .message, .assistant .message { color: #cbd5e1 !important; }

/* Upload zone */
.upload-container { min-height: 140px !important; cursor: pointer !important; }
"""

with gr.Blocks(title="SilverAI by LunarTech") as demo:

    with gr.Column(elem_id="header-block"):
        gr.Markdown(
            "# SilverAI — Handbook Generator\n"
            "Upload PDFs  ·  Ask questions  ·  Generate 20,000-word handbooks"
        )

    with gr.Tab("Upload Documents"):
        gr.Markdown("Drop PDF files below. They'll be extracted and indexed into a LightRAG knowledge graph for semantic search and handbook generation.")
        upload_input = gr.File(label="PDF Files", file_types=[".pdf"], file_count="multiple")
        upload_btn = gr.Button("Process & Index", variant="primary")
        upload_output = gr.Textbox(label="Status", lines=5, interactive=False, placeholder="Upload results appear here...")
        upload_btn.click(upload_pdf, inputs=upload_input, outputs=upload_output)

    with gr.Tab("Chat"):
        gr.Markdown("Ask questions about uploaded documents. Type **create a handbook on [topic]** to generate a full handbook from chat.")
        chatbot = gr.Chatbot(height=460, label="", show_label=False)
        with gr.Row():
            chat_input = gr.Textbox(placeholder="Ask a question or say 'create a handbook on...'", show_label=False, scale=5, container=False)
            chat_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)
        clear_btn = gr.Button("Clear", size="sm", variant="secondary")
        chat_btn.click(chat, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot])
        chat_input.submit(chat, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, chat_input])

    with gr.Tab("Generate Handbook"):
        gr.Markdown("Enter any topic. The LongWriter pipeline will produce a 20,000+ word structured handbook grounded in your documents.")
        topic_input = gr.Textbox(label="Handbook Topic", placeholder="e.g. Comprehensive handbook on Retrieval-Augmented Generation for AI practitioners", lines=3)
        gen_btn = gr.Button("Generate Handbook  —  20,000+ words", variant="primary")
        download_btn = gr.DownloadButton("Download Markdown", visible=False)
        handbook_output = gr.Markdown(label="")

        def on_generate(topic, progress=gr.Progress()):
            text, filepath = generate_handbook_tab(topic, progress)
            return text, gr.update(value=filepath, visible=filepath is not None)

        gen_btn.click(on_generate, inputs=topic_input, outputs=[handbook_output, download_btn])

if __name__ == "__main__":
    demo.launch(share=False, theme=theme, css=CSS)
