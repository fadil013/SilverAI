import os
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

uploaded_docs = []

def upload_pdf(files):
    global uploaded_docs
    if not files:
        return "No files uploaded."
    msgs = []
    for f in files:
        text = pdf_parser.extract_text(f.name)
        chunks = pdf_parser.chunk_text(text)
        try:
            rag_setup.insert(text)
        except Exception as e:
            msgs.append(f"⚠️ RAG indexing failed for {os.path.basename(f.name)}: {e}")
            continue
        try:
            supabase_store.store_chunks(chunks, source=os.path.basename(f.name))
        except Exception:
            pass
        uploaded_docs.append(os.path.basename(f.name))
        msgs.append(f"✅ {os.path.basename(f.name)} — {len(chunks)} chunks indexed")
    return "\n".join(msgs)

def chat(message, history):
    try:
        context = rag_setup.query(message)
        response = llm_module.chat(
            f"Context from documents:\n{context}\n\nUser question: {message}",
            system="You are a helpful AI assistant. Answer based on the provided document context."
        )
    except Exception as e:
        response = f"Error: {str(e)}"
    history.append((message, response))
    return "", history

def generate_handbook(topic, progress=gr.Progress()):
    if not topic.strip():
        return "Please enter a topic or instruction for the handbook.", None

    try:
        rag_context = rag_setup.query(topic)
    except Exception:
        rag_context = ""

    def progress_cb(i, total, step):
        progress((i + 1) / total, desc=f"Writing section {i+1}/{total}: {step[:50]}")

    try:
        result = handbook_gen.generate_handbook(topic, rag_context, progress_cb)
    except Exception as e:
        return f"Generation failed: {str(e)}", None

    word_count = len(result.split())
    header = f"# Handbook: {topic}\n\n*Generated {word_count:,} words*\n\n---\n\n"
    full_text = header + result

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8",
        prefix=f"handbook_{topic[:20].replace(' ', '_')}_"
    )
    tmp.write(full_text)
    tmp.close()

    return full_text, tmp.name

with gr.Blocks(title="SilverAI Handbook Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SilverAI — AI Handbook Generator")
    gr.Markdown("Upload PDFs, ask questions, and generate comprehensive 20,000+ word handbooks.")

    with gr.Tab("Upload Documents"):
        upload_input = gr.File(label="Upload PDF(s)", file_types=[".pdf"], file_count="multiple")
        upload_btn = gr.Button("Process & Index", variant="primary")
        upload_output = gr.Textbox(label="Status", lines=5)
        upload_btn.click(upload_pdf, inputs=upload_input, outputs=upload_output)

    with gr.Tab("Chat with Documents"):
        chatbot = gr.Chatbot(height=450, label="Document Q&A")
        chat_input = gr.Textbox(placeholder="Ask anything about your documents...", label="Your question")
        chat_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear")
        chat_btn.click(chat, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot])
        chat_input.submit(chat, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot])
        clear_btn.click(lambda: [], outputs=chatbot)

    with gr.Tab("Generate Handbook"):
        topic_input = gr.Textbox(
            label="Handbook topic or instruction",
            placeholder="e.g. 'Write a comprehensive handbook on Machine Learning for beginners'",
            lines=3
        )
        gen_btn = gr.Button("Generate Handbook (20,000+ words)", variant="primary")
        handbook_output = gr.Markdown(label="Generated Handbook")
        download_btn = gr.DownloadButton("Download as Markdown", visible=False)

        def on_generate(topic, progress=gr.Progress()):
            text, filepath = generate_handbook(topic, progress)
            visible = filepath is not None
            return text, gr.update(value=filepath, visible=visible)

        gen_btn.click(
            on_generate,
            inputs=topic_input,
            outputs=[handbook_output, download_btn]
        )

if __name__ == "__main__":
    demo.launch(share=False)
