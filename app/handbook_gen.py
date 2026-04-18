"""
LongWriter plan-then-write pipeline.
Prompts are adapted directly from LongWriter-main/agentwrite/prompts/
"""
import re
from llm import chat

PLAN_PROMPT = """I need you to help me break down the following long-form writing instruction into multiple subtasks. Each subtask will guide the writing of one paragraph in the essay, and should include the main points and word count requirements for that paragraph.

The writing instruction is as follows:

{instruction}

Please break it down in the following format, with each subtask taking up one line:

Paragraph 1 - Main Point: [Describe the main point of the paragraph, in detail] - Word Count: [Word count requirement, e.g., 400 words]

Paragraph 2 - Main Point: [Describe the main point of the paragraph, in detail] - Word Count: [word count requirement, e.g. 1000 words].

...

Make sure that each subtask is clear and specific, and that all subtasks cover the entire content of the writing instruction. Do not split the subtasks too finely; each subtask's paragraph should be no less than 400 words and no more than 1000 words. Target at least 25 paragraphs to reach 20,000+ total words. Do not output any other content."""

WRITE_PROMPT = """You are an excellent writing assistant. I will give you an original writing instruction and my planned writing steps. I will also provide you with the text I have already written. Please help me continue writing the next paragraph based on the writing instruction, writing steps, and the already written text.

Writing instruction:

{instruction}

Writing steps:

{plan}

Already written text:

{written}

Please integrate the original writing instruction, writing steps, and the already written text, and now continue writing {step}. If needed, you can add a small subtitle at the beginning. Remember to only output the paragraph you write, without repeating the already written text. As this is an ongoing work, omit open-ended conclusions or other rhetorical hooks."""


def generate_plan(instruction: str) -> list[str]:
    raw = chat(PLAN_PROMPT.format(instruction=instruction), max_tokens=4096)
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip().startswith("Paragraph")]
    return lines


def _extract_subtitle(text: str) -> str:
    """Pull the first subtitle/heading out of written text for TOC."""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
        if line and len(line) < 80 and not line.endswith("."):
            return line
    return ""


def write_section(instruction: str, plan: str, written: str, step: str) -> str:
    return chat(
        WRITE_PROMPT.format(
            instruction=instruction,
            plan=plan,
            written=written[-4000:],
            step=step,
        ),
        max_tokens=2048,
    )


def generate_handbook(instruction: str, rag_context: str = "", progress_cb=None) -> str:
    full_instruction = instruction
    if rag_context and rag_context.strip():
        full_instruction = (
            f"{instruction}\n\n"
            f"Use the following source material as factual reference and cite it where relevant:\n"
            f"{rag_context[:5000]}"
        )

    steps = generate_plan(full_instruction)
    if not steps:
        return "Failed to generate a writing plan. Please try a more specific topic."

    plan_text = "\n".join(steps)
    written = ""
    toc_entries = []

    for i, step in enumerate(steps):
        if progress_cb:
            progress_cb(i, len(steps), step)
        section = write_section(full_instruction, plan_text, written, step)
        subtitle = _extract_subtitle(section)
        if subtitle:
            toc_entries.append(subtitle)
        written += "\n\n" + section

    # Build table of contents
    toc_lines = ["## Table of Contents\n"]
    for entry in toc_entries:
        anchor = re.sub(r"[^\w\s-]", "", entry).strip().lower().replace(" ", "-")
        toc_lines.append(f"- [{entry}](#{anchor})")
    toc = "\n".join(toc_lines)

    return toc + "\n\n---\n\n" + written.strip()
