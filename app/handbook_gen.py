import re
from llm import chat

PLAN_PROMPT = """You are a professional handbook author. Break the following writing instruction into detailed chapter-level sections for a comprehensive handbook.

Writing instruction:
{instruction}

Output format (one section per line, no extra text, no blank lines):
Section 1 - [Chapter Title]: [Detailed description of what to cover] - Word Count: [500-1000]
Section 2 - [Chapter Title]: [Detailed description of what to cover] - Word Count: [500-1000]
...

Rules:
- Generate at least 25 sections to ensure 20,000+ total words
- Each section should be a distinct topic with a clear chapter title
- Vary word counts between 500 and 1000 per section
- Total target: 25,000 words"""

WRITE_PROMPT = """You are an expert technical writer producing a professional handbook.

Handbook topic:
{instruction}

Full chapter plan:
{plan}

Previously written content (last portion):
{written}

Now write this chapter section:
{step}

Instructions:
- Write only this section's content (do NOT repeat prior text)
- Begin with a clear ## heading using the chapter title
- Use subheadings (###) where appropriate
- Write in a professional, educational tone
- Include practical examples and explanations
- Do not add a conclusion or "next chapter" note"""

def generate_plan(instruction: str) -> list[str]:
    raw = chat(PLAN_PROMPT.format(instruction=instruction), max_tokens=4096)
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip().startswith("Section")]
    return lines

def _build_toc(steps: list[str]) -> str:
    toc_lines = ["## Table of Contents\n"]
    for step in steps:
        m = re.search(r"Section \d+ - (.+?):", step)
        if m:
            title = m.group(1).strip()
            anchor = title.lower().replace(" ", "-").replace(",", "").replace("(", "").replace(")", "")
            toc_lines.append(f"- [{title}](#{anchor})")
    return "\n".join(toc_lines)

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
    if rag_context:
        full_instruction = f"{instruction}\n\nUse this source material as reference:\n{rag_context[:4000]}"

    steps = generate_plan(full_instruction)
    if not steps:
        return "Failed to generate plan. Please try again with a more specific topic."

    toc = _build_toc(steps)
    plan_text = "\n".join(steps)
    written = ""

    for i, step in enumerate(steps):
        if progress_cb:
            progress_cb(i, len(steps), step)
        section = write_section(full_instruction, plan_text, written, step)
        written += "\n\n" + section

    return toc + "\n\n---\n\n" + written.strip()
