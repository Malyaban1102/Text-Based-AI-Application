"""
Architecture of a Text-Based AI Application
===========================================

Pipeline (each stage is its own section / function below):

    1. User Interface        -> Streamlit web app
    2. User Input            -> text, question, paragraph or instruction
    3. Input Processing      -> validation, cleaning, text selection
    4. Prompt Engineering    -> task-specific instruction
    5. LLM Processing        -> API + LLM model
    6. Output Processing     -> format and validate the generated response
    7. Display Result        -> show the response to the user

Setup:
    pip install streamlit groq python-dotenv
    Create a file named .env next to this script containing:
        GROQ_API_KEY=your-key-here
        GROQ_MODEL=llama-3.3-70b-versatile      (optional)
Run:
    streamlit run text_ai_app.py
"""

import os
import re

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
load_dotenv()  # reads GROQ_API_KEY / GROQ_MODEL from the .env file

API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_INPUT_CHARS = 8000
MIN_INPUT_CHARS = 3
MAX_OUTPUT_TOKENS = 1000

TASKS = {
    "Answer a question": {
        "system": "You are a knowledgeable assistant. Answer accurately and concisely. "
                  "If you are unsure, say so instead of guessing.",
        "template": "Question:\n{text}\n\nProvide a clear, well-structured answer.",
    },
    "Summarize text": {
        "system": "You are an expert summarizer who preserves the key facts and meaning.",
        "template": "Summarize the following text in 3-5 bullet points:\n\n{text}",
    },
    "Explain simply": {
        "system": "You are a patient teacher who explains concepts to beginners.",
        "template": "Explain the following in simple language with one short example:\n\n{text}",
    },
    "Rewrite / improve": {
        "system": "You are a professional editor. Keep the original meaning intact.",
        "template": "Rewrite the following text to be clearer and more polished:\n\n{text}",
    },
    "Follow my instruction": {
        "system": "You are a helpful assistant. Follow the user's instruction carefully.",
        "template": "{text}",
    },
}


# ----------------------------------------------------------------------------
# 3. INPUT PROCESSING  (validation, cleaning, text selection)
# ----------------------------------------------------------------------------
def clean_text(raw: str) -> str:
    """Remove control characters and normalise whitespace."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)  # control chars
    text = re.sub(r"[ \t]+", " ", text)                            # repeated spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)                         # excess blank lines
    return text.strip()


def validate_text(text: str) -> tuple[bool, str]:
    """Return (is_valid, message)."""
    if not text:
        return False, "Please enter some text, a question or an instruction."
    if len(text) < MIN_INPUT_CHARS:
        return False, f"Input is too short (minimum {MIN_INPUT_CHARS} characters)."
    if len(text) > MAX_INPUT_CHARS:
        return False, f"Input is too long ({len(text)} chars). Maximum is {MAX_INPUT_CHARS}."
    return True, ""


def select_text(text: str, limit: int = MAX_INPUT_CHARS) -> str:
    """Text selection: keep only the portion that fits the model budget."""
    return text[:limit]


def process_input(raw: str) -> tuple[bool, str, str]:
    """Run the full input-processing stage. Returns (ok, processed_text, error)."""
    cleaned = clean_text(raw)
    ok, msg = validate_text(cleaned)
    if not ok:
        return False, "", msg
    return True, select_text(cleaned), ""


# ----------------------------------------------------------------------------
# 4. PROMPT ENGINEERING  (task-specific instruction)
# ----------------------------------------------------------------------------
def build_prompt(task: str, text: str, tone: str, language: str) -> tuple[str, str]:
    """Create the system prompt and user prompt for the chosen task."""
    cfg = TASKS[task]
    system_prompt = (
        f"{cfg['system']} "
        f"Use a {tone.lower()} tone. Respond in {language}. "
        "Do not invent facts."
    )
    user_prompt = cfg["template"].format(text=text)
    return system_prompt, user_prompt


# ----------------------------------------------------------------------------
# 5. LLM PROCESSING  (API + LLM model)
# ----------------------------------------------------------------------------
def call_llm(system_prompt: str, user_prompt: str, api_key: str, model: str,
             temperature: float) -> str:
    """Send the prompt to the LLM API and return the raw text response."""
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


# ----------------------------------------------------------------------------
# 6. OUTPUT PROCESSING  (format and validate the generated response)
# ----------------------------------------------------------------------------
def process_output(raw_output: str) -> tuple[bool, str]:
    """Validate and format the model response. Returns (ok, formatted_text)."""
    if not raw_output or not raw_output.strip():
        return False, "The model returned an empty response. Please try again."

    formatted = raw_output.strip()
    formatted = re.sub(r"\n{3,}", "\n\n", formatted)  # tidy spacing
    return True, formatted


# ----------------------------------------------------------------------------
# 1. USER INTERFACE + 2. USER INPUT + 7. DISPLAY RESULT
# ----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Text-Based AI App", page_icon="🤖", layout="centered")
    st.title("🤖 Text-Based AI Application")
    st.caption("UI → Input → Processing → Prompt → LLM → Output Processing → Display")

    # ---- Sidebar settings ----
    with st.sidebar:
        st.header("Settings")
        temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.1)
        tone = st.selectbox("Tone", ["Neutral", "Friendly", "Formal", "Concise"])
        language = st.selectbox("Response language", ["English", "Bengali", "Hindi"])

    # ---- 2. User input ----
    task = st.selectbox("What do you want to do?", list(TASKS.keys()))
    user_text = st.text_area(
        "Enter your text, question, paragraph or instruction",
        height=200,
        placeholder="Type or paste here...",
    )
    st.caption(f"{len(user_text)} / {MAX_INPUT_CHARS} characters")

    if st.button("Generate", type="primary"):
        # ---- 3. Input processing ----
        ok, processed_text, error = process_input(user_text)
        if not ok:
            st.warning(error)
            return
        if not API_KEY:
            st.error("GROQ_API_KEY not found. Add it to your .env file and restart the app.")
            return

        # ---- 4. Prompt engineering ----
        system_prompt, user_prompt = build_prompt(task, processed_text, tone, language)

        # ---- 5. LLM processing ----
        try:
            with st.spinner("Thinking..."):
                raw_output = call_llm(system_prompt, user_prompt, API_KEY, MODEL, temperature)
        except Exception as exc:  # network / auth / rate-limit errors
            st.error(f"LLM request failed: {exc}")
            return

        # ---- 6. Output processing ----
        ok, final_output = process_output(raw_output)
        if not ok:
            st.error(final_output)
            return

        # ---- 7. Display result ----
        st.subheader("Response")
        st.markdown(final_output)
        st.download_button("Download response", final_output, file_name="response.txt")

        with st.expander("Show the prompt that was sent"):
            st.code(f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}")


if __name__ == "__main__":
    main()