"""A simple user interface for text translation."""

import streamlit as st
from deep_translator import GoogleTranslator


LANGUAGES = {
    "Auto detect": "auto",
    "English": "en",
    "Hindi": "hi",
    "Arabic": "ar",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Japanese": "ja",
    "Korean": "ko",
    "Tamil": "ta",
    "Telugu": "te",
}


def translate_text(text: str, source_name: str, target_name: str) -> str:
    """Send text to the translation service and return its response."""
    translator = GoogleTranslator(
        source=LANGUAGES[source_name], target=LANGUAGES[target_name]
    )
    return translator.translate(text)


st.set_page_config(page_title="Language Translation Tool", page_icon="")
st.title("Language Translation Tool")
st.caption("Translate a sentence or paragraph between common languages.")

input_text = st.text_area(
    "Text to translate", height=180, placeholder="Type or paste text here..."
)

left_column, right_column = st.columns(2)
with left_column:
    source_language = st.selectbox("Source language", list(LANGUAGES), index=0)
with right_column:
    target_options = [language for language in LANGUAGES if language != "Auto detect"]
    target_language = st.selectbox("Target language", target_options, index=0)

if st.button("Translate", type="primary"):
    if not input_text.strip():
        st.warning("Please enter some text before translating.")
    elif source_language == target_language:
        st.info("The source and target languages are the same.")
    else:
        try:
            with st.spinner("Translating..."):
                result = translate_text(input_text.strip(), source_language, target_language)
            st.subheader("Translation")
            st.text_area("Translated text", value=result, height=180)
        except Exception as error:
            st.error("The translation service could not complete the request.")
            st.caption(f"Details: {error}")
