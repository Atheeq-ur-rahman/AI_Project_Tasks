"""FAQ chatbot based on TF-IDF and cosine similarity."""

import json
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


FAQ_PATH = Path(__file__).with_name("faqs.json")
MINIMUM_CONFIDENCE = 0.16


@st.cache_resource
def load_matcher():
    """Load FAQ data and fit the text vectorizer once per server process."""
    with FAQ_PATH.open(encoding="utf-8") as file:
        faqs = json.load(file)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    question_matrix = vectorizer.fit_transform(item["question"] for item in faqs)
    return faqs, vectorizer, question_matrix


def find_answer(message: str) -> tuple[str, float]:
    """Return the answer that best matches a user question and its score."""
    faqs, vectorizer, question_matrix = load_matcher()
    message_vector = vectorizer.transform([message])
    scores = cosine_similarity(message_vector, question_matrix).ravel()
    best_index = scores.argmax()
    return faqs[best_index]["answer"], float(scores[best_index])


st.set_page_config(page_title="FAQ Chatbot", page_icon="")
st.title("Learning Platform FAQ Chatbot")
st.caption("Ask a question about accounts, courses, payments, or certificates.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Type your question")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    answer, confidence = find_answer(prompt)
    if confidence < MINIMUM_CONFIDENCE:
        answer = (
            "I could not find a close answer for that question. "
            "Please try using different wording or contact support."
        )

    with st.chat_message("assistant"):
        st.write(answer)
        st.caption(f"Match confidence: {confidence:.0%}")
    st.session_state.messages.append({"role": "assistant", "content": answer})
