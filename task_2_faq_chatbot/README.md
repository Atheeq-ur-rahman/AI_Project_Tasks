# FAQ Chatbot

This Streamlit chatbot answers questions about an online learning platform. It cleans the FAQ text with a lightweight tokenizer, represents questions with TF-IDF vectors, and returns the answer whose question has the highest cosine similarity to the user's message.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Try questions such as `How can I reset my password?` or `Can I get a certificate?`.

## Customising the knowledge base

The FAQ data lives in `faqs.json`. Add entries with `question` and `answer` fields, then restart the application. The model is fitted from that file when the app starts.

## How matching works

1. FAQ questions are lowercased and tokenized by the TF-IDF vectorizer.
2. The user message is transformed into the same vector space.
3. Cosine similarity ranks every FAQ question.
4. Low-confidence matches use a fallback response instead of guessing.
