# Language Translation Tool

A small Streamlit application that translates user-entered text between common languages. It uses the Google Translate endpoint supplied by the `deep-translator` package, so no API key is required for a basic demonstration.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The command opens the application in a browser. Enter text, choose the source and target languages, then select **Translate**.

## Notes

- The automatic source-language option lets the translation service detect the input language.
- Translation is performed by an online service, so an internet connection is required.
- For a production application, use an authenticated provider such as Google Cloud Translation or Microsoft Translator and keep credentials outside source control.
