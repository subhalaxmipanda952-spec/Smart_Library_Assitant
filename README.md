# 📚 Smart Library Assistant

A demo project for the "Smart Library Assistant" brief (Section B, Q1).
It maps each required capability onto a concrete AI technique and wires
them together into one Streamlit app:

| # | Capability | Technique used |
|---|---|---|
| i | Read scanned book-cover images to identify title/author | **CV** — OCR (`pytesseract`) + fuzzy text matching |
| ii | Answer questions from the library's own catalogue, not generic knowledge | **RAG** — TF-IDF retriever + template generator over a local knowledge base |
| iii | Learn from borrowing history to predict high-demand books | **ML** — supervised regression (`RandomForestRegressor`) |
| iv | Improve as new documents are added | Re-runnable indexing/retraining (see "Updating the data" below) |

Full written answers to parts (a)–(d) of the question are in
[`ANSWERS.md`](ANSWERS.md) and are also shown inside the app's
"Written Answers" tab.

---

## Project structure

```
smart_library_assistant/
├── app.py                          # Streamlit front-end (entry point)
├── cv_module.py                    # OCR + cover-to-catalogue matching
├── rag_module.py                   # Retriever + knowledge base + generator
├── ml_module.py                    # Feature engineering + demand-prediction model
├── ANSWERS.md                      # Written answers to Q1 (a)-(d)
├── requirements.txt
├── README.md
└── data/
    ├── catalogue.json              # Sample library catalogue (knowledge base)
    ├── generate_borrowing_history.py  # Creates synthetic borrowing_history.csv
    ├── borrowing_history.csv       # (generated) synthetic loan records
    └── demand_model.joblib         # (generated) trained ML model
```

---

## 1. Requirements

- Python 3.9+
- (Optional, for real OCR) the **Tesseract OCR engine** installed on your system,
  in addition to the `pytesseract` Python package. If Tesseract isn't installed,
  the app still runs — the OCR tab will just show an empty result, and you can
  use the manual text-entry fallback instead.

  - **Windows:** download the installer from
    https://github.com/UB-Mannheim/tesseract/wiki and add it to your PATH.
  - **macOS:** `brew install tesseract`
  - **Linux (Debian/Ubuntu):** `sudo apt-get install tesseract-ocr`

---

## 2. Setup

```bash
# 1. Clone / unzip the project, then move into the folder
cd smart_library_assistant

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 3. Generate sample data (one-time)

The app needs a synthetic borrowing history to train the demand-prediction
model. Generate it once:

```bash
python data/generate_borrowing_history.py
```

This creates `data/borrowing_history.csv` from `data/catalogue.json`.
(You can also click "Generate sample borrowing history" inside the app's
"Demand Prediction" tab — it runs the same script.)

---

## 4. Run the app

```bash
streamlit run app.py
```

Streamlit will print a local URL (typically **http://localhost:8501**) —
open it in your browser.

---

## 5. Using the app

- **🖼️ Book Cover Query (CV tab):** upload a book-cover image (jpg/png).
  The app OCRs the text and matches it against the catalogue. If Tesseract
  isn't installed, type the cover text into the manual fallback box instead.
- **💬 Library Q&A (RAG tab):** ask a natural-language question like
  *"Do you have anything on operating systems?"* — the app retrieves the
  most relevant catalogue entries and generates an answer grounded in them.
- **📈 Demand Prediction (ML tab):** click "Train / retrain demand model" to
  fit a `RandomForestRegressor` on the synthetic borrowing history, then view
  the ranked list/bar chart of predicted high-demand books.
- **📝 Written Answers tab:** shows the full exam-style answers to (a)–(d).

---

## 6. Updating the catalogue (capability iv: "improves as new documents are added")

1. Add a new entry (title, author, isbn, section, shelf, summary, copies) to
   `data/catalogue.json`.
2. Restart the app (or clear Streamlit's cache) — the RAG retriever rebuilds
   its TF-IDF index from the catalogue automatically on startup.
3. Optionally re-run `data/generate_borrowing_history.py` and retrain the
   demand model from the ML tab to incorporate the new book into demand
   predictions.

---

## 7. Notes on design choices

- The RAG "generator" is a lightweight, dependency-free template composer so
  the project runs with no external API keys. In a production system you
  would swap `rag_module.LibraryRAG.generate()` for a call to an LLM (e.g.
  the Claude API), still grounded in the same retrieved catalogue context.
- The demand-prediction labels are derived by holding out the most recent
  90 days of the synthetic history as the "future" period to predict —
  a simple, transparent way to simulate a supervised learning setup without
  needing real multi-year library data.
