"""
Smart Library Assistant — Streamlit front-end.

Run with:
    streamlit run app.py
"""
import json
from pathlib import Path

import streamlit as st
from PIL import Image

from cv_module import identify_book_cover, load_catalogue, TESSERACT_AVAILABLE
from rag_module import LibraryRAG
from ml_module import train_model, predict_demand_for_all, MODEL_PATH

st.set_page_config(page_title="Smart Library Assistant", page_icon="📚", layout="wide")

st.title("📚 Smart Library Assistant")
st.caption("A demo mapping CV / NLP / RAG / ML capabilities onto one application, "
           "built for the Section B project brief.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Book Cover Query (CV)",
    "💬 Library Q&A (RAG)",
    "📈 Demand Prediction (ML)",
    "📝 Written Answers (a–d)",
])

# ---------------------------------------------------------------------------
# TAB 1 — Computer Vision: identify a book from its cover image
# ---------------------------------------------------------------------------
with tab1:
    st.header("Identify a book from its cover")
    st.write(
        "Upload a (scanned) book-cover image. The app runs OCR (Computer "
        "Vision) to read the text on the cover, then matches it against the "
        "library's own catalogue to identify the title/author."
    )

    if not TESSERACT_AVAILABLE:
        st.warning(
            "`pytesseract` / the Tesseract OCR engine isn't available in this "
            "environment, so OCR will return empty text. See README.md for "
            "install instructions. You can still try the catalogue lookup "
            "below by typing text manually."
        )

    uploaded = st.file_uploader("Upload a book cover image", type=["jpg", "jpeg", "png"])
    manual_text = st.text_input("...or type/paste text from the cover manually (fallback)")

    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded cover", width=250)
        with st.spinner("Running OCR + catalogue matching..."):
            result = identify_book_cover(image)
        st.subheader("OCR text extracted")
        st.code(result["ocr_text"] or "(no text detected)")
        st.subheader("Best catalogue match")
        if result["match"]:
            b = result["match"]
            st.success(f"**{b['title']}** by {b['author']}  (confidence: {result['confidence']})")
            st.json(b)
        else:
            st.error("No confident match found in the catalogue.")

    elif manual_text:
        from cv_module import match_book_from_text
        result = match_book_from_text(manual_text)
        if result["match"]:
            b = result["match"]
            st.success(f"**{b['title']}** by {b['author']}  (confidence: {result['confidence']})")
            st.json(b)
        else:
            st.error("No confident match found in the catalogue.")

# ---------------------------------------------------------------------------
# TAB 2 — RAG: answer questions using the library's own catalogue
# ---------------------------------------------------------------------------
with tab2:
    st.header("Ask a question about the library's collection")
    st.write(
        "This uses Retrieval-Augmented Generation: a **retriever** (TF-IDF "
        "similarity) finds the most relevant entries in the library's own "
        "**knowledge base** (catalogue), and a **generator** composes an "
        "answer grounded only in those retrieved entries — not generic "
        "world knowledge."
    )

    @st.cache_resource
    def get_rag():
        return LibraryRAG()

    rag = get_rag()
    query = st.text_input("Your question", placeholder="e.g. Do you have any books on databases?")
    top_k = st.slider("Number of catalogue entries to retrieve", 1, 5, 3)

    if st.button("Ask") and query.strip():
        with st.spinner("Retrieving from catalogue and generating answer..."):
            result = rag.answer(query, top_k=top_k)
        st.markdown("### Answer")
        st.write(result["answer"])
        st.markdown("### Retrieved evidence (retriever output)")
        st.table(result["retrieved"])

    with st.expander("View raw catalogue (knowledge base)"):
        st.json(load_catalogue())

# ---------------------------------------------------------------------------
# TAB 3 — ML: predict demand for high-demand books
# ---------------------------------------------------------------------------
with tab3:
    st.header("Predict high-demand books")
    st.write(
        "Demand prediction is framed as a **supervised regression** problem: "
        "the model learns from historical borrowing patterns (labeled data) "
        "to predict each book's expected borrow count for the next period."
    )

    history_csv = Path(__file__).parent / "data" / "borrowing_history.csv"
    if not history_csv.exists():
        st.warning("Borrowing history not found yet. Click below to generate sample data.")
        if st.button("Generate sample borrowing history"):
            import subprocess, sys
            subprocess.run([sys.executable, str(Path(__file__).parent / "data" / "generate_borrowing_history.py")])
            st.success("Sample data generated. Please rerun this tab.")
    else:
        if st.button("Train / retrain demand model"):
            with st.spinner("Training RandomForestRegressor on borrowing history..."):
                metrics = train_model()
            st.success(f"Model trained. Mean Absolute Error on held-out test set: {metrics['mae']:.2f}")

        if MODEL_PATH.exists():
            with st.spinner("Generating predictions..."):
                preds = predict_demand_for_all()
            st.subheader("Predicted demand (next period) — sorted high to low")
            st.dataframe(preds, use_container_width=True)
            st.bar_chart(preds.set_index("title")["predicted_demand"])
        else:
            st.info("Train the model above to see predictions.")

# ---------------------------------------------------------------------------
# TAB 4 — Written exam answers, embedded for convenience
# ---------------------------------------------------------------------------
with tab4:
    st.header("Written answers to Q1 (a)–(d)")
    answers_path = Path(__file__).parent / "ANSWERS.md"
    if answers_path.exists():
        st.markdown(answers_path.read_text())
    else:
        st.info("ANSWERS.md not found.")
