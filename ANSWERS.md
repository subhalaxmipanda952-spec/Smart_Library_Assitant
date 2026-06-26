# Written Answers — Q1 (a)–(d)

## (a) Mapping each capability to AI/ML/NLP/RAG/CV

| Capability | Field | Justification |
|---|---|---|
| (i) Reads scanned book-cover images to identify title/author | **CV (Computer Vision)** | Identifying text/layout on an image requires image preprocessing and Optical Character Recognition — a core computer-vision task. |
| (ii) Answers questions using the library's own documents/catalogue, not generic knowledge | **RAG (Retrieval-Augmented Generation)**, built on **NLP** | The system must *retrieve* relevant passages from the library's private knowledge base and *generate* a grounded answer from them, rather than relying on a model's generic pretrained knowledge. NLP techniques (embeddings/TF-IDF, text similarity, language generation) underpin both the retriever and generator. |
| (iii) Learns from borrowing history to predict high-demand books | **ML (Machine Learning) — Supervised Learning** | Past borrowing records provide labeled historical outcomes (how many times a book was borrowed), so a regression/classification model can learn patterns and predict future demand. |
| (iv) Improves as new documents are added | **AI / ML (continual / incremental learning)** | The system needs to incorporate new catalogue entries and borrowing data over time without full retraining from scratch — i.e., continual learning / incremental indexing, which is a general AI-systems capability spanning the ML and RAG components above. |

## (b) How CV, NLP, and RAG interact to answer a book-cover query

1. **CV stage:** The user (or library staff) scans/uploads a book cover image. The CV component preprocesses the image (grayscale, contrast normalization) and runs OCR to extract raw text — e.g., the title, author name, or publisher printed on the cover.
2. **NLP stage:** The extracted raw OCR text is noisy, so an NLP step cleans and normalizes it (lower-casing, removing OCR artifacts) and may use fuzzy/semantic text matching to interpret what was actually printed on the cover, since OCR output rarely matches catalogue text exactly.
3. **RAG stage:** The cleaned text becomes a query into the RAG pipeline:
   - The **retriever** searches the library's catalogue (the **knowledge base**) using vector/TF-IDF similarity to find the best-matching record(s) for that title/author.
   - The **knowledge base** holds structured catalogue data (title, author, shelf location, availability, summary) — this is the library's own private data, never generic web knowledge.
   - The **generator** takes the retrieved record(s) and composes a natural-language answer for the user, e.g., "This is *Database System Concepts* by Silberschatz, located on shelf CS-204, 2 of 4 copies available."

So the pipeline is: **image → CV/OCR → NLP text cleaning → RAG retrieval over catalogue → RAG generation of final answer.** Each stage's output is the next stage's input, and the final answer is always grounded in the retrieved catalogue record(s) rather than the language model's own background knowledge.

## (c) Best ML paradigm for demand prediction + useful features

**Best paradigm: Supervised Learning (regression).**
Justification: the library has historical borrowing records with a known, measurable outcome (how many times each book was borrowed in a given period). This labeled history allows a regression model (e.g., Random Forest Regressor, Gradient Boosting, or even simple Linear Regression) to learn the relationship between a book's features and its future borrow count. Unsupervised learning isn't appropriate here because there's no predefined "demand" label to discover, and reinforcement learning isn't appropriate because there's no sequential decision/reward loop — it's a straightforward prediction-from-history problem.

**Useful data features:**
- Number of times borrowed in the last 30 / 90 / 180 days (recency-weighted demand)
- Total copies owned and currently available (supply context)
- Average loan duration (how long readers keep the book — proxy for engagement)
- Subject/section (e.g., "Computer Science" books may have different demand cycles than "Fiction")
- Time of year / semester (academic calendar effects, e.g., textbooks spike before exams)
- Number of holds/reservations currently placed on the book
- Publication/edition recency (newer editions may be more in demand)
- New-arrival flag (newly catalogued books may need cold-start handling)

## (d) One real-world challenge and a solution

**Challenge: Cold start for newly added books.**
A brand-new book has no borrowing history, so the ML demand-prediction model has no signal to predict its future popularity, and the RAG knowledge base may not yet have good summary/metadata text for retrieval either.

**Solution:** Use *content-based* similarity as a stand-in until real usage data accumulates — e.g., find similar existing books (same subject/section/author/popular keywords) and use their historical demand as an initial estimate, while also auto-generating a catalogue summary from the book's existing metadata (title, abstract, table of contents) so the RAG retriever can already surface it. As real borrowing data comes in, the system gradually shifts weight from this content-based estimate to the learned, history-based prediction (a hybrid cold-start approach), satisfying capability (iv) "improves as new documents are added."
