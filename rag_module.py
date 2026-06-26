"""
RAG module (Section B Q1: capability ii, and part (b) of the question).

Implements the three RAG pieces explicitly so the architecture maps
clearly to the exam answer:
    1. Knowledge base : the library's own catalogue (data/catalogue.json)
    2. Retriever      : TF-IDF + cosine similarity over book summaries
    3. Generator      : a lightweight template-based answer composer
                         (kept dependency-free; swap in an LLM call here
                         for a production system)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CATALOGUE_PATH = Path(__file__).parent / "data" / "catalogue.json"


class LibraryRAG:
    """A minimal Retrieval-Augmented Generation pipeline over the catalogue."""

    def __init__(self, catalogue_path: Path = CATALOGUE_PATH):
        with open(catalogue_path) as f:
            self.catalogue: list[dict] = json.load(f)

        # --- Knowledge base text representation ---
        self.documents = [
            f"{b['title']} by {b['author']}. Section: {b['section']}. "
            f"Summary: {b['summary']} Shelf: {b['shelf']}. "
            f"Copies available: {b['copies_available']} of {b['copies_total']}."
            for b in self.catalogue
        ]

        # --- Retriever ---
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(self.documents)

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[dict, float]]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix).flatten()
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.catalogue[i], float(scores[i])) for i in ranked_idx if scores[i] > 0]

    def generate(self, query: str, retrieved: list[tuple[dict, float]]) -> str:
        """
        Template-based generator that grounds its answer ONLY in the
        retrieved catalogue entries (not generic/world knowledge), which
        is exactly requirement (iii) in the question.
        """
        if not retrieved:
            return ("I couldn't find anything in the library's catalogue that "
                    "matches your question. Try rephrasing, or ask the front "
                    "desk to add this resource.")

        top_book, top_score = retrieved[0]
        availability = (
            f"{top_book['copies_available']} of {top_book['copies_total']} copies "
            f"currently available" if top_book["copies_available"] > 0
            else "currently fully checked out"
        )

        answer = (
            f"Based on the library catalogue, **{top_book['title']}** by "
            f"{top_book['author']} (Section: {top_book['section']}, Shelf: "
            f"{top_book['shelf']}) looks most relevant to your question.\n\n"
            f"Summary: {top_book['summary']}\n\n"
            f"Availability: {availability}."
        )

        if len(retrieved) > 1:
            others = ", ".join(b["title"] for b, _ in retrieved[1:])
            answer += f"\n\nOther related titles in the catalogue: {others}."

        return answer

    def answer(self, query: str, top_k: int = 3) -> dict:
        retrieved = self.retrieve(query, top_k=top_k)
        response = self.generate(query, retrieved)
        return {
            "query": query,
            "retrieved": [
                {"title": b["title"], "author": b["author"], "score": round(s, 3)}
                for b, s in retrieved
            ],
            "answer": response,
        }
