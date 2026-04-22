"""Streamlit app for the Amazon Books Retrieval System.

Provides four search modes over a 110,167-document corpus of Amazon book
reviews: BM25 keyword search, semantic embedding search, weighted hybrid
search, and retrieval-augmented generation (RAG) via the Groq API.
"""

# Silence transformers 5.x deprecation spam BEFORE any transformers import.
# sentence-transformers pulls in transformers, which on recent 5.x releases
# prints an "Accessing __path__ from .models.*.image_processing_*" line for
# every vision model at import time. These are harmless but flood the log.
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import sys
import pickle
from dotenv import load_dotenv

load_dotenv()

app_dir = Path(__file__).parent
project_root = app_dir.parent

sys.path.insert(0, str(project_root))

from src.bm25 import BM25Retriever
from src.semantic_retriever import SemanticRetriever
from src.rag_pipeline import RAGPipeline
from src.utils import load_data, load_corpus


class SimpleLLM:
    """Simple LLM for demo - always available."""

    def invoke(self, text: str) -> str:
        """Return a fixed placeholder response.

        Used when no GROQ_API_KEY is configured so the RAG tab still renders
        a sensible, non-crashing output and the retrieval portion can still
        be demonstrated end-to-end.
        """
        return (
            "Based on the provided reviews, this book appears to be well-received "
            "by customers who value its content, writing style, and overall quality."
        )


class GroqLLM:
    """Groq API wrapper with automatic model fallback.

    The Streamlit app uses this class when the user selects "Groq (Production)"
    and a valid API key is present. It attempts the preferred model first and
    cascades through AVAILABLE_MODELS if a model has been decommissioned or
    is otherwise unavailable, so the app stays functional across Groq's
    ongoing model lifecycle changes.
    """

    # Ordered fallback chain. LLaMA 3.3 70B is the model the LLM experiment
    # (notebooks/08_llm_comparison.ipynb and results/final_discussion.md)
    # selected as the preferred default for this task.
    AVAILABLE_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
        "mixtral-8x7b-32768",
    ]

    def __init__(self, api_key: str = None, model: str = None):
        """Initialize the Groq client and record the preferred model.

        Parameters
        ----------
        api_key : str, optional
            Groq API key. If omitted, falls back to the GROQ_API_KEY env var
            loaded from .env.
        model : str, optional
            Preferred Groq model ID. If omitted, defaults to the top of
            AVAILABLE_MODELS (llama-3.3-70b-versatile).
        """
        try:
            from groq import Groq
            self.api_key = api_key or os.getenv("GROQ_API_KEY")
            if not self.api_key:
                raise ValueError("GROQ_API_KEY not found in .env file")
            self.client = Groq(api_key=self.api_key)
            self.model = model or self.AVAILABLE_MODELS[0]
            self.available = True
        except ImportError:
            self.available = False

    def invoke(self, text: str) -> str:
        """Send a prompt to Groq and return the generated answer.

        Tries the preferred model first, then cascades through AVAILABLE_MODELS
        on "decommissioned" or "not supported" errors. Any other error is
        returned to the caller as a string for display in the UI.
        """
        if not self.available:
            return "Groq library not installed. Run: pip install groq"

        models_to_try = [self.model] + [m for m in self.AVAILABLE_MODELS if m != self.model]

        for model in models_to_try:
            try:
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": text}],
                    model=model,
                    max_tokens=500,
                    temperature=0.7,
                )
                return response.choices[0].message.content
            except Exception as e:
                error_msg = str(e)
                if "decommissioned" in error_msg or "not supported" in error_msg:
                    continue
                else:
                    return f"Error: {error_msg}"

        return "All available models failed. Try SimpleLLM mode or check your API key."


@st.cache_resource
def load_retrieval_system():
    """Load the corpus, both indexes, and an initialised RAG pipeline.

    Cached so subsequent reruns of the Streamlit page do not re-read the
    pickled indexes from disk on every interaction.
    """
    data_dir = project_root / "data" / "processed"

    try:
        df = load_data(str(data_dir / "books_sample.parquet"))
    except Exception:
        df = pd.DataFrame()

    try:
        with open(data_dir / "corpus.pkl", "rb") as f:
            corpus = pickle.load(f)
    except Exception:
        corpus = []

    bm25 = BM25Retriever()
    bm25.load(str(data_dir / "bm25_index.pkl"))
    bm25.corpus = corpus

    semantic = SemanticRetriever()
    semantic.load(str(data_dir / "semantic_index" / "faiss_index"))
    semantic.corpus = corpus

    llm = SimpleLLM()
    rag_pipeline = RAGPipeline(bm25, semantic, llm, prompt_version="balanced")
    rag_pipeline.corpus = corpus

    return df, corpus, bm25, semantic, rag_pipeline


def normalize_scores(scores: List[float]) -> List[float]:
    """Min-max normalise a list of scores to the [0, 1] interval.

    Returns a list of 0.5 when all scores are equal (no useful signal),
    avoiding a divide-by-zero.
    """
    if not scores:
        return scores
    scores = np.array(scores)
    min_val, max_val = scores.min(), scores.max()
    if max_val == min_val:
        return [0.5] * len(scores)
    return ((scores - min_val) / (max_val - min_val)).tolist()


def hybrid_search(query: str, bm25_retriever, semantic_retriever, top_k: int = 5, bm25_weight: float = 0.5):
    """Weighted hybrid search combining BM25 and semantic results.

    Each retriever returns its own ranked list; scores are min-max normalised
    (with FAISS distances first converted to similarity via 1/(1+d)) and then
    linearly combined with the user-selected weight.
    """
    bm25_results = bm25_retriever.search(query, top_k=top_k)
    semantic_results = semantic_retriever.search(query, top_k=top_k)

    bm25_dict = {idx: score for idx, score in bm25_results}
    semantic_dict = {idx: score for idx, score in semantic_results}
    all_indices = set(bm25_dict.keys()) | set(semantic_dict.keys())

    bm25_normalized = normalize_scores(list(bm25_dict.values()))
    semantic_normalized = normalize_scores([1 / (1 + d) for d in semantic_dict.values()])

    bm25_dict_norm = {idx: bm25_normalized[i] for i, idx in enumerate(bm25_dict.keys())}
    semantic_dict_norm = {idx: semantic_normalized[i] for i, idx in enumerate(semantic_dict.keys())}

    semantic_weight = 1.0 - bm25_weight
    hybrid_scores = {}
    for idx in all_indices:
        bm25_score = bm25_dict_norm.get(idx, 0)
        semantic_score = semantic_dict_norm.get(idx, 0)
        hybrid_scores[idx] = bm25_weight * bm25_score + semantic_weight * semantic_score

    sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


def display_book_result(rank: int, doc_id: int, df: pd.DataFrame, score: float = 0, method: str = ""):
    """Render a single retrieved book row with title, review snippet, rating and score."""
    if doc_id >= len(df) or doc_id < 0:
        return

    row = df.iloc[doc_id]
    title = row.get("product_title", "Unknown")
    review_text = row.get("text", "")
    rating = row.get("rating", 0)

    with st.container(border=True):
        col_rank, col_title = st.columns([0.08, 0.92])
        with col_rank:
            st.markdown(
                f"<div style='font-size: 18px; font-weight: bold; color: #1f77b4;'>{rank}</div>",
                unsafe_allow_html=True,
            )
        with col_title:
            st.markdown(
                f"<div style='font-size: 16px; font-weight: 600;'>{title}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        col_review, col_meta = st.columns([0.7, 0.3])
        with col_review:
            review_display = (
                (review_text[:250] + "..." if len(review_text) > 250 else review_text)
                if review_text else "(No review)"
            )
            st.caption(review_display)
        with col_meta:
            st.metric("Review Rating", f"{rating}/5.0", label_visibility="collapsed")

        col_score, col_id = st.columns(2)
        with col_score:
            if score > 0:
                if method == "semantic":
                    st.caption(f"Distance: {score:.4f} (lower = more similar)")
                elif method == "bm25":
                    st.caption(f"BM25 Score: {score:.2f} (higher = stronger keyword match)")
                elif method == "hybrid":
                    st.caption(f"Hybrid Score: {score:.3f} (normalised 0-1, higher = better)")
        with col_id:
            st.caption(f"Book ID: {doc_id}")


def setup_groq_sidebar():
    """Render the sidebar LLM selection controls and return the user's choice."""
    st.sidebar.markdown("### LLM Configuration")
    st.sidebar.markdown("---")

    llm_choice = st.sidebar.radio(
        "Select LLM",
        ["SimpleLLM (Demo)", "Groq (Production)"],
        help="SimpleLLM always works. Groq requires API key in .env file.",
    )

    if llm_choice == "Groq (Production)":
        st.sidebar.markdown("**Groq Setup**")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.sidebar.warning(
                "GROQ_API_KEY not found\n\n"
                "1. Get key: https://console.groq.com\n"
                "2. Create .env: GROQ_API_KEY=your_key\n"
                "3. Restart app"
            )
            return "simple", None

        st.sidebar.success("API key loaded")
        st.sidebar.info("Using auto-fallback model selection (LLaMA 3.3 70B primary)")

        return "groq", api_key

    return "simple", None


def render_about_tab():
    """Render the About tab with a user-facing guide, examples, and system notes."""
    st.markdown("### About This App")
    st.markdown(
        "This app is a search system over **110,167 Amazon book reviews** "
        "(sampled from the 11.7M-review Amazon Books 2023 dataset). It offers "
        "four different ways to find relevant books, each suited to a different "
        "style of query. You can compare their results side by side."
    )

    st.markdown("---")

    st.markdown("### Quick Start")
    st.markdown(
        "1. Type a query in the search bar at the top of the page.\n"
        "2. Click **Search**.\n"
        "3. Click through the tabs (**BM25**, **Semantic**, **Hybrid**, **RAG**) to "
        "see how each mode ranks results for the same query.\n"
        "4. Adjust per-tab controls (number of results, hybrid weighting, RAG prompt style) "
        "to see how the ranking changes."
    )

    st.markdown("---")

    st.markdown("### The Four Search Modes")

    with st.expander("BM25 Keyword Search"):
        st.markdown(
            "**What it does:** classical inverted-index retrieval using the Okapi "
            "BM25 algorithm. Matches exact tokens and scores by term frequency and "
            "inverse document frequency.\n\n"
            "**Best for:** well-known book titles, author names, concrete topics "
            "that will appear verbatim in reviews.\n\n"
            "**Example queries:**"
        )
        st.code(
            "python programming\n"
            "mystery novel\n"
            "cookbook recipes\n"
            "science fiction space",
            language=None,
        )
        st.caption("Speed: under 100 ms per query. Score: higher is better.")

    with st.expander("Semantic Meaning Search"):
        st.markdown(
            "**What it does:** encodes the query into a 384-dim vector using "
            "`all-MiniLM-L6-v2` and retrieves the nearest neighbours in a "
            "FAISS index of 110,167 document embeddings.\n\n"
            "**Best for:** paraphrased queries, intent-based searches, cases where "
            "the user does not know the exact words that appear in the review.\n\n"
            "**Example queries:**"
        )
        st.code(
            "book to help with anxiety\n"
            "guide for first time parents\n"
            "story about finding yourself\n"
            "something to read on a long flight",
            language=None,
        )
        st.caption("Speed: about 100 ms per query. Score: lower distance is better.")

    with st.expander("Hybrid Weighted Search"):
        st.markdown(
            "**What it does:** runs both BM25 and Semantic retrieval, normalises "
            "the two score ranges, and combines them with a weight slider.\n\n"
            "**Best for:** mixed queries that have both a concrete keyword and "
            "a looser intent. Adjust the BM25 weight slider to bias the result "
            "toward exact matching (right) or meaning matching (left).\n\n"
            "**Example queries:**"
        )
        st.code(
            "best machine learning book no math background\n"
            "historical fiction world war 2 female perspective\n"
            "self help book for procrastination",
            language=None,
        )
        st.caption(
            "Tip: start with the default 0.5 weight. If you get too many off-topic "
            "semantic results, slide toward BM25. If BM25 misses the point of the "
            "query, slide toward Semantic."
        )

    with st.expander("RAG (Retrieval-Augmented Generation)"):
        st.markdown(
            "**What it does:** retrieves the top-k books using hybrid search, "
            "stuffs the retrieved review text into a prompt, and sends it to "
            "Groq (LLaMA 3.3 70B by default). The LLM writes a grounded natural-language "
            "recommendation that references the retrieved reviews.\n\n"
            "**Best for:** open-ended recommendation questions, follow-up style "
            "queries, and any case where you want an explanation rather than a "
            "raw ranked list.\n\n"
            "**Prompt styles:**\n"
            "- **balanced** (default): lets the model synthesise and interpret the reviews.\n"
            "- **strict**: forces the model to stay very close to the retrieved text, "
            "  reducing hallucination risk but producing shorter answers.\n\n"
            "**Example queries:**"
        )
        st.code(
            "recommend a book for someone who just started coding\n"
            "what is a good introduction to machine learning without heavy math\n"
            "I need a relaxing read after a stressful week, any ideas",
            language=None,
        )
        st.caption("Requires a Groq API key in .env. Without one the app uses a SimpleLLM placeholder.")

    st.markdown("---")

    st.markdown("### Score Interpretation")
    st.markdown(
        "| Mode | Score displayed | Direction |\n"
        "|---|---|---|\n"
        "| BM25 | raw BM25 score | higher = better |\n"
        "| Semantic | L2 distance in embedding space | lower = better |\n"
        "| Hybrid | weighted normalised score | higher = better (scale 0-1) |\n"
        "| RAG | no score; answer is generated from top-k books | n/a |"
    )

    st.markdown("---")

    st.markdown("### Groq LLM Setup (optional, needed only for RAG)")
    st.markdown(
        "1. Sign up at https://console.groq.com (free, no credit card).\n"
        "2. Create an API key starting with `gsk_`.\n"
        "3. Copy `env.example` to `.env` in the project root and paste your key as "
        "`GROQ_API_KEY=gsk_...`.\n"
        "4. Restart the app.\n"
        "5. In the sidebar, switch from **SimpleLLM (Demo)** to **Groq (Production)**.\n\n"
        "The app uses an auto-fallback chain so if any model is decommissioned by "
        "Groq it cascades to the next one without user intervention."
    )

    st.markdown("---")

    st.markdown("### Data and Models")
    st.markdown(
        "- **Corpus:** 110,167 enriched review documents (sampled from 120K reviews, "
        "filtered to entries ≥20 characters). Built by `notebooks/02_data_preparation.ipynb`.\n"
        "- **BM25 index:** Okapi BM25 via `rank_bm25`. Built by `notebooks/03_bm25_keyword_search.ipynb`.\n"
        "- **Semantic index:** `all-MiniLM-L6-v2` embeddings (384-dim) indexed with "
        "FAISS `IndexFlatL2`. Built by `notebooks/04_semantic_embedding_search.ipynb`.\n"
        "- **RAG generation:** LLaMA 3.3 70B (primary) via Groq API, selected in "
        "`notebooks/08_llm_comparison.ipynb`.\n\n"
        "Full design discussion in `results/final_discussion.md`."
    )


def main():
    """Main Streamlit entry point."""
    st.set_page_config(page_title="Amazon Books Retrieval", page_icon="📚", layout="wide")

    st.markdown(
        """
    <style>
    .main-header { font-size: 2.5em; font-weight: 700; }
    .subtitle { font-size: 1.1em; color: #666; margin-bottom: 1.5em; }
    .badge { display: inline-block; padding: 0.5em 1em; border-radius: 20px; font-weight: 600; }
    .badge-bm25 { background-color: #FF6B6B; color: white; }
    .badge-semantic { background-color: #4ECDC4; color: white; }
    .badge-hybrid { background-color: #45B7D1; color: white; }
    .badge-rag { background-color: #96CEB4; color: white; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-header">Amazon Books Retrieval System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Final Submission: Dual-Method Retrieval with RAG over 110,167 Reviews</div>',
        unsafe_allow_html=True,
    )

    llm_type, groq_key = setup_groq_sidebar()

    st.markdown("---")
    st.markdown("### Search")

    col_input, col_btn = st.columns([0.92, 0.08])
    with col_input:
        query = st.text_input(
            "Enter search query",
            placeholder="E.g., 'machine learning for beginners'",
            label_visibility="collapsed",
        )
    with col_btn:
        search_button = st.button("Search", use_container_width=True, type="primary")

    if not query.strip() and search_button:
        st.warning("Please enter a search query")
        return

    st.markdown("---")

    tab_bm25, tab_semantic, tab_hybrid, tab_rag, tab_about = st.tabs(
        ["BM25", "Semantic", "Hybrid", "RAG", "About"]
    )

    try:
        df, corpus, bm25_retriever, semantic_retriever, rag_pipeline = load_retrieval_system()
    except Exception as e:
        st.error(f"Error loading system: {e}")
        return

    with tab_bm25:
        st.markdown("<span class='badge badge-bm25'>BM25</span> Keyword Search", unsafe_allow_html=True)
        st.markdown("Fast exact-match retrieval using term frequency scoring.")
        top_k = st.slider("Results", 1, 10, 5, key="bm25_k")

        if search_button and query.strip():
            try:
                with st.spinner("Searching..."):
                    results = bm25_retriever.search(query, top_k=top_k)
                st.success(f"Found {len(results)} results")
                st.markdown("---")
                for rank, (doc_id, score) in enumerate(results, 1):
                    display_book_result(rank, doc_id, df, score, method="bm25")
            except Exception as e:
                st.error(f"Search error: {e}")

    with tab_semantic:
        st.markdown("<span class='badge badge-semantic'>Semantic</span> Meaning-Based Search", unsafe_allow_html=True)
        st.markdown("Embedding-based search that understands semantic meaning and synonyms.")
        top_k = st.slider("Results", 1, 10, 5, key="semantic_k")

        if search_button and query.strip():
            try:
                with st.spinner("Searching..."):
                    results = semantic_retriever.search(query, top_k=top_k)
                st.success(f"Found {len(results)} results")
                st.markdown("---")
                for rank, (doc_id, distance) in enumerate(results, 1):
                    display_book_result(rank, doc_id, df, distance, method="semantic")
            except Exception as e:
                st.error(f"Search error: {e}")

    with tab_hybrid:
        st.markdown("<span class='badge badge-hybrid'>Hybrid</span> Combined Search", unsafe_allow_html=True)
        st.markdown("Balances keyword precision with semantic understanding.")
        col_k, col_w = st.columns(2)
        with col_k:
            top_k = st.slider("Results", 1, 10, 5, key="hybrid_k")
        with col_w:
            bm25_weight = st.slider(
                "BM25 Weight",
                0.0,
                1.0,
                0.5,
                step=0.1,
                help="0.0 = pure semantic, 1.0 = pure BM25, 0.5 = balanced",
            )

        if search_button and query.strip():
            try:
                with st.spinner("Searching..."):
                    results = hybrid_search(
                        query, bm25_retriever, semantic_retriever,
                        top_k=top_k, bm25_weight=bm25_weight,
                    )
                st.success(f"Found {len(results)} results")
                st.markdown("---")
                for rank, (doc_id, score) in enumerate(results, 1):
                    display_book_result(rank, doc_id, df, score, method="hybrid")
            except Exception as e:
                st.error(f"Search error: {e}")

    with tab_rag:
        st.markdown("<span class='badge badge-rag'>RAG</span> Question Answering", unsafe_allow_html=True)
        st.markdown("Retrieval-augmented generation: finds books and generates an AI answer grounded in their reviews.")

        col_k, col_p = st.columns(2)
        with col_k:
            top_k = st.slider("Top Books", 1, 10, 5, key="rag_k")
        with col_p:
            prompt_version = st.selectbox(
                "Prompt Style",
                ["balanced", "strict"],
                help="balanced = more fluent synthesis; strict = stays closer to retrieved text",
            )

        if search_button and query.strip():
            try:
                with st.spinner("Generating answer..."):
                    if llm_type == "groq" and groq_key:
                        llm = GroqLLM(api_key=groq_key)
                        rag_pipeline.llm = llm

                    result = rag_pipeline.invoke(query, top_k=top_k)

                st.success("Answer generated")
                st.markdown("---")

                st.markdown("#### Generated Answer")
                st.info(result.get("answer", "No answer"))

                st.markdown("#### Retrieved Books")

                retrieved_ids = result.get("retrieved_doc_ids", [])
                with st.expander(f"Show {len(retrieved_ids)} source documents"):
                    if retrieved_ids and len(retrieved_ids) > 0:
                        st.markdown("---")
                        for i, doc_id in enumerate(retrieved_ids, 1):
                            display_book_result(i, doc_id, df, 0, method="")
                    else:
                        st.caption("No documents retrieved")

            except Exception as e:
                st.error(f"RAG Error: {e}")

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()