import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bm25 import BM25Retriever
from src.semantic import SemanticRetriever
from src.utils import load_data, load_corpus


@st.cache_resource
def load_retrieval_system():
    """Load indexes and corpus on app startup."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    
    df = load_data(str(data_dir / "books_sample.parquet"))
    corpus = load_corpus(str(data_dir / "corpus.pkl"))
    
    bm25 = BM25Retriever()
    bm25.load(str(data_dir / "bm25_index.pkl"))
    bm25.corpus = corpus
    
    semantic = SemanticRetriever()
    semantic.load(str(data_dir / "semantic_index"))
    semantic.corpus = corpus
    
    return df, corpus, bm25, semantic


def normalize_scores(scores: List[float], method: str = "minmax") -> List[float]:
    """Normalize scores to 0-1 range for display."""
    if not scores:
        return scores
    scores = np.array(scores)
    if method == "minmax":
        min_val, max_val = scores.min(), scores.max()
        if max_val == min_val:
            return [0.5] * len(scores)
        return ((scores - min_val) / (max_val - min_val)).tolist()
    return scores.tolist()


def hybrid_search(
    query: str,
    bm25_retriever: BM25Retriever,
    semantic_retriever: SemanticRetriever,
    top_k: int = 5,
    bm25_weight: float = 0.5,
    semantic_weight: float = 0.5
) -> List[Tuple[int, float]]:
    """Combine BM25 and semantic search results."""
    bm25_results = bm25_retriever.search(query, top_k=top_k)
    semantic_results = semantic_retriever.search(query, top_k=top_k)
    
    bm25_dict = {idx: score for idx, score in bm25_results}
    semantic_dict = {idx: score for idx, score in semantic_results}
    
    all_indices = set(bm25_dict.keys()) | set(semantic_dict.keys())
    
    bm25_normalized = normalize_scores(list(bm25_dict.values()), "minmax")
    semantic_normalized = normalize_scores(
        [1 / (1 + d) for d in semantic_dict.values()], "minmax"
    )
    
    bm25_dict_norm = {
        idx: bm25_normalized[i] for i, idx in enumerate(bm25_dict.keys())
    }
    semantic_dict_norm = {
        idx: semantic_normalized[i] for i, idx in enumerate(semantic_dict.keys())
    }
    
    hybrid_scores = {}
    for idx in all_indices:
        bm25_score = bm25_dict_norm.get(idx, 0)
        semantic_score = semantic_dict_norm.get(idx, 0)
        hybrid_scores[idx] = (
            bm25_weight * bm25_score + semantic_weight * semantic_score
        )
    
    sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


def display_result(
    rank: int,
    doc_id: int,
    title: str,
    review_text: str,
    rating: float,
    score: float,
    method: str
):
    """Display a single search result with formatting."""
    with st.container(border=True):
        col1, col2 = st.columns([0.15, 0.85])
        with col1:
            st.markdown(f"### #{rank}")
        with col2:
            st.markdown(f"#### {title}")
        
        col_a, col_b = st.columns([0.7, 0.3])
        with col_a:
            review_display = (
                review_text[:200] + "..."
                if len(review_text) > 200
                else review_text
            )
            st.caption(f"__{review_display}__")
        with col_b:
            st.metric("Rating", f"{rating}/5", label_visibility="collapsed")
        
        col_x, col_y = st.columns(2)
        with col_x:
            if method == "semantic":
                st.caption(f"Distance: {score:.4f}")
            else:
                st.caption(f"Score: {score:.2f}")
        with col_y:
            st.caption(f"Doc ID: {doc_id}")


def main():
    st.set_page_config(
        page_title="Amazon Books Retrieval",
        page_icon="book",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Amazon Books Retrieval System")
    st.markdown(
        "Explore books using **BM25 keyword search** and **semantic embeddings**"
    )
    
    try:
        df, corpus, bm25_retriever, semantic_retriever = load_retrieval_system()
    except Exception as e:
        st.error(f"Error loading retrieval system: {e}")
        st.info(
            "Make sure you've run the data preparation notebooks and have "
            "the processed files in `data/processed/`"
        )
        return
    
    with st.sidebar:
        st.header("Settings")
        
        search_method = st.radio(
            "Search Method",
            ["BM25 (Keyword)", "Semantic (Embeddings)", "Hybrid (Combined)"],
            help="BM25: Fast keyword matching | Semantic: Meaning-aware | Hybrid: Best of both"
        )
        
        top_k = st.slider(
            "Number of Results",
            min_value=1,
            max_value=10,
            value=5,
            help="Top-k results to retrieve"
        )
        
        if search_method == "Hybrid (Combined)":
            st.markdown("**Hybrid Weights**")
            bm25_weight = st.slider(
                "BM25 Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
            semantic_weight = 1.0 - bm25_weight
            st.caption(f"Semantic Weight: {semantic_weight:.1f}")
    
    query = st.text_input(
        "Search Query",
        placeholder="E.g., 'machine learning for beginners' or 'self-help book for anxiety'",
        help="Enter your search query. Try both simple keyword queries and complex multi-part requests."
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col2:
        search_button = st.button("Search", use_container_width=True)
    
    if search_button and query.strip():
        try:
            with st.spinner(f"Searching with {search_method}..."):
                if search_method == "BM25 (Keyword)":
                    results = bm25_retriever.search(query, top_k=top_k)
                    method_name = "bm25"
                elif search_method == "Semantic (Embeddings)":
                    results = semantic_retriever.search(query, top_k=top_k)
                    method_name = "semantic"
                else:
                    results = hybrid_search(
                        query,
                        bm25_retriever,
                        semantic_retriever,
                        top_k=top_k,
                        bm25_weight=bm25_weight,
                        semantic_weight=semantic_weight
                    )
                    method_name = "hybrid"
            
            st.success(f"Found {len(results)} results")
            st.divider()
            
            for rank, (doc_id, score) in enumerate(results, 1):
                if doc_id < len(df):
                    row = df.iloc[doc_id]
                    title = row.get("title", "Unknown Title")
                    review_text = row.get("text", "")
                    rating = row.get("rating", 0)
                    
                    display_result(
                        rank=rank,
                        doc_id=doc_id,
                        title=title,
                        review_text=review_text,
                        rating=rating,
                        score=score,
                        method=method_name
                    )
        
        except Exception as e:
            st.error(f"Search error: {e}")
    
    elif search_button and not query.strip():
        st.warning("Please enter a search query")
    
    with st.expander("About This System"):
        st.markdown("""
        ### Retrieval Methods
        
        **BM25 (Keyword Search)**
        - Fast, exact keyword matching
        - Best for: Simple, direct queries with specific terms
        - Weakness: Struggles with semantic meaning and word sense disambiguation
        
        **Semantic Search (Embeddings)**
        - Understands meaning through learned embeddings
        - Best for: Complex, intent-based queries; synonyms and paraphrasing
        - Weakness: Slower; less transparent; requires good training data
        
        **Hybrid Search**
        - Combines strengths of both methods
        - Uses weighted combination for balanced results
        - Best for: Queries where you want both precision and semantic understanding
        
        ### Dataset
        - **Source**: Amazon Reviews 2023 (Books category)
        - **Size**: ~20,000 stratified book samples
        - **Fields**: Book title + review text (concatenated for better context)
        """)


if __name__ == "__main__":
    main()