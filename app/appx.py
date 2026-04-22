import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import sys
import pickle
import os
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
        return "Based on the provided reviews, this book appears to be well-received by customers who value its content, writing style, and overall quality."


class GroqLLM:
    """Groq API - with model fallback for deprecation."""
    AVAILABLE_MODELS = [
        "llama-3.2-90b-vision-preview",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
        "mixtral-8x7b-32768"
    ]
    
    def __init__(self, api_key: str = None, model: str = None):
        try:
            from groq import Groq
            self.api_key = api_key or os.getenv("GROQ_API_KEY")
            if not self.api_key:
                raise ValueError("GROQ_API_KEY not found in .env file")
            self.client = Groq(api_key=self.api_key)
            self.model = model or "llama-3.2-90b-vision-preview"
            self.available = True
        except ImportError:
            self.available = False
    
    def invoke(self, text: str) -> str:
        if not self.available:
            return "Groq library not installed. Run: pip install groq"
        
        models_to_try = [self.model] + [m for m in self.AVAILABLE_MODELS if m != self.model]
        
        for model in models_to_try:
            try:
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": text}],
                    model=model,
                    max_tokens=500,
                    temperature=0.7
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
    """Load all components."""
    data_dir = project_root / "data" / "processed"
    
    try:
        df = load_data(str(data_dir / "books_sample.parquet"))
    except:
        df = pd.DataFrame()
    
    try:
        with open(data_dir / "corpus.pkl", "rb") as f:
            corpus = pickle.load(f)
    except:
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
    """Normalize scores."""
    if not scores:
        return scores
    scores = np.array(scores)
    min_val, max_val = scores.min(), scores.max()
    if max_val == min_val:
        return [0.5] * len(scores)
    return ((scores - min_val) / (max_val - min_val)).tolist()


def hybrid_search(query: str, bm25_retriever, semantic_retriever, top_k: int = 5, bm25_weight: float = 0.5):
    """Hybrid search combining BM25 and semantic."""
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
    """Display book with metadata and score."""
    if doc_id >= len(df) or doc_id < 0:
        return
    
    row = df.iloc[doc_id]
    title = row.get("product_title", "Unknown")
    review_text = row.get("text", "")
    rating = row.get("rating", 0)
    
    with st.container(border=True):
        col_rank, col_title = st.columns([0.08, 0.92])
        with col_rank:
            st.markdown(f"<div style='font-size: 18px; font-weight: bold; color: #1f77b4;'>{rank}</div>", unsafe_allow_html=True)
        with col_title:
            st.markdown(f"<div style='font-size: 16px; font-weight: 600;'>{title}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        col_review, col_meta = st.columns([0.7, 0.3])
        with col_review:
            review_display = (review_text[:250] + "..." if len(review_text) > 250 else review_text) if review_text else "(No review)"
            st.caption(review_display)
        with col_meta:
            st.metric("Review Rating", f"{rating}/5.0", label_visibility="collapsed")
        
        col_score, col_id = st.columns(2)
        with col_score:
            if score > 0:
                if method == "semantic":
                    st.caption(f"Distance: {score:.4f}")
                elif method == "bm25":
                    st.caption(f"BM25 Score: {score:.2f}")
                elif method == "hybrid":
                    st.caption(f"Hybrid Score: {score:.3f}")
        with col_id:
            st.caption(f"Book ID: {doc_id}")


def setup_groq_sidebar():
    """Configure Groq LLM in sidebar."""
    st.sidebar.markdown("### LLM Configuration")
    st.sidebar.markdown("---")
    
    llm_choice = st.sidebar.radio(
        "Select LLM",
        ["SimpleLLM (Demo)", "Groq (Production)"],
        help="SimpleLLM always works. Groq requires API key in .env file."
    )
    
    if llm_choice == "Groq (Production)":
        st.sidebar.markdown("**Groq Setup**")
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.sidebar.warning(
                "⚠️ GROQ_API_KEY not found\n\n"
                "1. Get key: https://console.groq.com\n"
                "2. Create .env: GROQ_API_KEY=your_key\n"
                "3. Restart app"
            )
            return "simple", None
        
        st.sidebar.success("✓ API key loaded")
        st.sidebar.info("⚡ Using auto-fallback model selection")
        
        return "groq", api_key
    
    return "simple", None


def main():
    """Main app."""
    st.set_page_config(page_title="Amazon Books Retrieval", page_icon="📚", layout="wide")
    
    st.markdown("""
    <style>
    .main-header { font-size: 2.5em; font-weight: 700; }
    .subtitle { font-size: 1.1em; color: #666; margin-bottom: 1.5em; }
    .badge { display: inline-block; padding: 0.5em 1em; border-radius: 20px; font-weight: 600; }
    .badge-bm25 { background-color: #FF6B6B; color: white; }
    .badge-semantic { background-color: #4ECDC4; color: white; }
    .badge-hybrid { background-color: #45B7D1; color: white; }
    .badge-rag { background-color: #96CEB4; color: white; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">📚 Amazon Books Retrieval System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Milestone 2: Multi-Modal Search with RAG</div>', unsafe_allow_html=True)
    
    llm_type, groq_key = setup_groq_sidebar()
    
    st.markdown("---")
    st.markdown("### Search")
    
    col_input, col_btn = st.columns([0.92, 0.08])
    with col_input:
        query = st.text_input("Enter search query", placeholder="E.g., 'machine learning for beginners'", label_visibility="collapsed")
    with col_btn:
        search_button = st.button("Search", use_container_width=True, type="primary")
    
    if not query.strip() and search_button:
        st.warning("Please enter a search query")
        return
    
    st.markdown("---")
    
    tab_bm25, tab_semantic, tab_hybrid, tab_rag, tab_about = st.tabs(["BM25", "Semantic", "Hybrid", "RAG", "About"])
    
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
            bm25_weight = st.slider("BM25 Weight", 0.0, 1.0, 0.5, step=0.1)
        
        if search_button and query.strip():
            try:
                with st.spinner("Searching..."):
                    results = hybrid_search(query, bm25_retriever, semantic_retriever, top_k=top_k, bm25_weight=bm25_weight)
                st.success(f"Found {len(results)} results")
                st.markdown("---")
                for rank, (doc_id, score) in enumerate(results, 1):
                    display_book_result(rank, doc_id, df, score, method="hybrid")
            except Exception as e:
                st.error(f"Search error: {e}")
    
    with tab_rag:
        st.markdown("<span class='badge badge-rag'>RAG</span> Question Answering", unsafe_allow_html=True)
        st.markdown("Retrieval-augmented generation: finds books and generates AI answer.")
        
        col_k, col_p = st.columns(2)
        with col_k:
            top_k = st.slider("Top Books", 1, 10, 5, key="rag_k")
        with col_p:
            prompt_version = st.selectbox("Prompt Style", ["balanced", "strict"])
        
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
                st.info(result.get('answer', 'No answer'))
                
                st.markdown("#### Retrieved Books")
                
                retrieved_ids = result.get('retrieved_doc_ids', [])
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
        st.markdown("### System Overview")
        st.markdown("""
**4 Search Methods:**
- BM25: Keyword matching
- Semantic: Meaning-based
- Hybrid: Both combined
- RAG: AI-powered answers

**Groq Integration:**
Auto-fallback to available models
Models tested: 5 latest versions

**Getting Started:**
1. https://console.groq.com (free)
2. Create .env with GROQ_API_KEY
3. Select Groq mode
4. Done!
""")


if __name__ == "__main__":
    main()
