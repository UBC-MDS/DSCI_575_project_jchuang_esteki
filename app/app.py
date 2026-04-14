import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import sys
import pickle

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bm25 import BM25Retriever
from src.semantic_retriever import SemanticRetriever
from src.rag_pipeline import RAGPipeline
from src.utils import load_data, load_corpus


class SimpleLLM:
    """Simple LLM wrapper for Milestone 2 demo."""
    def invoke(self, text: str) -> str:
        return "Based on the provided reviews, this book appears to be well-received by customers who value its content, writing style, and overall quality. Readers particularly appreciate books that are informative, engaging, and provide clear value."


@st.cache_resource
def load_retrieval_system():
    """Load indexes, corpus, and RAG pipeline on app startup."""
    project_root = Path(__file__).parent.parent
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


def display_rag_result(result: Dict, df: pd.DataFrame):
    """Display RAG result with retrieved documents and generated answer."""
    with st.container(border=True):
        st.markdown("### Generated Answer")
        st.info(result['answer'])
        
        with st.expander(f"Retrieved Documents ({result['documents_retrieved']})"):
            st.caption(f"Context length: {result['context_length']} characters")
            
            if 'retrieved_docs' in result:
                for i, doc_id in enumerate(result['retrieved_docs'], 1):
                    if doc_id < len(df):
                        row = df.iloc[doc_id]
                        st.markdown(f"**Document {i}** - {row.get('title', 'Unknown')}")
                        st.caption(row.get('text', '')[:300] + "...")


def main():
    st.set_page_config(
        page_title="Amazon Books Retrieval - Milestone 2",
        page_icon="book",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Amazon Books Retrieval System")
    st.markdown("Milestone 2: **BM25**, **Semantic**, **Hybrid**, and **RAG** Retrieval")
    
    try:
        df, corpus, bm25_retriever, semantic_retriever, rag_pipeline = load_retrieval_system()
    except Exception as e:
        st.error(f"Error loading retrieval system: {e}")
        st.info(
            "Make sure you've run all data preparation notebooks and have "
            "the processed files in `data/processed/` (corpus.pkl, bm25_index.pkl, semantic_index/)"
        )
        return
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["BM25", "Semantic", "Hybrid", "RAG", "About"]
    )
    
    query_input = st.text_input(
        "Search Query",
        placeholder="E.g., 'machine learning for beginners' or 'self-help for anxiety'",
        help="Enter your search query"
    )
    
    col1, col2 = st.columns([0.85, 0.15])
    with col2:
        search_button = st.button("Search", use_container_width=True)
    
    with tab1:
        st.header("BM25 Keyword Search")
        st.markdown("Fast keyword matching using term frequency and document length normalization.")
        
        col_a, col_b = st.columns(2)
        with col_a:
            top_k_bm25 = st.slider("Top K Results", 1, 10, 5, key="bm25_k")
        
        if search_button and query_input.strip():
            try:
                with st.spinner("Searching with BM25..."):
                    results = bm25_retriever.search(query_input, top_k=top_k_bm25)
                
                st.success(f"Found {len(results)} results")
                st.divider()
                
                for rank, (doc_id, score) in enumerate(results, 1):
                    if doc_id < len(df):
                        row = df.iloc[doc_id]
                        title = row.get("title", "Unknown")
                        review_text = row.get("text", "")
                        rating = row.get("rating", 0)
                        
                        display_result(rank, doc_id, title, review_text, rating, score, "bm25")
            
            except Exception as e:
                st.error(f"Search error: {e}")
        elif search_button and not query_input.strip():
            st.warning("Please enter a search query")
    
    with tab2:
        st.header("Semantic Embedding Search")
        st.markdown("Meaning-aware search using sentence embeddings (all-MiniLM-L6-v2).")
        
        col_a, col_b = st.columns(2)
        with col_a:
            top_k_semantic = st.slider("Top K Results", 1, 10, 5, key="semantic_k")
        
        if search_button and query_input.strip():
            try:
                with st.spinner("Searching with Semantic embeddings..."):
                    results = semantic_retriever.search(query_input, top_k=top_k_semantic)
                
                st.success(f"Found {len(results)} results")
                st.divider()
                
                for rank, (doc_id, distance) in enumerate(results, 1):
                    if doc_id < len(df):
                        row = df.iloc[doc_id]
                        title = row.get("title", "Unknown")
                        review_text = row.get("text", "")
                        rating = row.get("rating", 0)
                        
                        display_result(rank, doc_id, title, review_text, rating, distance, "semantic")
            
            except Exception as e:
                st.error(f"Search error: {e}")
        elif search_button and not query_input.strip():
            st.warning("Please enter a search query")
    
    with tab3:
        st.header("Hybrid Retrieval")
        st.markdown("Combines BM25 and semantic search with weighted scoring.")
        
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            top_k_hybrid = st.slider("Top K Results", 1, 10, 5, key="hybrid_k")
        with col_b:
            bm25_weight = st.slider("BM25 Weight", 0.0, 1.0, 0.5, step=0.1, key="bm25_w")
        with col_c:
            semantic_weight = 1.0 - bm25_weight
            st.metric("Semantic Weight", f"{semantic_weight:.1f}", label_visibility="collapsed")
        
        if search_button and query_input.strip():
            try:
                with st.spinner("Searching with Hybrid approach..."):
                    results = hybrid_search(
                        query_input,
                        bm25_retriever,
                        semantic_retriever,
                        top_k=top_k_hybrid,
                        bm25_weight=bm25_weight,
                        semantic_weight=semantic_weight
                    )
                
                st.success(f"Found {len(results)} results")
                st.divider()
                
                for rank, (doc_id, score) in enumerate(results, 1):
                    if doc_id < len(df):
                        row = df.iloc[doc_id]
                        title = row.get("title", "Unknown")
                        review_text = row.get("text", "")
                        rating = row.get("rating", 0)
                        
                        display_result(rank, doc_id, title, review_text, rating, score, "hybrid")
            
            except Exception as e:
                st.error(f"Search error: {e}")
        elif search_button and not query_input.strip():
            st.warning("Please enter a search query")
    
    with tab4:
        st.header("RAG: Retrieval-Augmented Generation")
        st.markdown("Hybrid retrieval + LLM generation for question answering over book reviews.")
        
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            top_k_rag = st.slider("Top K Documents", 1, 10, 5, key="rag_k")
        with col_b:
            prompt_version = st.selectbox("Prompt Version", ["balanced", "strict"], key="prompt_v")
        
        st.info("SimpleLLM (demo) - Use Qwen or Groq for production")
        
        if search_button and query_input.strip():
            try:
                with st.spinner("Running RAG pipeline..."):
                    result = rag_pipeline.invoke(query_input, top_k=top_k_rag)
                
                st.success("RAG generation complete")
                st.divider()
                
                display_rag_result(result, df)
            
            except Exception as e:
                st.error(f"RAG error: {e}")
                st.info("Make sure src/rag_pipeline.py, src/prompts.py, and src/chunking.py are generated")
        elif search_button and not query_input.strip():
            st.warning("Please enter a question for RAG")
    
    with tab5:
        st.header("About This System")
        
        st.markdown("### Milestone 2 Features")
        st.markdown("""
        #### Retrieval Methods
        
        **1. BM25 (Keyword Search)**
        - Fast, exact keyword matching using Okapi BM25
        - Best for: Simple queries with specific terms
        - Pros: Fast, interpretable, requires no training
        - Cons: Struggles with semantic meaning
        
        **2. Semantic Search (Embeddings)**
        - Meaning-aware search using sentence embeddings
        - Model: all-MiniLM-L6-v2 from HuggingFace
        - Best for: Complex, intent-based queries; synonyms
        - Pros: Understands semantic relationships
        - Cons: Slower; requires good training data
        
        **3. Hybrid Retrieval**
        - Combines BM25 and semantic with weighted scoring
        - Configurable weights for different use cases
        - Best for: Balanced precision and recall
        - Pros: Leverages strengths of both methods
        - Cons: Slightly slower than BM25 alone
        
        **4. RAG (Retrieval-Augmented Generation)**
        - Hybrid retrieval + LLM for question answering
        - Uses document chunking for context management
        - Includes prompt templates (balanced/strict)
        - Best for: Questions requiring synthesis of multiple reviews
        - Pros: Natural language answers, context-aware
        - Cons: Depends on LLM quality
        
        #### Dataset
        - **Source**: Amazon Reviews 2023 (Books)
        - **Size**: 20,000 stratified samples
        - **Fields**: Book title + review text + rating
        
        #### Architecture
        - Corpus enrichment: Title + rating + reviews
        - Chunking: 500 character chunks, 50 char overlap
        - Max context: 2000 tokens per RAG query
        - LLM: SimpleLLM (demo) / Qwen / Groq (production)
        """)
        
        st.markdown("### Performance Tips")
        st.markdown("""
        - **BM25**: Use for keyword-heavy queries
        - **Semantic**: Best for conceptual questions
        - **Hybrid**: Default choice for most queries
        - **RAG**: Use for complex questions needing synthesis
        """)


if __name__ == "__main__":
    main()
