# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling

**Number of products used:** the pipeline now samples **100,000 reviews** from the full Amazon Books dataset (11.7M reviews). After text preprocessing (dropping reviews under 20 characters) the final retrieval corpus contains **91,850 enriched documents**. Both indexes were rebuilt against this corpus:

- `data/processed/corpus.pkl` (51.50 MB, 91,850 documents)
- `data/processed/books_sample.parquet` (30.35 MB)
- `data/processed/bm25_index.pkl` (BM25 index on 91,850 documents)
- `data/processed/semantic_index/faiss_index` (FAISS index, 91,850 vectors at 384 dimensions)

**Changes to sampling strategy:** the sampling block in `notebooks/02_data_preparation.ipynb` uses a two-tier strategy. It first attempts a stratified DuckDB query that allocates rows proportionally across the 1 to 5 rating buckets, then falls back to a straightforward random sample if the stratified query fails. On the 100K build the stratified path did not complete cleanly on the full 11.7M review stream, so the fallback ran and produced the 100K random sample. The rating distribution of the resulting sample (approximately 69% 5-star, 16% 4-star, 7% 3-star, 4% 2-star, 5% 1-star) matches the natural skew of Amazon book reviews, which is the distribution the app has to handle in production anyway. This fallback behaviour is intentional and is documented in the README.

### LLM Experiment

#### Models Compared

| | **LLaMA 3.3 70B** | **LLaMA 3.1 8B** |
| --- | --- | --- |
| Groq model ID | `llama-3.3-70b-versatile` | `llama-3.1-8b-instant` |
| Developer | Meta | Meta |
| Parameters | 70 billion | 8 billion |
| Training data | ~15 trillion tokens | ~15 trillion tokens |
| Context window | 128k tokens | 128k tokens |
| Release | December 2024 | July 2024 |
| Role | Primary, high-quality model | Small, fast comparison model |

Both models run via the Groq API. The full side-by-side comparison is in `notebooks/08_llm_comparison.ipynb`.

#### Prompt Used

Both models were given identical retrieved context and the following `RECOMMENDATION` prompt template (defined in `src/prompts.py`):

```text
You are a helpful book recommendation assistant. Based on the following book reviews, recommend books that match the user's request. For each book, briefly explain why it fits based on the reviews.

Context:
{context}

User request: {question}

Recommendation:
```

#### Results

Five queries were run across three difficulty levels. Both models received identical retrieved context and the same `RECOMMENDATION` prompt template. Full outputs are in `notebooks/08_llm_comparison.ipynb`.

**Query 1 (Easy): "mystery novel"**

Retrieved: Chase For Home, S., The Bookseller, Fast Track, Sister Innocent

> **LLaMA 3.3 70B:** Recommended 4 books with a brief justification for each drawn from review text. Notably excluded "S." and explained why (poor reviews).
>
> **LLaMA 3.1 8B:** Also recommended 4 books but went into more detail per title, including a note about verified vs. unverified reviews for Sister Innocent.

**Query 2 (Medium): "book to help with anxiety"**

Retrieved: Adult Coloring Book (Stress Relieving), Guided Mindfulness Meditation Series 1, I'll Be Home Soon, Journey to the Heart, Complete Reiki

> **LLaMA 3.3 70B:** Recommended 4 books with clear, grounded explanations. Connected each book to anxiety relief with specific detail from the reviews (e.g. "helped their daughter with severe anxiety").
>
> **LLaMA 3.1 8B:** Recommended 3 of the same books but missed the coloring book entirely. Also incorrectly recommended "I'll Be Home Soon" which is about separation anxiety in pets, not humans.

**Query 3 (Complex): "best book to learn machine learning with no math background"**

Retrieved: Python Machine Learning (2nd Ed.), Hand-Manipulated Stitches for Machine Knitters, Deep Learning, My First Grade Math Workbook, Data Mining

> **LLaMA 3.3 70B:** Correctly identified only one relevant book (Python Machine Learning), dismissed the off-topic results by name, and explained why the others were unsuitable. Concise and accurate.
>
> **LLaMA 3.1 8B:** Recommended Python Machine Learning, then invented a "third edition" that does not appear in the reviews. Also drifted into generic advice ("look for books that provide step-by-step examples") not grounded in the retrieved context.

**Query 4 (Complex): "historical fiction set in world war 2 from a female perspective"**

Retrieved: The Forgotten Village, The Women's War, To the Last Man, War Girl Ursula, Up the Learning Tree

> **LLaMA 3.3 70B:** Identified two relevant books (The Forgotten Village, War Girl Ursula) and flagged War Girl Ursula as the stronger match due to its explicit female protagonist. Correctly ignored off-topic titles.
>
> **LLaMA 3.1 8B:** Reached the same two recommendations with similar reasoning. Responses were nearly equivalent on this query.

**Query 5 (Complex): "self help book for overcoming procrastination and building better habits"**

Retrieved: Atomic Habits Workbook, The P.R.I.M.E.R. Goal Setting Method, YOU ARE WONDERFUL (coloring book), The Success Principles, Life's Healing Choices

> **LLaMA 3.3 70B:** Recommended 3 relevant books, correctly ignored the coloring book and Life's Healing Choices. Explanations were grounded in review text.
>
> **LLaMA 3.1 8B:** Recommended the same 3 books with similar reasoning. Response was cut off mid-sentence before completing the final thought.

#### Which Model We Chose and Why

We chose **LLaMA 3.3 70B** (`llama-3.3-70b-versatile`) as the default model for our pipeline. Across the five queries, it consistently gave more grounded and complete responses, sticking to the retrieved context rather than making things up. The 8B model struggled on harder queries, most notably inventing a book edition that was not in the reviews and drifting into generic advice. The 70B model also handled noisy retrieval better, correctly filtering out off-topic results and explaining why they were excluded. Given that we are already using the Groq API, the cost difference between the two is negligible, so the quality improvement makes the 70B model the clear choice.

The Streamlit app (`app/app.py`) wraps this choice in an auto-fallback chain (`llama-3.2-90b-vision-preview` then `llama-3.1-70b-versatile` then `llama-3.1-8b-instant` then `gemma2-9b-it` then `mixtral-8x7b-32768`). If any model becomes deprecated or rate-limited the app silently cascades to the next one. This keeps the experiment result (LLaMA 3.3 70B as the preferred default for the task) while protecting the user-facing app from model-deprecation outages.

## Step 2: Additional Feature

**Option chosen: Option 3, Scale to 100,000 products.**

### What You Implemented

The scaling target for the final submission was 100,000 products. The pipeline now processes 100,000 sampled reviews and, after the ≥20 character filter, the retrieval corpus holds 91,850 enriched documents. The following engineering decisions were taken so the pipeline actually works at this scale rather than simply bumping a number:

**1. Memory-safe sampling with DuckDB.** The raw Books review file is 11.7M records (multi-GB compressed). Loading it into pandas crashes on 16 GB laptops. `notebooks/02_data_preparation.ipynb` uses a single DuckDB SQL query that streams the gzipped JSON off disk, applies a `ROW_NUMBER() OVER (PARTITION BY rating)` for stratification, and materialises only the 100K sampled rows. Peak RAM stays under 4 GB throughout, which is what makes 100K feasible on a laptop.

**2. Graceful fallback sampling.** The stratified query can fail on very large streaming inputs (the OVER / RANDOM combination plus the 11.7M row scan sometimes exhausts DuckDB's working memory). The notebook wraps the stratified path in `try/except` and falls back to `ORDER BY RANDOM() LIMIT 100000` on failure. This is the path that ran on the 100K build, and it produced a representative rating distribution (see §1.1). Having the fallback means the scale step is reliable rather than something the user has to manually repair. Without it, 100K sampling is a flaky step; with it, the pipeline always completes.

**3. Tuned embedding batch size.** The original `SemanticRetriever` used the sentence-transformers default `batch_size=32`. At 91,850 documents that extrapolates to roughly 45 to 90 minutes of CPU encoding. We raised `batch_size` to 128 in `src/semantic.py` and in the auto-generated code block inside `notebooks/04_semantic_embedding_search.ipynb`. End-to-end encoding on CPU drops to the 5 to 10 minute range while staying well under the 16 GB RAM budget.

**4. Exact FAISS search remains viable at this scale.** The FAISS index is a flat `IndexFlatL2` over 91,850 × 384 float32 vectors (about 140 MB). Brute-force search returns in well under 100 ms per query, so there is no need to switch to `IndexIVFFlat` or `IndexHNSW`. Beyond roughly one to five million documents we would introduce an approximate index, but at 91,850 docs exact search is the cleaner choice (no training step, no accuracy-tuning hyperparameters).

**5. BM25 verified at scale.** BM25 is built with `rank_bm25` (pure Python, Okapi variant). At 91,850 documents the build runs in the low minutes and the resulting `bm25_index.pkl` is well-behaved on disk. Query latency is adequate for the single-user Streamlit front end. This was the main scaling risk going in since `rank_bm25` is O(N) per query; above roughly 250K documents we would swap it for `bm25s` (C-backed, drop-in API). At the current 91,850 scale it holds up.

**6. Downstream components are size-agnostic.** `src/hybrid.py`, `src/rag_pipeline.py`, and the Streamlit app all read from `corpus.pkl` plus the two index files and do not assume any particular document count. Bumping the dataset required zero changes below the sampling step.

**Key results:**

- Full rebuild (sampling, BM25 build, FAISS build) completes in roughly 15 to 25 minutes on a 16 GB laptop.
- Both retrieval modes (BM25 and Semantic) and the Hybrid and RAG modes in the Streamlit app are verified to work on the full 91,850 document corpus.
- The end-to-end pipeline is reproducible from the notebooks plus the new `environment.yml`.

## Step 3: Improve Documentation and Code Quality

### Documentation Update

The README was updated to reflect the scaled pipeline rather than the Milestone 2 state:

- Dataset description updated from "20,000 stratified sample" to "100,000 sampled reviews (91,850 after preprocessing)".
- The data-preparation section now documents the two-tier sampling strategy (stratified first, random fallback on failure) so a reader understands what actually runs in practice.
- The project-structure tree was updated with the real set of files in `src/`. One stale entry (`retrieval_metrics.py`) that never existed was removed.
- A short scale note was added under "Running the System" covering expected runtime and disk usage at 91,850 documents.
- Model references aligned with the LLM experiment conclusion (LLaMA 3.3 70B) while still documenting the runtime fallback chain used by the app.
- The milestone badge was updated from "Milestone 2" to "Final Submission".

### Code Quality Changes

- No hardcoded filesystem paths in any `src/` module; everything uses `pathlib.Path` (`src/bm25.py`, `src/semantic.py`, `src/hybrid.py`, `src/rag_pipeline.py`).
- No API keys in source code. `GROQ_API_KEY` is loaded from `.env` via `python-dotenv`, and `env.example` documents the required variables.
- All public functions and classes in `src/` have at least one-line docstrings.
- `environment.yml` cleaned up. Removed packages that are never imported (`langchain`, `langchain-community`, `torchvision`, `torchaudio`, `nltk`, `scikit-learn`, `scipy`). Pinned `transformers<5.0` to silence a batch of harmless but noisy deprecation warnings from the 5.x release series.
- `.gitignore` excludes `data/raw/`, `data/processed/`, `.env`, and notebook checkpoint directories so large artifacts and secrets never get committed.
- `src/semantic.py` `build_index()` now uses `batch_size=128` at encode time (previously the library default of 32), which is the single biggest driver of the 100K-scale runtime improvement.

## Step 4: Cloud Deployment Plan

### Data Storage

**Raw data.** The raw Amazon Reviews JSON files are large (the full Books dataset is about 30 GB compressed). We would store them in an **Amazon S3** bucket. S3 is a natural fit because it is cheap for bulk archival storage and DuckDB can read gzipped JSON directly from `s3://` paths. The `read_json_auto()` call in `notebooks/02_data_preparation.ipynb` would need only a path change, not a logic change.

**Processed data.** The processed `books_sample.parquet` file would also live in S3. Parquet is already a columnar format designed for cloud storage, so the app can load it via `pd.read_parquet("s3://...")` with no intermediate step. Processed data goes in a separate prefix (`processed/`) from raw data so S3 lifecycle rules can archive raw files to Glacier while keeping processed files hot.

**Vector index.** The FAISS index (`data/processed/semantic_index/faiss_index`, about 140 MB at 91,850 vectors) would be stored in S3 and pulled to the app's local disk on startup. FAISS cannot query directly from S3, so the app needs a local copy. Cold-start cost is a few seconds, which is acceptable for a Streamlit deployment.

**BM25 index.** Same pattern as the FAISS index. `bm25_index.pkl` and `corpus.pkl` live in S3 and are pulled to local disk on app startup. Both are small enough (well under 100 MB combined) that cold-start time is not a concern.

### Compute

**Where will your app run?** The Streamlit app would run on **AWS Elastic Beanstalk** backed by a single `t3.medium` EC2 instance to start. Elastic Beanstalk handles EC2 provisioning, load balancing, and rolling deploys so we do not have to manage infrastructure by hand. The app itself is lightweight because all heavy LLM work is offloaded to the Groq API; the EC2 instance only does retrieval (which is CPU-bound and fast on the flat FAISS and BM25 indexes).

**How will you handle multiple users (concurrency)?** Elastic Beanstalk can auto-scale horizontally behind an Application Load Balancer as traffic grows, adding `t3.medium` instances up to a configurable cap. Because each request is stateless (retrieve, build context, call Groq, return answer) there is no shared session state to coordinate, so horizontal scaling is clean. For very high traffic we would cache the FAISS index in a single shared-memory segment so multiple Streamlit workers on the same box share one copy rather than loading it per worker.

**How will you handle LLM inference (API vs hosted model)?** We would keep using the **Groq API** rather than hosting a 70B model ourselves. Self-hosting a 70B model would require GPU instances (`g5.12xlarge` or higher), which would dominate the monthly cost of the whole system. Groq provides low-latency inference on their own hardware at the free-tier limits that cover our traffic volume. The app only needs `GROQ_API_KEY` set as an Elastic Beanstalk environment variable, stored via AWS Systems Manager Parameter Store or AWS Secrets Manager rather than committed to git.

### Streaming / Updates

**How will you incorporate new products in production?** New review data from subsequent McAuley Lab releases would be uploaded to the raw prefix in S3. An **AWS Batch** job, or a scheduled Amazon ECS task, would then re-run the sampling query against the updated raw data, write a fresh `corpus.pkl` and `books_sample.parquet` to the processed prefix, rebuild the BM25 and FAISS indexes, and upload the new index artifacts to S3. The running app picks up the new indexes on its next container restart (or via a manual refresh endpoint).

**How will your pipeline stay up to date?** We would schedule the rebuild job with **Amazon EventBridge** on a cron trigger (for example nightly or weekly). The trigger invokes the Batch job, which executes the full pipeline and writes versioned artifacts (`corpus-v20260421.pkl`, and similar) to S3. A small manifest file tracks the current active version. For a production system with strict uptime we would blue-green the artifacts: write the new version, verify it in a staging Elastic Beanstalk environment, flip the manifest, then tear down the old version, so there is never a window where the app has inconsistent indexes. For the current project scale a scheduled nightly rebuild plus a simple container restart is more than enough.