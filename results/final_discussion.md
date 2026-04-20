# Final Discussion

## Step 1: Improve Your Workflow

### 1.1 Dataset Scaling

Our current pipeline already processes 20,000 products (reviews), so no changes are needed here. Both indexes have been built on this dataset:

- **Vector index (FAISS)**: `data/processed/semantic_index/faiss_index`
- **BM25 index**: `data/processed/bm25_index.pkl` (corpus stored in `data/processed/corpus.pkl`)

### 1.2 LLM Experiment

#### Models Compared

| | **LLaMA 3.3 70B** | **LLaMA 3.1 8B** |
| --- | --- | --- |
| Groq model ID | `llama-3.3-70b-versatile` | `llama-3.1-8b-instant` |
| Developer | Meta | Meta |
| Parameters | 70 billion | 8 billion |
| Training data | ~15 trillion tokens | ~15 trillion tokens |
| Context window | 128k tokens | 128k tokens |
| Release | December 2024 | July 2024 |
| Role | Primary/high-quality model | Small/fast comparison model |

Both models are run via the Groq API. The full comparison is in `notebooks/08_llm_comparison.ipynb`.

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

<!-- TODO: paste the 5-query side-by-side outputs from notebook 08 here -->

#### Which Model We Chose and Why

<!-- TODO: fill in after reviewing results -->

## Step 2: Additional Feature (state which option you chose)

### What You Implemented

- Description of the feature
- Key results or examples

## Step 3: Improve Documentation and Code Quality

### Documentation Update

- Summary of `README` improvements

### Code Quality Changes

- Summary of cleanups

## Step 4: Cloud Deployment Plan

### Data Storage

#### Raw Data

The raw Amazon Reviews JSON files are large (the full Books dataset is ~30 GB), so we would store them in an **S3 bucket**. S3 is a natural fit here because it is cheap for bulk storage and we can point DuckDB directly at it. In our current pipeline, DuckDB samples 20k reviews without loading the full file into memory, and that same query works against an S3 path with no real changes to the code.

#### Processed Data

The processed `books_sample.parquet` file would also live in S3. Parquet is already a columnar format built for cloud storage, so this is a straightforward lift-and-shift. Any downstream step that reads this file would just swap the local path for an S3 URI.

#### Vector Index

The FAISS index (`data/processed/semantic_index/faiss_index`) would be stored in S3 and downloaded to the app's local disk on startup. FAISS does not support querying directly from S3, so the app needs it locally. Since it is only around 100 MB for our dataset size, pulling it on boot is fast enough.

#### BM25 Index

Same story as the FAISS index. The `bm25_index.pkl` and `corpus.pkl` files go in S3 and get pulled to disk when the app starts up. Both are small enough that cold-start time should not be a problem.

### Compute

#### Where will your app run?

We would deploy the Streamlit app on **AWS Elastic Beanstalk**. Elastic Beanstalk is a good choice because it handles the underlying EC2 instance setup, load balancing, and auto-scaling without us having to manage any of that manually. We would point it at a `t3.medium` or similar instance. The app itself is lightweight since all the heavy LLM work is offloaded to the Groq API.

#### How will you handle multiple users (concurrency)?

Elastic Beanstalk can automatically spin up additional EC2 instances behind a load balancer when traffic increases. Since each request is stateless (retrieve, build context, call Groq API, return answer), horizontal scaling works cleanly here. We would not need anything fancier than that for a course project scale.

#### How will you handle LLM inference (API vs hosted model)?

We would keep using the **Groq API** rather than hosting our own model. Hosting a 70B model ourselves would require a GPU instance (expensive), whereas Groq gives us fast inference on their hardware for free at this scale. The app just needs a `GROQ_API_KEY` environment variable set in Elastic Beanstalk's configuration, which is straightforward.

### Streaming/Updates

#### How will you incorporate new products in production?

When new product data arrives, we would drop the updated raw JSON into the same S3 bucket and re-run the DuckDB sampling query to generate a fresh `books_sample.parquet`. From there, both the FAISS and BM25 indexes need to be rebuilt and re-uploaded to S3. This could be triggered manually or on a schedule.

#### How will your pipeline stay up to date?

We could set up a scheduled job (e.g. a cron on an EC2 instance or an AWS Lambda trigger) that periodically checks for new data in S3, rebuilds the indexes, and uploads the updated artifacts. The running Elastic Beanstalk app would then pick up the new indexes on its next restart. For a production system we would want zero-downtime index swaps, but for our use case a scheduled nightly rebuild would be more than sufficient.
