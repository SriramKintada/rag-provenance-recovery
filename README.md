# RAG Document Provenance Recovery

An inverse problem approach to recovering source documents from AI-generated answers in Retrieval Augmented Generation (RAG) systems.

## Overview

This project tackles the inverse problem of RAG systems: given an AI-generated answer, can we identify which source documents were used? I implemented and compared 4 different provenance recovery methods:

1. **Embedding Similarity** - Semantic matching using OpenAI embeddings
2. **TF-IDF Matching** - Keyword-based document similarity
3. **N-gram Overlap** - Exact phrase matching
4. **LLM Attribution** - Using GPT-4 to identify sources

## Results

On 3 ArXiv papers (220 chunks, 5 test queries):

| Method | Precision | Recall | F1 Score |
|--------|-----------|--------|----------|
| Embedding Similarity | 60% | 60% | **60%** |
| TF-IDF Matching | 24% | 24% | 24% |
| LLM Attribution | 37% | 20% | 26% |
| N-gram Overlap | 4% | 4% | 4% |

**Key Finding:** Semantic embeddings significantly outperform keyword-based methods for this task.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

Create a `.env` file:
```bash
OPENAI_API_KEY=your_api_key_here
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

### 3. Run the Pipeline

```bash
# Full pipeline with data collection
python main.py --num-queries 5

# Skip data collection (use existing data)
python main.py --skip-collection --num-queries 5
```

## Project Structure

```
rag-provenance-recovery/
├── src/
│   ├── data_collection.py    # ArXiv/Wikipedia fetching
│   ├── forward_rag.py         # RAG system implementation
│   ├── inverse_methods.py     # 4 provenance recovery methods
│   ├── evaluation.py          # Metrics and visualization
│   └── utils.py               # Helper functions
├── data/
│   ├── raw/                   # Source documents
│   └── metadata.json          # Document metadata
├── results/
│   ├── evaluation_results.json
│   └── figures/               # Performance plots
├── logs/                      # Execution logs
├── config.py                  # Configuration management
├── main.py                    # Main pipeline orchestrator
└── requirements.txt
```

## How It Works

### Phase 1: Data Collection
- Fetch documents from ArXiv and Wikipedia
- Chunk documents (1000 chars, 200 overlap)
- Generate embeddings using OpenAI's text-embedding-3-small

### Phase 2: Forward RAG (Ground Truth)
- For each query, retrieve top-5 most similar chunks
- Generate answer using GPT-4o-mini
- Record which chunks were used (ground truth)

### Phase 3: Inverse Recovery
- Given only the generated answer, apply each inverse method
- Predict which chunks were likely sources
- Compare predictions against ground truth

### Phase 4: Evaluation
- Compute Precision, Recall, F1 for each method
- Generate comparison visualizations
- Save detailed results to JSON

## Methods Explained

### Embedding Similarity
Embed the answer and all chunks in the same semantic space, then find nearest neighbors using cosine similarity. Works best because it captures meaning, not just keywords.

### TF-IDF Matching
Traditional information retrieval approach using term frequency-inverse document frequency. Struggles when LLM paraphrases.

### N-gram Overlap
Checks for exact phrase matches using trigrams and Jaccard similarity. Fails because LLMs synthesize rather than copy text.

### LLM Attribution
Ask GPT-4 to identify which chunks support the answer. Expensive and inconsistent, but sometimes catches nuanced connections.

## Use Cases

- **AI Transparency**: Verify where AI-generated content came from
- **Hallucination Detection**: Identify when AI makes unsupported claims
- **Automatic Citations**: Generate references for RAG systems
- **Debugging**: Trace why a RAG system gave a particular answer

## Cost Estimate

Running on 5 queries with 220 chunks:
- Embedding generation: ~$1.50
- Answer generation: ~$0.15
- Evaluation: ~$0.05
- **Total: ~$1.70 per run**

## Technical Details

### Models Used
- **Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
- **LLM**: GPT-4o-mini (128K context window)
- **Vector Store**: ChromaDB with cosine similarity

### Configuration
All parameters configurable via `config.py`:
- Chunk size and overlap
- Top-k retrieval count
- Temperature settings
- Batch sizes for embeddings

### Logging
Detailed execution logs in `logs/`:
- `data_collection.log` - Document fetching
- `forward_rag.log` - RAG system execution
- `evaluation.log` - Metrics computation

## Results

Full evaluation results available in `results/evaluation_results.json`. Includes:
- Per-query metrics for all methods
- Confusion matrices (TP, FP, FN)
- Average performance across queries
- Execution timestamps

Visualizations in `results/figures/`:
- `method_comparison.png` - Bar chart comparing all methods
- Additional analysis plots

## Limitations

- Small test set (5 queries, 3 papers)
- Limited to English text
- Depends on OpenAI API availability
- Binary matching (doesn't measure partial usage)
- Semantic drift when LLM adds reasoning

## Future Work

- Ensemble methods combining embedding + TF-IDF
- Attention-based attribution using open-source models
- Larger-scale evaluation (50+ queries, 20+ papers)
- Support for multi-language documents
- Real-time deployment as Chrome extension

## License

Academic project for Parameter Estimation course, IISER Pune.

## Author

Sriram Kintada
IISER Pune
sriram.kintada@students.iiserpune.ac.in
