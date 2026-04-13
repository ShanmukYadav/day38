# Day 38 — NLP Foundations: Embeddings, Polysemy & Semantic Similarity
**IIT Gandhinagar · Cohort 1 · Week 07 · Tuesday Assignment**

---

## Overview

This assignment explores two core NLP challenges using the **ShopSense E-Commerce Reviews** dataset:

1. **Word2Vec, Polysemy & Window-Size Effects (Q1)** — Understanding how static embeddings handle ambiguous words like *"cheap"*, building a context-based disambiguation system, and comparing how window size affects syntactic vs semantic capture.

2. **Semantic Similarity Across Representations (Q2)** — Computing cosine similarity between two semantically equivalent but lexically different reviews using BOW, TF-IDF, Word2Vec averaging, and Sentence-BERT, then analysing the *semantic gap* and how each method progressively closes it.

---

## Project Structure

```
day38/
├── day38_nlp_embeddings.ipynb   # Main notebook (all Q1 & Q2 solutions)
├── README.md                    # This file
├── window_pca_comparison.png    # Generated: PCA plot (window=2 vs window=10)
└── similarity_comparison.png    # Generated: bar chart of similarity methods
```

---

## Questions Covered

### Q1 — Word2Vec, Polysemy & Window Size
| Sub-task | What it does |
|---|---|
| **Q1a** | Trains Word2Vec on ShopSense reviews. Shows `cheap` gets ONE vector. Computes `cosine(cheap, affordable)` vs `cosine(cheap, flimsy)`. |
| **Q1b** | Builds a disambiguation system: given a sentence, determines whether `cheap` means *affordable* or *low-quality* using context word embeddings vs. sense anchor vectors. |
| **Q1c** | Trains two models (`window=2`, `window=10`), compares nearest neighbours, cosine sims, and PCA plots to illustrate syntactic vs semantic relationship capture. |

### Q2 — Similarity: BOW vs TF-IDF vs Word2Vec vs Sentence-BERT
| Sub-task | What it does |
|---|---|
| **Q2a** | Computes cosine similarity between Review A & B using all four methods. Identifies which correctly recognises their semantic equivalence. |
| **Q2b** | Walks through the exact token overlap for BOW, explaining why near-zero similarity is returned despite matching meaning. |
| **Q2c** | Explains the *semantic gap* and how each representation progressively closes it, from fully-open (BOW) to most-closed (Sentence-BERT). |

---

## Dataset

The notebook generates a synthetic **ShopSense** dataset (2,000 reviews) with:
- Reviews using `cheap` in both **affordable** and **low-quality** contexts
- Mixed-sentiment reviews (good camera/photos, bad battery)
- Columns: `review_id`, `review_text`, `rating`, `sentiment_label`, `category`

No external data file is required — the dataset is generated in-notebook via `generate_shopsense_reviews()`.

---

## Requirements

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk gensim sentence-transformers
```

### `requirements.txt`

```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
nltk>=3.8
gensim>=4.3
sentence-transformers>=2.6
```

> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~80 MB) on first run. Ensure you have an internet connection.

---

## How to Run

### 1. Clone / Navigate to the repo

```bash
cd week07/tuesday/
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk gensim sentence-transformers
```

### 3. Launch Jupyter

```bash
jupyter notebook day38_nlp_embeddings.ipynb
```

Or run all cells via CLI:

```bash
jupyter nbconvert --to notebook --execute day38_nlp_embeddings.ipynb --output day38_nlp_embeddings_executed.ipynb
```

---

## Expected Outputs

| Output | Description |
|---|---|
| Printed vocabulary vector | One 100-dim vector for `cheap` — demonstrating single-vector limitation |
| Cosine scores (Q1a) | `cosine(cheap, affordable)` vs `cosine(cheap, flimsy)` |
| Disambiguation table (Q1b) | Predicted sense (`affordable` / `low_quality`) for 6 test sentences |
| Window comparison table (Q1c) | Nearest neighbours for `cheap` under `window=2` vs `window=10` |
| `window_pca_comparison.png` | PCA 2-D plot of embeddings for both window sizes |
| Similarity results table (Q2a) | Cosine scores for BOW, TF-IDF, Word2Vec avg, Sentence-BERT |
| Token overlap analysis (Q2b) | Shared/unique tokens, Jaccard score, written explanation |
| Semantic gap narrative (Q2c) | Scored ASCII progress bars + explanation per method |
| `similarity_comparison.png` | Bar chart of all four similarity scores |

---

## Key Concepts

### Polysemy & Static Embeddings
Word2Vec assigns **one vector per word type**. The word *cheap* can mean:
- **affordable** → "cheap price, great value for money"
- **low-quality** → "cheap plastic, broke after a week"

The embedding is a weighted average of all contexts seen in training — both senses collapse into a single point, losing the distinction.

### Disambiguation Strategy
The system builds **anchor vectors** (average of sense-specific words), then computes the **context vector** of an input sentence (average of non-target word embeddings) and checks which anchor it's closest to via cosine similarity.

### Window Size
| Window | Captures | Example |
|---|---|---|
| `window=2` | **Syntactic** — immediately adjacent grammar words | determiners, prepositions near `cheap` |
| `window=10` | **Semantic** — broad topical co-occurrence | `affordable`, `budget`, `fragile`, `poor` |

### Semantic Gap
| Method | Gap | Reason |
|---|---|---|
| BOW | Fully open | Orthogonal sparse vectors, zero synonymy |
| TF-IDF | Barely reduced | Same space, just re-weighted |
| Word2Vec avg | Partially closed | Static distributional similarity |
| Sentence-BERT | Most closed | Contextual + NLI/STS fine-tuning |

---

## Engineering Standards

- ✅ All tasks split into named functions with docstrings  
- ✅ No hardcoded thresholds — constants defined at top of notebook  
- ✅ `try/except` on all model lookups and file I/O  
- ✅ Input validation (OOV checks, None guards on context vectors)  
- ✅ Readable variable names (`cos_affordable`, `anchor_vec`, `context_vec`)

---

## Git Commit Trail (suggested)

```
git commit -m "Day 38: Add ShopSense dataset generator and tokenization utils"
git commit -m "Day 38: Q1a — Word2Vec training and polysemy demonstration"
git commit -m "Day 38: Q1b — Context-based cheap disambiguation system"
git commit -m "Day 38: Q1c — Window size comparison with PCA visualisation"
git commit -m "Day 38: Q2 — BOW, TF-IDF, Word2Vec, SBERT similarity + semantic gap analysis"
```

---

*Week 07 · Tuesday · IIT Gandhinagar NLP Cohort 1*
