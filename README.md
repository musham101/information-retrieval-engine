# Information Retrieval System (TF-IDF, BM25 & CLI Search Engine)

This repository contains a complete **Information Retrieval (IR) System** implemented for **Assignment 3** of Information Retrieval & Text Mining.
The system supports **TF-IDF Vector Space Search**, **BM25 Ranking**, and a **Command-Line Interface (CLI)** to interactively retrieve relevant news articles.

---

# Features

### **Document Preprocessing**

* Combines *Heading* + *Article* into a single searchable text field
* Lowercasing
* Simple whitespace tokenization

### **Vector Space Model (TF-IDF)**

* Uses Scikit-Learn `TfidfVectorizer`
* Cosine similarity scoring
* Stopword removal (`english`)
* Document frequency filtering (`max_df`, `min_df`)

### **BM25 Ranking**

* Implemented using `rank_bm25` (BM25Okapi)
* Tokenized corpus
* Score computation for each document

### **Command-Line Interface (CLI)**

Search documents directly from the terminal:

```
python main.py --user_query "inflation impact" --top_k 5 --document_path "dataset/Articles.csv" --model_type bm25
```

### **Formatted Output**

* Shows rank
* Heading
* Score
* First 200 characters of article

---

# Project Structure

```
.
├── main.py
├── dataset/
│   └── Articles.csv
├── requirements.txt
└── README.md
```

---

# Installation

### **1. Create a virtual environment** (recommended)

```
python -m venv ir_env
source ir_env/bin/activate   # Mac/Linux
ir_env\Scripts\activate      # Windows
```

### **2. Install required packages**

Your repo includes a requirements file:

```
pip install -r requirements.txt
```

---

# Running the IR System

Run the script using:

```
python main.py --user_query "<your query>" --top_k <number> --document_path "<path-to-csv>" --model_type <vector_space|bm25>
```

---

## **Example 1 — TF-IDF Search**

```
python main.py \
  --user_query "impact of inflation on markets" \
  --top_k 5 \
  --document_path "dataset/Articles.csv" \
  --model_type vector_space
```

---

## **Example 2 — BM25 Search**

```
python main.py \
  --user_query "government policy" \
  --top_k 5 \
  --document_path "dataset/Articles.csv" \
  --model_type bm25
```

---

# How It Works

### **1. Data Loading**

CSV is read using:

```python
pd.read_csv(file_path, encoding="Windows-1252")
```

### **2. Text Combination**

We merge heading + article:

```python
row['Heading'] + " " + row['Article']
```

### **3. TF-IDF Ranking**

* Build TF-IDF matrix using `TfidfVectorizer`
* Compute cosine similarity with the query
* Retrieve top-k documents

### **4. BM25 Ranking**

* Tokenize all documents
* Compute BM25 score for each document
* Retrieve top-k documents

---

# Example Output

```
Rank 1:
Heading: asian markets retreat as china fed feed cauti
Score: 17.0815
NewsType: business

Article: Hong Kong: A mixed reading on Chinese inflation Thursday kept Asian equities traders on edge in fresh volatility Thursday as markets retreated from a two-day rally, while fears of a US interest rate h...
--------------------------------------------------------------------------------
```

---

# Requirements

Your `requirements.txt` is used for dependency installation.
Typical packages include:

```
pandas
numpy
scikit-learn
rank-bm25
argparse
```

(If your file has more, they will be installed automatically.)

---

# Notes

* CSV must include the following columns:

  * **Heading**
  * **Article**
  * **Date**
  * **NewsType**

* Encoding is set to `"Windows-1252"` because many datasets contain special characters.

* BM25 expects **flat token lists**, e.g., `['inflation', 'market']`.
