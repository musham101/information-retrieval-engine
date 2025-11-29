import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def read_data(file_path):
    df = pd.read_csv(file_path, encoding="Windows-1252")
    return df

def tokenize_documents(corpus):
    tokenized_corpus = []
    for doc in corpus:
        tokens = doc.lower().split()
        tokenized_corpus.append(tokens)
    return tokenized_corpus

def combine_text_columns(row):
    heading = str(row.get("Heading", ""))
    article = str(row.get("Article", ""))
    return heading + " " + article

def display_results(results):
    for i, row in results.iterrows():
        print(f"Rank {i+1}:")
        print(f"Heading: {row['Heading']}")
        print(f"Score: {row['score']:.4f}")
        print(f"NewsType: {row['NewsType']}")
        print()
        print(f"Article: {row['Article'][:200]}...")
        print("-" * 80)

def tfidf_search(vectorizer, query, top_k=5):
    if query:
        query_vec = vectorizer.transform([query])
        
        sim_scores = cosine_similarity(query_vec, doc_tfidf).ravel()
        
        top_idx = np.argsort(sim_scores)[::-1][:top_k]
        
        results = df.loc[top_idx, ["Heading", "Article", "Date", "NewsType"]].copy()
        results["score"] = sim_scores[top_idx]
        
        return results.reset_index(drop=True)
    return pd.DataFrame()

def bm25_search(bm25, query, top_k=5):
    if query:
        query_tokens = query.lower().split()
        
        scores = bm25.get_scores(query_tokens)
        
        top_idx = np.argsort(scores)[::-1][:top_k]
        
        results = df.loc[top_idx, ["Heading", "Article", "Date", "NewsType"]].copy()
        results["score"] = scores[top_idx]
        
        return results.reset_index(drop=True)
    return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_query", type=str, required=True, help="User query string")
    parser.add_argument("--top_k", type=int, required=True, help="key number of documents to retrieve")
    parser.add_argument("--document_path", type=str, required=True, help="Path to the document CSV file")
    parser.add_argument("--model_type", type=str, required=True, help="Type of IR model to use: 'vector_space' or 'bm25'")
    args = parser.parse_args()

    df = read_data(args.document_path)
    df["doc_text"] = df.apply(combine_text_columns, axis=1)
    doc_text = df["doc_text"].fillna("").tolist()

    if args.model_type == "vector_space":
        vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=2)
        doc_tfidf = vectorizer.fit_transform(doc_text)

        result = tfidf_search(vectorizer, args.user_query, top_k=args.top_k)

        display_results(result)
    elif args.model_type == "bm25":
        tokenized_doc = tokenize_documents(doc_text)
        bm25 = BM25Okapi(tokenized_doc)

        result = bm25_search(bm25, args.user_query, top_k=args.top_k)
        
        display_results(result)
    else:
        print("Invalid model type. Please choose 'vector_space' or 'bm25'.")