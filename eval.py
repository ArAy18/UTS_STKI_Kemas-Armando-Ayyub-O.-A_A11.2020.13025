# src/eval.py
import json, argparse
from src.search_engine import vsm_search  # jika menjalankan dari root, gunakan src.search_engine

# Jika import error, ubah jalur import sesuai struktur Anda:
# from search_engine import vsm_search

def precision_at_k(retrieved, relevant, k):
    retrieved_k = [d for d,_ in retrieved[:k]]
    rel = set(relevant)
    return sum(1 for d in retrieved_k if d in rel) / k

def average_precision(retrieved, relevant):
    rel = set(relevant)
    hits = 0
    sum_prec = 0.0
    for i,(d,_) in enumerate(retrieved, start=1):
        if d in rel:
            hits += 1
            sum_prec += hits / i
    if hits == 0: return 0.0
    return sum_prec / len(rel)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, help="JSON file: {query: [relevant_doc_ids]}")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    data = json.load(open(args.queries, encoding='utf-8'))
    precisions = []
    aps = []
    for q,relevant in data.items():
        retrieved = vsm_search(q, k=50)  # get top 50
        p = precision_at_k(retrieved, relevant, args.k)
        ap = average_precision(retrieved, relevant)
        precisions.append(p)
        aps.append(ap)
        print(f"Query: {q}\n Precision@{args.k}: {p:.3f}  AP: {ap:.3f}\n")
    print("Mean Precision@", args.k, sum(precisions)/len(precisions))
    print("MAP:", sum(aps)/len(aps))
