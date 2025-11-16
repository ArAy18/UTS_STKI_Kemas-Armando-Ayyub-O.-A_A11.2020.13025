# src/build_index.py
import os, math, json
from collections import defaultdict, Counter

def build_indices(processed_dir="data/processed", out_dir="indexes"):
    os.makedirs(out_dir, exist_ok=True)
    docs = {}
    for fname in sorted(os.listdir(processed_dir)):
        if not fname.endswith('.txt'): continue
        doc_id = fname.replace('.txt','')
        with open(os.path.join(processed_dir,fname), encoding='utf-8') as f:
            toks = f.read().split()
        docs[doc_id] = toks

    N = len(docs)
    df = defaultdict(int)
    tf = {}
    for doc_id, toks in docs.items():
        c = Counter(toks)
        tf[doc_id] = dict(c)
        for term in c:
            df[term] += 1

    idf = {t: math.log((N+1)/(df[t]+1)) + 1.0 for t in df}  # smoothed IDF
    doc_vecs = {}
    for doc_id, terms in tf.items():
        vec = {}
        for t, f in terms.items():
            vec[t] = f * idf.get(t, 0.0)
        doc_vecs[doc_id] = vec

    inverted = {t: [] for t in df}
    for t in inverted:
        for doc_id in tf:
            if t in tf[doc_id]:
                inverted[t].append((doc_id, tf[doc_id][t]))

    with open(os.path.join(out_dir,'inverted.json'),'w',encoding='utf-8') as f:
        json.dump(inverted,f,ensure_ascii=False,indent=2)
    with open(os.path.join(out_dir,'tf.json'),'w',encoding='utf-8') as f:
        json.dump(tf,f,ensure_ascii=False,indent=2)
    with open(os.path.join(out_dir,'idf.json'),'w',encoding='utf-8') as f:
        json.dump(idf,f,ensure_ascii=False,indent=2)
    with open(os.path.join(out_dir,'doc_vecs.json'),'w',encoding='utf-8') as f:
        json.dump(doc_vecs,f,ensure_ascii=False,indent=2)
    print("Indices saved to", out_dir)

if __name__ == "__main__":
    build_indices()
