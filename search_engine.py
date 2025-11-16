# src/search_engine.py
import argparse, json, math, re, os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

factory = StemmerFactory()
stemmer = factory.create_stemmer()
STOPWORDS = set(stopwords.words('indonesian'))

def simple_preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+',' ',text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+',' ', text).strip()
    toks = word_tokenize(text)
    toks = [t for t in toks if t not in STOPWORDS and len(t)>1]
    toks = [stemmer.stem(t) for t in toks]
    return toks

def load_indices(idx_dir="indexes"):
    with open(os.path.join(idx_dir,'doc_vecs.json'),encoding='utf-8') as f:
        doc_vecs = json.load(f)
    with open(os.path.join(idx_dir,'inverted.json'),encoding='utf-8') as f:
        inverted = json.load(f)
    with open(os.path.join(idx_dir,'idf.json'),encoding='utf-8') as f:
        idf = json.load(f)
    return doc_vecs, inverted, idf

def cosine_sim(qvec, dvec):
    num = 0.0
    for t, w in qvec.items():
        num += w * dvec.get(t, 0.0)
    denom_q = math.sqrt(sum(v*v for v in qvec.values()))
    denom_d = math.sqrt(sum(v*v for v in dvec.values()))
    if denom_q == 0 or denom_d == 0: return 0.0
    return num / (denom_q * denom_d)

def vsm_search(query, k=5, idx_dir="indexes"):
    doc_vecs, inverted, idf = load_indices(idx_dir)
    toks = simple_preprocess(query)
    qtf = {}
    for t in toks:
        qtf[t] = qtf.get(t,0)+1
    qvec = {t: qtf[t] * idf.get(t,1.0) for t in qtf}
    scores = []
    for doc_id, dvec in doc_vecs.items():
        s = cosine_sim(qvec, dvec)
        if s>0:
            scores.append((doc_id, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]

def boolean_search(query, k=10, idx_dir="indexes"):
    _, inverted, _ = load_indices(idx_dir)
    q = query.strip()
    if " AND " in q:
        parts = [p for p in q.split(" AND ")]
        sets = []
        for p in parts:
            toks = simple_preprocess(p)
            s = set()
            for t in toks:
                s.update([doc for doc,tf in inverted.get(t,[])])
            sets.append(s)
        res = set.intersection(*sets) if sets else set()
    elif " OR " in q:
        parts = [p for p in q.split(" OR ")]
        res = set()
        for p in parts:
            toks = simple_preprocess(p)
            for t in toks:
                res.update([doc for doc,tf in inverted.get(t,[])])
    elif q.startswith("NOT "):
        toks = simple_preprocess(q[4:])
        all_docs = set()
        for lst in inverted.values():
            for d, _ in lst:
                all_docs.add(d)
        exclude = set()
        for t in toks:
            exclude.update([d for d,_ in inverted.get(t,[])])
        res = all_docs - exclude
    else:
        toks = simple_preprocess(q)
        res = set()
        for t in toks:
            res.update([d for d,_ in inverted.get(t,[])])
    return [(d, 1.0) for i,d in enumerate(sorted(res))][:k]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vsm","boolean"], default="vsm")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--idx", default="indexes")
    args = parser.parse_args()
    if args.model == "vsm":
        res = vsm_search(args.query, args.k, args.idx)
    else:
        res = boolean_search(args.query, args.k, args.idx)
    print("Top results:")
    for doc,score in res:
        print(doc, score)
