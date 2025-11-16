# src/preprocess.py
import os, re, argparse
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

factory = StemmerFactory()
stemmer = factory.create_stemmer()
INDO_STOPWORDS = set(stopwords.words('indonesian'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+',' ',text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+',' ', text).strip()
    return text

def preprocess_file(in_path, out_path):
    with open(in_path, encoding='utf-8') as f:
        txt = f.read()
    txt = clean_text(txt)
    toks = word_tokenize(txt)
    toks = [t for t in toks if t not in INDO_STOPWORDS and len(t)>1]
    toks = [stemmer.stem(t) for t in toks]
    with open(out_path, 'w', encoding='utf-8') as out:
        out.write(" ".join(toks))

def main(input_dir="data/raw", output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith('.txt'): continue
        preprocess_file(os.path.join(input_dir,fname), os.path.join(output_dir,fname))
    print("Preprocessing done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/processed")
    args = parser.parse_args()
    main(args.input, args.output)
