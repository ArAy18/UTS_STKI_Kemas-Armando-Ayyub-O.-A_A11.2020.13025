# src/csv_to_txt.py
import csv, os, argparse

def csv_to_txt(csv_path, out_dir="data/raw", text_col="clean_content", max_docs=10):
    os.makedirs(out_dir, exist_ok=True)
    # csv_path boleh berupa local path atau URL (requests akan dibutuhkan untuk URL),
    # tapi kita gunakan built-in open untuk local; untuk URL gunakan wget/curl atau git raw link.
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        i = 0
        for row in reader:
            if i >= max_docs: break
            doc_id = row.get('no') or f"doc{i}"
            text = row.get(text_col) or " ".join(row.values())
            fname = os.path.join(out_dir, f"{doc_id}.txt")
            with open(fname, 'w', encoding='utf-8') as out:
                out.write(text.strip())
            i += 1
    print(f"Saved {i} documents to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--text_col", default="clean_content")
    parser.add_argument("--max", type=int, default=10)
    args = parser.parse_args()
    csv_to_txt(args.csv_path, args.out, args.text_col, args.max)
