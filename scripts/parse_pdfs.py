import os
from tika import parser
from PyPDF2 import PdfReader
from tqdm import tqdm

DATASET_DIR = "dataset"
OUT_DIR = "parsed_texts"
os.makedirs(OUT_DIR, exist_ok=True)

def extract_text(pdf_path):
    """Try Tika first; if it fails, use PyPDF2 fallback."""
    try:
        parsed = parser.from_file(pdf_path)
        text = parsed.get("content") or ""
        if len(text.strip()) < 50:  # too short → fallback
            raise ValueError("Using PyPDF2 fallback")
    except Exception:
        try:
            reader = PdfReader(pdf_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            print(f"⚠️ Failed to read {pdf_path}: {e}")
            text = ""
    return text.strip()

def main():
    for root, _, files in os.walk(DATASET_DIR):
        for f in tqdm(files, desc=f"Parsing PDFs in {root}"):
            if not f.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(root, f)
            rel = os.path.relpath(root, DATASET_DIR)
            out_dir = os.path.join(OUT_DIR, rel)
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f.replace(".pdf", ".txt"))
            if os.path.exists(out_file):
                continue
            text = extract_text(pdf_path)
            with open(out_file, "w", encoding="utf-8") as wf:
                wf.write(text)

if __name__ == "__main__":
    main()

