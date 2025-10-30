import os, json
from glob import glob

PARSED = "parsed_texts"
OUT_JSON = "results/authors_corpus.json"
os.makedirs("results", exist_ok=True)

authors = {}
for author_dir in os.listdir(PARSED):
    path = os.path.join(PARSED, author_dir)
    if not os.path.isdir(path):
        continue
    texts = []
    for txt_file in glob(os.path.join(path, "*.txt")):
        with open(txt_file, "r", encoding="utf-8") as rf:
            t = rf.read().strip()
            if t:
                texts.append(t)
    authors[author_dir] = {
        "papers": texts,
        "combined": "\n".join(texts)
    }

with open(OUT_JSON, "w", encoding="utf-8") as wf:
    json.dump(authors, wf, indent=2)
print("Saved:", OUT_JSON)

