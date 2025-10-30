import os, json, numpy as np, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ensure folders exist
os.makedirs("models", exist_ok=True)

# load author corpus
with open("results/authors_corpus.json", "r", encoding="utf-8") as f:
    authors = json.load(f)

# build document list (each paper text) and author labels
docs, doc_author = [], []
for author, val in authors.items():
    for paper in val.get("papers", []):
        docs.append(paper)
        doc_author.append(author)

# ---------------------- TF-IDF ----------------------
print("Building TF-IDF vectors ...")
tfidf = TfidfVectorizer(max_features=15000, stop_words="english")
X_tfidf = tfidf.fit_transform(docs)
joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")
joblib.dump(X_tfidf, "models/X_tfidf.npz")

# ------------------- SBERT Embeddings ----------------
print("Encoding with Sentence-BERT ...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs, show_progress_bar=True, batch_size=16)
np.save("models/doc_embeddings.npy", embeddings)

# compute author-level embedding (average of their papers)
author_embs = {}
for author in set(doc_author):
    idxs = [i for i, a in enumerate(doc_author) if a == author]
    author_embs[author] = np.mean([embeddings[i] for i in idxs], axis=0).tolist()

with open("models/author_embeddings.json", "w", encoding="utf-8") as wf:
    json.dump(author_embs, wf, indent=2)

print(" Saved TF-IDF + SBERT embeddings in 'models/'")

