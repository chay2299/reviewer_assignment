
# import os, json
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from PyPDF2 import PdfReader

# MODEL = SentenceTransformer('all-MiniLM-L6-v2')
# AUTHORS_PATH = os.path.join(os.path.dirname(__file__), "../models/author_embeddings.json")

# def extract_text(pdf_path):
#     """Extract text from a PDF using PyPDF2."""
#     try:
#         reader = PdfReader(pdf_path)
#         text = "\n".join(page.extract_text() or "" for page in reader.pages)
#         return text.strip()
#     except Exception as e:
#         print(f"⚠️ PDF read failed: {e}")
#         return ""

# def load_author_embeddings():
#     """Reload reviewer embeddings from JSON every time."""
#     if not os.path.exists(AUTHORS_PATH):
#         return {}
#     with open(AUTHORS_PATH, "r", encoding="utf-8") as f:
#         return json.load(f)

# def get_top_k(pdf_path, k=5):
#     """Find top-K reviewers for a given paper."""
#     text = extract_text(pdf_path)
#     if not text:
#         return []
    
#     authors = load_author_embeddings()
#     if not authors:
#         return []

#     paper_embed = MODEL.encode(text)
#     author_names = list(authors.keys())
#     author_embeds = np.array(list(authors.values()))

#     sims = cosine_similarity([paper_embed], author_embeds)[0]
#     sorted_idx = np.argsort(sims)[::-1][:k]
#     results = [(author_names[i], float(sims[i])) for i in sorted_idx]

#     return results
import os, json, faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from PyPDF2 import PdfReader

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Paths
AUTHORS_JSON = os.path.join(os.path.dirname(__file__), "../models/author_embeddings.json")
FAISS_INDEX = os.path.join(os.path.dirname(__file__), "../models/reviewer_index.faiss")
FAISS_MAP = os.path.join(os.path.dirname(__file__), "../models/reviewer_map.json")

def extract_text(pdf_path):
    """Extract text from a PDF using PyPDF2."""
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip()
    except Exception as e:
        print(f"⚠️ PDF read failed: {e}")
        return ""

# -------------------------------
# FAISS-based storage management
# -------------------------------

def build_faiss_index():
    """Build FAISS index from JSON embeddings (for first-time use)."""
    if not os.path.exists(AUTHORS_JSON):
        print("No author_embeddings.json found.")
        return None, []

    with open(AUTHORS_JSON, "r", encoding="utf-8") as f:
        authors = json.load(f)

    names = list(authors.keys())
    vectors = np.array(list(authors.values()), dtype="float32")
    vectors = normalize(vectors, axis=1)  # normalize for cosine similarity

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product ≈ cosine similarity on normalized vectors
    index.add(vectors)

    faiss.write_index(index, FAISS_INDEX)
    with open(FAISS_MAP, "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2)

    return index, names


def load_faiss_index():
    """Load FAISS index if available, otherwise build one."""
    if os.path.exists(FAISS_INDEX) and os.path.exists(FAISS_MAP):
        index = faiss.read_index(FAISS_INDEX)
        with open(FAISS_MAP, "r", encoding="utf-8") as f:
            names = json.load(f)
    else:
        index, names = build_faiss_index()
    return index, names


def add_reviewer_to_faiss(name, embedding):
    """Add a new reviewer to FAISS index."""
    index, names = load_faiss_index()
    emb = np.array([embedding], dtype="float32")
    emb = normalize(emb, axis=1)
    index.add(emb)
    names.append(name)
    faiss.write_index(index, FAISS_INDEX)
    with open(FAISS_MAP, "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2)


def delete_reviewer_from_faiss(name):
    """Delete a reviewer from FAISS index."""
    if not (os.path.exists(FAISS_INDEX) and os.path.exists(FAISS_MAP)):
        return
    with open(FAISS_MAP, "r", encoding="utf-8") as f:
        names = json.load(f)
    if name not in names:
        return
    idx = names.index(name)
    names.pop(idx)
    # Reload all vectors except deleted one
    with open(AUTHORS_JSON, "r", encoding="utf-8") as f:
        authors = json.load(f)
    if name in authors:
        del authors[name]
    with open(AUTHORS_JSON, "w", encoding="utf-8") as f:
        json.dump(authors, f, indent=2)
    # rebuild FAISS
    build_faiss_index()


# -------------------------------
# Main search function
# -------------------------------

def get_top_k(pdf_path, k=5):
    """Find top-K reviewers for a given paper using FAISS."""
    text = extract_text(pdf_path)
    if not text:
        return []

    # Load FAISS index
    index, names = load_faiss_index()
    if index is None or len(names) == 0:
        print("❌ No reviewer embeddings found.")
        return []

    paper_emb = MODEL.encode(text)
    paper_emb = np.array([paper_emb], dtype="float32")
    paper_emb = normalize(paper_emb, axis=1)

    scores, ids = index.search(paper_emb, k)
    results = [(names[i], float(scores[0][j])) for j, i in enumerate(ids[0]) if i < len(names)]
    return results


