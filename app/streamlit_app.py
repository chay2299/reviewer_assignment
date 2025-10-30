import sys, os, json
from pathlib import Path

# --- Fix import path ---
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# --- Imports ---
import streamlit as st
import tempfile
from sentence_transformers import SentenceTransformer
from scripts.match_reviewers import (
    get_top_k,
    add_reviewer_to_faiss,
    delete_reviewer_from_faiss,
    build_faiss_index,
)

# --- Constants ---
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
AUTHORS_PATH = os.path.join(ROOT_DIR, "models/author_embeddings.json")

# --- Streamlit page setup ---
st.set_page_config(page_title="Reviewer Recommendation System", layout="centered")
st.title("📄 Reviewer Recommendation System (FAISS Powered)")
st.caption("Find the most relevant reviewers for any research paper using semantic similarity search.")

# --- State management ---
if "results" not in st.session_state:
    st.session_state.results = None

# --- PDF Upload + K slider ---
uploaded = st.file_uploader("Upload a PDF file", type=["pdf"])
k = st.slider("Number of reviewers (K)", 1, 10, 5)

# --- Reload Button ---
if st.button("🔁 Reload Reviewers"):
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp.flush()
            st.info("Reloading reviewer data and recalculating matches ⏳...")
            st.session_state.results = get_top_k(tmp.name, k)
            os.unlink(tmp.name)
    else:
        st.warning("Please upload a PDF before reloading.")

# --- Process uploaded PDF if not loaded ---
if uploaded and not st.session_state.results:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        tmp.flush()
        st.info("Processing paper... please wait ⏳")
        st.session_state.results = get_top_k(tmp.name, k)
        os.unlink(tmp.name)

# --- Display Recommendations ---
if st.session_state.results:
    results = st.session_state.results
    if not results:
        st.error("⚠️ No text could be extracted from this PDF.")
    else:
        st.success("✅ Top-K Recommended Reviewers (via FAISS):")
        for name, score in results:
            st.markdown(f"**{name}** — similarity: `{score:.4f}`")

# ---------------------------------------
# 🧩 Reviewer Management Tools
# ---------------------------------------

st.divider()
st.subheader("🛠 Reviewer Management")

col1, col2, col3 = st.columns(3)

# --- Add Reviewer Button ---
with col1:
    if st.button("➕ Add Reviewer"):
        st.session_state.show_add = True
        st.session_state.show_delete = False
        st.session_state.show_reindex = False

# --- Delete Reviewer Button ---
with col2:
    if st.button("🗑️ Delete Reviewer"):
        st.session_state.show_delete = True
        st.session_state.show_add = False
        st.session_state.show_reindex = False

# --- Rebuild Index Button ---
with col3:
    if st.button("⚙️ Rebuild FAISS Index"):
        build_faiss_index()
        st.success("✅ FAISS index rebuilt successfully!")
        st.session_state.results = None

# --- Add Reviewer Form ---
if st.session_state.get("show_add", False):
    st.markdown("### ➕ Add a New Reviewer")
    with st.form("add_reviewer_form"):
        new_name = st.text_input("Reviewer Name")
        new_desc = st.text_area(
            "Reviewer Research Summary or Abstract",
            placeholder="Paste a few sentences describing their research interests...",
        )
        submitted = st.form_submit_button("Save Reviewer")

        if submitted:
            if not new_name.strip() or not new_desc.strip():
                st.warning("Please provide both name and description.")
            else:
                new_embed = MODEL.encode(new_desc).tolist()
                if os.path.exists(AUTHORS_PATH):
                    with open(AUTHORS_PATH, "r", encoding="utf-8") as f:
                        authors = json.load(f)
                else:
                    authors = {}
                authors[new_name] = new_embed
                with open(AUTHORS_PATH, "w", encoding="utf-8") as f:
                    json.dump(authors, f, indent=2)

                # Update FAISS index
                add_reviewer_to_faiss(new_name, new_embed)

                st.success(f"✅ Reviewer '{new_name}' added successfully!")
                st.cache_data.clear()
                st.session_state.results = None

# --- Delete Reviewer Form ---
if st.session_state.get("show_delete", False):
    st.markdown("### 🗑️ Delete a Reviewer")
    if os.path.exists(AUTHORS_PATH):
        with open(AUTHORS_PATH, "r", encoding="utf-8") as f:
            authors = json.load(f)
        all_reviewers = list(authors.keys())
        if all_reviewers:
            to_delete = st.selectbox("Select a reviewer to delete:", all_reviewers)
            if st.button("Confirm Delete"):
                del authors[to_delete]
                with open(AUTHORS_PATH, "w", encoding="utf-8") as f:
                    json.dump(authors, f, indent=2)
                # Rebuild FAISS index without deleted reviewer
                delete_reviewer_from_faiss(to_delete)
                st.success(f"🗑️ Reviewer '{to_delete}' deleted successfully!")
                st.cache_data.clear()
                st.session_state.results = None
        else:
            st.info("No reviewers found to delete.")
    else:
        st.info("No reviewer data available yet.")


