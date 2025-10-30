# reviewer_assignment

This project recommends the most suitable reviewers for a research paper 
based on semantic similarity using transformer-based embeddings.

### 🧠 Features
- Uses `allenai-specter2` model for scientific text embeddings.
- FAISS for efficient similarity search.
- Add / Delete reviewers dynamically.
- Works directly from PDF uploads.

### 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
