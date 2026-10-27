from app.rag.vector_store import faiss_store
from app.rag.bm25_retriever import bm25_retriever

def add_documents_to_all(chunks):
    faiss_store.add_chunks(chunks)
    bm25_retriever.add_chunks(chunks)

def hybrid_seach(query:str,top_k=3):
    vec_res = faiss_store.search(query,top_k)
    bm25_res = bm25_retriever.search(query,top_k)

    all_res = []
    seen = set()
    for doc in vec_res + bm25_res:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            all_res.append(doc)
    return all_res[:top_k]