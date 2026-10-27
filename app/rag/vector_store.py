import faiss
import numpy as np
import pickle
import os
from app.rag.embedding import embedding_model

INDEX_DIR = "data/indexes"
INDEX_PATH = os.path.join(INDEX_DIR,"faiss.index")
METADATA_PATH = os.path.join(INDEX_DIR,"index_meta.pkl")

os.makedirs(INDEX_DIR,exist_ok=True)

class FAISSStore:
    def __init__(self):
        self.index = None
        self.chunks_meta = []
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH,"rb") as f:
                self.chunks_meta = pickle.load(f)

    def add_chunks(self, chunks):
        texts = [c.page_content for c in chunks]
        embeds = embedding_model.embed_documents(texts)
        embeds_np = np.array(embeds).astype(np.float32)
        # 首次入库时按实际向量维度初始化索引，避免固定维度与模型不匹配
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeds_np.shape[1])
        if embeds_np.shape[1] != self.index.d:
            raise ValueError(
                f"向量维度不一致：索引为 {self.index.d}，当前模型输出 {embeds_np.shape[1]}，请更换模型后重建索引"
            )
        self.index.add(embeds_np)
        self.chunks_meta.extend(chunks)
        self.save()

    def search(self, query: str, top_k=3):
        if self.index is None or not self.chunks_meta:
            return []
        embed = embedding_model.embed_query(query)
        embed_np = np.array([embed]).astype(np.float32)
        scores, idx_list = self.index.search(embed_np, top_k)
        result = []
        for i in idx_list[0]:
            if 0 <= i < len(self.chunks_meta):
                result.append(self.chunks_meta[i])
        return result

    def save(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(self.chunks_meta, f)

faiss_store = FAISSStore()