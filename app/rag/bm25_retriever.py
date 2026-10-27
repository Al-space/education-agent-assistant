from rank_bm25 import BM25Okapi
import jieba

class BM25Retriever:
    def __init__(self):
        self.corpus = []
        self.chunks = []
        self.bm25 = None

    def add_chunks(self,chunks):
        for chunk in chunks:
            tokenized = jieba.lcut(chunk.page_content)
            self.corpus.append(tokenized)
            self.chunks.append(chunk)
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, top_k=3):
        if not self.bm25:
            return []
        tokens = jieba.lcut(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.chunks[i] for i in top_idx]

bm25_retriever = BM25Retriever()