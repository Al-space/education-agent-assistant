from langchain_core.embeddings import Embeddings
from typing import List
import requests
from app.services.config import settings

class SiliconFlowEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.siliconflow.cn/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self.model, "input": texts}
        resp = requests.post(self.url, json=payload, headers=self.headers)
        resp.raise_for_status()
        res_data = resp.json()
        return [item["embedding"] for item in res_data["data"]]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

embedding_model = SiliconFlowEmbeddings(api_key=settings.SILICONFLOW_API_KEY,model=settings.SILICONFLOW_MODEL)
