from fastapi import APIRouter,UploadFile,File
import os
from app.rag.document_loader import load_document
from app.rag.text_splitter import split_documents
from app.rag.hybrid_retriever import add_documents_to_all,hybrid_seach

router = APIRouter(prefix="/rag",tags=["RAG知识库"])
UPLOAD_FILE = "data/uploads"
os.makedirs(UPLOAD_FILE,exist_ok=True)

@router.post("/upload")
async def upload_file(file:UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_FILE,file.filename)
    with open(save_path,"wb") as f:
        f.write(await file.read())

    docs = load_document(save_path)
    chunks = split_documents(docs)
    add_documents_to_all(chunks)
    return {"msg": "文件上传并向量化入库成功", "chunk_count": len(chunks)}

@router.get("/search")
async def rag_search(q:str,top_k:int = 3):
    results = hybrid_seach(q,top_k)
    res_list = []
    for doc in results:
        res_list.append({
            "content":doc.page_content,
            "source":doc.metadata.get("source","")
        })
    return {"query":q,"retrieval_result":res_list}
