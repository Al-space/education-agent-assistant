from langchain_community.document_loaders import PyPDFLoader,TextLoader,CSVLoader,Docx2txtLoader
import os

def load_document(file_path:str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.txt':
        loader = TextLoader(file_path,encoding='utf-8')
    elif ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext == '.docx':
        loader = Docx2txtLoader(file_path)
    elif ext == '.csv':
        loader = CSVLoader(file_path)
    else:
        raise ValueError("不支持该文件！")
    docs = loader.load()
    return docs