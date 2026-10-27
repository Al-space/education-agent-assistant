import gradio as gr
import requests

BACKEND_URL = "http://127.0.0.1:8000"

def upload_knowledge(file):
    if file is None:
        return "请选择文件上传！"
    files = {"file": open(file, "rb")}
    resp = requests.post(f"{BACKEND_URL}/rag/upload", files=files)
    return resp.json()["msg"]

def rag_query(message, history):
    resp = requests.get(f"{BACKEND_URL}/rag/search", params={"q":message})
    data = resp.json()
    result_text = "检索结果：\n"
    for idx, item in enumerate(data["retrieval_result"]):
        result_text += f"\n【片段{idx+1}】来源：{item['source']}\n{item['content']}"
    return result_text

def create_ui():
    with gr.Blocks(title="教育智能题库助手【阶段2 RAG检索】") as demo:
        gr.Markdown("# 知识库上传 + RAG检索测试")
        file_input = gr.File(label="上传PDF/TXT知识库文档")
        upload_btn = gr.Button("上传并入库")
        upload_msg = gr.Textbox(label="上传结果")
        upload_btn.click(upload_knowledge, inputs=[file_input], outputs=upload_msg)

        gr.Markdown("## 检索问答（只返回检索片段，尚未调用LLM生成答案）")
        # 修复警告：增加 type="tuples", allow_tags=False
        chatbot = gr.Chatbot(type="tuples", allow_tags=False)
        msg = gr.Textbox(label="输入查询问题")
        submit_btn = gr.Button("检索")
        def submit_msg(text, hist):
            res = rag_query(text, hist)
            hist.append((text, res))
            return "", hist
        submit_btn.click(submit_msg, inputs=[msg, chatbot], outputs=[msg, chatbot])
    return demo

if __name__ == "__main__":
    ui = create_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860)