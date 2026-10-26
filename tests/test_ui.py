import gradio as gr

def chat_reply(message,history):
    # 阶段1：只做占位，还没接入Agent/RAG逻辑
    return f"收到你的消息：{message}，RAG/Agent尚未实现（阶段1演示界面）"

def create_ui():
    demo = gr.ChatInterface(
        fn=chat_reply,
        title="教育智能题库助手【阶段1演示】"
    )
    return demo

if __name__ == "__main__":
    ui = create_ui()
    ui.launch(server_name="0.0.0.0",server_port=7860)