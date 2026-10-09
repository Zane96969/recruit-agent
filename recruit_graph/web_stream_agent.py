import gradio as gr
from graph_tools import chat_model, retrieve_jd_info,analyze_jd_info
from langchain.agents import create_agent
from gradio import ChatMessage
from langchain_core.messages import AIMessage, ToolMessage
import json
from langgraph.checkpoint.memory import InMemorySaver


memory=InMemorySaver()
tools=[retrieve_jd_info,analyze_jd_info]
agent=create_agent(
    model=chat_model,
    tools=[retrieve_jd_info, analyze_jd_info],
    system_prompt=("你是招聘助手，有两个工具：retrieve_jd_info 检索公司已有的岗位JD库，analyze_jd_info 分析一段JD原文。先判断用户有没有直接给出JD原文：给了JD原文，必须调用 analyze_jd_info 做结构化分析；没给JD、只是询问岗位信息，才调用 retrieve_jd_info 检索。如果检索结果里确实没有，再如实说资料不足。回答简洁。"
                   ),checkpointer=memory
)
def stream_bot(history):
    user_text = history[-1]["content"]
    config = {"configurable":{"thread_id": "zhang-001"}}
    for chunk in agent.stream(
        {"messages": [("user", user_text)]},
        stream_mode="updates",
        config=config
    ):
        for node_update in chunk.values():
            for current_message in node_update["messages"]:
                if isinstance(current_message, AIMessage):
                    for tool_call in current_message.tool_calls:
                        history.append(ChatMessage(
                            role="assistant",
                            content=json.dumps(
                                tool_call["args"], ensure_ascii=False
                            ),
                            metadata={
                                "title": f"🛠️ 调用工具 {tool_call['name']}",
                                "status": "done",
                            },
                        ))
                        yield history
                    if current_message.content and not current_message.tool_calls:
                        history.append(ChatMessage(
                            role="assistant",
                            content=current_message.content,
                        ))
                        yield history
                if isinstance(current_message, ToolMessage):
                    tool_name = getattr(current_message, "name", "未知工具")
                    history.append(ChatMessage(
                        role="assistant",
                        content=str(current_message.content),
                        metadata={
                            "title": f"⚙️ 工具 {tool_name} 返回结果",
                            "status": "done",
                        },
                    ))
                    yield history
def add_user(user_text,history):
    return" " ,history + [{"role": "user", "content": user_text}]

theme = gr.themes.Soft(
    primary_hue="sky",
    neutral_hue="slate",
    radius_size="md",
    font=["Microsoft YaHei", "PingFang SC", "sans-serif"],
)

lock_light_js = """
function stripDark() {
  document.querySelectorAll('.dark').forEach(function (element) {
    element.classList.remove('dark');
  });
}
stripDark();
new MutationObserver(stripDark).observe(document.documentElement, {
  attributes: true, subtree: true, attributeFilter: ['class']
});
var originalMatchMedia = window.matchMedia.bind(window);
window.matchMedia = function (query) {
  if (query && query.indexOf('prefers-color-scheme') !== -1) {
    return originalMatchMedia('(prefers-color-scheme: light)');
  }
  return originalMatchMedia(query);
};
"""

with gr.Blocks(title="招聘Agent·思考过程可视化") as demo:
    gr.Markdown(
        """
        # 智能招聘助手
        粘贴 **岗位 JD**，我帮你解析成结构化信息；也可以直接问我岗位的薪资、要求、技术栈。
        """
    )
    chatbot=gr.Chatbot(show_label=False)
    with gr.Row():
        msg=gr.Textbox(placeholder="输入问题，回车发送",scale=8,
            show_label=False,)
        send_button=gr.Button("发送",
            variant="primary",
            scale=1,)
    examples = gr.Examples(
        examples=[
            "AI Agent 开发岗位的薪资范围是多少？",
            "Java 岗位要求几年工作经验？",
            "招聘专员主要负责哪些工作？",
        ],
        inputs=msg,
        label="试试这些问题（点击直接提问）",
    )
    msg.submit(add_user,[msg,chatbot],[msg,chatbot],queue=False).then(stream_bot,chatbot,chatbot)
    examples.load_input_event.then(
        add_user, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        stream_bot, chatbot, chatbot
    )
    send_button.click(add_user,[msg,chatbot],[msg,chatbot],queue=False).then(stream_bot,chatbot,chatbot)
    gr.Markdown("""
        <div style="text-align:center; color:#94a3b8; font-size:0.85em; margin-top:12px;">
        技术栈：LangChain · LangGraph · Chroma · BGE · Gradio ｜
        开源地址：<a href="https://github.com/Zane96969/recruit-agent" target="_blank">github.com/Zane96969/recruit-agent</a>
        </div>
        """)
   
demo.launch(theme=theme, js=lock_light_js)