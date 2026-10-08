import gradio as gr
from graph_tools import chat_model, retrieve_jd_info, analyze_jd_info
from langchain.agents import create_agent

tools = [retrieve_jd_info, analyze_jd_info]
agent = create_agent(
    model=chat_model,
    tools=[retrieve_jd_info, analyze_jd_info],
    system_prompt=(
        "你是招聘助手，有两个工具：retrieve_jd_info 检索公司已有的岗位JD库，analyze_jd_info 分析一段JD原文。"
        "先判断用户有没有直接给出JD原文：给了JD原文，必须调用 analyze_jd_info 做结构化分析；"
        "没给JD、只是询问岗位信息，才调用 retrieve_jd_info 检索。"
        "禁止凭记忆回答，也不要说自己没有这个功能。"
        "如果检索结果里确实没有或明显不相关，再如实说资料不足，不要硬套。回答简洁。"
    ),
)


def chat(message, history):
    messages = history + [{"role": "user", "content": message}]
    result = agent.invoke({"messages": messages})

    for msg in result["messages"]:
        tool_name = getattr(msg, "name", "")
        print("消息类型：", type(msg).__name__, "工具类型", tool_name)
    return result["messages"][-1].content


demo = gr.ChatInterface(
    fn=chat,
    title="招聘助手网页版",
)
demo.launch()
