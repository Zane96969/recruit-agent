import json

import gradio as gr
from gradio import ChatMessage
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from graph_tools import analyze_jd_info, chat_model, retrieve_jd_info

memory = InMemorySaver()

agent = create_agent(
    model=chat_model,
    tools=[retrieve_jd_info, analyze_jd_info],
    checkpointer=memory,
    system_prompt=(
        "你是招聘助手，有两个工具：retrieve_jd_info 检索公司已有的岗位JD库，"
        "analyze_jd_info 分析一段JD原文。先判断用户有没有直接给出JD原文："
        "给了JD原文，必须调用 analyze_jd_info 做结构化分析；"
        "没给JD、只是询问岗位信息，才调用 retrieve_jd_info 检索。"
        "如果检索结果里确实没有，再如实说资料不足。回答简洁。"
    ),
)


def stream_bot(history):
    user_text = history[-1]["content"]
    config = {"configurable": {"thread_id": "web-thread-1"}}
    for chunk in agent.stream(
        {"messages": [("user", user_text)]},
        config=config,
        stream_mode="updates",
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


def add_user(user_text, history):
    return "", history + [{"role": "user", "content": user_text}]


with gr.Blocks(title="招聘Agent·思考过程可视化") as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="输入问题，回车发送")
    msg.submit(
        add_user, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        stream_bot, chatbot, chatbot
    )


demo.launch()
