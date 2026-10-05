import sqlite3
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage
from pathlib import Path

# 复用 graph_tools 里现成的两个招聘工具和聊天模型，不重复创建
from graph_tools import retrieve_jd_info, analyze_jd_info, chat_model


# State：公共黑板；messages 用 add_messages 自动追加（不覆盖）
class RecruitState(TypedDict):
    messages: Annotated[list, add_messages]


# 主管模型：把两个真实招聘工具绑给它，由它自己判断该调哪个
supervisor_model = chat_model.bind_tools(
    [retrieve_jd_info, analyze_jd_info]
)


# 主管节点：把账本发给模型，再把模型回复（可能带 tool_calls）加回账本
def supervisor_node(state: RecruitState) -> dict:
    response = supervisor_model.invoke(state["messages"])
    return {"messages": [response]}


# 路由：看最后一条消息，有 tool_calls 去执行工具，没有就结束
def should_use_tools(state: RecruitState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# 预建工具执行工位：自动执行工具、包好 ToolMessage 回执
tool_node = ToolNode([retrieve_jd_info, analyze_jd_info])

# 搭图：START → 主管 →（条件边）工具 → 主管 回环
builder = StateGraph(RecruitState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", should_use_tools)
builder.add_edge("tools", "supervisor")

# 记忆：SqliteSaver 落本地文件、重启不丢；db 文件不进 git
db_path = Path(__file__).resolve().parent / "recruit_checkpoints.db"
conn = sqlite3.connect(str(db_path), check_same_thread=False)
checkpointer = SqliteSaver(conn)
app = builder.compile(checkpointer=checkpointer)

# 一个 thread_id = 一个招聘对话窗口
config = {"configurable": {"thread_id": "recruit-001"}}


if __name__ == "__main__":
    # 第一段：问已收录岗位薪资（走检索）
    result1 = app.invoke(
        {"messages": [HumanMessage(content="AI Agent 开发工程师的薪资是多少？")]},
        config=config,
    )
    print(result1["messages"][-1].content)

    # 第二段：追问城市，同一 thread 验证记忆
    result2 = app.invoke(
        {"messages": [HumanMessage(content="那这个岗位在哪个城市？")]},
        config=config,
    )
    print(result2["messages"][-1].content)

    # 第三段：给一份全新 JD 原文（走分析）；JD 要真正贴进消息里
    new_jd = (
        "岗位：前端开发工程师，地点：杭州，薪资：10K-15K，"
        "要求会 Vue、TypeScript。"
    )
    result3 = app.invoke(
        {"messages": [HumanMessage(content=f"帮我分析这份JD：\n{new_jd}")]},
        config=config,
    )
    print(result3["messages"][-1].content)
