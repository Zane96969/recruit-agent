import sqlite3
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage
from pathlib import Path

# 复用 graph_tools 里现成的两个招聘工具和聊天模型，不重复创建
from graph_tools import retrieve_jd_info, analyze_jd_info, chat_model


# State：公共黑板；messages 用 add_messages 自动追加（不覆盖）
class RecruitState(TypedDict):
    messages: Annotated[list, add_messages]
    pending_approval: str


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


# 人工审批节点：工具执行前暂停，把"要调的工具 + 参数"给人确认
def approval_node(state: RecruitState) -> dict:
    last_message = state["messages"][-1]
    request_form = {
        "即将执行的动作": [
            {"工具": tool_call["name"], "参数": tool_call["args"]}
            for tool_call in last_message.tool_calls
        ],
        "提示": "确认请回复“批准”，否则回复“拒绝”",
    }
    human_decision = interrupt(request_form)
    return {"pending_approval": human_decision}


# 审批后路由：批准 → 执行工具；拒绝 → 直接结束
def route_after_approval(state: RecruitState) -> str:
    if state.get("pending_approval") == "批准":
        return "tools"
    return END


# 搭图：START → 主管 → 审批 →（批准）工具 → 主管 回环；拒绝则结束
builder = StateGraph(RecruitState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("approval", approval_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    should_use_tools,
    {"tools": "approval", END: END},
)
builder.add_conditional_edges("approval", route_after_approval)
builder.add_edge("tools", "supervisor")

# 记忆：SqliteSaver 落本地文件、重启不丢；db 文件不进 git
db_path = Path(__file__).resolve().parent / "recruit_checkpoints.db"
conn = sqlite3.connect(str(db_path), check_same_thread=False)
checkpointer = SqliteSaver(conn)
app = builder.compile(checkpointer=checkpointer)

# 一个 thread_id = 一个招聘对话窗口
config = {"configurable": {"thread_id": "recruit-approval-001"}}


if __name__ == "__main__":
    # 一份待分析的新 JD 原文
    new_jd = (
        "岗位：前端开发工程师，地点：杭州，薪资：10K-15K，"
        "要求会 Vue、TypeScript。"
    )

    # 第一次 invoke：提交，图在“人工审批”处暂停（工具还没执行）
    first = app.invoke(
        {"messages": [HumanMessage(content=f"帮我分析这份JD：\n{new_jd}")]},
        config=config,
    )
    print("暂停，等待审批：")
    print(first.get("__interrupt__"))

    # 第二次 invoke：人“批准”后继续，工具执行、给出分析
    # 把“批准”改成“拒绝”，则不执行工具、直接结束
    final = app.invoke(Command(resume="批准"), config=config)
    print(final["messages"][-1].content)
