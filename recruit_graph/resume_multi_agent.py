"""多智能体串行双专家：简历分析专家 → 人岗匹配专家。

流程（Supervisor 编排）：
    START → 主管 supervisor ──派单──▶ 工具 experts（分析/匹配）
                    ▲                    │
                    └───── 结果回流 ──────┘
    两个专家都完成后，主管输出自然语言总结 → END

与 recruit_mcp 的区别：
    MCP 版 match_resume_jd 需要调用方同时给出简历和 JD 原文；
    本文件的匹配专家通过 InjectedState 自动读取上一步的画像，
    并主动从向量库检索 Top-K JD，用户只给一份简历即可完成“自动找岗 + 匹配度报告”。

运行：python recruit_graph/resume_multi_agent.py
"""

from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage

# 复用 graph_tools 里的聊天模型和两个专家工具，不重复创建
from graph_tools import chat_model, analyze_resume_worker, match_jd_worker


# State：公共黑板，messages 用 add_messages 自动追加（不覆盖）
class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]


# 总指挥的规矩：固定两步串行，模型只决定“调谁”，不负责搬运画像数据
supervisor_system = SystemMessage(content=(
    "你是招聘多智能体系统的总指挥，名下有两个专家工具：\n"
    "1. analyze_resume_worker：简历分析专家，把简历原文变成结构化画像；\n"
    "2. match_jd_worker：人岗匹配专家，自动读取画像、检索岗位库并给出匹配度报告。\n"
    "严格按固定流程执行，不允许跳步：\n"
    "第一步，用户发来简历后，立刻调用 analyze_resume_worker，并把简历原文传入；\n"
    "第二步，看到画像回传后，立刻无参数调用 match_jd_worker"
    "（工具会自动读取画像，你不要自己传画像内容）。\n"
    "两个工具都完成前，禁止输出最终结论、禁止结束；一次最多调用一个工具。\n"
    "两个工具都完成后，再用自然语言向用户总结：最匹配的岗位、匹配度、匹配点与差距建议。"
))

# 主管模型：绑定两个专家，由它按规矩依次派单
supervisor_model = chat_model.bind_tools(
    [analyze_resume_worker, match_jd_worker]
)


# 主管节点：带上总指挥规矩 + 全部历史消息一起发给模型
def supervisor_node(state: MultiAgentState) -> dict:
    response = supervisor_model.invoke(
        [supervisor_system] + state["messages"]
    )
    return {"messages": [response]}


# 路由：最后一条消息带 tool_calls 就去执行专家，否则结束
def route_after_supervisor(state: MultiAgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# 专家工位：ToolNode 自动执行工具，并把结果包成 ToolMessage 写回黑板
tool_node = ToolNode([analyze_resume_worker, match_jd_worker])


# 搭图：START → 主管 → 专家 → 主管 回环；主管不再派单则 END
builder = StateGraph(MultiAgentState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {"tools": "tools", END: END},
)
builder.add_edge("tools", "supervisor")
app = builder.compile()

# 一个 thread_id = 一次简历匹配会话
config = {"configurable": {"thread_id": "resume-match-001"}}


if __name__ == "__main__":
    # 一份待匹配的简历原文（拟真样例）
    resume_text = (
        "张三，23岁，大连外国语大学软件工程本科，"
        "掌握 Python、LangChain、RAG、Chroma 向量库，"
        "目标岗位 AI Agent 开发工程师，期望薪资 3K-5K（实习）。"
    )

    # stream 分步直播：主管派单 / 专家回传 / 最终总结
    final_answer = ""
    for step, update in enumerate(
        app.stream({"messages": [HumanMessage(content=resume_text)]}, config),
        start=1,
    ):
        for state_update in update.values():
            for message in state_update.get("messages", []):
                kind = message.__class__.__name__
                if kind == "AIMessage" and message.tool_calls:
                    names = [tc["name"] for tc in message.tool_calls]
                    print(f"第{step}步  总指挥派给：{names}")
                elif kind == "ToolMessage":
                    print(f"第{step}步  [{message.name}] 回传：{message.content[:120]}……")
                elif kind == "AIMessage":
                    final_answer = message.content

    print("\n===== 主管最终总结 =====")
    print(final_answer)
