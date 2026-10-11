"""多智能体并行三专家：一份简历 + 目标岗位 JD，三个维度专家同时评估、再汇总。

流程（静态 fan-out / fan-in）：
                 ┌─▶ 技术栈专家 tech_expert ─────┐
    START ───────┼─▶ 薪资专家   salary_expert ───┼─▶ collect 汇总 ─▶ END
                 └─▶ 稳定性专家 stability_expert ┘

与 resume_multi_agent（串行）的选型区别：
    串行版：简历分析 → 人岗匹配有先后依赖（后一步要用前一步结果），只给简历即可自动找岗；
    本文件：技术栈 / 薪资 / 到岗稳定性三个维度【互不依赖】，对同一份简历 + JD
            同时开工（静态并行），总耗时≈最慢的一个专家而非三者相加。
    数量不定的批量任务（如 N 份简历做同样的初筛）应改用 Send 动态扇出，
    见 resume_batch_screen.py。

工程要点：
    多个并行节点同时写 state 的同一字段会冲突，parts 用
    Annotated[list, operator.add] 作为 reducer（归并器）累加三块结论；
    汇总节点只把零件拼成 final_report（普通字段，写一次），不再回写 parts，
    否则 reducer 会把整份零件再累加一遍导致重复。

运行：python recruit_graph/resume_parallel_agents.py
"""

import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

# 复用 graph_tools 里的聊天模型，不重复创建
from graph_tools import chat_model


# 状态黑板：resume / job_desc 是三个专家共享的输入；
# parts 带 reducer 累加各专家分结论；final_report 是汇总成品（普通字段）
class HireState(TypedDict):
    resume: str
    job_desc: str
    parts: Annotated[list, operator.add]
    final_report: str


def tech_expert(state: HireState) -> dict:
    """技术栈专家：只看技术与项目，给技术匹配度分数。"""
    print("-- 技术栈专家 开工 --")
    reply = chat_model.invoke([
        SystemMessage(content="你是技术栈评估专家，只从技术栈、项目经验角度对照岗位评估，3 句以内，并给技术匹配度分数(0-100)。"),
        HumanMessage(content=f"【岗位要求】\n{state['job_desc']}\n\n【候选人简历】\n{state['resume']}"),
    ])
    print("-- 技术栈专家 完成 --")
    return {"parts": [f"【技术栈评估】\n{reply.content}"]}


def salary_expert(state: HireState) -> dict:
    """薪资专家：只评估期望薪资与岗位预算是否匹配。"""
    print("-- 薪资专家 开工 --")
    reply = chat_model.invoke([
        SystemMessage(content="你是薪资评估专家，只评估候选人期望薪资与岗位预算是否匹配，3 句以内，明确是否在区间内。"),
        HumanMessage(content=f"【岗位要求】\n{state['job_desc']}\n\n【候选人简历】\n{state['resume']}"),
    ])
    print("-- 薪资专家 完成 --")
    return {"parts": [f"【薪资评估】\n{reply.content}"]}


def stability_expert(state: HireState) -> dict:
    """到岗稳定性专家：只看每周到岗天数、可持续时长、到岗速度。"""
    print("-- 稳定性专家 开工 --")
    reply = chat_model.invoke([
        SystemMessage(content="你是到岗稳定性评估专家，只评估每周到岗天数、可持续实习时长、到岗速度是否满足岗位，3 句以内。"),
        HumanMessage(content=f"【岗位要求】\n{state['job_desc']}\n\n【候选人简历】\n{state['resume']}"),
    ])
    print("-- 稳定性专家 完成 --")
    return {"parts": [f"【稳定性评估】\n{reply.content}"]}


def collect(state: HireState) -> dict:
    """汇总节点：三个专家都交卷后才运行（LangGraph 自动屏障等齐），拼成一份报告。"""
    print("== 汇总：三块结论已收齐 ==")
    return {"final_report": "\n\n".join(state["parts"])}


builder = StateGraph(HireState)
builder.add_node("tech", tech_expert)
builder.add_node("salary", salary_expert)
builder.add_node("stability", stability_expert)
builder.add_node("collect", collect)
# 静态扇出（fan-out）：START 同时派给三个互不依赖的专家
builder.add_edge(START, "tech")
builder.add_edge(START, "salary")
builder.add_edge(START, "stability")
# 扇入（fan-in）：三个专家都汇到 collect，自动等齐三块结论
builder.add_edge("tech", "collect")
builder.add_edge("salary", "collect")
builder.add_edge("stability", "collect")
builder.add_edge("collect", END)
app = builder.compile()


if __name__ == "__main__":
    sample_resume = (
        "张三，23岁，大连外国语大学软件工程本科，"
        "熟悉 Python、LangChain、LangGraph、RAG、Chroma 向量库，"
        "独立做过招聘 Agent 开源项目，可实习 6 个月、每周 5 天，"
        "期望岗位 AI Agent 开发实习生，期望薪资 3K-5K。"
    )
    sample_job = (
        "AI Agent 开发实习生：熟悉 Python，了解 LangChain/LangGraph，"
        "理解 RAG、Function Calling，有大模型应用项目经验优先；"
        "实习薪资 4K-6K，要求每周到岗 4 天以上、可持续 3 个月以上。"
    )

    start = time.time()
    result = app.invoke({
        "resume": sample_resume,
        "job_desc": sample_job,
        "parts": [],
    })
    print("\n===== 三专家并行评估报告 =====")
    print(result["final_report"])
    print(f"\n总耗时 {time.time() - start:.1f} 秒（并行≈最慢的一个专家，而非三者相加）")
