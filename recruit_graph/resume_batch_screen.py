"""多智能体动态扇出（Send）：N 份简历批量初筛，自动复制 N 个筛选专家并行评估。

流程（动态 fan-out / fan-in）：
    START → dispatch（按简历份数生成一串 Send）
              ├─ Send ▶ 筛选专家·候选人1 ┐
              ├─ Send ▶ 筛选专家·候选人2 ┼─▶ collect 汇总 ─▶ END
              └─ Send ▶ 筛选专家·候选人N ┘
    运行时简历有几份就复制几个专家，改名单不用改图。

与 resume_parallel_agents（静态并行）的选型区别：
    静态并行：几个【不同维度】的不同专家、数量在写代码时固定（技术/薪资/稳定性）；
    本文件：【同一种】初筛处理套到【数量不定】的 N 份简历上，用 Send 动态扇出。
    口诀：不同的活、固定几摊用静态；同样的活、不知道几份用 Send。

工程要点：
    dispatch 是条件边函数，返回 Send 列表；Send("screener", 局部小包) 把每份
    简历的独立数据发给一个专家副本，worker 只收到该小包（参数注解写 dict），
    return 的结果经 screens 的 reducer（operator.add）累加回全局状态。
    注意名单永远是 list，哪怕只有一份简历也要包成 ["..."]。

运行：python recruit_graph/resume_batch_screen.py
"""

import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import HumanMessage, SystemMessage

# 复用 graph_tools 里的聊天模型，不重复创建
from graph_tools import chat_model


class BatchState(TypedDict):
    job_desc: str                          # 目标岗位 JD（所有筛选专家共享）
    resumes: list[str]                     # 待初筛简历名单（数量不定）
    screens: Annotated[list, operator.add]  # 每份简历的初筛结论：reducer 累加
    final_report: str                      # 汇总成品


def dispatch(state: BatchState):
    """简历有几份就开几张 Send 单子，都派给同一个 screener 节点（运行时自动复制）。"""
    return [
        Send("screener", {"index": i + 1, "resume": text, "job_desc": state["job_desc"]})
        for i, text in enumerate(state["resumes"])
    ]


def screener(state: dict) -> dict:
    """筛选专家：只收到 Send 发来的【单份简历】小包，做一次快速初筛。"""
    index = state["index"]
    print(f"-- 筛选专家 开工：候选人{index} --")
    reply = chat_model.invoke([
        SystemMessage(content=(
            "你是简历初筛专家，依据岗位要求对单份简历做快速初筛："
            "先给结论（建议推进 / 待定 / 不建议），再给 0-100 匹配分，"
            "最后一句话说明主要理由，总共不超过 3 句。"
        )),
        HumanMessage(content=(
            f"【岗位要求】\n{state['job_desc']}\n\n"
            f"【候选人{index}简历】\n{state['resume']}"
        )),
    ])
    print(f"-- 筛选专家 完成：候选人{index} --")
    return {"screens": [f"候选人{index}：{reply.content}"]}


def collect(state: BatchState) -> dict:
    """等所有复制出来的筛选专家交卷后（自动屏障），拼成一份批量初筛报告。"""
    print(f"== 汇总：{len(state['screens'])} 份简历初筛已收齐 ==")
    return {"final_report": "\n\n".join(state["screens"])}


builder = StateGraph(BatchState)
builder.add_node("screener", screener)
builder.add_node("collect", collect)
# 条件边：START 后交给 dispatch，它返回一串 Send 动态派单；
# 第三个参数 ["screener"] 是路径清单，告诉图这些单子都发往 screener 节点
builder.add_conditional_edges(START, dispatch, ["screener"])
builder.add_edge("screener", "collect")
builder.add_edge("collect", END)
app = builder.compile()


if __name__ == "__main__":
    sample_job = (
        "AI Agent 开发实习生：熟悉 Python，了解 LangChain/LangGraph，"
        "理解 RAG、Function Calling，有大模型应用项目经验优先；"
        "实习薪资 4K-6K，要求每周到岗 4 天以上、可持续 3 个月以上。"
    )
    sample_resumes = [
        "张三，软工本科，会 Python、LangChain、LangGraph、RAG，做过招聘 Agent 开源项目，可实习 6 个月每周 5 天，期望 3K-5K。",
        "李四，计科本科，主攻 Java 后端，熟悉 Spring、MySQL，没接触过大模型，每周可到 3 天，期望 4K-6K。",
        "王五，软工专科，会一点 Python，做过公众号运营，无 AI 项目，可随时到岗全职实习，期望 3K。",
    ]

    start = time.time()
    result = app.invoke({
        "job_desc": sample_job,
        "resumes": sample_resumes,
        "screens": [],
    })
    print("\n===== 批量简历初筛报告 =====")
    print(result["final_report"])
    print(f"\n初筛 {len(sample_resumes)} 份，总耗时 {time.time() - start:.1f} 秒（并行≈最慢的一份）")
