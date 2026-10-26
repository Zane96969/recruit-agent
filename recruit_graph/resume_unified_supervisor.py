"""统一调度毕业作品：大模型主管 + 三个子图车间。

主管读人话语义做三选一，派发到已编译好的子图；
模型只选路，车间流程由图结构在编译期锁死。
运行：python recruit_graph/resume_unified_supervisor.py
"""

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from recruit_graph.graph_tools import (
    chat_model,
    analyze_resume_worker,
    match_system,
)
from recruit_graph.hybrid_retriever import hybrid_search


class RecruitState(TypedDict):
    user_input: str  # 人话任务，主管靠它派单
    resume: str  # 单份简历（串行/并行车间用）
    job_desc: str  # 指定岗位 JD（并行/批量车间用）
    resumes: list[str]  # 批量简历名单（Send 车间用）
    profile_json: str  # 串行车间中间产物：画像 JSON
    parts: Annotated[list, operator.add]  # 并行三专家零件筐（reducer 累加）
    screens: Annotated[list, operator.add]  # 批量初筛零件筐（reducer 累加）
    final_report: str  # 最终报告（成品，只写一次）


def serial_analyze(state: RecruitState) -> dict:
    """串行工位①：把简历原文交给画像工具，产出结构化画像 JSON。"""
    print("-- 串行车间① 简历画像 开工 --")
    profile_json = analyze_resume_worker.invoke({"resume_text": state["resume"]})
    return {"profile_json": profile_json}


def serial_match(state: RecruitState) -> dict:
    """串行工位②：拿画像去岗位库检索 JD，再让匹配专家出报告。

    检索统一走 hybrid_search（BM25 + 向量 + RRF + rerank），
    不要再用裸的向量 retriever：同一套问题在两个入口必须命中同一批资料。
    top_k=2 与原 retriever 的 k=2 保持一致。
    """
    print("-- 串行车间② 检索JD + 人岗匹配 开工 --")
    jd_docs = hybrid_search(state["profile_json"], top_k=2)
    jd_text = "\n---\n".join(doc.page_content for doc in jd_docs)
    answer = chat_model.invoke(
        [
            match_system,
            HumanMessage(
                content=(
                    f"【候选人画像】\n{state['profile_json']}\n\n"
                    f"【相关岗位 JD】\n{jd_text}\n请给出匹配报告。"
                )
            ),
        ]
    )
    return {
        "final_report": f"【候选人画像】\n{state['profile_json']}\n\n{answer.content}"
    }


serial_builder = StateGraph(RecruitState)
serial_builder.add_node("serial_analyze", serial_analyze)
serial_builder.add_node("serial_match", serial_match)
serial_builder.add_edge(START, "serial_analyze")
serial_builder.add_edge("serial_analyze", "serial_match")
serial_builder.add_edge("serial_match", END)
serial_subgraph = serial_builder.compile()


def tech_expert(state: RecruitState) -> dict:
    """技术栈专家：只看技术与项目，给技术匹配度分数。"""
    print("-- 技术栈专家 开工 --")
    reply = chat_model.invoke(
        [
            SystemMessage(
                content="你是技术栈评估专家，只从技术栈、项目经验角度对照岗位评估，3 句以内，并给技术匹配度分数(0-100)。"
            ),
            HumanMessage(
                content=f"【岗位要求】\n{state['job_desc']}\n\n【候选人简历】\n{state['resume']}"
            ),
        ]
    )
    print("-- 技术栈专家 完成 --")
    return {"parts": [f"【技术栈评估】\n{reply.content}"]}


def salary_expert(state: RecruitState) -> dict:
    """薪资专家：只评估期望薪资与岗位预算是否匹配。"""
    print("-- 薪资专家 开工 --")
    reply = chat_model.invoke(
        [
            SystemMessage(
                content="你是薪资评估专家，只评估候选人期望薪资与岗位预算是否匹配，3 句以内，明确是否在区间内。"
            ),
            HumanMessage(
                content=f"【岗位要求】\n{state['job_desc']}\n\n【候选人简历】\n{state['resume']}"
            ),
        ]
    )
    print("-- 薪资专家 完成 --")
    return {"parts": [f"【薪资评估】\n{reply.content}"]}


def stability_expert(state: RecruitState) -> dict:
    """到岗稳定性专家：只看每周到岗天数、可持续时长、到岗速度。"""
    print("-- 稳定性专家 开工 --")
    reply = chat_model.invoke(
        [
            SystemMessage(
                content="你是到岗稳定性评估专家，只评估每周到岗天数、可持续实习时长、到岗速度是否满足岗位，3 句以内。"
            ),
            HumanMessage(
                content=f"【岗位要求】\n{state['job_desc']}\n\n【候选人简历】\n{state['resume']}"
            ),
        ]
    )
    print("-- 稳定性专家 完成 --")
    return {"parts": [f"【稳定性评估】\n{reply.content}"]}


def collect_p(state: RecruitState) -> dict:
    """汇总节点：三个专家都交卷后才运行（LangGraph 自动屏障等齐），拼成一份报告。"""
    print("== 汇总：三块结论已收齐 ==")
    return {"final_report": "\n\n".join(state["parts"])}


parallel_builder = StateGraph(RecruitState)
parallel_builder.add_node("tech", tech_expert)
parallel_builder.add_node("salary", salary_expert)
parallel_builder.add_node("stability", stability_expert)
parallel_builder.add_node("collect_p", collect_p)
parallel_builder.add_edge(START, "tech")
parallel_builder.add_edge(START, "salary")
parallel_builder.add_edge(START, "stability")
parallel_builder.add_edge("tech", "collect_p")
parallel_builder.add_edge("salary", "collect_p")
parallel_builder.add_edge("stability", "collect_p")
parallel_builder.add_edge("collect_p", END)
parallel_subgraph = parallel_builder.compile()


def batch_dispatch(state: RecruitState):
    """简历有几份就开几张 Send 单子，都派给同一个 screener 节点（运行时自动复制）。"""
    return [
        Send(
            "screener", {"index": i + 1, "resume": text, "job_desc": state["job_desc"]}
        )
        for i, text in enumerate(state["resumes"])
    ]


def screener(state: dict) -> dict:
    """筛选专家：只收到 Send 发来的【单份简历】小包，做一次快速初筛。"""
    index = state["index"]
    print(f"-- 筛选专家 开工：候选人{index} --")
    reply = chat_model.invoke(
        [
            SystemMessage(
                content=(
                    "你是简历初筛专家，依据岗位要求对单份简历做快速初筛："
                    "先给结论（建议推进 / 待定 / 不建议），再给 0-100 匹配分，"
                    "最后一句话说明主要理由，总共不超过 3 句。"
                )
            ),
            HumanMessage(
                content=(
                    f"【岗位要求】\n{state['job_desc']}\n\n"
                    f"【候选人{index}简历】\n{state['resume']}"
                )
            ),
        ]
    )
    print(f"-- 筛选专家 完成：候选人{index} --")
    return {"screens": [f"候选人{index}：{reply.content}"]}


def collect_b(state: RecruitState) -> dict:
    """等所有复制出来的筛选专家交卷后（自动屏障），拼成一份批量初筛报告。"""
    print(f"== 汇总：{len(state['screens'])} 份简历初筛已收齐 ==")
    return {"final_report": "\n\n".join(state["screens"])}


batch_builder = StateGraph(RecruitState)
batch_builder.add_node("screener", screener)
batch_builder.add_node("collect_b", collect_b)
batch_builder.add_conditional_edges(START, batch_dispatch, ["screener"])
batch_builder.add_edge("screener", "collect_b")
batch_builder.add_edge("collect_b", END)
batch_subgraph = batch_builder.compile()


class RouteDecision(BaseModel):
    """主管的结构化决策：只能从三个车间里选一个。"""

    route: Literal["serial_all", "parallel_all", "batch_all"] = Field(
        description="选择的车间节点名"
    )


router_model = chat_model.with_structured_output(
    RouteDecision,
    method="function_calling",
)
ROUTER_SYS = (
    "你是招聘智能体的调度主管，只负责决定任务派给哪个车间，不要自己干活：\n"
    "1. 有严格先后依赖、必须一步步来（如先分析简历、再拿分析结果匹配岗位）→ serial_all\n"
    "2. 同一份材料要从几个固定维度同时评估（如技术栈、薪资、稳定性三个专家并行）→ parallel_all\n"
    "3. 有一批数量不定的对象、每个都做同样处理（如 N 份简历逐份初筛）→ batch_all\n"
    "只返回 route 字段。"
)


def supervisor_route(state: RecruitState) -> str:
    """大图条件边函数：问主管该走哪个车间，返回节点名字符串。"""
    decision = router_model.invoke(
        [
            SystemMessage(content=ROUTER_SYS),
            HumanMessage(content=state["user_input"]),
        ]
    )
    print(f"主管决定派往：{decision.route}")
    return decision.route


builder = StateGraph(RecruitState)
builder.add_node("serial_all", serial_subgraph)
builder.add_node("parallel_all", parallel_subgraph)
builder.add_node("batch_all", batch_subgraph)
builder.add_conditional_edges(START, supervisor_route)
builder.add_edge("batch_all", END)
builder.add_edge("serial_all", END)
builder.add_edge("parallel_all", END)
graph = builder.compile()

if __name__ == "__main__":
    sample_resume = (
        "张三，23岁，大连外国语大学软件工程本科，"
        "熟悉 Python、LangChain、LangGraph、RAG、Chroma 向量库，"
        "独立做过招聘 Agent 开源项目，可实习 6 个月、每周 5 天，"
        "期望岗位 AI Agent 开发实习生，期望薪资 3K-5K。"
    )
    sample_job = (
        "AI Agent 开发实习生：熟悉 Python，了解 LangChain/LangGraph，"
        "理解 RAG、Function Calling，有大模型项目经验优先；"
        "实习薪资 4K-6K，要求每周到岗 4 天以上、可持续 3 个月以上。"
    )
    batch_resumes = [
        sample_resume,
        "李四，计科本科，主攻 Java 后端，熟悉 Spring、MySQL，没接触过大模型，每周可到 3 天，期望 4K-6K。",
        "王五，软工专科，会一点 Python，做过公众号运营，无 AI 项目，可随时到岗全职实习，期望 3K。",
    ]

    print("\n========== 场景1：只给简历，自动画像找岗 ==========")
    r1 = graph.invoke(
        {
            "user_input": "帮我分析这份简历，看看适合岗位库里的哪个岗位",
            "resume": sample_resume,
            "parts": [],
            "screens": [],
        }
    )
    print(r1["final_report"])

    print("\n========== 场景2：一份简历对指定 JD 多维评估 ==========")
    r2 = graph.invoke(
        {
            "user_input": "对照这个岗位，从技术栈、薪资、稳定性三个维度同时评估这份简历",
            "resume": sample_resume,
            "job_desc": sample_job,
            "parts": [],
            "screens": [],
        }
    )
    print(r2["final_report"])

    print("\n========== 场景3：一批简历批量初筛 ==========")
    r3 = graph.invoke(
        {
            "user_input": "这里有三份简历，对照岗位帮我逐份做个快速初筛",
            "job_desc": sample_job,
            "resumes": batch_resumes,
            "parts": [],
            "screens": [],
        }
    )
    print(r3["final_report"])
