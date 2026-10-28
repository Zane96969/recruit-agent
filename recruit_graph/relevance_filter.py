"""相关性筛选（LLM 裁判产品化）：判断候选人材料与候选 JD 是否真正相关。

两层过滤的分工（互补，不是替代）：
1. rerank 分数阈值（在 hybrid_search 里）：挡"完全不相关"——这类候选分数全都很低；
2. 本模块（LLM 裁判）：挡"表面相似但实质不匹配"——rerank 只看文本相似度，
   读不懂否定语义（如简历里有"沟通/内容"但明确拒绝销售外呼），LLM 能读懂。
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from recruit_graph.graph_tools import chat_model


class RelevanceVerdict(BaseModel):
    """裁判对"候选人材料—岗位"相关性的判断。"""

    relevant: bool = Field(description="这份候选人材料与该岗位是否真正相关/匹配")
    reason: str = Field(description="判断依据：相关点或不相关点，一句话")


judge_system = SystemMessage(
    content=(
        "你是中立的招聘裁判。你会收到一份候选人材料（简历或画像）和一个岗位 JD，"
        "只依据这两份材料判断二者是否真正相关：候选人的技能、方向、经历、求职意愿"
        "与该 JD 的职责、要求是否对口。"
        "特别注意：1）材料里明确表达拒绝或不接受某类工作（如拒绝销售、外呼、加班）"
        "的，即使有词汇重叠也判不相关；2）只有文本相似、但没有对口经历或技能的，"
        "判不相关；3）方向明显不同（如数据分析师、新媒体文案对技术/招聘岗）判不相关。"
        "只依据材料判断，禁止编造。"
    )
)

judge_model = chat_model.with_structured_output(
    RelevanceVerdict, method="function_calling"
)


def judge_relevance(candidate_text: str, jd_text: str) -> RelevanceVerdict:
    """判断一份候选人材料与一个岗位是否真正相关。"""
    return judge_model.invoke(
        [
            judge_system,
            HumanMessage(
                content=f"【候选人材料】\n{candidate_text}\n\n【岗位JD】\n{jd_text}"
            ),
        ]
    )


def filter_relevant_jd_ids(candidate_text: str, jd_ids: list) -> list:
    """对候选 JD 逐个裁判，返回真正相关的 jd_id 列表（保持原顺序）。

    get_jd 在这里延迟导入，避免与 graph_tools 形成顶层循环依赖。
    """
    from recruit_graph.jd_store import get_jd

    relevant_ids = []
    for jid in jd_ids:
        verdict = judge_relevance(candidate_text, get_jd(jid)["jd_text"])
        if verdict.relevant:
            relevant_ids.append(jid)
    return relevant_ids
