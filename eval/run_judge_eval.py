"""LLM 裁判评测：用 DeepSeek 当裁判，判断简历与检索到的 JD 是否相关。

解决两个问题：
1. 负样本（无对应岗位）：裁判说"全都不相关"时，系统应输出"无匹配"，而不是硬塞；
2. 统计端到端匹配正确率：正样本该匹配对、负样本该正确拒答。
"""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from recruit_graph.graph_tools import SCORE_THRESHOLD, chat_model
from recruit_graph.hybrid_retriever import hybrid_search
from recruit_graph.jd_store import get_jd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESUME_DIR = REPO_ROOT / "data" / "resumes"
DATASET_FILE = REPO_ROOT / "eval" / "eval_dataset.json"
TOP_K = 3


# 第一组：裁判的结构化输出（判断结果 + 理由）
class RelevanceVerdict(BaseModel):
    """裁判对'简历—岗位'相关性的判断。"""

    relevant: bool = Field(description="这份简历与该岗位是否相关/匹配")
    reason: str = Field(description="判断依据：相关点或不相关点，一句话")


judge_model = chat_model.with_structured_output(
    RelevanceVerdict, method="function_calling"
)

judge_system = SystemMessage(
    content=(
        "你是中立的招聘裁判。你会收到一份简历和一个岗位 JD，"
        "只依据这两份材料判断二者是否相关："
        "候选人的技能、方向、目标岗位与该 JD 的职责、要求是否对口。"
        "方向明显不同（如数据分析师、新媒体文案对技术/招聘岗）应判不相关。"
        "只依据材料判断，禁止编造。"
    )
)


# 第二组：判断一份简历与一个岗位是否相关
def judge_relevance(resume_text: str, jd_text: str) -> RelevanceVerdict:
    return judge_model.invoke(
        [
            judge_system,
            HumanMessage(content=f"【简历】\n{resume_text}\n\n【岗位JD】\n{jd_text}"),
        ]
    )


# 第三组：读评测数据集
dataset = json.loads(DATASET_FILE.read_text(encoding="utf-8"))

# 第四组：逐条检索 + 裁判判断
results = []
for case in dataset:
    resume_text = (RESUME_DIR / case["resume_file"]).read_text(encoding="utf-8")
    # 第一层：rerank 相关性阈值（第二层就是下面的 judge_relevance 裁判）
    docs = hybrid_search(resume_text, top_k=TOP_K, score_threshold=SCORE_THRESHOLD)

    # 检索结果按 jd_id 去重，再读完整 JD
    unique_jd_ids = []
    for doc in docs:
        jid = doc.metadata["jd_id"]
        if jid not in unique_jd_ids:
            unique_jd_ids.append(jid)

    # 裁判对每个岗位逐个判断
    verdicts = []
    for jid in unique_jd_ids:
        jd_text = get_jd(jid)["jd_text"]
        verdict = judge_relevance(resume_text, jd_text)
        verdicts.append((jid, verdict.relevant, verdict.reason))

    # 第一个被裁判认可的岗位作为最终匹配；都不认可则为 None（无匹配）
    matched = next((jid for jid, relevant, _ in verdicts if relevant), None)
    results.append(
        {
            "case_id": case["case_id"],
            "expected": case["expected_jd_id"],
            "matched": matched,
            "verdicts": verdicts,
        }
    )

# 第五组：对比期望，统计正确率
correct = 0
for r in results:
    if r["expected"] is None:
        is_correct = r["matched"] is None
    else:
        is_correct = r["matched"] == r["expected"]
    r["correct"] = is_correct
    correct += is_correct

accuracy = correct / len(results)

# 第六组：逐条打印 + 汇总
print("=== 裁判判断结果 ===")
for r in results:
    print(
        f"用例{r['case_id']}  期望JD={r['expected']}  "
        f"最终匹配={r['matched']}  正确={r['correct']}"
    )
    for jid, relevant, reason in r["verdicts"]:
        print(f"    JD{jid} 相关={relevant}  {reason}")

print("\n=== 汇总 ===")
print(f"端到端匹配正确率: {accuracy:.2%}")
