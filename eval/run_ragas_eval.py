"""RAGAS 标准化评测：用 RAGAS 给招聘匹配 RAG 打行业通用指标。

DeepSeek 既当生成模型（产出匹配结论），又当评测裁判（打分）。
三个指标（RAG 黄金三件套）：
1. LLMContextRecall（上下文召回）：该检索到的依据有没有覆盖；
2. LLMContextPrecisionWithReference（上下文精确率）：检索到的材料是否相关；
3. Faithfulness（忠实度）：匹配结论是否严格基于材料，有无瞎编。
匹配正确性（岗位名对不对）由自建裁判 run_judge_eval.py 负责。
"""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from recruit_graph.graph_tools import SCORE_THRESHOLD, chat_model
from recruit_graph.hybrid_retriever import hybrid_search
from recruit_graph.jd_store import get_jd
from recruit_graph.relevance_filter import filter_relevant_jd_ids

REPO_ROOT = Path(__file__).resolve().parent.parent
RESUME_DIR = REPO_ROOT / "data" / "resumes"
DATASET_FILE = REPO_ROOT / "eval" / "eval_dataset.json"
TOP_K = 3

# 第一组：生成匹配结论的系统提示
gen_system = SystemMessage(
    content=(
        "你是招聘匹配助手。根据检索到的岗位 JD，判断这份简历匹配哪个岗位，"
        "先给出匹配的岗位名称，再给一句理由。"
        "如果与所有岗位都不匹配，明确回答“无匹配”。"
        "只依据简历和 JD 作答，禁止编造。"
    )
)


# 第二组：让系统基于检索到的 JD 生成匹配结论
def generate_response(resume_text: str, jd_texts: list[str]) -> str:
    jd_block = "\n---\n".join(jd_texts)
    answer = chat_model.invoke(
        [
            gen_system,
            HumanMessage(
                content=f"【简历】\n{resume_text}\n\n【检索到的岗位JD】\n{jd_block}"
            ),
        ]
    )
    return answer.content


# 第三组：读评测数据集，逐条收集 RAG 数据（含真实匹配/拒答）
dataset = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
all_data = []

for case in dataset:
    resume_text = (RESUME_DIR / case["resume_file"]).read_text(encoding="utf-8")

    # 第一层：rerank 相关性阈值，挡掉完全不相关的块
    docs = hybrid_search(resume_text, top_k=TOP_K, score_threshold=SCORE_THRESHOLD)
    candidate_jd_ids = []
    for doc in docs:
        jid = doc.metadata["jd_id"]
        if jid not in candidate_jd_ids:
            candidate_jd_ids.append(jid)

    # 第二层：LLM 裁判过滤"表面相似但实质不匹配"
    jd_ids = filter_relevant_jd_ids(resume_text, candidate_jd_ids)

    # 上下文 = 候选人简历 + 过滤后真正相关的 JD（结论同时引用两者）
    contexts = [f"【候选人简历】\n{resume_text}"]
    contexts.extend(get_jd(jid)["jd_text"] for jid in jd_ids)

    # 两层后无相关岗位：回答"无匹配"；否则生成匹配结论
    if jd_ids:
        response = generate_response(
            resume_text, [get_jd(jid)["jd_text"] for jid in jd_ids]
        )
    else:
        response = "无匹配"
    reference = case["expected_jd_name"] or "无匹配"

    all_data.append(
        {
            "case_id": case["case_id"],
            "case_type": case["case_type"],
            "user_input": "这份简历匹配哪个岗位？",
            "retrieved_contexts": contexts,
            "response": response,
            "reference": reference,
        }
    )

# 第四组：分两类评测。
# 正样本是标准 RAG（有检索、有生成），用 RAGAS 三件套；
# 负样本是"拒答"场景：结论"无匹配"不对应任何检索到的片段，
# RAGAS 三件套会误判（它问"无匹配在 contexts 里有依据吗"，必然没有），
# 故负样本改用"拒答正确率"：系统是否正确输出"无匹配"。
positive_data = [item for item in all_data if item["case_type"] == "positive"]
negative_data = [item for item in all_data if item["case_type"] == "negative"]

# 去掉 RAGAS 不认识的辅助字段（case_id / case_type）
positive_ragas = [
    {key: value for key, value in item.items() if key not in ("case_id", "case_type")}
    for item in positive_data
]
evaluation_dataset = EvaluationDataset.from_list(positive_ragas)

# 第五组：正样本用 RAGAS 跑三件套
evaluator_llm = LangchainLLMWrapper(chat_model)
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
        Faithfulness(),
    ],
    llm=evaluator_llm,
)

# 负样本：统计拒答正确率（结论里出现"无匹配"即正确拒答）
rejection_correct = sum("无匹配" in item["response"] for item in negative_data)
rejection_rate = rejection_correct / len(negative_data)

print("=== 各条生成结论 ===")
for item in all_data:
    print(f"用例{item['case_id']}({item['case_type']}) " f"参考={item['reference']}")
    print(f"结论={item['response']}\n")

print("=== 正样本 RAGAS 三件套 ===")
print(result)
print("\n=== 负样本拒答正确率 ===")
print(
    f"拒答正确率: {rejection_rate:.2%} " f"({rejection_correct}/{len(negative_data)})"
)
