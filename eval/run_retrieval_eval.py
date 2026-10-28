"""最小检索评测：拿简历当 query，评测 hybrid_search 能否命中期望 JD。

两个指标：
1. Hit Rate（命中率）：top_k 结果里包含期望 JD 的用例占比；
2. MRR（平均倒数排名）：期望 JD 排第几，第1名=1、第2名=1/2、第3名=1/3，没命中=0。
"""
import json
from pathlib import Path

from recruit_graph.hybrid_retriever import hybrid_search

REPO_ROOT = Path(__file__).resolve().parent.parent
RESUME_DIR = REPO_ROOT / "data" / "resumes"
DATASET_FILE = REPO_ROOT / "eval" / "eval_dataset.json"
TOP_K = 3

# 第一组：读评测数据集（8 个用例）
dataset = json.loads(DATASET_FILE.read_text(encoding="utf-8"))

# 第二组：逐条跑检索，记录命中情况
results = []
for case in dataset:
    resume_text = (RESUME_DIR / case["resume_file"]).read_text(encoding="utf-8")
    docs = hybrid_search(resume_text, top_k=TOP_K)
    retrieved = [doc.metadata["jd_id"] for doc in docs]

    expected = case["expected_jd_id"]
    if case["case_type"] == "positive":
        hit = expected in retrieved
        rank = retrieved.index(expected) + 1 if hit else 0
        rr = 1 / rank if hit else 0
    else:
        hit = None
        rr = None

    results.append({
        "case_id": case["case_id"],
        "expected": expected,
        "retrieved": retrieved,
        "hit": hit,
        "rr": rr,
    })

# 第三组：汇总指标（只统计正样本）
positive = [r for r in results if r["hit"] is not None]
hit_rate = sum(r["hit"] for r in positive) / len(positive)
mrr = sum(r["rr"] for r in positive) / len(positive)

# 第四组：逐条打印 + 汇总
print("=== 逐条检索结果 ===")
for r in results:
    print(
        f"用例{r['case_id']}  期望JD={r['expected']}  "
        f"检索到={r['retrieved']}  命中={r['hit']}"
    )

print("\n=== 汇总指标（正样本）===")
print(f"Hit Rate 命中率: {hit_rate:.2%}")
print(f"MRR 平均倒数排名: {mrr:.3f}")
