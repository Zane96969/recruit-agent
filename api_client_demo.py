import requests

BASE = "http://127.0.0.1:8000"

# 1) 发一份新简历：POST /candidates → 分析、打分、落库
new_resume = """王五，25岁，大连理工大学软件工程硕士。
精通 Python、LangChain、LangGraph、RAG、Function Calling，有大模型应用落地项目，曾在人工智能公司实习半年。
目标岗位：AI Agent 开发工程师，期望薪资 12-18K。
电话 13800000003。
"""
print("=== 1) POST /candidates 入库 ===")
resp = requests.post(f"{BASE}/candidates", json={"resume_text": new_resume}, timeout=90)
resp.raise_for_status()
saved = resp.json()
print("姓名:", saved["profile"]["name"])
print("匹配岗位:", saved["match"]["matched_job"])
print(
    "匹配分数:",
    saved["match"]["match_score"],
    type(saved["match"]["match_score"]).__name__,
)

# 2) GET /candidates 看全部排名
print("\n=== 2) GET /candidates 全部排名 ===")
resp = requests.get(f"{BASE}/candidates", timeout=30)
resp.raise_for_status()
for c in resp.json()["candidates"]:
    print(c["name"], c["match_score"], c["matched_job"])

# 3) GET /candidates?min_score=80 只看高分
print("\n=== 3) GET /candidates?min_score=80 高分 ===")
resp = requests.get(f"{BASE}/candidates", params={"min_score": 80}, timeout=30)
resp.raise_for_status()
for c in resp.json()["candidates"]:
    print(c["name"], c["match_score"])
