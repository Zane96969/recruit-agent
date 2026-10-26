import hashlib
import json

from recruit_graph.db import connect, upsert_candidate
from recruit_graph.graph_tools import analyze_resume_worker, match_resume_worker

conn = connect()

resume_text = """
张三，23岁，大连外国语大学软件工程本科。
熟悉 Python、LangChain、RAG、LangGraph，做过一个招聘 Agent 项目。
目标岗位：AI Agent 开发实习生，期望薪资 3-5K。
电话 13800000001。
"""
profile_json = analyze_resume_worker.invoke({"resume_text": resume_text})
print("大模型返回的原始结果：")
print(profile_json)
print(type(profile_json))

profile = json.loads(profile_json)
print("解析成字典后：")
print(profile)
print(profile["name"], profile["skills"])
match_result = json.loads(match_resume_worker.invoke({"profile_json": profile_json}))
print("匹配结果：", match_result["matched_job"], match_result["match_score"])

# 落库统一走 db.upsert_candidate：按 content_hash / 手机号去重，且重复保存时 id 不变。
# 不用 INSERT OR REPLACE —— 它是"删旧行再插新行"，id 会变，评分记录（matchings）会悬空。
row = {
    "name": profile["name"],
    "age": profile["age"],
    "education": profile["education"],
    "skills": ",".join(profile["skills"]),
    "projects": ",".join(profile["projects"]),
    "target_position": profile["target_position"],
    "expected_salary": profile["expected_salary"],
    "phone": profile["phone"] if profile["phone"] != "未知" else None,
    "content_hash": hashlib.md5(resume_text.strip().encode("utf-8")).hexdigest(),
    "matched_job": match_result["matched_job"],
    "match_score": match_result["match_score"],
}
with conn:
    candidate_id, is_update = upsert_candidate(conn, row)
print("已把这位候选人的画像落库：id =", candidate_id, "| 是否覆盖已有记录 =", is_update)

for row in conn.execute("SELECT name, match_score, matched_job FROM candidates"):
    print(row["name"], "|", row["match_score"], "|", row["matched_job"])
conn.close()
