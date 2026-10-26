import hashlib
import json

from recruit_graph.db import connect, upsert_candidate
from recruit_graph.graph_tools import analyze_resume_worker, match_resume_worker

conn = connect()


def save_one(resume_text: str) -> None:
    """分析一份简历、打分并落库；手机号/内容指纹重复时原地更新（id 不变）。"""
    profile_json = analyze_resume_worker.invoke({"resume_text": resume_text})
    profile = json.loads(profile_json)
    match_result = json.loads(
        match_resume_worker.invoke({"profile_json": profile_json})
    )
    print(
        profile["name"], "→", match_result["matched_job"], match_result["match_score"]
    )

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
    upsert_candidate(conn, row)


# 1）准备 3 份简历的列表
# 2）循环逐份调用 save_one，整批放进一个事务里提交
# 3）按 match_score 从高到低排名打印（ORDER BY ... DESC）
# 4）筛出 80 分以上的高分候选人（WHERE match_score >= 80）
resumes = [
    """张三，23岁，大连外国语大学软件工程本科。
熟悉 Python、LangChain、RAG、LangGraph，做过一个招聘 Agent 项目。
目标岗位：AI Agent 开发实习生，期望薪资 3-5K。
电话 13800000001。""",
    """李四，24岁，大连交通大学计算机科学与技术本科。
熟悉 Java、Spring Boot、MySQL，做过一个电商后台管理系统。
目标岗位：Java 开发实习生，期望薪资 4-6K。
电话 13800000002。""",
    """王五，25岁，大连理工大学软件工程硕士。
精通 Python、LangChain、LangGraph、RAG、Function Calling，有大模型应用落地项目，曾在人工智能公司实习半年。
目标岗位：AI Agent 开发工程师，期望薪资 12-18K。
电话 13800000003。""",
]
with conn:  # 三份简历要么都写进去，要么都不写
    for resume_text in resumes:
        save_one(resume_text)

print("\n=== 按匹配度排名（高→低）===")
for row in conn.execute(
    "SELECT name, match_score, matched_job FROM candidates ORDER BY match_score DESC"
):
    print(row["name"], row["match_score"], row["matched_job"])

print("\n=== 80 分以上的高分候选人 ===")
for row in conn.execute(
    "SELECT name, match_score FROM candidates WHERE match_score >= 80 "
    "ORDER BY match_score DESC"
):
    print(row["name"], "|", row["match_score"])
conn.close()
