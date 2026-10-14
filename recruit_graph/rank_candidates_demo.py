import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from graph_tools import analyze_resume_worker, match_resume_worker

DB_PATH = Path(__file__).resolve().parent / "recruit_real.db"
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# 表已存在就跳过（幂等），保证这个脚本单独跑也不会报错
cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age TEXT,
        education TEXT,
        skills TEXT,
        projects TEXT,
        target_position TEXT,
        expected_salary TEXT,
        phone TEXT UNIQUE,
        content_hash TEXT UNIQUE,
        created_at TEXT,
        matched_job TEXT,
        match_score INTEGER
    )
    """)


def save_one(resume_text: str) -> None:
    """分析一份简历、打分并落库；手机号/内容指纹重复时覆盖旧记录。"""
    profile_json = analyze_resume_worker.invoke({"resume_text": resume_text})
    profile = json.loads(profile_json)

    content_hash = hashlib.md5(resume_text.strip().encode("utf-8")).hexdigest()
    phone = profile["phone"] if profile["phone"] != "未知" else None

    match_result = json.loads(
        match_resume_worker.invoke({"profile_json": profile_json})
    )
    print(
        profile["name"], "→", match_result["matched_job"], match_result["match_score"]
    )

    cursor.execute(
        """
        INSERT OR REPLACE INTO candidates
            (name, age, education, skills, projects, target_position,
             expected_salary, phone, content_hash, created_at,
             matched_job, match_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            profile["name"],
            profile["age"],
            profile["education"],
            ",".join(profile["skills"]),
            ",".join(profile["projects"]),
            profile["target_position"],
            profile["expected_salary"],
            phone,
            content_hash,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            match_result["matched_job"],
            match_result["match_score"],
        ),
    )


# 1）准备 3 份简历的列表
# 2）循环逐份调用 save_one，循环结束后 conn.commit()
# 3）按 match_score 从高到低排名打印（ORDER BY ... DESC）
# 4）筛出 80 分以上的高分候选人（WHERE match_score >= 80）
# 最后 conn.close()
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
for resume_text in resumes:
    save_one(resume_text)
conn.commit()
print("\n=== 按匹配度排名（高→低）===")
cursor.execute(
    "SELECT name, match_score, matched_job FROM candidates ORDER BY match_score DESC"
)
for row in cursor.fetchall():
    print(row["name"], row["match_score"], row["matched_job"])

print("\n=== 80 分以上的高分候选人 ===")
cursor.execute(
    "SELECT name,match_score FROM candidates WHERE match_score >= 80 ORDER BY match_score DESC"
)
for row in cursor.fetchall():
    print(row["name"], "|", row["match_score"])
conn.close()
