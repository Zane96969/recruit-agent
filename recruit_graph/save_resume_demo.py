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
content_hash = hashlib.md5(resume_text.strip().encode("utf-8")).hexdigest()
phone = profile["phone"] if profile["phone"] != "未知" else None
match_result = json.loads(match_resume_worker.invoke({"profile_json": profile_json}))
print("匹配结果：", match_result["matched_job"], match_result["match_score"])

cursor.execute(
    """
    INSERT OR REPLACE INTO candidates
        (name, age, education, skills,projects,target_position,
         expected_salary,phone,content_hash, created_at,matched_job,match_score)
    VALUES (?, ?, ?, ?, ?, ?, ?,?,?,? ,?,? )
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
conn.commit()
print("已把这位候选人的画像落库")
cursor.execute("SELECT name, match_score, matched_job FROM candidates")
for row in cursor.fetchall():
    print(row["name"], "|", row["match_score"], "|", row["matched_job"])
conn.close()
