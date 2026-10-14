"""候选人画像与匹配结果的 SQLite 存储层（供 FastAPI 调用，也可单独复用）。"""

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from recruit_graph.graph_tools import analyze_resume_worker, match_resume_worker

DB_PATH = Path(__file__).resolve().parent / "recruit_real.db"

_CREATE_TABLE = """
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
"""


def _connect():
    """开一个新连接，并让查询结果能用列名取值（row['name']）。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def save_candidate(resume_text: str) -> dict:
    """分析简历 → 打分 → 落库，返回画像与匹配结果；重复简历会覆盖。"""
    profile_json = analyze_resume_worker.invoke({"resume_text": resume_text})
    profile = json.loads(profile_json)

    match_result = json.loads(
        match_resume_worker.invoke({"profile_json": profile_json})
    )

    content_hash = hashlib.md5(resume_text.strip().encode("utf-8")).hexdigest()
    phone = profile["phone"] if profile["phone"] != "未知" else None

    conn = _connect()
    try:
        conn.execute(_CREATE_TABLE)
        conn.execute(
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
        conn.commit()
    finally:
        conn.close()

    return {"profile": profile, "match": match_result}


def list_candidates(min_score: int = 0) -> list[dict]:
    """按匹配度从高到低返回候选人；min_score 可只保留该分数以上的人。"""
    conn = _connect()
    try:
        conn.execute(_CREATE_TABLE)
        rows = conn.execute(
            """
            SELECT name, match_score, matched_job, target_position, id
            FROM candidates
            WHERE match_score >= ?
            ORDER BY match_score DESC
            """,
            (min_score,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def analyze_profile(resume_text: str) -> dict:
    """第一步：只分析简历，返回画像 dict，不落库（供前端纠错）。"""
    profile_json = analyze_resume_worker.invoke({"resume_text": resume_text})
    return json.loads(profile_json)


def save_candidate_from_profile(profile: dict, resume_text: str) -> dict:
    """第二步：把（可能被用户改过的）画像入库。
    同一份简历原文（content_hash）或手机号已存在 → UPDATE（保持原 id）；
    否则 → INSERT 新行。
    """
    content_hash = hashlib.md5(resume_text.strip().encode("utf-8")).hexdigest()
    phone = profile["phone"] if profile["phone"] != "未知" else None

    match_result = json.loads(
        match_resume_worker.invoke(
            {"profile_json": json.dumps(profile, ensure_ascii=False)}
        )
    )

    conn = _connect()
    try:
        conn.execute(_CREATE_TABLE)
        row = conn.execute(
            """
            SELECT id FROM candidates
            WHERE content_hash = ? OR (phone IS NOT NULL AND phone = ?)
            """,
            (content_hash, phone),
        ).fetchone()
        if row:
            conn.execute(
                """
                UPDATE candidates SET
                    name = ?, age = ?, education = ?, skills = ?,
                    projects = ?, target_position = ?, expected_salary = ?,
                    phone = ?, matched_job = ?, match_score = ?, content_hash = ?
                WHERE id = ?
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
                    match_result["matched_job"],
                    match_result["match_score"],
                    content_hash,
                    row["id"],
                ),
            )
            is_update = True
        else:
            conn.execute(
                """
                INSERT INTO candidates
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
            is_update = False
        conn.commit()
    finally:
        conn.close()

    return {"profile": profile, "match": match_result, "is_update": is_update}


def get_candidate_id(name: str) -> int | None:
    """按姓名查候选人 id；找不到返回 None（供评分记录关联用）。"""
    conn = _connect()
    try:
        conn.execute(_CREATE_TABLE)
        row = conn.execute(
            "SELECT id FROM candidates WHERE name = ?", (name,)
        ).fetchone()
        return row["id"] if row else None
    finally:
        conn.close()


def delete_candidate(candidate_id: int):
    """按 id 删除一个候选人，返回是否真的删掉了一行。"""
    conn = _connect()
    try:
        conn.execute(_CREATE_TABLE)
        cur = conn.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
