"""候选人画像与匹配结果的 SQLite 存储层（供 FastAPI 调用，也可单独复用）。"""

import hashlib
import json
import sqlite3
import time
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


def _run_step(step_name: str, func, payload: dict, retries: int = 2):
    """执行串联链里的一个步骤，失败时整体重试，全失败就抛带步骤名的错误。

    - step_name：步骤名（如“简历分析”），出错时用来标明卡在哪一步；
    - func：这一步要执行的可调用对象，通常是 LangChain 工具的 .invoke；
    - payload：传给 func 的参数（dict）；
    - retries：失败后最多整体重试几次，重试间隔逐次拉长（1s、2s）。
    """
    last_error = None
    for attempt in range(retries + 1):
        try:
            return func(payload)
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < retries:
                time.sleep(attempt + 1)  # 退避一下，不立刻猛打
    raise RuntimeError(f"步骤【{step_name}】失败：{last_error}") from last_error


def save_candidate(resume_text: str) -> dict:
    """分析简历 → 打分 → 落库，返回画像与匹配结果；重复简历会覆盖。"""
    # 第一步：简历分析（依赖 LLM）。它是整条链的刚需，失败整体重试后仍不行就短路。
    try:
        profile_json = _run_step(
            "简历分析",
            analyze_resume_worker.invoke,
            {"resume_text": resume_text},
        )
    except Exception as e:
        # 画像没拿到，后面匹配/写库都无从谈起，直接返回，绝不碰数据库。
        return {"error": "简历分析失败，未进行匹配与入库", "detail": str(e)}

    profile = json.loads(profile_json)
    # 产出校验：没抛异常但关键字段缺失，同样视同上游不合格，立刻短路。
    if not profile.get("name") or profile.get("name") == "未知":
        return {"error": "简历画像缺少姓名，无法入库，请补充简历后重试"}

    # 第二步：人岗匹配，吃的是第一步产出的画像；失败同样短路，不进写库。
    try:
        match_result = json.loads(
            _run_step(
                "人岗匹配",
                match_resume_worker.invoke,
                {"profile_json": profile_json},
            )
        )
    except Exception as e:
        return {"error": "人岗匹配失败，候选人未入库", "detail": str(e)}

    # 第三步：写库。前两步都成功、数据都齐了才走到这里。
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
    except Exception as e:
        return {"error": "写库失败，画像与匹配结果未保存", "detail": str(e)}
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
