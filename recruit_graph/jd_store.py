"""岗位 JD 的 SQLite 存储层（第二张表：jds）。"""

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from recruit_graph.jd_vector_store import (
    upsert_jd,
    delete_jd_vectors,
    rename_jd_vectors,
)

DB_PATH = Path(__file__).resolve().parent / "recruit_real.db"

_CREATE_JD_TABLE = """
    CREATE TABLE IF NOT EXISTS jds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        jd_name TEXT,
        jd_text TEXT,
        content_hash TEXT UNIQUE,
        created_at TEXT
    )
"""


def _connect():
    """开一个新连接，并让查询结果能用列名取值（row['jd_name']）。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def save_jd(jd_name: str, jd_text: str) -> dict:
    """保存一份 JD；同一份原文重复保存会 UPDATE（保持原 id，评分记录不悬空）。"""
    content_hash = hashlib.md5(jd_text.strip().encode("utf-8")).hexdigest()
    conn = _connect()

    try:
        conn.execute(_CREATE_JD_TABLE)
        row = conn.execute(
            "SELECT id FROM jds WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        if row:
            jd_id = row["id"]
            conn.execute(
                """
                UPDATE jds SET jd_name = ?, jd_text = ?, created_at = ?
                WHERE id = ?
                """,
                (
                    jd_name,
                    jd_text,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    row["id"],
                ),
            )
        else:
            cur = conn.execute(
                """
                INSERT INTO jds (jd_name, jd_text, content_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    jd_name,
                    jd_text,
                    content_hash,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            jd_id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()

    delete_jd_vectors(jd_id)
    upsert_jd(jd_id, jd_name, jd_text)
    return {"jd_name": jd_name, "content_hash": content_hash, "jd_id": jd_id}


def update_jd(jd_id: int, jd_name: str, jd_text: str) -> dict:
    """按 id 更新已有岗位（改名 / 补充内容），保留原 id 与创建时间。

    - 正文没变、只改名：只更新 SQLite 名字与向量块 metadata，不重新向量化（快）。
    - 正文有改动：更新原文后删除旧向量块、按新文本切块重建；
      若新原文与别的岗位完全相同（会撞唯一约束），提前返回 error 不改任何东西。
    """
    content_hash = hashlib.md5(jd_text.strip().encode("utf-8")).hexdigest()
    conn = _connect()
    try:
        conn.execute(_CREATE_JD_TABLE)
        row = conn.execute(
            "SELECT content_hash FROM jds WHERE id = ?", (jd_id,)
        ).fetchone()
        if not row:
            return {"error": f"岗位 #{jd_id} 不存在，无法更新"}
        content_changed = row["content_hash"] != content_hash
        if content_changed:
            # 新内容若与“别的”岗位完全相同，会撞 content_hash 唯一约束，提前拦下
            dup = conn.execute(
                "SELECT id FROM jds WHERE content_hash = ? AND id != ?",
                (content_hash, jd_id),
            ).fetchone()
            if dup:
                return {
                    "error": f"修改后的内容与另一个岗位 #{dup['id']} 完全相同，已取消"
                }
            conn.execute(
                "UPDATE jds SET jd_name = ?, jd_text = ?, content_hash = ? WHERE id = ?",
                (jd_name, jd_text, content_hash, jd_id),
            )
        else:
            conn.execute("UPDATE jds SET jd_name = ? WHERE id = ?", (jd_name, jd_id))
        conn.commit()
    finally:
        conn.close()

    if content_changed:
        delete_jd_vectors(jd_id)
        upsert_jd(jd_id, jd_name, jd_text)
    else:
        rename_jd_vectors(jd_id, jd_name)
    return {"jd_id": jd_id, "jd_name": jd_name, "content_hash": content_hash}


def list_jds() -> list[dict]:
    """返回所有已存 JD：id + 岗位名 + 正文字数 + 保存时间，最新在前。"""
    conn = _connect()
    try:
        conn.execute(_CREATE_JD_TABLE)
        rows = conn.execute(
            "SELECT id, jd_name, length(jd_text) AS jd_len, created_at "
            "FROM jds ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_jd(jd_id: int) -> dict:
    """按 id 取一份 JD 的完整原文，找不到返回空字典。"""
    conn = _connect()
    try:
        conn.execute(_CREATE_JD_TABLE)
        row = conn.execute("SELECT * FROM jds WHERE id = ?", (jd_id,)).fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


def delete_jd(jd_id: int) -> bool:
    """按 id 删除一份 JD，返回是否真的删掉了一行。"""
    conn = _connect()
    try:
        conn.execute(_CREATE_JD_TABLE)
        cur = conn.execute("DELETE FROM jds WHERE id = ?", (jd_id,))
        conn.commit()
        deleted = cur.rowcount > 0
    finally:
        conn.close()

    if deleted:
        delete_jd_vectors(jd_id)
    return deleted


# ============ 第三张表：评分记录 matchings（候选人 × 指定JD 的多对多） ============
_CREATE_MATCHINGS_TABLE = """
    CREATE TABLE IF NOT EXISTS matchings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        jd_id INTEGER,
        candidate_id INTEGER,
        match_score INTEGER,
        created_at TEXT
    )
"""


def record_match(jd_id: int, candidate_id: int, match_score: int) -> dict:
    """记录一次"候选人 × 指定JD"的评分。"""
    conn = _connect()
    try:
        conn.execute(_CREATE_JD_TABLE)
        conn.execute(_CREATE_MATCHINGS_TABLE)
        conn.execute(
            """
            INSERT INTO matchings (jd_id, candidate_id, match_score, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                jd_id,
                candidate_id,
                match_score,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return {"jd_id": jd_id, "candidate_id": candidate_id, "match_score": match_score}


def list_matchings() -> list[dict]:
    """JOIN 三张表：评分记录 + JD名 + 候选人名，最新在前。"""
    conn = _connect()
    try:
        conn.execute(_CREATE_JD_TABLE)
        conn.execute(_CREATE_MATCHINGS_TABLE)
        rows = conn.execute("""
            SELECT m.id, j.jd_name, c.name AS candidate_name,
                   m.match_score, m.created_at
            FROM matchings m
            JOIN jds j ON m.jd_id = j.id
            JOIN candidates c ON m.candidate_id = c.id
            ORDER BY m.id DESC
            """).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def delete_matchings_by_candidate(candidate_id: int) -> int:
    """删除某候选人的全部评分记录（删候选人时级联清理），返回删掉几行。"""
    conn = _connect()
    try:
        conn.execute(_CREATE_MATCHINGS_TABLE)
        cur = conn.execute(
            "DELETE FROM matchings WHERE candidate_id = ?", (candidate_id,)
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()
