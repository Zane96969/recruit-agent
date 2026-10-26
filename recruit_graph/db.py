"""SQLite 公共层：数据库路径、连接、候选人写入（UPSERT）。

单独抽一个模块，只为三件事：

1. **一份路径**：`jd_store` 与 `candidate_store` 原来各写一遍
   `Path(__file__).parent / "recruit_real.db"`，两份常量一旦 drift，
   就会出现"岗位在 A 库、候选人在 B 库"的怪事；
2. **一个入口**：候选人写入的"去重 + 保持 id"逻辑只写一次；
3. **可测试**：本模块只依赖标准库，不 import 大模型 / 向量库，
   因此单元测试可以直接导入它，不需要 API_KEY 和 Chroma。
"""

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "recruit_real.db"

# 候选人画像 + 最近一次匹配结果。phone / content_hash 都建了唯一索引，用来去重。
CREATE_CANDIDATES = """
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

# 写入/更新涉及的字段（不含自增 id 与 created_at）
CANDIDATE_FIELDS = (
    "name",
    "age",
    "education",
    "skills",
    "projects",
    "target_position",
    "expected_salary",
    "phone",
    "content_hash",
    "matched_job",
    "match_score",
)


def connect() -> sqlite3.Connection:
    """开一个连接，并让查询结果能用列名取值（row["id"]）。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def upsert_candidate(conn: sqlite3.Connection, row: dict) -> tuple[int, bool]:
    """按 content_hash / phone 去重写入一名候选人，返回 (候选人 id, 是否为更新)。

    **为什么不用 `INSERT OR REPLACE`**：SQLite 的 REPLACE 语义是"删掉冲突的旧行、
    再插入新行"，命中唯一键时 `id` 会变。而 `matchings.candidate_id` 引用的是旧 id，
    同一份简历再筛一次，历史评分记录就全部指向一条不存在的候选人 —— 数据静默损坏。
    这里显式 SELECT → UPDATE / INSERT，id 始终稳定。

    调用方负责事务（`with conn:`），本函数只执行语句，方便与别的写操作放进同一事务。
    """
    conn.execute(CREATE_CANDIDATES)

    phone = row.get("phone")
    # 手机号为空时不能写 `phone = NULL`（永远为 NULL，等于匹配不到），
    # 所以先判断 `? IS NOT NULL` 再用；空手机号只按 content_hash 去重。
    existing = conn.execute(
        """
        SELECT id FROM candidates
        WHERE content_hash = ? OR (? IS NOT NULL AND phone = ?)
        ORDER BY id
        LIMIT 1
        """,
        (row["content_hash"], phone, phone),
    ).fetchone()

    values = {field: row.get(field) for field in CANDIDATE_FIELDS}

    if existing is not None:
        candidate_id = existing["id"]
        assignments = ", ".join(f"{field} = ?" for field in CANDIDATE_FIELDS)
        conn.execute(
            f"UPDATE candidates SET {assignments} WHERE id = ?",
            tuple(values[field] for field in CANDIDATE_FIELDS) + (candidate_id,),
        )
        return candidate_id, True

    values["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    columns = list(values)
    placeholders = ", ".join("?" for _ in columns)
    cur = conn.execute(
        f"INSERT INTO candidates ({', '.join(columns)}) VALUES ({placeholders})",
        tuple(values[column] for column in columns),
    )
    return int(cur.lastrowid), False
