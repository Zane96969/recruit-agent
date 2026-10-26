"""候选人画像与匹配结果的 SQLite 存储层（供 FastAPI / Gradio 调用，也可单独复用）。"""

import hashlib
import json
import time

from recruit_graph.db import DB_PATH, connect, upsert_candidate
from recruit_graph.graph_tools import analyze_resume_worker, match_resume_worker

__all__ = [
    "DB_PATH",
    "analyze_profile",
    "delete_candidate",
    "get_candidate_id",
    "list_candidates",
    "save_candidate",
    "save_candidate_from_profile",
]


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


def _build_row(profile: dict, resume_text: str, match_result: dict) -> dict:
    """把「画像 + 匹配结果 + 简历原文」组装成一行候选人记录。

    技能/项目在库里用逗号拼成一列（岗位量级小，够用且直观）；
    手机号识别不到时存 NULL —— 唯一索引允许多个 NULL，不会把多个"未知"当成同一个人。
    """
    return {
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


def _write_candidate(row: dict) -> tuple[int, bool]:
    """在一个事务里写入候选人，返回 (候选人 id, 是否为更新)。"""
    conn = connect()
    try:
        with conn:  # 成功提交、异常回滚：要么整行写进去，要么一行都不写
            return upsert_candidate(conn, row)
    finally:
        conn.close()


def save_candidate(resume_text: str) -> dict:
    """分析简历 → 打分 → 落库，返回画像与匹配结果；重复简历原地更新（id 不变）。"""
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
    try:
        candidate_id, is_update = _write_candidate(
            _build_row(profile, resume_text, match_result)
        )
    except Exception as e:  # noqa: BLE001
        return {"error": "写库失败，画像与匹配结果未保存", "detail": str(e)}

    return {
        "profile": profile,
        "match": match_result,
        "candidate_id": candidate_id,
        "is_update": is_update,
    }


def list_candidates(min_score: int = 0) -> list[dict]:
    """按匹配度从高到低返回候选人；min_score 可只保留该分数以上的人。"""
    conn = connect()
    try:
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

    同一份简历原文（content_hash）或手机号已存在 → 原地 UPDATE（id 不变）；
    否则 → INSERT 新行。
    """
    # 姓名兜底：识别不到时统一成“未知姓名”，绝不让模型示例名（如张三）或
    # 板块标题混进库里；真实姓名以用户在界面对 JSON 的核对为准。
    name = (profile.get("name") or "").strip()
    if name.lower() in {"", "未知", "resumeprofile", "resume_profile", "none", "null"}:
        name = "未知姓名"
    profile["name"] = name

    match_result = json.loads(
        match_resume_worker.invoke(
            {"profile_json": json.dumps(profile, ensure_ascii=False)}
        )
    )

    candidate_id, is_update = _write_candidate(
        _build_row(profile, resume_text, match_result)
    )
    return {
        "profile": profile,
        "match": match_result,
        "candidate_id": candidate_id,
        "is_update": is_update,
    }


def get_candidate_id(name: str) -> int | None:
    """按姓名查候选人 id；找不到返回 None（供评分记录关联用）。"""
    conn = connect()
    try:
        row = conn.execute(
            "SELECT id FROM candidates WHERE name = ?", (name,)
        ).fetchone()
        return row["id"] if row else None
    finally:
        conn.close()


def delete_candidate(candidate_id: int):
    """按 id 删除一个候选人，返回是否真的删掉了一行。"""
    conn = connect()
    try:
        with conn:
            cur = conn.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
        return cur.rowcount > 0
    finally:
        conn.close()
