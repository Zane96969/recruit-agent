"""JD 管理页：保存 JD、下拉切换、用已存 JD 排名、删除。
跑：python web_jd_manager.py（仓库根目录）
"""

import gradio as gr

from recruit_graph.candidate_store import get_candidate_id
from recruit_graph.jd_store import (
    delete_jd,
    get_jd,
    list_jds,
    list_matchings,
    record_match,
    save_jd,
)
from web_jd_match import rank_for_jd


def _matchings_table() -> list[list]:
    """评分记录转表格：JD名、候选人名、分数、时间。"""
    return [
        [m["jd_name"], m["candidate_name"], m["match_score"], m["created_at"]]
        for m in list_matchings()
    ]


def _refresh_choices() -> list[str]:
    """下拉选项：'岗位名（#id）'，id 用于取原文。"""
    return [f"{jd['jd_name']}（#{jd['id']}）" for jd in list_jds()]


def _jd_id_from_label(label: str) -> int:
    """从 '岗位名（#3）' 里解出 id=3。"""
    return int(label.rsplit("#", 1)[1].rstrip("）"))


def save_handler(jd_name: str, jd_text: str):
    """保存 JD，刷新下拉；底层失败转成界面消息，不把 traceback 甩给使用者。"""
    if not jd_name.strip() or not jd_text.strip():
        return gr.update(choices=_refresh_choices()), "请填岗位名和 JD 原文"
    try:
        save_jd(jd_name.strip(), jd_text.strip())
    except (ValueError, RuntimeError) as e:
        # ValueError=输入用法错（如超长），RuntimeError=系统故障（可重试），都只显示消息
        return gr.update(choices=_refresh_choices()), f"未保存：{e}"
    return gr.update(choices=_refresh_choices()), "已保存：" + jd_name.strip()


def delete_handler(jd_label: str):
    """删除选中的 JD，刷新下拉；底层失败转成界面消息。"""
    if not jd_label:
        return gr.update(choices=_refresh_choices()), "请先选择要删除的 JD"
    try:
        ok = delete_jd(_jd_id_from_label(jd_label))
    except (ValueError, RuntimeError) as e:
        return gr.update(choices=_refresh_choices()), f"未删除：{e}"
    msg = "已删除：" + jd_label if ok else "删除失败"
    return gr.update(choices=_refresh_choices()), msg


def rank_handler(jd_label: str, resumes_text: str):
    """用选中的 JD 给多份简历排名；已入库的候选人顺便记一条评分。"""
    if not jd_label:
        return [], _matchings_table()
    jd_id = _jd_id_from_label(jd_label)
    jd_text = get_jd(jd_id).get("jd_text", "")
    rows = rank_for_jd(jd_text, resumes_text)
    for row in rows:
        candidate_id = get_candidate_id(row[0])
        if candidate_id:
            record_match(jd_id, candidate_id, row[1])
    return rows, _matchings_table()


with gr.Blocks(title="JD 管理") as demo:
    gr.Markdown("## JD 管理\n存 JD → 下拉选择 → 用选中的 JD 给简历排名")
    with gr.Row():
        jd_name_box = gr.Textbox(
            label="岗位名", lines=1, placeholder="AI Agent 开发实习生"
        )
        jd_text_box = gr.Textbox(
            label="JD 原文",
            lines=6,
            placeholder="岗位：…\n地点：…\n薪资：…\n要求：…",
        )
    save_btn = gr.Button("保存 JD")
    jd_dropdown = gr.Dropdown(
        choices=_refresh_choices(),
        label="已存 JD（下拉切换）",
    )
    delete_btn = gr.Button("删除选中的 JD")
    status = gr.Textbox(label="状态", lines=1, interactive=False)
    resumes_box = gr.Textbox(
        label="多份简历（用 --- 分隔）",
        lines=8,
        placeholder="张三，23岁…\n---\n王五，25岁…",
    )
    rank_btn = gr.Button("用选中 JD 排名")
    result_table = gr.Dataframe(
        headers=["姓名", "匹配分", "匹配岗位", "评估意见"],
        datatype=["str", "number", "str", "str"],
        wrap=True,
    )
    history_btn = gr.Button("刷新评分记录")
    history_table = gr.Dataframe(
        headers=["JD", "候选人", "分数", "时间"],
        datatype=["str", "str", "number", "str"],
        label="评分记录（已入库候选人 × 指定JD）",
        value=_matchings_table(),
    )

    save_btn.click(save_handler, [jd_name_box, jd_text_box], [jd_dropdown, status])
    delete_btn.click(delete_handler, [jd_dropdown], [jd_dropdown, status])
    rank_btn.click(
        rank_handler,
        [jd_dropdown, resumes_box],
        [result_table, history_table],
    )
    history_btn.click(lambda: _matchings_table(), outputs=history_table)


if __name__ == "__main__":
    demo.launch()
