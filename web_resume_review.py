"""简历画像纠错页：分析 → 可编辑画像 → 确认入库（两步走）。
跑：python web_resume_review.py（仓库根目录）
"""

import json

import gradio as gr

from recruit_graph.candidate_store import analyze_profile, save_candidate_from_profile


def analyze_handler(resume_text: str):
    """第一步：只分析不落库，返回可编辑的画像 JSON。"""
    if not resume_text.strip():
        return "", "请先粘贴简历"
    profile = analyze_profile(resume_text)
    profile_json = json.dumps(profile, ensure_ascii=False, indent=2)
    return profile_json, "已分析。检查画像，改完点“确认入库”"


def save_handler(profile_json: str, resume_text: str):
    """第二步：用（改过的）画像入库；已存在则 UPDATE 保持原 id。"""
    if not resume_text.strip() or not profile_json.strip():
        return "请先分析再入库"
    try:
        profile = json.loads(profile_json)
    except json.JSONDecodeError:
        return "画像不是合法 JSON，请检查格式"
    result = save_candidate_from_profile(profile, resume_text)
    m = result["match"]
    verb = "已更新（原记录保留 id）" if result["is_update"] else "已新增"
    return (
        f"{verb}\n"
        f"姓名：{result['profile']['name']}\n"
        f"匹配岗位：{m['matched_job']}　匹配分数：{m['match_score']}\n"
        f"点评：{m['report']}"
    )


with gr.Blocks(title="简历画像纠错入库") as demo:
    gr.Markdown(
        "## 简历画像纠错入库\n先分析 → 检查/修改画像 → 再确认入库。"
        "已存在的候选人会 UPDATE（保留原 id），不会换新行。"
    )
    resume_box = gr.Textbox(
        label="粘贴简历", lines=8, placeholder="把候选人简历文本贴在这里…"
    )
    analyze_btn = gr.Button("① 分析画像", variant="primary")
    profile_box = gr.Textbox(
        label="画像 JSON（可编辑）",
        lines=12,
        placeholder='{\n  "name": "…"',
    )
    save_btn = gr.Button("② 确认入库")
    status = gr.Textbox(label="状态", lines=6, interactive=False)

    analyze_btn.click(analyze_handler, resume_box, [profile_box, status])
    save_btn.click(save_handler, [profile_box, resume_box], status)


if __name__ == "__main__":
    demo.launch()
