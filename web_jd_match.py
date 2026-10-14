"""指定JD简历排名页：粘贴一份JD + 多份简历（---分隔），按匹配分排序。
当场计算，不落库。跑：python web_jd_match.py（仓库根目录）
"""

import json

import gradio as gr

from recruit_graph.graph_tools import analyze_resume_worker, match_resume_by_jd


def rank_for_jd(jd_text: str, resumes_text: str) -> list[list]:
    """按指定 JD 给多份简历打分排序。返回 [[姓名, 分数, 岗位, 意见], ...]。"""
    rows = []
    for resume in [r.strip() for r in resumes_text.split("---") if r.strip()]:
        # 第1步：简历原文 -> 结构化画像
        profile = json.loads(analyze_resume_worker.invoke({"resume_text": resume}))
        # 第2步：画像 + 指定JD -> 匹配结果
        match = json.loads(
            match_resume_by_jd.invoke(
                {
                    "profile_json": json.dumps(profile, ensure_ascii=False),
                    "jd_text": jd_text,
                }
            )
        )
        rows.append(
            [
                profile.get("name"),
                match.get("match_score"),
                match.get("matched_job"),
                match.get("report"),
            ]
        )
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


with gr.Blocks(title="指定JD简历排名") as demo:
    gr.Markdown(
        "## 指定 JD 简历排名\n粘贴一份 JD 和多份简历（用 `---` 分隔），按匹配分从高到低排序。"
    )
    jd_box = gr.Textbox(
        label="目标岗位 JD",
        lines=6,
        placeholder="岗位：AI Agent 开发实习生\n地点：大连\n薪资：3-5K\n要求：…",
    )
    resumes_box = gr.Textbox(
        label="多份简历（用 --- 分隔）",
        lines=12,
        placeholder="张三，23岁…\n---\n王五，25岁…",
    )
    rank_btn = gr.Button("开始排名")
    result_table = gr.Dataframe(
        headers=["姓名", "匹配分", "匹配岗位", "评估意见"],
        datatype=["str", "number", "str", "str"],
        wrap=True,
    )
    rank_btn.click(rank_for_jd, inputs=[jd_box, resumes_box], outputs=result_table)


if __name__ == "__main__":
    demo.launch()
