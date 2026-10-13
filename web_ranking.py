import gradio as gr

from recruit_graph.candidate_store import save_candidate, list_candidates


def get_table(min_score):
    """按最低分查候选人，转成表格要的二维列表。"""
    rows = list_candidates(int(min_score))
    return [[r["name"], r["match_score"], r["matched_job"]] for r in rows]


def add_resume(resume_text):
    """贴简历 → 分析打分入库 → 返回结果文字，同时刷新排名。"""
    result = save_candidate(resume_text)
    p = result["profile"]
    m = result["match"]
    summary = (
        f"✅ 已入库\n"
        f"姓名：{p['name']}\n"
        f"目标岗位：{p['target_position']}\n"
        f"技能：{','.join(p['skills'])}\n"
        f"匹配岗位：{m['matched_job']}\n"
        f"匹配分数：{m['match_score']}\n"
        f"点评：{m['report']}"
    )
    return summary, get_table(0)


with gr.Blocks(title="招聘候选人智能匹配") as demo:
    gr.Markdown("# 招聘候选人智能匹配系统")
    with gr.Row():
        # 左栏：贴简历 → 分析入库
        with gr.Column():
            resume_box = gr.Textbox(
                label="粘贴简历", lines=12, placeholder="把候选人简历文本贴在这里…"
            )
            add_btn = gr.Button("分析并入库", variant="primary")
            result_box = gr.Textbox(label="分析结果", lines=10)

        # 右栏：滑块筛选 + 排名表格
        with gr.Column():
            min_score = gr.Slider(0, 100, value=0, step=5, label="最低匹配分")
            refresh_btn = gr.Button("刷新排名")
            ranking = gr.Dataframe(
                headers=["姓名", "匹配分", "匹配岗位"],
                datatype=["str", "number", "str"],
                label="候选人排名（分高→低）",
                value=get_table(0),
            )

    # 绑定事件：点按钮 / 拖滑块时，跑哪个函数、用什么输入、结果放哪
    add_btn.click(add_resume, inputs=resume_box, outputs=[result_box, ranking])
    refresh_btn.click(get_table, inputs=min_score, outputs=ranking)
    min_score.change(get_table, inputs=min_score, outputs=ranking)


if __name__ == "__main__":
    demo.launch()
