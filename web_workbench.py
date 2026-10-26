# -*- coding: utf-8 -*-
"""统一招聘工作台：一个网页聚合 岗位管理 / 逐份简历初筛 / 知识库问答。

解决真实使用中的痛点：
1. 只能粘纯文本 —— 支持上传 PDF / Word / 图片，也支持直接 Ctrl+V 粘贴截图；
2. 只能多份简历一起 —— 一份一份处理：分析→核对→入库，
   入库后输入框自动清空，已入库的人全部保留，右侧实时累计排名。

跑（仓库根目录）：
    python web_workbench.py
浏览器打开：http://127.0.0.1:7860
"""

import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import gradio as gr
except ModuleNotFoundError as exc:  # 用错解释器时给一句人话，而不是裸 traceback
    if exc.name not in {"gradio", "langchain_core"}:
        raise
    raise SystemExit(
        "\n启动失败：当前这个 Python 没装项目依赖（缺 "
        f"{exc.name}）。\n"
        f"当前解释器：{sys.executable}\n\n"
        "两种解法：\n"
        "  1) 已有虚拟环境 → 用它的 python 启动本文件，例如：\n"
        "     <项目>\\.venv\\Scripts\\python.exe web_workbench.py\n"
        "  2) 还没建 → 在仓库根目录执行：\n"
        "     python -m venv .venv\n"
        "     .\\.venv\\Scripts\\python.exe -m pip install -e .\n"
        "然后 .\\.venv\\Scripts\\Activate.ps1 激活，再 python web_workbench.py\n\n"
        "注意：不要用系统自带的 python.exe 直接跑，它没有项目依赖。\n"
    ) from exc

from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recruit_graph.candidate_store import (
    analyze_profile,
    delete_candidate,
    list_candidates,
    save_candidate_from_profile,
)
from recruit_graph.deep_research import run_deep_research
from recruit_graph.doc_parser import extract_text
from recruit_graph.graph_tools import chat_model
from recruit_graph.hybrid_retriever import hybrid_search
from recruit_graph.jd_store import (
    delete_jd,
    delete_matchings_by_candidate,
    get_jd,
    list_jds,
    save_jd,
    update_jd,
)
from recruit_graph.jd_generator import generate_jd_draft
from recruit_mcp.observability import ObservabilityHandler

FILE_TYPES = [".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp", ".bmp"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


# ===================== 公共小工具：下拉、排名、计数 =====================
# 下拉项统一格式：'岗位名/姓名（#7）'。用正则只认结尾的 "#数字"，
# 而不是按 '#' 切分 —— 岗位名自己就常带 '#'（如「C# 后端开发」），
# 按 '#' 切会在这种名字上切错位；标签没有编号时也要能识别出来，而不是崩。
_LABEL_ID = re.compile(r"#\s*(\d+)\s*[）)]?\s*$")


def cid_from_label(label: str) -> int | None:
    """从 '岗位名/姓名（#7）' 里解出 id=7；解析不出来返回 None（调用方提示刷新列表）。"""
    match = _LABEL_ID.search(label or "")
    return int(match.group(1)) if match else None


def jd_choices() -> list[str]:
    return [f"{j['jd_name']}（#{j['id']}）" for j in list_jds()]


def jd_rows() -> list[list]:
    """岗位库总览表：编号、岗位名（完整不截断）、正文字数、录入时间。"""
    return [
        [j["id"], j["jd_name"], j.get("jd_len", 0), j.get("created_at", "")]
        for j in list_jds()
    ]


def candidate_choices() -> list[str]:
    return [f"{c['name']}（#{c['id']}）" for c in list_candidates()]


def rank_rows(min_score: int = 0) -> list[list]:
    """已入库候选人排名表（分高→低）。"""
    return [
        [
            c["id"],
            c["name"],
            c["match_score"],
            c["matched_job"],
            c.get("target_position") or "",
        ]
        for c in list_candidates(int(min_score))
    ]


def count_md() -> str:
    return f"### 已入库候选人：**{len(list_candidates())}** 人（都会保留，不受清空输入框影响）"


def refresh_all():
    """打开页面 / 手动刷新时，从库实时拉取岗位、候选人下拉与排名。"""
    return (
        gr.update(choices=jd_choices()),
        gr.update(value=jd_rows()),
        gr.update(choices=candidate_choices()),
        count_md(),
        rank_rows(0),
    )


# ===================== Tab 1：岗位管理 =====================
def jd_extract(file):
    """上传岗位文件 / 粘贴截图 → 提取文本；提取新文件即退出“编辑岗位”模式。"""
    if not file:
        return "", "", gr.update(), "请先选择文件或粘贴截图", gr.update()
    try:
        text = extract_text(file)
    except Exception as e:  # noqa: BLE001
        return "", "", gr.update(), f"提取失败：{e}", gr.update()
    # 粘贴的截图是临时文件名（无意义），岗位名留空让用户自己填
    stem = "" if Path(file).suffix.lower() in IMAGE_SUFFIXES else Path(file).stem
    return (
        stem,
        text,
        None,
        "已提取文本，请核对后点「② 保存新岗位」",
        gr.update(value="② 保存新岗位"),
    )


def jd_load(label: str):
    """把选中的已存岗位加载到编辑框，进入“编辑该岗位”模式。"""
    if not label:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            "请先选择要编辑的岗位",
            gr.update(),
        )
    jd_id = cid_from_label(label)
    if jd_id is None:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            "这个选项没有岗位编号（可能是组件状态过期），请点「刷新列表」后重新选择",
            gr.update(),
        )
    jd = get_jd(jd_id)
    if not jd:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            "没有找到该岗位，请刷新列表",
            gr.update(),
        )
    return (
        jd.get("jd_name", ""),
        jd.get("jd_text", ""),
        jd_id,
        f"已加载岗位：{jd.get('jd_name')}（#{jd_id}），可在上方改名/补充后点「② 保存修改」",
        gr.update(value="② 保存修改"),
    )


_NAME_PREFIXES = (
    "职位名称：",
    "职位名称:",
    "岗位名称：",
    "岗位名称:",
    "职位：",
    "职位:",
)


def auto_jd_name(text: str) -> str:
    """岗位名留空时用首行当名字：去掉常见前缀，放宽到 40 字，避免名字被截断。"""
    first = text.splitlines()[0].strip()
    for pre in _NAME_PREFIXES:
        if first.startswith(pre):
            first = first[len(pre) :].strip()
            break
    return first[:40]


def jd_save(name: str, text: str, edit_id):
    """保存岗位：edit_id 为空=新增，有值=更新该岗位；两者都同步写 SQLite + 向量库。"""
    name = (name or "").strip()
    text = (text or "").strip()
    if not text:
        return (
            gr.update(choices=jd_choices()),
            gr.update(),
            gr.update(),
            "请先提取或粘贴岗位原文",
            gr.update(),
        )
    if not name:
        name = auto_jd_name(text)
    if edit_id:
        result = update_jd(int(edit_id), name, text)
        if "error" in result:
            return (
                gr.update(choices=jd_choices()),
                gr.update(),
                gr.update(),
                result["error"],
                gr.update(),
            )
        msg = f"已更新岗位：{name}（编号 #{result['jd_id']}），知识库向量已同步重建"
    else:
        result = save_jd(name, text)
        msg = f"已保存岗位：{name}（编号 #{result['jd_id']}），并写入知识库可检索"
    return (
        gr.update(choices=jd_choices()),
        gr.update(value=jd_rows()),
        None,
        msg,
        gr.update(value="② 保存新岗位"),
    )


def jd_delete(label: str):
    if not label:
        return gr.update(choices=jd_choices()), gr.update(), "请先选择要删除的岗位"
    jd_id = cid_from_label(label)
    if jd_id is None:
        return (
            gr.update(choices=jd_choices()),
            gr.update(),
            "这个选项没有岗位编号（可能是组件状态过期），请点「刷新列表」后重新选择",
        )
    ok = delete_jd(jd_id)
    msg = ("已删除：" + label) if ok else "删除失败"
    return gr.update(choices=jd_choices()), gr.update(value=jd_rows()), msg


def jd_refresh():
    """重新从库读取岗位下拉与总览表（解决重启 / 外部新增后列表不更新）。"""
    return gr.update(choices=jd_choices()), gr.update(value=jd_rows())


def jd_ai_generate(job_name: str, customer_requirements: str):
    """输入岗位名（可附客户要求）→ 检索知识库相似 JD 当范例 → 生成草稿，填进编辑框（不入库）。

    客户明确给的要求会优先写进草稿；草稿只生成不保存，用户补完“待确认”、
    点「② 保存新岗位」才真正入库。
    """
    job_name = (job_name or "").strip()
    requirements = (customer_requirements or "").strip()
    if not job_name:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            "请先输入岗位名称，再点「AI 生成草稿」",
            gr.update(),
        )
    try:
        result = generate_jd_draft(
            job_name,
            n=2,
            customer_requirements=requirements,
        )
    except Exception as e:  # noqa: BLE001
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            f"生成失败：{type(e).__name__}（{e}），可改用上传 / 粘贴方式录入",
            gr.update(),
        )
    if result is None:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            "知识库没有相似岗位范例，第一版无法生成（未来联网兜底），请先录入同类岗位",
            gr.update(),
        )
    used = "、".join(f"{j['jd_name']}（#{j['id']}）" for j in result["examples"])
    req_note = "（已结合客户要求）" if requirements else ""
    return (
        result["job_name"],
        result["jd_text"],
        None,
        f"草稿已生成{req_note}，参考了：{used}。请核对客户要求是否都体现、"
        "把“待确认”补全、删掉套话，再点「② 保存新岗位」",
        gr.update(value="② 保存新岗位"),
    )


# ===================== Tab 2：逐份简历初筛（累积） =====================
def r_extract(file):
    """上传简历文件 / 粘贴截图 → 提取文本。"""
    if not file:
        return "", "请先选择文件或粘贴截图"
    try:
        text = extract_text(file)
    except Exception as e:  # noqa: BLE001
        return "", f"提取失败：{e}"
    return text, "已提取文本，核对后点「② 分析画像」"


def build_narrative(text: str, profile: dict) -> str:
    """根据简历原文和已抽取字段，生成一段自然语言人才画像。"""
    profile_json = json.dumps(profile, ensure_ascii=False)
    reply = chat_model.invoke(
        [
            SystemMessage(
                content=(
                    "你是资深猎头顾问。请依据候选人简历与已抽取的结构化字段，"
                    "写一段 150-250 字、面向用人经理的自然语言人才画像，"
                    "依次覆盖：一句话定位（背景/年限/方向）、核心技能与技术栈、"
                    "主要项目或经历亮点（最多2条，含做法和结果）、明显优势与待核实短板、"
                    "适合的岗位方向。只依据简历，不编造、不夸大；简历没有的信息写"
                    "「简历未体现」；写成通顺的自然段落，不要编号罗列，不要用 emoji。"
                )
            ),
            HumanMessage(content=f"结构化字段：\n{profile_json}\n\n简历原文：\n{text}"),
        ]
    )
    return reply.content.strip()


def r_analyze(text: str):
    """只分析、不入库，返回可编辑画像 JSON + 自然语言画像。"""
    if not (text or "").strip():
        return "", "", "请先提供简历（上传文件 / 粘贴截图 / 直接粘贴）"
    try:
        profile = analyze_profile(text)
        narrative = build_narrative(text, profile)
    except Exception as e:  # noqa: BLE001
        return "", "", f"分析失败：{e}"
    tip = "画像已生成，核对/修改后点「③ 确认入库」"
    if (profile.get("name") or "").strip() in {"", "未知"}:
        tip = (
            "⚠ 没识别到姓名，请在上方 JSON 的 name 字段补上真实姓名，再点「③ 确认入库」"
        )
    return (
        json.dumps(profile, ensure_ascii=False, indent=2),
        narrative,
        tip,
    )


def r_save(profile_json: str, text: str):
    """确认入库：保留前面所有人，清空输入框，刷新计数与排名。"""
    if not (text or "").strip():
        return (
            gr.update(value=None),
            "",
            "",
            "",
            count_md(),
            rank_rows(0),
            gr.update(choices=candidate_choices()),
            "请先提供简历",
        )
    try:
        profile = json.loads(profile_json)
    except Exception:  # noqa: BLE001
        return (
            gr.update(),
            text,
            profile_json,
            gr.update(),
            count_md(),
            rank_rows(0),
            gr.update(choices=candidate_choices()),
            "画像不是合法 JSON，请重新点「② 分析画像」",
        )
    try:
        result = save_candidate_from_profile(profile, text)
    except Exception as e:  # noqa: BLE001
        return (
            gr.update(),
            text,
            profile_json,
            gr.update(),
            count_md(),
            rank_rows(0),
            gr.update(choices=candidate_choices()),
            f"入库失败：{e}",
        )

    profile = result["profile"]
    match = result["match"]
    n = len(list_candidates())
    head = "已更新该候选人" if result.get("is_update") else f"已保留第 {n} 人"
    msg = (
        f"✅ {head}：{profile['name']}｜匹配「{match['matched_job']}」"
        f"{match['match_score']} 分\n点评：{match['report']}\n"
        "输入框已清空，前面的人都在库里，可继续下一份。"
    )
    if profile["name"] == "未知姓名":
        msg += "\n⚠ 这一份没识别到姓名，已暂存为「未知姓名」；建议在右侧删掉后，在画像 JSON 的 name 补真实姓名再入库。"
    # 截图、简历文本、两种画像都清空（方便下一份），右侧计数/排名/下拉刷新
    return (
        gr.update(value=None),
        "",
        "",
        "",
        count_md(),
        rank_rows(0),
        gr.update(choices=candidate_choices()),
        msg,
    )


def r_delete(label: str):
    if not label:
        return (
            gr.update(choices=candidate_choices()),
            rank_rows(0),
            count_md(),
            "请先选择要删除的人",
        )
    cid = cid_from_label(label)
    if cid is None:
        return (
            gr.update(choices=candidate_choices()),
            rank_rows(0),
            count_md(),
            "这个选项没有候选人编号（可能是组件状态过期），请点「刷新排名 / 候选人列表」后重新选择",
        )
    delete_matchings_by_candidate(cid)
    delete_candidate(cid)
    return (
        gr.update(choices=candidate_choices()),
        rank_rows(0),
        count_md(),
        f"已删除 {label}",
    )


# ===================== Tab 3：知识库问答 =====================
async def qa_ask(question: str, mode: str):
    """基于已录入 JD 做混合检索，再让模型只依据资料回答；同时返回本轮可观测指标。"""
    if not (question or "").strip():
        return "请输入问题", ""
    # 这轮专用的记录仪：verbose=False 只在最后汇总，不在控制台逐步刷屏
    started = time.perf_counter()
    handler = ObservabilityHandler(run_title="知识库问答", verbose=False)
    use_rerank = mode.startswith("精排")
    mode_tag = "精排（BM25+向量+重排）" if use_rerank else "粗排（BM25+向量）"
    docs = hybrid_search(question, use_rerank=use_rerank)
    if not docs:
        return "岗位知识库是空的，请先在【岗位管理】录入岗位。", ""

    context = "\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))
    sources = set()
    for d in docs:
        # 出处直接读检索结果的 metadata（jd_id 由向量层写入），
        # 不再解析 "jd3_5" 这种块 id —— 那是实现细节，格式一变就会标错出处。
        jd_id = d.metadata.get("jd_id")
        if jd_id is None:
            continue
        jd = get_jd(jd_id)
        if jd:
            sources.add(jd.get("jd_name"))
    # 用 ainvoke（异步调用）并挂上回调：模型每开始/结束一次，记录仪都会自动计数
    reply = await chat_model.ainvoke(
        [
            SystemMessage(
                content="你是招聘助手，只能依据下面给出的岗位资料回答；"
                "资料里没有的内容明确说「资料不足」，不要编造。"
            ),
            HumanMessage(content=f"岗位资料：\n{context}\n\n问题：{question}"),
        ],
        config={"callbacks": [handler]},
    )
    answer = reply.content.strip()
    answer += f"\n\n— 检索方式：{mode_tag}，命中 {len(docs)} 段"
    if sources:
        answer += "｜出处：" + "、".join(sources)
    # 复用记录仪的统一指标格式，再补上这轮特有的检索命中段数
    total = time.perf_counter() - started
    metrics = handler.metrics_text(total)
    metrics += f"｜命中 {len(docs)} 段（{mode_tag}）"
    return answer, metrics


async def deep_research_ask(question: str):
    """深度研究：多智能体（主管规划 → 两个研究员并行检索 → 综合报告）。

    返回 (报告, 过程轨迹, 运行指标)，界面分别填进三个框。
    """
    if not (question or "").strip():
        return "请输入要研究的岗位或主题。", "", ""
    started = time.perf_counter()
    handler = ObservabilityHandler(run_title="深度研究", verbose=False)
    result = await run_deep_research(question, handler=handler)
    total = time.perf_counter() - started
    report = result["report"]
    if result["files"]:
        report += "\n\n— 研究员共享文件：" + "、".join(result["files"])
    return report, result["trace"], handler.metrics_text(total)


# ===================== 界面 =====================
CUSTOM_CSS = """
.gradio-container {max-width: 1240px !important;}
.app-header {
  background: linear-gradient(135deg, #115e59 0%, #0d9488 55%, #14b8a6 100%);
  border-radius: 18px; padding: 26px 32px; margin: 6px 0 20px 0;
  color: #fff; box-shadow: 0 12px 32px rgba(13,148,136,0.28);
}
.app-title {font-size: 28px; font-weight: 800; letter-spacing: 1px;}
.app-sub {margin-top: 8px; font-size: 15px; opacity: .92;}
.app-tags {margin-top: 16px; display: flex; gap: 10px; flex-wrap: wrap;}
.app-tags span {
  background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.34);
  padding: 5px 14px; border-radius: 999px; font-size: 13px;
}
.card {
  background: #fff; border: 1px solid #e6edf3; border-radius: 14px;
  padding: 16px 18px; margin-bottom: 10px;
  box-shadow: 0 1px 2px rgba(15,23,42,0.05), 0 6px 18px rgba(15,23,42,0.05);
}
.section-title {margin-top: 6px;}
.section-title h3 {margin: 2px 0 4px 0; color: #0f766e; font-weight: 700;}
button.gr-button-primary, .gr-button-primary {
  background: linear-gradient(135deg, #0d9488, #0f766e) !important;
  border: none !important; box-shadow: 0 4px 12px rgba(13,148,136,0.25) !important;
}
button.gr-button-primary:hover, .gr-button-primary:hover {filter: brightness(1.07);}
.tab-nav {gap: 6px;}
.tab-nav button {font-size: 15px; font-weight: 600;}
.table-wrap thead th, thead th {
  background: #f0fdfa !important; color: #0f766e !important; font-weight: 700 !important;
}
footer {display: none !important;}
"""

brand = gr.themes.Color(
    c50="#f0fdfa",
    c100="#ccfbf1",
    c200="#99f6e4",
    c300="#5eead4",
    c400="#2dd4bf",
    c500="#14b8a6",
    c600="#0d9488",
    c700="#0f766e",
    c800="#115e59",
    c900="#134e4a",
    c950="#042f2e",
)

_LIGHT_BG = "#f1f5f9"
_CARD = "#ffffff"
_BORDER = "#e6edf3"
_INK = "#0f172a"
_SUB = "#64748b"
_INPUT_BG = "#f8fafc"
_INPUT_BORDER = "#cbd5e1"
_TEAL = "#0f766e"
_TEAL_SOFT = "#f0fdfa"

theme = gr.themes.Soft(
    primary_hue=brand,
    neutral_hue="slate",
    font=[
        "Microsoft YaHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "system-ui",
        "sans-serif",
    ],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    # 无论设备是亮色还是暗色模式，都强制呈现明亮专业的浅色界面
    body_background_fill=_LIGHT_BG,
    body_background_fill_dark=_LIGHT_BG,
    block_background_fill=_CARD,
    block_background_fill_dark=_CARD,
    block_border_color=_BORDER,
    block_border_color_dark=_BORDER,
    body_text_color=_INK,
    body_text_color_dark=_INK,
    body_text_color_subdued=_SUB,
    body_text_color_subdued_dark=_SUB,
    block_title_text_color=_INK,
    block_title_text_color_dark=_INK,
    block_label_text_color=_TEAL,
    block_label_text_color_dark=_TEAL,
    block_label_background_fill=_TEAL_SOFT,
    block_label_background_fill_dark=_TEAL_SOFT,
    input_background_fill=_INPUT_BG,
    input_background_fill_dark=_INPUT_BG,
    input_border_color=_INPUT_BORDER,
    input_border_color_dark=_INPUT_BORDER,
    table_even_background_fill=_CARD,
    table_even_background_fill_dark=_CARD,
    table_odd_background_fill=_INPUT_BG,
    table_odd_background_fill_dark=_INPUT_BG,
    button_secondary_background_fill=_CARD,
    button_secondary_background_fill_dark=_CARD,
    button_secondary_text_color=_TEAL,
    button_secondary_text_color_dark=_TEAL,
    button_secondary_border_color="#5eead4",
    button_secondary_border_color_dark="#5eead4",
    # 基础表面 / 面板 / 选中态：dark 下也强制浅色，修表格工具栏与选中行
    background_fill_primary=_CARD,
    background_fill_primary_dark=_CARD,
    background_fill_secondary=_LIGHT_BG,
    background_fill_secondary_dark=_LIGHT_BG,
    color_accent_soft=_TEAL_SOFT,
    color_accent_soft_dark=_TEAL_SOFT,
    border_color_primary=_BORDER,
    border_color_primary_dark=_BORDER,
    border_color_accent="#5eead4",
    border_color_accent_dark="#5eead4",
    border_color_accent_subdued="#ccfbf1",
    border_color_accent_subdued_dark="#ccfbf1",
    panel_background_fill=_CARD,
    panel_background_fill_dark=_CARD,
    panel_border_color=_BORDER,
    panel_border_color_dark=_BORDER,
    table_text_color=_INK,
    table_text_color_dark=_INK,
    checkbox_background_color=_CARD,
    checkbox_background_color_dark=_CARD,
    block_info_text_color=_TEAL,
    block_info_text_color_dark=_TEAL,
    code_background_fill=_LIGHT_BG,
    code_background_fill_dark=_LIGHT_BG,
    block_shadow="0 1px 2px rgba(15,23,42,0.05), 0 6px 18px rgba(15,23,42,0.05)",
    button_large_text_weight="600",
)

with gr.Blocks(title="招聘智能体工作台") as demo:
    gr.HTML("""
        <div class="app-header">
          <div class="app-title">招聘智能体工作台</div>
          <div class="app-sub">岗位知识库 · 简历智能初筛 · 混合检索问答</div>
          <div class="app-tags">
            <span>岗位 JD 管理</span>
            <span>简历逐份初筛与排名</span>
            <span>知识库问答</span>
          </div>
        </div>
        """)

    with gr.Tab("岗位管理"):
        jd_edit_state = gr.State(value=None)
        gr.Markdown("#### 岗位库总览", elem_classes="section-title")
        with gr.Group(elem_classes="card"):
            jd_table = gr.Dataframe(
                headers=["编号", "岗位名（完整）", "正文字数", "录入时间"],
                datatype=["number", "str", "number", "str"],
                value=jd_rows(),
                label="长名字自动换行；新增/删除/编辑后点「刷新列表」更新",
                wrap=True,
                max_height=230,
                show_label=False,
            )
        gr.Markdown("#### 录入 / 修改岗位", elem_classes="section-title")
        with gr.Row(elem_classes="card"):
            with gr.Column():
                jd_file = gr.File(
                    label="上传岗位文件（PDF / Word / 图片）",
                    file_types=FILE_TYPES,
                    type="filepath",
                )
                jd_image = gr.Image(
                    label="或者：点一下这里，再按 Ctrl+V 粘贴岗位截图",
                    type="filepath",
                    format="png",
                    sources=["upload", "clipboard"],
                    height=200,
                )
            with gr.Column():
                jd_name_box = gr.Textbox(label="岗位名（留空则自动取首行）", lines=1)
                jd_extract_btn = gr.Button("① 从文件提取文本", size="lg")
                jd_save_btn = gr.Button("② 保存新岗位", variant="primary", size="lg")
        gr.Markdown(
            "#### AI 生成 JD 草稿（不联网，参考知识库相似岗位）",
            elem_classes="section-title",
        )
        with gr.Row(elem_classes="card"):
            jd_gen_box = gr.Textbox(
                label="岗位名称",
                placeholder="例如：AI Agent 开发工程师",
                lines=1,
                scale=3,
            )
            jd_gen_btn = gr.Button("AI 生成草稿", variant="primary", size="lg")
        jd_req_box = gr.Textbox(
            label="客户给的要求（选填，会优先采纳，越具体越好）",
            placeholder=(
                "例如：base上海；本科及以上、5年以上；薪资30-45K·15薪；"
                "必须做过 Agent / RAG；招聘2人，向算法负责人汇报；金融行业背景优先"
            ),
            lines=4,
        )
        jd_text_box = gr.Textbox(
            label="岗位原文（草稿会填到这里，补全“待确认”后保存；也可直接粘贴）",
            lines=8,
        )
        jd_status = gr.Textbox(label="状态", interactive=False)
        gr.Markdown("#### 管理已有岗位", elem_classes="section-title")
        with gr.Row(elem_classes="card"):
            jd_dropdown = gr.Dropdown(
                label="已存岗位（选中后可加载修改）", choices=jd_choices(), scale=3
            )
            jd_refresh_btn = gr.Button("刷新列表", size="md")
            jd_load_btn = gr.Button("加载到编辑框", size="md")
            jd_delete_btn = gr.Button("删除选中岗位", variant="stop")

    with gr.Tab("简历初筛（逐份累积）"):
        gr.Markdown("#### 逐份初筛并自动累计排名", elem_classes="section-title")
        gr.Markdown(
            "上传文件 / 直接 Ctrl+V 粘贴截图 → ①提取 → ②分析画像 → ③确认入库；"
            "入库后自动清空、已入库的人全部保留，右侧实时排名。"
        )
        with gr.Row():
            with gr.Column(scale=1, elem_classes="card"):
                r_file = gr.File(
                    label="上传简历文件（PDF / Word / 图片）",
                    file_types=FILE_TYPES,
                    type="filepath",
                )
                r_image = gr.Image(
                    label="或者：点一下这里，再按 Ctrl+V 粘贴简历截图",
                    type="filepath",
                    format="png",
                    sources=["upload", "clipboard"],
                    height=200,
                )
                r_extract_btn = gr.Button("① 从文件提取文本", size="lg")
                r_text = gr.Textbox(
                    label="简历文本（提取后可编辑，也可直接粘贴）", lines=8
                )
                r_analyze_btn = gr.Button("② 分析画像（不入库）", size="lg")
                r_profile = gr.Textbox(label="画像 JSON（可编辑核对）", lines=10)
                r_narrative = gr.Textbox(label="人才画像（自然语言，可编辑）", lines=8)
                r_save_btn = gr.Button(
                    "③ 确认入库（保留前面所有人）", variant="primary", size="lg"
                )
                r_status = gr.Textbox(label="结果", interactive=False, lines=5)
            with gr.Column(scale=1, elem_classes="card"):
                r_count = gr.Markdown(value=count_md())
                r_rank = gr.Dataframe(
                    headers=["编号", "姓名", "匹配分", "匹配岗位", "目标岗位"],
                    datatype=["number", "str", "number", "str", "str"],
                    value=rank_rows(0),
                    label="候选人排名（分高→低，已入库的都在）",
                )
                with gr.Row():
                    r_dropdown = gr.Dropdown(
                        label="要删除的人", choices=candidate_choices()
                    )
                    r_delete_btn = gr.Button("删除此人", variant="stop")
                r_refresh_btn = gr.Button("刷新排名 / 候选人列表")

    with gr.Tab("知识库问答"):
        with gr.Column(elem_classes="card"):
            gr.Markdown(
                "#### 问岗位知识（基于已录入的 JD）", elem_classes="section-title"
            )
            q_box = gr.Textbox(
                label="你的问题",
                lines=2,
                placeholder="例如：AI Agent 实习岗要求会哪些框架？",
            )
            with gr.Row():
                q_rank = gr.Radio(
                    choices=["精排（更准，推荐）", "粗排（更快）"],
                    value="精排（更准，推荐）",
                    label="检索方式",
                )
                q_btn = gr.Button("提问", variant="primary")
            q_ans = gr.Textbox(label="回答", lines=12)
            q_metrics = gr.Textbox(label="运行指标（可观测）", lines=3)

    with gr.Tab("深度研究"):
        with gr.Column(elem_classes="card"):
            gr.Markdown(
                "#### 多智能体深度研究（主管规划 → 两研究员并行检索 → 综合报告）",
                elem_classes="section-title",
            )
            d_box = gr.Textbox(
                label="要研究的岗位 / 主题",
                lines=2,
                placeholder="例如：AI Agent 开发实习需要会什么？",
            )
            d_btn = gr.Button("开始深度研究", variant="primary")
            d_report = gr.Textbox(label="最终报告（带来源引用）", lines=16)
            d_trace = gr.Textbox(label="过程轨迹（主管规划 / 派活 / 回报）", lines=10)
            d_metrics = gr.Textbox(label="运行指标（可观测）", lines=3)

    # 事件绑定 —— 岗位
    jd_extract_btn.click(
        jd_extract,
        inputs=jd_file,
        outputs=[jd_name_box, jd_text_box, jd_edit_state, jd_status, jd_save_btn],
    )
    jd_image.change(
        jd_extract,
        inputs=jd_image,
        outputs=[jd_name_box, jd_text_box, jd_edit_state, jd_status, jd_save_btn],
    )
    jd_load_btn.click(
        jd_load,
        inputs=jd_dropdown,
        outputs=[jd_name_box, jd_text_box, jd_edit_state, jd_status, jd_save_btn],
    )
    jd_save_btn.click(
        jd_save,
        inputs=[jd_name_box, jd_text_box, jd_edit_state],
        outputs=[jd_dropdown, jd_table, jd_edit_state, jd_status, jd_save_btn],
    )
    jd_delete_btn.click(
        jd_delete, inputs=jd_dropdown, outputs=[jd_dropdown, jd_table, jd_status]
    )
    jd_refresh_btn.click(jd_refresh, outputs=[jd_dropdown, jd_table])
    jd_gen_btn.click(
        jd_ai_generate,
        inputs=[jd_gen_box, jd_req_box],
        outputs=[jd_name_box, jd_text_box, jd_edit_state, jd_status, jd_save_btn],
    )

    # 事件绑定 —— 简历
    r_extract_btn.click(r_extract, inputs=r_file, outputs=[r_text, r_status])
    r_image.change(r_extract, inputs=r_image, outputs=[r_text, r_status])
    r_analyze_btn.click(
        r_analyze, inputs=r_text, outputs=[r_profile, r_narrative, r_status]
    )
    r_save_btn.click(
        r_save,
        inputs=[r_profile, r_text],
        outputs=[
            r_image,
            r_text,
            r_profile,
            r_narrative,
            r_count,
            r_rank,
            r_dropdown,
            r_status,
        ],
    )
    r_delete_btn.click(
        r_delete,
        inputs=r_dropdown,
        outputs=[r_dropdown, r_rank, r_count, r_status],
    )
    r_refresh_btn.click(
        lambda: (
            gr.update(choices=candidate_choices()),
            count_md(),
            rank_rows(0),
        ),
        outputs=[r_dropdown, r_count, r_rank],
    )

    # 事件绑定 —— 问答（回答 + 运行指标两个输出）
    q_btn.click(qa_ask, inputs=[q_box, q_rank], outputs=[q_ans, q_metrics])

    # 事件绑定 —— 深度研究（报告 + 过程轨迹 + 运行指标三个输出）
    d_btn.click(deep_research_ask, inputs=d_box, outputs=[d_report, d_trace, d_metrics])

    # 页面每次打开 / 刷新，都从库实时拉取最新岗位与候选人，避免显示旧快照
    demo.load(refresh_all, outputs=[jd_dropdown, jd_table, r_dropdown, r_count, r_rank])


if __name__ == "__main__":
    # 公网分享：设置环境变量 GRADIO_SHARE=1 才生成公网链接；
    # 同时必须设置 APP_USER / APP_PWD 开启访问密码，避免真实简历裸奔。
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    auth_user = os.environ.get("APP_USER")
    auth_pwd = os.environ.get("APP_PWD")
    auth = (auth_user, auth_pwd) if auth_user and auth_pwd else None
    demo.launch(
        server_port=7860,
        theme=theme,
        css=CUSTOM_CSS,
        share=share,
        auth=auth,
    )
