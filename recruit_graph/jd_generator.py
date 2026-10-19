"""岗位 JD 生成器：只给岗位名称，从知识库找最相似的整份 JD 当范例，
再让大模型照着范例的结构写一份新 JD（不联网、只靠本地知识库）。"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from recruit_graph.hybrid_retriever import hybrid_search
from recruit_graph.jd_store import get_jd, list_jds

load_dotenv()
_API_KEY = os.getenv("API_KEY")
_BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com")
_MODEL = os.getenv("MODEL", "deepseek-chat")


# ============ 第一步：检索最相似的整份 JD 当范例 ============
def _chunk_id_to_jd_id(chunk_id: str) -> int:
    """把向量块 id（形如 'jd3_5'）还原成它所属的岗位 id（数字 3）。"""
    head = chunk_id.split("_")[0]  # 'jd3_5' 按下划线切开取第一段 → 'jd3'
    return int(head.replace("jd", ""))  # 去掉开头的 'jd'，剩下的转成整数 3


def retrieve_examples(job_name: str, n: int = 2) -> list[dict]:
    """输入岗位名，返回 n 份最相似的整份 JD 当范例（去重，每份含岗位名+全文）。

    知识库为空或一份都检索不到时，返回空列表 []，交给上层提示用户。
    """
    # 第 0 组：知识库一份 JD 都没有就直接返回空，不调用检索（空库检索会报错）
    if not list_jds():
        return []

    # 第一组：混合检索捞一批最相似的"句子块"（每份 JD 有很多块，所以多捞几块）
    # 关掉精排（use_rerank=False）：只找同类岗位，粗排足够，还省一次模型调用
    docs = hybrid_search(
        job_name,
        candidate_n=10,
        top_k=n * 3,
        use_rerank=False,
    )

    # 第二组：从块 id 还原岗位 id，按出现顺序去重
    # （同一份 JD 会有多个块排进来，范例只保留一份，不要重复）
    jd_ids = []
    for doc in docs:
        jd_id = _chunk_id_to_jd_id(doc.metadata["id"])
        if jd_id not in jd_ids:
            jd_ids.append(jd_id)

    # 第三组：按岗位 id 去 SQLite 取整份 JD 原文，只取前 n 份
    examples = []
    for jd_id in jd_ids[:n]:
        whole_jd = get_jd(jd_id)
        if whole_jd:
            examples.append(whole_jd)
    return examples


# ============ 第二步：让大模型照着范例生成结构化 JD 草稿 ============
class GeneratedJD(BaseModel):
    """大模型生成的 JD 草稿（结构化字段，便于后面人工编辑和拼成正文入库）。"""

    job_name: str = Field(description="岗位名称，与用户输入一致")
    education: str = Field(description="学历要求，如 本科/硕士；无法判断填 待确认")
    experience: str = Field(description="工作年限要求，如 三年以上；无法判断填 待确认")
    location: str = Field(description="工作地点；属于客户信息、无法推断时填 待确认")
    salary: str = Field(description="薪资范围；无法确定填 待确认，禁止编造数字")
    headcount: str = Field(description="招聘人数；客户未提供填 待确认，禁止编造")
    responsibilities: list[str] = Field(
        description="岗位职责列表，每条一句话、具体可执行，4-8 条"
    )
    requirements: list[str] = Field(
        description="任职要求列表，每条一句话、具体可执行，4-8 条"
    )


# 生成要一点点发挥，所以温度（temperature）比纯抽取的 0 稍高；其余超时/重试沿用项目配置
_gen_model = ChatOpenAI(
    api_key=_API_KEY,
    base_url=_BASE_URL,
    model=_MODEL,
    temperature=0.3,
    timeout=60,
    max_retries=3,
)
# 让模型直接返回符合 GeneratedJD 结构的对象（function_calling 是它支持的结构化输出方式）
_gen_structured = _gen_model.with_structured_output(
    GeneratedJD,
    method="function_calling",
)

_GENERATE_SYSTEM = SystemMessage(
    content=(
        "你是一名资深招聘专家，负责为猎头撰写岗位 JD（职位描述）。"
        "你会收到一个目标岗位名称、客户提出的具体要求（可能为空），以及若干份来自知识库的真实 JD 范例。"
        "请参照范例的【板块结构、颗粒度、措辞风格】，并结合客户要求，为目标岗位写一份新的 JD。必须遵守："
        "1) 客户明确提出的要求是最高优先级事实：薪资、地点、学历、年限、招聘人数、团队/汇报对象、"
        "硬性技能等，凡客户给了就必须准确写进对应字段或条目，不得再标“待确认”、不得遗漏或改写；"
        "2) 只借范例的结构和写法，严禁照抄范例里的公司名、部门名、具体业务名、岗位层级数字、日期；"
        "3) 所有职责和要求必须围绕目标岗位和客户要求重新撰写，不能把范例的条目原样搬过来；"
        "客户点名的硬性条件必须体现在岗位职责或任职要求中；"
        "4) 客户没有提供、又无法按行业惯例合理推断的客户专属信息（薪资、地点、招聘人数、部门等），"
        "一律填“待确认”，禁止编造；"
        "5) 岗位职责和任职要求各写 4-8 条，每条具体、可执行，用行业通用表述；"
        "6) 学历、年限若客户已给就照填；没给但能按行业惯例合理推断就给常见要求，无法判断填“待确认”。"
    )
)


def _build_examples_text(examples: list[dict]) -> str:
    """把若干整份 JD 范例拼成一段带编号的参考文本（每份额外标出岗位名）。"""
    blocks = []
    for i, jd in enumerate(examples, start=1):
        blocks.append(f"【范例{i}】岗位：{jd['jd_name']}\n{jd['jd_text']}")
    return "\n\n".join(blocks)


def generate_jd(
    job_name: str, examples: list[dict], customer_requirements: str = ""
) -> GeneratedJD | None:
    """输入岗位名 + 整份范例 + 客户要求，让大模型返回结构化 JD 草稿。

    第一版只做"本地范例生成"：没有范例就返回 None（不硬写），
    等第二阶段再加"本地没有就联网"的兜底。
    customer_requirements 是客户给的具体要求（可空），给了就优先采纳。
    """
    if not examples:
        return None

    # 第一组：客户要求为空时给一句占位说明，提醒模型客户信息仍标“待确认”
    requirements = (customer_requirements or "").strip()
    if requirements:
        req_block = requirements
    else:
        req_block = "（客户暂未提供额外要求，凡客户才知道的信息一律填“待确认”）"

    # 第二组：把目标岗位名、客户要求、范例全文一起放进人类消息（HumanMessage）
    human = HumanMessage(
        content=(
            f"目标岗位名称：{job_name}\n\n"
            f"客户提出的具体要求（最高优先级，必须据此撰写）：\n{req_block}\n\n"
            f"以下是知识库中相似岗位的真实 JD 范例：\n{_build_examples_text(examples)}\n\n"
            "请参照范例结构、结合客户要求，为目标岗位写一份新的 JD。"
        )
    )

    # 第三组：系统消息定规则 + 人类消息给数据，发给结构化模型，直接得到 GeneratedJD
    return _gen_structured.invoke([_GENERATE_SYSTEM, human])


# ============ 第三步：把结构化草稿拼成可入库的 JD 正文，并提供一步生成入口 ============
def format_jd(draft: GeneratedJD) -> str:
    """把结构化草稿拼成一段和知识库风格一致的 JD 正文（第一行是岗位名）。"""
    # 第一组：头部基础信息（学历/年限/地点/薪资），逐行列出
    lines = [
        draft.job_name,
        "",
        f"学历要求： {draft.education}",
        f"工作年限要求： {draft.experience}",
        f"工作地点： {draft.location}",
        f"薪资范围： {draft.salary}",
        f"招聘人数： {draft.headcount}",
        "",
        "职位详情",
        "",
        "岗位描述",
    ]

    # 第二组：岗位职责，每条加项目符号 •，结尾没有句号就补一个（方便后续按句号切块）
    for item in draft.responsibilities:
        item = item.strip()
        if not item.endswith("。"):
            item += "。"
        lines.append(f"• {item}")

    # 第三组：岗位要求，同样处理
    lines.append("")
    lines.append("岗位要求")
    for item in draft.requirements:
        item = item.strip()
        if not item.endswith("。"):
            item += "。"
        lines.append(f"• {item}")

    return "\n".join(lines)


def generate_jd_draft(
    job_name: str, n: int = 2, customer_requirements: str = ""
) -> dict | None:
    """一步生成：检索范例 → 结合客户要求生成 → 拼成正文（给界面直接调用）。

    返回 {job_name, examples, draft, jd_text}；知识库没有相似范例时返回 None。
    注意：此时只生成、不入库；要等用户在界面补完"待确认"并点保存，才调 save_jd。
    """
    # 第一组：先找范例，再把客户要求一并传给生成；任一环节没有范例就返回 None
    examples = retrieve_examples(job_name, n=n)
    draft = generate_jd(
        job_name,
        examples,
        customer_requirements=customer_requirements,
    )
    if draft is None:
        return None

    # 第二组：拼成正文，连同用到的范例和草稿一起返回（范例可展示给用户看来源）
    return {
        "job_name": draft.job_name,
        "examples": examples,
        "draft": draft,
        "jd_text": format_jd(draft),
    }
