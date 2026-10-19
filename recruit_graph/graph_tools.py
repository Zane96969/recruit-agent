import os
import re
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from recruit_graph.hybrid_retriever import hybrid_search

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("MODEL", "deepseek-chat")


@tool
def retrieve_jd_info(question: str, use_rerank: bool = True) -> str:
    """
    根据用户问题检索相关的JD信息。
    use_rerank 为 True 时走“BM25+向量粗排 + bge-reranker 精排”，结果更准、稍慢；
    为 False 时只做 BM25+向量粗排，更快，且不依赖精排服务（适合精排不可用时）。
    """
    docs = hybrid_search(question, use_rerank=use_rerank)
    content = [document.page_content for document in docs]
    return "\n".join(content)


class TechStack(BaseModel):
    core_tech: list[str] = Field(description="核心技术栈")
    optional_tech: list[str] = Field(description="可选技术栈", default=None)


class JobList(BaseModel):
    name: str = Field(description="岗位名称")
    salary: str = Field(description="薪资范围")
    location: str = Field(description="工作地点")
    requirements: str = Field(description="岗位要求", default=None)
    tech_stack: TechStack = Field(description="技术栈", default=None)


chat_model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,
    temperature=0,
    # 健壮化：单次请求最长等 60 秒，避免网络抽风时无限挂起；
    # 遇到限流/超时自动重试最多 3 次，把瞬时故障挡在外面。
    timeout=60,
    max_retries=3,
)

structured_model = chat_model.with_structured_output(JobList, method="function_calling")


@tool
def analyze_jd_info(jd_text: str) -> str:
    """
    用户提供一份 JD 原文、要求分析岗位或生成人才画像时使用
    """
    result = structured_model.invoke(
        [
            SystemMessage(
                content="你是一名招聘信息解析专家，请将JD信息提取为结构化数据。对于岗位要求，如果没有明确要求，请返回空字符串。"
            ),
            HumanMessage(content=f"请将JD信息提取为结构化数据。\n{jd_text}"),
        ]
    )
    return result.model_dump_json(ensure_ascii=False)


# ===================== 多智能体：简历分析 + 人岗匹配 =====================
class ResumeProfile(BaseModel):
    """从简历中抽取的候选人画像（简历分析专家的结构化产物）。"""

    name: str = Field(
        description="候选人的真实中文姓名（2-4个汉字），只能从简历原文逐字提取；"
        "简历确实没有姓名时填“未知”，禁止填写示例、英文名、类名或任何编造的名字"
    )
    age: str = Field(description="年龄，简历没写填未知")
    education: str = Field(description="最高学历与专业，简历没写填未知")
    skills: list[str] = Field(
        description="掌握的技能/技术栈列表，只放技术、框架、工具名称（如 Python、LangChain、RAG），不要放项目"
    )
    target_position: str = Field(description="目标岗位，简历没写填未知")
    expected_salary: str = Field(description="期望薪资，简历没写填未知")
    phone: str = Field(description="手机号码（11位数字），简历没写填未知")
    projects: list[str] = Field(
        description="项目经历或个人作品名称列表，如“招聘 Agent 项目”；简历没写项目则返回空列表"
    )


resume_system = SystemMessage(
    content=(
        "你是简历分析专家。只根据简历原文抽取结构化信息："
        "姓名、手机号、年龄、学历、技能、项目经历、目标岗位、期望薪资。"
        "skills 只放技术、框架、工具的名词（如 Python、LangChain、RAG），"
        "项目或作品经历一律放到 projects，不要放进 skills；"
        "简历没写项目时 projects 返回空列表。"
        "简历没写的字段填“未知”，禁止编造。"
        "姓名字段只能从简历原文里逐字提取候选人的真实中文姓名，"
        "简历确实没有姓名时填“未知”，绝不允许照抄示例、自行编造，"
        "也禁止填写英文、类名、ResumeProfile 等字样。"
    )
)

# 常见简历板块/标题词，不是人名，用正则抓姓名时要排除
_NON_NAME_WORDS = {
    "个人简历",
    "求职意向",
    "基本信息",
    "联系方式",
    "教育背景",
    "教育经历",
    "工作经历",
    "工作经验",
    "项目经历",
    "项目经验",
    "技能特长",
    "技能专长",
    "专业技能",
    "自我评价",
    "校园经历",
    "实习经历",
    "岗位匹配",
}


class MatchResult(BaseModel):
    """人岗匹配的结构化结果。"""

    matched_job: str = Field(description="最匹配的岗位名称")
    match_score: int = Field(
        description="匹配度分数，0-100 的整数，只填数字，不要带百分号"
    )
    report: str = Field(description="匹配理由：关键匹配点、差距点、一句建议")


match_system = SystemMessage(
    content=(
        "你是人岗匹配专家。你会收到候选人画像 JSON 和岗位库中的全部 JD，"
        "每份 JD 用 --- 分隔，第一行“职位名称：”后面就是该岗位的完整名称。"
        "请从这些 JD 中选出与画像最匹配的 1 个岗位，并对照岗位要求打分。评分锚点："
        "90-100：满足全部硬性要求，且具备 JD 中“优先/熟悉”的核心方向实战经验"
        "（如 RAG/LangChain/LangGraph、微服务/JVM 调优、完整猎头寻访流程），可直接推荐；"
        "75-84：满足硬性要求，但“优先”项里的关键能力只停留在课程或了解层面，缺少对应项目经验；"
        "60-74：方向沾边、会基础工具，但缺多项核心技能；"
        "40-59：仅学历或通用技能等个别条件沾边，没有对口方向的项目或实习；"
        "0-39：完全不对口，或与库中所有岗位都无关。"
        "硬性规则：1）只会基础语言（如 Python、Java）但缺少该岗位“优先”项的核心实战经验时，分数不得超过 80；"
        "2）对于“专业不限/学历门槛低”的岗位，也要看是否有对口实习或经历（招聘、销售、寻访、沟通等），"
        "完全没有相关经历的，分数不得超过 55；"
        "3）matched_job 必须逐字复制所选 JD 第一行“职位名称：”后的完整岗位名，"
        "禁止填写“JD1”“第一个岗位”等代号，也禁止填候选人自己的求职意向；"
        "4）若候选人与所有 JD 都不对口，matched_job 填“无”，分数给 0-39；"
        "5）不同候选人要严格横向比较、拉开差距，不要都给同一个分；"
        "6）report 写关键匹配点、差距点、一句建议。"
        "只允许依据画像和 JD 内容作答，禁止编造 JD 里不存在的信息。"
    )
)


@tool
def analyze_resume_worker(resume_text: str) -> str:
    """简历分析专家：输入一段简历原文，返回结构化候选人画像 JSON。
    当用户发来简历、或要求“分析这份简历 / 看看这个候选人”时，必须最先调用。
    """
    resume_model = chat_model.with_structured_output(
        ResumeProfile, method="function_calling"
    )
    profile = resume_model.invoke([resume_system, HumanMessage(content=resume_text)])
    # 兜底：模型没抽到姓名时，只认“姓名：XXX”这类明确标签，
    # 避免把“个人简历/求职意向”等板块标题误当姓名；抓不到就留“未知”。
    bad_names = {"resumeprofile", "resume_profile", "", "未知"}
    if profile.name.strip().lower() in bad_names:
        found = re.search(
            r"(?:姓名|名字)\s*[:：]?\s*([\u4e00-\u9fa5]{2,4})", resume_text
        )
        if found and found.group(1) not in _NON_NAME_WORDS:
            profile.name = found.group(1)
        else:
            profile.name = "未知"
    return profile.model_dump_json(ensure_ascii=False)


@tool
def match_jd_worker(state: Annotated[list, InjectedState("messages")]) -> str:
    """人岗匹配专家：简历分析完成后调用。自动读取分析专家回传的画像 JSON，
    从岗位库检索相关 JD 并给出匹配度报告。无需传入任何参数。
    """
    # 从共享状态里取最近一条工具回执（即简历分析专家回传的画像 JSON）
    tool_messages = [m for m in state if m.__class__.__name__ == "ToolMessage"]
    profile_json = tool_messages[-1].content if tool_messages else ""

    # 用画像去向量库检索最相关的 JD
    jd_docs = hybrid_search(profile_json)
    jd_text = "\n---\n".join(doc.page_content for doc in jd_docs)

    answer = chat_model.invoke(
        [
            match_system,
            HumanMessage(
                content=f"【候选人画像】\n{profile_json}\n\n【相关岗位 JD】\n{jd_text}\n请给出匹配报告。"
            ),
        ]
    )
    return answer.content


@tool
def match_resume_worker(profile_json: str) -> str:
    """人岗匹配打分专家：输入候选人画像 JSON，读取岗位库全部 JD 完整原文，
    横向比较后选出最匹配的 1 个岗位，返回结构化结果（岗位、整数分数、理由）的 JSON。
    岗位数量很少时，用完整 JD 全量比对（而非检索碎片），保证岗位名准确、分数有区分度。
    """
    from recruit_graph.jd_store import get_jd, list_jds

    # 岗位只有个位数，直接读全部完整 JD，按 id 升序拼接
    jd_texts = []
    for jd in sorted(list_jds(), key=lambda item: item["id"]):
        jd_texts.append(get_jd(jd["id"])["jd_text"])
    jd_text = "\n---\n".join(jd_texts)
    match_model = chat_model.with_structured_output(
        MatchResult,
        method="function_calling",
    )
    result = match_model.invoke(
        [
            match_system,
            HumanMessage(
                content=f"【候选人画像】\n{profile_json}\n\n【岗位库全部 JD】\n{jd_text}"
            ),
        ]
    )
    return result.model_dump_json(ensure_ascii=False)


match_by_jd_system = SystemMessage(
    content=(
        "你是人岗匹配专家。你会收到候选人画像 JSON 和一份用户指定的目标岗位 JD。"
        "请完成：1）只依据这份 JD 评估候选人，matched_job 填该 JD 的岗位名称；"
        "2）给出匹配度整数分数（0-100）；3）report 写关键匹配点、差距点、一句建议。"
        "只允许依据画像和 JD 内容作答，禁止编造 JD 里不存在的信息。"
    )
)


@tool
def match_resume_by_jd(profile_json: str, jd_text: str) -> str:
    """人岗匹配打分专家（指定JD版）：输入候选人画像 JSON 和目标岗位 JD 原文，
    只依据传入的这份 JD 打分，不做向量库检索。
    当用户指定了某份 JD（而不是让系统自己挑岗位）时使用。
    """
    match_model = chat_model.with_structured_output(
        MatchResult,
        method="function_calling",
    )
    result = match_model.invoke(
        [
            match_by_jd_system,
            HumanMessage(
                content=f"【候选人画像】\n{profile_json}\n\n【目标岗位 JD】\n{jd_text}"
            ),
        ]
    )
    return result.model_dump_json(ensure_ascii=False)


@tool
def analyze_and_match_resume(resume_text: str) -> str:
    """简历分析 + 人岗匹配一站式工具：输入简历原文，先产出候选人画像，
    再自动从岗位库检索相关 JD 并给出匹配度报告。
    当用户发来简历、或要求“分析简历并匹配岗位 / 看看这个候选人适合什么岗”时使用，
    一次调用即可完成，不要再调用其他简历工具。
    """
    # 第一步：复用简历分析专家，得到画像 JSON
    profile_json = analyze_resume_worker.invoke({"resume_text": resume_text})

    # 第二步：拿画像去向量库检索相关 JD
    jd_docs = hybrid_search(profile_json)
    jd_text = "\n---\n".join(doc.page_content for doc in jd_docs)

    # 第三步：匹配专家对照画像与 JD 出报告
    answer = chat_model.invoke(
        [
            match_system,
            HumanMessage(
                content=(
                    f"【候选人画像】\n{profile_json}\n\n【相关岗位 JD】\n{jd_text}\n请给出匹配报告。"
                )
            ),
        ]
    )
    return f"【候选人画像】\n{profile_json}\n\n{answer.content}"


if __name__ == "__main__":
    print(retrieve_jd_info.invoke({"question": "agent开发工程师薪资是多少"}))
    sample_jd_info = (
        "岗位：Python 工程师，地点：成都，薪资：8K-12K，要求会 Python、Django。"
    )
    print(analyze_jd_info.invoke({"jd_text": sample_jd_info}))
