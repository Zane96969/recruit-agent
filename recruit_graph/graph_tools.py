import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langchain_openai import  OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import InjectedState

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("MODEL", "deepseek-chat")

embeddings = OpenAIEmbeddings(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="BAAI/bge-large-zh-v1.5",
    check_embedding_ctx_length=False)

REPO_ROOT = Path(__file__).resolve().parent.parent

vectorstore = Chroma(
    collection_name="job_jd",
    persist_directory=str(REPO_ROOT / "chroma_db"),
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
@tool
def retrieve_jd_info(question: str) -> str:
    """
    根据用户问题检索相关的JD信息。
    """
    docs = retriever.invoke(question)
    content=[document.page_content  for document in docs]
    return "\n".join(content)

class TechStack(BaseModel):
    core_tech: list[str] = Field(description="核心技术栈")
    optional_tech: list[str] = Field(description="可选技术栈", default=None)

class JobList(BaseModel):
    name: str = Field(description="岗位名称")
    salary: str = Field( description="薪资范围")
    location: str = Field(description="工作地点")
    requirements: str = Field(description="岗位要求", default=None)
    tech_stack: TechStack = Field(description="技术栈", default=None)
    
chat_model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL
)
    
structured_model = chat_model.with_structured_output(
    JobList, method="function_calling"
)
@tool
def analyze_jd_info(jd_text: str) -> str: 
    """
    用户提供一份 JD 原文、要求分析岗位或生成人才画像时使用
    """
    result = structured_model.invoke([SystemMessage(content="你是一名招聘信息解析专家，请将JD信息提取为结构化数据。对于岗位要求，如果没有明确要求，请返回空字符串。"),
                                 HumanMessage(content=f"请将JD信息提取为结构化数据。\n{jd_text}")])
    return result.model_dump_json(ensure_ascii=False)


# ===================== 多智能体：简历分析 + 人岗匹配 =====================
class ResumeProfile(BaseModel):
    """从简历中抽取的候选人画像（简历分析专家的结构化产物）。"""

    name: str = Field(description="候选人的真实中文姓名，例如张三；禁止填写英文、类名或 ResumeProfile 字样")
    age: str = Field(description="年龄，简历没写填未知")
    education: str = Field(description="最高学历与专业，简历没写填未知")
    skills: list[str] = Field(description="掌握的技能/技术栈列表")
    target_position: str = Field(description="目标岗位，简历没写填未知")
    expected_salary: str = Field(description="期望薪资，简历没写填未知")


resume_system = SystemMessage(content=(
    "你是简历分析专家。只根据简历原文抽取结构化信息："
    "姓名、年龄、学历、技能、目标岗位、期望薪资。"
    "简历没写的字段填“未知”，禁止编造。"
    "姓名字段必须填候选人的真实中文姓名（如张三），"
    "禁止填写任何英文、类名或 ResumeProfile 字样。"
))

match_system = SystemMessage(content=(
    "你是人岗匹配专家。你会收到候选人画像 JSON 和从岗位库检索到的若干份 JD。"
    "请完成：1）从检索到的 JD 中选出与画像最匹配的 1 个岗位；"
    "2）给出匹配度百分比（0-100%）；3）列出关键匹配点（技能、学历、方向等）；"
    "4）列出差距点（如期望薪资与岗位薪资范围不符、经验或技能缺口）；"
    "5）给出一句求职/招聘建议。"
    "只允许依据画像和 JD 内容作答，禁止编造 JD 里不存在的信息。"
))


@tool
def analyze_resume_worker(resume_text: str) -> str:
    """简历分析专家：输入一段简历原文，返回结构化候选人画像 JSON。
    当用户发来简历、或要求“分析这份简历 / 看看这个候选人”时，必须最先调用。
    """
    resume_model = chat_model.with_structured_output(
        ResumeProfile, method="function_calling"
    )
    profile = resume_model.invoke([resume_system, HumanMessage(content=resume_text)])
    # 兜底：个别模型会把 Pydantic 类名误填进姓名字段，发现后纠正为“未知”
    if profile.name.strip().lower() in {"resumeprofile", "resume_profile", ""}:
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
    jd_docs = retriever.invoke(profile_json)
    jd_text = "\n---\n".join(doc.page_content for doc in jd_docs)

    answer = chat_model.invoke([
        match_system,
        HumanMessage(content=f"【候选人画像】\n{profile_json}\n\n【相关岗位 JD】\n{jd_text}\n请给出匹配报告。"),
    ])
    return answer.content


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
    jd_docs = retriever.invoke(profile_json)
    jd_text = "\n---\n".join(doc.page_content for doc in jd_docs)

    # 第三步：匹配专家对照画像与 JD 出报告
    answer = chat_model.invoke([
        match_system,
        HumanMessage(content=(
            f"【候选人画像】\n{profile_json}\n\n【相关岗位 JD】\n{jd_text}\n请给出匹配报告。"
        )),
    ])
    return f"【候选人画像】\n{profile_json}\n\n{answer.content}"



if __name__ == "__main__":
    print(retrieve_jd_info.invoke({"question": "agent开发工程师薪资是多少"}))
    sample_jd_info = "岗位：Python 工程师，地点：成都，薪资：8K-12K，要求会 Python、Django。"
    print(analyze_jd_info.invoke({"jd_text": sample_jd_info}))
