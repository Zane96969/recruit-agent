"""招聘业务 MCP Server。

把两个招聘能力按 MCP 标准做成工具：
- analyze_resume：输入简历原文，返回结构化候选人画像；
- match_resume_jd：输入简历 + 岗位 JD，返回匹配等级、匹配点、差距点、建议。

任何支持 MCP 的 Agent（本项目用 LangChain + langchain-mcp-adapters）
都能一次握手发现并自主调用这两个工具。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# 本文件在 recruit_mcp/ 下，上一级即仓库根目录，统一从根目录读 .env
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

server = FastMCP("recruit_server")

chat_model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,
)


# ===================== 结构化输出 Schema =====================
class RecruitProfile(BaseModel):
    """从简历中抽取的候选人画像。"""

    profile_name: str = Field(..., description="候选人姓名，如 张三")
    tech_stack: list[str] | None = Field(default=None, description="候选人技术栈")
    years_of_experience: str | None = Field(
        default=None, description="工作经验年限，只填如 5年"
    )
    expected_salary: str | None = Field(
        default=None, description="期望薪资，如 20K-25K"
    )
    education: str | None = Field(default=None, description="学历，只填如 本科")
    highlight_projects: list[str] | None = Field(
        default=None, description="亮点项目"
    )


class MatchResult(BaseModel):
    """简历与岗位 JD 的对照结果。"""

    match_level: str = Field(..., description="匹配等级，只填：高/中/低")
    match_points: list[str] | None = Field(default=None, description="匹配点")
    gap_points: list[str] | None = Field(
        default=None, description="差距点，不满足的要求"
    )
    suggestion: str | None = Field(default=None, description="给招聘方的一句建议")


# ===================== MCP 工具 =====================
@server.tool()
def analyze_resume(resume_text: str) -> str:
    """分析一份简历，返回候选人画像（技术栈、年限、期望薪资、学历、亮点）。
    当用户发来一段简历，或要求“分析这份简历 / 看看这个候选人”时使用。
    """
    structured_model = chat_model.with_structured_output(
        RecruitProfile,
        method="function_calling",
    )
    profile = structured_model.invoke(
        [
            SystemMessage(
                content="你是简历分析助手，只根据简历内容抽取信息；简历没写的字段留空，禁止编造。"
            ),
            HumanMessage(content=resume_text),
        ]
    )
    return profile.model_dump_json(ensure_ascii=False)


@server.tool()
def match_resume_jd(resume_text: str, jd_text: str) -> str:
    """对照简历和岗位 JD，返回匹配等级、匹配点、差距点和建议。
    当用户同时提供简历和岗位，或要求“看看这个候选人配不配这个岗”时使用。
    """
    structured_model = chat_model.with_structured_output(
        MatchResult,
        method="function_calling",
    )
    result = structured_model.invoke(
        [
            SystemMessage(
                content="你是简历匹配助手，只根据简历和岗位JD分析；匹配等级只填高/中/低，没有差距就留空，禁止编造。"
            ),
            HumanMessage(
                content=f"简历内容：{resume_text}\n岗位描述：{jd_text}"
            ),
        ]
    )
    return result.model_dump_json(ensure_ascii=False)


if __name__ == "__main__":
    server.run()
