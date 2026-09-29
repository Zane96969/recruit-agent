import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel,Field   
from langchain_core.messages import HumanMessage, SystemMessage
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("MODEL", "deepseek-chat")

chat_model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL
)
jd_path = PROJECT_ROOT / "data" / "jd_ai_agent.txt"
with open(jd_path, "r", encoding="utf-8") as file:
    jd_text = file.read()

class TechStack(BaseModel):
    core_tech: list[str] = Field(description="核心技术栈")
    optional_tech: list[str] = Field(description="可选技术栈", default=None)

class JobList(BaseModel):
    name: str = Field(description="岗位名称")
    salary: str = Field( description="薪资范围")
    location: str = Field(description="工作地点")
    requirements: str = Field(description="岗位要求", default=None)
    tech_stack: TechStack = Field(description="技术栈", default=None)
    
structured_model = chat_model.with_structured_output(JobList)
result = structured_model.invoke([SystemMessage(content="你是一名招聘信息解析专家，请将JD信息提取为结构化数据。对于岗位要求，如果没有明确要求，请返回空字符串。"),
                                 HumanMessage(content=f"请将JD信息提取为结构化数据。\n{jd_text}")])
print(result.name)  # 输出岗位名称
print(result.salary)  # 输出薪资范围
print(result.location)  # 输出工作地点
print(result.requirements)  # 输出岗位要求
print(result.tech_stack.core_tech)  # 输出核心技术栈
print(result.tech_stack.optional_tech)  # 输出可选技术栈