import os

from dotenv import load_dotenv
from langchain_openai import  OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("MODEL", "deepseek-chat")

embeddings = OpenAIEmbeddings(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="BAAI/bge-large-zh-v1.5",
    check_embedding_ctx_length=False)

vectorstore = Chroma(
    collection_name="job_jd",
    persist_directory=str(PROJECT_ROOT / "chroma_db"),
    embedding_function=embeddings,
  
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
@tool
def retrieve_jd_info(question: str) -> str:
    """
    当用户询问已收录岗位的薪资、工作地点、技能要求、岗位职责等信息时使用。
    输入用户的完整问题，输出检索到的 JD 相关段落。
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
    
structured_model = chat_model.with_structured_output(JobList)
@tool
def analyze_jd_info(jd_text: str) -> str: 
    """
    用户提供一份 JD 原文、要求分析岗位或生成人才画像时使用
    """
    result = structured_model.invoke([SystemMessage(content="你是一名招聘信息解析专家，请将JD信息提取为结构化数据。对于岗位要求，如果没有明确要求，请返回空字符串。"),
                                 HumanMessage(content=f"请将JD信息提取为结构化数据。\n{jd_text}")])
    return result.model_dump_json(ensure_ascii=False)



if __name__ == "__main__":
    print(retrieve_jd_info.invoke({"question": "agent开发工程师薪资是多少"}))
    sample_jd_info = "岗位：Python 工程师，地点：成都，薪资：8K-12K，要求会 Python、Django。"
    print(analyze_jd_info.invoke({"jd_text": sample_jd_info}))
