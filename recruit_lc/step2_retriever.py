import os

from dotenv import load_dotenv
from langchain_openai import  OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import Tool
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
question="AI Agent 工程师的薪资是多少？"
docs = retriever.invoke(question)
for doc in docs:
    print("相关JD信息:", doc.page_content)
    print("来源文件:", doc.metadata.get("source", "未知"))
    print("--"*40)
    