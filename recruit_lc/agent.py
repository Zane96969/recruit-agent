import os
from dotenv import load_dotenv
from langchain_openai import  OpenAIEmbeddings
from tools import retrieve_jd_info , analyze_jd_info
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("MODEL", "deepseek-chat")
memory=InMemorySaver()
chat_model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL
)
agent=create_agent(
    model=chat_model,
    tools=[retrieve_jd_info, analyze_jd_info],
    system_prompt=(
        "你是招聘助手，提供两类能力："
        "1）用户询问岗位的薪资、地点、技能、职责、要求等信息时，必须先调用 retrieve_jd_info 工具检索，再依据检索结果回答，不要不查就直接说查不到；"
        "2）用户给出一份新 JD、要求分析或生成画像时，调用 analyze_jd_info。"
        "查不到就明说，不要编造。"
    ),
    checkpointer=memory
)
config = {"configurable":{"thread_id": "zhang-001"}}
result1 = agent.invoke({"messages": [HumanMessage(content="agent开发工程师的薪资是多少？")]},
                        config=config,)

print(result1["messages"][-1].content)

result2 = agent.invoke({"messages": [HumanMessage(content="地点呢")]},
                        config=config,)
print(result2["messages"][-1].content)
new_jd = "岗位：前端工程师，地点：杭州，薪资：10K-15K，要求熟练 Vue、TypeScript。"
result3 = agent.invoke({"messages": [HumanMessage(content=f"请分析这份 JD 生成岗位画像：\n{new_jd}")]},
                       config= config,)


print(result3["messages"][-1].content)
