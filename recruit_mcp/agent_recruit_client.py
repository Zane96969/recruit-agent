"""Agent 经 MCP 自主调用 recruit_server 的演示。

运行后 Agent 会：
1. 启动并连接 recruit_server（stdio），自动发现两个工具；
2. 根据问题自主选择工具、提取参数、调用并给出结论：
   - “分析这份简历”   → analyze_resume
   - “看看配不配这个岗” → match_resume_jd
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

# 本文件在 recruit_mcp/ 下，上一级即仓库根目录，统一从根目录读 .env
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

# 同目录下的 Server 文件（绝对路径，跨平台）
SERVER_FILE = Path(__file__).resolve().parent / "recruit_server.py"


async def main():
    # 1. 配置 MCP 连接：用“当前 Python 解释器”启动 Server（stdio）
    mcp_client = MultiServerMCPClient(
        {
            "recruit_server": {
                "command": sys.executable,
                "args": [str(SERVER_FILE)],
                "transport": "stdio",
            }
        }
    )

    # 2. 自动启动 Server、握手、拿到工具列表
    tools = await mcp_client.get_tools()
    print("Agent 发现的工具：", [tool.name for tool in tools])

    # 3. 建模型、建 Agent（挂上 MCP 工具）
    chat_model = ChatOpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        model=os.getenv("MODEL"),
    )
    agent = create_agent(
        chat_model,
        tools,
        system_prompt="你是招聘助手，始终用简体中文简洁回答。",
    )

    # 内置示例数据（练习用假数据）
    resume_text = (
        "张三，28岁，本科软件工程，工作5年，熟练 Python、FastAPI、MySQL、Redis，"
        "了解 Docker、Kubernetes，期望20K-25K，大连或远程。"
    )
    jd_text = (
        "Python 后端工程师，大连，18K-25K，本科，3年以上 Python，"
        "熟练 FastAPI、MySQL、Redis，熟悉 Docker，Kubernetes 优先，"
        "有大模型/RAG 经验优先。"
    )

    # 场景1：只分析简历 → 应选 analyze_resume
    result1 = await agent.ainvoke(
        {
            "messages": [
                {"role": "user", "content": f"帮我分析这份简历：{resume_text}"}
            ]
        }
    )
    print("\n=== 场景1（分析简历）===")
    print(result1["messages"][-1].content)

    # 场景2：人岗对照 → 应选 match_resume_jd
    result2 = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"看看这个候选人配不配这个岗。\n简历：{resume_text}\nJD：{jd_text}",
                }
            ]
        }
    )
    print("\n=== 场景2（人岗对照）===")
    print(result2["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
