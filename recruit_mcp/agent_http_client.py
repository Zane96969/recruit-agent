"""Agent 经 MCP（Streamable HTTP）自主调用服务的演示。

和 agent_recruit_client.py 的唯一区别是“连接方式”：
- stdio 版：给一个命令，由本地“拉起”一个服务进程，一个客户端配一个进程；
- HTTP 版：给一个 URL，服务在别处常驻，多个客户端通过网络共享同一个服务。

启动顺序：
1. 先在一个终端启动服务（它会同时提供 REST 与 MCP）：
       python recruit_mcp/http_app.py
2. 再在另一个终端运行本客户端：
       python recruit_mcp/agent_http_client.py

本客户端只跑只读场景（岗位列表 / 知识库检索 / 简历只读分析），
不写库；写库工具由部署阶段的端到端验证脚本单独验证。
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

# 本文件在 recruit_mcp/ 下，上一级即仓库根目录，统一从根目录读 .env
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

# 常驻的 MCP 服务地址（http_app.py 里 /mcp 端点）
MCP_URL = "http://127.0.0.1:8000/mcp"

# 演示用虚构简历（与 data/resumes 同为拟真数据）
SAMPLE_RESUME = (
    "李明，24岁，大连外国语大学软件工程专业本科，2027届应届。"
    "熟练 Python，熟悉 LangChain、LangGraph、RAG、Chroma，"
    "独立做过一个招聘 Agent 项目：Function Calling 工具调用、"
    "BM25 加向量混合检索、人岗匹配打分。"
    "目标岗位 AI Agent 开发实习生，期望薪资 8K-12K，电话 13800000099。"
)


async def ask(agent, title: str, question: str):
    """向 Agent 提一个问题，打印标题与最终回答。"""
    print(f"\n===== {title} =====")
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    print(result["messages"][-1].content)


async def main():
    # 1. 配置 MCP 连接：Streamable HTTP，给一个 URL 即可（不再需要 command/args）
    mcp_client = MultiServerMCPClient(
        {"recruit_server": {"url": MCP_URL, "transport": "streamable_http"}}
    )

    # 2. 经 HTTP 握手、发现全部工具
    tools = await mcp_client.get_tools()
    print(f"已连接 {MCP_URL}")
    print("Agent 发现的工具：", sorted(tool.name for tool in tools))

    # 3. 建模型、建 Agent（挂上 MCP 工具）——这部分与 stdio 版完全一致
    chat_model = ChatOpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        model=os.getenv("MODEL"),
    )
    agent = create_agent(
        chat_model,
        tools,
        system_prompt=(
            "你是招聘助手，始终用简体中文简洁回答；需要事实信息时优先调用工具，"
            "不要自己编造岗位或候选人信息。"
        ),
    )

    # 场景1：岗位列表 → list_positions
    await ask(
        agent,
        "场景1：岗位列表（list_positions）",
        "现在岗位库里一共有哪些岗位？列出编号和岗位名。",
    )

    # 场景2：知识库混合检索 → search_jd_knowledge
    await ask(
        agent,
        "场景2：知识库混合检索（search_jd_knowledge）",
        "我想找要求会 RAG、LangChain 或 LangGraph 的岗位，帮我检索一下。",
    )

    # 场景3：简历画像只读分析 → analyze_resume（不落库）
    await ask(
        agent,
        "场景3：简历画像只读分析（analyze_resume）",
        f"先只分析这份简历、抽出画像，不要写入数据库：\n{SAMPLE_RESUME}",
    )


if __name__ == "__main__":
    asyncio.run(main())
