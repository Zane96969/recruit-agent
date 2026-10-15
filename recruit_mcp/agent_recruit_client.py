"""Agent 经 MCP 自主调用 recruit_server（真实后端版）的演示。

运行后 Agent 会：
1. 用 stdio 拉起 recruit_server，一次握手发现全部招聘工具；
2. 就几个真实问题自主选工具、提参、调用并作答：
   - “岗位库里有哪些岗位”          → list_positions
   - “检索要 RAG/LangChain 的岗位”  → search_jd_knowledge
   - “只分析这份简历、先别入库”      → analyze_resume（只读）

写库工具（screen_candidate / rank_candidates / delete_candidate）默认不触发，
避免每次演示都往候选人库写假人；把下方 WRITE_DEMO 改为 True 可追加演示
“完整初筛落库 + 查看排名”。
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
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from recruit_mcp.observability import ObservabilityHandler  # noqa: E402

# 同目录下的 Server 文件（绝对路径，跨平台）
SERVER_FILE = Path(__file__).resolve().parent / "recruit_server.py"

# 每轮 Agent 运行的可观测汇总（JSONL：一行一条记录），已在 .gitignore 排除
LOG_FILE = REPO_ROOT / "logs" / "agent_runs.jsonl"

# 写库演示开关：默认 False（只跑只读场景）；改成 True 会额外演示
# screen_candidate（完整初筛并写入候选人库）+ rank_candidates（看排名）。
WRITE_DEMO = False

# 演示用虚构简历（与 data/resumes 同为拟真数据）
SAMPLE_RESUME = (
    "李明，24岁，大连外国语大学软件工程专业本科，2027届应届。"
    "熟练 Python，熟悉 LangChain、LangGraph、RAG、Chroma，"
    "独立做过一个招聘 Agent 项目：Function Calling 工具调用、"
    "BM25 加向量混合检索、人岗匹配打分。"
    "目标岗位 AI Agent 开发实习生，期望薪资 8K-12K，电话 13800000099。"
)


async def ask(agent, title: str, question: str):
    """向 Agent 提一个问题，打印标题与最终回答，并记录可观测日志。"""
    print(f"\n===== {title} =====")
    handler = ObservabilityHandler(run_title=title, log_path=LOG_FILE)
    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": 15, "callbacks": [handler]},
        )
        print(result["messages"][-1].content)
    except Exception as e:
        print(f"[Agent 执行异常] {type(e).__name__}: {e}")
    finally:
        handler.print_summary()


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
    print("Agent 发现的工具：", sorted(tool.name for tool in tools))

    # 3. 建模型、建 Agent（挂上 MCP 工具）
    chat_model = ChatOpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        model=os.getenv("MODEL"),
        timeout=60,
        max_retries=3,
    )
    agent = create_agent(
        chat_model,
        tools,
        system_prompt=(
            "你是招聘助手，始终用简体中文简洁回答；需要事实信息时优先调用工具，"
            "不要自己编造岗位或候选人信息。"
            "如果工具返回的结果里带 error 字段，说明这一步失败了："
            "不要再调用依赖它的后续工具，也不要编造结果，"
            "直接如实告诉用户失败原因，并建议用户稍后重试或补充信息。"
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

    # 可选：写库演示（默认关闭）
    if WRITE_DEMO:
        await ask(
            agent,
            "场景4：完整初筛并落库（screen_candidate）",
            f"把这份候选人收进系统做初筛并匹配岗位：\n{SAMPLE_RESUME}",
        )
        await ask(
            agent,
            "场景5：候选人排名（rank_candidates）",
            "给我看一下现在库里候选人的匹配排名。",
        )


if __name__ == "__main__":
    asyncio.run(main())
