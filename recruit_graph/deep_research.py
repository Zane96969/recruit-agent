"""深度研究 Agent（Deep Agents）：主管规划 + 多研究员并行检索 + 综合报告。

和「知识库问答」的区别（这是本模块的价值）：
- 知识库问答：单个模型一次检索、一次回答，快但浅；
- 深度研究：主管先把大问题拆成子任务（write_todos），再把每个子任务
  派给一个研究员（task），研究员各自检索本地 JD 知识库、把结论写进
  共享文件，主管读回文件后综合成一份带引用的报告——慢但深、可并行。

消息是隔离的（研究员之间看不到彼此的思考），但共享同一个虚拟文件系统，
所以研究员之间靠「写文件」、主管靠「读文件」传递结果。
"""

from langchain_core.tools import tool
from langchain.agents.middleware import TodoListMiddleware
from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgent

from recruit_graph.graph_tools import chat_model
from recruit_graph.hybrid_retriever import hybrid_search


# 第一组：检索工具——研究员唯一的资料来源，和知识库问答用同一个混合检索
@tool
def search_jd_knowledge(question: str) -> str:
    """在本地岗位 JD 知识库里检索与问题最相关的真实资料片段，每段带来源块 id。"""
    docs = hybrid_search(question, use_rerank=True)
    lines = []
    for i, doc in enumerate(docs, start=1):
        source_id = doc.metadata.get("id", "?")
        lines.append(f"[资料{i} 来源块:{source_id}] {doc.page_content}")
    return "\n".join(lines)


# 第二组：研究员的工作纪律；把输出文件路径写死在每个研究员自己的角色设定里
# （比主管在 task 里口头指定更可靠，14B 这类小模型更不容易乱改文件名）。
def _researcher_prompt(role: str, file_path: str) -> str:
    """生成某个研究员的系统提示，file_path 是它固定的输出文件路径。"""
    return (
        f"你是{role}。必须先调用 search_jd_knowledge 检索本地真实 JD 再下结论，"
        "每条结论后标注来源块 id（来源:块id）；资料里没有就明说，禁止编造。\n"
        "研究完成后必须做两件事：\n"
        f"1）调用 write_file，把结论写到固定路径 {file_path}，"
        "文件名一字不差，禁止改名、禁止换目录；\n"
        "2）在给主管的文字回报里直接贴出你写下的结论全文，"
        "方便主管在读取文件失败时也能直接采用。"
    )


skill_researcher = SubAgent(
    name="skill_researcher",
    description="技能研究员：研究一个岗位/主题要掌握哪些技术、框架、工具、能力时派给它。",
    tools=[search_jd_knowledge],
    system_prompt=_researcher_prompt("技能研究员", "/workdir/skill.md"),
)

requirement_researcher = SubAgent(
    name="requirement_researcher",
    description="任职要求研究员：研究学历、经验、年限、软素质等任职门槛时派给它。",
    tools=[search_jd_knowledge],
    system_prompt=_researcher_prompt("任职要求研究员", "/workdir/requirement.md"),
)


# 第三组：构建研究主管——只负责规划、派活、综合，自己不直接查资料
def _build_agent():
    """每次研究新建一个主管 agent，避免不同任务互相串状态。"""
    return create_deep_agent(
        model=chat_model,
        tools=[],
        subagents=[skill_researcher, requirement_researcher],
        middleware=[TodoListMiddleware()],
        system_prompt=(
            "你是研究主管，负责产出一份岗位/主题研究报告。\n"
            "最高纪律：每一步都必须通过工具调用完成，禁止只用文字描述打算做什么。\n"
            "第一条回复必须立刻调用 write_todos 列出子任务，不要用文字代替。\n"
            "工作流程：\n"
            "1. 调 write_todos，列出「核心技能」「任职要求」两个子任务；\n"
            "2. 并行调用两次 task：技能类派给 skill_researcher、"
            "任职门槛类派给 requirement_researcher，在 description 里要求它们"
            "分别用 write_file 写到 /workdir/skill.md 与 /workdir/requirement.md；\n"
            "3. 两个研究员都回报后，调用 read_file 读回这两个文件；"
            "若某个文件不存在，直接采用该研究员 task 回报里贴出的结论全文，不要重复派活；"
            "然后综合成中文报告，报告分「核心技能」「任职要求」两节，"
            "保留每条结论的来源块 id；\n"
            "4. 每完成一步立刻调 write_todos 标记 completed；\n"
            "只有报告真正综合完成后，才输出最终文字报告，不要提前结束。"
        ),
    )


# 第四组：把消息流整理成「过程轨迹」，让人能看清多智能体每一步在干什么
def _build_trace(messages) -> str:
    """遍历主管这一层的消息，提取规划、派活、工具回报，拼成可读日志。"""
    steps = []
    for index, msg in enumerate(messages):
        kind = msg.__class__.__name__
        if kind == "AIMessage":
            for call in getattr(msg, "tool_calls", None) or []:
                name = call.get("name", "")
                args = call.get("args", {}) or {}
                if name == "write_todos":
                    todos = args.get("todos", [])
                    names = "、".join(t.get("content", "") for t in todos)
                    steps.append(f"[{index}] 🧠 主管规划：{names}")
                elif name == "task":
                    target = (
                        args.get("subagent_type")
                        or args.get("name")
                        or args.get("subagent")
                        or "研究员"
                    )
                    desc = str(args.get("description", ""))[:110]
                    steps.append(f"[{index}] 📤 派活→{target}：{desc}")
                else:
                    steps.append(f"[{index}] 🔧 调用工具：{name}")
        elif kind == "ToolMessage":
            text = str(getattr(msg, "content", "")).replace("\n", " ")
            steps.append(f"[{index}] 📥 回报：{text[:150]}")
    return "\n".join(steps)


# 第五组：对外唯一入口——输入研究问题，返回报告、过程轨迹、共享文件清单
def run_deep_research(question: str) -> dict:
    """跑一次深度研究。

    返回 dict：
    - report：主管综合后的最终报告（带引用）；
    - trace：规划/派活/回报的过程日志（可观测雏形）；
    - files：研究员写入共享文件系统的文件列表。
    """
    question = (question or "").strip()
    if not question:
        return {"report": "请先输入要研究的岗位或主题。", "trace": "", "files": []}

    agent = _build_agent()
    state = agent.invoke({"messages": [{"role": "user", "content": question}]})

    trace = _build_trace(state["messages"])
    files = list(state.get("files", {}).keys())
    report = str(state["messages"][-1].content).strip()
    return {"report": report, "trace": trace, "files": files}


if __name__ == "__main__":
    result = run_deep_research("AI Agent 开发实习")
    print("=" * 50)
    print("过程轨迹：")
    print(result["trace"])
    print("=" * 50)
    print("共享文件：", result["files"])
    print("=" * 50)
    print("最终报告：")
    print(result["report"])
