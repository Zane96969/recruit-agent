# 招聘 Agent（Recruit Agent）

> 面向招聘场景的 AI Agent：读一份 JD（职位描述）生成**结构化人才画像**，并基于 JD 知识库**检索问答（带出处、防幻觉）**。
> 项目含三套实现，呈现一条完整的"底层 → 组件 → 编排"路径：**手写版**（主循环与 RAG 全部手写、吃透底层）→ **LangChain 框架版**（组件化）→ **LangGraph 编排版**（状态机 + 持久记忆 + 人工审批）。
> 另有 **MCP 工具服务（recruit_mcp）**：把"简历分析 / 人岗匹配"做成标准 MCP Server，可被任意 MCP Agent 即插即用。

## 功能特性

- **JD → 人才画像**：抽取薪资、地点、岗位名称，归纳核心与可选技术栈；JD 没写的内容（如发展前景）标注"资料不足"，禁止编造。
- **知识库问答**：JD 切块入库、向量检索，答案末尾注明出自哪份 JD。
- **工具调用（Function Calling）**：Agent 自动判断该"查知识库"还是"现场分析一份新 JD"。
- **多轮记忆**：同一会话连续提问（如先问薪资、再追问地点），模型记得上文。
- **人工审批（Human-in-the-loop）**：工具执行前暂停，把“要调的工具 + 参数”交人确认，批准才执行、拒绝则中止。
- **MCP 工具服务**：把“简历分析、人岗匹配”按 MCP 标准做成独立 Server，Agent 一次握手发现两个工具、自主选择并提参；换 Agent 不改工具、换工具不改 Agent。
- **三版递进对照**：手写 ReAct/RAG → LangChain 组件化 → LangGraph 状态机编排 + checkpoint 持久记忆，看清每一层框架到底封装了什么。

## 效果演示

```
问：AI Agent 开发工程师的薪资是多少？
答：薪资范围是 12K-20K · 14薪（出自 AI Agent 开发工程师 JD）。

追问：地点呢？
答：工作地点是北京 / 远程。

给一份新 JD（前端 / 杭州 / 10K-15K / Vue、TypeScript）：
答：岗位画像 —— 核心技术 Vue，加分技术 TypeScript。
```

## 技术栈

| 用途 | 选型 |
| --- | --- |
| 聊天模型 | Qwen/Qwen2.5-14B-Instruct（硅基流动，OpenAI 兼容接口） |
| 向量模型 Embedding | BAAI/bge-large-zh-v1.5 |
| 精排模型 Rerank | BAAI/bge-reranker-v2-m3 |
| 向量数据库 | Chroma |
| 框架 | LangChain 1.4 + LangGraph 1.2 |
| 工具协议 | MCP（FastMCP）+ langchain-mcp-adapters |
| 语言 | Python 3.11 |

## 三版实现对照

同一套招聘能力（查岗位、分析新 JD），用三种技术深度各实现一遍，是本项目的核心设计：

| 维度 | 手写版（根目录） | LangChain 版（recruit_lc） | LangGraph 版（recruit_graph） |
| --- | --- | --- | --- |
| 对话与循环 | 手写 `while` + `finish_reason` | `create_agent` 封装 | `StateGraph` + 条件边回环 |
| 工具调用 | 手写 `TOOLS_MAP` 分发 | `@tool` + `bind_tools` | `ToolNode` + 条件边 |
| 工具回执 | 手写 `role="tool"` 消息 | 框架自动 | `ToolNode` 自动 |
| 结构化画像 | `response_format` + `json.loads` | `with_structured_output` | 复用工具内结构化 |
| 知识库检索 | 手写 Chroma `query` | `Chroma.as_retriever` | 复用检索工具 |
| 记忆 | 无 | `InMemorySaver`（易失） | `SqliteSaver`（重启不丢） |
| 人工审批 | 无 | 无 | `interrupt`（批准才执行、拒绝则中止） |
| 价值 | 吃透底层原理 | 组件复用、少写样板 | 复杂流程编排、持久记忆、人工审批、多智能体 |

## 项目结构

```
recruit-agent/
├── agent_code_templates.py   # 共享模板：客户端、切块、检索、护栏、ReAct 骨架
├── recruit_ingest.py         # 手写：JD 切块并写入向量库
├── recruit_qa.py             # 手写：检索问答（带出处）
├── recruit_profile.py        # 手写：JD 人才画像
├── data/                     # 示例 JD（拟真数据）
│   ├── jd_ai_agent.txt
│   ├── jd_java.txt
│   └── jd_recruiter.txt
├── recruit_lc/               # LangChain 框架版
│   ├── step1_structured.py   # with_structured_output 画像
│   ├── step2_retriever.py    # Chroma 检索器
│   ├── tools.py              # @tool 两个工具
│   └── agent.py              # create_agent + 记忆
├── recruit_graph/            # LangGraph 编排版
│   ├── graph_tools.py        # 两个招聘工具（检索 + JD 分析）
│   └── app.py                # StateGraph ReAct 主管 + SqliteSaver 记忆 + interrupt 人工审批
├── recruit_mcp/              # MCP 工具服务（能力标准化，可被任意 MCP Agent 复用）
│   ├── recruit_server.py     # FastMCP Server：analyze_resume + match_resume_jd
│   └── agent_recruit_client.py  # Agent 经 MCP 自主选工具的演示
└── docs/
    └── 招聘Agent-架构图.html
```

## 快速开始（Windows PowerShell）

```powershell
# 1. 克隆并进入
git clone https://github.com/<your-username>/recruit-agent.git
cd recruit-agent

# 2. 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. 配置密钥：复制模板，编辑 .env 填入 API_KEY
copy .env.example .env

# 4. 安装依赖
pip install -r requirements.txt

# 5. 构建向量库（切块入库，生成 chroma_db/）
python recruit_ingest.py

# 6. 运行
python recruit_qa.py            # 手写：检索问答
python recruit_profile.py       # 手写：人才画像
python recruit_lc/agent.py      # LangChain 版：工具调用 + 记忆
python recruit_graph/app.py     # LangGraph 版：状态机 + 持久记忆 + 人工审批
python recruit_mcp/agent_recruit_client.py  # MCP：Agent 自主调用简历分析/人岗匹配
```

> 需要一个硅基流动 API Key（注册：https://cloud.siliconflow.cn ，免费额度可跑通本项目）。
> 也可换成 DeepSeek、通义、Kimi、OpenAI 等任意 OpenAI 兼容服务，只改 `.env` 三行，代码不动。

## 数据说明

`data/` 下为**拟真示例 JD**，由作者编写、不对应任何真实公司，可自由用于演示。接入真实 JD 时请先脱敏（公司名、联系人、保密薪资）。

## 路线图（Roadmap）

- [x] **LangGraph**：StateGraph 状态机、循环/分支、checkpoint 持久化（Sqlite）、多智能体（subagents-as-tools）。
- [x] **人工审批 interrupt**：工具执行前暂停、把待执行动作（工具名 + 参数）交人确认，批准执行 / 拒绝中止；可扩展到跨平台认人等场景。
- [x] **MCP**：招聘能力（简历分析、人岗匹配）做成标准 MCP Server（recruit_mcp），Agent 一次握手、自主选工具；可扩展检索类工具。
- [ ] **Web 界面 + 部署**：Streamlit/Gradio 界面、真实数据、上线可访问。
- [ ] **能力 B（对标分析）**：一批同类 JD 反推合理技术栈与薪资行情；高级版用 Deep Agents 做"招聘行业深度研究 Agent"。

## License

[MIT](LICENSE)
