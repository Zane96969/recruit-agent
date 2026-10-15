# 招聘 Agent（Recruit Agent）

> 面向招聘场景的 AI Agent：读一份 JD（职位描述）生成**结构化人才画像**，并基于 JD 知识库**检索问答（带出处、防幻觉）**。
> 项目含三套实现，呈现一条完整的"底层 → 组件 → 编排"路径：**手写版**（主循环与 RAG 全部手写、吃透底层）→ **LangChain 框架版**（组件化）→ **LangGraph 编排版**（状态机 + 持久记忆 + 人工审批）。
> 另有 **MCP 工具服务（recruit_mcp）**：把岗位库、混合检索、简历分析、初筛落库、候选人排名等**真实后端能力**做成标准 MCP Server（8 个工具），任意 MCP 客户端一次握手即可调用同一套 SQLite + Chroma + RAG 后端，结果与本地 Agent、网页端完全一致。
> 新增 **LangGraph 多智能体（resume_multi_agent）**：Supervisor 编排"简历分析专家 → 人岗匹配专家"串行接力，只给一份简历就自动从岗位库检索 JD、给出匹配度报告。
> 再增 **多智能体并行（resume_parallel_agents / resume_batch_screen）**：技术栈 / 薪资 / 稳定性三个**互不依赖**的专家静态并行（fan-out/fan-in）同时评估一份简历；并用 **Send 动态扇出**对 N 份简历批量初筛、自动复制 N 个专家。串行 / 静态并行 / 动态扇出按任务依赖关系选型。
> 收官 **统一调度主管（resume_unified_supervisor）**：一个大模型 Supervisor 读人话语义做三选一，把任务派给三个已编译的 **subgraph（子图）**——串行画像自动找岗 / 三专家静态并行 / Send 批量初筛；模型只负责"选路"，车间内部的确定性流程由图结构在编译期锁死。

## 功能特性

- **JD → 人才画像**：抽取薪资、地点、岗位名称，归纳核心与可选技术栈；JD 没写的内容（如发展前景）标注"资料不足"，禁止编造。
- **混合检索 RAG**：JD 切块同时写 SQLite 与 Chroma；检索走 **BM25 关键词（jieba 分词）+ 向量召回 → RRF 融合（k=60）→ bge-reranker 精排**，比纯向量更能同时命中关键词与语义，答案末尾注明出自哪份 JD。
- **结构化数据层**：SQLite 存岗位（jds）、候选人（candidates）、匹配（matchings），Chroma 存向量，岗位增删改走"双写"保证两边一致；见 `recruit_graph/jd_store.py`、`candidate_store.py`、`jd_vector_store.py`。
- **简历初筛闭环**：一份简历进系统，自动完成"LLM 画像抽取 → 读取岗位库全部 JD 横向匹配打分 → 落库 → 按匹配分排名"，重复简历按内容指纹/手机号去重；`seed_candidates.py` 可批量灌入，`web_ranking.py` 可视化排名。
- **工具调用（Function Calling）**：Agent 自动判断该"查知识库"还是"现场分析一份新 JD"。
- **多轮记忆**：同一会话连续提问（如先问薪资、再追问地点），模型记得上文。
- **人工审批（Human-in-the-loop）**：工具执行前暂停，把“要调的工具 + 参数”交人确认，批准才执行、拒绝则中止。
- **MCP 工具服务（真实后端）**：8 个标准 MCP 工具直接复用项目真实的存储层与检索层——岗位侧 `list_positions / get_position / search_jd_knowledge`，简历侧 `analyze_resume / match_resume_with_jd`（只读、不写库）、`screen_candidate / rank_candidates / delete_candidate`（写库）；MCP 端与 Gradio、种子脚本共用同一套函数，结果一致。Agent 一次握手发现全部工具、自主选择并提参；换 Agent 不改工具、换工具不改 Agent。
- **多智能体串行双专家**：Supervisor（总指挥）按固定流程依次派单——简历分析专家先产出结构化画像，人岗匹配专家再经 **InjectedState** 自动读取画像（大段数据走共享状态，模型只决定“调谁”、不当搬运工），从向量库检索 Top-K JD 并给出匹配度百分比、匹配点与差距建议；专家结果回流主管后再派下一单，串行顺序由状态机强制保证。与 MCP 版的区别：MCP 版匹配需手动同时提供简历与 JD 原文，本版**只给一份简历即可自动找岗**。网页对话层另把两步封成一个复合工具，避免把流程正确性赌在模型自律上。
- **多智能体并行编排**：① **静态并行**（`resume_parallel_agents.py`）——一份简历 + 目标 JD，技术栈 / 薪资 / 到岗稳定性三个互不依赖的专家**同时开工**（fan-out），三块结论经 reducer（`Annotated[list, operator.add]`）累加、汇总节点自动等齐（fan-in），总耗时≈最慢的一个专家（实测约 1.4 秒，而非三者相加）；② **Send 动态扇出**（`resume_batch_screen.py`）——N 份简历批量初筛，`dispatch` 返回 `Send` 列表、运行时按简历份数自动复制 N 个专家，改名单不用改图。三者选型：**有依赖走串行、无依赖的多维评估走静态并行、数量不定的批量任务走 Send**。
- **统一调度主管（Supervisor + subgraph）**：顶层大模型主管用 `with_structured_output` + `Literal` 把人话语义收敛成三个合法车间之一（`serial_all / parallel_all / batch_all`），经条件边派发到对应**已编译子图**；语义判断交模型、确定性流程交图结构（模型只能选路、不能造路），三个真实场景实测全部正确路由（`resume_unified_supervisor.py`）。
- **网页界面（Gradio）**：一条命令在浏览器里聊天，工具路由过程清晰可观测。
- **思考过程可视化**：网页端把“调用工具 → 工具返回 → 最终答案”渲染成可折叠块、按执行顺序实时直播，每一步可点开追溯。
- **网页跨轮记忆**：checkpointer + thread_id 让网页版记得上文（先问薪资、再追问“它在哪工作”能正确指代），且不重复调用已有结果的工具。
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
| 聊天模型 | DeepSeek（deepseek-chat，OpenAI 兼容接口；可换 Qwen/通义/Kimi） |
| 向量模型 Embedding | BAAI/bge-large-zh-v1.5 |
| 精排模型 Rerank | BAAI/bge-reranker-v2-m3 |
| 关键词检索 | jieba 分词 + rank_bm25（BM25Okapi） |
| 向量数据库 | Chroma |
| 结构化存储 | SQLite（岗位 jds / 候选人 candidates / 匹配 matchings） |
| 框架 | LangChain 1.4 + LangGraph 1.2 |
| 工具协议 | MCP（FastMCP）+ langchain-mcp-adapters |
| Web 界面 | Gradio 6.x |
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
| 价值 | 吃透底层原理 | 组件复用、少写样板 | 复杂流程编排、持久记忆、人工审批、多智能体（串行 + 并行） |

## 多智能体编排选型

| 编排方式 | 适用场景 | 本项目脚本 |
| --- | --- | --- |
| **串行（Supervisor）** | 步骤间有先后依赖，后一步要用前一步结果（分析完才能匹配） | `resume_multi_agent.py` |
| **静态并行（fan-out/fan-in）** | 几个**不同维度**的不同专家、数量写代码时固定 | `resume_parallel_agents.py`（技术/薪资/稳定性三专家） |
| **Send 动态扇出** | **同一种**处理套到**数量不定**的 N 份数据上，运行时复制 N 个专家 | `resume_batch_screen.py`（N 份简历批量初筛） |
| **统一主管路由（Supervisor + subgraph）** | 三种编排都要支持，由大模型按人话语义自动选一个车间，车间内流程仍由图锁死 | `resume_unified_supervisor.py`（三子图 + 意图三选一） |

> 工程要点：多个并行节点同时写 state 同一字段会冲突，必须给该字段配 reducer（`Annotated[list, operator.add]`）让结果累加而非覆盖；汇总节点用独立的成品字段写一次，不要把带 reducer 的零件字段整体回写，否则会重复累加。

## 项目结构

```
recruit-agent/
├── agent_code_templates.py   # 共享模板：客户端、切块、检索、护栏、ReAct 骨架
├── recruit_ingest.py         # 手写：JD 切块并写入向量库
├── recruit_qa.py             # 手写：检索问答（带出处）
├── recruit_profile.py        # 手写：JD 人才画像
├── seed_jds.py               # 岗位种子灌库：清空后全量 JD 双写（SQLite + Chroma，幂等）
├── seed_candidates.py        # 简历种子灌库：批量画像→匹配→落库→排名
├── web_ranking.py            # Gradio 候选人匹配排名页
├── data/                     # 拟真数据（虚构，不对应真实公司）
│   ├── jd_ai_agent.txt
│   ├── jd_java.txt
│   ├── jd_recruiter.txt
│   └── resumes/              # 8 份虚构简历样本（resume_01 ~ resume_08）
├── recruit_lc/               # LangChain 框架版
│   ├── step1_structured.py   # with_structured_output 画像
│   ├── step2_retriever.py    # Chroma 检索器
│   ├── tools.py              # @tool 两个工具
│   └── agent.py              # create_agent + 记忆
├── recruit_graph/            # LangGraph 编排版 + 真实数据层
│   ├── graph_tools.py        # 招聘工具：JD 画像 + 简历分析/人岗匹配（含 InjectedState、全量 JD 评分）
│   ├── jd_store.py           # SQLite 岗位层：jds/matchings 增删查 + 触发向量双写
│   ├── candidate_store.py    # SQLite 候选人层：save_candidate 初筛落库 / list_candidates 排名
│   ├── jd_vector_store.py    # Chroma 向量层：切块、embedding、岗位向量增删
│   ├── hybrid_retriever.py   # 混合检索：BM25 + 向量 + RRF + rerank（总入口 hybrid_search）
│   ├── resume_multi_agent.py # 多智能体：Supervisor 串行双专家（简历分析 → RAG 人岗匹配）
│   ├── resume_parallel_agents.py # 多智能体：静态并行三专家（技术/薪资/稳定性 fan-out/fan-in）
│   ├── resume_batch_screen.py   # 多智能体：Send 动态扇出，N 份简历批量初筛
│   ├── resume_unified_supervisor.py # 毕业作品：大模型主管 + 三子图统一调度（串行/静态并行/Send 意图三选一）
│   ├── web_agent.py          # Gradio 网页版：create_agent + 双工具、工具调用可观测
│   ├── web_stream_agent.py   # 网页完整版：思考过程直播 + 跨轮记忆 + 简历一键匹配
│   └── app.py                # StateGraph ReAct 主管 + SqliteSaver 记忆 + interrupt 人工审批
├── recruit_mcp/              # MCP 工具服务（真实后端能力，可被任意 MCP Agent 复用）
│   ├── recruit_server.py     # FastMCP Server：8 个工具复用 SQLite/Chroma/混合检索
│   └── agent_recruit_client.py  # Agent 经 MCP 自主选工具的演示（写库场景默认关闭）
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

# 5. 构建向量库（手写版切块入库，生成 chroma_db/）
python recruit_ingest.py

# 5b.（可选）灌入数据层样例：全量岗位双写 + 8 份简历初筛落库
python seed_jds.py              # 全量 JD 双写 SQLite + Chroma（幂等，可重灌）
python seed_candidates.py       # 简历：画像→匹配→落库→排名（会调模型，约 1-2 分钟）

# 6. 运行
python recruit_qa.py            # 手写：检索问答
python recruit_profile.py       # 手写：人才画像
python recruit_lc/agent.py      # LangChain 版：工具调用 + 记忆
python recruit_graph/app.py     # LangGraph 版：状态机 + 持久记忆 + 人工审批
python recruit_graph/resume_multi_agent.py      # 多智能体：简历分析→人岗匹配串行双专家
python recruit_graph/resume_parallel_agents.py  # 多智能体：三专家静态并行评估（fan-out/fan-in）
python recruit_graph/resume_batch_screen.py     # 多智能体：Send 动态扇出，N 份简历批量初筛
python recruit_graph/resume_unified_supervisor.py  # 毕业作品：主管意图路由 + 三子图统一调度（三场景实测）
python recruit_graph/web_agent.py  # Gradio 网页版：浏览器打开 http://127.0.0.1:7860
python recruit_graph/web_stream_agent.py  # 网页完整版：思考过程直播 + 跨轮记忆
python recruit_mcp/agent_recruit_client.py  # MCP：Agent 自主调用 8 个真实后端工具（stdio）
```

> 需要一份 OpenAI 兼容的 `.env`：聊天模型默认 DeepSeek（`MODEL=deepseek-chat`、`BASE_URL=https://api.deepseek.com`），检索还要用到 BAAI/bge-large-zh 向量模型与 bge-reranker 精排模型。
> 若所选平台（如 DeepSeek 官方）不提供 bge 向量/精排，可把 `BASE_URL` 指向同时提供这些模型的兼容服务（如硅基流动 SiliconFlow）并相应调整 `MODEL`；聊天、向量、精排三处都读同一份 `.env`，代码不用改。

## 数据说明

`data/` 下为**拟真示例 JD**，由作者编写、不对应任何真实公司，可自由用于演示。接入真实 JD 时请先脱敏（公司名、联系人、保密薪资）。

## 路线图（Roadmap）

- [x] **LangGraph**：StateGraph 状态机、循环/分支、checkpoint 持久化（Sqlite）；多智能体 Supervisor 串行编排——简历分析 → 人岗匹配双专家，InjectedState 自动传递画像、RAG 自动检索岗位（`recruit_graph/resume_multi_agent.py`）。
- [x] **多智能体并行编排**：静态 fan-out/fan-in 三专家（`resume_parallel_agents.py`，reducer 累加、自动屏障等齐）+ Send 动态扇出 N 份简历批量初筛（`resume_batch_screen.py`，运行时按份数复制专家）。
- [x] **人工审批 interrupt**：工具执行前暂停、把待执行动作（工具名 + 参数）交人确认，批准执行 / 拒绝中止；可扩展到跨平台认人等场景。
- [x] **混合检索 RAG**：BM25 关键词（jieba）+ 向量召回 → RRF 融合（k=60）→ bge-reranker 精排，替换原有纯向量检索（`recruit_graph/hybrid_retriever.py`）。
- [x] **结构化数据层**：SQLite（岗位/候选人/匹配）+ Chroma 双写，岗位增删改两边一致；`seed_jds.py` 幂等清空重灌。
- [x] **真实简历初筛闭环**：8 份拟真简历批量"画像 → 全量 JD 横向匹配打分 → 落库 → 排名"，修复评分区分度与岗位名脏值（`seed_candidates.py`）。
- [x] **MCP（真实后端）**：8 个标准 MCP 工具直接复用 SQLite + Chroma + 混合检索——岗位查询/检索、简历画像/对照、初筛落库/排名/删除；stdio 即插即用，任意 MCP 客户端一次握手调用同一套后端（`recruit_mcp`）。
- [ ] **MCP 服务化部署**：将 stdio 升级为 Streamable HTTP（多客户端共享），与 FastAPI、服务器部署一起做。
- [x] **Web 界面（Gradio）**：本地网页聊天、后端打印消息类型与工具名、工具路由可观测（`recruit_graph/web_agent.py`）。
- [x] **思考过程可视化 + 网页记忆**：工具调用折叠块按顺序直播、每步可追溯；checkpointer + thread_id 支撑跨轮记忆（`recruit_graph/web_stream_agent.py`）。
- [x] **统一主管（Supervisor）路由三种编排**：顶层大模型主管按人话语义三选一，派发到串行 / 静态并行 / Send 三个**已编译 subgraph**（`resume_unified_supervisor.py`）；`Literal` 锁死合法路由、条件边派单，模型只选路不造路，三个真实场景实测全部正确。
- [ ] **部署上线**：真实数据、服务器部署、公网可访问。
- [ ] **能力 B（对标分析）**：一批同类 JD 反推合理技术栈与薪资行情；高级版用 Deep Agents 做"招聘行业深度研究 Agent"。

## License

[MIT](LICENSE)
