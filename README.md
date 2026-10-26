# 招聘 Agent（Recruit Agent）

> 面向招聘场景的 AI Agent：读一份 JD（职位描述）生成**结构化人才画像**，并基于 JD 知识库**检索问答（带出处、防幻觉）**。
> 项目含三套实现，呈现一条完整的"底层 → 组件 → 编排"路径：**手写版**（主循环与 RAG 全部手写、吃透底层）→ **LangChain 框架版**（组件化）→ **LangGraph 编排版**（状态机 + 持久记忆 + 人工审批）。
> 另有 **MCP 工具服务（recruit_mcp）**：把岗位库、混合检索、简历分析、初筛落库、候选人排名等**真实后端能力**做成标准 MCP Server（8 个工具），任意 MCP 客户端一次握手即可调用同一套 SQLite + Chroma + RAG 后端，结果与本地 Agent、网页端完全一致。
> 新增 **LangGraph 多智能体（resume_multi_agent）**：Supervisor 编排"简历分析专家 → 人岗匹配专家"串行接力，只给一份简历就自动从岗位库检索 JD、给出匹配度报告。
> 再增 **多智能体并行（resume_parallel_agents / resume_batch_screen）**：技术栈 / 薪资 / 稳定性三个**互不依赖**的专家静态并行（fan-out/fan-in）同时评估一份简历；并用 **Send 动态扇出**对 N 份简历批量初筛、自动复制 N 个专家。串行 / 静态并行 / 动态扇出按任务依赖关系选型。
> 收官 **统一调度主管（resume_unified_supervisor）**：一个大模型 Supervisor 读人话语义做三选一，把任务派给三个已编译的 **subgraph（子图）**——串行画像自动找岗 / 三专家静态并行 / Send 批量初筛；模型只负责"选路"，车间内部的确定性流程由图结构在编译期锁死。
> 最后以 **Gradio 四模块统一工作台（web_workbench.py）** 收口：把“岗位管理（上传/新增/改/删）、简历逐份初筛累积排名、知识库问答、JD 智能生成、深度研究”整合成一个网页，支持上传 PDF / Word / 图片（OCR）真实文件。
> 深度研究用 **Deep Agents 多智能体（recruit_graph/deep_research.py）** 再上一层：一个**研究主管**先用 write_todos 把问题拆成子任务、并行 task 派给**技能研究员 / 任职要求研究员**两个子智能体（SubAgent，从属智能体）；它们各自检索本地 JD、把结论写进共享虚拟文件，主管再 read_file 读回、综合成一份每条带来源块 id 的报告。与“知识库问答”（单 Agent 单轮、快但浅）形成高低搭配：深度研究慢但深、可并行、还能看到完整规划-派活过程。

## 功能特性

- **JD → 人才画像**：抽取薪资、地点、岗位名称，归纳核心与可选技术栈；JD 没写的内容（如发展前景）标注"资料不足"，禁止编造。
- **混合检索 RAG**：JD 切块同时写 SQLite 与 Chroma；检索走 **BM25 关键词（jieba 分词）+ 向量召回 → RRF 融合（k=60）→ bge-reranker 精排**，比纯向量更能同时命中关键词与语义，答案末尾注明出自哪份 JD。
- **结构化数据层**：SQLite 存岗位（jds）、候选人（candidates）、匹配（matchings），Chroma 存向量，岗位增删改走"双写"保证两边一致；见 `recruit_graph/jd_store.py`、`candidate_store.py`、`jd_vector_store.py`。
- **简历初筛闭环**：一份简历进系统，自动完成"LLM 画像抽取 → 读取岗位库全部 JD 横向匹配打分 → 落库 → 按匹配分排名"，重复简历按内容指纹/手机号去重；`seed_candidates.py` 可批量灌入，`web_ranking.py` 可视化排名。
- **工具调用（Function Calling）**：Agent 自动判断该"查知识库"还是"现场分析一份新 JD"。
- **多轮记忆**：同一会话连续提问（如先问薪资、再追问地点），模型记得上文。
- **人工审批（Human-in-the-loop）**：工具执行前暂停，把“要调的工具 + 参数”交人确认，批准才执行、拒绝则中止。
- **MCP 工具服务（真实后端）**：8 个标准 MCP 工具直接复用项目真实的存储层与检索层——岗位侧 `list_positions / get_position / search_jd_knowledge`，简历侧 `analyze_resume / match_resume_with_jd`（只读、不写库）、`screen_candidate / rank_candidates / delete_candidate`（写库）；MCP 端与 Gradio、种子脚本共用同一套函数，结果一致。Agent 一次握手发现全部工具、自主选择并提参；换 Agent 不改工具、换工具不改 Agent。
- **服务化部署（FastAPI + MCP Streamable HTTP）**：`recruit_mcp/http_app.py` 起一个常驻进程，把同一套后端同时暴露成普通 REST（`/api/...`）与标准 MCP（`/mcp`）；传输从 stdio 的"一个客户端拉起一个本地进程"升级为"一个服务、多个客户端通过 URL 共享"，Agent 与工具代码零改动（只改连接配置），另附 Swagger 文档（`/docs`）与健康检查。
- **五层容错（健壮性）**：① LLM 客户端 `timeout=60 + max_retries=3` 兜网络抖动；② 每个 MCP 工具函数体内 try/except，返回结构化 `{error, detail}`（不用装饰器，避免破坏工具参数 schema）；③ 串联流程（简历分析→产出校验→匹配→写库）失败即短路、绝不写脏数据，关键步骤带指数退避重试；④ 编排层 system prompt 约定"工具返回 error 就停手、不编造结果"；⑤ 检索层 rerank 精排失败自动降级回 RRF 粗排，保证仍能出结果。
- **检索可切换精排/粗排（`use_rerank`）**：混合检索默认"BM25+向量 RRF 粗排 → bge-reranker 精排"拿最准结果；也可 `use_rerank=False` 只做 RRF 粗排，更快且不依赖精排服务。开关从 `hybrid_search` 一路透传到 LangGraph 工具、MCP 工具与 REST（`/api/search?use_rerank=`）。
- **可观测（Observability）**：基于 LangChain 异步回调 `AsyncCallbackHandler` 统一记录每次 **LLM 调用**（耗时、输入/输出 token）与**工具调用**（名称、入参、耗时、成败），控制台实时打印每一步、每轮结束给汇总，并追加写成 JSONL（`logs/agent_runs.jsonl`，已 gitignore）；两个 MCP 客户端零侵入挂上即可。底层机制与 Langfuse / LangSmith 相同（一个 callback handler），后续可一键替换为平台 handler。
- **多智能体串行双专家**：Supervisor（总指挥）按固定流程依次派单——简历分析专家先产出结构化画像，人岗匹配专家再经 **InjectedState** 自动读取画像（大段数据走共享状态，模型只决定“调谁”、不当搬运工），从向量库检索 Top-K JD 并给出匹配度百分比、匹配点与差距建议；专家结果回流主管后再派下一单，串行顺序由状态机强制保证。与 MCP 版的区别：MCP 版匹配需手动同时提供简历与 JD 原文，本版**只给一份简历即可自动找岗**。网页对话层另把两步封成一个复合工具，避免把流程正确性赌在模型自律上。
- **多智能体并行编排**：① **静态并行**（`resume_parallel_agents.py`）——一份简历 + 目标 JD，技术栈 / 薪资 / 到岗稳定性三个互不依赖的专家**同时开工**（fan-out），三块结论经 reducer（`Annotated[list, operator.add]`）累加、汇总节点自动等齐（fan-in），总耗时≈最慢的一个专家（实测约 1.4 秒，而非三者相加）；② **Send 动态扇出**（`resume_batch_screen.py`）——N 份简历批量初筛，`dispatch` 返回 `Send` 列表、运行时按简历份数自动复制 N 个专家，改名单不用改图。三者选型：**有依赖走串行、无依赖的多维评估走静态并行、数量不定的批量任务走 Send**。
- **统一调度主管（Supervisor + subgraph）**：顶层大模型主管用 `with_structured_output` + `Literal` 把人话语义收敛成三个合法车间之一（`serial_all / parallel_all / batch_all`），经条件边派发到对应**已编译子图**；语义判断交模型、确定性流程交图结构（模型只能选路、不能造路），三个真实场景实测全部正确路由（`resume_unified_supervisor.py`）。
- **统一工作台（Gradio 四模块，`web_workbench.py`）**：一个入口整合"岗位管理 / 简历逐份初筛 / 知识库问答与 JD 生成"——岗位可粘贴文本或上传文件新建、按 id 改名/补充/删除；简历**逐份上传、逐份解析匹配并累积到同一份候选人排名**（不互相覆盖），可反复查看与删除。
- **文档解析（PDF / Word / 图片 OCR，`recruit_graph/doc_parser.py`）**：统一把 PDF（PyMuPDF 读文字层，文字极少的扫描页自动转图片交视觉模型 OCR）、Word（python-docx 读正文段落与表格）、图片（Qwen-VL 视觉模型原样转录）、txt/md 提取成纯文本；旧版 .doc 提示先另存为 .docx。
- **JD 智能生成（few-shot RAG + 客户要求，`recruit_graph/jd_generator.py`）**：先用混合检索从知识库捞出同类真实 JD 作范例（只借结构、不抄内容），再经 Function Calling 输出结构化草稿（学历/年限/地点/薪资/招聘人数/职责/要求）；**客户明确给的要求（薪资、地点、学历、年限、招聘人数、硬性技能、汇报对象）作为最高优先级事实写入，客户没给且无法按惯例推断的才标"待确认"**，从源头避免凭空编造。
- **Deep Agents 深度研究（主管规划 + 多子智能体并行 + 共享文件，`recruit_graph/deep_research.py`）**：研究主管 `write_todos` 列子任务 → 并行 `task` 派给两个 SubAgent（技能研究员、任职要求研究员）→ 研究员各自调**项目自有的混合检索**查真实 JD、`write_file` 写进固定共享文件（`/workdir/skill.md`、`/workdir/requirement.md`）并在回报里贴全文 → 主管 `read_file` 读回，综合成**每条带来源块 id** 的报告；入口 `run_deep_research()` 一次返回「报告 / 过程轨迹 / 共享文件清单」。模型与检索完全复用 `chat_model`、`hybrid_search`，只新增编排层。针对 14B 小模型对“口头指定文件路径”指令遵从弱的问题做双保险：路径写死进每个研究员自己的 system_prompt（固定角色纪律比临时派活更可靠），并约定主管读不到文件就直接用回报结论、不重复派活。
- **岗位管理与脏数据防护**：岗位支持按 id 更新——只改名时原地改向量块 metadata、不重新向量化；改正文时删旧向量再重建，若与其他岗位撞重则提前拦截。简历识别不到姓名时统一落"未知姓名"，姓名正则只认"姓名："这类明确标签并排除"个人简历/求职意向"等板块标题，防止把标题误当人名。
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
| 多智能体深度研究 | Deep Agents 0.7（create_deep_agent + SubAgent + TodoListMiddleware：规划/派活/共享虚拟文件系统） |
| 工具协议 | MCP（FastMCP）+ langchain-mcp-adapters |
| HTTP 服务化 | FastAPI + Uvicorn（同一进程提供 REST `/api` 与 MCP Streamable HTTP `/mcp`） |
| Web 界面 | Gradio 6.x |
| 文档解析 | PyMuPDF（PDF 文字层）+ python-docx（Word）+ 视觉模型 Qwen-VL（图片/扫描件 OCR） |
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
├── run.py                    # 一键启动器：自动找/建 .venv，用对的解释器启动入口（python run.py --list）
├── start.bat                 # Windows 双击入口（等价于 python run.py）
├── pyproject.toml            # 项目元数据 + 依赖声明（pip install -e . 后包可在任何目录导入）
├── uv.lock                   # 锁定到具体版本（uv sync --frozen 可完全复现环境）
├── tests/                    # 回归测试（pytest，第一批守候选人写入的 id 不变式）
├── agent_code_templates.py   # 共享模板：客户端、切块、检索、护栏、ReAct 骨架
├── recruit_ingest.py         # 手写：JD 切块并写入向量库
├── recruit_qa.py             # 手写：检索问答（带出处）
├── recruit_profile.py        # 手写：JD 人才画像
├── seed_jds.py               # 岗位种子灌库：清空后全量 JD 双写（SQLite + Chroma，幂等）
├── seed_candidates.py        # 简历种子灌库：批量画像→匹配→落库→排名
├── web_ranking.py            # Gradio 候选人匹配排名页
├── web_workbench.py          # 统一工作台：岗位管理 + 简历逐份初筛累积 + 知识库问答/JD 生成
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
│   ├── db.py                 # SQLite 公共层：DB_PATH / 连接 / 候选人 UPSERT（重复入库 id 不变）
│   ├── candidate_store.py    # SQLite 候选人层：save_candidate 初筛落库 / list_candidates 排名
│   ├── jd_vector_store.py    # Chroma 向量层：切块、embedding、岗位向量增删改
│   ├── doc_parser.py         # 文档解析：PDF/Word 文字层 + 图片视觉 OCR + 文本 → 纯文本
│   ├── jd_generator.py       # JD 智能生成：few-shot 检索同类范例 + 客户要求优先 + 结构化输出
│   ├── hybrid_retriever.py   # 混合检索：BM25 + 向量 + RRF + rerank（总入口 hybrid_search）
│   ├── deep_research.py      # Deep Agents：研究主管 + 技能/任职要求两子智能体并行检索，写共享文件综合成带引用报告
│   ├── resume_multi_agent.py # 多智能体：Supervisor 串行双专家（简历分析 → RAG 人岗匹配）
│   ├── resume_parallel_agents.py # 多智能体：静态并行三专家（技术/薪资/稳定性 fan-out/fan-in）
│   ├── resume_batch_screen.py   # 多智能体：Send 动态扇出，N 份简历批量初筛
│   ├── resume_unified_supervisor.py # 毕业作品：大模型主管 + 三子图统一调度（串行/静态并行/Send 意图三选一）
│   ├── web_agent.py          # Gradio 网页版：create_agent + 双工具、工具调用可观测
│   ├── web_stream_agent.py   # 网页完整版：思考过程直播 + 跨轮记忆 + 简历一键匹配
│   └── app.py                # StateGraph ReAct 主管 + SqliteSaver 记忆 + interrupt 人工审批
├── recruit_mcp/              # MCP 工具服务 + HTTP 服务化（同一套真实后端，可被任意 MCP Agent 复用）
│   ├── recruit_server.py     # FastMCP Server：8 个工具复用 SQLite/Chroma/混合检索（stdio 入口）
│   ├── observability.py      # 可观测：异步回调记录每轮 LLM/工具调用、token、耗时（控制台 + JSONL）
│   ├── agent_recruit_client.py  # Agent 经 MCP 自主选工具的演示（stdio，写库场景默认关闭）
│   ├── http_app.py           # FastAPI 常驻服务：同一套后端同时暴露 REST(/api) 与 MCP(/mcp)
│   └── agent_http_client.py  # Agent 经 Streamable HTTP 连常驻服务的演示（只改连接方式）
└── docs/
    └── 招聘Agent-架构图.html
```

## 快速开始（Windows PowerShell）

> **最省事的方式**：`python run.py`（或直接双击 `start.bat`）。
> 这个启动器只依赖标准库，会用系统 Python 运行，但**自动切到 `.venv` 里的解释器**去跑真正的程序；`.venv` 不存在就自动创建并装依赖（有 `uv` 用 `uv sync --frozen` 按 `uv.lock` 装，版本完全一致；没有就退回 pip）。`python run.py --list` 看全部入口（`workbench / qa / app / deep / http` 等）。
> 机器上已经有能跑的环境，也可以借来用、不重新装：
> `python run.py --python D:\somewhere\.venv\Scripts\python.exe`
> 借用一次之后路径会被记到 `.python-path`（git 忽略），以后直接 `python run.py` 就行；想换回自动选择用 `python run.py --forget`。
> 这样就再也不用记"到底该用哪个 python"，也不会再撞上 `No module named 'gradio'`。
>
> 下面是从零开始的完整步骤（想手动控制时用）。

```powershell
# 1. 克隆并进入
git clone https://github.com/<your-username>/recruit-agent.git
cd recruit-agent

# 2. 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. 配置密钥：复制模板，编辑 .env 填入 API_KEY
copy .env.example .env

# 4. 安装（可编辑安装：装依赖 + 把项目注册成包）
#    requirements.txt 里只有一行 `-e .`，依赖统一在 pyproject.toml 里声明
pip install -e .
#    需要跑测试时：pip install -e ".[dev]"

# 5. 构建向量库（手写版切块入库，生成 chroma_db/）
python recruit_ingest.py

# 5b.（可选）灌入数据层样例：全量岗位双写 + 8 份简历初筛落库
python seed_jds.py              # 全量 JD 双写 SQLite + Chroma（幂等，可重灌）
python seed_candidates.py       # 简历：画像→匹配→落库→排名（会调模型，约 1-2 分钟）

# 6. 运行
python recruit_qa.py            # 手写：检索问答
python recruit_profile.py       # 手写：人才画像
python -m recruit_lc.agent      # LangChain 版：工具调用 + 记忆
python -m recruit_graph.app     # LangGraph 版：状态机 + 持久记忆 + 人工审批
python -m recruit_graph.resume_multi_agent      # 多智能体：简历分析→人岗匹配串行双专家
python -m recruit_graph.resume_parallel_agents  # 多智能体：三专家静态并行评估（fan-out/fan-in）
python -m recruit_graph.resume_batch_screen     # 多智能体：Send 动态扇出，N 份简历批量初筛
python -m recruit_graph.resume_unified_supervisor  # 毕业作品：主管意图路由 + 三子图统一调度（三场景实测）
python -m recruit_graph.web_agent  # Gradio 网页版：浏览器打开 http://127.0.0.1:7860
python -m recruit_graph.web_stream_agent  # 网页完整版：思考过程直播 + 跨轮记忆
python web_workbench.py          # 统一工作台：四模块网页（岗位管理/简历初筛/问答与JD生成/深度研究），打开 http://127.0.0.1:7860
python -m recruit_graph.deep_research  # 命令行直接跑一次深度研究（打印过程轨迹 / 共享文件 / 带引用报告）
python -m recruit_mcp.agent_recruit_client  # MCP：Agent 自主调用 8 个真实后端工具（stdio）
python -m recruit_mcp.http_app              # 服务化：起常驻 FastAPI，同时提供 REST(/api) 与 MCP(/mcp)
# 上一条服务启动后，另开一个终端：
python -m recruit_mcp.agent_http_client     # Agent 经 Streamable HTTP 连 http://127.0.0.1:8000/mcp
```

> **为什么子目录脚本用 `python -m 包名.模块` 而不是 `python 路径/文件.py`**：直接给文件路径时，Python 只把**脚本所在目录**加进 `sys.path`，仓库根不在里面，于是 `from recruit_graph.xxx import ...` 会报 `ModuleNotFoundError`。`-m` 方式把**当前目录**加进搜索路径，在仓库根执行必定能找到包；配合第 4 步的 `pip install -e .`，在任何目录都能跑。IDE 里能跑只是因为 IDE 偷偷加了项目根，别当成项目本身能跑。
>
> **常见报错 `ModuleNotFoundError: No module named 'gradio'`（或 langchain/langgraph）**：说明这条命令是用**系统自带的 python** 跑的，不是虚拟环境里的。先 `.\\.venv\\Scripts\\Activate.ps1`，或直接用 `.\\.venv\\Scripts\\python.exe xxx.py`，再运行。命令行前缀显示 `C:\\Python3xx\\python.exe` 就是踩了这个坑。
>
> 需要一份 OpenAI 兼容的 `.env`：聊天模型默认 DeepSeek（`MODEL=deepseek-chat`、`BASE_URL=https://api.deepseek.com`），检索还要用到 BAAI/bge-large-zh 向量模型与 bge-reranker 精排模型。
> 若所选平台（如 DeepSeek 官方）不提供 bge 向量/精排，可把 `BASE_URL` 指向同时提供这些模型的兼容服务（如硅基流动 SiliconFlow）并相应调整 `MODEL`；聊天、向量、精排三处都读同一份 `.env`，代码不用改。

## 服务化部署：REST + MCP（Streamable HTTP）

`recruit_mcp/http_app.py` 用 FastAPI 把整套招聘后端在一个常驻进程里同时开成两套接口：

- **REST 接口**（`/api/...`）：给浏览器、前端、Postman 等常规 HTTP 客户端用，返回标准 JSON；
- **MCP 接口**（`/mcp`）：2025 年版 Streamable HTTP 传输，任何 MCP 客户端（本项目 Agent / Claude Desktop / Cursor）连一个 URL 即可握手调用。

启动（仓库根目录）：

```powershell
python -m recruit_mcp.http_app
# 或：uvicorn recruit_mcp.http_app:app --host 127.0.0.1 --port 8000
```

启动后：

- 健康检查：GET http://127.0.0.1:8000/health
- 自动接口文档（Swagger）：http://127.0.0.1:8000/docs
- MCP 端点：http://127.0.0.1:8000/mcp

REST 接口一览（与 8 个 MCP 工具共用同一套后端函数）：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| GET | `/api/positions` | 岗位列表 |
| GET | `/api/positions/{jd_id}` | 单份 JD 详情（找不到返回 404） |
| GET | `/api/search?query=&top_k=&use_rerank=` | 岗位知识库混合检索（BM25+向量+RRF；`use_rerank` 默认 true 走 bge 精排，false 只粗排） |
| POST | `/api/analyze-resume` | 简历画像只读分析（不写库） |
| POST | `/api/match-resume` | 简历与给定 JD 临时对照（不写库） |
| POST | `/api/screen-candidate` | 完整初筛并写入候选人库 |
| GET | `/api/candidates?min_score=` | 候选人匹配排名（按匹配分降序） |
| DELETE | `/api/candidates/{id}` | 删除候选人 |

stdio 与 Streamable HTTP 对比：

| | stdio（recruit_server.py） | Streamable HTTP（http_app.py） |
| --- | --- | --- |
| 连接方式 | 给命令，客户端本地拉起一个进程 | 给一个 URL，服务常驻 |
| 客户端数量 | 一个客户端配一个进程 | 一个服务可被多个网络客户端共享 |
| 典型场景 | 本地桌面 Agent、Claude Desktop | 服务端部署、远程/多客户端、团队共享 |
| 业务代码 | 同一套 8 个工具、同一个 Agent | 只改连接配置，工具与 Agent 零改动 |

> 本地默认只监听回环地址 127.0.0.1，且 FastMCP 默认只允许 localhost 访问；部署到服务器时需把 host 改为 `0.0.0.0`、在 FastMCP 配置里放开访问来源（TransportSecurity），并自行加鉴权 / 网关与 CORS，再对公网开放。

## 工程化：健壮性、降级与可观测

Agent 不是"能跑通"就够，真实环境里模型会超时、工具会报错、精排服务会不可用。本项目在主流程之外补了三层工程能力。

### 1. 五层容错

| 层 | 防线 | 做法 |
| --- | --- | --- |
| ① LLM 层 | 网络抖动 / 模型偶发报错 | 模型客户端统一 `timeout=60`、`max_retries=3` |
| ② 工具层 | 单个工具内部异常 | 工具函数体内 try/except，返回结构化 `{"error","detail"}`（不用装饰器，避免 FastMCP 丢掉参数类型注解、破坏工具 schema） |
| ③ 串联层 | 后一步依赖前一步结果 | 写库链路拆成"简历分析 → 产出校验 → 匹配 → 写库"，任一步失败立即**短路**、绝不碰库；关键步骤用 `_run_step` 做指数退避重试 |
| ④ 编排层 | 工具失败后 Agent 还在瞎调 | system prompt 约定：看到工具返回 `error` 就停止调用依赖它的后续工具、不编造、如实告知 |
| ⑤ 检索层 | 精排服务不可用 | rerank 精排包在 try/except 里，失败自动降级回 BM25+向量 RRF 粗排结果 |

设计取舍：**核心数据依赖用快速失败（Fail Fast）**，避免脏数据；**非核心增强用降级**（精排挂了就用粗排），保证仍有可用结果。

### 2. 精排 / 粗排开关

`hybrid_search(query, top_k=3, use_rerank=True)` 多一个布尔参数：

- `use_rerank=True`（默认）：BM25 + 向量 RRF 粗排 → bge-reranker 精排，最准、多一次模型调用；
- `use_rerank=False`：只做 RRF 粗排，更快、完全不依赖精排服务。

开关一路透传到四个入口：`hybrid_search`（核心）、LangGraph 工具 `retrieve_jd_info`、MCP 工具 `search_jd_knowledge`（参数会进工具 schema，Agent 自己可见）、REST `/api/search?use_rerank=false`。

### 3. 可观测：每轮工具调用 / token / 耗时日志

用 LangChain 的**回调（Callback）**机制实现：框架在"模型开始/结束、工具开始/结束"这些节点自动回调你写好的函数，因此业务代码零侵入——只要在 `ainvoke` 的 `config` 里挂上 handler：

```python
from recruit_mcp.observability import ObservabilityHandler

handler = ObservabilityHandler(run_title="场景1", log_path=LOG_FILE)
await agent.ainvoke(
    {"messages": [{"role": "user", "content": "..."}]},
    config={"recursion_limit": 15, "callbacks": [handler]},
)
handler.print_summary()
```

运行时实时看到每一步（LLM 两次 + 工具一次），结束给汇总：

```text
  [LLM] 开始：ChatOpenAI
  [LLM] 结束：1.04s | 输入 198 / 输出 24 tokens
  [工具] 开始：add  入参={'a': 3, 'b': 5}
  [工具] 成功：add  0.00s
  [LLM] 开始：ChatOpenAI
  [LLM] 结束：0.53s | 输入 242 / 输出 10 tokens
  —— 本轮汇总：LLM 调用 2 次，工具调用 1 次（失败 0），token 输入 440 / 输出 34 / 合计 474
```

- 每轮汇总还会追加一行 JSON 到 `logs/agent_runs.jsonl`（JSONL：一行一条记录），便于事后统计成本、失败率、各工具耗时；
- token 兼容新旧两种返回位置（`message.usage_metadata` 与 `llm_output["token_usage"]`）；
- 这正是 Langfuse / LangSmith 的底层原理（都是一个 callback handler）。想要 Web 可视化界面时，把 `ObservabilityHandler` 换成 Langfuse 的 `CallbackHandler`（`config={"callbacks":[langfuse_handler]}`）即可，业务代码无需改动。

### 4. 回归测试

```powershell
pip install -e ".[dev]"
pytest          # 现有用例只依赖标准库，不需要 API_KEY / 向量库，可离线跑
```

第一批用例守的是**候选人写入的不变式**（`tests/test_db_upsert.py`）：重复入库必须原地更新、`id` 不能变，否则 `matchings.candidate_id` 会指向一条不存在的候选人 —— 这种数据损坏不会报错，只能靠测试拦。

## 数据说明

`data/` 下为**拟真示例 JD**，由作者编写、不对应任何真实公司，可自由用于演示。接入真实 JD 时请先脱敏（公司名、联系人、保密薪资）。

统一工作台支持上传 PDF / Word / 图片 / txt 的真实 JD 与简历（图片走视觉模型 OCR）。真实数据只写本地 SQLite（`*.db` 已 gitignore）与本地向量库（`chroma_db/` 已 gitignore），密钥在 `.env`（已 gitignore），均不会进入仓库。

## 路线图（Roadmap）

- [x] **LangGraph**：StateGraph 状态机、循环/分支、checkpoint 持久化（Sqlite）；多智能体 Supervisor 串行编排——简历分析 → 人岗匹配双专家，InjectedState 自动传递画像、RAG 自动检索岗位（`recruit_graph/resume_multi_agent.py`）。
- [x] **多智能体并行编排**：静态 fan-out/fan-in 三专家（`resume_parallel_agents.py`，reducer 累加、自动屏障等齐）+ Send 动态扇出 N 份简历批量初筛（`resume_batch_screen.py`，运行时按份数复制专家）。
- [x] **人工审批 interrupt**：工具执行前暂停、把待执行动作（工具名 + 参数）交人确认，批准执行 / 拒绝中止；可扩展到跨平台认人等场景。
- [x] **混合检索 RAG**：BM25 关键词（jieba）+ 向量召回 → RRF 融合（k=60）→ bge-reranker 精排，替换原有纯向量检索（`recruit_graph/hybrid_retriever.py`）。
- [x] **结构化数据层**：SQLite（岗位/候选人/匹配）+ Chroma 双写，岗位增删改两边一致；`seed_jds.py` 幂等清空重灌。
- [x] **真实简历初筛闭环**：8 份拟真简历批量"画像 → 全量 JD 横向匹配打分 → 落库 → 排名"，修复评分区分度与岗位名脏值（`seed_candidates.py`）。
- [x] **MCP（真实后端）**：8 个标准 MCP 工具直接复用 SQLite + Chroma + 混合检索——岗位查询/检索、简历画像/对照、初筛落库/排名/删除；stdio 即插即用，任意 MCP 客户端一次握手调用同一套后端（`recruit_mcp`）。
- [x] **MCP 服务化部署（Streamable HTTP）**：FastAPI 常驻进程同时暴露 REST（`/api`）与标准 MCP（`/mcp`），把 stdio 的"一个客户端拉起一个进程"升级为"一个服务、多个客户端通过 URL 共享"，Agent 与工具代码零改动；含会话管理器 lifespan 挂载、Swagger 文档，以及 8 个工具（含写库增删闭环）端到端验证（`recruit_mcp/http_app.py`、`agent_http_client.py`）。
- [x] **Web 界面（Gradio）**：本地网页聊天、后端打印消息类型与工具名、工具路由可观测（`recruit_graph/web_agent.py`）。
- [x] **思考过程可视化 + 网页记忆**：工具调用折叠块按顺序直播、每步可追溯；checkpointer + thread_id 支撑跨轮记忆（`recruit_graph/web_stream_agent.py`）。
- [x] **统一主管（Supervisor）路由三种编排**：顶层大模型主管按人话语义三选一，派发到串行 / 静态并行 / Send 三个**已编译 subgraph**（`resume_unified_supervisor.py`）；`Literal` 锁死合法路由、条件边派单，模型只选路不造路，三个真实场景实测全部正确。
- [x] **五层容错与健壮性**：LLM 超时重试、工具结构化 error、串联短路 + 退避重试、编排层停手、rerank 降级，避免脏数据与级联失败（`recruit_mcp/recruit_server.py`、`recruit_graph/candidate_store.py`、`hybrid_retriever.py`）。
- [x] **精排/粗排开关**：`hybrid_search(..., use_rerank=)` 一路透传到 LangGraph 工具 / MCP / REST，在最准（精排）与最快（粗排）间切换。
- [x] **可观测（Observability）**：基于 LangChain `AsyncCallbackHandler` 记录每轮 LLM/工具调用、token、耗时与成败，控制台实时日志 + JSONL 落盘，可平滑替换为 Langfuse（`recruit_mcp/observability.py`）。
- [x] **统一工作台 + 文档解析 + JD 生成**：Gradio 四模块工作台（岗位管理、简历逐份初筛累积、知识库问答与 JD 生成）；PDF/Word/图片 OCR 解析（`doc_parser.py`）；JD 生成器 few-shot RAG 检索同类范例 + 客户要求优先 + Function Calling 结构化输出；岗位按 id 更新与简历姓名防脏（`web_workbench.py`、`jd_generator.py`、`jd_store.py`）。
- [ ] **部署上线**：真实数据、服务器部署、公网可访问。
- [x] **Deep Agents 深度研究（多智能体规划/并行/共享文件）**：研究主管 write_todos 规划 + 两个 SubAgent 并行检索本地 JD、写共享虚拟文件，主管读回综合成每条带来源块 id 的报告；已接进统一工作台第 4 个标签「深度研究」，同时返回可读过程轨迹（`recruit_graph/deep_research.py`）。
- [ ] **能力 B（对标分析）**：一批同类 JD 反推合理技术栈与薪资行情。

## License

[MIT](LICENSE)
