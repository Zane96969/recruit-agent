# MCP 接入指南：让别的智能体把 recruit-agent 当工具用

> 本项目的定位是「harness 的工具供应商」：不做通用智能体运行时，
> 而是把招聘领域的能力（岗位库 / 简历解析 / 人岗匹配）按 MCP 标准
> （Model Context Protocol，模型上下文协议）暴露出去，
> 任何支持 MCP 的智能体——Codex、Claude Desktop、Cursor、Cline、
> 豆包工作、你自己写的 Agent——一次握手就能发现并调用这些能力。

---

## 一、对外提供哪些工具（共 10 个）

| 工具名 | 干什么 | 是否写库 |
|---|---|---|
| `list_positions` | 列出岗位库全部岗位（编号、名称、时间） | 否 |
| `get_position` | 按编号查看一份 JD 全文 | 否 |
| `search_jd_knowledge` | 自然语言搜 JD 知识库（BM25+向量+可选精排） | 否 |
| `save_position` | 录入/更新一份 JD（幂等，超长拒绝，跨存储一致） | 是 |
| `delete_position` | 删除一份岗位（记录+向量块） | 是 |
| `analyze_resume` | 只解析简历出画像，不落库 | 否 |
| `match_resume_with_jd` | 简历 × 一段给定 JD 临时匹配，不落库 | 否 |
| `screen_candidate` | 完整初筛：画像→全库匹配→写候选人库 | 是 |
| `rank_candidates` | 查候选人匹配排名（可按分数线过滤） | 否 |
| `delete_candidate` | 删除一名候选人 | 是 |

所有工具的失败都返回结构化 `{"error": ..., "detail": ...}`，
不会把异常直接甩进 MCP 通道；调用方（智能体）能读懂失败原因并转告用户。

---

## 二、两种接入姿势

| | stdio（本地进程） | HTTP（Streamable HTTP） |
|---|---|---|
| 启动方式 | 客户端自动拉起，**零配置服务** | 先手动起服务：`python run.py http` |
| 服务地址 | 无（走标准输入输出） | `http://127.0.0.1:8000/mcp` |
| 谁在跑 | 每个客户端一个独立进程 | 一个常驻服务，多客户端共享 |
| 适合 | 本机的 Codex / Claude Desktop / Cursor / Cline | 网络另一端的智能体（豆包工作、扣子等） |

选型口诀：**客户端和项目在同一台电脑 → stdio；要跨机器/跨网 → HTTP。**

---

## 三、各客户端接入配置

以下 `PYTHON` 一律指解释器绝对路径：

```
D:\agent-learning\recruit-agent\.venv\Scripts\python.exe
```

`SERVER` 一律指：

```
D:\agent-learning\recruit-agent\recruit_mcp\recruit_server.py
```

### 1. Codex（stdio）

编辑 `C:\Users\<你>\.codex\config.toml`，追加：

```toml
[mcp_servers.recruit]
command = "D:\\agent-learning\\recruit-agent\\.venv\\Scripts\\python.exe"
args = ["D:\\agent-learning\\recruit-agent\\recruit_mcp\\recruit_server.py"]
```

重启 Codex 后输入"岗位库里有哪些岗位"即可验证。

### 2. Claude Desktop（stdio）

编辑 `%APPDATA%\Claude\claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "recruit": {
      "command": "D:\\agent-learning\\recruit-agent\\.venv\\Scripts\\python.exe",
      "args": ["D:\\agent-learning\\recruit-agent\\recruit_mcp\\recruit_server.py"]
    }
  }
}
```

### 3. Cursor（stdio）

项目级 `.cursor/mcp.json`（随仓库走）或全局 `~\.cursor\mcp.json`，
格式与 Claude Desktop 相同。

### 4. Cline / Trae（stdio）

在插件的 MCP 设置界面（Cline：设置 → MCP Servers → Configure MCP Servers）
添加一条，格式同上：`command` 填 PYTHON，`args` 填 SERVER 路径。

### 5. 豆包工作 / 扣子这类云端智能体（HTTP，注意一个现实约束）

1. 本机先起服务：`python run.py http`（默认只监听本机）；
2. 平台的 MCP 配置里填端点 URL。

**关键约束**：豆包工作/扣子的 Agent 跑在云端，它访问不到你电脑上的
`127.0.0.1:8000`。要打通只有三条路：

- 把服务部署到有公网 IP 的服务器（改 `uvicorn` 监听 `0.0.0.0` 并加鉴权）；
- 用内网穿透工具（如 cpolar）把本机端口暴露成一个公网 URL 再填进去；
- 仅限同一局域网（填你电脑的局域网 IP，如 `http://192.168.x.x:8000/mcp`）。

填好 URL 后在平台上问一句"现在岗位库里有哪些岗位"验证握手。
若平台只支持 SSE 而不支持 Streamable HTTP，需要等平台侧支持或加转换层。

### 6. 自己写的 LangChain Agent（项目内置示例）

```bash
python run.py mcp
```

即 `recruit_mcp/agent_recruit_client.py`：stdio 拉起服务 → 握手发现工具 →
`create_agent(model, tools)` 组装。照抄这份文件就能把招聘能力接进任何
LangChain 项目。

---

## 四、验证与排障

**最小验证**（不起大模型，只验证握手）：在仓库根目录跑

```bash
python run.py mcp
```

第一行会打印"Agent 发现的工具"清单——应有 10 个。若想不消耗模型调用，
看到工具清单后 Ctrl+C 即可（握手本身不花钱，回答才花钱）。

**常见坑**：

| 症状 | 原因与解法 |
|---|---|
| 工具发现失败 | `command` 没用 `.venv` 里的 python 绝对路径（系统 python 没装依赖） |
| 调用报 API key 错 | 服务进程没读到根目录 `.env`；确认文件在 `D:\agent-learning\recruit-agent\.env` |
| HTTP 客户端连不上 | 服务没起（`python run.py http`）或端点写错（要带 `/mcp`） |
| 云端平台连不通 | 它访问不到你的 `127.0.0.1`，见第三节第 5 条的三条路 |

---

## 五、设计边界（为什么只有这 10 个）

- **暴露能力，不暴露应用**：只给"JD 检索 / 简历解析 / 人岗匹配"这类
  无状态、可独立理解的原子能力；不给 Gradio 界面。
- **编排留给外层**：多轮有状态流程（如"深度研究"）不封装成工具——
  那是外层智能体自己该做的规划，不是工具的事。
- **粒度偏粗**：`screen_candidate` 一个工具完成"画像→匹配→入库"全流程，
  而不是拆成三个细工具让调用方自己串——减少外层智能体的决策负担。
- **写路径带防护**：录岗/删岗走的是健壮化后的跨存储一致性事务，
  超长输入在入口被拒绝，失败以结构化错误返回。
