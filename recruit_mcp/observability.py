"""Agent 可观测：用 LangChain 回调（Callback）记录每一步的工具调用、耗时与 token。

什么是回调（Callback）：
    框架在“模型开始/结束、工具开始/结束”这些关键节点会自动调用你写好的函数，
    你不用在业务代码里到处埋点，就能统一拿到日志和计数。
    Langfuse、LangSmith 这类平台，底层也都是这套回调机制。

记录三类信息：
    1. 每次 LLM 调用：模型、耗时、输入/输出 token；
    2. 每次工具调用：工具名、入参摘要、耗时、成功还是失败；
    3. 一轮对话结束的汇总：LLM 次数、工具次数、失败数、总 token。
"""

import json
import time
from datetime import datetime
from pathlib import Path

from langchain_core.callbacks import AsyncCallbackHandler


def _short(text, limit: int = 120) -> str:
    """把长文本压成一行、截到 limit 字，避免日志被超长入参/报错刷屏。"""
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "…"


class ObservabilityHandler(AsyncCallbackHandler):
    """记录一次 Agent 运行（一次 ask）里的 LLM 调用、工具调用与 token 用量。

    - verbose=True：每一步实时打印一行；
    - log_path：给了路径，就把每轮汇总追加写成一行 JSON（JSONL），方便事后统计。
    """

    def __init__(
        self,
        run_title: str = "",
        log_path: Path | None = None,
        verbose: bool = True,
    ):
        self.run_title = run_title
        self.log_path = log_path
        self.verbose = verbose

        self._starts = {}  # run_id（一次运行的唯一编号）-> 开始时间
        self._tool_names = {}  # run_id -> 工具名，用于把“开始”和“结束”配对

        self.llm_calls = 0
        self.tool_calls = 0
        self.tool_errors = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.events = []  # 结构化事件列表，可整体落盘

    # ---------------- LLM（大模型）----------------
    async def on_chat_model_start(self, serialized, messages, **kwargs):
        run_id = kwargs.get("run_id")
        self._starts[run_id] = time.perf_counter()
        model = (serialized or {}).get("name") or kwargs.get("name", "llm")
        self._log(f"[LLM] 开始：{model}")
        self.events.append({"type": "llm_start", "model": model})

    async def on_llm_end(self, response, **kwargs):
        run_id = kwargs.get("run_id")
        elapsed = self._elapsed(run_id)
        self.llm_calls += 1

        usage = self._extract_usage(response)
        self.prompt_tokens += usage["input"]
        self.completion_tokens += usage["output"]

        self._log(
            f"[LLM] 结束：{elapsed:.2f}s | "
            f"输入 {usage['input']} / 输出 {usage['output']} tokens"
        )
        self.events.append({"type": "llm_end", "elapsed": elapsed, **usage})

    # ---------------- Tool（工具）----------------
    async def on_tool_start(self, serialized, input_str, **kwargs):
        run_id = kwargs.get("run_id")
        self._starts[run_id] = time.perf_counter()
        name = (serialized or {}).get("name") or kwargs.get("name", "tool")
        self._tool_names[run_id] = name

        self._log(f"[工具] 开始：{name}  入参={_short(input_str)}")
        self.events.append({"type": "tool_start", "name": name})

    async def on_tool_end(self, output, **kwargs):
        run_id = kwargs.get("run_id")
        elapsed = self._elapsed(run_id)
        name = self._tool_names.pop(run_id, "tool")
        self.tool_calls += 1

        self._log(f"[工具] 成功：{name}  {elapsed:.2f}s")
        self.events.append(
            {"type": "tool_end", "name": name, "elapsed": elapsed, "ok": True}
        )

    async def on_tool_error(self, error, **kwargs):
        run_id = kwargs.get("run_id")
        elapsed = self._elapsed(run_id)
        name = self._tool_names.pop(run_id, "tool")
        self.tool_calls += 1
        self.tool_errors += 1

        self._log(f"[工具] 失败：{name}  {elapsed:.2f}s  {_short(error)}")
        self.events.append(
            {"type": "tool_end", "name": name, "elapsed": elapsed, "ok": False}
        )

    # ---------------- 汇总与落盘 ----------------
    def summary(self) -> dict:
        """返回本轮的计数汇总。"""
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }

    def print_summary(self):
        """打印一行汇总；若配置了 log_path，再追加一行 JSON 到 JSONL 文件。"""
        s = self.summary()
        self._log(
            "—— 本轮汇总："
            f"LLM 调用 {s['llm_calls']} 次，工具调用 {s['tool_calls']} 次"
            f"（失败 {s['tool_errors']}），"
            f"token 输入 {s['prompt_tokens']} / 输出 {s['completion_tokens']}"
            f" / 合计 {s['total_tokens']}"
        )
        if self.log_path:
            self._write_jsonl(s)

    def metrics_text(self, total: float | None = None) -> str:
        """把本轮汇总拼成一行中文指标，供网页"运行指标"框直接显示（各入口共用）。"""
        s = self.summary()
        text = (
            f"模型调用 {s['llm_calls']} 次｜工具调用 {s['tool_calls']} 次"
            f"（失败 {s['tool_errors']}）｜"
            f"token {s['prompt_tokens']} 入 / {s['completion_tokens']} 出"
        )
        if total is not None:
            text += f"｜总耗时 {total:.1f}s"
        return text

    def _write_jsonl(self, summary: dict):
        """把本轮汇总追加写成一行 JSON（JSONL：一行一条记录）。"""
        self.log_path = Path(self.log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "title": self.run_title,
            **summary,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---------------- 内部小工具 ----------------
    def _elapsed(self, run_id) -> float:
        """按 run_id 取出开始时间，算到现在经过了几秒。"""
        start = self._starts.pop(run_id, None)
        return time.perf_counter() - start if start else 0.0

    @staticmethod
    def _extract_usage(response) -> dict:
        """从模型响应里抠出输入/输出 token，兼容新版和旧版两种返回位置。"""
        # 新版 langchain-openai：token 在 message.usage_metadata
        try:
            message = response.generations[0][0].message
            usage = getattr(message, "usage_metadata", None)
            if usage:
                return {
                    "input": usage.get("input_tokens", 0),
                    "output": usage.get("output_tokens", 0),
                }
        except Exception:
            pass
        # 旧版位置：response.llm_output["token_usage"]
        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage", {}) or {}
        return {
            "input": token_usage.get("prompt_tokens", 0),
            "output": token_usage.get("completion_tokens", 0),
        }

    def _log(self, text: str):
        """统一的打印出口，方便以后一键改成 logging 或换成 Langfuse handler。"""
        if self.verbose:
            print(f"  {text}")
