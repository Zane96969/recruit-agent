# -*- coding: utf-8 -*-
"""项目启动器：永远用对的解释器，依赖缺了自己补。

为什么需要它：
项目依赖装在虚拟环境里，但命令行里敲 `python` 往往指向系统 Python（没装依赖），
于是动不动就是 `ModuleNotFoundError: No module named 'gradio'`。
这个脚本只依赖标准库（所以系统 Python 也能跑），它负责：

1. 找到可用的解释器：优先 `.venv`，其次 `--python` 指定的那个；
2. `.venv` 不存在就用 uv / pip 自动装好（只装一次，之后秒起）；
3. 永远在仓库根目录启动，相对路径（data/、chroma_db/）不会错。

用法：
    python run.py                    # 默认：统一工作台（http://127.0.0.1:7860）
    python run.py app                # LangGraph 版：状态机 + 记忆 + 人工审批
    python run.py deep               # 命令行跑一次深度研究
    python run.py --list             # 看全部入口
    python run.py --python D:\\xxx\\.venv\\Scripts\\python.exe   # 借用别人的环境（会记住）
    python run.py --forget           # 忘掉记住的解释器，重新自动选择
也可以直接双击 start.bat（等价于 python run.py）。
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
VENV_PY = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
# 记住上次用的解释器，下次不用再敲长路径（git 忽略，不随仓库走）
PIN_FILE = ROOT / ".python-path"

# 名称 -> (启动参数, 说明)。包内模块统一用 -m，不受当前目录影响。
ENTRIES = {
    "workbench": (["web_workbench.py"], "统一工作台：岗位 / 简历初筛 / 问答 / 深度研究"),
    "qa": (["recruit_qa.py"], "手写版：知识库检索问答（带出处）"),
    "profile": (["recruit_profile.py"], "手写版：JD → 人才画像"),
    "lc": (["-m", "recruit_lc.agent"], "LangChain 版：工具调用 + 记忆"),
    "app": (["-m", "recruit_graph.app"], "LangGraph 版：状态机 + 持久记忆 + 人工审批"),
    "web": (["-m", "recruit_graph.web_stream_agent"], "网页版：思考过程直播 + 跨轮记忆"),
    "multi": (["-m", "recruit_graph.resume_multi_agent"], "多智能体：串行双专家"),
    "parallel": (["-m", "recruit_graph.resume_parallel_agents"], "多智能体：三专家静态并行"),
    "batch": (["-m", "recruit_graph.resume_batch_screen"], "多智能体：Send 批量初筛"),
    "supervisor": (["-m", "recruit_graph.resume_unified_supervisor"], "统一调度主管 + 三子图"),
    "deep": (["-m", "recruit_graph.deep_research"], "命令行跑一次深度研究"),
    "http": (["-m", "recruit_mcp.http_app"], "REST(/api) + MCP(/mcp) 常驻服务"),
    "mcp": (["-m", "recruit_mcp.agent_recruit_client"], "MCP 客户端演示（stdio）"),
}
DEFAULT_ENTRY = "workbench"

# 判断"这个解释器能不能跑本项目"的最小依赖集合
PROBE = "import gradio, langchain, chromadb, jieba"


def print_entries(ready_python: Path | None = None) -> None:
    print("可用入口（python run.py <名字>）：")
    width = max(len(name) for name in ENTRIES)
    for name, (_, desc) in ENTRIES.items():
        mark = "（默认）" if name == DEFAULT_ENTRY else ""
        print(f"  {name.ljust(width)}  {desc}{mark}")
    print(f"\n解释器：{ready_python or '尚未确认（首次运行会自动安装依赖）'}")


def read_pin() -> Path | None:
    """读上次记住的解释器；没有或读不出来就返回 None。"""
    try:
        raw = PIN_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return Path(raw) if raw else None


def write_pin(python: Path) -> None:
    """记住解释器。写不进去也无所谓（只是下次要多敲一次路径）。"""
    try:
        PIN_FILE.write_text(str(python), encoding="utf-8")
    except OSError:
        pass


def find_uv() -> str | None:
    """找 uv（有就用它装依赖，能按 uv.lock 装出一模一样的版本）。"""
    found = shutil.which("uv")
    if found:
        return found
    guess = Path.home() / ".local" / "bin" / ("uv.exe" if os.name == "nt" else "uv")
    return str(guess) if guess.exists() else None


def deps_ok(python: Path) -> bool:
    """这个解释器能不能导入项目依赖。"""
    if not Path(python).exists():
        return False
    probe = subprocess.run([str(python), "-c", PROBE], capture_output=True)
    return probe.returncode == 0


def create_env() -> bool:
    """准备 `.venv`：能用 uv 就按 uv.lock 装（快且版本固定），否则退回 pip。"""
    uv = find_uv()
    if uv:
        print("首次运行：用 uv 按 uv.lock 安装依赖（几分钟，只需一次）…")
        if subprocess.run([uv, "sync", "--frozen"], cwd=ROOT).returncode == 0:
            return deps_ok(VENV_PY)
        print("uv 安装没成功，改用 pip 再试一次…")

    if not VENV_PY.exists():
        print(f"创建虚拟环境：{VENV_DIR}")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    print("用 pip 安装依赖…（第一次要几分钟）")
    result = subprocess.run([str(VENV_PY), "-m", "pip", "install", "-e", "."], cwd=ROOT)
    return result.returncode == 0 and deps_ok(VENV_PY)


def parse_args(argv: list[str]) -> tuple[str | None, list[str]]:
    """拆出 --python 覆盖值，其余原样返回。"""
    python_override = os.environ.get("RECRUIT_PYTHON")
    rest: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--python" and index + 1 < len(argv):
            python_override = argv[index + 1]
            index += 2
            continue
        if arg.startswith("--python="):
            python_override = arg.split("=", 1)[1]
            index += 1
            continue
        rest.append(arg)
        index += 1
    return python_override, rest


def resolve_python(override: str | None) -> Path | None:
    """决定用哪个解释器：--python/环境变量 > 记住的 > .venv > 自动安装。"""
    if override:
        candidate = Path(override).expanduser()
        if not deps_ok(candidate):
            raise SystemExit(
                f"指定的解释器不可用：{candidate}\n"
                f"（要么路径不对，要么它没装本项目依赖；可以用下面的命令自查：\n"
                f"  \"{candidate}\" -c \"{PROBE}\" ）"
            )
        write_pin(candidate)  # 记住了，下次不用再敲
        return candidate

    pinned = read_pin()
    if pinned:
        if deps_ok(pinned):
            print(f"沿用上次的解释器：{pinned}（想换掉：python run.py --forget）")
            return pinned
        # 记住的可能被删了/换了版本：静默忽略，别让用户先修环境才能启动
        print(f"上次用的解释器已不可用，忽略：{pinned}")

    if deps_ok(VENV_PY):
        return VENV_PY

    if create_env() and deps_ok(VENV_PY):
        write_pin(VENV_PY)
        return VENV_PY

    return None


def describe_python(override: str | None) -> Path | None:
    """--list 用：只探测现有环境，绝不触发安装。"""
    for candidate in (
        Path(override).expanduser() if override else None,
        read_pin(),
        VENV_PY,
    ):
        if candidate and deps_ok(candidate):
            return candidate
    return None


def main() -> int:
    python_override, rest = parse_args(sys.argv[1:])

    if rest and rest[0] == "--forget":
        if PIN_FILE.exists():
            PIN_FILE.unlink()
            print("已忘掉记住的解释器，下次运行会重新自动选择。")
        else:
            print("本来就没记住过解释器，无需清除。")
        return 0

    if rest and rest[0] in {"-h", "--help", "--list", "-l"}:
        print_entries(describe_python(python_override))
        return 0

    entry = rest[0] if rest else DEFAULT_ENTRY
    if entry not in ENTRIES:
        print(f"没有叫 {entry!r} 的入口。\n")
        print_entries()
        return 1

    python = resolve_python(python_override)
    if python is None:
        print(
            "\n依赖没装好，暂时跑不起来。两条路：\n"
            "  1) 再试一次安装：\n"
            f"     {VENV_PY} -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple\n"
            "  2) 机器上已有别的能跑的环境，直接借用：\n"
            "     python run.py --python <那个环境>\\Scripts\\python.exe\n"
            "     （或设环境变量 RECRUIT_PYTHON=...）\n"
        )
        return 1

    target = ENTRIES[entry][0] + rest[1:]
    print(f"解释器：{python}")
    print(f"启动：{' '.join(target)}")
    print("（Ctrl+C 停止；网页入口启动后打开 http://127.0.0.1:7860）\n")
    # cwd 固定在仓库根目录：README 里所有相对路径都以它为准
    return subprocess.call([str(python), *target], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
