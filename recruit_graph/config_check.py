"""启动前的配置自检（config check，配置检查）。

目的：把"配置缺失"从运行中途的深处异常（第一次调 embedding/LLM 时
在远程端 401），提前到启动那一刻——一句中文指引、立刻退出。

用法（启动器子进程里）：
    from dotenv import load_dotenv
    load_dotenv()
    missing = missing_keys()
    # missing 非空就打印指引并退出
"""

import os

# 必需配置项：模型与向量服务全都靠它。BASE_URL 有默认值，不算必需。
REQUIRED_KEYS = ("API_KEY",)


def missing_keys() -> list[str]:
    """返回缺失的必需配置项名称；空列表 = 配置齐全。

    只看进程环境变量（.env 的加载交给调用方，因为 .env 的位置
    随启动目录变化），保持本函数纯净、可单测。
    """
    return [name for name in REQUIRED_KEYS if not os.getenv(name)]
