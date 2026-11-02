"""能力 B：对标分析（Benchmark Analysis，对标分析）。

喂入一批同类 JD，反推这行现在普遍要会什么技术、薪资大概什么水平。
设计原则：薪资、技术栈的数字全部用确定性代码统计（保证准确），
LLM（大语言模型）只负责把统计结果写成通顺的人话报告，不碰原始数字。
"""

import re
import statistics
from collections import Counter

# 薪资数字对：匹配 "12K-20K"、"12k~20k"、"12-20K" 这类写法。
# (\d+(?:\.\d+)?) 抓一个数字（允许小数）；[kK]? 表示 K 可有可无；
# [-~—至到] 是连接符字符类，覆盖横杠、波浪线、破折号、汉字"至/到"。
_SALARY_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*[kK]?\s*[-~—至到]\s*(\d+(?:\.\d+)?)\s*[kK]?"
)


def parse_salary(salary_text):
    """把薪资字符串解析成 (下限K, 上限K) 两个数字；解析不了返回 None。

    例："12K-20K · 14薪" -> (12.0, 20.0)
        "6K-12K + 提成" -> (6.0, 12.0)
    """
    if not salary_text:
        return None
    match = _SALARY_PATTERN.search(salary_text)
    if not match:
        return None
    low = float(match.group(1))
    high = float(match.group(2))
    if low > high:
        # 万一 JD 把两个数字写反了，交换回来保证下限小、上限大
        low, high = high, low
    return (low, high)


# ===================== ② 聚合统计（纯函数，可离线测） =====================
# 一份 JD 抽取后的"事实"（fact）用这个结构表示：
# {
#   "name": 岗位名,
#   "salary_low"/"salary_high": 薪资下限/上限（K，解析不了为 None）,
#   "locations": 地点字符串列表（一份可能写多个城市）,
#   "core_tech"/"optional_tech": 核心/可选技术栈列表,
# }


def _pct(count, total):
    """占比（percentage）：出现份数 ÷ 总份数，转成百分比、保留 1 位小数。"""
    if total == 0:
        return 0.0
    return round(count / total * 100, 1)


# 技术名词别名表：key 用小写，value 是统一后的标准写法。
# 解决"LangChain 框架"和"LangChain"被当成两个技术的问题。
_TECH_ALIASES = {
    "langchain框架": "LangChain",
    "milvus向量数据库": "Milvus",
    "milvus数据库": "Milvus",
    "chroma向量数据库": "Chroma",
    "chroma数据库": "Chroma",
    "openai": "OpenAI 兼容接口",
}


def normalize_tech(name):
    """技术名词归一化（normalization）：去多余空格，把常见别名统一成标准名。"""
    key = name.strip()
    # 去掉内部空格再查别名，兼容“LangChain 框架”和“LangChain框架”两种写法
    compact = re.sub(r"\s+", "", key).lower()
    alias = _TECH_ALIASES.get(compact)
    return alias if alias else key


def count_tech(facts):
    """统计每个技术在多少份 JD 出现及占比，核心/可选分开。

    返回 {"core": [(技术, 份数, 占比), ...], "optional": [...]}，按份数降序。
    同一份 JD 里同一技术重复写，用 set 去重后只算一次；
    统计前先归一化，合并同一技术的不同写法。
    """
    total = len(facts)
    core_counter = Counter()
    optional_counter = Counter()
    for fact in facts:
        core = {normalize_tech(tech) for tech in fact["core_tech"]}
        optional = {normalize_tech(tech) for tech in fact["optional_tech"]}
        core_counter.update(core)
        optional_counter.update(optional)

    def to_rows(counter):
        rows = []
        for tech, count in counter.most_common():
            rows.append((tech, count, _pct(count, total)))
        return rows

    return {"core": to_rows(core_counter), "optional": to_rows(optional_counter)}


def _stats(values):
    """一组数字的统计：最小、最大、中位数（median）、均值（mean）、个数。"""
    if not values:
        return None
    return {
        "min": min(values),
        "max": max(values),
        "median": round(statistics.median(values), 1),
        "mean": round(statistics.mean(values), 1),
        "count": len(values),
    }


def summarize_salary(facts):
    """分别统计薪资下限、上限的最小/最大/中位数/均值（单位 K）。"""
    lows = []
    highs = []
    for fact in facts:
        if fact.get("salary_low") is not None:
            lows.append(fact["salary_low"])
        if fact.get("salary_high") is not None:
            highs.append(fact["salary_high"])
    return {"low": _stats(lows), "high": _stats(highs)}


def count_location(facts):
    """统计工作地点出现频次；一份 JD 里的多个城市按 / 、 ， 拆开。"""
    counter = Counter()
    for fact in facts:
        cities = []
        for loc in fact["locations"]:
            cities.extend(re.split(r"[/、，,]", loc))
        cities = [city.strip() for city in cities if city.strip()]
        counter.update(set(cities))  # 同一份 JD 去重
    return counter.most_common()


# ===================== ③ LLM 批量抽取 + 主入口 =====================
def extract_fact(jd_text):
    """对单份 JD 调结构化模型抽取，再转成内部事实 dict。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    from recruit_graph.graph_tools import structured_model

    job = structured_model.invoke(
        [
            SystemMessage(
                content="你是一名招聘信息解析专家，请将JD信息提取为结构化数据。"
                "岗位要求没有明确写就返回空字符串。"
                "技术名词一律用最简标准写法，例如写“LangChain”不要写“LangChain 框架”，"
                "写“Milvus”不要写“Milvus 向量数据库”，保持同一技术前后名称一致。"
            ),
            HumanMessage(content=f"请将JD信息提取为结构化数据。\n{jd_text}"),
        ]
    )
    salary = parse_salary(job.salary)
    tech_stack = job.tech_stack
    return {
        "name": job.name,
        "salary_low": salary[0] if salary else None,
        "salary_high": salary[1] if salary else None,
        "locations": [job.location] if job.location else [],
        "core_tech": tech_stack.core_tech if tech_stack else [],
        "optional_tech": tech_stack.optional_tech if tech_stack else [],
    }


def analyze_benchmark(jd_texts=None, jd_ids=None):
    """对标分析主入口：返回含技术频次、薪资统计、地点的结构化报告。

    两种喂数方式：
    - jd_texts：直接给一批 JD 文本（list[str]）；
    - jd_ids：给岗位库里的 id 列表，从 SQLite 读原文；
    两个都不给：默认读岗位库全部 JD。
    单份 JD 抽取失败会跳过、不拖垮整批，记进 failed。
    """
    if jd_texts is None:
        from recruit_graph.jd_store import get_jd, list_jds

        if jd_ids is None:
            jd_ids = [jd["id"] for jd in list_jds()]
        jd_texts = [get_jd(jid)["jd_text"] for jid in jd_ids]

    facts = []
    failed = []
    for index, text in enumerate(jd_texts):
        try:
            facts.append(extract_fact(text))
        except Exception as exc:
            failed.append({"index": index, "detail": str(exc)})

    return {
        "sample_count": len(facts),
        "failed": failed,
        "facts": facts,
        "tech": count_tech(facts),
        "salary": summarize_salary(facts),
        "location": count_location(facts),
    }


# ===================== ④ LLM 人话报告（不重算数字） =====================
def write_benchmark_report(result):
    """把确定性统计结果交给 LLM 写成通顺报告，严格基于给定数字、不自行计算。"""
    from recruit_graph.graph_tools import chat_model

    prompt = f"""以下是对 {result['sample_count']} 份同类 JD 的确定性统计结果。
请基于这些给定数字写一份对标分析报告，分三块：
1. 技术栈：哪些是高频必备技术、哪些是加分项；
2. 薪资行情：薪资下限/上限的中位数和区间；
3. 给求职者的一句准备建议。
严格使用给定数字，禁止自行计算或编造统计里没有的数字。
薪资数字单位是 K（千元/月，例如 12K 表示月薪 1.2 万元），
请统一用 K 表示月薪，不要写成"万元/年"，也不要自行换算。

技术统计：{result['tech']}
薪资统计：{result['salary']}
地点统计：{result['location']}
"""
    answer = chat_model.invoke(prompt)
    return answer.content


def run_benchmark(jd_texts=None, jd_ids=None):
    """一站式：先跑确定性对标分析，再生成人话报告。返回 (报告文字, 结构化结果)。"""
    result = analyze_benchmark(jd_texts=jd_texts, jd_ids=jd_ids)
    report = write_benchmark_report(result)
    return report, result


if __name__ == "__main__":
    # 离线验证：用构造的模拟事实跑聚合，不需要 API_KEY。
    demo_facts = [
        {
            "name": "AI Agent 工程师 A",
            "salary_low": 12.0,
            "salary_high": 20.0,
            "locations": ["北京 / 远程"],
            "core_tech": ["Python", "RAG"],
            "optional_tech": ["LangChain"],
        },
        {
            "name": "AI Agent 工程师 B",
            "salary_low": 14.0,
            "salary_high": 22.0,
            "locations": ["北京"],
            "core_tech": ["Python", "LangGraph"],
            "optional_tech": ["LangChain", "Milvus"],
        },
        {
            "name": "AI Agent 工程师 C",
            "salary_low": 10.0,
            "salary_high": 18.0,
            "locations": ["上海 / 远程"],
            "core_tech": ["Python", "RAG"],
            "optional_tech": ["LangChain"],
        },
    ]
    print("薪资解析样例：", parse_salary("12K-20K · 14薪"))
    print("技术频次：", count_tech(demo_facts))
    print("薪资统计：", summarize_salary(demo_facts))
    print("地点统计：", count_location(demo_facts))
