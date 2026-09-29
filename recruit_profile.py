import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
from agent_code_templates import client, MODEL

jd_path = PROJECT_ROOT / "data" / "jd_ai_agent.txt"
with open(jd_path, "r", encoding="utf-8") as file:
    jd_text = file.read()

print("JD 字数:", len(jd_text))
print(jd_text)
profile_template = {
    "岗位本质": "",
    "必备硬技能": [],
    "加分技能": [],
    "技术栈": {"核心必用": [], "了解即可": []},
    "经验职级": "",
    "软素质": [],
    "薪资范围": "",
    "发展前景": "",
    "搜索关键词": [],
    "求职者自检清单": [],
}
profile_schema = json.dumps(profile_template, ensure_ascii=False, indent=2)
system_prompt = (
    "你是一位资深招聘专家，擅长把一份 JD 解读成清晰的人才画像。"
    "请严格按给定 JSON 模板填写，不要输出模板以外的内容。"
    "薪资、经验等 JD 写明的直接如实抽取；岗位本质、必备技能等基于 JD 归纳；"
    "发展前景 JD 里没有，资料不足就填‘资料不足，仅供参考’，"
    "禁止编造薪资行情、增长数字或发展趋势。只输出 JSON。"
)
user_prompt = f"【JD 原文】\n{jd_text}\n\n【JSON 模板】\n{profile_schema}\n\n请按模板输出人才画像 JSON。"

print(profile_schema)
resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type":"json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],)
profile_json_text = resp.choices[0].message.content
profile = json.loads(profile_json_text or "{}")   # 模型返回空串时兜底成 {}

print("=" * 40)
print("人才画像")
print("=" * 40)
print(json.dumps(profile, ensure_ascii=False, indent=2))
    