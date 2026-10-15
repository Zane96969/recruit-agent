# seed_candidates.py —— 一次性重建候选人数据：清空旧假人，再用 save_candidate 逐份灌入简历
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recruit_graph.candidate_store import save_candidate, list_candidates, DB_PATH

# resume = 简历；RESUME_DIR 是仓库里的 data/resumes 目录
RESUME_DIR = Path(__file__).resolve().parent / "data" / "resumes"

# 第一组：清空旧候选人。候选人只在 SQLite（不像 JD 还要双写 Chroma 向量库），
# 所以这里只清数据库，不碰向量库。
conn = sqlite3.connect(DB_PATH)
# 先删评分记录 matchings（它用 candidate_id 关联候选人，先清子表，避免悬空评分）
conn.execute("DELETE FROM matchings")
# 再删候选人 candidates（把旧假人清掉）
conn.execute("DELETE FROM candidates")
# 重置自增，让新候选人 id 从 1 开始
conn.execute("DELETE FROM sqlite_sequence WHERE name = 'candidates'")
conn.commit()
conn.close()
print("SQLite 已清空 candidates、matchings")

# 第二组：遍历 data/resumes 每份简历，调 save_candidate 做“解析 + 匹配 + 落库”
# 注意：每份要跑 2 次大模型（解析画像、匹配打分），比较慢，逐份打印进度
files = sorted(RESUME_DIR.glob("*.txt"))
print(f"共 {len(files)} 份简历，开始逐份灌库（每份约十几秒，请耐心等待）…")
for path in files:
    resume_text = path.read_text(encoding="utf-8")
    result = save_candidate(resume_text)
    name = result["profile"]["name"]
    matched = result["match"]["matched_job"]
    score = result["match"]["match_score"]
    print(f"{path.name} → {name} | {matched} | {score}")

# 第三组：打印最终排名（分高→低），一眼看闭环是否跑通
print("\n===== 候选人排名（匹配分高 → 低）=====")
for c in list_candidates():
    print(f"{c['match_score']:>3}  {c['name']}  → {c['matched_job']}")
