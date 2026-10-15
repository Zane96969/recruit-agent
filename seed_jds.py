# seed_jds.py —— 一次性重建 JD 数据：先清空旧库，再用 save_jd 双写灌入种子 JD
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recruit_graph.jd_store import save_jd, DB_PATH
from recruit_graph.jd_vector_store import vectorstore

DATA_DIR = Path(__file__).resolve().parent / "data"

# 第一组：清空 Chroma 旧向量（保留 collection，倒空里面的块）
col = vectorstore._collection
old_ids = col.get()["ids"]  # 空①：col.get() 取出全部 id（回顾体检脚本怎么写的）
col.delete(ids=old_ids)  # 空②：按 id 批量删，参数名 ids=
print(f"Chroma 已清空 {len(old_ids)} 个旧块")

# 第二组：清空 SQLite 的 jds、悬空评分，并重置自增让 id 从 1 开始
conn = sqlite3.connect(DB_PATH)
conn.execute("DELETE FROM matchings")
conn.execute("DELETE FROM jds")
conn.execute(
    "DELETE FROM sqlite_sequence WHERE name = 'jds'"
)  # 空③：sqlite_sequence 里 name='jds' 那条计数
conn.commit()
conn.close()
print("SQLite 已清空 jds、matchings")

# 第三组：遍历 data 目录每份 txt，读全文、从首行取岗位名，调 save_jd 双写
for path in sorted(DATA_DIR.glob("*.txt")):
    jd_text = path.read_text(encoding="utf-8")
    jd_name = jd_text.splitlines()[0].replace("职位名称：", "").strip()
    result = save_jd(jd_name, jd_text)
    print(f"{path.name} → {result}")
