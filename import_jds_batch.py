# import_jds_batch.py —— 增量批量导入 JD（只追加 / 更新，绝不清空已有岗位）
#
# 用法三步：
#   1. 在 data/batch_jds/ 文件夹里，每份 JD 存成一个 .txt（首行写“职位名称：xxx”）
#   2. 在项目根目录运行：
#        D:\agent-learning\workspace\.venv\Scripts\python.exe import_jds_batch.py
#   3. 脚本逐个调用 save_jd，自动双写 SQLite + 向量库，可反复跑、不会重复
#      （同一份原文重复导入会更新原记录，不会产生新岗位）
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recruit_graph.jd_store import save_jd

BATCH_DIR = Path(__file__).resolve().parent / "data" / "batch_jds"


def main():
    if not BATCH_DIR.exists():
        BATCH_DIR.mkdir(parents=True, exist_ok=True)
        print(f"已创建文件夹：{BATCH_DIR}")
        print("请把每份 JD 的 .txt 文件放进去，然后重新运行本脚本。")
        return

    files = sorted(BATCH_DIR.glob("*.txt"))
    if not files:
        print(f"文件夹 {BATCH_DIR} 里没有 .txt 文件。")
        return

    count = 0
    for path in files:
        jd_text = path.read_text(encoding="utf-8").strip()
        if not jd_text:
            print(f"跳过空文件：{path.name}")
            continue

        first_line = jd_text.splitlines()[0]
        jd_name = first_line.replace("职位名称：", "").replace("职位名称:", "").strip()
        if not jd_name:
            jd_name = path.stem  # 首行没写岗位名，就用文件名当岗位名

        result = save_jd(jd_name, jd_text)
        count += 1
        print(f"{path.name} → 岗位 #{result['jd_id']}：{jd_name}")

    print(f"完成：本次导入 / 更新 {count} 份 JD（未清空、未删除任何已有岗位）。")


if __name__ == "__main__":
    main()
