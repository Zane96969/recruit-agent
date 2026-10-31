import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com")

embeddings = OpenAIEmbeddings(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="BAAI/bge-large-zh-v1.5",
    check_embedding_ctx_length=False,
    # 显式超时/重试：embedding 是远程调用，不能靠底层默认值，
    # 单次 30 秒上限、瞬时故障自动重试 3 次，和聊天模型同一套防护口径。
    timeout=30,
    max_retries=3,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


vectorstore = Chroma(
    collection_name="job_jd",
    persist_directory=str(REPO_ROOT / "chroma_db"),
    embedding_function=embeddings,
)

# 向量库内容版本号：每次增/删/重建都 +1。检索层（hybrid_retriever）靠它判断
# 本地缓存（全量块快照 + BM25 索引）是否失效，避免每次查询都重新拉库、重新分词。
_index_version = 0


def index_version() -> int:
    """当前内容版本号（进程内计数，配合集合总数一起作为缓存键）。"""
    return _index_version


def _bump_version() -> None:
    """内容变化时推进版本号，让检索层缓存立刻失效。"""
    global _index_version
    _index_version += 1


# 单块字符上限：bge 模型有 token 上限，无句号的整段长文若不切，
# 会退化成一个巨型块——超限部分被截断、语义检索整体失真。
MAX_CHUNK_CHARS = 300


def split_sentences(text):
    """按句号切成句子列表：补回句号、滤掉空白块。

    兜底：单句超过 MAX_CHUNK_CHARS（典型是无句号的整段粘贴文本）时
    再按长度硬切，保证每块都不超出 embedding 的处理能力。
    """
    sentences = [s.strip() + "。" for s in text.split("。") if s.strip()]
    chunks = []
    for sent in sentences:
        while len(sent) > MAX_CHUNK_CHARS:
            chunks.append(sent[:MAX_CHUNK_CHARS])
            sent = sent[MAX_CHUNK_CHARS:]
        chunks.append(sent)
    return chunks


def upsert_jd(jd_id: int, jd_name: str, jd_text: str) -> int:
    """把一份 JD 切块、向量化、写进 job_jd，按 jd_id 编号，返回块数。"""
    chunks = split_sentences(jd_text)
    ids = [f"jd{jd_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"jd_id": jd_id, "jd_name": jd_name} for _ in chunks]
    vectorstore.add_texts(texts=chunks, ids=ids, metadatas=metadatas)
    _bump_version()
    return len(chunks)


def delete_jd_vectors(jd_id: int) -> None:
    """按 jd_id 删掉一份 JD 的所有向量块。"""
    vectorstore._collection.delete(where={"jd_id": jd_id})
    _bump_version()


def rename_jd_vectors(jd_id: int, jd_name: str) -> None:
    """只改岗位名（正文未变）：原地更新各块 metadata 的 jd_name，不重新向量化。"""
    data = vectorstore._collection.get(where={"jd_id": jd_id})
    ids = data.get("ids") or []
    if not ids:
        return
    metadatas = [{"jd_id": jd_id, "jd_name": jd_name} for _ in ids]
    vectorstore._collection.update(ids=ids, metadatas=metadatas)
