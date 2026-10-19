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
)

REPO_ROOT = Path(__file__).resolve().parent.parent


vectorstore = Chroma(
    collection_name="job_jd",
    persist_directory=str(REPO_ROOT / "chroma_db"),
    embedding_function=embeddings,
)


def split_sentences(text):
    """按句号切成句子列表：补回句号、滤掉空白块。"""
    return [s.strip() + "。" for s in text.split("。") if s.strip()]


def upsert_jd(jd_id: int, jd_name: str, jd_text: str) -> int:
    """把一份 JD 切块、向量化、写进 job_jd，按 jd_id 编号，返回块数。"""
    chunks = split_sentences(jd_text)
    ids = [f"jd{jd_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"jd_id": jd_id, "jd_name": jd_name} for _ in chunks]
    vectorstore.add_texts(texts=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def delete_jd_vectors(jd_id: int) -> None:
    """按 jd_id 删掉一份 JD 的所有向量块。"""
    vectorstore._collection.delete(where={"jd_id": jd_id})


def rename_jd_vectors(jd_id: int, jd_name: str) -> None:
    """只改岗位名（正文未变）：原地更新各块 metadata 的 jd_name，不重新向量化。"""
    data = vectorstore._collection.get(where={"jd_id": jd_id})
    ids = data.get("ids") or []
    if not ids:
        return
    metadatas = [{"jd_id": jd_id, "jd_name": jd_name} for _ in ids]
    vectorstore._collection.update(ids=ids, metadatas=metadatas)
