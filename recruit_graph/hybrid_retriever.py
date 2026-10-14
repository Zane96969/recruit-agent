import jieba
import requests
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from recruit_graph.jd_vector_store import API_KEY, BASE_URL, embeddings, vectorstore

RRF_K = 60
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def bm25_search(query: str, top_n: int = 5):
    """BM25 关键词检索，返回按相关度排序的若干块，每项为 (块id, 文本)。"""
    data = vectorstore._collection.get()
    ids = data["ids"]
    documents = data["documents"]

    query_tokens = list(jieba.cut(query))
    tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]

    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query_tokens)

    ranked_index = sorted(
        range(len(bm25_scores)),
        key=lambda index: bm25_scores[index],
        reverse=True,
    )
    return [(ids[i], documents[i]) for i in ranked_index[:top_n]]


def hybrid_retrieve(query: str, top_n: int = 5):
    """向量 + BM25 经 RRF 融合，返回粗排后的若干块，每项 (块id, 文本)。"""
    # 第一组：取全量块，再用 zip(拉链)把 id 和文本一一配对，建"id→文本"字典
    data = vectorstore._collection.get()
    ids = data["ids"]
    documents = data["documents"]
    id_to_doc = dict(zip(ids, documents))

    # 第二组：BM25 路——top_n 给全量，拿到所有块按关键词排好的 id 顺序
    bm25_hits = bm25_search(query, top_n=len(ids))
    bm25_ranked_ids = [cid for cid, _ in bm25_hits]

    # 第三组：向量路——语义召回，n_results 给全量，拿到按相似度排好的 id 顺序
    query_embedding = embeddings.embed_query(query)
    vec_result = vectorstore._collection.query(
        query_embeddings=[query_embedding],
        n_results=len(ids),
    )
    vector_ranked_ids = vec_result["ids"][0]

    # 第四组：每个 id 先记 0 分，再按它在两路里的名次各加一次 1/(常数+名次)
    rrf_scores = {cid: 0.0 for cid in ids}
    for rank, cid in enumerate(bm25_ranked_ids):
        rrf_scores[cid] += 1 / (RRF_K + rank)
    for rank, cid in enumerate(vector_ranked_ids):
        rrf_scores[cid] += 1 / (RRF_K + rank)

    # 第五组：按融合分降序，取前 top_n 个 id，再用字典换回文本
    fused_ids = sorted(ids, key=lambda cid: rrf_scores[cid], reverse=True)
    return [(cid, id_to_doc[cid]) for cid in fused_ids[:top_n]]


def rerank_candidates(query: str, candidates, top_k: int = 3):
    """用 bge-reranker 对粗排候选精排，返回 [(块id, 文本), ...]，长度 top_k。"""
    # 第一组：把候选里的文本抽成列表（rerank 接口只认文本，不认我们的块 id）
    documents = [doc for _, doc in candidates]

    # 第二组：发 POST 请求到 /rerank，把问题和候选文本一起发过去
    resp = requests.post(
        f"{BASE_URL}/rerank",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_k,
            "return_documents": False,
        },
    )
    results = resp.json()["results"]

    # 第三组：results 已按相关度从高到低排好，每项 item["index"] 是"它在你传入 documents 里的局部位置"
    # 用这个局部位置去 candidates 里取，才能把 (块id, 文本) 拿回来
    return [candidates[item["index"]] for item in results]


def hybrid_search(query: str, candidate_n: int = 8, top_k: int = 3):
    """混合检索总入口：BM25+向量 RRF 粗排 → bge-reranker 精排，
    返回 LangChain Document 列表（与 retriever.invoke 同类型）。"""
    # 第一组：粗排，多捞一些候选（candidate_n 个）
    candidates = hybrid_retrieve(query, top_n=candidate_n)

    # 第二组：精排，压到 top_k 个
    ranked = rerank_candidates(query, candidates, top_k=top_k)

    # 第三组：把每个 (块id, 文本) 包成 Document 盒子，返回类型和 retriever.invoke 一致
    return [Document(page_content=text, metadata={"id": cid}) for cid, text in ranked]
