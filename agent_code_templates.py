# =====================================================================
# AI Agent 开发 · 常用代码模板速查（个人模板库）
# ---------------------------------------------------------------------
# 三种用法：
#   1) 直接运行本文件：看底部“最小演示”；
#   2) 需要哪段：复制对应区块 / 函数；
#   3) 在别的文件里调用：from agent_code_templates import chat_once
# 前提：.env 文件里已配好  API_KEY / BASE_URL / MODEL
# 警告：密钥只放 .env，绝不要写死在代码里。
# =====================================================================


# ===== 区块0：固定 import（几乎每个项目开头都是这几行）=====
import os
import json
from pathlib import Path
import requests
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from rank_bm25 import BM25Okapi


# ===== 区块1：加载 .env + 建客户端（固定启动两步，照抄）=====
PROJECT_ROOT = Path(__file__).resolve().parent      # 本文件所在目录 = 项目根
load_dotenv(PROJECT_ROOT / ".env")                  # 无论从哪运行，都能找到根的 .env
API_KEY = os.getenv("API_KEY")                     # 密钥
BASE_URL = os.getenv("BASE_URL")                   # 接口地址
MODEL = os.getenv("MODEL")                         # 聊天模型（.env 里配的 Qwen）
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)  # 建好客户端，后面一直用

# 向量 / 精排模型名（硅基流动，按需替换）
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


# ===== 区块2：向量化器（新版 Chroma 1.x 用官方 OpenAIEmbeddingFunction）=====
def make_embedding_function():
    """返回可直接传给 Chroma 集合的向量化器（指向硅基流动、BGE 模型）。

    说明：Chroma 1.5.x 要求向量化器实现 name / get_config / build_from_config 一整套，
    手写一个只有 __call__ 的类会报 “'BGE' object has no attribute 'name'”。
    官方 OpenAIEmbeddingFunction 已实现这套，硅基流动又兼容 OpenAI 接口，直接用最稳。
    """
    return OpenAIEmbeddingFunction(
        api_key=API_KEY,
        api_base=BASE_URL,
        model_name=EMBEDDING_MODEL,
    )


# ===== 区块3：最简单的对话生成（一句问、一句答）=====
def chat_once(question, system_prompt=None):
    """输入：question 问题；system_prompt 可选的角色/规则。输出：模型回答（字符串）。"""
    messages = []
    if system_prompt:                              # 给了 system 就先放
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    resp = client.chat.completions.create(model=MODEL, messages=messages)
    return resp.choices[0].message.content         # 取回答文本


# ===== 区块4：Rerank 精排（requests 万能四件套：url/headers/json/解析）=====
def rerank(query, documents, top_n=3):
    """输入：问题、若干文档、留前几个。输出：[{index, score}, ...] 已按相关度降序。"""
    resp = requests.post(
        url=BASE_URL + "/rerank",
        headers={                                  # 鉴权 + 格式，三处都别拼错
            "Authorization": "Bearer " + API_KEY,  # Bearer 后有空格
            "Content-Type": "application/json",    # 中间是减号不是下划线
        },
        json={
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_n,                        # 只返回分数最高前 N 个
            "return_documents": False,
        },
    )
    results = resp.json()["results"]
    return [
        {"index": item["index"], "score": item["relevance_score"]}
        for item in results
    ]


# ===== 区块5：Chroma 向量库（持久化 + 集合 + 检索）=====
def get_collection(db_path="./chroma_db", collection_name="policy"):
    """打开/创建一个持久化集合。db_path 是本地文件夹，关掉再开数据还在。"""
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=make_embedding_function(),
    )
    return collection


def add_documents(collection, documents, ids, metadatas=None, use_upsert=True):
    """把文本写进集合：documents 文本列表、ids 唯一编号、metadatas 可选来源标签。

    use_upsert=True（默认）：有就更新、没有就新增，重跑不因 id 重复报错。
    """
    if use_upsert:
        collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
    else:
        collection.add(documents=documents, ids=ids, metadatas=metadatas)


def vector_search(collection, question, n_results=3):
    """向量检索：返回和问题最相近的 n_results 段文本。"""
    result = collection.query(query_texts=[question], n_results=n_results)
    return result["documents"][0]                  # 取第一条查询的结果列表


# ===== 区块6：BM25 关键词检索 =====
def bm25_ranking(question_tokens, tokenized_corpus):
    """输入：问题分词、语料分词。输出：按 BM25 分从高到低的“块下标”名次列表。"""
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(question_tokens)
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


# ===== 区块7：RRF 融合（把两路名次融合成一路，只看名次不看分）=====
def rrf_fuse(ranking_a, ranking_b, n_total, k=60):
    """输入：两路名次列表、总块数。输出：融合后的名次列表。"""
    scores = [0.0] * n_total                       # n_total 个格子，先清零
    for rank, index in enumerate(ranking_a):       # enumerate 同时给名次 rank 和块号
        scores[index] += 1.0 / (k + rank + 1)
    for rank, index in enumerate(ranking_b):
        scores[index] += 1.0 / (k + rank + 1)
    return sorted(range(n_total), key=lambda i: scores[i], reverse=True)


# ===== 区块8：拼资料 + 防幻觉护栏 + 生成（RAG 最后一步，直接抄）=====
def answer_with_context(question, context_documents):
    """输入：问题、检索回来的资料段。输出：带依据章节、防幻觉的自然语言答案。"""
    context = "\n".join(context_documents)         # 用换行把几段资料拼成一段
    system_prompt = (
        "你是严谨的制度问答助手。只根据【参考资料】回答，并在答案末尾注明依据章节；"
        "资料中没有的信息，直接回答‘资料中没有相关信息’，禁止编造。"
    )
    user_prompt = f"【参考资料】\n{context}\n\n【问题】\n{question}"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


# ===== 区块9：文本切块（按中文句号切句，RAG 常用第一步）=====
def split_sentences(text):
    """按句号切成句子列表：补回句号、滤掉空串。"""
    return [s + "。" for s in text.split("。") if s]


# ===== 区块10：ReAct 主循环骨架（需要工具时用，完整可抄版见 react_solo.py）=====
# 流程（边写边对照）：
#   while True:
#       调 chat.completions.create(model, messages, tools=菜单)
#       取 msg 和 finish_reason
#       若 finish_reason == "tool_calls"：
#           messages.append(msg)                   # 先把模型这轮(含申请单)记上
#           对每张申请单 tc：
#               args = json.loads(tc.function.arguments)  # 参数 JSON 字符串转字典
#               result = TOOLS_MAP[tc.function.name](args) # 分发表执行
#               messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
#           continue                               # 回到循环，让模型看结果再想
#       否则（stop）：
#           print(msg.content)                     # 最终答案
#           break
#   安全：加轮数计数，超过 6 轮强制 break，防止工具调个不停。


# =====================================================================
# 最小演示：直接运行本文件时执行（只调一次对话，不碰向量库）
# =====================================================================
if __name__ == "__main__":
    answer = chat_once("用一句话鼓励一个正在学编程的人")
    print(answer)
