

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
import chromadb
from agent_code_templates import split_sentences, API_KEY, BASE_URL, EMBEDDING_MODEL
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from agent_code_templates import make_embedding_function
import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from rank_bm25 import BM25Okapi
from agent_code_templates import split_sentences, API_KEY, BASE_URL, EMBEDDING_MODEL, make_embedding_function, client, MODEL



chromadb_path=str(PROJECT_ROOT / "chroma_db")
chroma_client = chromadb.PersistentClient(path=chromadb_path)
embedding_model = OpenAIEmbeddingFunction(
    api_key=API_KEY,
    api_base=BASE_URL,
    model_name=EMBEDDING_MODEL,
)
jd_collection = chroma_client.get_or_create_collection(
        name="job_jd" ,
        embedding_function=embedding_model,)
question="AI Agent 开发工程师的薪资是多少"
print("集合条数:", jd_collection.count())
result = jd_collection.query(query_texts=[question], n_results=2)
retrieved_documents=result["documents"][0]
retrieved_metadatas=result["metadatas"][0]
print("=== 检索到的 JD 块 ===")
for document,metadata in zip(retrieved_documents,retrieved_metadatas):
    print("来源:", metadata["source"])
    print("内容:", document)
    print("-" * 30)
context="\n".join(retrieved_documents)

system_prompt = (
        "你是严谨的招聘岗位问答助手。只根据【参考资料】回答，"
    "并在答案末尾注明出自哪份 JD；"
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
print("=== 最终答案 ===")
print(resp.choices[0].message.content)
