import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
import chromadb
from agent_code_templates import split_sentences, API_KEY, BASE_URL, EMBEDDING_MODEL
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

all_chunks=[]
all_ids=[]
all_metadatas=[]
date_dir=str(PROJECT_ROOT / "data")
for doc_name in os.listdir(date_dir):
    if doc_name.endswith(".txt"):
        file_path = os.path.join(date_dir,doc_name)
        with open(file_path,encoding="utf-8") as f:
            content=f.read()
            print(doc_name,len(content))
        chunks=split_sentences(content)
        base_name=doc_name.replace(".txt","")
        
        
        for i,chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{base_name}_{i}")
            all_metadatas.append({"source":doc_name})
print("总块数:", len(all_chunks))
print("前3个id:", all_ids[:3])
chromadb_path=str(PROJECT_ROOT / "chroma_db")
chroma_client = chromadb.PersistentClient(path=chromadb_path)
embedding_model = OpenAIEmbeddingFunction(
    api_key=API_KEY,
    api_base=BASE_URL,
    model_name=EMBEDDING_MODEL,
)
jd_collection = chroma_client.get_or_create_collection(
        name="job_jd" ,
        embedding_function=embedding_model,
    )
jd_collection.upsert(documents=all_chunks,metadatas=all_metadatas,ids=all_ids)

