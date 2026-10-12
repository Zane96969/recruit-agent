import requests

url = "http://127.0.0.1:8000/analyze"

data = {
    "resume_text": "张三，熟悉Python、LangChain、LangGraph，期望AI Agent开发实习。",
}

try:
    resp = requests.post(url, json=data, timeout=60)
    resp.raise_for_status()
    print(resp.json())
except requests.exceptions.ConnectionError:
    print("连不上后端：请先启动 uvicorn")
except requests.exceptions.HTTPError as error:
    print("接口返回错误：", error)
