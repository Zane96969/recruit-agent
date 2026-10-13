from fastapi import FastAPI
from pydantic import BaseModel
from recruit_graph.graph_tools import analyze_and_match_resume
from recruit_graph.candidate_store import save_candidate, list_candidates

app = FastAPI()


@app.get("/")
def home():
    return {"message": "你好，招聘Agent上线了"}


class ResumeRequest(BaseModel):
    resume_text: str


@app.post("/analyze")
def analyze(req: ResumeRequest):
    report = analyze_and_match_resume.invoke({"resume_text": req.resume_text})
    return {"report": report}


@app.post("/candidates")
def add_candidate(req: ResumeRequest):
    """发一份简历：分析、打分、落库，返回画像和匹配结果。"""
    return save_candidate(req.resume_text)


@app.get("/candidates")
def get_candidates(min_score: int = 0):
    """看候选人排名；网址加 ?min_score=80 只看高分。"""
    return {"candidates": list_candidates(min_score)}
