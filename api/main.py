from fastapi import FastAPI
from pydantic import BaseModel
from recruit_graph.graph_tools import analyze_and_match_resume
app=FastAPI()
@app.get("/")
def home():
    return {"message": "你好，招聘Agent上线了"}
class ResumeRequest(BaseModel):
    resume_text:str
@app.post("/analyze")
def analyze(req:ResumeRequest):
    report = analyze_and_match_resume.invoke({"resume_text": req.resume_text})
    return{"report":report}
 