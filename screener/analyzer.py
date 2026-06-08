"""AI resume analyzer and job fit scorer."""
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field
from typing import List, Optional
import json

class CandidateProfile(BaseModel):
    name: str
    skills: List[str]
    years_experience: float
    education_level: str
    achievements: List[str]
    languages: List[str]
    fit_score: Optional[float] = None

SYSTEM_PROMPT = """You are an expert HR analyst. Extract structured information from resumes.
Always return valid JSON. Be objective and factual. Do not make assumptions about gender, age or background."""

class ResumeAnalyzer:
    def __init__(self):
        self.llm = ChatVertexAI(model_name="gemini-1.5-pro-002", temperature=0)

    def extract_profile(self, resume_text: str) -> CandidateProfile:
        prompt = f"""{SYSTEM_PROMPT}

Resume:
{resume_text[:4000]}

Extract to JSON: name, skills (list), years_experience (float), education_level, achievements (list), languages (list)"""
        resp = self.llm.invoke(prompt).content
        data = json.loads(resp.split("```json")[-1].split("```")[0] if "```" in resp else resp)
        return CandidateProfile(**data)

    def score_fit(self, profile: CandidateProfile, job_description: str) -> float:
        prompt = f"""Score how well this candidate fits the job (0.0-1.0). Return only the score.

Candidate: Skills={profile.skills}, Experience={profile.years_experience}y, Education={profile.education_level}

Job: {job_description[:2000]}"""
        score = float(self.llm.invoke(prompt).content.strip())
        return min(max(score, 0.0), 1.0)

    def generate_interview_questions(self, profile: CandidateProfile, job_description: str) -> List[str]:
        prompt = f"""Generate 8 tailored interview questions for this candidate.
Profile: {profile.model_dump()}
Job: {job_description[:1000]}
Mix: technical depth, behavioral, situational. Return as JSON list."""
        resp = self.llm.invoke(prompt).content
        data = json.loads(resp.split("```json")[-1].split("```")[0] if "```" in resp else resp)
        return data if isinstance(data, list) else data.get("questions", [])

    def detect_bias_risk(self, decision_history: list) -> dict:
        genders = [d.get("inferred_gender") for d in decision_history if d.get("inferred_gender")]
        if not genders: return {"bias_detected": False}
        male_rate = sum(1 for g in genders if g == "male") / len(genders)
        return {"bias_detected": abs(male_rate - 0.5) > 0.2, "male_selection_rate": male_rate,
                "alert": "Potential gender bias: review screening criteria" if abs(male_rate - 0.5) > 0.2 else "OK"}
