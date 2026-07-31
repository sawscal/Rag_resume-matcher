"""
Local RAG Fit Analyzer — No external API required.

Replaces the Gemini generative AI with a deterministic, rule-based
NLP analysis engine that:

  1. Parses both the resume and the job description with the NLP pipeline
  2. Computes skill overlap, skill gaps, and experience comparison
  3. Runs TF-IDF cosine similarity for keyword relevance
  4. Produces a structured fit report in the 1)/2)/3) format expected by the frontend

Zero API keys, zero network calls, zero cost, works fully offline.
"""

from __future__ import annotations

import re
from typing import List, Set

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.nlp_parser import parse_resume, extract_skills


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_jd_experience_requirement(jd_text: str) -> float:
    """
    Tries to pull a minimum years-of-experience requirement from the JD.
    e.g. "3+ years", "at least 2 years", "minimum 5 years experience"
    Returns 0.0 if not found.
    """
    patterns = [
        r"(\d+)\s*\+\s*years?\s+(?:of\s+)?(?:experience|exp)",
        r"(?:minimum|at\s+least|minimum\s+of)\s+(\d+)\s+years?",
        r"(\d+)\s+years?\s+(?:of\s+)?(?:relevant|related|professional)\s+experience",
    ]
    for pat in patterns:
        m = re.search(pat, jd_text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return 0.0


def _tfidf_score(resume_text: str, jd_text: str) -> float:
    """Returns TF-IDF cosine similarity between resume and JD (0–1)."""
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        mat = vec.fit_transform([resume_text, jd_text])
        score = float(cosine_similarity(mat[0], mat[1])[0][0])
        return round(score, 4)
    except Exception:
        return 0.0


def _bullet(items: List[str], prefix: str = "•") -> str:
    return "\n".join(f"  {prefix} {i}" for i in items) if items else "  (none detected)"


def _experience_verdict(resume_exp: float, required_exp: float) -> str:
    if required_exp == 0:
        return f"{resume_exp} yrs experience detected" if resume_exp > 0 else "Experience duration not specified"
    if resume_exp >= required_exp:
        return f"[OK] Meets requirement ({resume_exp} yrs detected / {required_exp} yrs required)"
    diff = required_exp - resume_exp
    return f"[!!] May fall short by ~{diff:.0f} yr(s) ({resume_exp} yrs detected / {required_exp} yrs required)"


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------

class RAGExplanationGenerator:
    """
    Rule-based NLP fit analyzer.
    Produces a structured 3-section fit report without any external API.
    """

    def generate_match_analysis(self, resume_text: str, job_description: str) -> str:
        """
        Analyzes candidate fit against the job description.

        Returns a formatted string with sections:
          1) Matching Skills & Strengths
          2) Skill Gaps
          3) Final Fit Verdict
        """
        # --- Parse both documents ---
        resume_parsed = parse_resume(resume_text)
        jd_skills = set(extract_skills(job_description))
        resume_skills = set(resume_parsed.skills)

        # --- Skill analysis ---
        matching_skills: List[str] = sorted(resume_skills & jd_skills)
        missing_skills: List[str] = sorted(jd_skills - resume_skills)
        extra_skills: List[str] = sorted(resume_skills - jd_skills)  # bonus skills

        # --- Experience comparison ---
        required_exp = _extract_jd_experience_requirement(job_description)
        exp_line = _experience_verdict(resume_parsed.experience_years, required_exp)

        # --- TF-IDF keyword relevance ---
        tfidf = _tfidf_score(resume_text, job_description)
        tfidf_pct = round(tfidf * 100)

        # --- Education ---
        edu_line = f"Education: {resume_parsed.education}" if resume_parsed.education else ""

        # --- Compute overall fit score ---
        # Weighted: skill match 60%, TF-IDF relevance 25%, experience 15%
        skill_match_ratio = len(matching_skills) / max(len(jd_skills), 1)
        exp_score = min(1.0, resume_parsed.experience_years / max(required_exp, 1.0)) if required_exp > 0 else 0.8
        overall = (skill_match_ratio * 0.60) + (tfidf * 0.25) + (exp_score * 0.15)
        overall_pct = round(overall * 100)

        # --- Decide fit label ---
        if overall_pct >= 65:
            fit_label = "[STRONG FIT]"
            fit_summary = "Candidate profile closely aligns with the job requirements."
        elif overall_pct >= 40:
            fit_label = "[PARTIAL FIT]"
            fit_summary = "Candidate has relevant experience but may need upskilling in key areas."
        else:
            fit_label = "[LOW FIT]"
            fit_summary = "Significant skill or experience gaps relative to the job requirements."

        # --- Build the 3-section report ---
        section1_parts = []
        if matching_skills:
            section1_parts.append(f"Matched {len(matching_skills)} of {len(jd_skills)} required skills:")
            section1_parts.append(_bullet(matching_skills))
        else:
            section1_parts.append("No exact skill keyword matches found against the job description.")
        if extra_skills:
            section1_parts.append(f"\nAdditional candidate skills not in JD ({len(extra_skills)}):")
            section1_parts.append(_bullet(extra_skills[:8]))  # cap at 8
        if edu_line:
            section1_parts.append(f"\n{edu_line}")
        section1_parts.append(f"\nKeyword relevance (TF-IDF): {tfidf_pct}%")

        section2_parts = []
        if missing_skills:
            section2_parts.append(f"{len(missing_skills)} required skill(s) not found in resume:")
            section2_parts.append(_bullet(missing_skills))
        else:
            section2_parts.append("No skill gaps detected — candidate has all listed required skills.")
        section2_parts.append(f"\nExperience: {exp_line}")

        section3_parts = [
            f"{fit_label}",
            f"{fit_summary}",
            f"\nOverall Fit Score: {overall_pct}%",
            f"  • Skill match:  {round(skill_match_ratio * 100)}% ({len(matching_skills)}/{max(len(jd_skills),1)} skills)",
            f"  • TF-IDF relevance: {tfidf_pct}%",
            f"  • Experience score: {round(exp_score * 100)}%",
        ]

        report = (
            f"1) {chr(10).join(section1_parts)}\n\n"
            f"2) {chr(10).join(section2_parts)}\n\n"
            f"3) {chr(10).join(section3_parts)}"
        )
        return report


# Global instance
rag_generator = RAGExplanationGenerator()
