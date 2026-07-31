"""
NLP Text Parsing Pipeline
Extracts structured metadata from raw resume text:
  - Key skill sets (matched against a built-in skills taxonomy)
  - Experience duration (regex-based: explicit mentions + date range detection)
  - Education level (degree keyword matching)
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Skills taxonomy — 150+ skills across major domains
# ---------------------------------------------------------------------------
SKILLS_TAXONOMY = {
    # Programming Languages
    "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#", "Go",
    "Rust", "Kotlin", "Swift", "Ruby", "PHP", "Scala", "R", "MATLAB",
    "Bash", "Shell", "Perl", "Dart", "Lua",

    # Web / Frontend
    "React", "Vue", "Angular", "Next.js", "Nuxt.js", "HTML", "CSS",
    "Tailwind", "Bootstrap", "SASS", "SCSS", "jQuery", "Webpack", "Vite",
    "Redux", "GraphQL", "REST", "WebSockets",

    # Backend / Frameworks
    "FastAPI", "Flask", "Django", "Express", "Spring Boot", "Node.js",
    "Laravel", "Rails", "ASP.NET", "gRPC",

    # Data & ML
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "XGBoost",
    "LightGBM", "Hugging Face", "Transformers", "LangChain", "RAG",
    "FAISS", "Pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn",
    "Plotly", "OpenCV", "NLTK", "SpaCy",

    # Databases
    "MySQL", "PostgreSQL", "SQLite", "MongoDB", "Redis", "Cassandra",
    "DynamoDB", "Elasticsearch", "Neo4j", "Firebase", "Supabase",
    "SQL", "NoSQL", "Vector Database",

    # Cloud & DevOps
    "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform",
    "Ansible", "Jenkins", "GitHub Actions", "CI/CD", "Linux",
    "Nginx", "Apache", "Vercel", "Railway",

    # Data Engineering
    "Apache Spark", "Kafka", "Airflow", "dbt", "Snowflake", "BigQuery",
    "ETL", "Data Pipeline", "Hadoop",

    # Tools & Methodologies
    "Git", "GitHub", "Agile", "Scrum", "Jira", "Confluence",
    "Postman", "Swagger", "Unit Testing", "pytest", "TDD",

    # Design & Other
    "Figma", "Adobe XD", "UX", "UI", "API Design",
    "Microservices", "System Design", "OOP", "Functional Programming",
}

# Pre-build a lowercase → canonical name lookup for fast, case-insensitive matching
_SKILLS_LOWER: dict = {s.lower(): s for s in SKILLS_TAXONOMY}

# ---------------------------------------------------------------------------
# Education keyword patterns
# ---------------------------------------------------------------------------
EDUCATION_PATTERNS = [
    (r"\bph\.?d\.?\b", "PhD"),
    (r"\bm\.?sc\.?\b|\bmaster(?:\'s)?\s+(?:of\s+)?(?:science|arts|engineering|technology|business)\b", "Master's"),
    (r"\bmba\b", "MBA"),
    (r"\bb\.?tech\.?\b|\bbachelor(?:\'s)?\s+of\s+technology\b", "B.Tech"),
    (r"\bb\.?e\.?\b|\bbachelor(?:\'s)?\s+of\s+engineering\b", "B.E."),
    (r"\bb\.?sc\.?\b|\bbachelor(?:\'s)?\s+of\s+science\b", "B.Sc."),
    (r"\bb\.?a\.?\b|\bbachelor(?:\'s)?\s+of\s+arts\b", "B.A."),
    (r"\bassociate(?:\'s)?\s+degree\b", "Associate's"),
    (r"\bhigh\s+school\b|\bhsc\b|\bssc\b", "High School"),
]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class ParsedResume:
    skills: List[str] = field(default_factory=list)
    experience_years: float = 0.0
    education: Optional[str] = None
    raw_text_length: int = 0


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------

def extract_skills(text: str) -> List[str]:
    """
    Matches resume text against the skills taxonomy.
    Uses word-boundary-aware matching to avoid false positives (e.g. 'R' in 'React').
    """
    found = set()
    text_lower = text.lower()

    for skill_lower, skill_canonical in _SKILLS_LOWER.items():
        # Escape special regex chars in the skill name
        pattern = re.escape(skill_lower)
        # Require word boundaries (handles multi-word skills too)
        if re.search(rf"\b{pattern}\b", text_lower):
            found.add(skill_canonical)

    return sorted(found)


def extract_experience_years(text: str) -> float:
    """
    Extracts total years of professional experience from resume text.
    Strategy (highest confidence wins):
      1. Explicit mention: "5 years", "3+ yrs", "over 2 years"
      2. Date range: "Jan 2019 – Mar 2023" or "2019 – present"
    Returns the highest value found (assumes candidate lists most relevant experience first).
    """
    years_found: List[float] = []

    # --- Pattern 1: Explicit mention ---
    explicit_patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?(?:experience|exp|work)",
        r"(\d+(?:\.\d+)?)\s*\+?\s*yrs?\s+(?:of\s+)?(?:experience|exp|work)",
        r"over\s+(\d+(?:\.\d+)?)\s+years?",
        r"more\s+than\s+(\d+(?:\.\d+)?)\s+years?",
    ]
    for pat in explicit_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            try:
                years_found.append(float(m.group(1)))
            except (IndexError, ValueError):
                pass

    # --- Pattern 2: Date ranges ---
    # Matches "Month YYYY – Month YYYY" or "YYYY – YYYY" or "YYYY – present"
    current_year = 2025
    range_pattern = re.compile(
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?\.?\s*"
        r"(20\d{2}|19\d{2})"
        r"\s*[-–—to]+\s*"
        r"(?:(20\d{2}|19\d{2})|present|current|now)",
        re.IGNORECASE,
    )
    total_range_years = 0.0
    for m in range_pattern.finditer(text):
        start_year = int(m.group(1))
        end_str = m.group(2)
        end_year = int(end_str) if end_str else current_year
        duration = max(0, end_year - start_year)
        total_range_years += duration

    if total_range_years > 0:
        years_found.append(total_range_years)

    if not years_found:
        return 0.0
    return round(max(years_found), 1)


def extract_education(text: str) -> Optional[str]:
    """
    Detects the highest education level mentioned in the resume.
    Returns the canonical degree label or None.
    """
    # Priority order: highest degree first
    for pattern, label in EDUCATION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return label
    return None


def parse_resume(text: str) -> ParsedResume:
    """
    Full pipeline: runs all extractors and returns a ParsedResume dataclass.
    """
    return ParsedResume(
        skills=extract_skills(text),
        experience_years=extract_experience_years(text),
        education=extract_education(text),
        raw_text_length=len(text),
    )
