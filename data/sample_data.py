import os

# Sample Resume data
SAMPLE_RESUMES = [
    {
        "filename": "john_doe_backend.txt",
        "text": (
            "John Doe - Senior Backend Engineer\n\n"
            "Technical Skills:\n"
            "- Languages: Python, Go, SQL\n"
            "- Frameworks: FastAPI, Flask, Django\n"
            "- Tools & Infrastructure: Docker, Kubernetes, AWS EC2, S3, PostgreSQL, Redis\n"
            "- Concepts: REST APIs, Microservices, System Design, CI/CD\n\n"
            "Experience:\n"
            "Backend Developer at CloudScale (2023 - Present):\n"
            "- Built and deployed scalable REST APIs containerized with Docker on AWS EC2, handling 500+ daily candidate queries.\n"
            "- Designed database schemas in PostgreSQL, improving query response times by 30%.\n"
            "- Built secure user authentication systems and API gateways.\n"
        )
    },
    {
        "filename": "alice_smith_data_scientist.txt",
        "text": (
            "Alice Smith - Machine Learning & NLP Scientist\n\n"
            "Technical Skills:\n"
            "- Languages: Python, R, C++\n"
            "- Libraries: PyTorch, Hugging Face, Transformers, Scikit-learn, Pandas, NumPy\n"
            "- Databases: FAISS, Pinecone, MongoDB\n"
            "- Concepts: Natural Language Processing (NLP), Retrieval-Augmented Generation (RAG), Sentence-BERT, Vector Embeddings\n\n"
            "Experience:\n"
            "AI Research Engineer at ML Labs (2022 - Present):\n"
            "- Built an end-to-end Retrieval-Augmented Generation (RAG) job-matching system using Sentence-BERT embeddings and FAISS vector database.\n"
            "- Increased matching accuracy by 38% over baseline TF-IDF by implementing dense semantic retrievers.\n"
            "- Fine-tuned BERT and T5 models for custom document classification and summary generation.\n"
        )
    },
    {
        "filename": "bob_johnson_frontend.txt",
        "text": (
            "Bob Johnson - Lead Frontend Engineer\n\n"
            "Technical Skills:\n"
            "- Languages: TypeScript, JavaScript, HTML5, CSS3\n"
            "- Frameworks: React, Next.js, Vue.js, Tailwind CSS\n"
            "- Tools: Webpack, Vite, Git, Jest\n"
            "- Concepts: Responsive Design, SPA, State Management (Redux, Zustand), Web Performance Optimization\n\n"
            "Experience:\n"
            "Frontend Architect at WebFlow (2021 - Present):\n"
            "- Rebuilt the core client dashboard using Next.js and Tailwind CSS, improving load times by 40%.\n"
            "- Implemented responsive, pixel-perfect user interfaces, boosting user engagement by 15%.\n"
            "- Managed a team of 4 frontend developers and established TypeScript coding guidelines.\n"
        )
    },
    {
        "filename": "sarah_lee_devops.txt",
        "text": (
            "Sarah Lee - DevOps & Cloud Infrastructure Engineer\n\n"
            "Technical Skills:\n"
            "- Cloud: AWS (EC2, VPC, EKS, RDS, S3, IAM), GCP\n"
            "- Automation/IaC: Terraform, Ansible\n"
            "- CI/CD: GitHub Actions, Jenkins, GitLab CI\n"
            "- Containerization & Orchestration: Docker, Kubernetes, Helm\n"
            "- Monitoring: Prometheus, Grafana, ELK Stack\n\n"
            "Experience:\n"
            "DevOps Engineer at InfraTech (2022 - Present):\n"
            "- Automated multi-region AWS cloud infrastructure deployment using Terraform, reducing setup time by 80%.\n"
            "- Maintained production Kubernetes (EKS) clusters, scaling deployments to handle peak traffic spikes.\n"
            "- Constructed end-to-end CI/CD pipelines, shortening code-to-production cycles to under 10 minutes.\n"
        )
    },
    {
        "filename": "david_kim_pm.txt",
        "text": (
            "David Kim - Technical Product Manager\n\n"
            "Technical Skills:\n"
            "- Methodologies: Agile, Scrum, Kanban, Product Lifecycle Management\n"
            "- Tools: Jira, Confluence, Figma, Mixpanel, Tableau\n"
            "- Analytics: SQL, Google Analytics, Excel\n"
            "- Core Competencies: Roadmap Strategy, User Research, A/B Testing, Stakeholder Management\n\n"
            "Experience:\n"
            "Technical Product Manager at FinTech Corp (2020 - Present):\n"
            "- Owned the product lifecycle for a mobile banking app, growing active user base from 10k to 100k.\n"
            "- Gathered requirements, designed user stories, and collaborated with engineering teams to ship 15+ major features.\n"
            "- Analyzed conversion funnels using SQL and Mixpanel to identify user friction points, improving conversion rate by 12%.\n"
        )
    }
]

# Sample Job Descriptions
SAMPLE_JOB_DESCRIPTIONS = [
    {
        "id": "jd_ml",
        "title": "Machine Learning Engineer (NLP Focus)",
        "text": (
            "We are looking for a Machine Learning Engineer with deep expertise in Natural Language Processing (NLP).\n"
            "Responsibilities include designing semantic search algorithms, building RAG systems, and utilizing vector databases.\n"
            "Required skills: Python, PyTorch, Hugging Face Transformers, Sentence-BERT, FAISS vector search, and model fine-tuning."
        ),
        "expected_match": "alice_smith_data_scientist.txt"
    },
    {
        "id": "jd_backend",
        "title": "Senior Python Backend Developer",
        "text": (
            "Seeking a Senior Backend Engineer to design and maintain high-performance REST APIs.\n"
            "The candidate must be proficient in Python and have hands-on experience building lightweight microservices containerized with Docker.\n"
            "Key tech stack: Python, FastAPI, Docker, AWS EC2, PostgreSQL database, and CI/CD pipelines."
        ),
        "expected_match": "john_doe_backend.txt"
    },
    {
        "id": "jd_frontend",
        "title": "React Frontend Engineer",
        "text": (
            "We are hiring a Frontend Developer specializing in React and modern UI libraries.\n"
            "Must have extensive experience with TypeScript, Next.js, and creating highly interactive web components.\n"
            "Skills: JavaScript, TypeScript, React, Next.js, CSS, HTML5, UI/UX optimization."
        ),
        "expected_match": "bob_johnson_frontend.txt"
    }
]

def write_sample_files(dest_dir: str = "data/resumes"):
    """
    Helper function to write sample resumes to text files.
    """
    os.makedirs(dest_dir, exist_ok=True)
    file_paths = []
    for resume in SAMPLE_RESUMES:
        path = os.path.join(dest_dir, resume["filename"])
        with open(path, "w", encoding="utf-8") as f:
            f.write(resume["text"])
        file_paths.append(path)
    return file_paths
