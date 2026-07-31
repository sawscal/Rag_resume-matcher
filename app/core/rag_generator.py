import os
from google import genai

class RAGExplanationGenerator:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initializes the sequence-to-sequence generator for prompt-based analysis using Gemini API.
        """
        self.model_name = model_name
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = genai.Client()
        return self._client

    def generate_match_analysis(self, resume_text: str, job_description: str) -> str:
        """
        Runs RAG generation to explain the match between a resume and job description.
        """
        # Gemini can handle much larger contexts, but we can still truncate reasonably
        truncated_resume = resume_text[:5000]
        truncated_jd = job_description[:5000]

        prompt = (
            f"You are an AI recruiter. Analyze the fit between the candidate's resume and the job description.\n"
            f"Resume snippet: {truncated_resume}\n"
            f"Job Description snippet: {truncated_jd}\n\n"
            f"Provide a brief evaluation. Address: 1) Matches 2) Skill Gaps 3) Final Recommendation (Fit/No Fit)."
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            return f"Error during generation: {str(e)}"

# Global generator instance
rag_generator = RAGExplanationGenerator()
