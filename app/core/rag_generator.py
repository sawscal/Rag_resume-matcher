import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

class RAGExplanationGenerator:
    def __init__(self, model_name: str = "google/flan-t5-small"):
        """
        Initializes the sequence-to-sequence generator for prompt-based analysis.
        Uses a lightweight Hugging Face model for fast execution.
        """
        self.model_name = model_name
        self._generator = None

    @property
    def tokenizer(self):
        if not hasattr(self, "_tokenizer") or self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @property
    def model(self):
        if not hasattr(self, "_model") or self._model is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        return self._model

    def generate_match_analysis(self, resume_text: str, job_description: str) -> str:
        """
        Runs RAG generation to explain the match between a resume and job description.
        """
        # Truncate text inputs to avoid exceeding token limit of the small model (usually 512 or 1024 tokens)
        truncated_resume = resume_text[:1200]
        truncated_jd = job_description[:800]

        prompt = (
            f"You are an AI recruiter. Analyze the fit between the candidate's resume and the job description.\n"
            f"Resume snippet: {truncated_resume}\n"
            f"Job Description snippet: {truncated_jd}\n\n"
            f"Provide a brief evaluation. Address: 1) Matches 2) Skill Gaps 3) Final Recommendation (Fit/No Fit)."
        )

        try:
            tok = self.tokenizer
            mod = self.model
            
            inputs = tok(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = mod.generate(**inputs, max_length=512)
            
            generated_text = tok.decode(outputs[0], skip_special_tokens=True)
            return generated_text.strip()
        except Exception as e:
            return f"Error during generation: {str(e)}"

# Global generator instance
rag_generator = RAGExplanationGenerator()
