# resume_analyzer.py - Updated with Cover Letter Generation

import os
import re
from typing import Dict, List, Optional, Tuple
import docx
import PyPDF2
from dotenv import load_dotenv
import logging

# Hugging Face Inference API for conversational models
from huggingface_hub import InferenceClient

# Load environment variables from .env file
load_dotenv()

class ResumeAnalyzer:
    def __init__(self):
        self.client = None
        self.llm = None
        self._initialize_llm()
        
        self.technical_skills = [
            'python', 'java', 'javascript', 'react', 'node.js', 'html', 'css', 
            'sql', 'mongodb', 'mysql', 'postgresql', 'git', 'docker', 'aws', 
            'azure', 'kubernetes', 'linux', 'windows', 'machine learning', 
            'data science', 'artificial intelligence', 'flask', 'django', 
            'spring boot', 'angular', 'vue.js', 'tensorflow', 'pytorch',
            'c++', 'c#', '.net', 'php', 'ruby', 'golang', 'scala', 'kotlin',
            'swift', 'rust', 'power bi', 'tableau', 'figma', 'generative ai', 
            'llm', 'hugging face', 'api', 'rest', 'microservices', 'devops',
            'jenkins', 'terraform', 'ansible', 'spark', 'hadoop', 'kafka',
            'nosql', 'redis', 'elasticsearch', 'graphql', 'flutter', 'xamarin'
        ]
        self.soft_skills = [
            'leadership', 'communication', 'teamwork', 'problem solving',
            'project management', 'analytical', 'creative', 'organized',
            'time management', 'collaboration', 'adaptable', 'innovative',
            'critical thinking', 'decision making', 'mentoring', 'strategic planning',
            'conflict resolution', 'emotional intelligence', 'negotiation'
        ]
        self.job_profiles = {
            "Software Development": ['python', 'java', 'c++', 'c#', 'javascript', 'html', 'css', 'git', 'sql', 'docker', 'react', 'angular', 'vue.js', 'django', 'flask', 'spring boot', '.net'],
            "Data Science": ['python', 'r', 'sql', 'machine learning', 'data science', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'tableau', 'power bi'],
            "Cyber Security": ['cyber security', 'networking', 'linux', 'firewall', 'penetration testing', 'encryption', 'siem', 'python'],
            "Cloud Engineering / DevOps": ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible', 'ci/cd', 'linux', 'python', 'git', 'jenkins']
        }
        self.resume_sections = ['experience', 'education', 'skills', 'projects', 'work experience', 'employment', 'qualifications', 'achievements', 'certifications', 'summary', 'objective', 'profile', 'about', 'contact', 'professional experience']
        self.education_keywords = ['university', 'college', 'degree', 'bachelor', 'master', 'phd', 'diploma', 'certification', 'course', 'training', 'institute', 'school', 'graduated', 'gpa', 'cgpa']

    def _initialize_llm(self):
        try:
            huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
            if not huggingface_api_token:
                logging.warning("HUGGINGFACE_API_TOKEN not found. AI features will be disabled.")
                return
            
            self.client = InferenceClient(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                token=huggingface_api_token
            )
            self.llm = self.client
            logging.info("Hugging Face conversational client initialized successfully.")
            
        except Exception as e:
            logging.error(f"Failed to initialize Hugging Face client: {e}")
            self.client = self.llm = None

    def _have_conversation_with_llm(self, messages: List[Dict[str, str]], max_tokens: int = 500) -> Optional[str]:
        if not self.client:
            logging.warning("LLM client not available.")
            return None
        
        try:
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"Error during LLM conversation: {e}")
            return None

    def generate_ai_feedback(self, text: str, skills: Dict, score: int) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful and professional resume coach. Your task is to provide 5 specific, actionable tips to improve a resume based on its content and score. Each tip must start with a bullet point."},
            {"role": "user", "content": f"Please analyze this resume and give me 5 actionable improvements. The resume has an ATS score of {score}/100. Here is the resume text:\n\n---\n{text[:1500]}\n---"}
        ]
        feedback = self._have_conversation_with_llm(messages, max_tokens=400)
        if feedback:
            lines = [line.strip() for line in feedback.split('\n') if line.strip()]
            formatted_lines = [f"• {line.lstrip('*- ')}" for line in lines]
            return '\n'.join(formatted_lines[:5])
        return self._generate_fallback_feedback(skills, score)

    def ai_enhanced_job_comparison(self, resume_text: str, jd_text: str, resume_skills: List[str]) -> Optional[Dict]:
        if not jd_text or not jd_text.strip():
            return None
            
        jd_skills = self.extract_skills(jd_text)['technical']
        matching_skills = sorted(list(set(resume_skills) & set(jd_skills)))
        missing_skills = sorted(list(set(jd_skills) - set(resume_skills)))
        match_score = int((len(matching_skills) / len(jd_skills)) * 100) if jd_skills else 0
        
        result = {
            'match_score': match_score,
            'matching_skills': matching_skills,
            'missing_skills': missing_skills,
            'ai_insights': "AI insights are unavailable at this moment.",
            'jd_text': jd_text # Pass the JD text back for later use
        }
        
        if self.client:
            messages = [
                {"role": "system", "content": "You are a career advisor. You will receive a resume and a job description. Your job is to provide 3 concise insights: 1. Key strengths for the role. 2. Critical gaps to address. 3. One actionable tip to improve the candidate's fit for this specific job."},
                {"role": "user", "content": f"Compare this resume snippet with the job description.\n\n**Resume:**\n{resume_text[:800]}\n\n**Job Description:**\n{jd_text[:800]}\n\nBased on this, what are the key strengths, critical gaps, and one actionable tip?"}
            ]
            
            ai_insights = self._have_conversation_with_llm(messages, max_tokens=300)
            if ai_insights:
                result['ai_insights'] = ai_insights
        
        return result

    def enhance_bullet_points(self, text: str) -> list:
        suggestions = []
        bullet_points = re.findall(r'^\s*[\*•-]\s*(.*)', text, re.MULTILINE)
        
        if not self.client or not bullet_points:
            return []

        for bullet in bullet_points[:5]:
            if len(bullet.split()) < 5:
                continue
            messages = [
                {"role": "system", "content": "You are an expert resume editor. Your task is to rewrite a single resume bullet point to be more impactful. Use a strong action verb, focus on quantifiable results, and keep it concise (under 25 words)."},
                {"role": "user", "content": f"Rewrite this bullet point: \"{bullet}\""}
            ]
            enhanced_bullet = self._have_conversation_with_llm(messages, max_tokens=100)
            if enhanced_bullet and enhanced_bullet.lower() != bullet.lower():
                suggestions.append({'original': bullet, 'suggestion': enhanced_bullet.strip('*- ')})
        return suggestions

    # --- NEW METHOD FOR COVER LETTER GENERATION ---
    def generate_cover_letter(self, resume_text: str, jd_text: Optional[str]) -> Optional[str]:
        """Generates a draft cover letter using the LLM."""
        if not self.client:
            return None

        prompt = (
            f"Based on the following resume and job description, write a professional and compelling cover letter. "
            f"The tone should be enthusiastic but formal. The letter should highlight 2-3 key skills or experiences from the resume "
            f"that directly align with the requirements in the job description. Structure it with a clear introduction, body, and conclusion.\n\n"
            f"--- RESUME ---\n{resume_text[:2000]}\n\n"
        )
        if jd_text:
            prompt += f"--- JOB DESCRIPTION ---\n{jd_text[:1500]}\n\n"
        else:
            prompt += "--- JOB DESCRIPTION ---\n(No job description provided. Write a general-purpose cover letter for a role in the candidate's field.)\n\n"
        prompt += "Generate the cover letter now."

        messages = [
            {"role": "system", "content": "You are a world-class career coach specializing in writing persuasive cover letters."},
            {"role": "user", "content": prompt}
        ]
        return self._have_conversation_with_llm(messages, max_tokens=700)

    def extract_text_from_pdf(self, file_path: str) -> str:
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logging.error(f"Error extracting PDF: {e}")
            return ""

    def extract_text_from_docx(self, file_path: str) -> str:
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs]).strip()
        except Exception as e:
            logging.error(f"Error extracting DOCX: {e}")
            return ""

    def extract_text_from_txt(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read().strip()
        except Exception as e:
            logging.error(f"Error extracting TXT: {e}")
            return ""

    def extract_text(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf': return self.extract_text_from_pdf(file_path)
        elif ext in ['.docx', '.doc']: return self.extract_text_from_docx(file_path)
        elif ext == '.txt': return self.extract_text_from_txt(file_path)
        return ""

    def is_resume(self, text: str) -> bool:
        text_lower = text.lower()
        section_count = sum(1 for section in self.resume_sections if section in text_lower)
        has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        has_phone = bool(re.search(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text))
        word_count = len(text.split())
        return section_count >= 2 and (has_email or has_phone) and word_count > 50

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        text_lower = text.lower()
        found_technical = {skill for skill in self.technical_skills if re.search(r'\b' + re.escape(skill) + r'\b', text_lower)}
        found_soft = {skill for skill in self.soft_skills if re.search(r'\b' + re.escape(skill) + r'\b', text_lower)}
        return {'technical': sorted(list(found_technical)), 'soft': sorted(list(found_soft))}
    
    def _generate_fallback_feedback(self, skills: Dict, score: int) -> str:
        feedback = []
        if len(skills['technical']) < 5: feedback.append("• Add more relevant technical skills from job descriptions.")
        if len(skills['soft']) < 3: feedback.append("• Include important soft skills like leadership or teamwork.")
        if score < 70: feedback.append("• Use action verbs (e.g., 'Developed,' 'Managed') and quantify achievements with numbers.")
        generic_tips = ["• Proofread carefully for any spelling or grammar errors.", "• Keep your resume concise and easy to read, ideally one page."]
        for tip in generic_tips:
            if len(feedback) < 5: feedback.append(tip)
        return '\n'.join(feedback)

    def calculate_score_and_breakdown(self, text: str, skills: Dict) -> Tuple[int, Dict]:
        score_breakdown = {'Contact Info': 0, 'Content': 0, 'Education': 0, 'Experience': 0, 'Skills': 0, 'Structure': 0}
        text_lower = text.lower()
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text): score_breakdown['Contact Info'] += 5
        if re.search(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text): score_breakdown['Contact Info'] += 5
        score_breakdown['Content'] = min(15, int(len(text.split()) / 30))
        if any(keyword in text_lower for keyword in self.education_keywords): score_breakdown['Education'] = 10
        if any(keyword in text_lower for keyword in ['experience', 'work', 'employment']): score_breakdown['Experience'] = 20
        score_breakdown['Skills'] = min(25, (len(skills['technical']) + len(skills['soft'])) * 2)
        score_breakdown['Structure'] = min(20, sum(1 for section in self.resume_sections if section in text_lower) * 4)
        total_score = sum(score_breakdown.values())
        return min(100, total_score), score_breakdown

    def calculate_job_profile_match(self, found_skills: List[str]) -> Dict[str, int]:
        matches = {}
        for profile, keywords in self.job_profiles.items():
            matched_keywords = [skill for skill in found_skills if skill in keywords]
            score = (len(matched_keywords) / len(keywords)) * 100 if keywords else 0
            matches[profile] = min(100, int(score))
        return matches

    def analyze_resume(self, file_path: str, job_description_text: Optional[str] = None) -> Dict:
        try:
            text = self.extract_text(file_path)
            if not text: return {'success': False, 'error': 'Could not extract text from the file.'}
            if not self.is_resume(text): return {'success': False, 'error': 'The uploaded file does not appear to be a valid resume.'}
            
            skills = self.extract_skills(text)
            score, score_breakdown = self.calculate_score_and_breakdown(text, skills)
            job_profile_matches = self.calculate_job_profile_match(skills['technical'])
            job_comparison = self.ai_enhanced_job_comparison(text, job_description_text, skills['technical'])
            ai_feedback = self.generate_ai_feedback(text, skills, score)
            enhanced_bullets = self.enhance_bullet_points(text)

            return {
                'success': True, 
                'filename': os.path.basename(file_path), 
                'score': score, 
                'skills': skills, 
                'score_breakdown': score_breakdown, 
                'job_profile_matches': job_profile_matches, 
                'job_comparison': job_comparison, 
                'ai_feedback': ai_feedback, 
                'enhanced_bullets': enhanced_bullets,
                'full_text': text, # Pass the full text for the cover letter generator
                'ai_powered': self.client is not None
            }
        except Exception as e:
            logging.error(f"Error during resume analysis: {str(e)}")
            return {'success': False, 'error': 'An unexpected error occurred during analysis.'}