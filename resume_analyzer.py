# resume_analyzer.py

import os
import re
from typing import Dict, List, Optional, Tuple
import docx
import PyPDF2
from dotenv import load_dotenv
import logging
from huggingface_hub import InferenceClient
from rag_engine import ResumeRAGEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)


class ResumeAnalyzer:

    def __init__(self):
        self.client = None
        self.llm    = None
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
            'nosql', 'redis', 'elasticsearch', 'graphql', 'flutter', 'xamarin',
            'langchain', 'rag', 'chromadb', 'vector database', 'semantic search',
            'embeddings', 'retrieval augmented generation',
        ]

        self.soft_skills = [
            'leadership', 'communication', 'teamwork', 'problem solving',
            'project management', 'analytical', 'creative', 'organized',
            'time management', 'collaboration', 'adaptable', 'innovative',
            'critical thinking', 'decision making', 'mentoring',
            'strategic planning', 'conflict resolution',
            'emotional intelligence', 'negotiation',
        ]

        self.action_verbs = [
            'developed', 'designed', 'built', 'implemented', 'created', 'led',
            'managed', 'improved', 'increased', 'decreased', 'reduced',
            'achieved', 'delivered', 'launched', 'deployed', 'optimized',
            'automated', 'architected', 'engineered', 'established',
            'streamlined', 'spearheaded', 'collaborated', 'mentored',
            'trained', 'coordinated', 'directed', 'produced', 'resolved',
            'integrated', 'migrated', 'scaled', 'refactored', 'debugged',
        ]

        self.job_profiles = {
            "Software Dev": [
                'python', 'java', 'c++', 'c#', 'javascript', 'html', 'css',
                'git', 'sql', 'docker', 'react', 'angular', 'vue.js',
                'django', 'flask', 'spring boot', '.net',
            ],
            "Data Science": [
                'python', 'r', 'sql', 'machine learning', 'data science',
                'tensorflow', 'pytorch', 'pandas', 'numpy', 'tableau', 'power bi',
            ],
            "Cyber Security": [
                'cyber security', 'networking', 'linux', 'firewall',
                'penetration testing', 'encryption', 'siem', 'python',
            ],
            "Cloud / DevOps": [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
                'ansible', 'linux', 'python', 'git', 'jenkins',
            ],
            "AI / ML Eng": [
                'python', 'machine learning', 'langchain', 'rag', 'llm',
                'tensorflow', 'pytorch', 'hugging face', 'embeddings', 'generative ai',
            ],
        }

        self.resume_sections = [
            'experience', 'education', 'skills', 'projects', 'work experience',
            'employment', 'qualifications', 'achievements', 'certifications',
            'summary', 'objective', 'profile', 'about', 'contact',
            'professional experience', 'publications', 'awards', 'internship',
        ]

        self.education_keywords = [
            'university', 'college', 'degree', 'bachelor', 'master', 'phd',
            'diploma', 'certification', 'course', 'training', 'institute',
            'school', 'graduated', 'gpa', 'cgpa', 'b.tech', 'b.e', 'm.tech', 'mba',
        ]

    # ─── LLM init ─────────────────────────────────────────────────────────────
    def _initialize_llm(self):
        token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not token:
            logging.warning("HUGGINGFACE_API_TOKEN not set — AI features disabled")
            return
        try:
            self.client = InferenceClient(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                token=token
            )
            self.llm = self.client
            logging.info("HuggingFace InferenceClient initialised")
        except Exception as e:
            logging.error(f"LLM init failed: {e}")

    def _llm_call(self, messages: List[Dict], max_tokens: int = 500) -> Optional[str]:
        if not self.client:
            return None
        try:
            resp = self.client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"LLM call error: {e}")
            return None

    # ─── ATS Scoring (realistic 6-dimension rubric, max 100) ──────────────────
    def calculate_score_and_breakdown(self, text: str, skills: Dict) -> Tuple[int, Dict]:
        """
        Scores across 6 dimensions:
          Contact Info  →  10 pts   email, phone, LinkedIn, GitHub
          Content       →  15 pts   word count band, action verbs, quantified results
          Education     →  10 pts   degree, institution, year, GPA
          Experience    →  20 pts   section presence, titles, date ranges, bullets
          Skills        →  25 pts   tech count, soft count, dedicated section
          Structure     →  20 pts   distinct section categories found
        """
        bd = {
            'Contact Info': 0,
            'Content':      0,
            'Education':    0,
            'Experience':   0,
            'Skills':       0,
            'Structure':    0,
        }
        tl = text.lower()
        wc = len(text.split())

        # Contact Info (max 10)
        ci = 0
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text): ci += 4
        if re.search(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text): ci += 3
        if 'linkedin.com' in tl: ci += 2
        if 'github.com'   in tl: ci += 1
        bd['Contact Info'] = min(10, ci)

        # Content quality (max 15)
        cq = 0
        if wc >= 150: cq += 2
        if wc >= 300: cq += 2
        if wc >= 450: cq += 1
        if wc > 950:  cq -= 2   # too verbose penalty
        quant = len(re.findall(
            r'\b\d+\s*(%|percent|users?|clients?|projects?|members?|'
            r'hours?|months?|years?|times?|\bx\b)', tl))
        cq += min(5, quant)
        verb_hits = sum(1 for v in self.action_verbs if v in tl)
        cq += min(5, verb_hits)
        bd['Content'] = min(15, max(0, cq))

        # Education (max 10)
        ed = 0
        if any(k in tl for k in ['university','college','institute','school']): ed += 3
        if any(k in tl for k in ['bachelor','b.tech','b.e','master','m.tech','phd','diploma']): ed += 3
        if re.search(r'\b(19|20)\d{2}\b', text): ed += 2
        if any(k in tl for k in ['gpa','cgpa','percentage','grade']): ed += 2
        bd['Education'] = min(10, ed)

        # Experience (max 20)
        ex = 0
        if any(k in tl for k in ['experience','work experience','professional experience']): ex += 4
        if any(k in tl for k in ['intern','internship','trainee']): ex += 3
        if any(k in tl for k in ['engineer','developer','analyst','consultant','manager','lead']): ex += 3
        if any(k in tl for k in ['project','projects']): ex += 3
        date_hits = len(re.findall(
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,.-]+20\d{2}', tl))
        ex += min(4, date_hits * 2)
        bullet_count = len(re.findall(r'^\s*[•\-\*]', text, re.MULTILINE))
        ex += min(3, bullet_count // 2)
        bd['Experience'] = min(20, ex)

        # Skills (max 25)
        sk = 0
        sk += min(14, len(skills['technical']))
        sk += min(5,  len(skills['soft']))
        if any(k in tl for k in ['skills','technical skills','core competencies','technologies']): sk += 4
        if len(skills['soft']) == 0: sk -= 4   # penalty for zero soft skills
        bd['Skills'] = min(25, max(0, sk))

        # Structure (max 20)
        cats = set()
        mapping = {
            'summary': ['summary','objective','profile','about'],
            'contact': ['contact'],
            'education': ['education'],
            'experience': ['experience','work experience','professional experience','employment'],
            'projects': ['projects'],
            'skills': ['skills','technical skills','core competencies'],
            'achievements': ['achievements','certifications','awards','publications'],
            'internship': ['internship'],
        }
        for cat, variants in mapping.items():
            if any(v in tl for v in variants):
                cats.add(cat)
        st = len(cats) * 3
        if re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', text[:200]): st += 2  # name detected
        if len(cats) < 3: st -= 4
        bd['Structure'] = min(20, max(0, st))

        return min(100, max(0, sum(bd.values()))), bd

    # ─── AI feedback ─────────────────────────────────────────────────────────
    def generate_ai_feedback(self, text: str, skills: Dict, score: int) -> str:
        msg = [
            {"role": "system", "content": (
                "You are a professional resume coach. "
                "Give exactly 5 specific, actionable improvement tips. "
                "Each tip must start with '•'. "
                "Focus on weakest areas: missing quantified results, "
                "weak action verbs, missing sections, or thin skills."
            )},
            {"role": "user", "content": (
                f"ATS score: {score}/100.\n\nResume:\n---\n{text[:1500]}\n---\n\n"
                "Give 5 concrete improvement tips."
            )},
        ]
        feedback = self._llm_call(msg, max_tokens=400)
        if feedback:
            lines = [l.strip() for l in feedback.split('\n') if l.strip()]
            return '\n'.join(f"• {l.lstrip('•*- ')}" for l in lines[:5])
        return self._fallback_feedback(skills, score)

    def _fallback_feedback(self, skills: Dict, score: int) -> str:
        tips = []
        if len(skills['technical']) < 7:
            tips.append("• Add more technical skills — aim for at least 7.")
        if len(skills['soft']) < 3:
            tips.append("• Include soft skills: 'leadership', 'teamwork', 'problem solving'.")
        if score < 60:
            tips.append("• Use strong action verbs: 'Developed', 'Built', 'Led', 'Improved'.")
        if score < 75:
            tips.append("• Quantify achievements — add numbers, percentages, or scale.")
        tips.append("• Ensure clear sections: Summary, Experience, Skills, Education.")
        return '\n'.join(tips[:5])

    # ─── Job comparison ───────────────────────────────────────────────────────
    def ai_enhanced_job_comparison(
        self, resume_text: str, jd_text: str, resume_skills: List[str]
    ) -> Optional[Dict]:
        if not jd_text or not jd_text.strip():
            return None
        jd_skills = self.extract_skills(jd_text)['technical']
        matching  = sorted(set(resume_skills) & set(jd_skills))
        missing   = sorted(set(jd_skills) - set(resume_skills))
        match_pct = int(len(matching) / len(jd_skills) * 100) if jd_skills else 0
        result = {
            'match_score':      match_pct,
            'matching_skills':  matching,
            'missing_skills':   missing,
            'ai_insights':      "AI insights unavailable.",
            'jd_text':          jd_text,
        }
        if self.client:
            ai = self._llm_call([
                {"role": "system", "content": "Career advisor. Give 3 concise insights: (1) key strengths, (2) critical gaps, (3) one actionable tip."},
                {"role": "user", "content": f"Resume:\n{resume_text[:800]}\n\nJD:\n{jd_text[:800]}"},
            ], max_tokens=300)
            if ai:
                result['ai_insights'] = ai
        return result

    # ─── Bullet enhancement ───────────────────────────────────────────────────
    def enhance_bullet_points(self, text: str) -> list:
        bullets = re.findall(r'^\s*[\*•-]\s*(.*)', text, re.MULTILINE)
        if not self.client or not bullets:
            return []
        suggestions = []
        for bullet in bullets[:5]:
            if len(bullet.split()) < 5:
                continue
            enhanced = self._llm_call([
                {"role": "system", "content": "Rewrite this resume bullet: stronger verb, quantified result, under 25 words."},
                {"role": "user", "content": f'Rewrite: "{bullet}"'},
            ], max_tokens=100)
            if enhanced and enhanced.lower() != bullet.lower():
                suggestions.append({'original': bullet, 'suggestion': enhanced.strip('*- ')})
        return suggestions

    # ─── Cover letter ─────────────────────────────────────────────────────────
    def generate_cover_letter(self, resume_text: str, jd_text: Optional[str]) -> Optional[str]:
        if not self.client:
            return None
        prompt = (
            "Write a professional cover letter. Highlight 2-3 key skills. "
            "Structure: introduction, body, conclusion.\n\n"
            f"RESUME:\n{resume_text[:2000]}\n\n"
            f"JOB DESCRIPTION:\n{jd_text[:1500] if jd_text else '(None — write general cover letter.)'}"
        )
        return self._llm_call([
            {"role": "system", "content": "World-class career coach who writes compelling cover letters."},
            {"role": "user",   "content": prompt},
        ], max_tokens=700)

    # ─── Text extraction ──────────────────────────────────────────────────────
    def extract_text_from_pdf(self, path: str) -> str:
        try:
            t = ""
            with open(path, 'rb') as f:
                for page in PyPDF2.PdfReader(f).pages:
                    t += (page.extract_text() or "") + "\n"
            return t.strip()
        except Exception as e:
            logging.error(f"PDF error: {e}"); return ""

    def extract_text_from_docx(self, path: str) -> str:
        try:
            return "\n".join(p.text for p in docx.Document(path).paragraphs).strip()
        except Exception as e:
            logging.error(f"DOCX error: {e}"); return ""

    def extract_text_from_txt(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read().strip()
        except Exception as e:
            logging.error(f"TXT error: {e}"); return ""

    def extract_text(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.pdf':             return self.extract_text_from_pdf(path)
        if ext in ('.docx', '.doc'): return self.extract_text_from_docx(path)
        if ext == '.txt':             return self.extract_text_from_txt(path)
        return ""

    def is_resume(self, text: str) -> bool:
        tl   = text.lower()
        secs = sum(1 for s in self.resume_sections if s in tl)
        email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text))
        phone = bool(re.search(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text))
        return secs >= 2 and (email or phone) and len(text.split()) > 50

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        tl   = text.lower()
        tech = sorted({s for s in self.technical_skills if re.search(r'\b' + re.escape(s) + r'\b', tl)})
        soft = sorted({s for s in self.soft_skills     if re.search(r'\b' + re.escape(s) + r'\b', tl)})
        return {'technical': tech, 'soft': soft}

    def calculate_job_profile_match(self, skills: List[str]) -> Dict[str, int]:
        return {
            p: min(100, int(len([s for s in skills if s in kws]) / len(kws) * 100))
            for p, kws in self.job_profiles.items()
        }

    # ─── Main entry point ─────────────────────────────────────────────────────
    def analyze_resume(
        self, file_path: str, job_description_text: Optional[str] = None
    ) -> Dict:
        try:
            text = self.extract_text(file_path)
            if not text:
                return {'success': False, 'error': 'Could not extract text from the file.'}
            if not self.is_resume(text):
                return {'success': False, 'error': 'The uploaded file does not appear to be a resume.'}

            skills           = self.extract_skills(text)
            score, breakdown = self.calculate_score_and_breakdown(text, skills)
            profile_matches  = self.calculate_job_profile_match(skills['technical'])
            job_comparison   = self.ai_enhanced_job_comparison(text, job_description_text, skills['technical'])
            ai_feedback      = self.generate_ai_feedback(text, skills, score)
            enhanced_bullets = self.enhance_bullet_points(text)

            # ── RAG Pipeline ─────────────────────────────────────────────────
            rag_insights = {"rag_available": False}
            hf_token     = os.getenv("HUGGINGFACE_API_TOKEN", "")

            if hf_token:
                try:
                    rag = ResumeRAGEngine(hf_api_token=hf_token)

                    # Check embeddings via .ready property (STEmbeddings always
                    # exists as an object, but may have failed to load the model)
                    if not rag._embeddings.ready:
                        logging.error(
                            "RAG SKIPPED: sentence-transformers embeddings failed to load.\n"
                            "  Fix: pip install sentence-transformers torch"
                        )
                    elif not rag.build_vectorstore(text, job_description_text):
                        logging.error("RAG SKIPPED: build_vectorstore() returned False")
                    else:
                        exp_fb = rag.get_targeted_feedback("work experience and projects")
                        ski_fb = rag.get_targeted_feedback("technical skills")
                        jd_sem = (
                            rag.get_semantic_jd_match_insights()
                            if job_description_text else None
                        )
                        rag_insights = {
                            "rag_available":       True,
                            "experience_feedback": exp_fb,
                            "skills_feedback":     ski_fb,
                            "jd_semantic_match":   jd_sem,
                        }
                        logging.info("RAG: all insights generated successfully")

                except Exception as e:
                    logging.error(f"RAG pipeline error: {e}", exc_info=True)
            else:
                logging.warning("RAG SKIPPED: HUGGINGFACE_API_TOKEN not set")
            # ─────────────────────────────────────────────────────────────────

            return {
                'success':             True,
                'filename':            os.path.basename(file_path),
                'score':               score,
                'skills':              skills,
                'score_breakdown':     breakdown,
                'job_profile_matches': profile_matches,
                'job_comparison':      job_comparison,
                'ai_feedback':         ai_feedback,
                'enhanced_bullets':    enhanced_bullets,
                'full_text':           text,
                'ai_powered':          self.client is not None,
                'rag_insights':        rag_insights,
            }

        except Exception as e:
            logging.error(f"analyze_resume error: {e}", exc_info=True)
            return {'success': False, 'error': 'An unexpected error occurred during analysis.'}