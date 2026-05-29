# Resume.AI 🚀

An AI-powered Resume Analysis Platform built using **Flask, LangChain, RAG, ChromaDB, and Mistral-7B** that helps users evaluate resumes, match them against job descriptions, generate cover letters, and interact with their resumes through natural language conversations.

---

## 📌 Features

### 📄 Resume Analysis
- Upload resumes in:
  - PDF
  - DOCX
  - TXT
- Automatic text extraction
- Resume structure evaluation
- ATS-style scoring

### 🎯 Job Description Matching
- Compare resume against a Job Description
- Identify:
  - Matching skills
  - Missing skills
  - Skill gap percentage
- Generate match score

### 🤖 AI-Powered Feedback
Using Mistral-7B through Hugging Face Inference API:

- Resume improvement suggestions
- ATS optimization recommendations
- Personalized feedback

### ✍️ Cover Letter Generation
Generate professional cover letters automatically based on:

- Resume content
- Job Description

### 🔍 RAG-Based Resume Chatbot
Ask questions directly about your resume:

Examples:

- "What are my strongest technical skills?"
- "Which projects match a Data Scientist role?"
- "Summarize my experience."
- "What skills are missing for this job?"

Implemented using:

- LangChain
- ChromaDB
- Semantic Retrieval
- Hugging Face LLM

### 🧠 Semantic Search
The system retrieves the most relevant resume sections before sending context to the LLM, improving answer quality and reducing hallucinations.

---

## 🏗️ Tech Stack

### Backend
- Flask
- Python

### AI / LLM
- Mistral-7B-Instruct
- Hugging Face Inference API

### RAG Pipeline
- LangChain
- ChromaDB
- Sentence Transformers
- Custom TF-IDF Fallback Embeddings

### Document Processing
- PyPDF2
- python-docx

### Frontend
- HTML
- CSS
- Jinja Templates

---
