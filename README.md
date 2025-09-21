# Resume.AI
An intelligent web application designed to help job seekers optimize their resumes, get actionable feedback, and automatically generate tailored cover letters using the power of Large Language Models.


## About The Project
In today's competitive job market, getting past the initial screening by Applicant Tracking Systems (ATS) is the biggest hurdle for applicants. Resume.AI was built to solve this problem by providing a comprehensive suite of tools to analyze, score, and improve career documents.

This project leverages the Hugging Face Inference API to provide intelligent, context-aware feedback, going far beyond simple keyword matching. It acts as a personal career assistant, helping users craft application materials that stand out to both bots and humans.

## Key Features
📄 Multi-Format Resume Parsing: Supports .pdf, .docx, and .txt file formats.

💯 Instant ATS Score: Calculates an overall score out of 100 based on key metrics like contact information, skills, experience, and structure.

🤖 AI-Powered Writing Coach: Provides specific, actionable suggestions to improve the grammar, style, and impact of your resume's content.

🎯 Job Description Match Analysis: Compares your resume against a job description to identify matching and missing keywords, with a compatibility score.

✍️ Automated Cover Letter Generation: Creates a professional, well-written cover letter tailored to your resume and the provided job description in seconds.

📊 Interactive Data Visualizations: Displays analysis results using dynamic charts (donut, radar, and bar charts) for an engaging user experience.

🖨️ Print & Save Results: Allows users to easily print or save their detailed analysis report as a PDF.

✨ Polished User Experience: Features a modern UI, a drag-and-drop file uploader, and loading spinners for a smooth and professional feel.

## Tech Stack
This project was built using a modern and robust set of technologies:

Backend: Python with the Flask web framework.

AI & Machine Learning: Hugging Face Inference API to interact with the mistralai/Mistral-7B-Instruct-v0.2 model.

Frontend: HTML5, CSS3, and JavaScript, styled with Bootstrap 5.

Data Visualization: Chart.js for interactive and responsive charts.

File Parsing:

PyPDF2 for PDF files.

python-docx for DOCX files.
