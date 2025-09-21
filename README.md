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


Of course. A great README is essential for showcasing your work on GitHub. Here is a comprehensive and professional README.md file for your project.

You can copy and paste the entire content below into a new file named README.md in your project's root directory.

Resume.AI - Your AI Application Co-Pilot
  ![Hugging Face](https://img.shields.io/badge/%F0%9F%A4% hugging%20face-Inference%20API-yellow)

An intelligent web application designed to help job seekers optimize their resumes, get actionable feedback, and automatically generate tailored cover letters using the power of Large Language Models.

## Project Demo
This demo showcases the complete user flow, from uploading a resume to generating an AI-powered cover letter.

(Note: To create a GIF like this, you can use a free online tool like ezgif.com to convert the Resume analyzer.mov video you created into a GIF. Then, upload the GIF to your GitHub repository and replace the placeholder URL above.)

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

## Setup and Installation
To get a local copy up and running, follow these simple steps.

### Prerequisites
Make sure you have Python 3.9+ installed on your system.

### Installation
Clone the repository:
Bash

git clone https://github.com/your_username/Resume.AI.git
cd Resume.AI
Create and activate a virtual environment:

Bash

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
Create a requirements.txt file:
If you don't have one, you can create it by running this command in your activated virtual environment after installing the necessary packages:

Bash

pip freeze > requirements.txt
(Necessary packages include: Flask, python-dotenv, werkzeug, huggingface_hub, pypdf2, python-docx)

Install the dependencies:

Bash

pip install -r requirements.txt
Set up environment variables:

Create a new file named .env in the root directory of the project.

Open the .env file and add your Hugging Face API token:

HUGGINGFACE_API_TOKEN="hf_YourApiTokenHere"
You will also need a secret key for Flask sessions:

SECRET_KEY="your-super-secret-key"
Run the application:

Bash

flask run
The application will be available at http://127.0.0.1:5000.

File Parsing:

PyPDF2 for PDF files.

python-docx for DOCX files.
