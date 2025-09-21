# app.py - Enhanced Flask App with Langchain Integration

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
from werkzeug.utils import secure_filename
from resume_analyzer import ResumeAnalyzer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the resume analyzer
try:
    analyzer = ResumeAnalyzer()
    logger.info("Resume analyzer initialized successfully")
    if analyzer.llm:
        logger.info("AI-powered analysis is available")
    else:
        logger.warning("AI-powered analysis is not available - falling back to rule-based analysis")
except Exception as e:
    logger.error(f"Failed to initialize resume analyzer: {e}")
    analyzer = None

def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash('File is too large. Maximum size is 16MB.', 'danger')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {error}")
    flash('An internal server error occurred. Please try again.', 'danger')
    return redirect(url_for('index'))

@app.route('/')
def index():
    """Home page - displays upload form."""
    # Check if AI is available and pass to template
    ai_available = analyzer is not None and analyzer.llm is not None
    return render_template('index.html', ai_available=ai_available)

@app.route('/upload', methods=['POST'])
def upload_resume():
    """Handle resume file upload and job description text."""
    
    # Check if analyzer is available
    if not analyzer:
        flash('Resume analyzer is currently unavailable. Please try again later.', 'danger')
        return redirect(url_for('index'))
    
    # Check if the post request has the file part
    if 'resume' not in request.files:
        flash('No file selected. Please choose a file to upload.', 'warning')
        return redirect(url_for('index'))
    
    file = request.files['resume']
    
    # Check if user actually selected a file
    if file.filename == '':
        flash('No file selected. Please choose a file to upload.', 'warning')
        return redirect(url_for('index'))
    
    # Validate file type and process
    if file and allowed_file(file.filename):
        try:
            # Secure the filename
            filename = secure_filename(file.filename)
            if not filename:
                flash('Invalid filename. Please rename your file and try again.', 'warning')
                return redirect(url_for('index'))
            
            # Save the file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"File saved: {filepath}")
            
            # Get the job description text from the form
            job_description = request.form.get('job_description', '').strip()
            if job_description:
                logger.info("Job description provided for AI-enhanced analysis")
            else:
                logger.info("No job description provided - performing general analysis")
            
            # Analyze the resume
            analysis_result = analyzer.analyze_resume(filepath, job_description)
            
            # Clean up the uploaded file
            try:
                os.remove(filepath)
                logger.info(f"Temporary file cleaned up: {filepath}")
            except Exception as e:
                logger.warning(f"Could not clean up file {filepath}: {e}")
            
            # Check if analysis was successful
            if not analysis_result.get('success', False):
                flash(f"Analysis failed: {analysis_result.get('error', 'Unknown error occurred')}", 'danger')
                return redirect(url_for('index'))
            
            # Add AI availability info to results
            analysis_result['ai_available'] = analyzer.llm is not None
            
            # Render results
            return render_template('results.html', result=analysis_result)
                
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            
            # Clean up file if it exists
            try:
                if 'filepath' in locals() and os.path.exists(filepath):
                    os.remove(filepath)
            except:
                pass
            
            flash(f'An error occurred while processing your file: {str(e)}', 'danger')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload PDF, DOC, DOCX, or TXT files only.', 'warning')
        return redirect(url_for('index'))

@app.route('/health')
def health_check():
    """Enhanced health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'analyzer_available': analyzer is not None,
        'ai_available': analyzer.llm is not None if analyzer else False,
        'huggingface_configured': os.getenv('HUGGINGFACE_API_TOKEN') is not None
    })
# In app.py, add this new route

@app.route('/generate-cover-letter', methods=['POST'])
def generate_cover_letter_route():
    if not analyzer or not analyzer.llm:
        return jsonify({'success': False, 'error': 'AI features are not available.'}), 503

    data = request.get_json()
    resume_text = data.get('resume_text')
    jd_text = data.get('jd_text')

    if not resume_text:
        return jsonify({'success': False, 'error': 'Resume text is required.'}), 400

    try:
        cover_letter = analyzer.generate_cover_letter(resume_text, jd_text)
        if cover_letter:
            return jsonify({'success': True, 'cover_letter': cover_letter})
        else:
            return jsonify({'success': False, 'error': 'Failed to generate cover letter.'}), 500
    except Exception as e:
        logger.error(f"Cover letter generation failed: {e}")
        return jsonify({'success': False, 'error': 'An internal error occurred.'}), 500

# This should be the last part of your file
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)