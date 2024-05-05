from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
import numpy as np
import pandas as pd
import PyPDF2  # For PDF processing
import docx  # For DOCX processing
import requests
from bs4 import BeautifulSoup
import os
from tempfile import NamedTemporaryFile
from werkzeug.utils import secure_filename
from flask import Flask, request

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_file):
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ''
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extractText()
        return text

# Function to extract text from DOCX files
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Function to extract job description from a webpage URL
def extract_job_description_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Assuming the job description is contained within a specific HTML tag, e.g., <div class="job-description">
    job_description = soup.find('div', class_='job-description').get_text()
    return job_description

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    resume_files = request.files.getlist('resume_files')
    job_description_url = request.form['job_description_url']
    job_description = extract_job_description_from_url(job_description_url)

    resumes = []
    for resume_file in resume_files:
        filename = secure_filename(resume_file.filename)
        temp_file = NamedTemporaryFile(delete=False)
        temp_file_path = temp_file.name
        resume_file.save(temp_file_path)
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(temp_file_path)
        elif filename.endswith('.docx'):
            text = extract_text_from_docx(temp_file_path)
        else:
            raise ValueError("Unsupported file format")
        resumes.append(text)
        os.unlink(temp_file_path)

    # Feature extraction using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X = tfidf_vectorizer.fit_transform(resumes + [job_description])
    y = [1] * len(resumes) + [0]  # Assuming all resumes match the job description

    # Split data into train and test sets (not needed if you're just predicting without training)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training (not needed if you're just predicting without training)
    # model = LogisticRegression()
    # model.fit(X_train, y_train)
    model = LogisticRegression()
    y_pred = cross_val_predict(model, X, y, cv=5)  # 5-fold cross-validation
    accuracy = np.mean(y_pred == y)  # Mean accuracy across folds

    # Calculate matching percentage
    matching_percentage = (sum(y_pred == 1) / len(y_pred)) * 100

    print("Resume Matching percentage:", matching_percentage)

    # Just returning a placeholder result for demonstration
    return "Resumes and job description processed."


if __name__ == '__main__':
    app.run(debug=True)
