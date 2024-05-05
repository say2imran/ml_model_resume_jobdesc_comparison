# About
It starts flask web instance.
'/upload' endpoint can be used to upload the resume/CV in either .pdf or .docx format

Start service:
```commandline
python3 model_resume.py
```

Run below command to test:
```
curl -F "resume_files=@/path/to/your_resume.pdf" -F "job_description_url=http://example.com/job_description" http://localhost:5000/upload
```
