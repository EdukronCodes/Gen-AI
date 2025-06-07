# Intelligent Resume Optimization Tool

A Flask-based web application for AI-powered resume analysis and optimization using classic HTML, CSS, and JavaScript frontend.

## Features
- Upload resume (PDF/CSV)
- View resume scoring and improvement suggestions
- ATS optimization
- Simple web UI (HTML/CSS/JS)

## Project Structure
```
intelligent_resume_optimization/
├── app.py
├── requirements.txt
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── templates/
│   ├── index.html
│   ├── scoring.html
│   └── suggestions.html
└── README.md
```

## Setup
1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open your browser at [http://localhost:5000](http://localhost:5000)

## Requirements
- Python 3.8+
- Flask
- scikit-learn
- pandas
- matplotlib

## License
MIT 