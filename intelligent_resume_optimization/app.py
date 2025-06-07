from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory storage for demo
resume_data = None
scoring_result = None
model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global resume_data
    file = request.files.get('file')
    if not file:
        flash('No file uploaded')
        return redirect(url_for('index'))
    df = pd.read_csv(file)
    resume_data = df
    flash('File uploaded successfully!')
    return redirect(url_for('scoring'))

@app.route('/scoring')
def scoring():
    global resume_data, scoring_result
    if resume_data is None:
        flash('Please upload resume data first.')
        return redirect(url_for('index'))
    # Simple scoring logic (dummy)
    scoring_result = resume_data.head(5)
    return render_template('scoring.html', scoring=scoring_result.to_html())

@app.route('/suggestions', methods=['GET', 'POST'])
def suggestions():
    global resume_data, model
    if request.method == 'POST':
        if resume_data is None:
            flash('Please upload resume data first.')
            return redirect(url_for('index'))
        # Train a simple model for demo
        if model is None:
            X = resume_data.select_dtypes(include=['float64', 'int64'])
            y = (X.sum(axis=1) > X.sum(axis=1).mean()).astype(int)  # Dummy target
            model = RandomForestClassifier().fit(X, y)
        # Get input from form
        try:
            values = [float(request.form.get(f'feature{i}')) for i in range(len(resume_data.columns)-1)]
            pred = model.predict([values])[0]
            flash(f'Suggestion: {"Optimize" if pred == 1 else "No Optimization Needed"}')
        except Exception as e:
            flash(f'Error in suggestion: {e}')
    return render_template('suggestions.html')

if __name__ == '__main__':
    app.run(debug=True) 