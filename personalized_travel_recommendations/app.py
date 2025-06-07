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
travel_data = None
recommendations = None
model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global travel_data
    file = request.files.get('file')
    if not file:
        flash('No file uploaded')
        return redirect(url_for('index'))
    df = pd.read_csv(file)
    travel_data = df
    flash('File uploaded successfully!')
    return redirect(url_for('recommendations'))

@app.route('/recommendations')
def recommendations():
    global travel_data, recommendations
    if travel_data is None:
        flash('Please upload travel data first.')
        return redirect(url_for('index'))
    # Simple recommendation logic (dummy)
    recommendations = travel_data.head(5)
    return render_template('recommendations.html', recommendations=recommendations.to_html())

@app.route('/marketing', methods=['GET', 'POST'])
def marketing():
    global travel_data, model
    if request.method == 'POST':
        if travel_data is None:
            flash('Please upload travel data first.')
            return redirect(url_for('index'))
        # Train a simple model for demo
        if model is None:
            X = travel_data.select_dtypes(include=['float64', 'int64'])
            y = (X.sum(axis=1) > X.sum(axis=1).mean()).astype(int)  # Dummy target
            model = RandomForestClassifier().fit(X, y)
        # Get input from form
        try:
            values = [float(request.form.get(f'feature{i}')) for i in range(len(travel_data.columns)-1)]
            pred = model.predict([values])[0]
            flash(f'Campaign Prediction: {"Success" if pred == 1 else "Failure"}')
        except Exception as e:
            flash(f'Error in prediction: {e}')
    return render_template('marketing.html')

if __name__ == '__main__':
    app.run(debug=True) 