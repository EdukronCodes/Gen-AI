from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Placeholder for prediction logic
    prediction = "Equipment is functioning normally."
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True) 