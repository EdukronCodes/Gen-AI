from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    # Placeholder for fraud detection logic
    result = "Transaction is legitimate."
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True) 