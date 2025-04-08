from flask import Flask, request, jsonify
from services.abuse_detector import hybrid_abuse_check

app = Flask(__name__)

@app.route('/check-abuse', methods=['POST'])
def check_abuse():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400

    result = hybrid_abuse_check(data['text'])
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
