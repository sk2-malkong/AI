# routes.py

from flask import Blueprint, request, jsonify
from services.abuse_detector import hybrid_abuse_check

routes = Blueprint('routes', __name__)

@routes.route('/check-abuse', methods=['POST'])
def check_abuse():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field'}), 400

    result = hybrid_abuse_check(data['text'])
    return jsonify(result)
