# Flask 
# app.py

from flask import Flask, request, jsonify
from detector import detect_profanity
from cleaner import clean_profanity

app = Flask(__name__)

# ✅ 서버 상태 확인용
@app.route("/", methods=["GET"])
def index():
    return "✅ Flask 욕설 탐지/정화 서버 실행 중입니다."


# ✅ 욕설 탐지 엔드포인트
@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "텍스트를 입력해주세요."}), 400

    is_profanity = detect_profanity(text)
    return jsonify({
        "original_text": text,
        "is_profanity": is_profanity,
        "message": "욕설 감지됨" if is_profanity else "정상"
    })


# ✅ 욕설 정화 엔드포인트
@app.route("/clean", methods=["POST"])
def clean():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "텍스트를 입력해주세요."}), 400

    cleaned = clean_profanity(text)
    return jsonify({
        "original_text": text,
        "cleaned_text": cleaned
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
