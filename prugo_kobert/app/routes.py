from flask import Blueprint, request, jsonify
from .model import predict
from .preprocess import replace_bad_words

main = Blueprint("main", __name__)
# predict에 POST형식으로 json형태 반환
@main.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    text = data.get("text", "")
    label, prob = predict(text)

    if label == 1:
        cleaned = replace_bad_words(text)
        return jsonify({
            "label": "욕설",
            "probability": prob,
            "original": text,
            "cleaned": cleaned
        })
    else:
        return jsonify({
            "label": "정상",
            "probability": prob,
            "original": text
        })
    
    # 클라이 언트 요청 -> Kobert 욕설 여부 판단 -> 결과 json 응답 
