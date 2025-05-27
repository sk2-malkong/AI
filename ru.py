from flask import Flask, request, jsonify
import torch
from transformers import BertModel, AutoTokenizer
import torch.nn as nn
import os
import re
import csv

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ KoBERT 모델 정의
class CussClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert", trust_remote_code=True)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)

# ✅ KoBERT 모델 로딩
model = CussClassifier().to(device)
model_path = os.path.join("purgo_kobert", "model", "purgo.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ✅ KoBERT 토크나이저 로딩
kobert_tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

# ✅ 욕설 단어 사전 로딩 (TXT 기반)
def load_badwords_from_txt():
    path = os.path.join("purgo_kobert", "app", "befasttext_filter", "befasttext_cuss_train_full.txt")
    badwords = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().split(",")[0]
            if word:
                badwords.add(word)
    return badwords

# ✅ 텍스트 전처리 (특수문자 제거)
def clean_text(text):
    text = re.sub(r"[^\u4E00-\u9FFF\u3400-\u4DBF가-힣ㄱ-ㅎㅏ-ㅣ0-9\s]", "", text)
    return text

# ✅ FastText 감지
def detect_fasttext(text, badword_set):
    cleaned_text = clean_text(text)
    detected = []
    for word in badword_set:
        if word and word in cleaned_text:
            detected.append(word)
    return detected

# ✅ KoBERT 감지
def detect_kobert(text):
    inputs = kobert_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()
    return pred, round(conf, 4)

# ✅ Flask 앱 생성
app = Flask(__name__)
badword_set = load_badwords_from_txt()

@app.route("/", methods=["GET"])
def home():
    return "✅ FastText 특수문자 제거 + KoBERT 직렬 조건부 욕설 탐지 서버 실행 중!"

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "text 필드가 필요합니다"}), 400

    print(f"\n입력 문장: {text}")

    # 1단계: FastText 감지
    fasttext_result = detect_fasttext(text, badword_set)
    fasttext_hit = 1 if len(fasttext_result) > 0 else 0
    print(f"🔍 FastText 탐지 결과: {fasttext_result}")

    response = {
        "fasttext": {
            "is_bad": fasttext_hit,
            "detected_words": fasttext_result
        },
        "kobert": {
            "is_bad": None,
            "confidence": None
        },
        "result": {
            "original_text": text,
            "rewritten_text": text  # 정제 없이 그대로 유지
        },
        "final_decision": 0
    }

    if fasttext_hit == 1:
        print("✅ FastText로 욕설 감지 완료.")
        response["final_decision"] = 1
    else:
        kobert_pred, kobert_conf = detect_kobert(text)
        kobert_hit = 1 if kobert_pred == 1 else 0
        print(f"📘 KoBERT 예측 결과: {kobert_pred} ({'욕설' if kobert_hit else '중립'}) | 신뢰도: {kobert_conf}")

        response["kobert"]["is_bad"] = kobert_hit
        response["kobert"]["confidence"] = kobert_conf

        if kobert_hit == 1:
            print("✅ KoBERT 문맥으로 욕설 감지 완료.")
            response["final_decision"] = 1
        else:
            print("⭕️ KoBERT 문맥으로도 욕설 아님. 정상 처리.")

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)