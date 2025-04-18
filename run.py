from flask import Flask, request, jsonify
import torch
import fasttext
from transformers import BertModel, AutoTokenizer, GPT2LMHeadModel, PreTrainedTokenizerFast
import torch.nn as nn
import os
import re

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
model_path = os.path.join("purgo_kobert", "model", "kobert_cuss_epoch(32)_batch_size=32.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ✅ 토크나이저
kobert_tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
kogpt_model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2").to(device)
kogpt_tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
kogpt_model.eval()

# ✅ FastText 모델 로딩
fasttext_model = fasttext.load_model("purgo_kobert/app/fasttext_filter/fasttext_cuss_model.bin")

# ✅ 정중한 문장 생성 함수
def clean_output(text):
    text = text.replace("\\n", " ").replace("\n", " ")
    return re.sub(r"[^가-힣0-9.,!?\s]", "", text).strip()

def rewrite_text_kogpt(text):
    prompt = f"'{text}' 문장을 공손하고 매우 정중한 말로 바꿔줘. 문맥도 자연스럽게 만들어줘."
    input_ids = kogpt_tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = kogpt_model.generate(
            input_ids,
            max_length=100,
            do_sample=True,
            top_p=0.85,
            temperature=0.7,
            repetition_penalty=1.5,
            no_repeat_ngram_size=2,
            num_beams=5,
            pad_token_id=kogpt_tokenizer.pad_token_id
        )
    result = kogpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_output(result.replace(prompt, "").strip())

# ✅ 단어 기반 욕설 탐지
def detect_fasttext(text):
    words = text.split()
    bad_words = []
    for word in words:
        label, prob = fasttext_model.predict(word)
        if label[0] == "__label__1" and prob[0] > 0.7:
            bad_words.append({"word": word, "prob": round(prob[0], 3)})
    return bad_words

# ✅ 문맥 기반 KoBERT 탐지
def detect_kobert(text):
    inputs = kobert_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
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

@app.route("/", methods=["GET"])
def home():
    return "✅ FastText + KoBERT + KoGPT 욕설 감지 및 정제 서버 작동 중!"

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "text 필드가 필요합니다"}), 400

    # 1단계: FastText 탐지
    fasttext_result = detect_fasttext(text)
    fasttext_hit = len(fasttext_result) > 0

    # 2단계: KoBERT 탐지
    kobert_pred, kobert_conf = detect_kobert(text)
    kobert_hit = kobert_pred == 1

    # 최종 판단
    is_abusive = fasttext_hit or kobert_hit

    # 3단계: KoGPT 정제
    rewritten = rewrite_text_kogpt(text) if is_abusive else text

    return jsonify({
        "original_text": text,
        "fasttext_bad_words": fasttext_result,
        "kobert_pred": kobert_pred,
        "kobert_confidence": kobert_conf,
        "is_abusive": is_abusive,
        "rewritten_text": rewritten
    })

if __name__ == "__main__":
    app.run(debug=True)
