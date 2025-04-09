# 머신러닝 모델
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from models.badword_check.BadWord import load_badword_model, preprocessing
from models.kobert.kobert_predict import predict_abuse_kobert

model = load_badword_model()


'''
def predict(text):
    x = preprocessing(text)
    score = model.predict(x)[0][0]
    return {
        "label": 1 if score > 0.5 else 0,
        "score": float(score)
    }
'''

def predict(text):
    x = preprocessing(text)
    score = float(model.predict(x)[0][0])
    is_abusive = score > 0.5
    return {"is_abusive": is_abusive, "score": score}

'''
def load_model(model_path="beomi/kcbert-base"):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = torch.argmax(probs, dim=1).item()
    score = probs[0][label].item()
    return {"label": label, "score": score}
'''
