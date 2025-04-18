# 딥러닝 프레임워크
import torch
# Kobert를 불러오기 위한 라이브러리
from transformers import BertTokenizer, BertModel
# 신경망을 만들기 위한 모듈 
import torch.nn as nn

# nn.Module 상속 받기 (모듈화, 상태관리)
# kobert가 뽑아낸 특징 벡터(768차원)을 받아 2가지로 분류 (욕인지 아닌지)
class CussClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert")
        self.classifier = nn.Linear(768, 2)

    # forward 함수는 모델이 실제로 예측할 대 호출되는 함수 
    # 토크나이저로 입력 데이터를 나눔 
    # output로 문장 전체 의미 요약한 768차원 벡터
    # classifier 거쳐서 0 or 1 형태 출력 
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)

# 모델 불러오기 
# 토크나이저 준비 
# pth 불러오고 model.eval은 모델 예측만 하겠다. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
model = CussClassifier().to(device)
model.load_state_dict(torch.load("model/kobert_cuss.pth", map_location=device))
model.eval()

# 사용자 입력값 text
# 토크나이저 숫자로 변환 
# output 모델 예측 결과 
# softmax 확률 형태로 변환
# argmax 확률이 높은 쪽 선택해 라벨로 반환
# prob[0][1] 욕설 확률 숫자로 추출 
def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=64)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        prob = torch.softmax(output, dim=1)
        label = torch.argmax(prob, dim=1).item()
        return label, float(prob[0][1])
