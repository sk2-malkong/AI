# 라이브러리 및 디바이스 설정 
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, AutoTokenizer
import os
import time
from sklearn.model_selection import train_test_split

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Dataset 클래스 정의
# Dataset 
# 토크나이저 숫자로 변환 
# output 모델 예측 결과 
# softmax 확률 형태로 변환
# argmax 확률이 높은 쪽 선택해 라벨로 반환
# prob[0][1] 욕설 확률 숫자로 추출 
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.texts = df['text'].tolist()
        self.labels = df['label'].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    # pyToch에서 텍스트 데이터를 쉽게 불러오도록 커스텀 Dataset 클래스 생성 
    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx])
        }

    # 전체 데이터 크기 반환 
    def __len__(self):
        return len(self.labels)

# ✅ 모델 정의
class CussClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert", trust_remote_code=True)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)

# ✅ 데이터 불러오기 및 병합 
df1 = pd.read_csv("data/sample_data1.csv")
df1 = df1[df1['label'].isin([0, 1])]
df1['label'] = df1['label'].astype(int)
# label은 0 또는 1만 사용 
df2 = pd.read_csv("data/sample_data0.csv")
df2 = df2[df2['label'].isin([0, 1])]
df2['label'] = df2['label'].astype(int)

# ✅ 데이터 병합 및 셔플
df = pd.concat([df1, df2]).sample(frac=1).reset_index(drop=True)

# ✅ 학습/검증 데이터 나누기 
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# ✅ Tokenizer 및 DataLoader 생성
# Kobert용 도크나이저 로드 
# TextDataset을 PyTorch DataLoader로 감싸서 미니배치 학습 
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
train_dataset = TextDataset(train_df, tokenizer)
val_dataset = TextDataset(val_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ✅ 모델 초기화
model = CussClassifier().to(device)
print("\n🆕 기존 모델 무시하고 새 모델을 학습합니다.")

# 학습률 2e-5
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# ✅ 학습 루프 시작
print("\n📚 학습 시작...")
num_batches = len(train_loader)

# 에폭 수 32
for epoch in range(32):
    start_time = time.time()
    total_loss = 0
    model.train()

    for i, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        progress = (i + 1) / num_batches * 100
        print(f"\r🌀 Epoch {epoch+1} 진행률: {progress:.2f}% | 현재 Loss: {loss.item():.4f}", end='')

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n📘 Epoch {epoch+1} 완료 - Train Loss: {total_loss:.4f} | 소요 시간: {elapsed:.2f}초")

    # ✅ 검증 루프
    # 모델을 평가 모델로 전환 후 검증 데이터셋에 대해 예측 
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            val_loss += loss_fn(outputs, labels).item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"✅ Validation Loss: {val_loss:.4f}, Accuracy: {acc:.4f}")

# ✅ 모델 저장
# 학습이 끝난 파일을 model 폴더에 저장 
# 모델 저장~ 
os.makedirs("model", exist_ok=True)
model_path = "model/kobert_cuss_epoch(32)_batch_size=32.pth"
torch.save(model.state_dict(), model_path)
print(f"\n✅ 모델 저장 완료: {model_path}")
print("🎉 전체 학습 완료! 이제 kobert_cuss_epoch(32)_batch_size=32 모델을 사용할 수 있어요.")

# 학습 손실 (Train Loss)

# 검증 손실 (Validation Loss)

# 정확도 (Accuracy)

# 에폭 시간 (소요 시간)
