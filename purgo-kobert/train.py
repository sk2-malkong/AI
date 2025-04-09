# purgo-kobert/train.py

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 커스텀 Dataset
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

# ✅ 분류기 모델
class CussClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert")
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)

# ✅ 데이터 불러오기
df1 = pd.read_csv("data/hate_speech.csv")
df2 = pd.read_csv("data/nsmc.csv")
df = pd.concat([df1, df2]).sample(frac=1).reset_index(drop=True)  # 셔플

tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
dataset = TextDataset(df, tokenizer)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# ✅ 모델 정의 및 학습
model = CussClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

print("📚 학습 시작...")
for epoch in range(3):
    total_loss = 0
    model.train()
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"📘 Epoch {epoch+1} - Loss: {total_loss:.4f}")

# ✅ 모델 저장
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/kobert_cuss.pth")
print("✅ 모델 저장 완료: model/kobert_cuss.pth")
