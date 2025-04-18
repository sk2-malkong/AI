# 💬 AI 욕설 탐지 프로젝트_Purgo 

### 👥 팀원
- 이현영
- 백욱진  
- 김태기  
- 김소현  
- 표상혁

### 🌿 브랜치
- **finish**

---

## 📌 프로젝트 개요
Flask 기반의 웹 서버에서 OpenAI LLM을 활용해  
**댓글/채팅의 욕설 탐지 및 정화 기능**을 수행하는 AI 서비스입니다.

### 딥러닝 프레임워크
import torch
### Kobert를 불러오기 위한 라이브러리
from transformers import BertTokenizer, BertModel
### 신경망을 만들기 위한 모듈 
import torch.nn as nn
### CSV 읽기 쓰기
import pandas as pd 
### API 요청 
import requests 
### 바 차트 시각화 
import matplotlib.pyplot as plt
### HTML 템플릿  
from jinja2 import Template
### PDF 변환
import pdfkit
### 경로 처리 및 파일 열기  
import os  

### Kobert 필요 라이브리러
pip install torch torchvision torchaudio
pip install transformers
pip install transformers gluonnlp sentencepiece
pip install torch torchvision
pip install transformers==4.10.0
pip install gluonnlp==0.10.0
pip install sentencepiece
pip install pandas tqdm
pip install kobert-tokenizer
pip install sentencepiece protobuf
pip install pandas 
pip install torch 
pip install scikit-learn

---

## 학습 파일
### C:\Users\r2com\Desktop\AI_Kobert_KoGPT\purgo_kobert\model\kobert_cuss_epoch(32)_batch_size=32.pth

## 학습 데이터 셋 
### C:\Users\r2com\Desktop\AI_Kobert_KoGPT\purgo_kobert\data\sample_data0.csv
### C:\Users\r2com\Desktop\AI_Kobert_KoGPT\purgo_kobert\data\sample_data1.csv

## fasttext 데이터 단어 셋
### C:\Users\r2com\Desktop\AI_Kobert_KoGPT\purgo_kobert\app\fasttext_filter\fasttext_cuss_train_full.txt

## 자동화 테스트 데이터 셋 
### C:\Users\r2com\Desktop\AI_Kobert_KoGPT\purgo_kobert\app\test_input.csv

---

### KoGPT 필요 라이브러리 - 그나마 괜찮
pip install transformers torch sentencepiece

### nlpai-lab/korean-paraphrase-t5-small 필요 라이브러리 - 사용 불가
pip install transformers sentencepiece

### paust/pko-t5-base - 답변 그대로 출력 성능 아쉽.

### beomi/KoParrot - 사용 불가

### digit82/kobart-summarization - 성능 별로

### 성능 테스트 
pip install jinja2 matplotlib pandas
pip install pandas requests matplotlib jinja2


### 1단계 fasttext
pip install fasttext

### 2단계 Kobert

### 3단계 KoGPT

### 실행 방법 - 서버 실행 후 새 터미널에서 진행 
학습 실행 방법 PS C:\Users\r2com\Desktop\AI_Kobert_KoGPT\purgo_kobert> python train.py 
서버 실행 방법 PS C:\Users\r2com\Desktop\AI_Kobert_KoGPT> python run.py
테스트 실행 방법 C:\Users\r2com\Desktop\AI_Kobert_KoGPT\purgo_kobert\app\test_api_from_csv_FINAL_FIXED.py


 