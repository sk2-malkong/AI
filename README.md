 # 💬 Purgo - AI 욕설 탐지 및 정화 시스템

Flask 기반 웹 서버에서 **KoBERT + KoGPT + FastText**를 활용하여  
댓글/채팅 등 텍스트 내 **욕설을 탐지하고 정제하는 AI 서비스**입니다.

---

## 👥 AI 팀원
| 이름 | 
| 이현영 | 
| 김태기 | 
| 백욱진 | 
| 김소현 | 
| 표상혁 | 

---

## 🌿 주요 브랜치
- `finish`: 최종 완성 브랜치

---

## 🛠 사용 기술 및 라이브러리

### 딥러닝 프레임워크
- PyTorch
- transformers
- KoBERT
- KoGPT

### 서버 및 시각화
- Flask
- jinja2
- matplotlib
- pdfkit

### 데이터 처리
- pandas
- scikit-learn
- sentencepiece
- requests
- os

---

## 🧠 욕설 탐지 파이프라인

```
1단계 FastText → 2단계 KoBERT → 3단계 KoGPT 직렬 조건부 구조 
```

- **FastText**: 단어 기반 필터링
- **KoBERT**: 문장/문맥 기반 욕설 감지
- **KoGPT**: 감지된 욕설을 정중하게 변환

---

## 📁 프로젝트 디렉터리 구조

```
purgo_kobert/
├── app/
│   ├── fasttext_filter/
│   │   └── fasttext_cuss_train_full.txt       ← FastText 단어 사전
│   └── test_api_from_csv_FINAL_FIXED.py       ← 자동화 테스트 실행
├── data/
│   ├── sample_data0.csv                       ← 학습 데이터
│   └── sample_data1.csv
├── model/
│   └── kobert_cuss_epoch(32)_batch_size=32.pth ← KoBERT 학습 모델
├── train.py                                    ← KoBERT 학습 스크립트
├── run.py                                      ← Flask 서버 실행
├── test_input.csv                              ← 테스트용 입력
```

---

## ⚙️ 주요 라이브러리 설치 
# ✅ 웹 서버
pip install flask
pip install fastapi
pip install uvicorn

# ✅ AI 모델 (딥러닝)
pip install torch torchvision torchaudio
pip install transformers==4.10.0
pip install gluonnlp==0.10.0
pip install sentencepiece
pip install kobert-tokenizer
pip install fasttext

# ✅ 데이터 처리
pip install pandas
pip install tqdm
pip install scikit-learn

# ✅ HTTP 통신
pip install requests

# ✅ 보고서 및 시각화
pip install jinja2
pip install matplotlib
pip install pdfkit
pip install tqdm

# pdf 변환
wkhtmltopdf

### 1단계 fasttext
### 2단계 Kobert
### 3단계 KoGPT


### KoBERT 관련
```bash
pip install torch torchvision torchaudio
pip install transformers==4.10.0
pip install gluonnlp==0.10.0
pip install sentencepiece pandas tqdm kobert-tokenizer scikit-learn
```

### KoGPT 관련
```bash
pip install transformers torch sentencepiece
```

### FastText 관련
```bash
pip install fasttext
```

### 보고서 및 시각화
```bash
pip install jinja2 matplotlib pandas pdfkit
```

---

## 🚀 실행 방법

### 1. KoBERT 모델 학습
```bash
python train.py
```

### 2. Flask 서버 실행
```bash
python run.py
```

### 3. 자동화 테스트 실행
```bash
python app/test_api_from_csv_FINAL_FIXED.py
```

> ⚠️ `run.py`로 서버를 먼저 실행한 후, 새 터미널에서 테스트를 진행하세요.

---

## 📌 모델 성능 참고 메모

| 모델 이름 | 사용 가능 여부 | 메모 |
|-----------|----------------|------|
| nlpai-lab/korean-paraphrase-t5-small | ❌ | 사용 불가 |
| paust/pko-t5-base | ⭕ | 성능 아쉬움 |
| beomi/KoParrot | ❌ | 사용 불가 |
| digit82/kobart-summarization | ⭕ | 성능 미흡 |
| **KoGPT** (사용 중) | ⭕ | 성능 양호 |

---

## 📞 문의
이슈나 버그는 GitHub Issues 또는 Pull Request를 통해 알려주세요. 감사합니다! 🙇





# pip install openai
# pip install dotenv
# pip install openai==0.28.1