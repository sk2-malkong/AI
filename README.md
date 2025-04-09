# KoBERT 기반 문맥 인식 욕설 감지 시스템

이 프로젝트는 한국어 문장의 **문맥을 이해하여 욕설을 인식하고 필터링**하는 AI 시스템을 구축합니다.  
PyTorch 기반 KoBERT 모델을 사용하며, Flask API 서버로 감싸 외부 서비스(Java, 프론트엔드 등)와 연동이 가능합니다.

---

## 📁 프로젝트 구조
kobert_abuse_filter/ ├── app.py # Flask API 서버 ├── model/ # 학습된 모델 및 토크나이저 저장 폴더 ├── utils/ │ └── predictor.py # KoBERT 기반 예측 함수 ├── venv/ # Python 가상환경 ├── requirements.txt └── README.md # 프로젝트 설명서


---

## 🛠 개발 환경

- macOS (Intel 기반)
- Python 3.8 ~ 3.11
- VS Code
- Virtual Environment (`venv`)

---

## 🧪 초기 설정 가이드

### 1️⃣ 가상환경 생성 및 활성화

```bash
python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip

pip install torch torchvision torchaudio \
--index-url https://download.pytorch.org/whl/cpu

pip install transformers flask sentencepiece
