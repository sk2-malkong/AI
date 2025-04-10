# 💬 AI 욕설 탐지 프로젝트_Purgo 

### 👥 팀원
- 이현영
- 백욱진  
- 김태기  
- 김소현  
- 표상혁

### 🌿 브랜치
- **Taegi**

---

## 📌 프로젝트 개요
Flask 기반의 웹 서버에서 OpenAI LLM을 활용해  
**댓글/채팅의 욕설 탐지 및 정화 기능**을 수행하는 AI 서비스입니다.

---

### 필요 외부 라이브러리
#### pip install openai python-dotenv flask , 필요 버전 pip install openai==0.28.1
#### 현재 코드에서는 OpenAI 라이브러리의 `0.28.1` 버전 사용을 권장합니다.  
#### 최신 버전(`1.0.0` 이상)에서는 `ChatCompletion.create()` 방식이 제거되어 작동하지 않습니다.

---

## 🛠️ 프로젝트 실행 방법
#### python app.py 

### ▶️가상환경 만들기
#### python -m venv venv

### ▶️가상환경 활성화 
#### .\venv\Scripts\activate

### ▶️의존성 저장
#### pip freeze > requirements.txt

### ▶️EC2설치
#### pip install -r requirements.txt
