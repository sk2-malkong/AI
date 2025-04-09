'''
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def refine_abusive_text(text):
    prompt = f"""
문장에서 욕설 여부를 판단하고, 욕설이면 해당 부분을 순화해줘.

문장: "{text}"

다음 형식으로 응답해줘:
욕설 여부: true/false
욕설 부분: (있으면 명시, 없으면 '해당 없음')
욕설 부분의 갯수: (욕설 부분이 몇개 존재하는 지 숫자형식으로 출력)
순화 문장: (순화된 문장 또는 원문 유지)
리포트: (욕설 횟수 타운팅 및 기록, 해당 욕설에 대한 원문과 의미, 욕설의 대체어를 모두 정리해서 리포트를 작성)
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    result = response['choices'][0]['message']['content']

    lines = result.lower().splitlines()
    abusive = "true" in lines[0]
    partnum = ""
    part = ""
    refined = ""
    report = ""

    for line in lines:
        if line.startswith("욕설 부분:"):
            part = line.replace("욕설 부분:", "").strip()
        elif line.startswith("욕설 부분의 갯수:"):
            partnum = line.replace("욕설 부분의 갯수:", "").strip()
        elif line.startswith("순화 문장:"):
            refined = line.replace("순화 문장:", "").strip()
        elif line.startswith("리포트:"):
            report = line.replace("리포트:", "").strip()

    return {
        "gpt_abusive": abusive,
        "abusive_part": part,
        "part_num": partnum,
        "refined": refined,
        "report": report
    }
'''

'''
from transformers import pipeline

# KoAlpaca-LoRA 모델 로딩
pipe = pipeline("text-generation", model="beomi/KoAlpaca-Polyglot-5.8B", device=-1)  # CPU 사용

def parse_output(output: str):
    lines = output.lower().splitlines()
    abusive = "true" in lines[0]
    partnum = ""
    part = ""
    refined = ""
    report = ""

    for line in lines:
        if line.startswith("욕설 부분:"):
            part = line.replace("욕설 부분:", "").strip()
        elif line.startswith("욕설 부분의 갯수:"):
            partnum = line.replace("욕설 부분의 갯수:", "").strip()
        elif line.startswith("순화 문장:"):
            refined = line.replace("순화 문장:", "").strip()
        elif line.startswith("리포트:"):
            report = line.replace("리포트:", "").strip()

    return {
        "gpt_abusive": abusive,
        "abusive_part": part,
        "part_num": partnum,
        "refined": refined,
        "report": report
    }

def refine_abusive_text(text):
    prompt = f"""
문장에서 욕설 여부를 판단하고, 욕설이면 해당 부분을 순화해줘.

문장: "{text}"

다음 형식으로 응답해줘:
욕설 여부: true/false
욕설 부분: (있으면 명시, 없으면 '해당 없음')
욕설 부분의 갯수: (욕설 부분이 몇개 존재하는 지 숫자형식으로 출력)
순화 문장: (순화된 문장 또는 원문 유지)
리포트: (욕설 횟수 타운팅 및 기록, 해당 욕설에 대한 원문과 의미, 욕설의 대체어를 모두 정리해서 리포트를 작성)
"""

    result = pipe(prompt, max_new_tokens=256, do_sample=True)[0]['generated_text']
    return parse_output(result)
'''

'''
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# Hugging Face API Key 환경변수 또는 직접 입력
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
#API_URL = "https://api-inference.huggingface.co/models/beomi/KoAlpaca-Polyglot-5.8B"
#API_URL = "https://api-inference.huggingface.co/models/gpt2"
API_URL = "https://api-inference.huggingface.co/models/EleutherAI/polyglot-ko-1.3B"


headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
}

def parse_output(output: str):
    lines = output.lower().splitlines()
    abusive = "true" in lines[0]
    partnum = ""
    part = ""
    refined = ""
    report = ""

    for line in lines:
        if line.startswith("욕설 부분:"):
            part = line.replace("욕설 부분:", "").strip()
        elif line.startswith("욕설 부분의 갯수:"):
            partnum = line.replace("욕설 부분의 갯수:", "").strip()
        elif line.startswith("순화 문장:"):
            refined = line.replace("순화 문장:", "").strip()
        elif line.startswith("리포트:"):
            report = line.replace("리포트:", "").strip()

    return {
        "gpt_abusive": abusive,
        "abusive_part": part,
        "part_num": partnum,
        "refined": refined,
        "report": report
    }

def refine_abusive_text(text):
    prompt = f"""
문장에서 욕설 여부를 판단하고, 욕설이면 해당 부분을 순화해줘.

문장: "{text}"

다음 형식으로 응답해줘:
욕설 여부: true/false
욕설 부분: (있으면 명시, 없으면 '해당 없음')
욕설 부분의 갯수: (욕설 부분이 몇개 존재하는 지 숫자형식으로 출력)
순화 문장: (순화된 문장 또는 원문 유지)
리포트: (욕설 횟수 타운팅 및 기록, 해당 욕설에 대한 원문과 의미, 욕설의 대체어를 모두 정리해서 리포트를 작성)
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "do_sample": True,
            "max_new_tokens": 256,
            "temperature": 0.7
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Inference API Error: {response.status_code}, {response.text}")

    generated_text = response.json()[0]["generated_text"]
    return parse_output(generated_text)
'''

'''
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# Hugging Face API Key 환경변수
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/EleutherAI/polyglot-ko-1.3B"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

def parse_output(output: str):
    lines = output.strip().splitlines()

    abusive = False
    partnum = ""
    part = ""
    refined = ""
    report = ""

    for line in lines:
        lower_line = line.lower()
        if "욕설 여부:" in lower_line:
            abusive = "true" in lower_line
        elif "욕설 부분:" in lower_line:
            part = line.split(":", 1)[1].strip()
        elif "욕설 부분의 갯수:" in lower_line:
            part_num = line.split(":", 1)[1].strip()
        elif "순화 문장:" in lower_line:
            refined = line.split(":", 1)[1].strip()
        elif "리포트:" in lower_line:
            report = line.split(":", 1)[1].strip()

    return {
        "gpt_abusive": abusive,
        "abusive_part": part,
        "part_num": part_num,
        "refined": refined,
        "report": report
    }

def refine_abusive_text(text):
    prompt = f"""
다음 문장의 욕설 여부를 판단하고, 아래 양식에 따라 실제 값을 채워서 응답해줘.

문장: "{text}"

---응답 양식---
욕설 여부: true 또는 false 중 하나로 대답해줘
욕설 부분: (존재하면 작성하고, 없으면 '해당 없음'이라고 적어)
욕설 부분의 갯수: (숫자로만 적어)
순화 문장: (순화된 문장 또는 원문 그대로)
리포트: (욕설 횟수, 원문 욕설, 의미, 대체어 등 정리)
---끝---

양식을 그대로 복사하지 말고, 실제 값으로 채워서 응답해줘.
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 256,
            "return_full_text": False
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Inference API Error: {response.status_code}, {response.text}")

    generated_text = response.json()[0]["generated_text"]
    print("===== Hugging Face 생성 텍스트 =====")
    print(generated_text)  # 여기 추가!

    return parse_output(generated_text)
'''




import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # .env에 저장: GROQ_API_KEY=your-key
API_URL = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def parse_output(output: str):
    lines = output.lower().splitlines()
    abusive = "true" in lines[0]
    partnum = ""
    part = ""
    refined = ""
    report = ""

    for line in lines:
        if line.startswith("욕설 여부:"):
            abusive = line.replace("욕설 여부:", "").strip()
        elif line.startswith("욕설 부분:"):
            part = line.replace("욕설 부분:", "").strip()
        elif line.startswith("욕설 여부2:"):
            partnum = line.replace("욕설 여부2:", "").strip()
        elif line.startswith("순화 문장:"):
            refined = line.replace("순화 문장:", "").strip()
        elif line.startswith("리포트:"):
            report = line.replace("리포트:", "").strip()

    return {
        "gpt_abusive": abusive,
        "abusive_part": part,
        "part_num": partnum,
        "refined": refined,
        "report": report
    }

def refine_abusive_text(text):
    prompt = f"""
응답은 아래 규칙을 다른다.
1. 문장에서 욕설 여부를 판단하고, 욕설이면 해당 부분을 순화해줘.
2. 욕설여부를 제외하고는 모두 한국어로 작성해줘.
3. 괄호안의 설명을 기준으로 응답을 작성한다.

문장: "{text}"

다음 형식으로 응답해줘:
욕설 여부: true/false
욕설 부분: (있으면 명시, 없으면 '해당 없음')
욕설 여부2: (욕설 여부가 true이면 1, False이면 0을 출력한다.)
순화 문장: (욕설이면 한국어로 이루어진 순화된 대체문장을 생성하고, 욕설이 아니면 원문을 그대로 적어줘.)
리포트: (욕설로 판단한 이유를 한국어로 정리하는 글을 작성)
"""

    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "너는 욕설 탐지 및 순화 전문가야."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Inference API Error: {response.status_code}, {response.text}")

    generated_text = response.json()["choices"][0]["message"]["content"]
    return parse_output(generated_text)

