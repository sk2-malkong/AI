# 욕설 탐지 

import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # ✅ 수정됨

def detect_profanity(text: str) -> dict:
    prompt = f"""
    문장에서 욕설이나 비속어가 포함되어 있는지 판단해.
    문장: "{text}"  

    결과는 JSON으로 응답해줘.
    {{
        "is_profanity": true 또는 false,
        "reason": "이유"
    }}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    result = response['choices'][0]['message']['content']
    return eval(result)  # 👉 json.loads(result)로 교체하면 더 안전함
