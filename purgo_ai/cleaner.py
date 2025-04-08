# 정화/정제 로직
# 욕설 정제 

import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def clean_profanity(text: str) -> str:
    prompt = f"""
    아래 문장에서 욕설, 비속어, 공격적인 표현을 순화하고 예의 바른 표현으로 정제해줘.
    원문: "{text}"
    
    정제된 문장만 출력해줘.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response['choices'][0]['message']['content'].strip()
