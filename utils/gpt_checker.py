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
