# # ✅ Flask 서버 직접 연결
# API_URL = "http://3.34.64.170:5000/analyze"

# PROXY_URL = "http://127.0.0.1:8001/proxy/analyze"   로컬일때 
# API_KEY = "abcd-1234-efgh-5678"  # 프록시 서버 API 키

#  ec2 가동일떄 
# PROXY_URL = "http://3.34.64.170:5000/proxy/analyze"   로컬일때 
# API_KEY = "abcd-1234-efgh-5678"  # 프록시 서버 API 키

# 로컬 AI 
# API_URL = "http://127.0.0.1:5000/analyze"

import pandas as pd
import requests
import time
from jinja2 import Template
import pdfkit
import os
from tqdm import tqdm  # ✅ 추가

API_URL = "http://127.0.0.1:5000/analyze"

# ✅ Flask 서버 직접 연결
# API_URL = "http://13.125.55.220:5000/analyze"


input_path = "최종 테스트용 욕설_5 copy.csv"
html_output_path = "test_result_report.html"
pdf_output_path = "test_result_report.pdf"

df = pd.read_csv(input_path)

print("🚀 테스트 실행 중...")

results = []
start_time = time.time()

# ✅ tqdm으로 진행률 표시
for idx, row in tqdm(df.iterrows(), total=len(df), desc="진행 상황", ncols=100):
    text = row["text"]
    try:
        response = requests.post(API_URL, json={"text": text})
        if response.status_code == 200:
            data = response.json()
            results.append({
                "문장": data.get("result", {}).get("original_text", ""),
                "단어욕설_감지_수": len(data.get("fasttext", {}).get("detected_words", [])),
                "문맥욕설_감지": data.get("kobert", {}).get("is_bad", ""),
                "문맥_신뢰도": data.get("kobert", {}).get("confidence", 0),
                "욕설_여부": data.get("final_decision", ""),
                "정제된문장": data.get("result", {}).get("rewritten_text", "")
            })
        else:
            results.append({
                "문장": text,
                "단어욕설_감지_수": "요청 실패",
                "문맥욕설_감지": "요청 실패",
                "문맥_신뢰도": 0,
                "욕설_여부": "요청 실패",
                "정제된문장": "요청 실패"
            })
    except Exception as e:
        results.append({
            "문장": text,
            "단어욕설_감지_수": "에러",
            "문맥욕설_감지": "에러",
            "문맥_신뢰도": 0,
            "욕설_여부": "에러",
            "정제된문장": str(e)
        })

end_time = time.time()
elapsed_time = end_time - start_time

result_df = pd.DataFrame(results)

total = len(result_df)
abusive_count = pd.to_numeric(result_df["욕설_여부"], errors='coerce').fillna(0).astype(int).sum()
normal_count = total - abusive_count

confidence_values = pd.to_numeric(result_df["문맥_신뢰도"], errors='coerce')
average_confidence = round(confidence_values.mean(), 4)

template_str = """
<html>
<head>
    <meta charset="UTF-8">
    <title>Purgo - 욕설 탐지 리포트</title>
    <style>
        body { font-family: Arial, sans-serif; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; }
        .summary { margin-bottom: 20px; padding: 10px; background-color: #f9f9f9; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <h2>Purgo - 욕설 탐지 리포트</h2>
    <div class="summary">
        <p>총 문장 수: {{ total }}</p>
        <p>욕설 감지 문장 수: {{ abusive_count }}</p>
        <p>정상 문장 수: {{ normal_count }}</p>
        <p>평균 문맥 신뢰도: {{ average_confidence }}</p>
        <p>총 처리 시간: {{ elapsed_time }} 초</p>
    </div>
    <table>
        <thead>
            <tr>
                <th>문장</th>
                <th>단어욕설 감지 수</th>
                <th>문맥욕설 감지</th>
                <th>문맥 신뢰도</th>
                <th>욕설 여부</th>
                <th>정제된 문장</th>
            </tr>
        </thead>
        <tbody>
        {% for row in rows %}
            <tr>
                <td>{{ row.문장 }}</td>
                <td>{{ row.단어욕설_감지_수 }}</td>
                <td>{{ row.문맥욕설_감지 }}</td>
                <td>{{ row.문맥_신뢰도 }}</td>
                <td>{{ row.욕설_여부 }}</td>
                <td>{{ row.정제된문장 }}</td>
            </tr>
        {% endfor %}
        </tbody>
    </table>
</body>
</html>
"""

template = Template(template_str)
html_content = template.render(
    total=total,
    abusive_count=abusive_count,
    normal_count=normal_count,
    average_confidence=average_confidence,
    elapsed_time=round(elapsed_time, 2),
    rows=result_df.to_dict(orient="records")
)

with open(html_output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"✅ HTML 리포트 생성 완료: {html_output_path}")

config = pdfkit.configuration(
    wkhtmltopdf=r"C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"
)
options = {
    'enable-local-file-access': None
}

pdfkit.from_file(html_output_path, pdf_output_path, configuration=config, options=options)

print(f"✅ PDF 리포트 생성 완료: {pdf_output_path}")

os.startfile(pdf_output_path) 
