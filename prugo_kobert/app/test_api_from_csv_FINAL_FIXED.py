import pandas as pd # CSV 읽기 쓰기 
import requests # API 요청 
import matplotlib.pyplot as plt # 바 차트 시각화 
from jinja2 import Template # HTML 템플릿 
import pdfkit # PDF 변환 
import os # 경로 처리 및 파일 열기 


# ✅ 한글 폰트 설정 한글 깨짐 설정 
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
API_URL = "http://127.0.0.1:5000/analyze" # Flask 서버 API
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 파일 위치 기준

# 테스트 데이터 받기 출력하기 리포트 출력 
input_path = os.path.join(BASE_DIR, "test_input.csv")
output_path = os.path.join(BASE_DIR, "test_results.csv")
report_html_path = os.path.join(BASE_DIR, "욕설_감지_리포트.html")
chart_path = os.path.join(BASE_DIR, "abuse_chart.png")
report_pdf_path = os.path.join(BASE_DIR, "욕설_감지_리포트.pdf")

# 데이터 읽기 처리
df = pd.read_csv(input_path)
results = []

# API 요청 및 결과 수집 
for _, row in df.iterrows():
    text = row["text"]
    try:
        response = requests.post(API_URL, json={"text": text})
        if response.status_code == 200:
            data = response.json()
            results.append({
                "문장": data.get("original_text", ""), 
                "단어욕설_감지_수": len(data.get("fasttext_bad_words", [])),
                "문맥욕설_감지": data.get("kobert_pred", ""),
                "문맥_신뢰도": data.get("kobert_confidence", ""),
                "욕설_여부": data.get("is_abusive", ""),
                "정제된문장": data.get("rewritten_text", "")
            })
    except Exception as e:
        results.append({
            "문장": text,
            "단어욕설_감지_수": "요청 실패",
            "문맥욕설_감지": "요청 실패",
            "문맥_신뢰도": "요청 실패",
            "욕설_여부": "요청 실패",
            "정제된문장": str(e)
        })

# 결과 저장
df_result = pd.DataFrame(results)
df_result.to_csv(output_path, index=False, encoding="utf-8-sig")

# 그래프 저장
df_result["욕설_여부"] = df_result["욕설_여부"].astype(str)
df_result["욕설_여부"].value_counts().plot(kind="bar", color=["red", "green"])
plt.title("욕설 여부 분포")
plt.ylabel("문장 수")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(chart_path)
plt.close()

# ✅ 절대 경로 생성
chart_path_for_html = "file:///" + chart_path.replace("\\", "/")

# HTML 템플릿
html_template = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>욕설 감지 리포트</title>
    <style>
        body { font-family: sans-serif; padding: 20px; }
        .entry { margin-bottom: 20px; border-bottom: 1px solid #ccc; padding: 10px; }
        .label { font-weight: bold; color: #333; }
    </style>
</head>
<body>
    <h1>욕설 감지 리포트</h1>
    <img src="{{ chart_path }}" alt="욕설 여부 그래프" width="400">
    {% for row in rows %}
    <div class="entry">
        <p><span class="label">📌 문장:</span> {{ row['문장'] }}</p>
        <p><span class="label">🔍 단어 욕설 감지 수:</span> {{ row['단어욕설_감지_수'] }}</p>
        <p><span class="label">🧠 문맥 판단:</span> {{ row['문맥욕설_감지'] }} (신뢰도: {{ row['문맥_신뢰도'] }})</p>
        <p><span class="label">🚨 최종 욕설 여부:</span> {{ row['욕설_여부'] }}</p>
        <p><span class="label">💬 정제된 문장:</span> {{ row['정제된문장'] }}</p>
    </div>
    {% endfor %}
</body>
</html>"""

# HTML 생성 및 저장
html_content = Template(html_template).render(
    rows=results,
    chart_path=chart_path_for_html
)

with open(report_html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

# pdf 경로 설정~ 
config = pdfkit.configuration(
    wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
)

# 옵션 추가
options = {
    'enable-local-file-access': None
}

# PDF 생성
pdfkit.from_file(report_html_path, report_pdf_path, configuration=config, options=options)


# PDF 자동 열기
os.startfile(report_pdf_path)
