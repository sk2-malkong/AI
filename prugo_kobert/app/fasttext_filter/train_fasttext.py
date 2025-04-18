import pandas as pd
import requests

API_URL = "http://127.0.0.1:5000/analyze"
input_path = "app/test_input.csv"
output_path = "app/test_results.csv"

df = pd.read_csv(input_path)
results = []

for _, row in df.iterrows():
    text = row["text"]
    try:
        response = requests.post(API_URL, json={"text": text})
        if response.status_code == 200:
            data = response.json()
            results.append({
                "문장": text,
                "단어욕설_감지_수": len(data.get("fasttext_bad_words", [])),
                "문맥욕설_감지": data.get("kobert_pred", "없음"),
                "욕설_여부": data.get("is_abusive", "없음"),
                "정제된문장": data.get("rewritten_text", "없음")
            })
        else:
            print(f"❌ 응답 오류: {response.status_code}")
    except Exception as e:
        print(f"❌ 요청 실패: {e}")
        results.append({
            "문장": text,
            "단어욕설_감지_수": "요청 실패",
            "문맥욕설_감지": "요청 실패",
            "욕설_여부": "요청 실패",
            "정제된문장": str(e)
        })

pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")
print("✅ 테스트 완료 → 결과 저장됨!")
