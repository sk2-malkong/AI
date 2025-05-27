# 학습된 fasttext 코드
import pandas as pd
import fasttext
import time
import os

# 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
train_csv = os.path.join(base_dir, "fasttext_cuss_train_full.txt")
train_txt = os.path.join(base_dir, "fasttext_train.txt")
model_bin = os.path.join(base_dir, "fasttext_cuss_model.bin")
test_input = os.path.join(base_dir, "test_input.csv")
test_output = os.path.join(base_dir, "test_results.csv")

# ⏱ 로그 출력 함수
def log(msg):
    now = time.strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {msg}")

# ✅ 1단계: CSV → FastText 학습 포맷 변환
def convert_csv_to_fasttext_format(input_csv, output_txt):
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    if "label" not in df.columns or "text" not in df.columns:
        df.columns = ["text", "label"]
    with open(output_txt, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            label = int(row["label"])
            text = str(row["text"]).strip().replace(" ", "")
            if text:
                f.write(f"__label__{label} {text}\n")
    log(f"✅ 학습용 텍스트 저장 완료 → {output_txt}")

# ✅ 2단계: FastText 모델 학습 및 저장
def train_fasttext_model(train_file, model_output):
    log("⏳ FastText 모델 학습 시작")
    start = time.time()
    model = fasttext.train_supervised(
        input=train_file,
        lr=1.0,
        epoch=25,
        wordNgrams=2,
        minCount=1,
        verbose=2
    )
    model.save_model(model_output)
    elapsed = round(time.time() - start, 4)
    log(f"✅ 모델 학습 및 저장 완료 → {model_output}")
    log(f"⏱️ 총 학습 시간: {elapsed}초")
    return model

# ✅ 3단계: 테스트 문장 예측 후 저장
def test_fasttext_model(model, input_csv, output_csv):
    log("🔍 테스트 시작")
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    results = []

    for _, row in df.iterrows():
        text = str(row["text"]).strip().replace(" ", "")
        if not text:
            continue
        label, prob = model.predict(text)
        is_abusive = "욕설" if label[0] == "__label__1" else "중립"
        results.append({
            "문장": text,
            "예측_라벨": label[0],
            "신뢰도": round(prob[0], 4),
            "욕설_여부": is_abusive
        })

    pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")
    log(f"✅ 테스트 결과 저장 완료 → {output_csv}")

# ✅ 전체 실행
if __name__ == "__main__":
    log("🚀 FastText 전체 프로세스 시작")
    convert_csv_to_fasttext_format(train_csv, train_txt)
    model = train_fasttext_model(train_txt, model_bin)
    test_fasttext_model(model, test_input, test_output)
    log("🏁 전체 완료")
