import pandas as pd

df = pd.read_table("data/ratings_train.txt")
df = df.dropna()
df = df[df['label'] == 1]  # 긍정 문장만 사용
df = df[['document']]
df['label'] = 0  # KoBERT용 정상문장 → label 0
df.rename(columns={'document': 'text'}, inplace=True)
df.to_csv("data/nsmc.csv", index=False)

print("✅ nsmc.csv 생성 완료")
