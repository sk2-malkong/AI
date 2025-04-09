import pandas as pd

df = pd.read_csv("data/labeled_data.csv")
df['text'] = df['tweet']
df['label'] = df['class'].apply(lambda x: 0 if x == 2 else 1)

df[['text', 'label']].to_csv("data/hate_speech.csv", index=False)
print("✅ hate_speech.csv 생성 완료")
