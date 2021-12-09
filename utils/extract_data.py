import pandas as pd

"""새 데이터만 뽑기 (새로 받은 csv에 기존 데이터가 합쳐져 있어서 만든 코드)"""

df = pd.read_csv('labels/data_100.csv', index_col=0)  # 기존 데이터
df_new = pd.read_csv('labels/data.csv', index_col=0)  # 새 데이터

i = 0
new_rows = pd.DataFrame([])
for col in range(len(df_new)):
    if df_new.index[col] not in df.index:
        new_rows = new_rows.append(df_new.iloc[col])
        i += 1

# column 순서는 항상 이걸로 통일 (100개로 학습한 모델이 이 순서로 학습했기 때문에)
new_rows = new_rows[['Cerebral WM Hypointensities* Total Percent Of Icv',
                     'Cortical Gray Matter Total Percent Of Icv',
                     'Ventricle Total Percent Of Icv',
                     'Cerebral White Matter Total Percent Of Icv',
                     'Whole Brain Total Percent Of Icv']]

new_rows.to_csv(f'data_{len(new_rows)}.csv', index=False)
