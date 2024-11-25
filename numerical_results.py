import pandas as pd
df = pd.read_csv('./evaluation_results.csv')
cols = df.columns
print(df.groupby("model")[cols[:10]].mean())