import pandas as pd

fields = ['Name', 'GencodeV41_Name']
df = pd.read_csv('EPIC-8v2-0_A1.csv', header=7)
df = df[fields]

df = df.drop(df['cg' not in df['Name']].index)
