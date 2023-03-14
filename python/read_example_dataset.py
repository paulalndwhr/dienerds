import pandas as pd

df = pd.read_csv('../data/export.csv')

print(df)

print(df.columns)

series = df['Response']

column_list = series.tolist()

with open('../data/logs.json', 'w') as file:
    for line in column_list:
        file.write(f'{line}\n')


