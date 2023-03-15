import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import plotly.express as px

df = pd.read_csv('../data/clean.csv', sep=';')

columns = df.columns
print(columns)
with PdfPages('dienerds.pdf') as pdf:
    for i in columns:
        print(f'i = {i}')
        for j in columns:
        # for j in ['/Section D-D/Circle 1/D']:
            try:
                x_mean = df[i].mean()
                y_mean = df[j].mean()
                x_var = df[i].var()
                y_var = df[j].var()
            except TypeError:
                x_mean = 'N/A'
                y_mean = 'N/A'
                x_var = 'N/A'
                y_var = 'N/A'
            print(f'j = {j}')
            plt.figure(figsize=(len(columns), len(columns)))
            plt.scatter(df[i], df[j], color='b', alpha=0.05)
            plt.xlabel(f'{i}, mean={x_mean}, var={x_var}')
            plt.ylabel(f'{j}, mean={y_mean}, var={y_var}')
            plt.title(f'{i} gegen {j}')
            plt.plot()
            pdf.savefig()
            plt.close

fig = px.scatter(df, x='/Section D-D/Circle 1/D', y='/Section E-E/Circle 10/D', color='nr')
fig.show()
fig.write_html("../data/example.html")

