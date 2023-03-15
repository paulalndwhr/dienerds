import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import plotly.express as px
import scipy as sp
import seaborn as sns
import numpy as np

df = pd.read_csv('../data/clean.csv', sep=';')

columns = df.columns
print(columns)
# with PdfPages('dienerds.pdf') as pdf:
#     for i in columns:
#         print(f'i = {i}')
#         for j in columns:
#         # for j in ['/Section D-D/Circle 1/D']:
#             try:
#                 x_mean = df[i].mean()
#                 y_mean = df[j].mean()
#                 x_var = df[i].var()
#                 y_var = df[j].var()
#             except TypeError:
#                 x_mean = 'N/A'
#                 y_mean = 'N/A'
#                 x_var = 'N/A'
#                 y_var = 'N/A'
#             print(f'j = {j}')
#             plt.figure(figsize=(len(columns), len(columns)))
#             plt.scatter(df[i], df[j], color='b', alpha=0.05)
#             plt.xlabel(f'{i}, mean={x_mean}, var={x_var}')
#             plt.ylabel(f'{j}, mean={y_mean}, var={y_var}')
#             plt.title(f'{i} gegen {j}')
#             plt.plot()
#             pdf.savefig()
#             plt.close
#
# fig = px.scatter(df, x='/Section D-D/Circle 1/D', y='/Section E-E/Circle 10/D', color='nr')
# fig.show()
# fig.write_html("../data/example.html")


# ======================
# Nun dasselbe Spiel in Seaborn...
# =====================

with PdfPages('dienerds.pdf') as pdf:
    for i in columns:
        print(f'i = {i}')
        # for j in columns:
        for j in ['/Section D-D/Circle 1/D']:
            print(f'j = {j}')
            try:
                x_mean = df[i].mean()
                y_mean = df[j].mean()
                x_var = df[i].var()
                y_var = df[j].var()

                fig, ax = plt.subplots(figsize=(10, 6))
                ax = sns.regplot(x=df[i], y=df[j], ci=95, order=1,
                                 # plot_kws={'line_kws': {'color': 'red', 'alpha': 1}, 'scatter_kws': {'alpha': 0.1}},
                                 # line_kws={'label': 'Linear regression line: $Y(X)=5.74+2.39\cdot 10^{-5} X$', 'color': 'm'},
                                 seed=1, truncate=False)  # , label="Original data")
                ax.legend(loc="upper left")
                r, p = sp.stats.pearsonr(df[i], df[j])
                plt.xlabel(f'{i}, mean={x_mean}, var={x_var}\n r^2={r**2}')
                plt.ylabel(f'{j}, mean={y_mean}, var={y_var}')
            except TypeError:
                x_mean = 'N/A'
                y_mean = 'N/A'
                x_var = 'N/A'
                y_var = 'N/A'
                plt.scatter(df[i], df[j], color='b', alpha=0.05)

                plt.xlabel(f'{i}, mean={x_mean}, var={x_var}')
                plt.ylabel(f'{j}, mean={y_mean}, var={y_var}')
            plt.title(f'{i} gegen {j}')
            plt.plot()
            pdf.savefig()
            plt.close

#
# plt.show()
