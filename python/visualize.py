import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import plotly.express as px
import scipy as sp
import seaborn as sns
import numpy as np
from typing import List
from math import exp


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


def all_plots_in_seaborn(df: pd.DataFrame) -> None:
    with PdfPages('dienerds-with-garbage.pdf') as pdf:
        for i in columns:
            print(f'i = {i}')
            for j in columns:
            # for j in ['/Section D-D/Circle 1/D']:
                print(f'j = {j}')
                # this section skips unnecessary diagrams which just increase the .pdf file size
                if i == j:
                    continue
                elif i == 'time':
                    continue
                elif j == 'time':
                    continue

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


def pdf_potentially_relevant_diagr(df: pd.DataFrame, savename: str = 'dienerds-without-garbage.pdf') -> None:
    """
    draws all potentially relevant diagrams into a .pdf, using seaborn and seaborn's regression fit.
    :param df:
    :param savename:
    :return:
    """

    with PdfPages(savename) as pdf:
        for i in columns:
            print(f'i = {i}')
            for j in columns:
                # for j in ['/Section D-D/Circle 1/D']:
                print(f'j = {j}')
                # this section skips unnecessary diagrams which just increase the .pdf file size
                if i == j:
                    continue
                elif i == 'time':
                    continue
                elif j == 'time':
                    continue

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

                    best_curves = find_best_curve(df=df, col_x=i, col_y=j)

                    plot2 = sns.lineplot(data=df, x=df[i], y=exponential(x, *best_curves['exp']), color='g', ax=ax)
                    r, p = sp.stats.pearsonr(df[i], df[j])
                    plt.xlabel(f'{i}, mean={x_mean}, var={x_var}\n r^2={r ** 2}')
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


def exponential(x, a, b, c):
    return a * np.exp(x-b) + c


def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


def cubic(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def bellcurve(x, a, mu, sigma, offset):
    return offset + a * np.exp(-((x - mu) / sigma) ** 2)


def quartic(x, a, b, c, d, e):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e


def mean_square_error(function, arg_opt: List[float], xdata: List[float], ydata: List[float]) -> float:
    residuals = ydata - function(xdata, *arg_opt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def find_best_curve(df: pd.DataFrame, col_x: str, col_y: str):
    """
    does the curve fitting for an exponential, a quadratic, a cubic and a quartic function, returns them as a dict.
    :param df:
    :param col_x:
    :param col_y:
    :return:
    """
    popt_exp, pcov_exp = sp.optimize.curve_fit(exponential, df[col_x].values, df[col_y].values)
    popt_quad, pcov_quad = sp.optimize.curve_fit(quadratic, df[col_x].values, df[col_y].values, p0=[1, 1, 1])
    popt_cube, pcov_cube = sp.optimize.curve_fit(cubic, df[col_x].values, df[col_y].values, p0=[1, 1, 1, 1])
    popt_quart, pcov_quart = sp.optimize.curve_fit(quartic, df[col_x].values, df[col_y].values, p0=[1, 1, 1, 1, 1])

    print(f'exp {popt_exp}, '
          f'r^2={mean_square_error(exponential, arg_opt=popt_exp, xdata=df[col_x].values, ydata=df[col_y].values)}')

    print(
        f'quad {popt_quad}, '
        f'r^2={mean_square_error(quadratic, arg_opt=popt_quad, xdata=df[col_x].values, ydata=df[col_y].values)}')

    print(
        f'cube {popt_cube}, '
        f'r^2={mean_square_error(cubic, arg_opt=popt_cube, xdata=df[col_x].values, ydata=df[col_y].values)}')

    print(
        f'quartic {popt_quart}, '
        f'r^2={mean_square_error(quartic, arg_opt=popt_quart, xdata=df[col_x].values, ydata=df[col_y].values)}')
    return {'exp': popt_exp, 'quad': popt_quad, 'cube': popt_cube, 'quart': popt_quart}


def pdf_potentially_relevant_with_fitted_curves(df: pd.DataFrame, savename='dienerds-lines-without-garbage.pdf'):
    with PdfPages(savename) as pdf:
        for i in columns:
            print(f'i = {i}')
            for j in columns:
                # for j in ['/Section D-D/Circle 1/D']:
                print(f'j = {j}')
                # this section skips unnecessary diagrams which just increase the .pdf file size
                if i == j:
                    continue
                elif i == 'time':
                    continue
                elif j == 'time':
                    continue

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

                    best_curves = find_best_curve(df=df, col_x=i, col_y=j)

                    plot2 = sns.lineplot(data=df, x=df[i], y=exponential(x, *best_curves['exp']), color='g', ax=ax)
                    r, p = sp.stats.pearsonr(df[i], df[j])
                    plt.xlabel(f'{i}, mean={x_mean}, var={x_var}\n r^2={r ** 2}')
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


if __name__ == '__main__':
    df = pd.read_csv('../data/clean.csv', sep=';')
    columns = df.columns
    print(columns)

    fig = px.scatter(df, x='/Section D-D/Circle 1/D', y='/Section E-E/Circle 10/D', color='nr')
    fig.show()
    fig.write_html("../data/example.html")

    #find_best_curve(df=df, col_x='/Section E-E/Circle 10/D', col_y='/Section D-D/Circle 10/D')
    #pdf_potentially_relevant_diagr(df=df)

    pdf_potentially_relevant_with_fitted_curves(df=df)



