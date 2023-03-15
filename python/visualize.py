import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import plotly.express as px

df = pd.read_csv('../data/clean.csv', sep=';')

columns = df.columns
print(columns)
# with PdfPages('dienerds.pdf') as pdf:
#     for i in columns:
#         print(f'i = {i}')
#         for j in columns:
#         # for j in ['/Section D-D/Circle 1/D']:
#             print(f'j = {j}')
#             plt.figure(figsize=(len(columns), len(columns)))
#             plt.scatter(df[i], df[j], color='b', alpha=0.05)
#             plt.xlabel(f'{i}')
#             plt.ylabel(f'{j}')
#             plt.title(f'{i} gegen {j}')
#             plt.plot()
#             pdf.savefig()
#             plt.close

fig = px.scatter(df, x='/Section D-D/Circle 1/D', y='/Section E-E/Circle 10/D', color='nr')
fig.show()
fig.write_html("../data/example.html")
# import datetime
# import numpy as np
# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.pyplot as plt
#
# # Create the PdfPages object to which we will save the pages:
# # The with statement makes sure that the PdfPages object is closed properly at
# # the end of the block, even if an Exception occurs.
# with PdfPages('multipage_pdf.pdf') as pdf:
#     plt.figure(figsize=(3, 3))
#     plt.plot(range(7), [3, 1, 4, 1, 5, 9, 2], 'r-o')
#     plt.title('Page One')
#     pdf.savefig()  # saves the current figure into a pdf page
#     plt.close()
#
#     # if LaTeX is not installed or error caught, change to `False`
#     plt.rcParams['text.usetex'] = True
#     plt.figure(figsize=(8, 6))
#     x = np.arange(0, 5, 0.1)
#     plt.plot(x, np.sin(x), 'b-')
#     plt.title('Page Two')
#     pdf.attach_note("plot of sin(x)")  # attach metadata (as pdf note) to page
#     pdf.savefig()
#     plt.close()
#
#     plt.rcParams['text.usetex'] = False
#     fig = plt.figure(figsize=(4, 5))
#     plt.plot(x, x ** 2, 'ko')
#     plt.title('Page Three')
#     pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
#     plt.close()

    # # We can also set the file's metadata via the PdfPages object:
    # d = pdf.infodict()
    # d['Title'] = 'Multipage PDF Example'
    # d['Author'] = 'Jouni K. Sepp\xe4nen'
    # d['Subject'] = 'How to create a multipage pdf file and set its metadata'
    # d['Keywords'] = 'PdfPages multipage keywords author title subject'
    # d['CreationDate'] = datetime.datetime(2009, 11, 13)
    # d['ModDate'] = datetime.datetime.today()