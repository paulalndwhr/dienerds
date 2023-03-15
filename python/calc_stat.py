import pandas as pd
import statistics

df = pd.read_csv('../data/teammessung.csv', sep=';')

#print(df)

dark = df.loc[df['comment'] == 'dark_M1_val']
sun = df.loc[df['comment'] == 'sunlight_M1_val']
art = df.loc[df['comment'] == 'artilight_M1_val']

#print(dark)
#print(sun)
#print(art)

for c in ["/Section D-D/Circle 1/D", 
            "/Section D-D/Circle 9/D",
            "/Section D-D/Circle 10/D",
            "/Section E-E/Circle 1/D",
            "/Section E-E/Circle 9/D",
            "/Section E-E/Circle 10/D",
            "/Cylinder (5)/D"]:
    print(c)
    characteristic = dark[c]
    print(statistics.mean(characteristic))
    print(statistics.stdev(characteristic))
    
    characteristic = sun[c]
    print(statistics.mean(characteristic))
    print(statistics.stdev(characteristic))

    characteristic = art[c]
    print(statistics.mean(characteristic))
    print(statistics.stdev(characteristic))