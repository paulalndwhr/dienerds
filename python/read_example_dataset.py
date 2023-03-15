import pandas as pd
import json

df = pd.read_csv('../data/export.csv')

df = df.values.tolist()

f = open("../data/clean.csv", "w")

for line in df:
    #print(line)
    f.write(line[0]) # time
    f.write(";")
    f.write(str(line[1])) # nr


    js = json.loads(line[4]) # column Request
    characteristics = js["PartState"]["Characteristics"]
    for i in range(1, 8):
        f.write(";")
        c = characteristics[i];
        f.write(str(c["Actual"]))
        
    f.write(";")
    f.write(str(line[2])) # Team
    f.write(";")
    f.write(str(line[3])) # Comment
    
    f.write("\n")