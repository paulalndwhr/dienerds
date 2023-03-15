import pandas as pd
import json

def isValid(line):
    js = json.loads(line[4]) # column Request
    characteristics = js["PartState"]["Characteristics"]

    for i in range(1, 8):
        c = characteristics[i]
        if(c["Actual"] == None):
            return False

    return True

df = pd.read_csv('../data/export.csv')

df = df.values.tolist()

f = open("../data/clean.csv", "w")

f.write("time;nr;/Section D-D/Circle 1/D;/Section D-D/Circle 9/D;/Section D-D/Circle 10/D;/Section E-E/Circle 1/D;/Section E-E/Circle 9/D;/Section E-E/Circle 10/D;/Cylinder (5)/D;team;comment\n") # alle header

comments = set()

for line in df:
    if not isValid(line):
        continue

    #print(line)
    f.write(line[0]) # time
    f.write(";")
    f.write(str(line[1])) # nr

    js = json.loads(line[4]) # column Request
    characteristics = js["PartState"]["Characteristics"]
    for i in range(1, 8):
        f.write(";")
        c = characteristics[i]
        
        actual = float(c["Actual"])
        nominal = c["Nominal"]
        rel = nominal - actual

        f.write(str(rel))
        
    f.write(";")
    f.write(str(line[2])) # Team
    f.write(";")
    f.write(str(line[3])) # Comment

    comments.add(str(line[3]))

    f.write("\n")

print(comments)