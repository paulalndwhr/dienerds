
import csv

clean_file = "../data/clean.csv"

reader = csv.reader(open(clean_file ),delimiter=';')
filtered = filter(lambda p: 'ohne Vibration am Scanner' == p[10], reader)
csv.writer(open(r"abx.csv",'w'),delimiter=';').writerows(filtered)