
import csv

clean_file = "../data/clean.csv"

reader = csv.reader(open(clean_file ),delimiter=';')
filtered = filter(lambda p: '_F' not in p[10], reader)
csv.writer(open(r"../data/clean_filtered.csv",'w'),delimiter=';').writerows(filtered)



clean_file = "../data/clean1-12.csv"

reader = csv.reader(open(clean_file ),delimiter=';')
filtered = filter(lambda p: '_F' not in p[10], reader)
csv.writer(open(r"../data/clean1-12_filtered.csv",'w'),delimiter=';').writerows(filtered)