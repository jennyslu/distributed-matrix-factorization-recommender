import csv
import json
import pandas as pd

df = pd.read_csv('all_languages.csv')
field_names = df['language_name'].values.tolist()
field_names.append('repo_name')

f_out = open('languages.csv', 'w', newline='')
writer = csv.DictWriter(f_out,
    fieldnames = field_names, restval = 0)
writer.writeheader()

i = 0
for i in range(4):
    input_file = 'languages{}.json'.format(i)
    with open(input_file) as f:
        for line in f:
            write_line = {}
            data = json.loads(line.rstrip())
            write_line['repo_name'] = data['repo_name']
            for x in data['language']:
                write_line[x['name']] = x['bytes']
            print(i)
            writer.writerow(write_line)
            i += 1
