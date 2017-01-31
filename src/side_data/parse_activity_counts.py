import pandas as pd

for year in range(2012, 2017):
    file_in = '{}.csv'.format(year)
    file_out = 'activity_{}.csv'.format(year)
    df = pd.read_csv(file_in)
    try:
        data = df.pivot(index='repo_name', columns='type', values='f0_')
    except KeyError as e:
        data = df.pivot(index='projects_repo_name', columns='type', values='f0_')
    data.fillna(0, inplace=True)
    data.to_csv(file_out)
