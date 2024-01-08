import pandas as pd

from codebase.workflows.data_reader import DataWorkflow
from config import ALL_DATASETS

SERIES_RESULTS_DIR = 'assets/results/by_series'

file_scores = []
for ds in ALL_DATASETS:
    print(ds)
    # ds = 'm4_weekly'
    files = DataWorkflow.get_listdir(SERIES_RESULTS_DIR, ds)

    for file_ in files:
        fp = f'{SERIES_RESULTS_DIR}/{file_}'

        try:
            r = DataWorkflow.read_results(fp)
        except TypeError:
            continue

        r_avg = r.mean()
        r_avg.name = ds

        file_scores.append(r_avg)

df = pd.DataFrame(file_scores).reset_index()
df = df.rename(columns={'index': 'Dataset'})
df = df.loc[:, ~df.columns.str.startswith('TSEPR')]

print(df.drop('Dataset', axis=1).rank(axis=1).mean().sort_values())
print(df.groupby('Dataset').apply(lambda x: x.rank(axis=1).mean()).T.round(2).T)
print(df.groupby('Dataset').apply(lambda x: x.mean()).T.round(4))

df.to_csv('assets/results/consolidated_full_horizon.csv', index=False)
