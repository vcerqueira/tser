import os
import numpy as np
import pandas as pd

from codebase.workflows.analysis import AnalysisWorkflow

DATA_DIR = 'assets/results/variants'
DATASET = 'nn5_daily_without_missing'

all_files = os.listdir(DATA_DIR)
max_id = np.max([AnalysisWorkflow.get_ts_id(x) for x in all_files])

perf = []
for i in range(max_id):
    df = pd.read_csv(f'{DATA_DIR}/{DATASET}_TS{i}.csv', index_col='Unnamed: 0')

    perf.append(df.mean())

perf_df = pd.DataFrame(perf).drop('Horizon', axis=1)

print(perf_df.rank(axis=1).mean())
print(perf_df.rank(axis=1).std())

avg_rank = perf_df.rank(axis=1).mean().round(2)
std_rank = perf_df.rank(axis=1).std().round(2)

avg_rank_d = avg_rank.to_dict()

df = pd.Series({k: f'{avg_rank_d[k]}\pm{std_rank[k]}' for k in avg_rank_d})
df = pd.DataFrame([df.values], columns=df.keys())

print(df.to_latex(caption='caption',
                  escape=False,
                  label='label'))
