import os
import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
from plotnine import *

from codebase.workflows.analysis import AnalysisWorkflow

DATA_DIR = 'assets/results/sensitivity'
DATASET = 'nn5_daily_without_missing'

all_files = os.listdir(DATA_DIR)
max_id = np.max([AnalysisWorkflow.get_ts_id(x) for x in all_files])

perf = []
for i in range(max_id):
    df = pd.read_csv(f'{DATA_DIR}/{DATASET}_TS{i}.csv', index_col='Unnamed: 0')

    perf.append(df.mean())

perf_df = pd.DataFrame(perf).drop('Horizon', axis=1)

print(perf_df.rank(axis=1).mean())
print(perf_df.shape)
print(perf_df.rank(axis=1).std())

avg_rank = perf_df.rank(axis=1).mean()
avg_rank_df = avg_rank.reset_index()
avg_rank_df.columns = ['Variant', 'Avg. Rank']
avg_rank_df['Variant'] = [
    'Global',
    'S. Ratio 1',
    'S. Ratio 2',
    'S. Ratio 3',
    'S. Ratio 4',
    'S. Ratio 5',
    'S. Ratio 6',
    'S. Ratio 7',
    'S. Ratio 8',
    'S. Ratio 9',
    'S. Ratio 10',
    'S. Ratio 11',
    'S. Ratio 12',
    'S. Ratio 13',
    'S. Ratio 14',
    'S. Ratio 15',
    'S. Ratio 16',
    'S. Ratio 17',
    'S. Ratio 18',
    'Balanced'
]

avg_rank_df['Variant'] = pd.Categorical(avg_rank_df['Variant'],
                                        categories=avg_rank_df['Variant'])

MY_THEME = theme_538(base_family='Palatino', base_size=12) + \
           theme(plot_margin=0.02,
                 axis_text_y=element_text(size=10),
                 panel_background=element_rect(fill='white'),
                 plot_background=element_rect(fill='white'),
                 strip_background=element_rect(fill='white'),
                 legend_background=element_rect(fill='white'),
                 axis_text_x=element_text(size=10, angle=80))

ar_plot = ggplot(avg_rank_df, aes(x='Variant', y='Avg. Rank')) + \
          MY_THEME + \
          geom_bar(position='dodge',
                   stat='identity',
                   fill=AnalysisWorkflow.COLOR) + \
          labs(x='', y='Avg. Rank')

ar_plot.save(f'avg_rank_sampling_ratios.pdf', height=5, width=12)
