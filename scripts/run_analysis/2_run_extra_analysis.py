import os
import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import pandas as pd

from plotnine import *

from codebase.workflows.analysis import AnalysisWorkflow

EXTRA_DATA_DIR = 'assets/results/extra_by_series'
ORIGINAL_DATA_DIR = 'assets/results/by_series'
DATASET = 'nn5_daily_without_missing'

MY_THEME = theme_538(base_family='Palatino', base_size=12) + \
           theme(plot_margin=0.01,
                 axis_text_y=element_text(size=10),
                 panel_background=element_rect(fill='white'),
                 plot_background=element_rect(fill='white'),
                 strip_background=element_rect(fill='white'),
                 legend_background=element_rect(fill='white'),
                 axis_text_x=element_text(size=10))


all_files = os.listdir(EXTRA_DATA_DIR)
max_id = np.max([AnalysisWorkflow.get_ts_id(x) for x in all_files])

perf_diff_l = []
for i in range(max_id):
    on_entity = pd.read_csv(f'{ORIGINAL_DATA_DIR}/{DATASET}_TS{i}.csv', index_col='Unnamed: 0')
    on_extra = pd.read_csv(f'{EXTRA_DATA_DIR}/{DATASET}_TS{i}.csv')

    entity = on_entity#.mean()
    extra = on_extra#.mean()
    # perf_diff = 100 * ((extra-entity) / entity)
    # perf_diff = (100 * ((on_extra-on_entity) / on_entity)).mean()
    perf_diff = (extra-entity).median()

    perf_diff_l.append(perf_diff)

perf_diff_df = pd.DataFrame(perf_diff_l).drop(['Horizon','Unnamed: 0'], axis=1)
perf_diff_df = perf_diff_df.loc[:,~perf_diff_df.columns.str.startswith('TSEPR')]

plot_df = perf_diff_df.mean().reset_index()
plot_df.columns = ['Method', 'value']
print(plot_df)

pd_plot = ggplot(plot_df, aes(x='Method', y='value')) + \
          MY_THEME + \
          geom_bar(position='dodge',
                   stat='identity',
                   fill='#2c5f78') + \
          labs(x='', y='Average MASE Diff')

print(pd_plot)
pd_plot.save(f'avg_pd_on_extra.pdf', height=5, width=10)

