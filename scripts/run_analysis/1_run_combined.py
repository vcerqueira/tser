from plotnine import *
import matplotlib

matplotlib.use('TkAgg')

import numpy as np
import pandas as pd

from codebase.workflows.analysis import AnalysisWorkflow

df = pd.read_csv('assets/results/consolidated_full_horizon.csv')
df['Dataset'].value_counts()
df = df.drop('Dataset', axis=1)

# Average Rank
avg_rank = df.rank(axis=1).mean().sort_values()
avg_rank_df = avg_rank.reset_index()
avg_rank_df.columns = ['Model', 'Avg. Rank']
avg_rank_df['Model'] = pd.Categorical(avg_rank_df['Model'], categories=avg_rank_df['Model'])

MY_THEME = theme_538(base_family='Palatino', base_size=12) + \
           theme(plot_margin=0.01,
                 axis_text_y=element_text(size=10),
                 panel_background=element_rect(fill='white'),
                 plot_background=element_rect(fill='white'),
                 strip_background=element_rect(fill='white'),
                 legend_background=element_rect(fill='white'),
                 axis_text_x=element_text(size=10))

ar_plot = ggplot(avg_rank_df, aes(x='Model', y='Avg. Rank')) + \
          MY_THEME + \
          geom_bar(position='dodge',
                   stat='identity',
                   fill=AnalysisWorkflow.COLOR) + \
          labs(x='', y='Avg. Rank')

# ar_plot.save(f'avg_rank_all.pdf', height=5, width=12)

# % Diff to Baselines
# pd_to_glb = AnalysisWorkflow.get_diff_df(df, 'Global')
pd_to_glb = AnalysisWorkflow.get_pd_df(df, 'Global')
# pd_to_ind = AnalysisWorkflow.get_diff_df(df, 'Local')
pd_to_ind = AnalysisWorkflow.get_pd_df(df, 'Local')

pd_data = pd.concat([pd_to_ind, pd_to_glb], axis=0, ignore_index=True)
pd_data = pd_data.drop(['Global', 'Local'], axis=1)

pdm_df = pd_data.melt('Baseline')
pdm_df['value'] = np.sign(pdm_df['value']) * np.log(np.abs(pdm_df['value']) + 1)

pd_plot = ggplot(pdm_df,
                 aes(x='variable',
                     y='value',
                     fill='Baseline')) + \
          MY_THEME + \
          geom_boxplot(width=.8) + \
          geom_hline(yintercept=0,
                     linetype='dashed',
                     color='steelblue',
                     size=1.1,
                     alpha=0.9) + \
          labs(x='', y='% Diff in MASE')

# pd_plot.save(f'pd_baseline_all.pdf', height=5, width=12)

# Proportion wrt Local
pd_to_ind2 = pd_to_ind.drop('Baseline', axis=1)

side_probs = AnalysisWorkflow.get_side_probs(pd_to_ind2, 'Local', AnalysisWorkflow.ROPE)
side_probs_m = side_probs.reset_index().melt('index')
side_probs_m = side_probs_m.rename(columns={'variable': 'Outcome'})
side_probs_m['Outcome'] = pd.Categorical(side_probs_m['Outcome'],
                                         categories=['Local wins',
                                                     'draw',
                                                     'Local loses'])

plot_probs = \
    ggplot(side_probs_m, aes(x='index',
                             y='value',
                             fill='Outcome')) + \
    geom_col(position='fill') + \
    MY_THEME + \
    scale_fill_brewer(type='div', palette=5) + \
    labs(x='', y='Proportion of probability')

# plot_probs.save(f'probs_to_local_all.pdf', height=6, width=12)

# Proportion wrt Global
pd_to_glb2 = pd_to_glb.drop('Baseline', axis=1)

side_probs = AnalysisWorkflow.get_side_probs(pd_to_glb2, 'Global', AnalysisWorkflow.ROPE)
side_probs = AnalysisWorkflow.get_side_probs(pd_to_glb2, 'Global', 5)
side_probs_m = side_probs.reset_index().melt('index')
side_probs_m['variable'] = pd.Categorical(side_probs_m['variable'],
                                          categories=['Global wins',
                                                      'draw',
                                                      'Global loses'])

plot_probs = \
    ggplot(side_probs_m, aes(x='index',
                             y='value',
                             fill='variable')) + \
    geom_col(position='fill') + \
    theme_classic(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.02,
          axis_text=element_text(size=12),
          legend_text=element_text(size=14),
          legend_title=element_blank(),
          legend_position='top') + \
    scale_fill_brewer(type='div', palette=5) + \
    labs(x='', y='Proportion of probability')

# plot_probs.save(f'probs_to_global_all.pdf', height=5, width=12)

# Bayesian statistical analysis
from baycomp.multiple import SignedRankTest, SignTest

pd_to_glb2 = pd_to_glb.drop('Baseline', axis=1)
pd_to_glb2['Global'] = 0

ba_results = []
for col in pd_to_glb2:
    print(col)
    if col == 'Global':
        continue
    # col = 'Local'
    probs = SignTest.probs(x=pd_to_glb2[col],
                           y=pd_to_glb2['Global'],
                           rope=5,
                           nsamples=10000)

    ba_results.append({
        'Global wins': probs[0],
        'draw': probs[1],
        'Global loses': probs[2],
    })

side_probs = pd.DataFrame(ba_results, index=pd_to_glb2.columns[:-1])
print(side_probs.to_latex(caption='sadsasda',
                          escape=False,
                          label='lasasdsa'))

# side_probs = AnalysisWorkflow.get_side_probs(pd_to_glb2, 'Global', AnalysisWorkflow.ROPE)
# side_probs = AnalysisWorkflow.get_side_probs(pd_to_glb2, 'Global', 5)
side_probs_m = side_probs.reset_index().melt('index')
side_probs_m['variable'] = pd.Categorical(side_probs_m['variable'],
                                          categories=['Global wins',
                                                      'draw',
                                                      'Global loses'])

plot_probs = \
    ggplot(side_probs_m, aes(x='index',
                             y='value',
                             fill='variable')) + \
    geom_col(position='fill') + \
    theme_classic(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.02,
          axis_text=element_text(size=12),
          legend_text=element_text(size=14),
          legend_title=element_blank(),
          legend_position='top') + \
    scale_fill_brewer(type='div', palette=5) + \
    labs(x='', y='Proportion of probability')

plot_probs.save(f'test_global.pdf', height=5, width=12)

# Local vs Global

ranks_df = df.rank(axis=1)
ranks_df['Best_BL'] = \
    (ranks_df['Local'] < ranks_df['Global']).map({True: 'When Local is better than Global',
                                                  False: 'When Global is better than Local'})

avg_rank_by_bl = ranks_df.groupby('Best_BL').mean().T
avg_rank_df2 = avg_rank_by_bl.reset_index().melt('index')
avg_rank_df2['index'] = pd.Categorical(avg_rank_df2['index'], categories=avg_rank_df['Model'])

ar_plot2 = ggplot(avg_rank_df2, aes(x='index', y='value')) + \
           MY_THEME + \
           geom_bar(position='dodge',
                    stat='identity',
                    fill=AnalysisWorkflow.COLOR) + \
           labs(x='', y='Avg. Rank') + \
           facet_wrap('~ Best_BL', ncol=1)

# ar_plot2.save(f'avg_rank_all_by_bl.pdf', height=7, width=13)
