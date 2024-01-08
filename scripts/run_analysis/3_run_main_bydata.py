import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv('assets/results/consolidated_full_horizon.csv')

df_grouped = df.groupby('Dataset')
avg_rank = {}
for name, ds in df_grouped:
    print(name)
    print(ds)
    avg_rank[name] = ds.drop('Dataset', axis=1).rank(axis=1).mean()

avg_rank_df = pd.DataFrame(avg_rank).T.round(1)

print(avg_rank_df.T.to_latex(caption='sadsada', label='lasadsa',
                             escape=False,
                             header=['\\rotatebox{90}{' + c + '}' for c in avg_rank_df.T.columns]))

print(avg_rank_df.T.to_latex(caption='sadsada',
                             escape=False,
                             label='lasadsa'))
