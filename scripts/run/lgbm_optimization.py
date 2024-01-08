from pprint import pprint
import pandas as pd

from codebase.methods.lightgbm import LightGBMOptim
from codebase.workflows.data_reader import DataWorkflow
from config import N_LAGS, ALL_DATASETS

best_params_by_ds = {}
for ds_name in ALL_DATASETS:
    print(ds_name)

    ds = DataWorkflow.get_from_gluonts(ds_name)

    train, *_ = \
        DataWorkflow.train_test_split(dataset=ds,
                                      n_lags=N_LAGS,
                                      horizon=1,
                                      test_size=0.3)

    train_df = pd.concat(train).reset_index(drop=True)

    X = train_df.drop('Series(t+1)', axis=1)
    y = train_df['Series(t+1)'].values

    best_params = LightGBMOptim.optimize_params(X, y)

    best_params_by_ds[ds_name] = best_params

pprint(best_params_by_ds)
