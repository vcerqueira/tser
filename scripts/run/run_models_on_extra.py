import os
import warnings

import pandas as pd

from codebase.methods.lightgbm import LightGBMOptim, BEST_PARAMS
from codebase.workflows.modeling import ModellingWorkflow
from codebase.workflows.data_reader import DataWorkflow

from config import (FORECASTING_HORIZON,
                    N_LAGS,
                    TEST_SIZE)

warnings.filterwarnings("ignore")

DS = 'nn5_daily_without_missing'
OUTPUT_DIR = 'assets/results/extra_by_series'

ds = DataWorkflow.get_from_gluonts(DS)

train, test, averages = \
    DataWorkflow.train_test_split(dataset=ds,
                                  n_lags=N_LAGS,
                                  horizon=FORECASTING_HORIZON[DS],
                                  test_size=TEST_SIZE)

ts_names = [*train]

for name in ts_names:
    # name = 'TS0'
    filepath = f'{OUTPUT_DIR}/{DS}_{name}.csv'

    if os.path.exists(filepath):
        continue
    else:
        pd.DataFrame().to_csv(filepath)

    print(f'MODELLING SERIES: {name}')
    mod = LightGBMOptim(params=BEST_PARAMS[DS])

    scores_df = \
        ModellingWorkflow.performance_estimation_extra(algorithm=mod,
                                                       train=train,
                                                       test=test,
                                                       averages=averages,
                                                       series_name=name)

    print(scores_df.mean())

    scores_df.to_csv(filepath, index=False)
