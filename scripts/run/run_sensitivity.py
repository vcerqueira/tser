import os
import warnings

import pandas as pd

from codebase.methods.lightgbm import LightGBMOptim, BEST_PARAMS
from codebase.workflows.modeling import SensitivityOnSampling
from codebase.workflows.data_reader import DataWorkflow
from codebase.common.errors import NotEnoughDataError

from config import (FORECASTING_HORIZON,
                    N_LAGS,
                    TEST_SIZE)

warnings.filterwarnings("ignore")

DS = 'nn5_daily_without_missing'
OUTPUT_DIR = 'assets/results/sensitivity'

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

    try:
        X_ts, Y_ts_or, Y_tr_or, avg = \
            SensitivityOnSampling.get_xy(train=train,
                                         test=test,
                                         averages=averages,
                                         series_name=name)
    except NotEnoughDataError:
        continue

    mod = LightGBMOptim(params=BEST_PARAMS[DS])

    scores_df = \
        SensitivityOnSampling.performance_estimation(algorithm=mod,
                                                     train=train,
                                                     X_test=X_ts,
                                                     Y_test=Y_ts_or,
                                                     Y_insample=Y_tr_or,
                                                     series_name=name,
                                                     series_avg=avg)

    scores_df.to_csv(filepath)
