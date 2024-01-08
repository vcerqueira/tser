import pandas as pd

from imblearn.over_sampling import SMOTE
from codebase.workflows.data_reader import DataWorkflow
from codebase.methods.tser import TSER

ds = DataWorkflow.get_from_gluonts('nn5_daily_without_missing')

train, test, averages = \
    DataWorkflow.train_test_split(dataset=ds,
                                  n_lags=5,
                                  horizon=7,
                                  test_size=0.3)

resampler = TSER(resampler=SMOTE(k_neighbors=10), target_entity='TS0')

train_augmented = resampler.resample(train)

train_original = pd.concat(train)
