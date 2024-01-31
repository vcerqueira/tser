import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error as mae
from imblearn.over_sampling import SMOTE

from codebase.workflows.data_reader import DataWorkflow
from codebase.methods.lightgbm import LightGBMOptim

# loading dataset
ds = DataWorkflow.get_from_gluonts('nn5_daily_without_missing')

# train test splitting plus normalizations
train, test, averages = \
    DataWorkflow.train_test_split(dataset=ds, n_lags=7, horizon=7, test_size=0.3)

# concatenating individual series into a single dataframe for a global modeling approach
train_df = pd.concat(train)
test_df = pd.concat(test)

# X, y split
is_y = train_df.columns.str.contains('\+')
y_col_names = np.array(train_df.columns)[np.where(is_y)[0]].tolist()

X_train, Y_train = train_df.drop(y_col_names, axis=1), train_df[y_col_names]
X_test, Y_test = test_df.drop(y_col_names, axis=1), test_df[y_col_names]

# fitting + predictions
model = MultiOutputRegressor(LightGBMOptim(iters=2))

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
Y_pred = pd.DataFrame(Y_pred, index=Y_test.index)

# error by sample

sample_error = []
for i, r in Y_test.iterrows():
    # print(r)
    err_i = mae(Y_pred.loc[i], r)  # not the best metric
    sample_error.append(err_i)

test_df['error'] = sample_error

# checking the dist of error
test_df['error'].hist()

# oversampling large errors using 95 percentile using SMOTE

large_errors = (test_df['error'] > test_df['error'].quantile(0.95)).astype(int)

resampled_data, err_class = SMOTE().fit_resample(test_df.drop(columns='error'), large_errors)

large_errors_df = resampled_data[err_class == 1]

# analyse large_errors_df
