import os
import re
from typing import Dict, Tuple

import pandas as pd
from gluonts.dataset.repository.datasets import get_dataset
from sklearn.model_selection import train_test_split

from codebase.common.preprocess import series_as_supervised


class DataWorkflow:

    @staticmethod
    def read_results(filepath):
        r = pd.read_csv(filepath, index_col='Horizon')
        if r.shape[0] < 1:
            return r

        if 'Unnamed: 0' in r.columns:
            r = r.drop('Unnamed: 0', axis=1)

        return r

    @staticmethod
    def get_listdir(dir: str, dataset_name: str):

        files = os.listdir(dir)
        files = [x for x in files if x != '.DS_Store']
        files = [x for x in files
                 if bool(re.search(f'^{dataset_name}', x))]

        return files

    @staticmethod
    def get_from_gluonts(dataset_name: str) -> pd.DataFrame:
        dataset = get_dataset(dataset_name, regenerate=False)

        train = list(dataset.train)

        # ds_list = [pd.Series(ds['target'],
        #                      index=pd.date_range(start=ds['start'],
        #                                          freq=ds['start'].freq,
        #                                          periods=len(ds['target'])))
        #            for ds in train]

        ds_list = [pd.Series(ds['target'],
                             index=pd.date_range(start=ds['start'].to_timestamp(),
                                                 freq=ds['start'].freq,
                                                 periods=len(ds['target']))) for ds in train]

        df = pd.concat(ds_list, axis=1)
        df.columns = [f'TS{x}' for x in df.columns]

        if dataset_name == 'electricity_nips':
            df = df.loc[:, ~df.isna().any()]
            df = df.drop('TS322', axis=1)

        return df

    @staticmethod
    def train_test_split(dataset: pd.DataFrame,
                         n_lags: int,
                         horizon: int,
                         test_size: float) -> Tuple[Dict, Dict, Dict]:

        ts_names = dataset.columns.tolist()

        train_size = 1 - test_size
        split_ind = int(len(dataset.index) * train_size)

        train_end = dataset.index[split_ind]
        test_start = dataset.index[split_ind + 1]

        train, test, averages = {}, {}, {}
        for series_name in ts_names:
            # print(series_name)

            series = dataset[series_name]
            series.name = 'Series'

            avg = series[:train_end].mean()

            series /= avg

            train_series = series[:train_end]
            test_series = series[test_start:]

            train_df = series_as_supervised(train_series, n_lags=n_lags, horizon=horizon)
            test_df = series_as_supervised(test_series, n_lags=n_lags, horizon=horizon)

            train[series_name] = train_df
            test[series_name] = test_df
            averages[series_name] = avg

        return train, test, averages

    @staticmethod
    def ind_train_test_split(dataset: pd.DataFrame,
                             n_lags: int,
                             horizon: int,
                             test_size: float) -> Tuple[Dict, Dict, Dict]:
        # if the time series in the collection are independent, such as m4_hourly

        ts_names = dataset.columns.tolist()

        train, test, averages = {}, {}, {}
        for series_name in ts_names:
            # print(series_name)

            series = dataset[series_name].dropna()
            series.name = 'Series'

            train_s, test_s = train_test_split(series, test_size=test_size, shuffle=False)

            avg = train_s.mean()

            series /= avg

            train_df = series_as_supervised(train_s, n_lags=n_lags, horizon=horizon)
            test_df = series_as_supervised(test_s, n_lags=n_lags, horizon=horizon)

            train[series_name] = train_df
            test[series_name] = test_df
            averages[series_name] = avg

        return train, test, averages
