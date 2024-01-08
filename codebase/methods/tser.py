import copy
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor


class TSER:

    def __init__(self,
                 resampler,
                 target_entity: str,
                 keep_local_only: bool = False):
        self.resampler = resampler
        self.target_entity = target_entity
        self.keep_local_only = keep_local_only

    def resample(self, train: Dict):
        """
        :param train: Training set. A collection of univariate time series in a tabular format
        :return: X, y, augmented
        """

        assert self.target_entity in train

        train_ = train.copy()

        train_extra = train_.copy()
        train_extra.pop(self.target_entity)

        train_extra_df = pd.concat(train_extra, axis=0).copy(deep=True)
        train_local_df = train_[self.target_entity]

        train_extra_df['class'] = 0
        train_local_df['class'] = 1

        train_df = pd.concat([train_extra_df, train_local_df], axis=0)
        y_aux = train_df['class']
        X_aux = train_df.drop('class', axis=1)

        train_df_aug, y_entity = self.resampler.fit_resample(X_aux, y_aux)

        if self.keep_local_only:
            train_df_aug = train_df_aug[y_entity > 0].reset_index(drop=True)

        return train_df_aug


class TSERModel(TSER):

    def __init__(self,
                 model,
                 resampler,
                 keep_local_only: bool,
                 target_entity: str):
        super().__init__(resampler, target_entity, keep_local_only)

        self.model = copy.deepcopy(MultiOutputRegressor(model))
        self.columns = []

    def fit(self, train: Dict):
        """
        :param train: Training set. A dict with all entities
        :return:
        """

        train_df = self.resample(train)

        y_cols = self.get_target_columns(train_df.columns)

        Y = train_df[y_cols]
        X = train_df.drop(y_cols, axis=1)

        self.columns = X.columns
        self.model.fit(X.values, Y)

    def predict(self, X_ts: pd.DataFrame):
        return self.model.predict(X_ts)

    @staticmethod
    def get_target_columns(col_names):
        is_y = col_names.str.contains('\+')

        y_col_names = np.array(col_names)[np.where(is_y)[0]].tolist()

        return y_col_names
