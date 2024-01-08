import copy
from typing import Dict

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import NearMiss

from codebase.common.metrics import MASE
from codebase.common.errors import NotEnoughDataError
from codebase.methods.baselines import LocalModel, GlobalModel
from codebase.methods.tser import TSERModel

MIN_N_OBSERVATIONS = 30
N_NEIGHBORS = 10


class ModellingWorkflow:

    @staticmethod
    def error_by_horizon(preds: pd.DataFrame,
                         insample: pd.DataFrame,
                         actual: pd.DataFrame):
        error = {col: MASE(insample[col].values,
                           actual[col].values,
                           preds[col].values)
                 for col in actual.columns}

        return error

    @staticmethod
    def get_xy(train: Dict,
               test: Dict,
               averages: Dict,
               series_name: str):
        test_series = test[series_name]
        train_series = train[series_name]

        if test_series.shape[0] < MIN_N_OBSERVATIONS or train_series.shape[0] < MIN_N_OBSERVATIONS:
            raise NotEnoughDataError('Minimum data size not met.')

        avg = averages[series_name]

        y_cols = TSERModel.get_target_columns(test_series.columns)

        X_ts = test_series.drop(y_cols, axis=1)
        Y_ts = test_series[y_cols]

        Y_ts_or = Y_ts.copy()
        Y_ts_or *= avg

        Y_tr = train_series[y_cols]
        Y_tr_or = Y_tr.copy()
        Y_tr_or *= avg

        return X_ts, Y_ts_or, Y_tr_or, avg

    @staticmethod
    def get_models(algorithm, series_name: str):

        global_model = GlobalModel(model=algorithm)
        local_model = LocalModel(model=algorithm, target_entity=series_name)

        tser_smote = \
            TSERModel(model=algorithm,
                      target_entity=series_name,
                      resampler=SMOTE(sampling_strategy='not majority', k_neighbors=N_NEIGHBORS),
                      keep_local_only=False)

        tser_adasyn = \
            TSERModel(model=algorithm,
                      target_entity=series_name,
                      resampler=ADASYN(sampling_strategy='not majority', n_neighbors=N_NEIGHBORS),
                      keep_local_only=False)

        tser_bsmote = \
            TSERModel(model=algorithm,
                      target_entity=series_name,
                      resampler=BorderlineSMOTE(sampling_strategy='not majority', k_neighbors=N_NEIGHBORS),
                      keep_local_only=False)

        tser_nm = \
            TSERModel(model=algorithm,
                      target_entity=series_name,
                      resampler=NearMiss(sampling_strategy='all', n_neighbors=N_NEIGHBORS),
                      keep_local_only=False)

        models = {
            'Global': global_model,
            'Local': local_model,
            'TSER(SMOTE)': tser_smote,
            'TSER(ADASYN)': tser_adasyn,
            'TSER(BSMOTE)': tser_bsmote,
            'TSER(NM)': tser_nm,
        }

        return models

    @classmethod
    def performance_estimation(cls,
                               train: Dict,
                               X_test: pd.DataFrame,
                               Y_test: pd.DataFrame,
                               Y_insample: pd.DataFrame,
                               algorithm,
                               series_name: str,
                               series_avg: float):

        models = cls.get_models(algorithm=algorithm, series_name=series_name)

        scores = {}
        for method, model in models.items():
            print(method)
            train_ = copy.deepcopy(train)

            model.fit(train_)

            preds = model.predict(X_test.values)
            preds = pd.DataFrame(preds, columns=Y_test.columns)
            preds *= series_avg

            error = cls.error_by_horizon(preds, insample=Y_insample, actual=Y_test)

            scores[method] = error

        scores_df = pd.DataFrame(scores).reset_index(drop=True)
        scores_df['Horizon'] = np.arange(1, scores_df.shape[0] + 1)

        return scores_df

    @classmethod
    def performance_estimation_extra(cls,
                                     train: Dict,
                                     test: Dict,
                                     algorithm,
                                     series_name,
                                     averages):

        models = cls.get_models(algorithm=algorithm, series_name=series_name)

        scores = {}
        for method, model in models.items():
            # print(method)
            train_ = copy.deepcopy(train)

            model.fit(train_)

            scores_on_extra_series = []
            for name in train:
                if name == series_name:
                    continue

                try:
                    X_ts, Y_ts_or, Y_tr_or, avg = \
                        cls.get_xy(train=train,
                                   test=test,
                                   averages=averages,
                                   series_name=name)
                except NotEnoughDataError:
                    continue

                preds = model.predict(X_ts.values)
                preds = pd.DataFrame(preds, columns=Y_ts_or.columns)
                preds *= avg

                error = cls.error_by_horizon(preds,
                                             insample=Y_tr_or,
                                             actual=Y_ts_or)

                scores_on_extra_series.append(error)

            avg_score = pd.DataFrame(scores_on_extra_series).mean()
            avg_score.index = np.arange(1, len(avg_score) + 1)

            scores[method] = avg_score

        scores_df = pd.DataFrame(scores).reset_index(drop=True)
        scores_df['Horizon'] = np.arange(1, scores_df.shape[0] + 1)

        return scores_df


class SensitivityOnSampling(ModellingWorkflow):

    @staticmethod
    def get_os_sampling_ratios(train, series_name):
        n_all = pd.concat(train).shape[0]
        n_tgt = train[series_name].shape[0]

        n_min = np.linspace(start=n_tgt, stop=n_all, num=20).astype(int)
        # sample_ratio = n_min / n_all

        sample_dict = [{0: n_all, 1: x} for x in n_min]

        return sample_dict

    @staticmethod
    def get_models(algorithm, series_name: str, sampling_strategy_list):

        tser_vars = {}
        for i, s_ratio in enumerate(sampling_strategy_list):
            # s_ratio = sampling_ratios[0]
            tser_v = \
                TSERModel(model=algorithm,
                          target_entity=series_name,
                          resampler=SMOTE(sampling_strategy=s_ratio, k_neighbors=N_NEIGHBORS),
                          keep_local_only=False)

            tser_vars[f'TSER_var{i}'] = tser_v

        return tser_vars

    @classmethod
    def performance_estimation(cls,
                               train: Dict,
                               X_test: pd.DataFrame,
                               Y_test: pd.DataFrame,
                               Y_insample: pd.DataFrame,
                               algorithm,
                               series_name: str,
                               series_avg: float):

        sampling_ratios = SensitivityOnSampling.get_os_sampling_ratios(train, series_name)

        models = cls.get_models(algorithm=algorithm, series_name=series_name, sampling_strategy_list=sampling_ratios)

        scores = {}
        for method, model in models.items():
            print(method)
            train_ = copy.deepcopy(train)

            model.fit(train_)

            preds = model.predict(X_test.values)
            preds = pd.DataFrame(preds, columns=Y_test.columns)
            preds *= series_avg

            error = cls.error_by_horizon(preds,
                                         insample=Y_insample,
                                         actual=Y_test)

            scores[method] = error

        scores_df = pd.DataFrame(scores).reset_index(drop=True)
        scores_df['Horizon'] = np.arange(1, scores_df.shape[0] + 1)

        return scores_df


class VariantAnalysis(ModellingWorkflow):

    @staticmethod
    def get_models(algorithm, series_name: str, n_all: int):
        global_model = GlobalModel(model=algorithm)
        local_model = LocalModel(model=algorithm, target_entity=series_name)

        tser_min = \
            TSERModel(model=algorithm,
                      target_entity=series_name,
                      resampler=SMOTE(sampling_strategy='not majority', k_neighbors=N_NEIGHBORS),
                      keep_local_only=False)

        tser_min_l = \
            TSERModel(model=algorithm,
                      target_entity=series_name,
                      resampler=SMOTE(sampling_strategy='not majority', k_neighbors=N_NEIGHBORS),
                      keep_local_only=True)

        tser_all = \
            TSERModel(model=algorithm,
                      target_entity=series_name,
                      resampler=SMOTE(sampling_strategy={0: int(n_all * 1.5), 1: n_all}, k_neighbors=N_NEIGHBORS),
                      keep_local_only=False)

        models = {
            'Global': global_model,
            'Local': local_model,
            'TSER': tser_min,
            'TSER(Local)': tser_min_l,
            'TSER(all)': tser_all,
        }

        return models

    @classmethod
    def performance_estimation(cls, train: Dict, X_test, Y_test, Y_insample, algorithm, series_name, series_avg):
        n_all = pd.concat(train).shape[0]

        models = cls.get_models(algorithm=algorithm, series_name=series_name, n_all=n_all)

        scores = {}
        for method, model in models.items():
            train_ = copy.deepcopy(train)

            model.fit(train_)

            preds = model.predict(X_test.values)
            preds = pd.DataFrame(preds, columns=Y_test.columns)
            preds *= series_avg

            error = cls.error_by_horizon(preds,
                                         insample=Y_insample,
                                         actual=Y_test)

            scores[method] = error

        scores_df = pd.DataFrame(scores).reset_index(drop=True)
        scores_df['Horizon'] = np.arange(1, scores_df.shape[0] + 1)

        return scores_df
