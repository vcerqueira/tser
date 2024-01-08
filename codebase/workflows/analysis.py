import pandas as pd
import re


class AnalysisWorkflow:
    COLOR = '#2c5f78'
    ROPE = 1

    @staticmethod
    def get_ts_id(file_name):
        ts_id = re.search('TS[0-9]?[0-9]?[0-9]?[0-9]', file_name).group()
        ts_number = int(re.sub('TS', '', ts_id))

        return ts_number

    @staticmethod
    def get_pd_df(results: pd.DataFrame, baseline: str) -> pd.DataFrame:
        assert baseline in results.columns

        pd_dict = {}
        for col in results.columns:
            if col == baseline:
                continue
            pd_dict[col] = 100 * ((results[col] - results[baseline]) / results[baseline])

        df_pd = pd.concat(pd_dict, axis=1)
        df_pd['Baseline'] = baseline

        return df_pd

    @staticmethod
    def get_side_probs(pd_df: pd.DataFrame, baseline: str, rope: float):
        ab_rope = (pd_df > rope).mean()
        bl_rope = (pd_df < -rope).mean()
        probs = pd.concat([
            bl_rope,
            1 - ab_rope - bl_rope,
            ab_rope,
        ], axis=1)

        probs.columns = [f'{baseline} loses', 'draw', f'{baseline} wins']

        return probs
