N_LAGS = 10
TEST_SIZE = 0.3

FORECASTING_HORIZON = {
    'nn5_daily_without_missing': 14,
    'solar-energy': 24,
    'traffic_nips': 24,
    'electricity_nips': 24,
    'taxi_30min': 48,
    'rideshare_without_missing': 24,
    'm4_hourly': 24,
    'm4_weekly': 12,
    'm4_daily': 24,
}

FREQUENCY = {
    'nn5_daily_without_missing': 7,
    'solar-energy': 24,
    'traffic_nips': 24,
    'electricity_nips': 24,
    'taxi_30min': 48,
    'rideshare_without_missing': 24,
    'm4_daily': 7,
    'm4_hourly': 24,
    'm4_weekly': 52,
}

ALL_DATASETS = [*FORECASTING_HORIZON]
