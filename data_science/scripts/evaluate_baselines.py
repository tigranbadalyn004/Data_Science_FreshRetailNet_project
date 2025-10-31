# scripts/evaluate_baselines.py
import yaml
import pandas as pd
from src.data.feature_engineering import create_time_features, create_lag_features, fill_na_features
from src.models.baseline.naive_models import SimpleNaiveBaseline, SeasonalNaiveBaseline, MovingAverageBaseline
from src.models.baseline.linear_models import LinearBaseline
from src.models.baseline.tree_models import RandomForestBaseline, XGBoostBaseline
from src.models.baseline.ensemble_models import EnsembleBaseline
from src.evaluation.metrics import evaluate_df, evaluate_business_metrics

# Load config
with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

train = pd.read_csv(cfg['data']['train_path'])
eval = pd.read_csv(cfg['data']['eval_path'])

# Feature engineering
train = create_time_features(train, datetime_col=cfg['data']['datetime_column'])
eval = create_time_features(eval, datetime_col=cfg['data']['datetime_column'])
lags = cfg['preprocessing']['lag_periods']
group_cols = ['store_id', 'product_id']
train = create_lag_features(train.sort_values(group_cols + [cfg['data']['datetime_column']]), group_cols, cfg['data']['target_column'], lags)
eval = create_lag_features(eval.sort_values(group_cols + [cfg['data']['datetime_column']]), group_cols, cfg['data']['target_column'], lags)

lag_cols = [f"{cfg['data']['target_column']}_lag_{l}" for l in lags]
fill_value = cfg['preprocessing'].get('fillna_value', 0.0)
train = fill_na_features(train, lag_cols, fill_value)
eval = fill_na_features(eval, lag_cols, fill_value)

feature_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'] + lag_cols

# Initialize models
models = {
    "SimpleNaive": SimpleNaiveBaseline(target_col=cfg['data']['target_column']),
    "SeasonalNaive": SeasonalNaiveBaseline(target_col=cfg['data']['target_column'], seasonal_period=168),
    "MovingAverage": MovingAverageBaseline(feature_cols=feature_cols, target_col=cfg['data']['target_column'], windows=[6,24,168]),
    "LinearRegression": LinearBaseline(feature_cols=feature_cols, target_col=cfg['data']['target_column']),
    "RandomForest": RandomForestBaseline(feature_cols=feature_cols, target_col=cfg['data']['target_column']),
    "XGBoost": XGBoostBaseline(feature_cols=feature_cols, target_col=cfg['data']['target_column']),
    # Ensemble could be added if needed
}

results = {}
for name, model in models.items():
    print(f"Training & evaluating: {name}")
    model.fit(train)
    eval['prediction'] = model.predict(eval)
    core_metrics = evaluate_df(eval)
    business_metrics = evaluate_business_metrics(eval)
    results[name] = {"core": core_metrics, "business": business_metrics}

# Save comparison
comparison_df = pd.DataFrame({k: v['core'] for k,v in results.items()}).T
comparison_df.to_csv("baseline_models_comparison.csv")
print("✅ Comparison saved to baseline_models_comparison.csv")
