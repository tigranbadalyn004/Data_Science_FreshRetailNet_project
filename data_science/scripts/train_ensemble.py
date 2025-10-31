# scripts/train_ensemble.py
import yaml
import pandas as pd
from src.data.feature_engineering import create_time_features, create_lag_features, fill_na_features
from src.models.baseline.ensemble_models import EnsembleBaseline
from src.models.baseline.linear_models import LinearBaseline
from src.models.baseline.tree_models import XGBoostBaseline

# Load config
with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

train = pd.read_csv(cfg['data']['train_path'])
eval = pd.read_csv(cfg['data']['eval_path'])

# Create time features
train = create_time_features(train, datetime_col=cfg['data']['datetime_column'])
eval = create_time_features(eval, datetime_col=cfg['data']['datetime_column'])

# Lag features
lags = cfg['models']['baseline']['ensemble']['features']['lags']
group_cols = ['store_id', 'product_id']
train = create_lag_features(train.sort_values(group_cols + [cfg['data']['datetime_column']]), group_cols, cfg['data']['target_column'], lags)
eval = create_lag_features(eval.sort_values(group_cols + [cfg['data']['datetime_column']]), group_cols, cfg['data']['target_column'], lags)

# Fill NA
lag_cols = [f"{cfg['data']['target_column']}_lag_{l}" for l in lags]
fill_value = cfg['models']['baseline']['ensemble']['features'].get('fillna_value', 0.0)
train = fill_na_features(train, lag_cols, fill_value)
eval = fill_na_features(eval, lag_cols, fill_value)

# Feature columns
feature_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'] + lag_cols

# Initialize base models
linear_model = LinearBaseline(feature_cols=feature_cols, target_col=cfg['data']['target_column'])
linear_model.fit(train)
xgb_model = XGBoostBaseline(feature_cols=feature_cols, target_col=cfg['data']['target_column'])
xgb_model.fit(train)

base_models = {"linear": linear_model, "xgboost": xgb_model}

# Instantiate Ensemble
ensemble_cfg = cfg['models']['baseline']['ensemble']
ensemble_model = EnsembleBaseline(
    feature_cols=feature_cols,
    target_col=cfg['data']['target_column'],
    base_models=base_models,
    meta_model_type=ensemble_cfg['meta_model']['type'],
    test_size=ensemble_cfg['test_size'],
    random_state=ensemble_cfg['random_state'],
    fillna_value=fill_value
)

# Fit and predict
print("Fitting Ensemble baseline...")
ensemble_model.fit(train)

print("Predicting on eval...")
preds = ensemble_model.predict(eval)
eval['prediction'] = preds
eval[['store_id','product_id',cfg['data']['datetime_column'],'prediction']].to_csv('ensemble_baseline_preds.csv', index=False)
print("Predictions saved to ensemble_baseline_preds.csv")
