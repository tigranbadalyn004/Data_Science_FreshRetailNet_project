# scripts/train_tree.py
import yaml
import pandas as pd
from src.data.feature_engineering import create_time_features, create_lag_features, fill_na_features
from src.models.baseline.tree_models import RandomForestBaseline

# Load config
with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

train = pd.read_csv(cfg['data']['train_path'])
eval = pd.read_csv(cfg['data']['eval_path'])

# Time features
train = create_time_features(train, datetime_col=cfg['data']['datetime_column'])
eval = create_time_features(eval, datetime_col=cfg['data']['datetime_column'])

# Lags
lags = cfg['models']['baseline']['tree']['random_forest']['features']['lags']
group_cols = ['store_id', 'product_id']
train = create_lag_features(train.sort_values(group_cols + [cfg['data']['datetime_column']]), group_cols, cfg['data']['target_column'], lags)
eval = create_lag_features(eval.sort_values(group_cols + [cfg['data']['datetime_column']]), group_cols, cfg['data']['target_column'], lags)

# Fill NA
lag_cols = [f"{cfg['data']['target_column']}_lag_{l}" for l in lags]
fill_value = cfg['models']['baseline']['tree']['random_forest']['features'].get('fillna_value', 0.0)
train = fill_na_features(train, lag_cols, fill_value)
eval = fill_na_features(eval, lag_cols, fill_value)

# Feature columns
feature_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'] + lag_cols

# Instantiate model
rf_cfg = cfg['models']['baseline']['tree']['random_forest']
model = RandomForestBaseline(
    feature_cols=feature_cols,
    target_col=cfg['data']['target_column'],
    n_estimators=rf_cfg['n_estimators'],
    max_depth=rf_cfg['max_depth'],
    min_samples_split=rf_cfg['min_samples_split'],
    min_samples_leaf=rf_cfg['min_samples_leaf'],
    test_size=rf_cfg['training']['test_size'],
    random_state=rf_cfg['random_state'],
    fillna_value=fill_value
)

# Fit and predict
print("Fitting Random Forest baseline...")
model.fit(train)

print("Predicting on eval...")
preds = model.predict(eval)
eval['prediction'] = preds
eval[['store_id','product_id',cfg['data']['datetime_column'],'prediction']].to_csv('rf_baseline_preds.csv', index=False)
print("Predictions saved to rf_baseline_preds.csv")

