# scripts/train_linear.py
import yaml
import pandas as pd
from src.data.feature_engineering import create_time_features, create_lag_features, fill_na_features
from src.models.baseline.linear_models import LinearBaseline

# load config
with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

train = pd.read_csv(cfg['data']['train_path'])
eval = pd.read_csv(cfg['data']['eval_path'])

# basic time features
train = create_time_features(train, datetime_col=cfg['data']['datetime_column'])
eval = create_time_features(eval, datetime_col=cfg['data']['datetime_column'])

# create lags (example: 1,7,168)
lags = cfg['models']['baseline']['linear']['features']['lags']
group_cols = ['store_id', 'product_id']
train = create_lag_features(train.sort_values(group_cols + [cfg['data']['datetime_column']]), group_cols, cfg['data']['target_column'], lags)
eval = create_lag_features(eval.sort_values(group_cols + [cfg['data']['datetime_column']]), group_cols, cfg['data']['target_column'], lags)

# fill NA in lag features
lag_cols = [f"{cfg['data']['target_column']}_lag_{l}" for l in lags]
fill_value = cfg['models']['baseline']['linear']['features'].get('fillna_value', 0.0)
train = fill_na_features(train, lag_cols, fill_value=fill_value)
eval = fill_na_features(eval, lag_cols, fill_value=fill_value)

# choose feature columns: cyclical encodings + lags (you can extend)
feature_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'] + lag_cols

# instantiate and train model
model_cfg = cfg['models']['baseline']['linear']
model = LinearBaseline(
    feature_cols=feature_cols,
    target_col=cfg['data']['target_column'],
    model_type=model_cfg.get('model_type', 'ridge'),
    ridge_alpha=model_cfg.get('ridge_alpha', 1.0),
    use_scaler=model_cfg.get('training', {}).get('use_scaler', True),
    test_size=model_cfg.get('training', {}).get('test_size', 0.2),
    random_state=cfg['training'].get('random_state', 42)
)

print("Fitting linear baseline...")
model.fit(train)

print("Predicting on eval...")
preds = model.predict(eval)
# save or evaluate preds
eval['prediction'] = preds
eval[['store_id','product_id',cfg['data']['datetime_column'],'prediction']].to_csv('linear_baseline_preds.csv', index=False)
print("Predictions saved to linear_baseline_preds.csv")
