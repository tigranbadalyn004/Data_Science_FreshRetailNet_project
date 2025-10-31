# src/models/baseline/ensemble_models.py
import logging
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.models.baseline.linear_models import LinearBaseline
from src.models.baseline.tree_models import XGBoostBaseline

logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


class EnsembleBaseline:
    """
    Ensemble regression baseline using stacking.
    """

    def __init__(self,
                 feature_cols: List[str],
                 target_col: str = "sale_amount",
                 base_models: Dict[str, object] = None,
                 meta_model_type: str = "linear",
                 test_size: float = 0.2,
                 random_state: int = 42,
                 fillna_value: float = 0.0):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.fillna_value = fillna_value

        # Initialize base models
        self.base_models = base_models if base_models else {
            "linear": LinearBaseline(feature_cols=feature_cols, target_col=target_col),
            "xgboost": XGBoostBaseline(feature_cols=feature_cols, target_col=target_col)
        }

        # Meta-model for stacking
        self.meta_model = LinearRegression() if meta_model_type == "linear" else None
        self.model = None

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        X = df[self.feature_cols].fillna(self.fillna_value)
        y = df[self.target_col]

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Prepare sklearn estimators for stacking
        estimators = []
        for name, model in self.base_models.items():
            # model.fit() will be called inside stacking
            estimators.append((name, model.model if hasattr(model, "model") else model))

        self.model = StackingRegressor(
            estimators=estimators,
            final_estimator=self.meta_model,
            n_jobs=-1,
            passthrough=True
        )

        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_val)
        preds = np.clip(preds, a_min=0.0, a_max=None)

        mae = mean_absolute_error(y_val, preds)
        rmse = mean_squared_error(y_val, preds, squared=False)
        logger.info(f"EnsembleBaseline fit: val MAE={mae:.4f}, RMSE={rmse:.4f}")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
        X = df[self.feature_cols].fillna(self.fillna_value)
        preds = self.model.predict(X)
        preds = np.clip(preds, a_min=0.0, a_max=None)
        return pd.Series(preds, index=df.index, name="prediction")
