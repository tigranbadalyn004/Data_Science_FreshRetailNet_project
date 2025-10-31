# src/models/baseline/naive_models.py
import pandas as pd
import numpy as np
from typing import Dict, Optional
from ..base_model import BaseForecastingModel
import logging

logger = logging.getLogger(__name__)

class NaiveForecaster(BaseForecastingModel):
    """
    Naive forecasting models - the starting point for any forecasting project.
    
    Teaching insight: Always start with simple baselines! These models:
    1. Are easy to understand and implement
    2. Provide a benchmark for more complex models
    3. Often work surprisingly well
    4. Help identify if complex models are actually adding value
    
    This class implements several naive strategies:
    - Last value (persistence model)
    - Seasonal naive (same time last week/day)
    - Average (overall mean)
    """
    
    def __init__(self, strategy: str = 'seasonal', seasonal_period: int = 24, config: Dict = None):
        """
        Initialize the naive forecaster.
        
        Args:
            strategy: 'last', 'seasonal', 'mean', or 'drift'
            seasonal_period: For seasonal strategy (24 for daily pattern, 168 for weekly)
            config: Additional configuration
        """
        super().__init__(f"Naive_{strategy}", config)
        self.strategy = strategy
        self.seasonal_period = seasonal_period
        self.fitted_values = {}
        
        if strategy not in ['last', 'seasonal', 'mean', 'drift']:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from ['last', 'seasonal', 'mean', 'drift']")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'NaiveForecaster':
        """
        Fit the naive model by storing necessary historical values.
        
        Teaching point: Naive models don't really "fit" in the traditional sense,
        but they do need to store some historical information for predictions.
        """
        self.validate_input(X, y)
        
        # We need store_id and product_id to group time series
        if 'store_id' not in X.columns or 'product_id' not in X.columns:
            raise ValueError("Naive forecaster requires 'store_id' and 'product_id' columns")
        
        # Combine features with target for easier grouping
        data = X.copy()
        data['target'] = y
        
        logger.info(f"Fitting {self.model_name} model...")
        
        # Store the fitted values based on strategy
        for (store_id, product_id), group in data.groupby(['store_id', 'product_id']):
            group = group.sort_values('dt')
            key = (store_id, product_id)
            
            if self.strategy == 'last':
                # Use the last observed value
                self.fitted_values[key] = group['target'].iloc[-1]
                
            elif self.strategy == 'seasonal':
                # Store the last seasonal_period values for seasonal naive
                self.fitted_values[key] = group['target'].iloc[-self.seasonal_period:].values
                
            elif self.strategy == 'mean':
                # Store the historical mean
                self.fitted_values[key] = group['target'].mean()
                
            elif self.strategy == 'drift':
                # Linear trend from first to last observation
                if len(group) > 1:
                    first_val = group['target'].iloc[0]
                    last_val = group['target'].iloc[-1]
                    trend = (last_val - first_val) / (len(group) - 1)
                    self.fitted_values[key] = {'last_value': last_val, 'trend': trend}
                else:
                    self.fitted_values[key] = {'last_value': group['target'].iloc[0], 'trend': 0}
        
        self.is_fitted = True
        self.training_history['num_time_series'] = len(self.fitted_values)
        
        logger.info(f"Fitted naive model for {len(self.fitted_values)} store-product combinations")
        return self
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate naive predictions based on the fitted strategy.
        
        Teaching insight: Notice how different naive strategies capture
        different aspects of time series behavior - trends, seasonality, etc.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        
        for _, row in X.iterrows():
            key = (row['store_id'], row['product_id'])
            
            if key not in self.fitted_values:
                # Handle new store-product combinations with overall mean
                # This is a common issue in production systems
                logger.warning(f"Unknown store-product combination: {key}. Using fallback prediction.")
                pred = 0  # Conservative fallback for new combinations
            else:
                if self.strategy == 'last':
                    pred = self.fitted_values[key]
                    
                elif self.strategy == 'seasonal':
                    # Use seasonal pattern - this requires knowing position in season
                    seasonal_values = self.fitted_values[key]
                    # For simplicity, use the mean of seasonal values
                    # In practice, you'd want to track the exact seasonal position
                    pred = np.mean(seasonal_values)
                    
                elif self.strategy == 'mean':
                    pred = self.fitted_values[key]
                    
                elif self.strategy == 'drift':
                    values = self.fitted_values[key]
                    # For simplicity, predict next value in trend
                    pred = values['last_value'] + values['trend']
            
            predictions.append(max(0, pred))  # Ensure non-negative predictions
        
        return np.array(predictions)

class SeasonalNaiveForecaster(BaseForecastingModel):
    """
    Advanced seasonal naive forecaster that properly handles seasonality.
    
    This is a more sophisticated version that tracks actual seasonal positions
    and can handle multiple seasonal patterns simultaneously.
    """
    
    def __init__(self, seasonal_periods: list[int] = [24, 168], config: Dict = None):
        """
        Initialize seasonal naive forecaster.
        
        Args:
            seasonal_periods: List of seasonal periods (24=daily, 168=weekly)
            config: Additional configuration
        """
        super().__init__("SeasonalNaive", config)
        self.seasonal_periods = seasonal_periods
        self.seasonal_data = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'SeasonalNaiveForecaster':
        """Fit by storing seasonal patterns for each time series."""
        self.validate_input(X, y)
        
        data = X.copy()
        data['target'] = y
        data['dt'] = pd.to_datetime(data['dt'])
        
        logger.info("Fitting seasonal naive model...")
        
        for (store_id, product_id), group in data.groupby(['store_id', 'product_id']):
            group = group.sort_values('dt').reset_index(drop=True)
            key = (store_id, product_id)
            
            seasonal_patterns = {}
            for period in self.seasonal_periods:
                # Create seasonal indices
                seasonal_indices = np.arange(len(group)) % period
                pattern = {}
                for i in range(period):
                    mask = seasonal_indices == i
                    if mask.sum() > 0:
                        pattern[i] = group.loc[mask, 'target'].mean()
                    else:
                        pattern[i] = 0
                seasonal_patterns[period] = pattern
            
            self.seasonal_data[key] = seasonal_patterns
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Generate predictions using seasonal patterns."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        
        for _, row in X.iterrows():
            key = (row['store_id'], row['product_id'])
            
            if key not in self.seasonal_data:
                predictions.append(0)
                continue
            
            # Use the primary seasonal period (first one)
            primary_period = self.seasonal_periods[0]
            pattern = self.seasonal_data[key][primary_period]
            
            # Calculate seasonal index based on hour of day
            hour = pd.to_datetime(row['dt']).hour
            seasonal_idx = hour % primary_period
            
            pred = pattern.get(seasonal_idx, 0)
            predictions.append(max(0, pred))
        
        return np.array(predictions)
    

    # src/models/baseline/naive_models.py
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)


class BaseForecastingModel:
    """
    Minimal interface expected by our pipeline.
    If you have src/models/base_model.py with a different interface,
    adapt these classes to inherit from it.
    """
    def fit(self, df: pd.DataFrame):
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


# Utility: ensure group keys present
def _check_required_columns(df: pd.DataFrame, required: List[str]):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


class SimpleNaive(BaseForecastingModel):
    """
    Predict last observed value per (store_id, product_id).
    If no history, fall back to global median (or 0).
    Expects df with columns: ['store_id','product_id','dt','sale_amount']
    """

    def __init__(self, fallback: Optional[float] = None):
        self.fallback = fallback
        self.last_map = {}  # (store,product) -> last_value

    def fit(self, df: pd.DataFrame):
        _check_required_columns(df, ['store_id', 'product_id', 'dt', 'sale_amount'])
        # sort by datetime and take last per group
        df_sorted = df.sort_values('dt')
        last = df_sorted.groupby(['store_id', 'product_id'])['sale_amount'].last()
        self.last_map = last.to_dict()
        if self.fallback is None:
            try:
                self.fallback = float(df['sale_amount'].median())
            except Exception:
                self.fallback = 0.0
        logger.info(f"SimpleNaive fit: stored {len(self.last_map)} last-values, fallback={self.fallback}")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        _check_required_columns(df, ['store_id', 'product_id'])
        preds = []
        for idx, row in df[['store_id', 'product_id']].iterrows():
            key = (row['store_id'], row['product_id'])
            val = self.last_map.get(key, self.fallback)
            preds.append(val)
        return pd.Series(preds, index=df.index, name='prediction')


class SeasonalNaive(BaseForecastingModel):
    """
    Predict using value from same hour last week (seasonal_period hours).
    Requires historical data with hourly granularity and aligned dt.
    """

    def __init__(self, seasonal_period: int = 168, fallback: Optional[float] = None):
        self.seasonal_period = int(seasonal_period)
        self.fallback = fallback
        # we'll store a mapping: (store,product,dt) -> past_value but for scalability we store by index shift
        self.history = None  # DataFrame indexed by (store,product,dt)

    def fit(self, df: pd.DataFrame):
        _check_required_columns(df, ['store_id', 'product_id', 'dt', 'sale_amount'])
        df_local = df.copy()
        df_local['dt'] = pd.to_datetime(df_local['dt'])
        df_local.set_index(['store_id', 'product_id', 'dt'], inplace=True)
        self.history = df_local['sale_amount'].sort_index()
        if self.fallback is None:
            self.fallback = float(df['sale_amount'].median())
        logger.info(f"SeasonalNaive fit: history length={len(self.history)}, seasonal_period={self.seasonal_period}")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        _check_required_columns(df, ['store_id', 'product_id', 'dt'])
        preds = []
        df_local = df.copy()
        df_local['dt'] = pd.to_datetime(df_local['dt'])
        for idx, row in df_local.iterrows():
            key = (row['store_id'], row['product_id'], row['dt'] - pd.Timedelta(hours=self.seasonal_period))
            val = self.history.get(key, np.nan) if self.history is not None else np.nan
            if pd.isna(val):
                val = self.fallback
            preds.append(float(val))
        return pd.Series(preds, index=df.index, name='prediction')


class MovingAverage(BaseForecastingModel):
    """
    Moving average with multiple windows and optional weights.
    Produces a weighted average of last available values within window per (store,product).
    Implementation: compute mean of last N observations (excluding current).
    """

    def __init__(self, windows: List[int] = [6, 24, 168], weights: Optional[List[float]] = None):
        self.windows = sorted([int(w) for w in windows])
        self.weights = weights
        self.group_series = None  # dict of Series per group for fast lookup

    def fit(self, df: pd.DataFrame):
        _check_required_columns(df, ['store_id', 'product_id', 'dt', 'sale_amount'])
        df_local = df.copy()
        df_local['dt'] = pd.to_datetime(df_local['dt'])
        # store series per group (sorted by dt)
        self.group_series = {
            grp: grp_df.set_index('dt')['sale_amount'].sort_index()
            for grp, grp_df in df_local.groupby(['store_id', 'product_id'], sort=False)
        }
        logger.info(f"MovingAverage fit: prepared series for {len(self.group_series)} groups")

    def _weighted_avg_last_hours(self, series: pd.Series, window_hours: int) -> float:
        # For a given group's series (indexed by datetime), take most recent 'window_hours' non-null values
        if series is None or len(series) == 0:
            return np.nan
        # last timestamp in series
        last_ts = series.index.max()
        cutoff = last_ts - pd.Timedelta(hours=window_hours)
        window_values = series[series.index > cutoff].dropna().values
        if len(window_values) == 0:
            return np.nan
        # by default simple mean; could be weighted by recency
        # weight by exponential recency: newer values get higher weight
        weights = np.exp(np.linspace(-1, 0, len(window_values)))
        return float(np.sum(window_values * weights) / np.sum(weights))

    def predict(self, df: pd.DataFrame) -> pd.Series:
        _check_required_columns(df, ['store_id', 'product_id', 'dt'])
        preds = []
        for idx, row in df.iterrows():
            grp = (row['store_id'], row['product_id'])
            series = self.group_series.get(grp, pd.Series(dtype=float))
            # compute per-window averages (exclude any value at prediction dt if present)
            per_window_vals = []
            for w in self.windows:
                val = self._weighted_avg_last_hours(series, w)
                per_window_vals.append(val)
            # combine per-window using normalized weights (recent windows higher weight)
            # default weights = inverse of window length
            wins = np.array(self.windows, dtype=float)
            if self.weights is None:
                wts = 1.0 / wins
            else:
                wts = np.array(self.weights, dtype=float)
                if len(wts) != len(wins):
                    raise ValueError("weights must match windows length")
            # handle nan values: set their weight to zero
            vals = np.array([v if (v is not None and not pd.isna(v)) else np.nan for v in per_window_vals], dtype=float)
            valid_mask = ~np.isnan(vals)
            if not valid_mask.any():
                preds.append(np.nan)
                continue
            wts = wts[valid_mask]
            vals = vals[valid_mask]
            wts = wts / wts.sum()
            pred = float(np.sum(vals * wts))
            preds.append(pred)
        # fallback: replace NaN preds with global median if needed
        preds_series = pd.Series(preds, index=df.index, name='prediction')
        median = np.median([v for s in self.group_series.values() for v in s.dropna().values]) if len(self.group_series)>0 else 0.0
        preds_series.fillna(median, inplace=True)
        return preds_series
