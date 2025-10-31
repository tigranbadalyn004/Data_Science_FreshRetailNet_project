import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Comprehensive feature engineering for retail demand forecasting.
    
    This class teaches students the art and science of feature engineering,
    which is often more impactful than choosing sophisticated algorithms.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize with preprocessing configuration.
        
        Teaching point: Feature engineering parameters should be configurable
        to enable easy experimentation and hyperparameter tuning.
        """
        self.config = config
        self.lag_periods = config['lag_periods']
        self.rolling_windows = config['rolling_windows']
        self.seasonal_periods = config['seasonal_periods']
        
    def create_temporal_features(self, df: pd.DataFrame, datetime_col: str = 'dt') -> pd.DataFrame:
        """
        Create comprehensive time-based features.
        
        Teaching insight: Time-based features are crucial for retail forecasting
        because demand patterns are highly temporal (hourly, daily, weekly cycles).
        """
        df = df.copy()
        
        # Ensure datetime column is properly formatted
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        logger.info("Creating temporal features...")
        
        # Basic temporal components
        # These capture different levels of seasonality
        df['hour'] = df[datetime_col].dt.hour
        df['day_of_week'] = df[datetime_col].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df[datetime_col].dt.day
        df['month'] = df[datetime_col].dt.month
        df['quarter'] = df[datetime_col].dt.quarter
        df['week_of_year'] = df[datetime_col].dt.isocalendar().week
        
        # Cyclical encoding - This is a key teaching moment!
        # Linear encoding (hour=23, hour=0) suggests these times are very different
        # Cyclical encoding correctly represents that 23:00 and 01:00 are close
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Business-relevant time indicators
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_weekday'] = (df['day_of_week'] < 5).astype(int)
        
        # Retail-specific time periods (teaching: domain knowledge matters!)
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_lunch_time'] = ((df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)
        df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
        df['is_late_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        
        # Time since start of dataset (trend component)
        df['days_since_start'] = (df[datetime_col] - df[datetime_col].min()).dt.days
        df['hours_since_start'] = (df[datetime_col] - df[datetime_col].min()).dt.total_seconds() / 3600
        
        logger.info(f"Created {len([c for c in df.columns if c not in df.columns[:20]])} temporal features")
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'sale_amount') -> pd.DataFrame:
        """
        Create lagged versions of the target variable.
        
        Teaching concept: Lag features capture the autocorrelation in time series.
        They're essential because past sales often predict future sales.
        """
        df = df.copy()
        
        # Sort data properly for lag calculation
        df = df.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)
        
        logger.info(f"Creating lag features for periods: {self.lag_periods}")
        
        # Create lag features for each store-product combination
        for lag in self.lag_periods:
            lag_col = f'{target_col}_lag_{lag}'
            
            # Use groupby to ensure lags are calculated within each time series
            df[lag_col] = df.groupby(['store_id', 'product_id'])[target_col].shift(lag)
            
            # For educational purposes, let's also create lag features for stockout status
            if lag <= 7:  # Only short-term lags for stockout (memory constraints)
                stockout_lag_col = f'stockout_lag_{lag}'
                df[stockout_lag_col] = df.groupby(['store_id', 'product_id'])['hours_stock_status'].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'sale_amount') -> pd.DataFrame:
        """
        Create rolling statistics features.
        
        Teaching insight: Rolling features capture local trends and patterns.
        They help the model understand if demand is increasing, decreasing, or stable.
        """
        df = df.copy()
        df = df.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)
        
        logger.info(f"Creating rolling features for windows: {self.rolling_windows}")
        
        for window in self.rolling_windows:
            # Rolling mean (local average demand)
            df[f'{target_col}_rolling_mean_{window}'] = (
                df.groupby(['store_id', 'product_id'])[target_col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
            
            # Rolling standard deviation (demand volatility)
            df[f'{target_col}_rolling_std_{window}'] = (
                df.groupby(['store_id', 'product_id'])[target_col]
                .rolling(window=window, min_periods=1)
                .std()
                .fillna(0)  # Fill NaN with 0 for single observations
                .reset_index(level=[0, 1], drop=True)
            )
            
            # Rolling maximum (peak demand in window)
            df[f'{target_col}_rolling_max_{window}'] = (
                df.groupby(['store_id', 'product_id'])[target_col]
                .rolling(window=window, min_periods=1)
                .max()
                .reset_index(level=[0, 1], drop=True)
            )
            
            # Rolling median (robust central tendency)
            df[f'{target_col}_rolling_median_{window}'] = (
                df.groupby(['store_id', 'product_id'])[target_col]
                .rolling(window=window, min_periods=1)
                .median()
                .reset_index(level=[0, 1], drop=True)
            )
        
        return df
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from categorical variables.
        
        Teaching point: Categorical features often contain rich information
        that needs to be extracted thoughtfully.
        """
        df = df.copy()
        
        logger.info("Creating categorical and interaction features...")
        
        # Store-level aggregations (store characteristics)
        store_stats = df.groupby('store_id').agg({
            'sale_amount': ['mean', 'std', 'max'],
            'product_id': 'nunique'  # Number of products per store
        }).round(2)
        
        store_stats.columns = ['store_avg_sales', 'store_sales_volatility', 'store_max_sales', 'store_product_count']
        df = df.merge(store_stats, on='store_id', how='left')
        
        # Product-level aggregations (product characteristics)
        product_stats = df.groupby('product_id').agg({
            'sale_amount': ['mean', 'std'],
            'store_id': 'nunique'  # Number of stores selling this product
        }).round(2)
        
        product_stats.columns = ['product_avg_sales', 'product_sales_volatility', 'product_store_count']
        df = df.merge(product_stats, on='product_id', how='left')
        
        # City-level features (regional effects)
        city_stats = df.groupby('city_id').agg({
            'sale_amount': 'mean',
            'hours_stock_status': 'mean'  # Average stock availability by city
        }).round(2)
        
        city_stats.columns = ['city_avg_sales', 'city_stock_availability']
        df = df.merge(city_stats, on='city_id', how='left')
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Teaching concept: Sometimes the combination of features is more
        informative than individual features alone.
        """
        df = df.copy()
        
        logger.info("Creating interaction features...")
        
        # Weather-time interactions (weather effect varies by time)
        df['temp_hour_interaction'] = df['avg_temperature'] * df['hour']
        df['rain_weekend_interaction'] = df['precpt'] * df['is_weekend']
        
        # Promotion-time interactions (promotion effectiveness varies by time)
        df['discount_weekend'] = df['discount'] * df['is_weekend']
        df['discount_evening'] = df['discount'] * df['is_evening_rush']
        
        # Stock-time interactions
        df['stock_hour_interaction'] = df['hours_stock_status'] * df['hour']
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps in the correct order.
        
        Teaching point: Feature engineering pipeline order matters!
        Some features depend on others being created first.
        """
        logger.info("Starting comprehensive feature engineering pipeline...")
        
        original_cols = len(df.columns)
        
        # Step 1: Temporal features (these don't depend on other features)
        df = self.create_temporal_features(df)
        
        # Step 2: Categorical features (these create new base features)
        # df = self.create_categorical_features(df)
        
        # Step 3: Lag features (these depend on having clean temporal data)
        # df = self.create_lag_features(df)
        
        # Step 4: Rolling features (these depend on temporal ordering)
        # df = self.create_rolling_features(df)
        
        # Step 5: Interaction features (these depend on base features existing)
        df = self.create_interaction_features(df)
        
        final_cols = len(df.columns)
        logger.info(f"Feature engineering complete: {original_cols} → {final_cols} features (+{final_cols - original_cols})")
        
        return df
    

    # src/data/feature_engineering.py
import pandas as pd
import numpy as np

def create_time_features(df: pd.DataFrame, datetime_col: str = "dt") -> pd.DataFrame:
    """
    Create basic temporal features: hour, day_of_week, month and cyclical encodings.
    """
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df['hour'] = df[datetime_col].dt.hour
    df['day_of_week'] = df[datetime_col].dt.dayofweek  # 0=Mon
    df['month'] = df[datetime_col].dt.month

    # cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

    return df


def create_lag_features(df: pd.DataFrame, group_cols: list, target_col: str, lags: list):
    """
    Add lag features (shift by hours) per group (store_id, product_id).
    Note: expects df sorted by group_cols + datetime beforehand.
    """
    df = df.copy()
    for lag in lags:
        col_name = f"{target_col}_lag_{lag}"
        df[col_name] = df.groupby(group_cols)[target_col].shift(lag)
    return df


def fill_na_features(df: pd.DataFrame, feature_cols: list, fill_value: float = 0.0):
    """
    Simple imputation for newly created features.
    We do a conservative fill (e.g., 0 or median), configurable in config.
    """
    df = df.copy()
    for col in feature_cols:
        if col in df.columns:
            df[col].fillna(fill_value, inplace=True)
    return df


# src/data/feature_engineering.py

import pandas as pd
import numpy as np

def add_time_features(df, datetime_col="dt", holiday_col="holiday_flag"):
    """
    Add temporal features to the dataset
    - Hour, Day of Week, Month
    - Cyclical encodings for hour, day, month
    - Interactions between time scales
    - Advanced holiday features (days before/after holidays)
    """

    df = df.copy()
    
    # Basic time features
    df["hour"] = df[datetime_col].dt.hour
    df["day_of_week"] = df[datetime_col].dt.dayofweek
    df["month"] = df[datetime_col].dt.month

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Interaction features
    df["hour_dow_interaction"] = df["hour"] * df["day_of_week"]
    df["hour_month_interaction"] = df["hour"] * df["month"]

    # Advanced holiday features
    if holiday_col in df.columns:
        df["day_before_holiday"] = df[holiday_col].shift(-1).fillna(0)
        df["day_after_holiday"] = df[holiday_col].shift(1).fillna(0)

    return df

def add_lag_rolling_features(df, target_col="sale_amount", 
                             lag_periods=[1,2,3,6,24,48,168],
                             rolling_windows=[6,24,168]):
    """
    Add lag and rolling statistical features
    - Lag features at multiple horizons
    - Rolling mean, std, min, max, quantiles, trends
    - Exponentially weighted features
    """
    df = df.copy()
    
    # Lag features
    for lag in lag_periods:
        df[f"{target_col}_lag_{lag}"] = df.groupby(['store_id','product_id'])[target_col].shift(lag)
    
    # Rolling statistics
    for window in rolling_windows:
        df[f"{target_col}_roll_mean_{window}"] = df.groupby(['store_id','product_id'])[target_col]\
                                                .transform(lambda x: x.shift(1).rolling(window).mean())
        df[f"{target_col}_roll_std_{window}"] = df.groupby(['store_id','product_id'])[target_col]\
                                                .transform(lambda x: x.shift(1).rolling(window).std())
        df[f"{target_col}_roll_min_{window}"] = df.groupby(['store_id','product_id'])[target_col]\
                                                .transform(lambda x: x.shift(1).rolling(window).min())
        df[f"{target_col}_roll_max_{window}"] = df.groupby(['store_id','product_id'])[target_col]\
                                                .transform(lambda x: x.shift(1).rolling(window).max())
        df[f"{target_col}_roll_q25_{window}"] = df.groupby(['store_id','product_id'])[target_col]\
                                                .transform(lambda x: x.shift(1).rolling(window).quantile(0.25))
        df[f"{target_col}_roll_q75_{window}"] = df.groupby(['store_id','product_id'])[target_col]\
                                                .transform(lambda x: x.shift(1).rolling(window).quantile(0.75))
        # Exponentially weighted mean
        df[f"{target_col}_ewm_{window}"] = df.groupby(['store_id','product_id'])[target_col]\
                                           .transform(lambda x: x.shift(1).ewm(span=window, adjust=False).mean())
    
    # Fill missing values (cold start) with 0
    df.fillna(0, inplace=True)
    return df


def add_weather_features(df, temp_col="temperature", precip_col="precipitation"):
    """
    Create weather-related features
    - Temperature categories: hot, mild, cold
    - Precipitation intensity: none, light, moderate, heavy
    - Temperature and precipitation changes from previous day
    - Interaction with time features can be added later
    """
    df = df.copy()
    
    # Temperature categories (thresholds can be tuned via config)
    df["temp_category"] = pd.cut(df[temp_col],
                                 bins=[-np.inf, 10, 25, np.inf],
                                 labels=["cold","mild","hot"])
    
    # Precipitation intensity levels
    df["precip_category"] = pd.cut(df[precip_col],
                                   bins=[-np.inf, 0, 2, 5, np.inf],
                                   labels=["none","light","moderate","heavy"])
    
    # Weather changes
    df["temp_change"] = df[temp_col].diff().fillna(0)
    df["precip_change"] = df[precip_col].diff().fillna(0)
    
    return df



def add_promo_features(df, discount_col="discount", store_col="store_id", category_col="product_category"):
    """
    Create promotion-related features
    - Discount magnitude categories: small, medium, large
    - Discount change indicators
    - Average discount in store/category
    - Interaction features with time and weather
    """
    df = df.copy()
    
    # Discount categories
    df["discount_category"] = pd.cut(df[discount_col],
                                     bins=[-np.inf, 0, 10, 20, np.inf],
                                     labels=["none","small","medium","large"])
    
    # Discount change indicator
    df["discount_change"] = df[discount_col].diff().fillna(0)
    
    # Average discount per store/category
    df["avg_discount_store"] = df.groupby(store_col)[discount_col].transform("mean")
    df["avg_discount_category"] = df.groupby(category_col)[discount_col].transform("mean")
    
    return df


def add_interaction_features(df):
    """
    Create interaction features between:
    - Weather and time (e.g., hot weekend afternoons)
    - Promotions and time (e.g., weekend discount effect)
    """
    df = df.copy()
    
    # Example: hot weekend afternoons
    df["hot_weekend_afternoon"] = ((df["temp_category"]=="hot") &
                                   (df["day_of_week"]>=5) &  # 5=Saturday, 6=Sunday
                                   (df["hour"].between(12,18))).astype(int)
    
    # Example: rainy morning promotions
    df["rainy_morning_discount"] = ((df["precip_category"]=="moderate") &
                                    (df["hour"].between(6,12)) &
                                    (df["discount"]>0)).astype(int)
    
    return df


def add_store_features(df, store_col="store_id", sales_col="sale_amount", product_col="product_id"):
    """
    Create features that capture store characteristics:
    - Store size proxies: number of products, total sales
    - Average sales per product
    - Stockout frequency
    - Optional: clustering stores based on sales patterns
    """
    df = df.copy()
    
    # Store size proxies
    store_agg = df.groupby(store_col).agg(
        store_total_sales = (sales_col, "sum"),
        store_num_products = (product_col, "nunique"),
        store_avg_sales_per_product = (sales_col, "mean"),
        store_stockout_freq = ("hours_stock_status", lambda x: (x==0).mean())
    ).reset_index()
    
    # Merge back to main df
    df = df.merge(store_agg, on=store_col, how="left")
    
    return df


def add_product_features(df, product_col="product_id", category_col="product_category", sales_col="sale_amount"):
    """
    Create features capturing product characteristics:
    - Popularity (sales rank within category)
    - Lifecycle stage (new/mature/declining)
    - Substitutability indicators (correlation with similar products)
    - Product velocity: intensity of demand
    """
    df = df.copy()
    
    # Product popularity: rank within category
    df["product_sales_rank"] = df.groupby(category_col)[sales_col].rank(method="dense", ascending=False)
    
    # Product velocity: rolling sum of sales
    df["product_velocity"] = df.groupby(product_col)[sales_col].transform(lambda x: x.rolling(7).sum().fillna(0))
    
    # Lifecycle stage based on average sales over time
    avg_sales = df.groupby(product_col)[sales_col].transform("mean")
    df["product_stage"] = pd.cut(avg_sales, bins=[-np.inf, 10, 50, np.inf], labels=["new","mature","declining"])
    
    return df


def add_cross_features(df, store_col="store_id", product_col="product_id", category_col="product_category"):
    """
    - Market basket effects: category-level spillovers, complementary products
    - Spatial/hierarchical features: city-level aggregations, category hierarchy
    - Cannibalization indicators for similar products
    """
    df = df.copy()
    
    # Category-level demand spillovers
    df["category_sales_sum"] = df.groupby(category_col)["sale_amount"].transform("sum")
    
    # Complementary product: count of other products sold in same transaction/hour
    df["comp_products_count"] = df.groupby([store_col, "dt"])[product_col].transform("nunique") - 1
    
    # City-level aggregation (if city_col exists)
    if "city" in df.columns:
        df["city_total_sales"] = df.groupby("city")["sale_amount"].transform("sum")
    
    return df


def feature_engineering_pipeline(df):
    """
    Full feature engineering pipeline:
    1. Time features
    2. Weather features
    3. Promo features
    4. Store-level features
    5. Product-level features
    6. Cross-product/store features
    7. Interaction features
    """
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_promo_features(df)
    df = add_store_features(df)
    df = add_product_features(df)
    df = add_cross_features(df)
    df = add_interaction_features(df)
    
    return df



