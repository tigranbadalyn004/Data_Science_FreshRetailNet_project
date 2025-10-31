from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from typing import Dict, Optional
from ..base_model import BaseForecastingModel
import logging
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
logger = logging.getLogger(__name__)

class LinearForecastingModel(BaseForecastingModel):
    """
    Linear regression models for demand forecasting.
    
    Teaching value: Linear models are interpretable, fast, and often surprisingly effective.
    They provide a good baseline and help students understand the relationship between
    features and predictions through coefficients.
    
    Supports multiple regularization techniques:
    - Linear Regression: No regularization
    - Ridge: L2 regularization (handles multicollinearity)
    - Lasso: L1 regularization (feature selection)
    - ElasticNet: Combined L1 + L2 regularization
    """
    
    def __init__(self, model_type: str = 'linear', config: Dict = None):
        """
        Initialize linear model with specified type and configuration.
        
        Args:
            model_type: 'linear', 'ridge', 'lasso', or 'elasticnet'
            config: Model configuration including regularization parameters
        """
        super().__init__(f"LinearRegression_{model_type}", config)
        self.model_type = model_type
        self.scaler = None
        self.imputer = None
        
        # Configuration with sensible defaults
        self.use_scaling = config.get('use_scaling', True) if config else True
        self.scaling_method = config.get('scaling_method', 'standard') if config else 'standard'
        self.handle_missing = config.get('handle_missing', True) if config else True
        
        # Initialize the appropriate sklearn model
        if model_type == 'linear':
            self.model = LinearRegression()
            
        elif model_type == 'ridge':
            alpha = config.get('alpha', 1.0) if config else 1.0
            self.model = Ridge(alpha=alpha, random_state=42)
            
        elif model_type == 'lasso':
            alpha = config.get('alpha', 1.0) if config else 1.0
            self.model = Lasso(alpha=alpha, random_state=42, max_iter=1000)
            
        elif model_type == 'elasticnet':
            alpha = config.get('alpha', 1.0) if config else 1.0
            l1_ratio = config.get('l1_ratio', 0.5) if config else 0.5
            self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=1000)
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def _prepare_features(self, X: pd.DataFrame, is_training: bool = False) -> np.ndarray:
        """
        Prepare features for modeling: handle missing values, scale, etc.
        
        Teaching point: Feature preprocessing is crucial and must be consistent
        between training and prediction time.
        """
        X_processed = X.copy()
        
        # Step 1: Handle missing values
        if self.handle_missing:
            if is_training:
                # Fit imputer on training data
                self.imputer = SimpleImputer(strategy='median')
                X_processed = pd.DataFrame(
                    self.imputer.fit_transform(X_processed),
                    columns=X_processed.columns,
                    index=X_processed.index
                )
            else:
                # Transform using fitted imputer
                if self.imputer is None:
                    raise ValueError("Imputer not fitted. This suggests an error in training.")
                X_processed = pd.DataFrame(
                    self.imputer.transform(X_processed),
                    columns=X_processed.columns,
                    index=X_processed.index
                )
        
        # Step 2: Feature scaling
        if self.use_scaling:
            if is_training:
                # Choose scaler based on configuration
                if self.scaling_method == 'standard':
                    self.scaler = StandardScaler()
                elif self.scaling_method == 'robust':
                    self.scaler = RobustScaler()  # Less sensitive to outliers
                else:
                    raise ValueError(f"Unknown scaling method: {self.scaling_method}")
                
                X_scaled = self.scaler.fit_transform(X_processed)
            else:
                if self.scaler is None:
                    raise ValueError("Scaler not fitted. This suggests an error in training.")
                X_scaled = self.scaler.transform(X_processed)
        else:
            X_scaled = X_processed.values
        
        return X_scaled
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'LinearForecastingModel':
        """
        Fit the linear model with comprehensive preprocessing.
        
        Teaching insight: The fit method should be robust to common data issues
        and provide informative logging for debugging.
        """
        self.validate_input(X, y)
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Remove non-numeric columns for linear regression
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]
        
        if len(numeric_cols) < len(X.columns):
            dropped_cols = set(X.columns) - set(numeric_cols)
            logger.warning(f"Dropped non-numeric columns: {list(dropped_cols)}")
            self.feature_names = numeric_cols.tolist()
        
        logger.info(f"Fitting {self.model_name} with {len(self.feature_names)} features...")
        
        # Prepare features
        X_processed = self._prepare_features(X_numeric, is_training=True)
        
        # Handle any remaining infinite values
        X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Fit the model
        self.model.fit(X_processed, y)
        self.is_fitted = True
        
        # Store training metrics for analysis
        train_score = self.model.score(X_processed, y)
        self.training_history['train_r2'] = train_score
        self.training_history['n_features'] = X_processed.shape[1]
        self.training_history['n_samples'] = X_processed.shape[0]
        
        logger.info(f"Training completed. R² score: {train_score:.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate predictions with the same preprocessing as training.
        
        Teaching point: Prediction preprocessing must exactly match training preprocessing.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Use only the features that were used during training
        X_features = X[self.feature_names]
        
        # Apply the same preprocessing pipeline
        X_processed = self._prepare_features(X_features, is_training=False)
        
        # Handle infinite values
        X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Generate predictions
        predictions = self.model.predict(X_processed)
        
        # Ensure non-negative predictions for demand forecasting
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def get_coefficients(self) -> Dict[str, float]:
        """
        Get model coefficients for interpretability.
        
        Teaching value: Understanding which features are most important
        and how they influence predictions is crucial for model trust and debugging.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get coefficients")
        
        if hasattr(self.model, 'coef_'):
            coefficients = dict(zip(self.feature_names, self.model.coef_))
            
            # Add intercept if available
            if hasattr(self.model, 'intercept_'):
                coefficients['intercept'] = self.model.intercept_
            
            return coefficients
        else:
            return {}
        
        # src/models/baseline/linear_models.py


logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


class LinearBaseline:
    """
    Linear baseline with preprocessing (imputation + scaling) and validation.
    """

    def __init__(self,
                 feature_cols: List[str],
                 target_col: str = "sale_amount",
                 model_type: str = "ridge",
                 ridge_alpha: float = 1.0,
                 use_scaler: bool = True,
                 test_size: float = 0.2,
                 random_state: int = 42):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.model_type = model_type
        self.ridge_alpha = ridge_alpha
        self.use_scaler = use_scaler
        self.test_size = test_size
        self.random_state = random_state

        self.pipeline = None
        self.model = None

    def _build_pipeline(self):
        steps = []
        # impute missing numeric with 0 (or median)
        steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=0.0)))
        if self.use_scaler:
            steps.append(("scaler", StandardScaler()))
        # final estimator
        if self.model_type == "ridge":
            estimator = Ridge(alpha=self.ridge_alpha, random_state=self.random_state)
        else:
            estimator = LinearRegression()
        steps.append(("estimator", estimator))
        self.pipeline = Pipeline(steps)

    def fit(self, df: pd.DataFrame):
        if self.pipeline is None:
            self._build_pipeline()

        X = df[self.feature_cols].copy()
        y = df[self.target_col].copy()

        # train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        self.pipeline.fit(X_train, y_train)
        preds = self.pipeline.predict(X_val)
        # ensure non-negative predictions
        preds = np.clip(preds, a_min=0.0, a_max=None)

        mae = mean_absolute_error(y_val, preds)
        rmse = mean_squared_error(y_val, preds, squared=False)
        logger.info(f"LinearBaseline fit: val MAE={mae:.4f}, RMSE={rmse:.4f}")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.pipeline is None:
            raise RuntimeError("Model not fitted yet")
        X = df[self.feature_cols].copy()
        preds = self.pipeline.predict(X)
        preds = np.clip(preds, a_min=0.0, a_max=None)
        return pd.Series(preds, index=df.index, name="prediction")

    def cross_validate(self, df: pd.DataFrame, cv: int = 5):
        X = df[self.feature_cols].copy()
        y = df[self.target_col].copy()
        if self.pipeline is None:
            self._build_pipeline()
        scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error")
        logger.info(f"Cross-val MAE (negated): {np.mean(scores):.4f} (std {np.std(scores):.4f})")
        return scores
