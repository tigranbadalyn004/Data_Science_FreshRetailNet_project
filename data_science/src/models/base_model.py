# src/models/base_model.py
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, List
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseForecastingModel(ABC):
    """
    Abstract base class for all forecasting models.
    
    Teaching purpose: This demonstrates good software engineering practices
    - inheritance, abstraction, and consistent interfaces across different models.
    
    All models in our framework will inherit from this class, ensuring
    consistent behavior and making it easy to swap between different algorithms.
    """
    
    def __init__(self, model_name: str, config: Dict = None):
        """
        Initialize the base model with common attributes.
        
        Args:
            model_name: Human-readable name for this model
            config: Dictionary of model-specific configuration parameters
        """
        self.model_name = model_name
        self.config = config or {}
        self.is_fitted = False
        self.feature_names = None
        self.model = None
        self.training_history = {}
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseForecastingModel':
        """
        Fit the model to training data.
        
        This method must be implemented by all subclasses.
        It should set self.is_fitted = True upon successful completion.
        
        Args:
            X: Feature matrix (pandas DataFrame)
            y: Target variable (pandas Series)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Self for method chaining (allows model.fit().predict() syntax)
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate predictions for given features.
        
        Args:
            X: Feature matrix (pandas DataFrame)
            **kwargs: Additional prediction parameters
            
        Returns:
            Array of predictions
        """
        pass
    
    def validate_input(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Validate input data for common issues.
        
        Teaching point: Always validate your inputs! This prevents mysterious
        errors later and makes debugging much easier.
        """
        # Check for completely empty data
        if X.empty:
            raise ValueError("Input DataFrame is empty")
            
        # Check for infinite values
        # if np.isinf(X.select_dtypes(include=[np.number]).values).any():
        #     logger.warning("Infinite values detected in features")
            
        # Check for extremely high missing value rates
        missing_rates = X.isnull().mean()
        high_missing = missing_rates[missing_rates > 0.5]
        if not high_missing.empty:
            logger.warning(f"Features with >50% missing values: {high_missing.index.tolist()}")
        
        # Validate target if provided
        if y is not None:
            if len(X) != len(y):
                raise ValueError(f"Feature matrix ({len(X)} rows) and target ({len(y)} rows) have different lengths")
            
            if y.isnull().sum() > 0:
                logger.warning(f"Target variable has {y.isnull().sum()} missing values")
    
    def save_model(self, filepath: str) -> None:
        """Save the fitted model to disk for later use."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
            
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the entire model object
        joblib.dump(self, filepath)
        logger.info(f"Model {self.model_name} saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BaseForecastingModel':
        """Load a previously saved model from disk."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance if the underlying model supports it.
        
        Teaching insight: Not all models provide feature importance,
        but when they do, it's incredibly valuable for understanding
        what drives your predictions.
        """
        if not self.is_fitted:
            logger.warning("Model not fitted yet")
            return None
            
        # Check for sklearn-style feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        # Check for linear model coefficients
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
        else:
            logger.info(f"Feature importance not available for {self.model_name}")
            return None
        
        if self.feature_names is not None:
            return dict(zip(self.feature_names, importances))
        else:
            return dict(zip([f"feature_{i}" for i in range(len(importances))], importances))
    
    def get_model_summary(self) -> Dict:
        """Get a summary of the model's key characteristics."""
        return {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'num_features': len(self.feature_names) if self.feature_names else None,
            'config': self.config,
            'training_history': self.training_history
        }