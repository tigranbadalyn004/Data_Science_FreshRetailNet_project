# src/data/data_loader.py
import pandas as pd
import numpy as np
from datasets import load_dataset
from typing import Dict, Tuple, Optional, List
import logging
from pathlib import Path
import yaml

# Set up logging to help students debug issues
logging.basicConfig(filename='app.log', filemode='a',level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FreshRetailDataLoader:
    """
    Production-ready data loader for FreshRetailNet-50K dataset.
    
    This class demonstrates best practices for data loading in ML pipelines:
    - Error handling and logging
    - Data validation
    - Caching for efficiency
    - Clear documentation
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the data loader with configuration.
        
        Teaching point: Always use configuration files rather than hardcoded values.
        This makes your code more maintainable and allows easy experimentation.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.dataset = None
        self.train_data = None
        self.eval_data = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the FreshRetailNet-50K dataset from HuggingFace.
        
        Returns:
            Tuple of (train_df, eval_df)
            
        Teaching note: This method demonstrates proper error handling
        and informative logging for production systems.
        """
        try:
            logger.info("Loading dataset...")
            
            self.train_data = pd.read_csv(self.config['data']["train_path"]).sample(frac=self.config['data']['fracture'], random_state=42)
            self.eval_data = pd.read_csv(self.config['data']["eval_path"]).sample(frac=self.config['data']['fracture'], random_state=42)

            datetime_col = self.data_config['datetime_column']
            self.train_data[datetime_col] = pd.to_datetime(self.train_data[datetime_col])
            self.eval_data[datetime_col] = pd.to_datetime(self.eval_data[datetime_col])
            
            logger.info(f"Successfully loaded:")
            logger.info(f"  Training samples: {len(self.train_data):,}")
            logger.info(f"  Evaluation samples: {len(self.eval_data):,}")
            logger.info(f"  Date range: {self.train_data[datetime_col].min()} to {self.train_data[datetime_col].max()}")
            
            return self.train_data, self.eval_data
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def get_data_summary(self) -> Dict:
        """
        Generate comprehensive data summary for exploratory analysis.
        
        Teaching point: Always start with data understanding before modeling.
        This method provides the foundation for EDA notebooks.
        """
        if self.train_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        datetime_col = self.data_config['datetime_column']
        target_col = self.data_config['target_column']
        
        summary = {
            'dataset_shape': {
                'train': self.train_data.shape,
                'eval': self.eval_data.shape
            },
            'time_span': {
                'start': str(self.train_data[datetime_col].min()),
                'end': str(self.train_data[datetime_col].max()),
                'duration_days': (self.train_data[datetime_col].max() - self.train_data[datetime_col].min()).days
            },
            'business_dimensions': {
                'unique_stores': self.train_data['store_id'].nunique(),
                'unique_products': self.train_data['product_id'].nunique(),
                'unique_cities': self.train_data['city_id'].nunique(),
                'total_store_product_combinations': len(self.train_data.groupby(['store_id', 'product_id']))
            },
            'target_statistics': {
                'mean_sales': self.train_data[target_col].mean(),
                'median_sales': self.train_data[target_col].median(),
                'zero_sales_percentage': (self.train_data[target_col] == 0).mean() * 100,
                'max_sales': self.train_data[target_col].max()
            },
            'stockout_analysis': {
                'total_observations': len(self.train_data),
                'stockout_hours': (self.train_data['hours_stock_status'] == 0).sum(),
                'stockout_rate_percent': (self.train_data['hours_stock_status'] == 0).mean() * 100,
                'stores_with_stockouts': self.train_data[self.train_data['hours_stock_status'] == 0]['store_id'].nunique()
            },
            'data_quality': {
                'missing_values': self.train_data.isnull().sum().to_dict(),
                'duplicate_rows': self.train_data.duplicated().sum()
            }
        }
        
        return summary