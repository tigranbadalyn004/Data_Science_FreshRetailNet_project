# Data_Science_FreshRetailNet_project
Full data science project: retail demand forecasting, feature engineering, baseline and advanced ML models.


# Data Science FreshRetailNet Project

## Overview

This project is a comprehensive retail demand forecasting system built using Python, designed to analyze hourly sales data, handle stockouts, incorporate external factors like weather and promotions, and provide actionable insights for business operations. The project follows a production-ready structure with clear separation of data processing, modeling, evaluation, and configuration.

---

## Project Structure

📂 data_science/
│
├─ 📂 config/
│ └─ config.yaml # Central configuration file for data paths, model hyperparameters, and settings
│
├─ 📂 src/
│ ├─ 📂 data/
│ │ ├─ data_loader.py # Functions to load datasets
│ │ └─ feature_engineering.py # Functions for feature creation and preprocessing
│ │
│ ├─ 📂 models/
│ │ ├─ base_model.py # Base class for models (fit, predict, evaluate)
│ │ ├─ 📂 baseline/
│ │ │ ├─ naive_models.py # Simple Naive Forecasts
│ │ │ ├─ linear_models.py # Linear Regression Baseline
│ │ │ └─ tree_models.py # Random Forest/XGBoost Baseline
│ │
│ └─ 📂 evaluation/
│ └─ metrics.py # Evaluation metrics (MAE, RMSE, MAPE, Bias, business metrics)
│
├─ 📂 scripts/
│ └─ train_baseline.py # Script to run baseline model training and evaluation
│
├─ 📂 notebook/
│ ├─ 01_data_exploration.ipynb # Data exploration and visualization
│ └─ assignment2_feature_analysis.ipynb # Feature analysis and importance study
│
└─ README.md # This documentation file

markdown
Copy code

---

## Features and Capabilities

### 1. Data Exploration
- Load and summarize training and evaluation datasets.
- Analyze distributions of `sale_amount` (mean, median, std, skewness).
- Visualize zero-sales proportion and implications for forecasting.
- Identify temporal patterns (hourly, daily, weekly, monthly trends).
- Analyze stockout effects (censored demand).

### 2. Baseline Models
- **Naive Models**
  - Simple Naive: last observed value.
  - Seasonal Naive: same hour last week.
  - Moving Average: weighted averages with multiple windows.
- **Linear Regression**
  - Temporal features (hour, day_of_week, month) with cyclical encoding.
  - Lag features for 1, 7, 168 hours.
  - Feature scaling and validation.
- **Tree-Based Models**
  - Random Forest & XGBoost regression baselines.
  - Handles high-dimensional sparse data.
  - Ensures non-negative predictions.
- **Ensemble Regression**
  - Combines predictions from multiple models.
  - Improved stability and performance.

### 3. Advanced Feature Engineering
- Multi-scale seasonality: intra-day, weekly, monthly patterns.
- Holiday and event features: pre/post holiday impact, school calendar effects.
- Lag and rolling features with multiple horizons (short, medium, long-term).
- Exponentially weighted rolling statistics.
- Weather features: temperature, precipitation, and interaction effects.
- Promotional features: discount categories, changes, and interactions.
- Store and product characteristics: size, performance, popularity, lifecycle.
- Cross-product and hierarchical features: market basket effects, category hierarchies, cannibalization indicators.

### 4. Evaluation Framework
- Core metrics: MAE, RMSE, MAPE, Bias.
- Business-relevant metrics: high vs low volume products, peak vs off-peak hours, stockout vs in-stock periods.
- Comparative evaluation across all baseline models.
- Residual analysis and systematic bias detection.
- Insightful reports highlighting model strengths and weaknesses.

---

## Configuration
- All paths, model hyperparameters, and feature engineering settings are defined in `config/config.yaml`.
- Centralized configuration allows easy switching between models, data subsets, and parameters without changing code.

---

## How to Run

1. **Clone Repository**
```bash
git clone https://github.com/<your-username>/Data_Science_FreshRetailNet_project.git
cd Data_Science_FreshRetailNet_project
Install Dependencies

bash
Copy code
pip install -r requirements.txt
Run Data Exploration

bash
Copy code
jupyter notebook notebook/01_data_exploration.ipynb
Train Baseline Models

bash
Copy code
python scripts/train_baseline.py
Analyze Features

bash
Copy code
jupyter notebook notebook/assignment2_feature_analysis.ipynb
Project Highlights
Production-ready project structure.

Configurable via config.yaml.

Comprehensive feature engineering capturing temporal, weather, promotion, and business domain effects.

Multiple baseline models with clear interfaces and evaluation metrics.

Designed to handle stockout scenarios and censored demand.

Notebook-based exploration for clear visualization and insights.

Contributing
Fork the repository and submit pull requests.

Add unit tests for any new feature engineering or modeling function.

Document all changes and maintain Markdown conventions for notebooks and scripts.
