"""
Machine Learning Models for Sea Level Prediction.
Implements XGBoost, Prophet, and ARIMA models for time series forecasting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from utils.logger import get_logger, log_execution_time
from utils.helpers import save_model, load_model
from utils.config import Config

logger = get_logger(__name__)

class SeaLevelPredictor:
    """Main class for sea level prediction using multiple models."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_metrics = {}
        
    def prepare_features(self, data: pd.DataFrame, target_column: str = 'water_level') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for machine learning models."""
        df = data.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        
        # Time-based features
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['hour'] = df['datetime'].dt.hour
        
        # Cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Lag features
        for lag in [1, 7, 30]:
            df[f'lag_{lag}'] = df[target_column].shift(lag)
        
        # Rolling statistics
        for window in [7, 30]:
            df[f'rolling_mean_{window}'] = df[target_column].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df[target_column].rolling(window=window).std()
        
        # Remove NaN values
        df = df.dropna()
        
        # Select features
        feature_columns = [col for col in df.columns if col not in ['datetime', 'date', target_column, 'station_id']]
        X = df[feature_columns]
        y = df[target_column]
        
        return X, y
    
    @log_execution_time
    def train_xgboost(self, data: pd.DataFrame, target_column: str = 'water_level') -> Dict[str, Any]:
        """Train XGBoost model for sea level prediction."""
        logger.info("Training XGBoost model...")
        
        X, y = self.prepare_features(data, target_column)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Store model and results
        self.models['xgboost'] = model
        self.scalers['xgboost'] = scaler
        self.feature_importance['xgboost'] = dict(zip(X.columns, model.feature_importances_))
        self.model_metrics['xgboost'] = metrics
        
        logger.info(f"XGBoost training completed. R²: {metrics['r2']:.4f}")
        return metrics
    
    @log_execution_time
    def train_prophet(self, data: pd.DataFrame, target_column: str = 'water_level') -> Dict[str, Any]:
        """Train Prophet model for sea level prediction."""
        try:
            from prophet import Prophet
        except ImportError:
            logger.error("Prophet not installed. Skipping Prophet model.")
            return {}
        
        logger.info("Training Prophet model...")
        
        df = data.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        
        # Prepare data for Prophet
        prophet_df = df[['datetime', target_column]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df = prophet_df.dropna()
        
        # Split data
        split_idx = int(len(prophet_df) * 0.8)
        train_df = prophet_df.iloc[:split_idx]
        test_df = prophet_df.iloc[split_idx:]
        
        # Train model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        
        model.fit(train_df)
        
        # Make predictions
        future = model.make_future_dataframe(periods=len(test_df))
        forecast = model.predict(future)
        
        # Extract test predictions
        test_predictions = forecast.iloc[split_idx:]['yhat'].values
        y_test = test_df['y'].values
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, test_predictions)
        
        # Store model and results
        self.models['prophet'] = model
        self.model_metrics['prophet'] = metrics
        
        logger.info(f"Prophet training completed. R²: {metrics['r2']:.4f}")
        return metrics
        
    @log_execution_time
    def train_arima(self, data: pd.DataFrame, target_column: str = 'water_level') -> Dict[str, Any]:
        """Train ARIMA model for sea level prediction."""
        logger.info("Training ARIMA model...")
        
        df = data.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        df = df.set_index('datetime')
        
        # Resample to daily frequency
        daily_data = df[target_column].resample('D').mean().dropna()
        
        # Split data
        split_idx = int(len(daily_data) * 0.8)
        train_data = daily_data.iloc[:split_idx]
        test_data = daily_data.iloc[split_idx:]
        
        # Train ARIMA model
        model = ARIMA(train_data, order=(1, 1, 1))
        fitted_model = model.fit()
        
        # Make predictions
        forecast = fitted_model.forecast(steps=len(test_data))
        
        # Calculate metrics
        metrics = self._calculate_metrics(test_data.values, forecast.values)
        
        # Store model and results
        self.models['arima'] = fitted_model
        self.model_metrics['arima'] = metrics
        
        logger.info(f"ARIMA training completed. R²: {metrics['r2']:.4f}")
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate prediction metrics."""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    def predict_future(self, days: int = 30, model_name: str = 'xgboost') -> pd.DataFrame:
        """Make future predictions using the specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        logger.info(f"Making {days}-day predictions using {model_name}")
        
        if model_name == 'xgboost':
            return self._predict_xgboost_future(days)
        elif model_name == 'prophet':
            return self._predict_prophet_future(days)
        elif model_name == 'arima':
            return self._predict_arima_future(days)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def _predict_xgboost_future(self, days: int) -> pd.DataFrame:
        """Make future predictions using XGBoost."""
        # This is a simplified version - in practice, you'd need the last N days of data
        # to create the lag features for future predictions
        
        future_dates = pd.date_range(
            start=datetime.now(),
            periods=days + 30,  # Extra days for lag features
            freq='D'
        )
        
        # Create future features (simplified)
        future_df = pd.DataFrame({'datetime': future_dates})
        future_df['year'] = future_df['datetime'].dt.year
        future_df['month'] = future_df['datetime'].dt.month
        future_df['day'] = future_df['datetime'].dt.day
        future_df['day_of_year'] = future_df['datetime'].dt.dayofyear
        future_df['day_of_week'] = future_df['datetime'].dt.dayofweek
        
        # Cyclical features
        future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
        future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
        future_df['day_sin'] = np.sin(2 * np.pi * future_df['day_of_year'] / 365.25)
        future_df['day_cos'] = np.cos(2 * np.pi * future_df['day_of_year'] / 365.25)
        
        # Select features (excluding lag features for now)
        feature_columns = [col for col in future_df.columns if col != 'datetime']
        X_future = future_df[feature_columns]
        
        # Scale features
        X_future_scaled = self.scalers['xgboost'].transform(X_future)
        
        # Make predictions
        predictions = self.models['xgboost'].predict(X_future_scaled)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'datetime': future_dates,
            'predicted_water_level': predictions
        })
        
        return result_df.tail(days)  # Return only the requested number of days
    
    def _predict_prophet_future(self, days: int) -> pd.DataFrame:
        """Make future predictions using Prophet."""
        future = self.models['prophet'].make_future_dataframe(periods=days)
        forecast = self.models['prophet'].predict(future)
        
        result_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)
        result_df.columns = ['datetime', 'predicted_water_level', 'lower_bound', 'upper_bound']
        
        return result_df
    
    def _predict_arima_future(self, days: int) -> pd.DataFrame:
        """Make future predictions using ARIMA."""
        forecast = self.models['arima'].forecast(steps=days)
        
        future_dates = pd.date_range(
            start=datetime.now(),
            periods=days,
            freq='D'
        )
        
        result_df = pd.DataFrame({
            'datetime': future_dates,
            'predicted_water_level': forecast.values
        })
        
        return result_df
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all trained models."""
        return self.model_metrics
    
    def save_models(self, base_path: str = None) -> None:
        """Save all trained models."""
        if base_path is None:
            base_path = Config.MODEL_PATH
        
        for model_name, model in self.models.items():
            model_path = f"{base_path}/{model_name}_model.pkl"
            save_model(model, model_path)
            
            if model_name in self.scalers:
                scaler_path = f"{base_path}/{model_name}_scaler.pkl"
                save_model(self.scalers[model_name], scaler_path)
    
    def load_models(self, base_path: str = None) -> None:
        """Load trained models."""
        if base_path is None:
            base_path = Config.MODEL_PATH
        
        model_files = {
            'xgboost': f"{base_path}/xgboost_model.pkl",
            'prophet': f"{base_path}/prophet_model.pkl",
            'arima': f"{base_path}/arima_model.pkl"
        }
        
        for model_name, model_path in model_files.items():
            try:
                self.models[model_name] = load_model(model_path)
                logger.info(f"Loaded {model_name} model from {model_path}")
            except FileNotFoundError:
                logger.warning(f"Model file not found: {model_path}")
        
        # Load scalers
        scaler_files = {
            'xgboost': f"{base_path}/xgboost_scaler.pkl"
        }
        
        for scaler_name, scaler_path in scaler_files.items():
            try:
                self.scalers[scaler_name] = load_model(scaler_path)
                logger.info(f"Loaded {scaler_name} scaler from {scaler_path}")
            except FileNotFoundError:
                logger.warning(f"Scaler file not found: {scaler_path}")

def main():
    """Main function for testing the models."""
    # This would typically load real data
    # For demonstration, create synthetic data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Create synthetic sea level data with trend and seasonality
    trend = np.linspace(0, 0.1, len(dates))  # Rising trend
    seasonality = 0.05 * np.sin(2 * np.pi * dates.dayofyear / 365.25)  # Annual cycle
    noise = 0.01 * np.random.randn(len(dates))  # Random noise
    
    water_levels = 1.0 + trend + seasonality + noise
    
    data = pd.DataFrame({
        'datetime': dates,
        'water_level': water_levels
    })
    
    # Train models
    predictor = SeaLevelPredictor()
    
    print("Training XGBoost model...")
    xgb_metrics = predictor.train_xgboost(data)
    print(f"XGBoost metrics: {xgb_metrics}")
    
    print("Training ARIMA model...")
    arima_metrics = predictor.train_arima(data)
    print(f"ARIMA metrics: {arima_metrics}")
    
    # Make predictions
    print("Making future predictions...")
    future_predictions = predictor.predict_future(days=30, model_name='xgboost')
    print(f"Future predictions:\n{future_predictions.head()}")
    
    # Save models
    predictor.save_models()

if __name__ == "__main__":
    main() 