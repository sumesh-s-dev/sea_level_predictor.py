import os
import pandas as pd
from datetime import datetime, timedelta
from data.collectors.noaa_collector import NOAACollector
from models.ml_models import SeaLevelPredictor
from utils.helpers import save_data
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

def main():
    # 1. Collect latest data for all default stations
    collector = NOAACollector()
    all_data = []
    for station_id in Config.DEFAULT_STATIONS:
        logger.info(f"Collecting historical data for station {station_id}")
        data = collector.get_historical_data(station_id, years=Config.HISTORICAL_DATA_YEARS)
        if not data.empty:
            data['station_id'] = station_id
            all_data.append(data)
        else:
            logger.warning(f"No data collected for station {station_id}")
    if not all_data:
        logger.error("No data collected for any station. Exiting pipeline.")
        return
    full_data = pd.concat(all_data, ignore_index=True)
    # Save collected data
    os.makedirs('data/processed', exist_ok=True)
    data_path = 'data/processed/sea_level_data.csv'
    full_data.to_csv(data_path, index=False)
    logger.info(f"Saved collected data to {data_path}")

    # 2. Train models
    predictor = SeaLevelPredictor()
    logger.info("Training XGBoost model...")
    predictor.train_xgboost(full_data)
    logger.info("Training Prophet model...")
    predictor.train_prophet(full_data)
    logger.info("Training ARIMA model...")
    predictor.train_arima(full_data)
    # Save trained models
    predictor.save_models()
    logger.info("Trained models saved.")

    # 3. Generate predictions for next 30 days for each model
    predictions = {}
    for model_name in ['xgboost', 'prophet', 'arima']:
        try:
            pred_df = predictor.predict_future(days=30, model_name=model_name)
            predictions[model_name] = pred_df
            pred_path = f'data/processed/predictions_{model_name}.csv'
            pred_df.to_csv(pred_path, index=False)
            logger.info(f"Saved {model_name} predictions to {pred_path}")
        except Exception as e:
            logger.error(f"Failed to generate predictions for {model_name}: {e}")

    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main() 