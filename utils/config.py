"""
Configuration management for Sea Level Predictor.
Handles environment variables and application settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration class."""
    
    # API Keys
    NOAA_API_KEY = os.getenv('NOAA_API_KEY', '')
    NASA_API_KEY = os.getenv('NASA_API_KEY', '')
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///sea_level_data.db')
    
    # Model Configuration
    MODEL_PATH = Path(os.getenv('MODEL_PATH', './data/models/'))
    MODEL_RETRAIN_INTERVAL_DAYS = int(os.getenv('MODEL_RETRAIN_INTERVAL_DAYS', 7))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/sea_level_predictor.log')
    
    # Server Configuration
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    PORT = int(os.getenv('PORT', 5000))
    DASH_PORT = int(os.getenv('DASH_PORT', 8050))
    
    # Data Collection
    DATA_COLLECTION_INTERVAL_HOURS = int(os.getenv('DATA_COLLECTION_INTERVAL_HOURS', 6))
    HISTORICAL_DATA_YEARS = int(os.getenv('HISTORICAL_DATA_YEARS', 10))
    
    # Prediction Configuration
    PREDICTION_DAYS = int(os.getenv('PREDICTION_DAYS', 30))
    CONFIDENCE_INTERVAL = float(os.getenv('CONFIDENCE_INTERVAL', 0.95))
    
    # External APIs
    NOAA_BASE_URL = os.getenv('NOAA_BASE_URL', 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter')
    NASA_BASE_URL = os.getenv('NASA_BASE_URL', 'https://api.nasa.gov/planetary/earth/assets')
    
    # Station Configuration
    DEFAULT_STATIONS = [
        '8727520',  # Fort Myers, FL
        '8724580',  # Key West, FL
        '9447130',  # Seattle, WA
        '9410230',  # San Diego, CA
        '8518750',  # The Battery, NY
    ]
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.MODEL_PATH,
            Path(cls.LOG_FILE).parent,
            Path('data/collectors'),
            Path('data/processors'),
            Path('logs'),
            Path('tests'),
            Path('docs'),
            Path('notebooks'),
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings."""
        errors = []
        
        if not cls.NOAA_API_KEY:
            errors.append("NOAA_API_KEY is required for data collection")
        
        if cls.PREDICTION_DAYS <= 0:
            errors.append("PREDICTION_DAYS must be positive")
        
        if not (0 < cls.CONFIDENCE_INTERVAL < 1):
            errors.append("CONFIDENCE_INTERVAL must be between 0 and 1")
        
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")
        
        return True

# Create directories on import
Config.create_directories() 