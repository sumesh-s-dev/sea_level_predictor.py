"""
Logging configuration for Sea Level Predictor.
Provides consistent logging across all modules.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from utils.config import Config

def setup_logger(name: str = 'sea_level_predictor', level: str = None) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    if level is None:
        level = Config.LOG_LEVEL
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = Path(Config.LOG_FILE)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (optional)
    
    Returns:
        Logger instance
    """
    if name is None:
        name = 'sea_level_predictor'
    
    return setup_logger(name)

# Create default logger
logger = get_logger()

def log_function_call(func):
    """Decorator to log function calls with parameters."""
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned {type(result).__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

def log_execution_time(func):
    """Decorator to log function execution time."""
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        logger.debug(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.info(f"{func.__name__} completed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    return wrapper 