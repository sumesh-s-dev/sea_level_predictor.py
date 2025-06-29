"""
NOAA Data Collector for Sea Level Predictor.
Fetches real-time and historical sea level data from NOAA CO-OPS API.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time
from utils.config import Config
from utils.logger import get_logger, log_execution_time
from utils.helpers import validate_data, save_data

logger = get_logger(__name__)

class NOAACollector:
    """Collector for NOAA sea level data."""
    
    def __init__(self):
        self.base_url = Config.NOAA_BASE_URL
        self.api_key = Config.NOAA_API_KEY
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SeaLevelPredictor/1.0 (Educational Project)'
        })
    
    @log_execution_time
    def get_station_data(self, station_id: str, start_date: str, end_date: str, 
                        product: str = 'water_level', datum: str = 'MLLW') -> pd.DataFrame:
        """
        Fetch water level data for a specific station and date range.
        
        Args:
            station_id: NOAA station ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            product: Data product type ('water_level', 'predictions', 'hourly_height')
            datum: Vertical datum ('MLLW', 'NAVD88', 'MSL')
        
        Returns:
            DataFrame with station data
        """
        params = {
            'begin_date': start_date.replace('-', ''),
            'end_date': end_date.replace('-', ''),
            'station': station_id,
            'product': product,
            'datum': datum,
            'time_zone': 'lst_ldt',
            'units': 'metric',
            'format': 'json',
            'application': 'SeaLevelPredictor'
        }
        
        try:
            logger.info(f"Fetching data for station {station_id} from {start_date} to {end_date}")
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'error' in data:
                logger.error(f"NOAA API error: {data['error']}")
                return pd.DataFrame()
            
            if 'data' not in data or not data['data']:
                logger.warning(f"No data available for station {station_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data['data'])
            
            # Clean and process data
            df = self._process_station_data(df, station_id)
            
            logger.info(f"Successfully fetched {len(df)} records for station {station_id}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for station {station_id}: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing data for station {station_id}: {str(e)}")
            return pd.DataFrame()
    
    def _process_station_data(self, df: pd.DataFrame, station_id: str) -> pd.DataFrame:
        """
        Process and clean station data.
        
        Args:
            df: Raw station data
            station_id: Station ID for reference
        
        Returns:
            Processed DataFrame
        """
        if df.empty:
            return df
        
        # Add station ID
        df['station_id'] = station_id
        
        # Convert date and time columns
        if 't' in df.columns:
            df['datetime'] = pd.to_datetime(df['t'])
            df['date'] = df['datetime'].dt.date
            df['time'] = df['datetime'].dt.time
        
        # Convert water level to numeric
        if 'v' in df.columns:
            df['water_level'] = pd.to_numeric(df['v'], errors='coerce')
        
        # Convert quality flags
        if 'q' in df.columns:
            df['quality'] = df['q']
        
        # Convert type flags
        if 'f' in df.columns:
            df['type'] = df['f']
        
        # Select relevant columns
        columns_to_keep = ['station_id', 'datetime', 'date', 'time', 'water_level', 'quality', 'type']
        available_columns = [col for col in columns_to_keep if col in df.columns]
        
        return df[available_columns].dropna(subset=['water_level'])
    
    @log_execution_time
    def get_stations_list(self) -> List[Dict[str, Any]]:
        """
        Get list of available NOAA stations.
        
        Returns:
            List of station information dictionaries
        """
        try:
            # This would typically fetch from NOAA stations API
            # For now, return a predefined list of major stations
            stations = [
                {
                    'id': '8727520',
                    'name': 'Fort Myers, FL',
                    'state': 'FL',
                    'latitude': 26.6473,
                    'longitude': -81.8722
                },
                {
                    'id': '8724580',
                    'name': 'Key West, FL',
                    'state': 'FL',
                    'latitude': 24.5553,
                    'longitude': -81.8080
                },
                {
                    'id': '9447130',
                    'name': 'Seattle, WA',
                    'state': 'WA',
                    'latitude': 47.6023,
                    'longitude': -122.3393
                },
                {
                    'id': '9410230',
                    'name': 'San Diego, CA',
                    'state': 'CA',
                    'latitude': 32.7143,
                    'longitude': -117.1733
                },
                {
                    'id': '8518750',
                    'name': 'The Battery, NY',
                    'state': 'NY',
                    'latitude': 40.7004,
                    'longitude': -74.0141
                }
            ]
            
            logger.info(f"Retrieved {len(stations)} stations")
            return stations
            
        except Exception as e:
            logger.error(f"Error fetching stations list: {str(e)}")
            return []
    
    @log_execution_time
    def get_current_data(self, station_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get current water level data for specified stations.
        
        Args:
            station_ids: List of station IDs (uses default stations if None)
        
        Returns:
            DataFrame with current data
        """
        if station_ids is None:
            station_ids = Config.DEFAULT_STATIONS
        
        all_data = []
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        for station_id in station_ids:
            logger.info(f"Fetching current data for station {station_id}")
            station_data = self.get_station_data(station_id, start_date, end_date)
            
            if not station_data.empty:
                all_data.append(station_data)
            
            # Rate limiting
            time.sleep(1)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Collected current data for {len(combined_data)} records")
            return combined_data
        else:
            logger.warning("No current data collected")
            return pd.DataFrame()
    
    @log_execution_time
    def get_historical_data(self, station_id: str, years: int = 10) -> pd.DataFrame:
        """
        Get historical data for a station.
        
        Args:
            station_id: NOAA station ID
            years: Number of years of historical data to fetch
        
        Returns:
            DataFrame with historical data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        logger.info(f"Fetching {years} years of historical data for station {station_id}")
        
        # Fetch data in chunks to avoid API limits
        chunk_size = 365  # days
        all_data = []
        
        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=chunk_size), end_date)
            
            chunk_data = self.get_station_data(
                station_id,
                current_start.strftime('%Y-%m-%d'),
                current_end.strftime('%Y-%m-%d')
            )
            
            if not chunk_data.empty:
                all_data.append(chunk_data)
            
            current_start = current_end
            time.sleep(2)  # Rate limiting
        
        if all_data:
            historical_data = pd.concat(all_data, ignore_index=True)
            historical_data = historical_data.drop_duplicates(subset=['datetime'])
            historical_data = historical_data.sort_values('datetime')
            
            logger.info(f"Collected {len(historical_data)} historical records for station {station_id}")
            return historical_data
        else:
            logger.warning(f"No historical data collected for station {station_id}")
            return pd.DataFrame()
    
    def save_station_data(self, data: pd.DataFrame, station_id: str, data_type: str = 'current') -> None:
        """
        Save station data to file.
        
        Args:
            data: DataFrame to save
            station_id: Station ID
            data_type: Type of data ('current', 'historical')
        """
        if data.empty:
            logger.warning(f"No data to save for station {station_id}")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data/noaa_{station_id}_{data_type}_{timestamp}.csv"
        
        save_data(data, filename, 'csv')
        logger.info(f"Data saved to {filename}")

def main():
    """Main function for testing the collector."""
    collector = NOAACollector()
    
    # Test current data collection
    logger.info("Testing current data collection...")
    current_data = collector.get_current_data()
    
    if not current_data.empty:
        logger.info(f"Successfully collected {len(current_data)} current records")
        print(current_data.head())
    else:
        logger.error("Failed to collect current data")
    
    # Test historical data collection
    logger.info("Testing historical data collection...")
    historical_data = collector.get_historical_data('8727520', years=1)
    
    if not historical_data.empty:
        logger.info(f"Successfully collected {len(historical_data)} historical records")
        print(historical_data.head())
    else:
        logger.error("Failed to collect historical data")

if __name__ == "__main__":
    main() 