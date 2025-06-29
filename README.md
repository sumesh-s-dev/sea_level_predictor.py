# Sea Level Predictor

A professional machine learning system for real-time sea level prediction and analysis using historical data from NOAA and NASA.

## 🌊 Features

- **Live Data Collection**: Real-time sea level data from NOAA/NASA APIs
- **Multiple ML Models**: XGBoost, Prophet, and ARIMA for robust predictions
- **Interactive Dashboard**: Real-time visualization with Plotly and Dash
- **Historical Analysis**: Comprehensive trend analysis and anomaly detection
- **API Endpoints**: RESTful API for integration with other systems
- **Automated Retraining**: Scheduled model updates with new data
- **Professional Documentation**: Complete API docs and user guides

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sea_level_predictor.git
cd sea_level_predictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. Run the application:
```bash
python app.py
```

The dashboard will be available at `http://localhost:8050`

## 📊 Project Structure

```
sea_level_predictor/
├── app.py                 # Main Flask application
├── dashboard.py           # Dash dashboard
├── data/
│   ├── collectors/        # Data collection modules
│   ├── processors/        # Data processing modules
│   └── models/           # Trained model files
├── models/
│   ├── ml_models.py      # ML model implementations
│   ├── feature_engineering.py
│   └── evaluation.py     # Model evaluation metrics
├── utils/
│   ├── config.py         # Configuration management
│   ├── logger.py         # Logging utilities
│   └── helpers.py        # Helper functions
├── api/
│   └── routes.py         # API endpoints
├── tests/                # Unit tests
├── docs/                 # Documentation
└── notebooks/            # Jupyter notebooks for analysis
```

## 🔧 Configuration

Create a `.env` file with the following variables:

```env
NOAA_API_KEY=your_noaa_api_key
NASA_API_KEY=your_nasa_api_key
DATABASE_URL=your_database_url
MODEL_PATH=./data/models/
LOG_LEVEL=INFO
```

## 📈 Usage

### Web Dashboard
- Visit `http://localhost:8050` for the interactive dashboard
- View real-time sea level data and predictions
- Explore historical trends and anomalies

### API Endpoints

```bash
# Get current sea level data
GET /api/current

# Get predictions for next 30 days
GET /api/predictions?days=30

# Get historical data
GET /api/historical?start_date=2020-01-01&end_date=2023-12-31

# Retrain models
POST /api/retrain
```

### Command Line

```bash
# Collect latest data
python -m data.collectors.noaa_collector

# Train models
python -m models.train_models

# Generate predictions
python -m models.predict --days 30
```

## 🤖 Machine Learning Models

### XGBoost
- Gradient boosting for short-term predictions
- Handles non-linear relationships
- Feature importance analysis

### Prophet (Facebook)
- Time series forecasting
- Handles seasonality and trends
- Robust to missing data

### ARIMA
- Statistical time series model
- Captures autocorrelation patterns
- Good for trend analysis

## 📊 Data Sources

- **NOAA CO-OPS**: Real-time tide gauge data
- **NASA JPL**: Satellite altimetry data
- **PSMSL**: Permanent Service for Mean Sea Level
- **GLOSS**: Global Sea Level Observing System

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=. tests/
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NOAA for providing sea level data
- NASA for satellite altimetry data
- The scientific community for research and models

## 📞 Support

For support, email support@sealevelpredictor.com or create an issue in the repository.

## 🔬 Research

This project is based on scientific research in:
- Climate change and sea level rise
- Time series forecasting
- Machine learning applications in environmental science

---

**Disclaimer**: This tool is for educational and research purposes. Always consult official sources for critical applications.
