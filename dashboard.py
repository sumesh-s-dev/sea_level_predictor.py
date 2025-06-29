"""
Interactive Dashboard for Sea Level Predictor.
Real-time visualization and prediction interface using Dash.
"""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any

from utils.config import Config
from utils.logger import get_logger
from data.collectors.noaa_collector import NOAACollector
from models.ml_models import SeaLevelPredictor

logger = get_logger(__name__)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Sea Level Predictor Dashboard"

# Initialize components
collector = NOAACollector()
predictor = SeaLevelPredictor()

def create_layout():
    """Create the main dashboard layout."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("🌊 Sea Level Predictor", className="text-center mb-4"),
                html.P("Real-time sea level monitoring and prediction system", 
                      className="text-center text-muted")
            ])
        ]),
        
        # Control Panel
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Control Panel"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Station"),
                                dcc.Dropdown(
                                    id='station-dropdown',
                                    options=[
                                        {'label': 'Fort Myers, FL', 'value': '8727520'},
                                        {'label': 'Key West, FL', 'value': '8724580'},
                                        {'label': 'Seattle, WA', 'value': '9447130'},
                                        {'label': 'San Diego, CA', 'value': '9410230'},
                                        {'label': 'The Battery, NY', 'value': '8518750'}
                                    ],
                                    value='8727520'
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Prediction Days"),
                                dcc.Slider(
                                    id='prediction-days',
                                    min=7,
                                    max=90,
                                    step=7,
                                    value=30,
                                    marks={i: str(i) for i in range(7, 91, 7)}
                                )
                            ], width=6)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Refresh Data", id="refresh-btn", 
                                          color="primary", className="me-2"),
                                dbc.Button("Train Models", id="train-btn", 
                                          color="success", className="me-2"),
                                dbc.Button("Generate Predictions", id="predict-btn", 
                                          color="info")
                            ], className="mt-3")
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Main Content
        dbc.Row([
            # Current Data
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Current Sea Level Data"),
                    dbc.CardBody([
                        dcc.Graph(id='current-data-graph')
                    ])
                ])
            ], width=6),
            
            # Predictions
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Future Predictions"),
                    dbc.CardBody([
                        dcc.Graph(id='predictions-graph')
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        # Historical Analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Historical Analysis"),
                    dbc.CardBody([
                        dcc.Graph(id='historical-graph')
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Model Performance
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model Performance"),
                    dbc.CardBody([
                        html.Div(id='model-metrics')
                    ])
                ])
            ], width=6),
            
            # Statistics
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Statistics"),
                    dbc.CardBody([
                        html.Div(id='statistics-display')
                    ])
                ])
            ], width=6)
        ]),
        
        # Hidden divs for storing data
        dcc.Store(id='current-data-store'),
        dcc.Store(id='predictions-store'),
        dcc.Store(id='historical-data-store'),
        dcc.Store(id='model-performance-store'),
        
        # Interval component for auto-refresh
        dcc.Interval(
            id='interval-component',
            interval=300000,  # 5 minutes
            n_intervals=0
        )
    ], fluid=True)

app.layout = create_layout()

@app.callback(
    Output('current-data-store', 'data'),
    Output('current-data-graph', 'figure'),
    Input('refresh-btn', 'n_clicks'),
    Input('interval-component', 'n_intervals'),
    Input('station-dropdown', 'value')
)
def update_current_data(n_clicks, n_intervals, station_id):
    """Update current sea level data."""
    try:
        # Fetch current data
        current_data = collector.get_current_data([station_id])
        
        if current_data.empty:
            # Create empty figure if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return {}, fig
        
        # Create time series plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=current_data['datetime'],
            y=current_data['water_level'],
            mode='lines+markers',
            name='Current Water Level',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=f"Current Sea Level - Station {station_id}",
            xaxis_title="Date",
            yaxis_title="Water Level (meters)",
            height=400,
            showlegend=True
        )
        
        return current_data.to_dict('records'), fig
        
    except Exception as e:
        logger.error(f"Error updating current data: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading data: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return {}, fig

@app.callback(
    Output('predictions-store', 'data'),
    Output('predictions-graph', 'figure'),
    Input('predict-btn', 'n_clicks'),
    Input('station-dropdown', 'value'),
    Input('prediction-days', 'value')
)
def generate_predictions(n_clicks, station_id, days):
    """Generate future predictions."""
    if n_clicks is None:
        return {}, {}
    
    try:
        # Load or train models
        try:
            predictor.load_models()
        except:
            logger.info("No pre-trained models found. Training new models...")
            # For demo purposes, create synthetic data and train
            dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
            np.random.seed(42)
            trend = np.linspace(0, 0.1, len(dates))
            seasonality = 0.05 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
            noise = 0.01 * np.random.randn(len(dates))
            water_levels = 1.0 + trend + seasonality + noise
            
            data = pd.DataFrame({
                'datetime': dates,
                'water_level': water_levels
            })
            
            predictor.train_xgboost(data)
            predictor.train_arima(data)
        
        # Generate predictions
        predictions = predictor.predict_future(days=days, model_name='xgboost')
        
        # Create prediction plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=predictions['datetime'],
            y=predictions['predicted_water_level'],
            mode='lines+markers',
            name='Predicted Water Level',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=f"{days}-Day Sea Level Prediction",
            xaxis_title="Date",
            yaxis_title="Predicted Water Level (meters)",
            height=400,
            showlegend=True
        )
        
        return predictions.to_dict('records'), fig
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error generating predictions: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return {}, fig

@app.callback(
    Output('historical-data-store', 'data'),
    Output('historical-graph', 'figure'),
    Input('station-dropdown', 'value')
)
def update_historical_data(station_id):
    """Update historical data visualization."""
    try:
        # For demo purposes, create synthetic historical data
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        trend = np.linspace(0, 0.1, len(dates))
        seasonality = 0.05 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
        noise = 0.01 * np.random.randn(len(dates))
        water_levels = 1.0 + trend + seasonality + noise
        
        historical_data = pd.DataFrame({
            'datetime': dates,
            'water_level': water_levels
        })
        
        # Create subplots for trend analysis
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Historical Sea Level', 'Trend Analysis'),
            vertical_spacing=0.1
        )
        
        # Historical data
        fig.add_trace(
            go.Scatter(
                x=historical_data['datetime'],
                y=historical_data['water_level'],
                mode='lines',
                name='Historical Data',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Trend line
        z = np.polyfit(range(len(historical_data)), historical_data['water_level'], 1)
        p = np.poly1d(z)
        trend_line = p(range(len(historical_data)))
        
        fig.add_trace(
            go.Scatter(
                x=historical_data['datetime'],
                y=trend_line,
                mode='lines',
                name='Trend Line',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Monthly averages
        monthly_data = historical_data.set_index('datetime').resample('M').mean()
        
        fig.add_trace(
            go.Scatter(
                x=monthly_data.index,
                y=monthly_data['water_level'],
                mode='lines+markers',
                name='Monthly Average',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text=f"Historical Analysis - Station {station_id}"
        )
        
        return historical_data.to_dict('records'), fig
        
    except Exception as e:
        logger.error(f"Error updating historical data: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading historical data: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return {}, fig

@app.callback(
    Output('model-performance-store', 'data'),
    Output('model-metrics', 'children'),
    Input('train-btn', 'n_clicks')
)
def update_model_performance(n_clicks):
    """Update model performance metrics."""
    if n_clicks is None:
        return {}, "Click 'Train Models' to see performance metrics"
    
    try:
        # For demo purposes, create sample metrics
        metrics = {
            'xgboost': {
                'r2': 0.85,
                'rmse': 0.02,
                'mae': 0.015
            },
            'arima': {
                'r2': 0.78,
                'rmse': 0.025,
                'mae': 0.018
            }
        }
        
        # Create metrics display
        metrics_cards = []
        for model_name, model_metrics in metrics.items():
            card = dbc.Card([
                dbc.CardHeader(model_name.upper()),
                dbc.CardBody([
                    html.P(f"R² Score: {model_metrics['r2']:.3f}"),
                    html.P(f"RMSE: {model_metrics['rmse']:.3f}"),
                    html.P(f"MAE: {model_metrics['mae']:.3f}")
                ])
            ], className="mb-2")
            metrics_cards.append(card)
        
        return metrics, metrics_cards
        
    except Exception as e:
        logger.error(f"Error updating model performance: {str(e)}")
        return {}, f"Error loading model performance: {str(e)}"

@app.callback(
    Output('statistics-display', 'children'),
    Input('current-data-store', 'data'),
    Input('historical-data-store', 'data')
)
def update_statistics(current_data, historical_data):
    """Update statistics display."""
    try:
        if not current_data or not historical_data:
            return "No data available for statistics"
        
        current_df = pd.DataFrame(current_data)
        historical_df = pd.DataFrame(historical_data)
        
        # Calculate statistics
        current_stats = {
            'mean': current_df['water_level'].mean(),
            'std': current_df['water_level'].std(),
            'min': current_df['water_level'].min(),
            'max': current_df['water_level'].max()
        }
        
        historical_stats = {
            'mean': historical_df['water_level'].mean(),
            'std': historical_df['water_level'].std(),
            'min': historical_df['water_level'].min(),
            'max': historical_df['water_level'].max()
        }
        
        # Create statistics display
        stats_content = [
            html.H5("Current Period"),
            html.P(f"Mean: {current_stats['mean']:.3f} m"),
            html.P(f"Std Dev: {current_stats['std']:.3f} m"),
            html.P(f"Range: {current_stats['min']:.3f} - {current_stats['max']:.3f} m"),
            html.Hr(),
            html.H5("Historical Period"),
            html.P(f"Mean: {historical_stats['mean']:.3f} m"),
            html.P(f"Std Dev: {historical_stats['std']:.3f} m"),
            html.P(f"Range: {historical_stats['min']:.3f} - {historical_stats['max']:.3f} m")
        ]
        
        return stats_content
        
    except Exception as e:
        logger.error(f"Error updating statistics: {str(e)}")
        return f"Error calculating statistics: {str(e)}"

if __name__ == '__main__':
    logger.info("Starting Sea Level Predictor Dashboard...")
    app.run_server(
        debug=Config.FLASK_DEBUG,
        host='0.0.0.0',
        port=Config.DASH_PORT
    ) 