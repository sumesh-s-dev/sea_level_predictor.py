import dash
from dash import html

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Sea Level Predictor Dashboard'),
    html.P('Welcome! This is a placeholder dashboard. Replace this with your real-time sea level prediction and analysis UI.')
])

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050) 