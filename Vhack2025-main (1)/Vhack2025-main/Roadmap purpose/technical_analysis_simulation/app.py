from flask import Flask, render_template, send_from_directory, jsonify
import json
import os
import pandas as pd
from pathlib import Path
import glob

app = Flask(__name__)

# Configure the data directory
DATA_DIR = Path.home() / "Downloads" / "TradingSystem"

@app.route('/')
def index():
    """Render the main dashboard page"""
    # Load the strategy results from JSON
    results_path = DATA_DIR / "trading_system_results.json"
    
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}
    
    # Get list of available charts
    chart_files = []
    for chart_path in DATA_DIR.glob("*.png"):
        chart_files.append(chart_path.name)
    
    # Get list of available strategy CSVs
    strategy_files = []
    for strategy_path in DATA_DIR.glob("BTC-USD_*_strategy.csv"):
        strategy_name = strategy_path.stem.replace("BTC-USD_", "").replace("_strategy", "")
        strategy_files.append({
            "name": strategy_name,
            "file": strategy_path.name
        })
    
    return render_template('index.html', 
                          results=results, 
                          chart_files=chart_files,
                          strategy_files=strategy_files)

@app.route('/charts/<chart_name>')
def get_chart(chart_name):
    """Serve chart images"""
    return send_from_directory(DATA_DIR, chart_name)

@app.route('/data/<filename>')
def get_data_file(filename):
    """Serve data files"""
    return send_from_directory(DATA_DIR, filename)

@app.route('/api/strategy/<strategy_name>')
def get_strategy_data(strategy_name):
    """API endpoint to get strategy data as JSON"""
    strategy_file = f"BTC-USD_{strategy_name}_strategy.csv"
    strategy_path = DATA_DIR / strategy_file
    
    if strategy_path.exists():
        # Load only the last 100 rows for performance
        df = pd.read_csv(strategy_path, parse_dates=['Date'], index_col='Date').tail(100)
        
        # Return key strategy data
        data = {
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'close': df['Close'].tolist(),
            'equity': df['Equity'].tolist(),
            'drawdown': df['Drawdown'].tolist(),
            'signals': df['Trade_Signal'].tolist()
        }
        return jsonify(data)
    else:
        return jsonify({'error': 'Strategy file not found'}), 404

@app.route('/api/compare')
def get_comparison_data():
    """API endpoint to get strategy comparison data"""
    results_path = DATA_DIR / "trading_system_results.json"
    
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Extract key metrics for comparison
        comparison = []
        for strategy_name, strategy_data in results.items():
            metrics = strategy_data.get('metrics', {})
            strategy_metrics = metrics.get('strategy', {})
            trade_stats = metrics.get('trade_stats', {})
            
            comparison.append({
                'name': strategy_name,
                'total_return': strategy_metrics.get('total_return', 0),
                'annual_return': strategy_metrics.get('annual_return', 0),
                'sharpe_ratio': strategy_metrics.get('sharpe_ratio', 0),
                'max_drawdown': strategy_metrics.get('max_drawdown', 0),
                'win_rate': trade_stats.get('win_rate', 0),
                'total_trades': trade_stats.get('total_trades', 0)
            })
        
        return jsonify(comparison)
    else:
        return jsonify({'error': 'Results file not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)