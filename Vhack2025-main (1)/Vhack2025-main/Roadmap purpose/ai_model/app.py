from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import os
import sys

app = Flask(__name__)

# Path to the model files
DEFAULT_MODEL_PATH = r"C:\two\Vhack\ai_model\saved_models\best_trading_agent.keras"
BITCOIN_MODEL_PATH = r"C:\two\Vhack\ai_model\saved_models\bitcoin_model.keras"
ADVANCED_MODEL_PATH = r"C:\two\Vhack\ai_model\saved_models\final_trading_agent_2.keras"

# Custom layer needed for advanced model loading
class AdvantageNormalization(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AdvantageNormalization, self).__init__(**kwargs)
    
    def call(self, inputs):
        # Subtract mean from advantage
        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        return inputs - mean
    
    def get_config(self):
        config = super(AdvantageNormalization, self).get_config()
        return config

# Check if models exist
if not os.path.exists(DEFAULT_MODEL_PATH):
    print(f"ERROR: Default model not found at {DEFAULT_MODEL_PATH}")
if not os.path.exists(BITCOIN_MODEL_PATH):
    print(f"ERROR: Bitcoin model not found at {BITCOIN_MODEL_PATH}")
if not os.path.exists(ADVANCED_MODEL_PATH):
    print(f"ERROR: Advanced model not found at {ADVANCED_MODEL_PATH}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_model', methods=['GET'])
def check_model():
    """Check if a specific model exists"""
    model_id = request.args.get('model_id', 'default')
    
    if model_id == 'bitcoin':
        model_path = BITCOIN_MODEL_PATH
    elif model_id == 'advanced':
        model_path = ADVANCED_MODEL_PATH
    else:
        model_path = DEFAULT_MODEL_PATH
    
    if os.path.exists(model_path):
        return jsonify({
            'status': 'success',
            'message': f'Model found at {model_path}'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': f'Model not found at {model_path}'
        })

@app.route('/load_model', methods=['GET'])
def load_model():
    """Check if model exists and if so, load it and return basic info"""
    try:
        model_id = request.args.get('model_id', 'default')
        
        if model_id == 'bitcoin':
            model_path = BITCOIN_MODEL_PATH
            model_name = "Bitcoin Trading Model"
        elif model_id == 'advanced':
            model_path = ADVANCED_MODEL_PATH
            model_name = "Advanced Trading Model"
        else:
            model_path = DEFAULT_MODEL_PATH
            model_name = "Default Trading Model"
        
        if not os.path.exists(model_path):
            return jsonify({
                'status': 'error',
                'message': f'Model not found at {model_path}'
            })
        
        # Try to load the model - just to confirm it works
        # For advanced model, we need to include the custom layer
        if model_id == 'advanced':
            try:
                # Try loading with custom objects
                custom_objects = {'AdvantageNormalization': AdvantageNormalization}
                model = keras.models.load_model(model_path, custom_objects=custom_objects)
            except Exception as e:
                print(f"First loading attempt failed: {str(e)}")
                # Alternative loading method with compile=False
                try:
                    model = keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
                except Exception as e2:
                    return jsonify({
                        'status': 'error',
                        'message': f'Model loading failed: {str(e2)}'
                    })
        else:
            # Regular model loading for other models
            model = keras.models.load_model(model_path)
        
        # Get basic model info
        model_info = {
            'id': model_id,
            'name': model_name,
            'path': model_path,
            'layers': len(model.layers),
            'type': model.name
        }
        
        return jsonify({
            'status': 'success',
            'message': 'Model loaded successfully',
            'model_info': model_info
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error loading model: {str(e)}'
        })

@app.route('/get_stock_data', methods=['POST'])
def get_stock_data():
    """Get stock data for a given ticker"""
    data = request.json
    ticker = data.get('ticker', 'TSLA')
    
    try:
        # Default to 1 year of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Download data
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        if stock_data.empty:
            return jsonify({
                'status': 'error',
                'message': f'No data found for {ticker}'
            })
        
        # Create a simple price chart
        plt.figure(figsize=(10, 6))
        plt.plot(stock_data.index, stock_data['Close'])
        plt.title(f'{ticker} Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.grid(True, alpha=0.3)
        
        # Save chart to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Calculate basic stats - FIXED: Extract float values directly
        current_price = float(stock_data['Close'].iloc[-1])
        prev_price = float(stock_data['Close'].iloc[-2])
        price_change = ((current_price / prev_price) - 1) * 100
        
        # Format the data for the response
        price_data = []
        for date, row in stock_data.tail(30).iterrows():
            price_data.append({
                'date': date.strftime('%Y-%m-%d'),
                # FIXED: Convert pandas values to native Python types
                'price': float(round(row['Close'], 2)),
                'volume': int(row['Volume'])
            })
        
        return jsonify({
            'status': 'success',
            'ticker': ticker,
            'current_price': round(current_price, 2),
            'price_change': round(price_change, 2),
            'price_data': price_data,
            'chart': f'data:image/png;base64,{image_base64}'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error getting stock data: {str(e)}'
        })

@app.route('/get_backtest_results', methods=['GET'])
def get_backtest_results():
    """Return the backtest results from the model"""
    try:
        model_id = request.args.get('model_id', 'default')
        
        # In a real app, you might run the backtest or load the results from a file
        # For now, we'll just return the hardcoded results from the logs
        
        # Different results based on the model selected
        if model_id == 'bitcoin':
            backtest_results = {
                'buy_actions': 212,
                'hold_actions': 0,
                'sell_actions': 31,
                'total_trades': 52,
                'initial_portfolio': 10000.00,
                'final_portfolio': 14650.32,
                'performance': 46.50,
                'maximum_drawdown': 0.17,
                'buy_and_hold_value': 12500.75,
                'vs_buy_hold': 17.20
            }
        elif model_id == 'advanced':
            backtest_results = {
                'buy_actions': 278,
                'hold_actions': 12,
                'sell_actions': 36,
                'total_trades': 62,
                'initial_portfolio': 10000.00,
                'final_portfolio': 16320.45,
                'performance': 63.20,
                'maximum_drawdown': 0.15,
                'buy_and_hold_value': 12890.30,
                'vs_buy_hold': 26.61
            }
        else:
            backtest_results = {
                'buy_actions': 244,
                'hold_actions': 0,
                'sell_actions': 22,
                'total_trades': 45,
                'initial_portfolio': 10000.00,
                'final_portfolio': 13790.85,
                'performance': 37.91,
                'maximum_drawdown': 0.19,
                'buy_and_hold_value': 11772.03,
                'vs_buy_hold': 17.15
            }
        
        # Generate mock trade history
        trade_history = []
        
        # Generate sample trade data
        import random
        from datetime import datetime, timedelta
        
        # Start date for our mock data
        current_date = datetime.now() - timedelta(days=180)
        
        # Mock cryptocurrency price starting at around $20,000
        price = 20000
        
        # Portfolio and crypto holdings
        portfolio_usd = 10000
        crypto_holdings = 0
        
        # Generate trade history with realistic dates, prices, and actions
        for i in range(60):  # Generate 60 sample trades
            current_date += timedelta(days=random.randint(1, 5))
            
            # Simulate some price movement
            price_change = random.uniform(-0.05, 0.05)  # -5% to +5% change
            price = price * (1 + price_change)
            
            # Decide action based on some basic probability
            # Roughly match the proportions from the backtest results
            action_roll = random.random()
            
            if model_id == 'bitcoin':
                buy_probability = 0.75  # Slightly different proportions for Bitcoin model
            elif model_id == 'advanced':
                buy_probability = 0.70  # Different proportions for Advanced model
                # Sometimes do a hold action for the advanced model
                if action_roll > 0.92:
                    trade_history.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'action': "HOLD",
                        'price': round(price, 2),
                        'amount': 0,
                        'value': 0,
                        'portfolio_value': round(portfolio_usd + (crypto_holdings * price), 2)
                    })
                    continue
            else:
                buy_probability = 0.8
            
            if action_roll < buy_probability:
                action = "BUY"
                amount = round(random.uniform(0.01, 0.1), 4)  # Buy between 0.01 and 0.1 BTC
                crypto_holdings += amount
                portfolio_usd -= amount * price
            else:
                action = "SELL"
                if crypto_holdings > 0:
                    amount = round(min(random.uniform(0.01, 0.1), crypto_holdings), 4)
                    crypto_holdings -= amount
                    portfolio_usd += amount * price
                else:
                    # Skip this iteration if we have nothing to sell
                    continue
            
            trade_history.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'action': action,
                'price': round(price, 2),
                'amount': amount,
                'value': round(amount * price, 2),
                'portfolio_value': round(portfolio_usd + (crypto_holdings * price), 2)
            })
        
        return jsonify({
            'status': 'success',
            'backtest_results': backtest_results,
            'trade_history': trade_history
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error getting backtest results: {str(e)}'
        })

if __name__ == '__main__':
    print("Starting Flask app...")
    print(f"Looking for default model at: {DEFAULT_MODEL_PATH}")
    print(f"Looking for Bitcoin model at: {BITCOIN_MODEL_PATH}")
    print(f"Looking for Advanced model at: {ADVANCED_MODEL_PATH}")
    print("Default model exists:", os.path.exists(DEFAULT_MODEL_PATH))
    print("Bitcoin model exists:", os.path.exists(BITCOIN_MODEL_PATH))
    print("Advanced model exists:", os.path.exists(ADVANCED_MODEL_PATH))
    app.run(debug=True)