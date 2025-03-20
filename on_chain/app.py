from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
import copy
import io
import base64
import os
import sys
import traceback
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Try to import backtrader for strategy backtesting
try:
    import backtrader as bt
    BACKTRADER_AVAILABLE = True
except ImportError:
    BACKTRADER_AVAILABLE = False
    print("Backtrader not available. Using simplified backtesting only.")

# Patch matplotlib to avoid plt.show() in on_chain_2.py module
original_show = plt.show

def patched_show(*args, **kwargs):
    # This function does nothing - it prevents interactive plots from being shown
    pass

# Monkey patch plt.show to prevent it from being called
plt.show = patched_show

# Try to import on-chain analysis module
try:
    from on_chain_2 import IntegratedBitcoinStrategy, run_integrated_strategy
    ON_CHAIN_AVAILABLE = True
    print("On-chain analysis module loaded successfully.")
except ImportError:
    ON_CHAIN_AVAILABLE = False
    print("On-chain analysis module not available. Only standard analysis will be used.")

# Restore original plt.show for other parts of the code
plt.show = original_show

# Try to import TensorFlow to check if it's available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow is available.")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow is not available. Some advanced features will be disabled.")

# Set display options and random seeds for reproducibility
pd.set_option('display.float_format', '{:.4f}'.format)
plt.style.use('ggplot')
np.random.seed(42)
random.seed(42)

app = Flask(__name__)
app.secret_key = 'trading_analysis_dashboard_secret_key'

def load_stock_data(ticker, start_date, end_date):
    """Load stock data using yfinance and standardize column names"""
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Standardize column names (flatten multi-index columns)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0].lower() for col in data.columns]
    else:
        data.columns = [col.lower() for col in data.columns]
    
    return data

def add_technical_features(df):
    """Add technical indicators as features for ML models"""
    df_features = df.copy()
    
    # Check for 'close' or 'adj close' column
    price_column = None
    volume_column = None
    
    if 'close' in df_features.columns:
        price_column = 'close'
    elif 'adj close' in df_features.columns:
        price_column = 'adj close'
    elif 'adjclose' in df_features.columns:
        price_column = 'adjclose'
    else:
        # Try to find any column with 'close' in it
        close_cols = [col for col in df_features.columns if 'close' in col.lower()]
        if close_cols:
            price_column = close_cols[0]
        else:
            raise ValueError("Could not find a 'close' price column in the data")
    
    if 'volume' in df_features.columns:
        volume_column = 'volume'
    else:
        volume_cols = [col for col in df_features.columns if 'volume' in col.lower()]
        if volume_cols:
            volume_column = volume_cols[0]
        else:
            # Create a dummy volume column
            df_features['volume'] = 1000000
            volume_column = 'volume'
    
    # Momentum factor: 5-day price change
    df_features['momentum_5'] = df_features[price_column] / df_features[price_column].shift(5) - 1
    
    # Volume factor: (5-day avg volume) / (10-day avg volume) - 1
    df_features['vol_ratio'] = (df_features[volume_column].rolling(5).mean() / 
                               df_features[volume_column].rolling(10).mean() - 1)
    
    # RSI (14-day)
    delta = df_features[price_column].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    # Avoid division by zero
    rs = gain / loss.replace(0, np.finfo(float).eps)
    df_features['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20-day, 2 std)
    sma_20 = df_features[price_column].rolling(window=20).mean()
    std_20 = df_features[price_column].rolling(window=20).std()
    df_features['BB_upper'] = sma_20 + 2 * std_20
    df_features['BB_lower'] = sma_20 - 2 * std_20
    
    # Target variable: next day return
    df_features['future_ret_1d'] = df_features[price_column].pct_change().shift(-1)
    
    # Add 'close' column for compatibility if it doesn't exist
    if price_column != 'close':
        df_features['close'] = df_features[price_column]
    
    # Remove NaN values
    df_features.dropna(inplace=True)
    
    return df_features

def split_data(df, train_pct=0.6, val_pct=0.2):
    """Split data into training, validation, and test sets"""
    train_idx = int(len(df) * train_pct)
    val_idx = int(len(df) * (train_pct + val_pct))
    
    train_data = df.iloc[:train_idx].copy()
    val_data = df.iloc[train_idx:val_idx].copy()
    test_data = df.iloc[val_idx:].copy()
    
    split_info = {
        'train': {
            'start': train_data.index.min().strftime('%Y-%m-%d'),
            'end': train_data.index.max().strftime('%Y-%m-%d'),
            'count': len(train_data)
        },
        'val': {
            'start': val_data.index.min().strftime('%Y-%m-%d'),
            'end': val_data.index.max().strftime('%Y-%m-%d'),
            'count': len(val_data)
        },
        'test': {
            'start': test_data.index.min().strftime('%Y-%m-%d'),
            'end': test_data.index.max().strftime('%Y-%m-%d'),
            'count': len(test_data)
        }
    }
    
    return train_data, val_data, test_data, split_info

def prepare_features_target(train_data, val_data, test_data, features):
    """Extract feature matrices and target vectors from dataframes"""
    X_train = train_data[features].values
    y_train = train_data['future_ret_1d'].values
    
    X_val = val_data[features].values
    y_val = val_data['future_ret_1d'].values
    
    X_test = test_data[features].values
    y_test = test_data['future_ret_1d'].values
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_linear_model(X_train, y_train, X_val, y_val):
    """Train a linear regression model with hyperparameter tuning"""
    pipeline = Pipeline([('lr', LinearRegression())])
    
    param_grid = {'lr__fit_intercept': [True, False]}
    
    best_score = float('-inf')
    best_params = None
    best_model = None
    
    for params in ParameterGrid(param_grid):
        pipeline.set_params(**params)
        pipeline.fit(X_train, y_train)
        
        val_pred = pipeline.predict(X_val)
        val_r2 = r2_score(y_val, val_pred)
        
        if val_r2 > best_score:
            best_score = val_r2
            best_params = params
            best_model = copy.deepcopy(pipeline)
    
    return best_model, best_score, best_params


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train a random forest model with reduced hyperparameter grid for web app"""
    pipeline = Pipeline([('rf', RandomForestRegressor(random_state=42))])
    
    # Simplified param grid for quicker web response
    param_grid = {
        'rf__n_estimators': [100, 500],
        'rf__max_depth': [3, 10],
        'rf__min_samples_split': [2, 5]
    }
    
    best_score = float('-inf')
    best_params = None
    best_model = None
    
    for params in ParameterGrid(param_grid):
        pipeline.set_params(**params)
        pipeline.fit(X_train, y_train)
        
        val_pred = pipeline.predict(X_val)
        val_r2 = r2_score(y_val, val_pred)
        
        if val_r2 > best_score:
            best_score = val_r2
            best_params = params
            best_model = copy.deepcopy(pipeline)
    
    return best_model, best_score, best_params


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train an XGBoost model with reduced hyperparameter grid for web app"""
    pipeline = Pipeline([('xgb', XGBRegressor(random_state=42, verbosity=0))])
    
    # Simplified param grid for quicker web response
    param_grid = {
        'xgb__n_estimators': [100, 500],
        'xgb__learning_rate': [0.01, 0.1],
        'xgb__max_depth': [3, 5]
    }
    
    best_score = float('-inf')
    best_params = None
    best_model = None
    
    for params in ParameterGrid(param_grid):
        pipeline.set_params(**params)
        pipeline.fit(X_train, y_train)
        
        val_pred = pipeline.predict(X_val)
        val_r2 = r2_score(y_val, val_pred)
        
        if val_r2 > best_score:
            best_score = val_r2
            best_params = params
            best_model = copy.deepcopy(pipeline)
    
    return best_model, best_score, best_params


def train_mlp(X_train, y_train, X_val, y_val):
    """Train a neural network (MLP) model with reduced hyperparameter grid for web app"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(random_state=42, max_iter=500))
    ])
    
    # Simplified param grid for quicker web response
    param_grid = {
        'mlp__hidden_layer_sizes': [(64, 64), (128, 128)],
        'mlp__alpha': [1e-3, 1e-2],
        'mlp__learning_rate_init': [1e-3, 1e-2]
    }
    
    best_score = float('-inf')
    best_params = None
    best_model = None
    
    for params in ParameterGrid(param_grid):
        pipeline.set_params(**params)
        pipeline.fit(X_train, y_train)
        
        val_pred = pipeline.predict(X_val)
        val_r2 = r2_score(y_val, val_pred)
        
        if val_r2 > best_score:
            best_score = val_r2
            best_params = params
            best_model = copy.deepcopy(pipeline)
    
    return best_model, best_score, best_params


def optimize_ensemble_weights(models, X_val, y_val, sum_to_1=True, nonnegative=True):
    """Optimize ensemble weights using R² scores for web app"""
    # Get predictions from each model
    predictions = np.column_stack([model.predict(X_val) for model in models])
    N, M = predictions.shape
    
    # Simple approach: use R² scores as weights
    r2_scores = []
    for i in range(M):
        r2 = r2_score(y_val, predictions[:, i])
        # Handle negative R² values
        r2_scores.append(max(0.0001, r2))
    
    # Normalize weights to sum to 1
    w_opt = np.array(r2_scores) / sum(r2_scores) if sum(r2_scores) > 0 else np.ones(M) / M
    
    # Calculate final prediction and R²
    y_val_pred = predictions @ w_opt
    r2_val = r2_score(y_val, y_val_pred)
    
    return w_opt, r2_val


def calculate_performance_metrics(y_true, y_pred, model_name):
    """Calculate key performance metrics for a model"""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'model': model_name,
        'mse': f"{mse:.8f}",
        'r2': f"{r2:.4f}"
    }


class VolumeIndicator(bt.Indicator):
    """Custom volume indicator for backtrader chart"""
    lines = ('vol',)
    plotinfo = dict(subplot=True, plotname='Volume')
    
    def __init__(self):
        self.lines.vol = self.data.volume


class MLEnsembleStrategy(bt.Strategy):
    """Trading strategy that uses ML ensemble predictions"""
    params = (
        ('target_percent', 0.98),  # Target position size
    )
    
    def __init__(self, models, weights, features):
        self.models = models
        self.weights = weights
        self.features = features
        
        # Turn off volume in main plot
        self.data.plotinfo.plotvolume = False
        
        # Add custom volume indicator
        self.vol_ind = VolumeIndicator(self.data)
        self.vol_5 = bt.indicators.SMA(self.vol_ind.vol, period=5)
        self.vol_10 = bt.indicators.SMA(self.vol_ind.vol, period=10)
        
        # Technical indicators
        self.momentum_5 = bt.indicators.PercentChange(self.data.close, period=5)
        self.rsi_14 = bt.indicators.RSI(self.data.close, period=14)
        self.bb = bt.indicators.BollingerBands(self.data.close, period=20, devfactor=2)
        
        # Track portfolio value history
        self.value_history_dates = []
        self.value_history_values = []
    
    def next(self):
        # Calculate features
        momentum = self.momentum_5[0]
        vol_ratio = (self.vol_5[0] / self.vol_10[0] - 1) if self.vol_10[0] != 0 else 0
        rsi = self.rsi_14[0]
        bb_upper = self.bb.top[0]
        bb_lower = self.bb.bot[0]
        
        # Create feature vector
        X = np.array([[momentum, vol_ratio, rsi, bb_upper, bb_lower]])
        
        # Get ensemble prediction
        predictions = np.array([model.predict(X)[0] for model in self.models])
        pred_ret = np.sum(predictions * self.weights)
        
        # Get current position
        current_position = self.getposition().size
        
        # Execute trades based on predictions
        if pred_ret > 0 and current_position == 0:
            # Buy signal
            self.order_target_percent(target=self.p.target_percent)
        elif pred_ret <= 0 and current_position > 0:
            # Sell signal
            self.order_target_percent(target=0.0)
        
        # Record portfolio value
        self.value_history_dates.append(self.data.datetime.date(0))
        self.value_history_values.append(self.broker.getvalue())


class BuyAndHoldStrategy(bt.Strategy):
    """Simple buy and hold strategy for comparison"""
    def __init__(self):
        self.value_history_dates = []
        self.value_history_values = []
    
    def next(self):
        # Buy on first day
        if len(self) == 1:
            self.order_target_percent(target=0.98)
        
        # Record portfolio value
        self.value_history_dates.append(self.data.datetime.date(0))
        self.value_history_values.append(self.broker.getvalue())


def calculate_strategy_metrics(strategy):
    """Calculate key performance metrics for a strategy"""
    # Extract portfolio values and dates
    values = np.array(strategy.value_history_values)
    dates = strategy.value_history_dates
    
    # Calculate returns (daily)
    returns = np.diff(values) / values[:-1]
    
    # Calculate total and annualized returns
    total_return_pct = (values[-1] / values[0] - 1) * 100
    days = (dates[-1] - dates[0]).days
    ann_return_pct = ((1 + total_return_pct/100) ** (365/days) - 1) * 100 if days > 0 else 0
    
    # Calculate Sharpe ratio (assuming 0% risk-free rate for simplicity)
    # Annualized Sharpe = (Mean Daily Return / Daily Std Dev) * sqrt(252)
    daily_return_mean = np.mean(returns)
    daily_return_std = np.std(returns)
    sharpe_ratio = (daily_return_mean / daily_return_std) * np.sqrt(252) if daily_return_std > 0 else 0
    
    # Calculate maximum drawdown
    peak = values[0]
    max_drawdown = 0
    for value in values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    max_drawdown_pct = max_drawdown * 100
    
    # Calculate Calmar ratio (annualized return / maximum drawdown)
    calmar_ratio = ann_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else float('inf')
    
    return {
        'total_return_pct': f"{total_return_pct:.2f}%",
        'annualized_return_pct': f"{ann_return_pct:.2f}%",
        'sharpe_ratio': f"{sharpe_ratio:.2f}",
        'max_drawdown_pct': f"{max_drawdown_pct:.2f}%",
        'calmar_ratio': f"{calmar_ratio:.2f}"
    }


def run_backtest(df, strategy_class, strategy_params=None, initial_cash=100000):
    """Run backtrader backtest with the given strategy"""
    cerebro = bt.Cerebro()
    
    # Add data feed
    data_feed = bt.feeds.PandasData(
        dataname=df,
        datetime=None,  # Use index as datetime
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=None
    )
    cerebro.adddata(data_feed)
    
    # Add strategy
    if strategy_params:
        cerebro.addstrategy(strategy_class, **strategy_params)
    else:
        cerebro.addstrategy(strategy_class)
    
    # Set initial cash and commission
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)  # 0.1%
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # Run backtest
    results = cerebro.run()
    
    return results[0]


def run_trading_strategies(test_data, models, weights, features):
    """Run both ML Ensemble and Buy & Hold strategies and compare results"""
    # Run ML Ensemble strategy
    ml_strategy = run_backtest(
        test_data,
        MLEnsembleStrategy,
        {'models': models, 'weights': weights, 'features': features, 'target_percent': 0.98},
        initial_cash=100000
    )
    
    # Run Buy and Hold strategy
    bh_strategy = run_backtest(
        test_data,
        BuyAndHoldStrategy,
        initial_cash=100000
    )
    
    # Calculate performance metrics
    ml_metrics = calculate_strategy_metrics(ml_strategy)
    bh_metrics = calculate_strategy_metrics(bh_strategy)
    
    # Create comparison chart
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ml_strategy.value_history_dates, ml_strategy.value_history_values, 
             label='ML Ensemble Strategy', linewidth=2)
    plt.plot(bh_strategy.value_history_dates, bh_strategy.value_history_values, 
             label='Buy and Hold Strategy', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Strategy Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    strategy_chart = fig_to_base64(fig)
    
    # Return metrics and chart
    return {
        'ml_metrics': ml_metrics,
        'bh_metrics': bh_metrics,
        'comparison_chart': strategy_chart
    }


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML display"""
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_data


def run_standard_analysis(ticker, start_date_str, end_date_str):
    """Run the complete stock analysis pipeline and return results for Flask"""
    try:
        # Convert date strings to datetime objects
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        
        # Load stock data
        data = load_stock_data(ticker, start_date, end_date)
        if data.empty:
            return {'error': f"No data found for ticker {ticker} in the specified date range."}
        
        # Add technical indicators as features
        df = add_technical_features(data)
        
        # Define feature set
        features = ['momentum_5', 'vol_ratio', 'RSI_14', 'BB_upper', 'BB_lower']
        
        # Data preview
        data_preview = df[['close'] + features].tail(5).reset_index()
        data_preview_html = data_preview.to_html(classes='table table-striped table-bordered')
        
        # Target variable stats
        target_stats = {
            'mean': f"{df['future_ret_1d'].mean():.6f}",
            'std': f"{df['future_ret_1d'].std():.6f}",
            'min': f"{df['future_ret_1d'].min():.6f}",
            'max': f"{df['future_ret_1d'].max():.6f}"
        }
        
        # Visualize returns distribution
        fig_dist = plt.figure(figsize=(10, 5))
        sns.histplot(df['future_ret_1d'], bins=50)
        plt.title('Next-Day Return Distribution')
        plt.xlabel('Return')
        fig_dist_base64 = fig_to_base64(fig_dist)
        
        # Correlation analysis
        fig_corr = plt.figure(figsize=(10, 8))
        corr = df[['close', 'future_ret_1d'] + features].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        fig_corr_base64 = fig_to_base64(fig_corr)
        
        # Split data into train/validation/test sets
        train_data, val_data, test_data, split_info = split_data(df)
        
        # Visualize data splits
        fig_splits = plt.figure(figsize=(15, 6))
        plt.plot(train_data.index, train_data['close'], label='Training', color='blue')
        plt.plot(val_data.index, val_data['close'], label='Validation', color='green')
        plt.plot(test_data.index, test_data['close'], label='Test', color='red')
        
        # Add split points
        split_date_1 = train_data.index[-1]
        split_date_2 = val_data.index[-1]
        plt.axvline(split_date_1, color='black', linestyle='--')
        plt.axvline(split_date_2, color='black', linestyle='--')
        
        plt.title('Data Split: Training, Validation, and Test Sets')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)
        fig_splits_base64 = fig_to_base64(fig_splits)
        
        # Prepare features and targets
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_features_target(
            train_data, val_data, test_data, features
        )
        
        # Train models with hyperparameter tuning (with reduced grids for web app)
        lr_model, lr_score, lr_params = train_linear_model(X_train, y_train, X_val, y_val)
        rf_model, rf_score, rf_params = train_random_forest(X_train, y_train, X_val, y_val)
        xgb_model, xgb_score, xgb_params = train_xgboost(X_train, y_train, X_val, y_val)
        mlp_model, mlp_score, mlp_params = train_mlp(X_train, y_train, X_val, y_val)
        
        # Model validation scores
        validation_scores = [
            {'model': 'Linear Regression', 'r2': f"{lr_score:.4f}", 'params': str(lr_params)},
            {'model': 'Random Forest', 'r2': f"{rf_score:.4f}", 'params': str(rf_params)},
            {'model': 'XGBoost', 'r2': f"{xgb_score:.4f}", 'params': str(xgb_params)},
            {'model': 'MLP', 'r2': f"{mlp_score:.4f}", 'params': str(mlp_params)}
        ]
        
        # Evaluate individual models on test set
        models = [lr_model, rf_model, xgb_model, mlp_model]
        model_names = ['Linear Regression', 'Random Forest', 'XGBoost', 'MLP']
        
        test_performance = []
        for name, model in zip(model_names, models):
            y_pred = model.predict(X_test)
            metrics = calculate_performance_metrics(y_test, y_pred, name)
            test_performance.append(metrics)
        
        # Feature importance from Random Forest
        rf_importances = rf_model.named_steps['rf'].feature_importances_
        feature_importance = []
        for feature, importance in zip(features, rf_importances):
            feature_importance.append({'feature': feature, 'importance': f"{importance:.4f}"})
        
        # Sort feature importances
        feature_importance.sort(key=lambda x: float(x['importance']), reverse=True)
        
        # Visualize feature importance
        fig_importance = plt.figure(figsize=(10, 6))
        importances = [float(item['importance']) for item in feature_importance]
        feature_names = [item['feature'] for item in feature_importance]
        plt.barh(range(len(importances)), importances, align='center')
        plt.yticks(range(len(importances)), feature_names)
        plt.xlabel('Feature Importance')
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        fig_importance_base64 = fig_to_base64(fig_importance)
        
        # Optimize ensemble weights
        weights, ensemble_val_r2 = optimize_ensemble_weights(
            models, X_val, y_val, sum_to_1=True, nonnegative=True
        )
        
        ensemble_weights = [
            {'model': name, 'weight': f"{weight:.4f}"}
            for name, weight in zip(model_names, weights)
        ]
        
        # Evaluate ensemble on test set
        test_predictions = np.column_stack([model.predict(X_test) for model in models])
        ensemble_pred = test_predictions @ weights
        ensemble_metrics = calculate_performance_metrics(y_test, ensemble_pred, 'Ensemble')
        
        # Run backtest strategies
        strategy_metrics = run_trading_strategies(
            test_data.copy(), 
            models, 
            weights, 
            features
        )
        
        # Visualize model predictions vs actual
        # Select a subset of test data for clearer visualization
        vis_start = max(0, len(test_data) - 100)
        test_subset = test_data.iloc[vis_start:].copy()
        X_test_subset = test_subset[features].values
        
        # Get predictions for each model and the ensemble
        pred_data = test_subset.copy()
        for i, (name, model) in enumerate(zip(model_names, models)):
            pred_data[f'pred_{name}'] = model.predict(X_test_subset)
        
        # Add ensemble prediction
        ensemble_subset_pred = np.column_stack([
            model.predict(X_test_subset) for model in models
        ]) @ weights
        pred_data['pred_Ensemble'] = ensemble_subset_pred
        
        # Create prediction vs actual visualization
        fig_predictions = plt.figure(figsize=(15, 8))
        plt.plot(pred_data.index, pred_data['future_ret_1d'], 'k-', label='Actual', linewidth=2)
        plt.plot(pred_data.index, pred_data['pred_Ensemble'], 'r--', label='Ensemble', linewidth=2)
        for name in model_names:
            plt.plot(pred_data.index, pred_data[f'pred_{name}'], '--', label=name, alpha=0.7)
        plt.title('Model Predictions vs Actual Returns')
        plt.xlabel('Date')
        plt.ylabel('Next-Day Return')
        plt.legend()
        plt.grid(True)
        fig_predictions_base64 = fig_to_base64(fig_predictions)
        
        # Return all results and visualizations
        return {
            'ticker': ticker,
            'start_date': start_date_str,
            'end_date': end_date_str,
            'data_preview': data_preview_html,
            'target_stats': target_stats,
            'split_info': split_info,
            'validation_scores': validation_scores,
            'test_performance': test_performance,
            'feature_importance': feature_importance,
            'ensemble_weights': ensemble_weights,
            'ensemble_metrics': ensemble_metrics,
            'strategy_metrics': strategy_metrics,
            'plots': {
                'returns_distribution': fig_dist_base64,
                'correlation_matrix': fig_corr_base64,
                'data_splits': fig_splits_base64,
                'feature_importance': fig_importance_base64,
                'predictions': fig_predictions_base64,
                'strategy_comparison': strategy_metrics['comparison_chart']
            },
            'analysis_type': 'standard',
            'success': True
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {'error': str(e), 'success': False}


def run_onchain_analysis(ticker, start_date_str, end_date_str, on_chain_data_dir=None, use_sample_data=True):
    """Run on-chain analysis for Bitcoin"""
    if not ON_CHAIN_AVAILABLE:
        return {'error': "On-chain analysis module not available.", 'success': False}
    
    try:
        # Convert date strings to datetime objects
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        
        # Run integrated strategy
        print(f"Running on-chain analysis for {ticker} from {start_date_str} to {end_date_str}")
        print(f"On-chain data directory: {on_chain_data_dir if on_chain_data_dir else 'None (using sample data)'}")
        
        # Create an integrated strategy object
        strategy = run_integrated_strategy(
            data_dir=on_chain_data_dir,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            use_backtrader=BACKTRADER_AVAILABLE,
            use_sample_data=use_sample_data,
            initial_cash=100000
        )
        
        if strategy is None:
            return {'error': "Failed to initialize on-chain analysis strategy.", 'success': False}
        
        # Gather results
        results = {
            'ticker': ticker,
            'start_date': start_date_str,
            'end_date': end_date_str,
            'analysis_type': 'on_chain',
            'success': True
        }
        
        # Add data information
        if strategy.features is not None:
            # Data preview
            data_preview = strategy.features.tail(5).reset_index()
            # Get the actual name of the index column after reset_index()
            index_column_name = data_preview.columns[0]  # First column after reset_index is the former index
            
            columns_to_show = ['close']
            for col in ['returns', 'volatility_14d', 'rsi_14', 'macd']:
                if col in data_preview.columns:
                    columns_to_show.append(col)
            if 'nvt_ratio' in data_preview.columns:
                columns_to_show.append('nvt_ratio')
            if 'hodl_ratio' in data_preview.columns:
                columns_to_show.append('hodl_ratio')
            
            # Ensure we don't try to display too many columns
            columns_to_show = columns_to_show[:8]
            # Use the actual index column name instead of assuming it's called 'index'
            data_preview_html = data_preview[[index_column_name] + columns_to_show].to_html(classes='table table-striped table-bordered')
            results['data_preview'] = data_preview_html
            
            # Get data splits
            train_data, val_data, test_data = strategy.split_data(train_pct=0.6, val_pct=0.2)
            results['split_info'] = {
                'train': {
                    'start': train_data.index.min().strftime('%Y-%m-%d'),
                    'end': train_data.index.max().strftime('%Y-%m-%d'),
                    'count': len(train_data)
                },
                'val': {
                    'start': val_data.index.min().strftime('%Y-%m-%d'),
                    'end': val_data.index.max().strftime('%Y-%m-%d'),
                    'count': len(val_data)
                },
                'test': {
                    'start': test_data.index.min().strftime('%Y-%m-%d'),
                    'end': test_data.index.max().strftime('%Y-%m-%d'),
                    'count': len(test_data)
                }
            }
            
            # Target stats
            if 'future_ret_7d' in strategy.features.columns:
                target_col = 'future_ret_7d'
            elif 'future_ret_1d' in strategy.features.columns:
                target_col = 'future_ret_1d'
            else:
                target_col = 'returns'
                
            results['target_stats'] = {
                'mean': f"{strategy.features[target_col].mean():.6f}",
                'std': f"{strategy.features[target_col].std():.6f}",
                'min': f"{strategy.features[target_col].min():.6f}",
                'max': f"{strategy.features[target_col].max():.6f}"
            }
        
        # Add model information
        # Check if random forest model is available
        if 'random_forest' in strategy.models:
            rf_model, rf_features = strategy.models['random_forest']
            
            # Feature importance from Random Forest
            try:
                rf_importances = rf_model.feature_importances_
                feature_importance = []
                for feature, importance in zip(rf_features, rf_importances):
                    feature_importance.append({'feature': feature, 'importance': f"{importance:.4f}"})
                
                # Sort feature importances
                feature_importance.sort(key=lambda x: float(x['importance']), reverse=True)
                results['feature_importance'] = feature_importance[:10]  # Top 10 features
            except:
                print("Could not extract feature importance from RF model")
        
        # Add ensemble weights (if available)
        if hasattr(strategy, 'ensemble_weights') and strategy.ensemble_weights is not None:
            ensemble_weights = []
            if 'linear_regression' in strategy.models:
                ensemble_weights.append({'model': 'Linear Regression', 'weight': f"{strategy.ensemble_weights[0]:.4f}"})
            if 'xgboost' in strategy.models:
                ensemble_weights.append({'model': 'XGBoost', 'weight': f"{strategy.ensemble_weights[1]:.4f}"})
            if 'mlp' in strategy.models:
                ensemble_weights.append({'model': 'MLP', 'weight': f"{strategy.ensemble_weights[2]:.4f}"})
            
            results['ensemble_weights'] = ensemble_weights
        
        # Add backtest performance
        if hasattr(strategy, 'performance') and strategy.performance:
            # Convert performance metrics to match standard format
            if isinstance(strategy.performance, dict):
                # If already in right format
                if 'total_return_pct' in strategy.performance:
                    strategy_metrics = {
                        'ml_metrics': strategy.performance,
                        'bh_metrics': {
                            'total_return_pct': "0.00%",
                            'annualized_return_pct': "0.00%",
                            'sharpe_ratio': "0.00",
                            'max_drawdown_pct': "0.00%",
                            'calmar_ratio': "0.00"
                        }
                    }
                else:
                    # Convert format
                    strategy_metrics = {
                        'ml_metrics': {
                            'total_return_pct': f"{strategy.performance.get('Annual Return', 0) * 100:.2f}%",
                            'annualized_return_pct': f"{strategy.performance.get('Annual Return', 0) * 100:.2f}%",
                            'sharpe_ratio': f"{strategy.performance.get('Sharpe Ratio', 0):.2f}",
                            'max_drawdown_pct': f"{abs(strategy.performance.get('Max Drawdown', 0)) * 100:.2f}%",
                            'calmar_ratio': f"{strategy.performance.get('Calmar Ratio', 0):.2f}"
                        },
                        'bh_metrics': {
                            'total_return_pct': "0.00%",
                            'annualized_return_pct': "0.00%",
                            'sharpe_ratio': "0.00",
                            'max_drawdown_pct': "0.00%",
                            'calmar_ratio': "0.00"
                        }
                    }
            else:
                # If format is unexpected, create placeholder
                strategy_metrics = {
                    'ml_metrics': {
                        'total_return_pct': "0.00%",
                        'annualized_return_pct': "0.00%",
                        'sharpe_ratio': "0.00",
                        'max_drawdown_pct': "0.00%",
                        'calmar_ratio': "0.00"
                    },
                    'bh_metrics': {
                        'total_return_pct': "0.00%",
                        'annualized_return_pct': "0.00%",
                        'sharpe_ratio': "0.00",
                        'max_drawdown_pct': "0.00%",
                        'calmar_ratio': "0.00"
                    }
                }
                
            results['strategy_metrics'] = strategy_metrics
        
        # Add plots - capture all plots generated by the on_chain strategy
        plt.figure(figsize=(15, 6))
        plt.plot(strategy.features.index, strategy.features['close'], label='Bitcoin Price')
        if 'market_regime' in strategy.features.columns:
            # Color based on regimes
            for regime in range(strategy.features['market_regime'].nunique()):
                mask = strategy.features['market_regime'] == regime
                plt.scatter(
                    strategy.features.index[mask], 
                    strategy.features['close'][mask],
                    label=f'Regime {regime}',
                    s=10,
                    alpha=0.7
                )
        plt.title('Bitcoin Price with Market Regimes')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        fig_regimes_base64 = fig_to_base64(plt.gcf())
        
        # Generate strategy plots
        if hasattr(strategy, 'signals') and strategy.signals is not None:
            # Returns distribution
            plt.figure(figsize=(10, 5))
            if 'returns' in strategy.signals.columns:
                sns.histplot(strategy.signals['returns'].dropna(), bins=50)
                plt.title('Daily Return Distribution')
                plt.xlabel('Return')
                fig_dist_base64 = fig_to_base64(plt.gcf())
            else:
                fig_dist_base64 = None
            
            # Trading signals visualization
            if 'signal' in strategy.signals.columns:
                plt.figure(figsize=(15, 8))
                plt.plot(strategy.signals.index, strategy.signals['close'], color='blue', alpha=0.5, label='Price')
                
                # Plot buy signals
                buys = strategy.signals[strategy.signals['signal'] > 0].index
                sells = strategy.signals[strategy.signals['signal'] < 0].index
                
                if len(buys) > 0:
                    plt.scatter(buys, strategy.signals.loc[buys, 'close'], 
                              marker='^', color='green', s=100, label='Buy Signal')
                    
                if len(sells) > 0:
                    plt.scatter(sells, strategy.signals.loc[sells, 'close'], 
                              marker='v', color='red', s=100, label='Sell Signal')
                
                plt.title('Trading Signals')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(True)
                fig_signals_base64 = fig_to_base64(plt.gcf())
            else:
                fig_signals_base64 = None
                
            # Strategy performance
            if 'cum_strategy_returns' in strategy.signals.columns and 'cum_market_returns' in strategy.signals.columns:
                plt.figure(figsize=(15, 8))
                plt.plot(strategy.signals.index, strategy.signals['cum_strategy_returns'] * 100, 
                       label='ML Strategy', linewidth=2)
                plt.plot(strategy.signals.index, strategy.signals['cum_market_returns'] * 100, 
                       label='Buy and Hold', linewidth=2)
                plt.title('Strategy Performance Comparison')
                plt.xlabel('Date')
                plt.ylabel('Cumulative Returns (%)')
                plt.legend()
                plt.grid(True)
                fig_strategy_base64 = fig_to_base64(plt.gcf())
            else:
                fig_strategy_base64 = None
        else:
            fig_dist_base64 = None
            fig_signals_base64 = None
            fig_strategy_base64 = None
        
        # Backtrader strategy plot
        if hasattr(strategy, 'bt_strategies') and strategy.bt_strategies:
            try:
                plt.figure(figsize=(15, 8))
                ml_strategy = strategy.bt_strategies['ml_strategy']
                bh_strategy = strategy.bt_strategies['bh_strategy']
                
                plt.plot(ml_strategy.value_history_dates, ml_strategy.value_history_values, 
                       label='ML Ensemble Strategy', linewidth=2)
                plt.plot(bh_strategy.value_history_dates, bh_strategy.value_history_values, 
                       label='Buy and Hold Strategy', linewidth=2)
                plt.xlabel('Date')
                plt.ylabel('Portfolio Value ($)')
                plt.title('Strategy Performance Comparison')
                plt.legend()
                plt.grid(True)
                bt_strategy_base64 = fig_to_base64(plt.gcf())
            except:
                bt_strategy_base64 = None
        else:
            bt_strategy_base64 = None
            
        # Add plots to results
        results['plots'] = {
            'market_regimes': fig_regimes_base64,
            'returns_distribution': fig_dist_base64 if fig_dist_base64 else (fig_regimes_base64 if fig_regimes_base64 else None),
            'trading_signals': fig_signals_base64,
            'strategy_comparison': bt_strategy_base64 if bt_strategy_base64 else (fig_strategy_base64 if fig_strategy_base64 else None),
            'correlation_matrix': None,  # Placeholder
            'data_splits': None,  # Placeholder
            'feature_importance': None,  # Placeholder
            'predictions': None  # Placeholder
        }
        
        # Generate additional information for on-chain analysis
        # Add market regime information if available
        if 'market_regime' in strategy.features.columns:
            regime_info = []
            for regime in range(strategy.features['market_regime'].nunique()):
                regime_data = strategy.features[strategy.features['market_regime'] == regime]
                avg_return = regime_data['returns'].mean() * 252 if 'returns' in regime_data.columns else 0
                avg_vol = regime_data['volatility_30d'].mean() * np.sqrt(252) if 'volatility_30d' in regime_data.columns else 0
                avg_rsi = regime_data['rsi_14'].mean() if 'rsi_14' in regime_data.columns else 0
                
                regime_info.append({
                    'regime': regime,
                    'count': len(regime_data),
                    'ann_return': f"{avg_return:.2%}",
                    'ann_vol': f"{avg_vol:.2%}",
                    'avg_rsi': f"{avg_rsi:.2f}"
                })
            
            results['regime_info'] = regime_info
        
        # Add CNN model performance if available
        if 'cnn' in strategy.models:
            results['cnn_info'] = {
                'available': True,
                'lookback_window': strategy.models['cnn'][2]
            }
        else:
            results['cnn_info'] = {'available': False}
        
        # Add on-chain metric information if available
        on_chain_metrics = []
        for metric in ['nvt_ratio', 'hodl_ratio', 'mvrv_ratio']:
            if metric in strategy.features.columns:
                on_chain_metrics.append({
                    'name': metric,
                    'mean': f"{strategy.features[metric].mean():.4f}",
                    'std': f"{strategy.features[metric].std():.4f}",
                    'min': f"{strategy.features[metric].min():.4f}",
                    'max': f"{strategy.features[metric].max():.4f}"
                })
        
        results['on_chain_metrics'] = on_chain_metrics
        
        # Add validation scores for ensemble models
        validation_scores = []
        if hasattr(strategy, 'models'):
            if 'random_forest' in strategy.models:
                validation_scores.append({
                    'model': 'Random Forest',
                    'r2': '0.0000',  # Placeholder as we don't have this info directly
                    'params': 'n_estimators=100, max_depth=10'
                })
            if 'gradient_boosting' in strategy.models:
                validation_scores.append({
                    'model': 'Gradient Boosting',
                    'r2': '0.0000',  # Placeholder
                    'params': 'n_estimators=100, max_depth=5'
                })
            if 'linear_regression' in strategy.models:
                validation_scores.append({
                    'model': 'Linear Regression',
                    'r2': '0.0000',  # Placeholder
                    'params': 'fit_intercept=True'
                })
            if 'xgboost' in strategy.models:
                validation_scores.append({
                    'model': 'XGBoost',
                    'r2': '0.0000',  # Placeholder
                    'params': 'n_estimators=100, max_depth=3'
                })
            if 'mlp' in strategy.models:
                validation_scores.append({
                    'model': 'MLP',
                    'r2': '0.0000',  # Placeholder
                    'params': 'hidden_layer_sizes=(64,)'
                })
                
        results['validation_scores'] = validation_scores
        
        # Add ensemble metrics placeholder
        results['ensemble_metrics'] = {
            'model': 'Ensemble',
            'mse': '0.0000',
            'r2': '0.0000'
        }
        
        # Add test performance placeholder
        test_performance = []
        for model in validation_scores:
            test_performance.append({
                'model': model['model'],
                'mse': '0.0000',
                'r2': '0.0000'
            })
        
        results['test_performance'] = test_performance
        
        return results
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {'error': str(e), 'success': False}


def run_analysis(ticker, start_date, end_date, analysis_type='standard', on_chain_data_dir=None, use_sample_data=True):
    """Run either standard or on-chain analysis based on analysis_type"""
    if analysis_type == 'on_chain':
        if not ON_CHAIN_AVAILABLE:
            flash('On-chain analysis module not available. Running standard analysis instead.', 'warning')
            return run_standard_analysis(ticker, start_date, end_date)
        else:
            return run_onchain_analysis(ticker, start_date, end_date, on_chain_data_dir, use_sample_data)
    else:
        return run_standard_analysis(ticker, start_date, end_date)


@app.route('/')
def index():
    """Render main dashboard template"""
    # Default to TSLA for past 3 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    
    return render_template('index.html', 
                          default_ticker='TSLA',
                          default_start_date=start_date.strftime('%Y-%m-%d'),
                          default_end_date=end_date.strftime('%Y-%m-%d'),
                          on_chain_available=ON_CHAIN_AVAILABLE,
                          tensorflow_available=TENSORFLOW_AVAILABLE)


@app.route('/analyze', methods=['POST'])
def analyze():
    """Run analysis with user-provided parameters"""
    ticker = request.form.get('ticker', 'TSLA')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    analysis_type = request.form.get('analysis_type', 'standard')
    
    # On-chain specific parameters
    on_chain_data_dir = request.form.get('on_chain_data_dir', None)
    use_sample_data = request.form.get('use_sample_data', 'true') == 'true'
    
    # Check if on-chain data directory is valid
    if analysis_type == 'on_chain' and on_chain_data_dir:
        if not os.path.isdir(on_chain_data_dir):
            flash(f"The directory '{on_chain_data_dir}' does not exist or is not accessible. Using sample data instead.", 'warning')
            on_chain_data_dir = None
    
    # Validate inputs
    if not ticker or not start_date or not end_date:
        flash('Missing required parameters', 'danger')
        return redirect(url_for('index'))
    
    try:
        # Run the analysis
        results = run_analysis(
            ticker=ticker, 
            start_date=start_date, 
            end_date=end_date,
            analysis_type=analysis_type,
            on_chain_data_dir=on_chain_data_dir,
            use_sample_data=use_sample_data
        )
        
        # Check for errors
        if 'error' in results:
            flash(f"Analysis error: {results['error']}", 'danger')
            return redirect(url_for('index'))
            
        # Ensure all required keys exist in the results dictionary to avoid template errors
        required_keys = ['split_info', 'target_stats', 'validation_scores', 'test_performance', 
                         'feature_importance', 'ensemble_weights', 'ensemble_metrics', 
                         'strategy_metrics', 'plots']
                         
        # Add default values for any missing keys
        for key in required_keys:
            if key not in results:
                if key == 'split_info':
                    results[key] = {
                        'train': {'start': 'N/A', 'end': 'N/A', 'count': 0},
                        'val': {'start': 'N/A', 'end': 'N/A', 'count': 0},
                        'test': {'start': 'N/A', 'end': 'N/A', 'count': 0}
                    }
                elif key == 'target_stats':
                    results[key] = {'mean': 'N/A', 'std': 'N/A', 'min': 'N/A', 'max': 'N/A'}
                elif key == 'validation_scores' or key == 'test_performance':
                    results[key] = []
                elif key == 'feature_importance' or key == 'ensemble_weights':
                    results[key] = []
                elif key == 'ensemble_metrics':
                    results[key] = {'model': 'Ensemble', 'mse': 'N/A', 'r2': 'N/A'}
                elif key == 'strategy_metrics':
                    results[key] = {
                        'ml_metrics': {
                            'total_return_pct': '0.00%',
                            'annualized_return_pct': '0.00%',
                            'sharpe_ratio': '0.00',
                            'max_drawdown_pct': '0.00%',
                            'calmar_ratio': '0.00'
                        },
                        'bh_metrics': {
                            'total_return_pct': '0.00%',
                            'annualized_return_pct': '0.00%',
                            'sharpe_ratio': '0.00',
                            'max_drawdown_pct': '0.00%',
                            'calmar_ratio': '0.00'
                        }
                    }
                elif key == 'plots':
                    results[key] = {
                        'returns_distribution': None,
                        'correlation_matrix': None,
                        'data_splits': None,
                        'feature_importance': None,
                        'predictions': None,
                        'strategy_comparison': None
                    }
        
        # Render template with results
        return render_template('results.html', results=results)
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(error_traceback)
        flash(f"An error occurred: {str(e)}", 'danger')
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)