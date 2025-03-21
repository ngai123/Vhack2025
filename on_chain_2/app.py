import os
import io
import base64
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import matplotlib
matplotlib.use('Agg')  # Set backend to Agg for server-side plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Import the trading strategy code
from on_chain_3 import IntegratedBitcoinStrategy, add_enhanced_onchain_analysis, run_integrated_strategy

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store the last run strategy in memory for accessing results
current_strategy = None

# Default values
DEFAULT_TICKER = "BTC-USD"
DEFAULT_START_DATE = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')  # 2 years ago
DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')  # Today
DEFAULT_INITIAL_CASH = 100000
DEFAULT_USE_SAMPLE_DATA = True
DEFAULT_USE_ENHANCED_ONCHAIN = True
DEFAULT_USE_ENHANCED_TECHNICAL = True

@app.route('/')
def index():
    """Render the index page with the form"""
    return render_template('index.html', 
                          default_ticker=DEFAULT_TICKER,
                          default_start_date=DEFAULT_START_DATE,
                          default_end_date=DEFAULT_END_DATE,
                          default_initial_cash=DEFAULT_INITIAL_CASH)

@app.route('/run_strategy', methods=['POST'])
def run_strategy():
    """Run the trading strategy with the parameters from the form"""
    global current_strategy
    
    # Get form parameters
    ticker = request.form.get('ticker', DEFAULT_TICKER)
    start_date = request.form.get('start_date', DEFAULT_START_DATE)
    end_date = request.form.get('end_date', DEFAULT_END_DATE)
    initial_cash = float(request.form.get('initial_cash', DEFAULT_INITIAL_CASH))
    use_sample_data = 'use_sample_data' in request.form
    use_enhanced_onchain = 'use_enhanced_onchain' in request.form
    use_enhanced_technical = 'use_enhanced_technical' in request.form
    
    # Handle file upload for on-chain data
    data_dir = None
    if 'data_file' in request.files and request.files['data_file'].filename:
        data_file = request.files['data_file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], data_file.filename)
        data_file.save(file_path)
        data_dir = app.config['UPLOAD_FOLDER']
    
    # Convert date strings to datetime
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Fix for the ZeroDivisionError in visualization
    # We'll patch the visualization function to handle this case
    def patched_visualization(strategy):
        try:
            # Create our own visualization function that saves figures to memory
            figures = []
            
            # 1. Cumulative returns plot
            if hasattr(strategy, 'signals') and 'cum_market_returns' in strategy.signals.columns and 'cum_strategy_returns' in strategy.signals.columns:
                fig = plt.figure(figsize=(10, 6))
                plt.plot(strategy.signals.index, strategy.signals['cum_market_returns'] * 100, 
                        label='Buy and Hold', color='blue', alpha=0.7)
                plt.plot(strategy.signals.index, strategy.signals['cum_strategy_returns'] * 100, 
                        label='Trading Strategy', color='green')
                plt.title('Cumulative Returns: Strategy vs Buy-and-Hold')
                plt.xlabel('Date')
                plt.ylabel('Returns (%)')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                figures.append(fig)
            
            # 2. Drawdown plot
            if hasattr(strategy, 'signals') and 'strategy_returns' in strategy.signals.columns:
                try:
                    fig = plt.figure(figsize=(10, 6))
                    cum_returns = (1 + strategy.signals['strategy_returns']).cumprod()
                    running_max = cum_returns.cummax()
                    drawdown = (cum_returns / running_max - 1) * 100
                    
                    plt.plot(strategy.signals.index, drawdown)
                    plt.title('Strategy Drawdown')
                    plt.xlabel('Date')
                    plt.ylabel('Drawdown (%)')
                    plt.grid(True)
                    plt.tight_layout()
                    figures.append(fig)
                except Exception as e:
                    print(f"Error generating drawdown plot: {str(e)}")
            
            # 3. Signal distribution
            if hasattr(strategy, 'signals') and 'signal' in strategy.signals.columns:
                try:
                    fig = plt.figure(figsize=(10, 6))
                    strategy.signals['signal'].value_counts().plot(kind='bar')
                    plt.title('Distribution of Trading Signals')
                    plt.xlabel('Signal (-1: Sell, 0: Hold, 1: Buy)')
                    plt.ylabel('Frequency')
                    plt.grid(True, axis='y')
                    plt.tight_layout()
                    figures.append(fig)
                except Exception as e:
                    print(f"Error generating signal distribution plot: {str(e)}")
            
            # 4. Feature importance plot
            if 'random_forest' in strategy.models:
                try:
                    fig = plt.figure(figsize=(10, 6))
                    model, features = strategy.models['random_forest']
                    feature_importance = pd.DataFrame({
                        'Feature': features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    # Plot top 15 features
                    top_features = feature_importance.head(15)
                    plt.barh(top_features['Feature'], top_features['Importance'])
                    plt.title('Top 15 Feature Importance')
                    plt.xlabel('Importance')
                    plt.gca().invert_yaxis()
                    plt.grid(True, axis='x')
                    plt.tight_layout()
                    figures.append(fig)
                except Exception as e:
                    print(f"Error generating feature importance plot: {str(e)}")
            
            # 5. Market regimes
            if hasattr(strategy, 'signals') and 'market_regime' in strategy.signals.columns:
                # Regime returns
                try:
                    fig = plt.figure(figsize=(10, 6))
                    regime_returns = strategy.signals.groupby('market_regime')['returns'].mean() * 252  # Annualized
                    
                    plt.bar(regime_returns.index, regime_returns * 100)
                    plt.title('Average Annual Returns by Market Regime')
                    plt.xlabel('Market Regime')
                    plt.ylabel('Annualized Returns (%)')
                    plt.grid(True, axis='y')
                    plt.tight_layout()
                    figures.append(fig)
                except Exception as e:
                    print(f"Error generating market regime returns plot: {str(e)}")
                
                # Regime distribution over time
                try:
                    fig = plt.figure(figsize=(10, 6))
                    # Get price column
                    price_col = None
                    for col in ['close', 'price', 'market_price_usd_bc']:
                        if col in strategy.signals.columns:
                            price_col = col
                            break
                    
                    if price_col:
                        # Create a colormap for regimes
                        n_regimes = strategy.signals['market_regime'].nunique()
                        colors = plt.cm.viridis(np.linspace(0, 1, n_regimes))
                        
                        # Plot price with colored backgrounds by regime
                        plt.plot(strategy.signals.index, strategy.signals[price_col], color='black', alpha=0.7)
                        
                        # Add colored backgrounds for regimes
                        for regime in range(n_regimes):
                            regime_mask = strategy.signals['market_regime'] == regime
                            if regime_mask.any():
                                regime_dates = strategy.signals.index[regime_mask]
                                plt.fill_between(regime_dates, 0, strategy.signals[price_col].max(),
                                               color=colors[regime], alpha=0.2)
                        
                        plt.title('Market Regimes Over Time')
                        plt.xlabel('Date')
                        plt.ylabel('Price')
                        plt.grid(True)
                        plt.tight_layout()
                        figures.append(fig)
                except Exception as e:
                    print(f"Error generating market regime distribution plot: {str(e)}")
            
            # 6. On-chain metrics if available
            if hasattr(strategy, 'onchain_analyzer') and strategy.onchain_analyzer is not None:
                if hasattr(strategy.onchain_analyzer, 'onchain_features') and strategy.onchain_analyzer.onchain_features is not None:
                    # Select key metrics
                    onchain_features = strategy.onchain_analyzer.onchain_features
                    key_metrics = [col for col in onchain_features.columns 
                                  if any(x in col for x in ['ratio', 'sentiment', 'hodl', 'mvrv', 'nvt', 'cycle'])
                                  and not col.endswith('_signal') and not col == 'onchain_regime'][:5]
                    
                    if key_metrics:
                        fig = plt.figure(figsize=(10, 6))
                        
                        # Plot each metric with normalization for comparison
                        for metric in key_metrics:
                            data = onchain_features[metric]
                            if data.max() > data.min():  # Prevent division by zero
                                normalized = (data - data.min()) / (data.max() - data.min())
                                plt.plot(onchain_features.index, normalized, label=metric)
                        
                        plt.title('Key On-Chain Metrics (Normalized)')
                        plt.xlabel('Date')
                        plt.ylabel('Normalized Value')
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        figures.append(fig)
            
            # 7. Technical Indicators Visualization
            if hasattr(strategy, 'signals'):
                signals = strategy.signals
                # Check for technical indicators
                tech_indicators = []
                if 'rsi_14' in signals.columns:
                    tech_indicators.append('rsi_14')
                if 'macd' in signals.columns:
                    tech_indicators.append('macd')
                if 'bb_width' in signals.columns:
                    tech_indicators.append('bb_width')
                if 'volatility_14d' in signals.columns:
                    tech_indicators.append('volatility_14d')
                    
                if tech_indicators:
                    fig = plt.figure(figsize=(10, 6))
                    
                    # Plot each technical indicator with normalization
                    for indicator in tech_indicators:
                        data = signals[indicator].copy()
                        # Handle NaN and inf values
                        data = data.replace([np.inf, -np.inf], np.nan).dropna()
                        if len(data) > 0 and data.max() > data.min():
                            normalized = (data - data.min()) / (data.max() - data.min())
                            plt.plot(data.index, normalized, label=indicator)
                    
                    plt.title('Technical Indicators (Normalized)')
                    plt.xlabel('Date')
                    plt.ylabel('Normalized Value')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    figures.append(fig)
                    
                # 8. Moving Average Crossovers
                if 'sma_50' in signals.columns and 'sma_200' in signals.columns:
                    fig = plt.figure(figsize=(10, 6))
                    
                    # Get price data
                    price_col = None
                    for col in ['close', 'price', 'market_price_usd_bc']:
                        if col in signals.columns:
                            price_col = col
                            break
                    
                    if price_col:
                        # Plot price and moving averages
                        plt.plot(signals.index, signals[price_col], label=f'Price ({price_col})', color='black', alpha=0.5)
                        plt.plot(signals.index, signals['sma_50'], label='50-day MA', color='blue')
                        plt.plot(signals.index, signals['sma_200'], label='200-day MA', color='red')
                        
                        plt.title('Moving Average Crossovers')
                        plt.xlabel('Date')
                        plt.ylabel('Price')
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        figures.append(fig)
                
                # 9. RSI with overbought/oversold levels
                if 'rsi_14' in signals.columns:
                    fig = plt.figure(figsize=(10, 6))
                    
                    # Plot RSI
                    plt.plot(signals.index, signals['rsi_14'], label='RSI-14', color='blue')
                    
                    # Add overbought/oversold lines
                    plt.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
                    plt.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
                    plt.axhline(y=50, color='k', linestyle='--', alpha=0.3)
                    
                    plt.title('Relative Strength Index (RSI-14)')
                    plt.xlabel('Date')
                    plt.ylabel('RSI Value')
                    plt.ylim(0, 100)  # RSI range is 0-100
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    figures.append(fig)
                
                # 10. MACD
                if all(x in signals.columns for x in ['macd', 'macd_signal']):
                    fig = plt.figure(figsize=(10, 6))
                    
                    # Create two subplots: MACD and Price
                    ax1 = plt.subplot(2, 1, 1)
                    
                    # Plot price on top subplot
                    price_col = None
                    for col in ['close', 'price', 'market_price_usd_bc']:
                        if col in signals.columns:
                            price_col = col
                            break
                    
                    if price_col:
                        ax1.plot(signals.index, signals[price_col], label=price_col, color='black')
                        ax1.set_ylabel('Price')
                        ax1.set_title('Price and MACD')
                        ax1.grid(True)
                        ax1.legend(loc='upper left')
                    
                    # Plot MACD on bottom subplot
                    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
                    ax2.plot(signals.index, signals['macd'], label='MACD Line', color='blue')
                    ax2.plot(signals.index, signals['macd_signal'], label='Signal Line', color='red')
                    
                    # Plot MACD histogram if available
                    if 'macd_hist' in signals.columns:
                        ax2.bar(signals.index, signals['macd_hist'], label='MACD Histogram', 
                              color=signals['macd_hist'].apply(lambda x: 'green' if x > 0 else 'red'),
                              alpha=0.3, width=5)
                    
                    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    ax2.set_xlabel('Date')
                    ax2.set_ylabel('MACD Value')
                    ax2.grid(True)
                    ax2.legend(loc='upper left')
                    
                    plt.tight_layout()
                    figures.append(fig)
                
                # 11. Bollinger Bands
                if all(x in signals.columns for x in ['bb_upper', 'bb_lower']) and price_col:
                    fig = plt.figure(figsize=(10, 6))
                    
                    # Plot price and Bollinger Bands
                    plt.plot(signals.index, signals[price_col], label=price_col, color='black')
                    plt.plot(signals.index, signals['bb_upper'], label='Upper Band', color='red', linestyle='--')
                    
                    # Plot middle band (20-day SMA) if available, otherwise calculate it
                    if 'sma_20' in signals.columns:
                        plt.plot(signals.index, signals['sma_20'], label='Middle Band (20-day SMA)', color='blue')
                    
                    plt.plot(signals.index, signals['bb_lower'], label='Lower Band', color='green', linestyle='--')
                    
                    # Fill between bands
                    plt.fill_between(signals.index, signals['bb_upper'], signals['bb_lower'], color='gray', alpha=0.1)
                    
                    plt.title('Bollinger Bands')
                    plt.xlabel('Date')
                    plt.ylabel('Price')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    figures.append(fig)
                
                # 12. Combined Signals Visualization
                if 'signal' in signals.columns and price_col:
                    fig = plt.figure(figsize=(10, 6))
                    
                    # Plot price
                    plt.plot(signals.index, signals[price_col], label=price_col, color='black', alpha=0.7)
                    
                    # Mark buy signals
                    buy_signals = signals[signals['signal'] == 1]
                    if not buy_signals.empty:
                        plt.scatter(buy_signals.index, buy_signals[price_col], 
                                  marker='^', color='green', s=100, label='Buy Signal')
                    
                    # Mark sell signals
                    sell_signals = signals[signals['signal'] == -1]
                    if not sell_signals.empty:
                        plt.scatter(sell_signals.index, sell_signals[price_col], 
                                  marker='v', color='red', s=100, label='Sell Signal')
                    
                    # If on-chain signals available, add them too
                    if 'onchain_signal' in signals.columns:
                        # Add only the onchain signals that differ from regular signals
                        onchain_buy = signals[(signals['onchain_signal'] == 1) & (signals['signal'] != 1)]
                        if not onchain_buy.empty:
                            plt.scatter(onchain_buy.index, onchain_buy[price_col], 
                                      marker='^', color='blue', s=80, label='On-Chain Buy')
                            
                        onchain_sell = signals[(signals['onchain_signal'] == -1) & (signals['signal'] != -1)]
                        if not onchain_sell.empty:
                            plt.scatter(onchain_sell.index, onchain_sell[price_col], 
                                      marker='v', color='purple', s=80, label='On-Chain Sell')
                    
                    plt.title('Trading Signals on Price Chart')
                    plt.xlabel('Date')
                    plt.ylabel('Price')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    figures.append(fig)
            
            # Convert figures to base64-encoded PNGs for web display
            encoded_images = []
            for fig in figures:
                buf = io.BytesIO()
                FigureCanvas(fig).print_png(buf)
                buf.seek(0)
                img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
                encoded_images.append(img_data)
                plt.close(fig)
            
            return encoded_images
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    try:
        # Run the strategy with the form parameters
        print(f"Running strategy with ticker: {ticker}, start: {start_date}, end: {end_date}")
        
        # Override the visualization method to prevent errors and redirect output
        orig_visualize = IntegratedBitcoinStrategy.visualize_results
        IntegratedBitcoinStrategy.visualize_results = lambda self: None
        
        # Run the strategy
        current_strategy = run_integrated_strategy(
            data_dir=data_dir,
            ticker=ticker,
            start_date=start_dt,
            end_date=end_dt,
            use_backtrader=False,  # Don't use backtrader for web app
            initial_cash=initial_cash,
            use_sample_data=use_sample_data,
            use_enhanced_onchain=use_enhanced_onchain
        )
        
        # Restore original visualization method
        IntegratedBitcoinStrategy.visualize_results = orig_visualize
        
        if current_strategy is None:
            return render_template('error.html', error="Failed to run strategy. Check logs for details.")
        
        # Generate visualizations for web display
        graphs = patched_visualization(current_strategy)
        
        # Get performance metrics
        performance = current_strategy.performance if hasattr(current_strategy, 'performance') else {}
        
        # Get signals sample
        signals = None
        if hasattr(current_strategy, 'signals') and current_strategy.signals is not None:
            sample_columns = ['signal', 'position', 'returns', 'strategy_returns']
            
            # Add on-chain columns if available
            if 'onchain_signal' in current_strategy.signals.columns:
                sample_columns.append('onchain_signal')
            if 'market_regime' in current_strategy.signals.columns:
                sample_columns.append('market_regime')
                
            # Get only the columns that exist
            available_columns = [col for col in sample_columns if col in current_strategy.signals.columns]
            
            # Get sample (last 10 rows)
            signals = current_strategy.signals[available_columns].tail(10).to_html(classes='table table-striped')
        
        # Get technical indicators sample
        technical_indicators = None
        if hasattr(current_strategy, 'signals') and current_strategy.signals is not None:
            # Look for technical indicator columns
            tech_columns = []
            indicator_prefixes = ['rsi', 'macd', 'bb_', 'sma_50', 'sma_200', 'volatility']
            
            for col in current_strategy.signals.columns:
                if any(prefix in col for prefix in indicator_prefixes):
                    tech_columns.append(col)
            
            # Add a price column if available
            price_columns = [col for col in current_strategy.signals.columns if col in ['close', 'price', 'market_price_usd_bc']]
            if price_columns:
                tech_columns = price_columns + tech_columns
            
            # Limit to a reasonable number of columns
            tech_columns = tech_columns[:6]  # Limit to 6 columns for readability
            
            if tech_columns:
                # Get sample (last 10 rows)
                technical_indicators = current_strategy.signals[tech_columns].tail(10).to_html(classes='table table-striped')
        
        # Get feature importance if available
        feature_importance = None
        if 'random_forest' in current_strategy.models:
            model, features = current_strategy.models['random_forest']
            importance = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            feature_importance = importance.head(15).to_html(classes='table table-striped')
        
        # Get on-chain metrics if available
        onchain_metrics = None
        if (hasattr(current_strategy, 'onchain_analyzer') and 
            hasattr(current_strategy.onchain_analyzer, 'onchain_features') and 
            current_strategy.onchain_analyzer.onchain_features is not None):
            
            onchain_sample = current_strategy.onchain_analyzer.onchain_features.tail(10)
            onchain_metrics = onchain_sample.to_html(classes='table table-striped')
        
        # Get combined signals analysis
        combined_signals = None
        if hasattr(current_strategy, 'signals') and current_strategy.signals is not None:
            if 'signal' in current_strategy.signals.columns and 'onchain_signal' in current_strategy.signals.columns:
                # Calculate agreement stats
                total = len(current_strategy.signals)
                agree = (current_strategy.signals['signal'] == current_strategy.signals['onchain_signal']).sum()
                agree_pct = agree / total * 100
                
                # Create performance by signal type
                signal_performance = {}
                
                # Regular signals performance
                reg_signals = current_strategy.signals[current_strategy.signals['signal'] != 0]
                if len(reg_signals) > 0:
                    reg_returns = reg_signals['strategy_returns'].mean() * 252 * 100  # Annualized
                    signal_performance['technical'] = reg_returns
                
                # On-chain signals performance
                onchain_signals = current_strategy.signals[current_strategy.signals['onchain_signal'] != 0]
                if len(onchain_signals) > 0:
                    onchain_returns = onchain_signals['strategy_returns'].mean() * 252 * 100  # Annualized
                    signal_performance['onchain'] = onchain_returns
                
                # Create HTML table
                combined_signals = f"""
                <table class="table table-striped">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Signal Agreement Rate</td>
                        <td>{agree_pct:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Technical Signals Annual Return</td>
                        <td>{signal_performance.get('technical', 0):.2f}%</td>
                    </tr>
                    <tr>
                        <td>On-Chain Signals Annual Return</td>
                        <td>{signal_performance.get('onchain', 0):.2f}%</td>
                    </tr>
                </table>
                """
        
        # Render the results template
        return render_template('results.html', 
                            ticker=ticker,
                            start_date=start_date,
                            end_date=end_date,
                            initial_cash=initial_cash,
                            use_sample_data=use_sample_data,
                            use_enhanced_onchain=use_enhanced_onchain,
                            use_enhanced_technical=use_enhanced_technical,
                            performance=performance,
                            signals=signals,
                            technical_indicators=technical_indicators,
                            feature_importance=feature_importance,
                            onchain_metrics=onchain_metrics,
                            combined_signals=combined_signals,
                            graphs=graphs)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error running strategy: {str(e)}")
        print(error_details)
        return render_template('error.html', 
                            error=f"Error running strategy: {str(e)}",
                            details=error_details)

@app.route('/download_signals')
def download_signals():
    """Download the signals data as CSV"""
    if current_strategy is None or not hasattr(current_strategy, 'signals'):
        return "No strategy data available", 404
    
    # Create a CSV in memory
    csv_data = io.StringIO()
    current_strategy.signals.to_csv(csv_data)
    csv_data.seek(0)
    
    # Send the file
    return send_file(
        io.BytesIO(csv_data.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='trading_signals.csv'
    )

@app.route('/download_performance')
def download_performance():
    """Download the performance metrics as CSV"""
    if current_strategy is None or not hasattr(current_strategy, 'performance'):
        return "No performance data available", 404
    
    # Convert performance dict to DataFrame
    perf_df = pd.DataFrame.from_dict(current_strategy.performance, orient='index', columns=['Value'])
    
    # Create a CSV in memory
    csv_data = io.StringIO()
    perf_df.to_csv(csv_data)
    csv_data.seek(0)
    
    # Send the file
    return send_file(
        io.BytesIO(csv_data.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='performance_metrics.csv'
    )

@app.route('/download_technical_indicators')
def download_technical_indicators():
    """Download the technical indicators as CSV"""
    if current_strategy is None or not hasattr(current_strategy, 'signals'):
        return "No technical indicator data available", 404
    
    # Get technical indicator columns
    signals = current_strategy.signals
    technical_columns = []
    
    # Look for common technical indicator names
    indicator_prefixes = ['rsi', 'macd', 'bb_', 'sma_', 'ema_', 'volatility', 'momentum']
    for col in signals.columns:
        if any(prefix in col for prefix in indicator_prefixes):
            technical_columns.append(col)
    
    # Add price columns
    price_columns = [col for col in signals.columns if 'price' in col.lower() or col in ['open', 'high', 'low', 'close', 'volume']]
    technical_columns.extend(price_columns)
    
    # Create DataFrame with only technical indicators
    if technical_columns:
        tech_df = signals[technical_columns]
        
        # Create a CSV in memory
        csv_data = io.StringIO()
        tech_df.to_csv(csv_data)
        csv_data.seek(0)
        
        # Send the file
        return send_file(
            io.BytesIO(csv_data.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='technical_indicators.csv'
        )
    else:
        return "No technical indicator data found", 404

@app.route('/download_onchain_metrics')
def download_onchain_metrics():
    """Download the on-chain metrics as CSV"""
    if (current_strategy is None or 
        not hasattr(current_strategy, 'onchain_analyzer') or 
        not hasattr(current_strategy.onchain_analyzer, 'onchain_features')):
        return "No on-chain metrics available", 404
    
    # Get on-chain metrics
    onchain_features = current_strategy.onchain_analyzer.onchain_features
    
    # Create a CSV in memory
    csv_data = io.StringIO()
    onchain_features.to_csv(csv_data)
    csv_data.seek(0)
    
    # Send the file
    return send_file(
        io.BytesIO(csv_data.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='onchain_metrics.csv'
    )

@app.template_filter('format_number')
def format_number(value):
    """Format numbers nicely for display"""
    if isinstance(value, (int, float)):
        if abs(value) < 0.01:
            return f"{value:.6f}"
        elif abs(value) < 1:
            return f"{value:.4f}"
        else:
            return f"{value:,.2f}"
    return value

if __name__ == '__main__':
    app.run(debug=True)