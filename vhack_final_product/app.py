from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import uuid
from datetime import datetime, timedelta
import warnings
import traceback
from werkzeug.utils import secure_filename

# Import the strategy class
from hiha import IntegratedCryptoStrategy

app = Flask(__name__)
app.secret_key = "crypto_strategy_secret_key"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store strategies by ID
strategies = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_strategy', methods=['POST'])
def run_strategy():
    try:
        # Get parameters from form
        use_sample = request.form.get('use_sample') == 'on'
        
        # Handle date inputs
        start_date_str = request.form.get('start_date', '')
        end_date_str = request.form.get('end_date', '')
        
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d') if start_date_str else None
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str else None
        
        # Create a unique ID for this strategy run
        strategy_id = str(uuid.uuid4())
        
        # Handle file uploads
        data_dir = None
        if 'data_files' in request.files:
            files = request.files.getlist('data_files')
            if files and files[0].filename:
                # Create a directory for this run
                data_dir = os.path.join(app.config['UPLOAD_FOLDER'], strategy_id)
                os.makedirs(data_dir, exist_ok=True)
                
                # Save all uploaded files
                for file in files:
                    if file.filename:
                        filename = secure_filename(file.filename)
                        file.save(os.path.join(data_dir, filename))
        
        # Create and run strategy
        strategy = IntegratedCryptoStrategy()
        
        # Store in global dict for later access
        strategies[strategy_id] = {
            'strategy': strategy,
            'status': 'running',
            'start_time': datetime.now(),
            'params': {
                'use_sample': use_sample,
                'start_date': start_date_str,
                'end_date': end_date_str,
                'data_dir': data_dir
            }
        }
        
        # Store ID in session for redirect
        session['current_strategy'] = strategy_id
        
        # Run strategy in a way that won't block
        # In a real app, you might want to use Celery or threads
        success = strategy.run_pipeline(data_dir=data_dir, use_sample=use_sample, 
                                    start_date=start_date, end_date=end_date)
        
        # Update status
        strategies[strategy_id]['status'] = 'completed'
        strategies[strategy_id]['success'] = success
        
        return redirect(url_for('results', strategy_id=strategy_id))
    
    except Exception as e:
        # Log the error and traceback
        print(f"Error running strategy: {str(e)}")
        print(traceback.format_exc())
        return render_template('error.html', error=str(e), traceback=traceback.format_exc())

@app.route('/results/<strategy_id>')
def results(strategy_id):
    if strategy_id not in strategies:
        return render_template('error.html', error="Strategy not found")
    
    strategy_data = strategies[strategy_id]
    strategy = strategy_data['strategy']
    
    # Check if strategy has run successfully
    if not hasattr(strategy, 'performance') or not strategy.performance:
        return render_template('error.html', error="Strategy has not completed running")
    
    # Performance metrics
    performance = strategy.performance
    
    # Create results plots
    plots = create_results_plots(strategy)
    
    # Get trade summary
    trade_summary = get_trade_summary(strategy)
    
    return render_template('results.html', 
                          strategy_id=strategy_id,
                          performance=performance,
                          trade_summary=trade_summary,
                          plots=plots)

@app.route('/trades/<strategy_id>')
def trades(strategy_id):
    if strategy_id not in strategies:
        return render_template('error.html', error="Strategy not found")
    
    strategy_data = strategies[strategy_id]
    strategy = strategy_data['strategy']
    
    # Check if strategy has trade history
    if not hasattr(strategy, 'trade_history') or not strategy.trade_history:
        return render_template('error.html', error="No trade history available")
    
    # Format trade history for display
    trades = []
    for trade in strategy.trade_history:
        trades.append({
            'entry_date': trade['entry_date'],
            'exit_date': trade['exit_date'],
            'trade_type': trade['trade_type'],
            'entry_price': f"${trade['entry_price']:.2f}",
            'exit_price': f"${trade['exit_price']:.2f}",
            'exit_reason': trade['exit_reason'],
            'position_size': f"{trade['position_size']*100:.2f}%",
            'return': f"{trade['return']*100:.2f}%",
            'profit_loss': "Profit" if trade['return'] > 0 else "Loss",
            'duration': (trade['exit_date'] - trade['entry_date']).days
        })
    
    # Create trades plots
    plots = create_trades_plots(strategy)
    
    return render_template('trades.html',
                          strategy_id=strategy_id, 
                          trades=trades,
                          plots=plots)

@app.route('/signals/<strategy_id>')
def signals(strategy_id):
    if strategy_id not in strategies:
        return render_template('error.html', error="Strategy not found")
    
    strategy_data = strategies[strategy_id]
    strategy = strategy_data['strategy']
    
    # Check if strategy has generated signals
    if not hasattr(strategy, 'signals') or strategy.signals is None:
        return render_template('error.html', error="No signals data available")
    
    # Create signals plots
    plots = create_signals_plots(strategy)
    
    # Calculate signals statistics
    signal_stats = calculate_signal_stats(strategy)
    
    return render_template('signals.html',
                          strategy_id=strategy_id,
                          signal_stats=signal_stats,
                          plots=plots)

@app.route('/download_data/<strategy_id>/<data_type>')
def download_data(strategy_id, data_type):
    if strategy_id not in strategies:
        return jsonify({"error": "Strategy not found"}), 404
    
    strategy = strategies[strategy_id]['strategy']
    
    try:
        if data_type == 'signals':
            if hasattr(strategy, 'signals') and strategy.signals is not None:
                return strategy.signals.to_csv()
        elif data_type == 'trades':
            if hasattr(strategy, 'trade_history') and strategy.trade_history:
                df = pd.DataFrame(strategy.trade_history)
                return df.to_csv()
        elif data_type == 'performance':
            if hasattr(strategy, 'performance') and strategy.performance:
                df = pd.DataFrame([strategy.performance])
                return df.to_csv()
        
        return jsonify({"error": "Data not available"}), 404
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def create_results_plots(strategy):
    plots = {}
    
    # Only create plots if signals data is available
    if hasattr(strategy, 'signals') and strategy.signals is not None:
        signals = strategy.signals
        
        # 1. Cumulative returns
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=signals.index, 
            y=signals['cum_strategy_returns'] * 100,
            mode='lines',
            name='Trading Strategy',
            line=dict(color='green', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=signals.index, 
            y=signals['cum_market_returns'] * 100,
            mode='lines',
            name='Buy and Hold',
            line=dict(color='blue', width=2, dash='dash')
        ))
        fig.update_layout(
            title='Cumulative Returns: Strategy vs Buy-and-Hold',
            xaxis_title='Date',
            yaxis_title='Returns (%)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white'
        )
        plots['returns'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # 2. Drawdown
        if hasattr(strategy.risk_manager, 'drawdown_history'):
            drawdown_data = strategy.risk_manager.drawdown_history
            
            # Handle length mismatch
            if len(drawdown_data) < len(signals):
                # Pad with zeros
                drawdown_data = [0] * (len(signals) - len(drawdown_data)) + drawdown_data
            elif len(drawdown_data) > len(signals):
                # Trim
                drawdown_data = drawdown_data[-len(signals):]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=signals.index, 
                y=[d * 100 for d in drawdown_data],
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=2),
                fill='tozeroy'
            ))
            fig.update_layout(
                title='Strategy Drawdown',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                template='plotly_white'
            )
            plots['drawdown'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # 3. Portfolio value
        if 'portfolio_value' in signals.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=signals.index, 
                y=signals['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='purple', width=2)
            ))
            fig.update_layout(
                title='Portfolio Value Over Time',
                xaxis_title='Date',
                yaxis_title='Value ($)',
                template='plotly_white'
            )
            plots['portfolio'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return plots

def create_trades_plots(strategy):
    plots = {}
    
    # Only create plots if trade history is available
    if hasattr(strategy, 'trade_history') and strategy.trade_history:
        trades = strategy.trade_history
        
        # 1. Trade returns distribution
        trade_returns = [trade['return'] * 100 for trade in trades]
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=trade_returns,
            nbinsx=20,
            marker_color='lightblue',
            name='Trade Returns'
        ))
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=0, y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )
        fig.update_layout(
            title='Distribution of Trade Returns',
            xaxis_title='Return (%)',
            yaxis_title='Frequency',
            template='plotly_white'
        )
        plots['returns_dist'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # 2. Trade duration vs returns
        durations = [(trade['exit_date'] - trade['entry_date']).days for trade in trades]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=durations,
            y=trade_returns,
            mode='markers',
            marker=dict(
                color=trade_returns,
                colorscale='RdYlGn',
                size=10,
                colorbar=dict(title="Return (%)"),
                cmin=-max(abs(min(trade_returns)), abs(max(trade_returns))),
                cmax=max(abs(min(trade_returns)), abs(max(trade_returns)))
            ),
            name='Trades'
        ))
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=max(durations), y1=0,
            line=dict(color="red", width=2, dash="dash")
        )
        fig.update_layout(
            title='Trade Duration vs Returns',
            xaxis_title='Duration (days)',
            yaxis_title='Return (%)',
            template='plotly_white'
        )
        plots['duration_vs_returns'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # 3. Trade performance by type
        long_returns = [trade['return'] * 100 for trade in trades if trade['trade_type'] == 'long']
        short_returns = [trade['return'] * 100 for trade in trades if trade['trade_type'] == 'short']
        
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=long_returns,
            name='Long Trades',
            marker_color='green',
            boxmean=True
        ))
        fig.add_trace(go.Box(
            y=short_returns,
            name='Short Trades',
            marker_color='red',
            boxmean=True
        ))
        fig.update_layout(
            title='Trade Performance by Type',
            yaxis_title='Return (%)',
            template='plotly_white'
        )
        plots['performance_by_type'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return plots

def create_signals_plots(strategy):
    plots = {}
    
    # Only create plots if signals data is available
    if hasattr(strategy, 'signals') and strategy.signals is not None:
        signals = strategy.signals
        
        # 1. Signal distribution
        if 'signal' in signals.columns:
            # Count signal occurrences
            signal_counts = signals['signal'].value_counts().to_dict()
            labels = {-1: 'Sell', 0: 'Hold', 1: 'Buy'}
            
            # Prepare data for the chart
            fig_data = []
            for signal_value, count in signal_counts.items():
                fig_data.append({
                    'Signal': labels.get(signal_value, str(signal_value)),
                    'Count': count
                })
            
            # Create dataframe
            df = pd.DataFrame(fig_data)
            
            # Create the figure
            fig = px.bar(
                df, 
                x='Signal', 
                y='Count',
                color='Signal',
                color_discrete_map={'Buy': 'green', 'Hold': 'gray', 'Sell': 'red'},
                title="Distribution of Trading Signals"
            )
            fig.update_layout(template='plotly_white')
            plots['signal_dist'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # 2. Signal timeline
        if 'signal' in signals.columns:
            # Prepare data for non-zero signals
            buy_signals = signals[signals['signal'] == 1]
            sell_signals = signals[signals['signal'] == -1]
            
            fig = go.Figure()
            
            # Add price line
            price_col = None
            for col in ['price_close', 'market_price_usd']:
                if col in signals.columns:
                    price_col = col
                    break
            
            if price_col:
                fig.add_trace(go.Scatter(
                    x=signals.index,
                    y=signals[price_col],
                    mode='lines',
                    name='Price',
                    line=dict(color='blue', width=1)
                ))
            
            # Add buy signals
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals[price_col] if price_col else [1] * len(buy_signals),
                    mode='markers',
                    name='Buy Signals',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ))
            
            # Add sell signals
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals[price_col] if price_col else [1] * len(sell_signals),
                    mode='markers',
                    name='Sell Signals',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ))
            
            fig.update_layout(
                title='Trading Signals Timeline',
                xaxis_title='Date',
                yaxis_title='Price' if price_col else 'Signal',
                template='plotly_white'
            )
            plots['signal_timeline'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        # 3. Position size over time
        if 'position_size' in signals.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=signals.index,
                y=signals['position_size'] * 100,  # Convert to percentage
                mode='lines',
                name='Position Size',
                line=dict(color='purple', width=2)
            ))
            fig.update_layout(
                title='Position Size Over Time',
                xaxis_title='Date',
                yaxis_title='Position Size (% of Portfolio)',
                template='plotly_white'
            )
            plots['position_size'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        # 4. Model contribution
        model_signal_cols = [col for col in signals.columns if col.endswith('_signal') and col != 'signal']
        if len(model_signal_cols) >= 2:
            # Prepare data for correlation matrix
            corr_matrix = signals[model_signal_cols + ['signal']].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text:.2f}'
            ))
            fig.update_layout(
                title='Signal Correlation Matrix',
                template='plotly_white'
            )
            plots['model_correlation'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return plots

def get_trade_summary(strategy):
    """Create a summary of trading activity"""
    summary = {}
    
    if hasattr(strategy, 'trade_history') and strategy.trade_history:
        trades = strategy.trade_history
        
        # Count trades by type
        long_trades = sum(1 for trade in trades if trade['trade_type'] == 'long')
        short_trades = sum(1 for trade in trades if trade['trade_type'] == 'short')
        
        # Calculate win rates
        winning_trades = sum(1 for trade in trades if trade['return'] > 0)
        win_rate = winning_trades / len(trades) if trades else 0
        
        long_winners = sum(1 for trade in trades if trade['trade_type'] == 'long' and trade['return'] > 0)
        long_win_rate = long_winners / long_trades if long_trades else 0
        
        short_winners = sum(1 for trade in trades if trade['trade_type'] == 'short' and trade['return'] > 0)
        short_win_rate = short_winners / short_trades if short_trades else 0
        
        # Average returns
        avg_return = sum(trade['return'] for trade in trades) / len(trades) if trades else 0
        avg_winner = sum(trade['return'] for trade in trades if trade['return'] > 0) / winning_trades if winning_trades else 0
        avg_loser = sum(trade['return'] for trade in trades if trade['return'] <= 0) / (len(trades) - winning_trades) if (len(trades) - winning_trades) > 0 else 0
        
        # Average trade duration
        avg_duration = sum((trade['exit_date'] - trade['entry_date']).days for trade in trades) / len(trades) if trades else 0
        
        # Exit reasons
        exit_reasons = {}
        for trade in trades:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
        summary = {
            'total_trades': len(trades),
            'long_trades': long_trades,
            'short_trades': short_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate * 100,
            'long_win_rate': long_win_rate * 100,
            'short_win_rate': short_win_rate * 100,
            'avg_return': avg_return * 100,
            'avg_winner': avg_winner * 100,
            'avg_loser': avg_loser * 100,
            'avg_duration': avg_duration,
            'exit_reasons': exit_reasons
        }
    
    return summary

def calculate_signal_stats(strategy):
    """Calculate statistics about the generated signals"""
    stats = {}
    
    if hasattr(strategy, 'signals') and strategy.signals is not None:
        signals = strategy.signals
        
        if 'signal' in signals.columns:
            # Count signal types
            signal_counts = signals['signal'].value_counts()
            
            # Total days with signals
            days_with_signals = (signals['signal'] != 0).sum()
            signal_ratio = days_with_signals / len(signals)
            
            # Signal transitions (how often signals change)
            signal_changes = (signals['signal'] != signals['signal'].shift(1)).sum()
            avg_signal_duration = len(signals) / signal_changes if signal_changes > 0 else 0
            
            # Model agreement
            model_signal_cols = [col for col in signals.columns if col.endswith('_signal') and col != 'signal' 
                                and col not in ['rsi_signal', 'ma_signal', 'flow_rule_signal']]
            
            if model_signal_cols:
                # Calculate how often models agree
                agreement_days = 0
                strong_agreement_days = 0
                
                for i, row in signals.iterrows():
                    # Count non-zero signals
                    model_signals = [row[col] for col in model_signal_cols if pd.notna(row[col])]
                    if not model_signals:
                        continue
                        
                    # Count signals in the same direction
                    positive_signals = sum(1 for s in model_signals if s > 0)
                    negative_signals = sum(1 for s in model_signals if s < 0)
                    
                    # If all models point in same direction
                    if positive_signals == len(model_signals) or negative_signals == len(model_signals):
                        strong_agreement_days += 1
                    
                    # If majority of models agree
                    if positive_signals > len(model_signals)/2 or negative_signals > len(model_signals)/2:
                        agreement_days += 1
                
                agreement_ratio = agreement_days / len(signals) if signals.shape[0] > 0 else 0
                strong_agreement_ratio = strong_agreement_days / len(signals) if signals.shape[0] > 0 else 0
            else:
                agreement_ratio = 0
                strong_agreement_ratio = 0
            
            stats = {
                'total_days': len(signals),
                'days_with_signals': int(days_with_signals),
                'signal_ratio': signal_ratio * 100,
                'buy_signals': int(signal_counts.get(1, 0)),
                'sell_signals': int(signal_counts.get(-1, 0)),
                'hold_signals': int(signal_counts.get(0, 0)),
                'signal_changes': int(signal_changes),
                'avg_signal_duration': avg_signal_duration,
                'model_agreement_ratio': agreement_ratio * 100,
                'strong_agreement_ratio': strong_agreement_ratio * 100
            }
    
    return stats

if __name__ == '__main__':
    app.run(debug=True)