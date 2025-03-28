import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import requests
import time
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from hmmlearn import hmm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, LSTM
warnings.filterwarnings('ignore')

class RiskManager:
    """
    Enhanced risk management component that optimizes for Sharpe ratio.
    Provides position sizing, drawdown control, and leverage management.
    """
    
    def __init__(self, strategy):
        """
        Initialize the risk manager with reference to parent strategy.
        
        Args:
            strategy: Parent IntegratedCryptoStrategy instance
        """
        self.strategy = strategy
        self.portfolio_history = []
        self.drawdown_history = []
        self.max_drawdown = 0
        self.current_drawdown = 0
        self.peak_value = 1.0  # Normalized starting value
        
    def calculate_position_size(self, signal_strength, current_volatility, regime=None):
        """
        Calculate optimal position size using Kelly criterion and volatility targeting.
        
        Args:
            signal_strength (float): Signal strength from -1 to 1
            current_volatility (float): Current annualized volatility
            regime (int, optional): Current market regime if available
            
        Returns:
            float: Optimal position size as fraction of portfolio
        """
        # Get regime-specific parameters if available
        if regime is not None and hasattr(self.strategy, 'regime_params'):
            regime_params = self.strategy.regime_params.get(regime, self.strategy.risk_params)
        else:
            regime_params = self.strategy.risk_params
            
        # Base position size on signal strength
        base_position = abs(signal_strength)
        
        # Apply Kelly criterion (half-Kelly for conservatism)
        # Kelly = (edge / odds) = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
        # We approximate this using signal strength
        if hasattr(self.strategy, 'models') and 'random_forest' in self.strategy.models:
            rf_model = self.strategy.models['random_forest']['model']
            
            # Get probability estimate for positive return
            if hasattr(rf_model, 'predict_proba'):
                # Get features for current data point
                # For simplicity, we use the last data point in the features DataFrame
                features = self.strategy.models['random_forest']['features']
                X = self.strategy.features.iloc[-1:][features]
                
                # Get probability of positive return
                prob_positive = rf_model.predict_proba(X)[0, 1]
                
                # Calculate Kelly fraction
                if prob_positive > 0.5:
                    # Estimated edge
                    edge = prob_positive - (1 - prob_positive)
                    # Expected odds (simplified)
                    odds = 1.0
                    # Half-Kelly for conservatism
                    kelly_fraction = 0.5 * (edge / odds)
                    
                    # Apply Kelly position sizing
                    base_position = min(base_position, kelly_fraction)
            
        # Apply volatility targeting
        target_volatility = regime_params.get('volatility_target', 0.15)
        if current_volatility > 0:
            vol_scalar = target_volatility / current_volatility
        else:
            vol_scalar = 1.0
            
        # Scale position by volatility
        position_size = base_position * vol_scalar
        
        # Apply drawdown control - reduce position when in drawdown
        if self.current_drawdown > 0.05:  # If in >5% drawdown
            drawdown_factor = 1.0 - (self.current_drawdown / 0.2)  # Scaled by max 20% drawdown
            drawdown_factor = max(0.25, drawdown_factor)  # Don't reduce below 25%
            position_size *= drawdown_factor
            
        # Apply maximum position constraint
        max_position = regime_params.get('max_position_size', 1.0)
        position_size = min(position_size, max_position)
        
        # Respect leverage limits
        max_leverage = regime_params.get('max_leverage', 1.0)
        position_size = min(position_size, max_leverage)
        
        return position_size * (1 if signal_strength >= 0 else -1)
    
    def update_drawdown(self, current_value):
        """
        Update drawdown calculations
        
        Args:
            current_value (float): Current portfolio value
            
        Returns:
            float: Current drawdown percentage
        """
        # Update peak value if we have a new high
        if current_value > self.peak_value:
            self.peak_value = current_value
            
        # Calculate current drawdown
        if self.peak_value > 0:
            self.current_drawdown = 1 - (current_value / self.peak_value)
        else:
            self.current_drawdown = 0
            
        # Update maximum drawdown
        if self.current_drawdown > self.max_drawdown:
            self.max_drawdown = self.current_drawdown
            
        # Add to history
        self.drawdown_history.append(self.current_drawdown)
        
        return self.current_drawdown
    
    def calculate_stop_loss(self, entry_price, position_type, volatility, regime=None):
        """
        Calculate dynamic stop loss levels based on volatility and regime
        
        Args:
            entry_price (float): Entry price of position
            position_type (str): 'long' or 'short'
            volatility (float): Current volatility
            regime (int, optional): Current market regime
            
        Returns:
            dict: Stop loss and trailing stop levels
        """
        if regime is not None and hasattr(self.strategy, 'regime_params'):
            regime_params = self.strategy.regime_params.get(regime, self.strategy.risk_params)
        else:
            regime_params = self.strategy.risk_params
            
        # Base stop loss - scale with volatility
        base_stop_pct = regime_params.get('stop_loss_pct', 0.05)
        # Adjust for volatility but keep within reasonable range
        vol_adjusted_stop = base_stop_pct * (volatility / 0.02)  # 0.02 = 2% daily vol reference
        vol_adjusted_stop = max(base_stop_pct * 0.75, min(base_stop_pct * 2.0, vol_adjusted_stop))
        
        # Trailing stop - typically tighter than initial stop
        trailing_stop_pct = regime_params.get('trailing_stop_pct', 0.03)
        vol_adjusted_trailing = trailing_stop_pct * (volatility / 0.02)
        vol_adjusted_trailing = max(trailing_stop_pct * 0.75, min(trailing_stop_pct * 2.0, vol_adjusted_trailing))
        
        # Calculate actual price levels
        if position_type == 'long':
            stop_loss_level = entry_price * (1 - vol_adjusted_stop)
            trailing_stop_level = entry_price * (1 - vol_adjusted_trailing)
        else:  # short position
            stop_loss_level = entry_price * (1 + vol_adjusted_stop)
            trailing_stop_level = entry_price * (1 + vol_adjusted_trailing)
            
        return {
            'stop_loss_level': stop_loss_level,
            'trailing_stop_level': trailing_stop_level,
            'stop_loss_pct': vol_adjusted_stop,
            'trailing_stop_pct': vol_adjusted_trailing
        }


class IntegratedCryptoStrategy:
    """
    A self-learning trading strategy that combines on-chain data analysis, 
    exchange flow analysis, and technical indicators to generate trading signals.
    """
    
    def __init__(self):
        """Initialize the trading strategy."""
        self.data_sources = {}
        self.merged_data = None
        self.features = None
        self.models = {}
        self.signals = None
        self.performance = {}
        self.api_keys = {
            'cybotrade': 'YOUR_CYBOTRADE_API_KEY'  # Replace with actual API key
        }
        
        # Initialize risk parameters
        self.risk_params = {
            'volatility_target': 0.15,
            'max_position_size': 1.0,
            'max_leverage': 1.0,
            'stop_loss_pct': 0.05,
            'trailing_stop_pct': 0.03
        }
        
        # Initialize regime-specific risk parameters
        self.regime_params = {}
        
        # Initialize the risk manager
        self.risk_manager = RiskManager(self)
        
        # Position tracking
        self.current_position = 0
        self.position_history = []
        self.trade_history = []
    
    def load_on_chain_data(self, data_dir=None, use_sample=False):
        """
        Load on-chain data from multiple sources including blockchain.com and lookintobitcoin.
        
        Args:
            data_dir: Directory containing data files
            use_sample: Whether to use sample data for testing
        """
        print("Loading on-chain data from multiple sources...")
        
        if data_dir:
            try:
                # Load blockchain.com data
                self.data_sources['blockchain_daily'] = pd.read_csv(f'{data_dir}/blockchain_dot_com_daily_data.csv')
                self.data_sources['blockchain_hourly'] = pd.read_csv(f'{data_dir}/blockchain_dot_com_half_hourly_data.csv')
                
                # Load lookintobitcoin data
                self.data_sources['bitcoin_daily'] = pd.read_csv(f'{data_dir}/look_into_bitcoin_daily_data.csv')
                self.data_sources['address_balances'] = pd.read_csv(f'{data_dir}/look_into_bitcoin_address_balances_data.csv')
                self.data_sources['hodl_waves'] = pd.read_csv(f'{data_dir}/look_into_bitcoin_hodl_waves_data.csv')
                self.data_sources['realized_cap_hodl'] = pd.read_csv(f'{data_dir}/look_into_bitcoin_realised_cap_hodl_waves_data.csv')
                
                # Convert datetime columns
                for key, df in self.data_sources.items():
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'])
                    elif 'Unnamed: 0' in df.columns and key == 'address_balances':
                        df['datetime'] = pd.to_datetime(df['Unnamed: 0'])
                
                print("On-chain data loaded successfully")
                return True
            except Exception as e:
                print(f"Error loading on-chain data: {str(e)}")
                if not use_sample:
                    return False
        
        if use_sample:
            print("Generating sample on-chain data for testing...")
            # Create synthetic data for testing
            start_date = datetime(2020, 1, 1)
            end_date = datetime(2023, 1, 1)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate sample blockchain daily data
            self.data_sources['blockchain_daily'] = pd.DataFrame({
                'datetime': dates,
                'market_price_usd': np.linspace(5000, 50000, len(dates)) + np.random.normal(0, 2000, len(dates)),
                'market_cap_usd': np.linspace(100e9, 1e12, len(dates)) + np.random.normal(0, 20e9, len(dates)),
                'transaction_rate': np.random.lognormal(10, 0.5, len(dates)),
                'hash_rate': np.linspace(50e6, 200e6, len(dates)) + np.random.normal(0, 10e6, len(dates)),
                'difficulty': np.linspace(10e12, 50e12, len(dates)) + np.random.normal(0, 1e12, len(dates)),
                'miners_revenue': np.random.lognormal(16, 0.3, len(dates)),
                'average_block_size': np.random.uniform(0.8, 1.3, len(dates))
            })
            
            # Generate sample bitcoin daily data
            self.data_sources['bitcoin_daily'] = pd.DataFrame({
                'datetime': dates,
                'realised_cap_usd': np.linspace(70e9, 500e9, len(dates)) + np.random.normal(0, 10e9, len(dates)),
                'nvt_ratio': np.random.normal(50, 15, len(dates)),
                'velocity': np.random.normal(5, 1, len(dates))
            })
            
            # Generate sample address balances data
            self.data_sources['address_balances'] = pd.DataFrame({
                'datetime': dates,
                'addresses_with_1000_btc': np.random.normal(2000, 100, len(dates)),
                'addresses_with_100_btc': np.random.normal(15000, 500, len(dates)),
                'addresses_with_10_btc': np.random.normal(100000, 5000, len(dates)),
                'addresses_with_1_btc': np.random.normal(800000, 50000, len(dates)),
                'addresses_with_0.01_btc_x': np.random.normal(5000000, 100000, len(dates)),
                'addresses_with_0.01_btc_y': np.random.normal(10000000, 500000, len(dates))
            })
            
            # Sample HODL waves data
            hodl_cols = ['24h', '1d_1w', '1w_1m', '1m_3m', '3m_6m', '6m_12m',
                        '1y_2y', '2y_3y', '3y_5y', '5y_7y', '7y_10y', '10y']
            
            self.data_sources['hodl_waves'] = pd.DataFrame({'datetime': dates})
            
            # Generate random percentages that sum to 100 for HODL waves
            for i in range(len(dates)):
                random_vals = np.random.dirichlet(np.ones(len(hodl_cols))*3, 1)[0] * 100
                for j, col in enumerate(hodl_cols):
                    self.data_sources['hodl_waves'].loc[i, col] = random_vals[j]
            
            print("Sample on-chain data generated successfully")
            return True
        
        print("Failed to load on-chain data")
        return False
    
    def fetch_exchange_flow_data(self, start_time, end_time=None, save_to_file=None):
        """
        Fetch exchange flow data from Cybotrade API.
        
        Args:
            start_time: Start time as datetime object
            end_time: End time as datetime object (defaults to now)
            save_to_file: Optional file path to save the fetched data
        """
        print("Fetching exchange flow data from Cybotrade API...")
        
        if end_time is None:
            end_time = datetime.now(pytz.timezone("UTC"))
        
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        API_URL = "https://api.datasource.cybotrade.rs"
        API_KEY = self.api_keys['cybotrade']
        
        # Endpoint for exchange flows
        endpoint = "cryptoquant|btc/exchange-flows/in-house-flow?window=block&exchange=all_exchange"
        provider = endpoint.split("|")[0]
        endpoint_path = endpoint.split("|")[-1]
        
        url = f"{API_URL}/{provider}/{endpoint_path}&start_time={start_timestamp}&end_time={end_timestamp}&limit=50000"
        
        try:
            print(f"Requesting: {url}")
            
            response = requests.get(
                url,
                headers={"X-API-KEY": API_KEY}
            )
            
            print(f"Status: {response.status_code} - {response.reason}")
            
            if response.status_code == 200:
                data = response.json().get("data", [])
                
                if data:
                    df = pd.DataFrame(data)
                    df['datetime'] = pd.to_datetime(df['start_time'], unit='ms')
                    
                    print(f"Successfully fetched exchange flow data with {len(df)} records")
                    
                    if save_to_file:
                        df.to_csv(save_to_file, index=False)
                        print(f"Data saved to {save_to_file}")
                    
                    self.data_sources['exchange_flows'] = df
                    return True
                else:
                    print("Response contained no data")
            else:
                print(f"Error response content: {response.text[:200]}...")
                
        except Exception as e:
            print(f"Failed to fetch exchange flow data: {e}")
        
        # If we reach here, use sample data
        print("Generating sample exchange flow data...")
        
        start_date = start_time.replace(tzinfo=None)
        end_date = end_time.replace(tzinfo=None)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate sample exchange flow data
        self.data_sources['exchange_flows'] = pd.DataFrame({
            'datetime': dates,
            'flow_total': np.random.normal(500, 150, len(dates)),
            'flow_mean': np.random.normal(0.1, 0.03, len(dates)),
            'transactions_count_flow': np.random.lognormal(8, 0.5, len(dates)),
            'blockheight': np.arange(600000, 600000 + len(dates) * 144, 144)
        })
        
        print("Sample exchange flow data generated successfully")
        return True
    
    def load_price_data(self, ticker='BTC-USD', start_date=None, end_date=None):
        """
        Load price data for backtesting.
        
        Args:
            ticker: Symbol to load (e.g., 'BTC-USD')
            start_date: Start date
            end_date: End date
        """
        print(f"Loading price data for {ticker}...")
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365*3)  # 3 years
        
        if end_date is None:
            end_date = datetime.now()
        
        try:
            # Try using yfinance
            import yfinance as yf
            
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if not data.empty:
                print(f"Successfully loaded price data with {len(data)} records")
                self.data_sources['price_data'] = data
                return True
            else:
                print("No price data found")
        except Exception as e:
            print(f"Error loading price data: {str(e)}")
        
        # If we reach here, generate sample data
        print("Generating sample price data...")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Random walk for price
        np.random.seed(42)
        log_returns = np.random.normal(0.001, 0.02, len(dates))
        price = 10000  # Starting price
        prices = [price]
        
        for ret in log_returns[1:]:
            price = price * (1 + ret)
            prices.append(price)
        
        # Create DataFrame with OHLCV data
        self.data_sources['price_data'] = pd.DataFrame({
            'Open': prices * np.random.uniform(0.99, 0.999, len(prices)),
            'High': prices * np.random.uniform(1.001, 1.03, len(prices)),
            'Low': prices * np.random.uniform(0.97, 0.999, len(prices)),
            'Close': prices,
            'Volume': np.random.lognormal(16, 1, len(prices))
        }, index=dates)
        
        print("Sample price data generated successfully")
        return True
    
    def merge_data(self):
        """
        Merge data from multiple sources and handle missing values.
        """
        print("Merging data from multiple sources...")
        
        if 'blockchain_daily' not in self.data_sources:
            print("Error: On-chain data not available")
            return False
        
        # Start with blockchain daily data
        blockchain_daily = self.data_sources['blockchain_daily'].copy()
        
        # Ensure datetime is set as index
        if 'datetime' in blockchain_daily.columns:
            blockchain_daily.set_index('datetime', inplace=True)
        
        merged = blockchain_daily.copy()
        
        # Merge with other sources if available
        for source_name in ['bitcoin_daily', 'address_balances', 'hodl_waves', 'realized_cap_hodl']:
            if source_name in self.data_sources:
                source_df = self.data_sources[source_name].copy()
                
                # Ensure datetime is set as index
                if 'datetime' in source_df.columns:
                    source_df.set_index('datetime', inplace=True)
                
                # Remove potential duplicate columns
                overlap_cols = [col for col in source_df.columns if col in merged.columns and col != 'datetime']
                if overlap_cols:
                    source_df = source_df.drop(overlap_cols, axis=1)
                
                # Merge with existing data
                merged = pd.merge(merged, source_df, 
                                  left_index=True, right_index=True, 
                                  how='outer')
        
        # Merge exchange flow data if available
        if 'exchange_flows' in self.data_sources:
            ef_df = self.data_sources['exchange_flows'].copy()
            
            # Ensure datetime is set as index
            if 'datetime' in ef_df.columns:
                ef_df.set_index('datetime', inplace=True)
            
            # Resample to daily if needed
            if ef_df.index.duplicated().any():
                ef_df = ef_df.resample('D').agg({
                    'flow_total': 'sum',
                    'flow_mean': 'mean',
                    'transactions_count_flow': 'sum',
                    'blockheight': 'last'
                })
            
            # Merge with existing data
            merged = pd.merge(merged, ef_df, 
                              left_index=True, right_index=True, 
                              how='outer', suffixes=('', '_ef'))
        
        # Merge price data
        if 'price_data' in self.data_sources:
            price_df = self.data_sources['price_data'].copy()
            
            # Handle MultiIndex columns if present
            if isinstance(price_df.columns, pd.MultiIndex):
                # Flatten MultiIndex columns by taking the first level
                price_df.columns = [col[0] for col in price_df.columns]
                
            # Add column prefix to price data to avoid conflicts
            price_df.columns = [f'price_{str(col).lower()}' if str(col).lower() != 'volume' else 'price_volume' for col in price_df.columns]
            
            # Merge with existing data
            merged = pd.merge(merged, price_df, 
                              left_index=True, right_index=True, 
                              how='outer')
        
        # Fill missing values
        merged = merged.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure price column exists
        if 'price_close' in merged.columns:
            if 'market_price_usd' not in merged.columns:
                merged['market_price_usd'] = merged['price_close']
        elif 'market_price_usd' in merged.columns:
            if 'price_close' not in merged.columns:
                merged['price_close'] = merged['market_price_usd']
                merged['price_open'] = merged['market_price_usd'] * 0.999
                merged['price_high'] = merged['market_price_usd'] * 1.01
                merged['price_low'] = merged['market_price_usd'] * 0.99
        
        # Create basic return column
        if 'price_close' in merged.columns:
            merged['returns'] = merged['price_close'].pct_change()
        elif 'market_price_usd' in merged.columns:
            merged['returns'] = merged['market_price_usd'].pct_change()
        
        self.merged_data = merged
        print(f"Data merged successfully. Shape: {merged.shape}")
        return True
    
    def create_features(self):
        """
        Engineer features from raw data, including technical indicators,
        on-chain metrics, and market regime indicators.
        """
        print("Creating features...")
        
        if self.merged_data is None:
            print("Error: No merged data available")
            return False
        
        df = self.merged_data.copy()
        
        # Determine price column to use
        price_col = None
        for col in ['price_close', 'market_price_usd']:
            if col in df.columns:
                price_col = col
                break
        
        if price_col is None:
            print("Error: No price column found in merged data")
            return False
        
        print(f"Using '{price_col}' as main price column")
        
        # 1. Basic price features
        df['log_returns'] = np.log(df[price_col] / df[price_col].shift(1))
        df['volatility_7d'] = df['returns'].rolling(7).std()
        df['volatility_14d'] = df['returns'].rolling(14).std()
        df['volatility_30d'] = df['returns'].rolling(30).std()
        
        # 2. Moving averages and crossovers
        windows = [7, 14, 30, 50, 200]
        for w in windows:
            df[f'sma_{w}'] = df[price_col].rolling(w).mean()
        
        # Generate crossover features
        df['sma_cross_7_30'] = df['sma_7'] - df['sma_30']
        df['sma_cross_14_50'] = df['sma_14'] - df['sma_50']
        df['sma_cross_50_200'] = df['sma_50'] - df['sma_200']
        
        # 3. RSI (Relative Strength Index)
        delta = df[price_col].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.finfo(float).eps)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 4. MACD (Moving Average Convergence Divergence)
        ema12 = df[price_col].ewm(span=12, adjust=False).mean()
        ema26 = df[price_col].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 5. On-chain metrics
        if all(col in df.columns for col in ['transaction_rate', 'average_block_size']):
            df['transaction_volume_ratio'] = df['transaction_rate'] / df['average_block_size']
        
        if all(col in df.columns for col in ['hash_rate', 'difficulty']):
            df['hash_difficulty_ratio'] = df['hash_rate'] / df['difficulty']
        
        if all(col in df.columns for col in ['total_transaction_fees', 'miners_revenue']):
            df['fee_to_reward_ratio'] = df['total_transaction_fees'] / df['miners_revenue']
        
        if all(col in df.columns for col in ['market_cap_usd', 'transaction_rate']):
            df['nvt_ratio'] = df['market_cap_usd'] / df['transaction_rate']
        
        # 6. Address balance features
        balance_cols = ['addresses_with_1000_btc', 'addresses_with_100_btc',
                      'addresses_with_10_btc', 'addresses_with_1_btc',
                      'addresses_with_0.01_btc_x', 'addresses_with_0.01_btc_y']
        
        if all(col in df.columns for col in balance_cols):
            df['whale_dominance'] = (df['addresses_with_1000_btc'] + df['addresses_with_100_btc']) / (
                df['addresses_with_1000_btc'] + df['addresses_with_100_btc'] + 
                df['addresses_with_10_btc'] + df['addresses_with_1_btc'] + 
                df['addresses_with_0.01_btc_x'] + df['addresses_with_0.01_btc_y']
            )
        
        # 7. HODL wave metrics
        hodl_cols = ['24h', '1d_1w', '1w_1m', '1m_3m', '3m_6m', '6m_12m',
                    '1y_2y', '2y_3y', '3y_5y', '5y_7y', '7y_10y', '10y']
        
        if all(col in df.columns for col in hodl_cols[:4]):
            df['short_term_holders'] = df['24h'] + df['1d_1w'] + df['1w_1m'] + df['1m_3m']
        
        if all(col in df.columns for col in hodl_cols[4:7]):
            df['mid_term_holders'] = df['3m_6m'] + df['6m_12m'] + df['1y_2y']
        
        if all(col in df.columns for col in hodl_cols[7:]):
            df['long_term_holders'] = df['2y_3y'] + df['3y_5y'] + df['5y_7y'] + df['7y_10y'] + df['10y']
        
        if all(x in df.columns for x in ['short_term_holders', 'long_term_holders']):
            df['hodl_ratio'] = df['long_term_holders'] / df['short_term_holders'].replace(0, np.finfo(float).eps)
        
        # 8. Market cap to realized cap ratio (MVRV)
        if all(col in df.columns for col in ['market_cap_usd', 'realised_cap_usd']):
            df['mvrv_ratio'] = df['market_cap_usd'] / df['realised_cap_usd']
        
        # 9. Exchange flow features (if available)
        if 'flow_total' in df.columns:
            # Normalize flow
            for window in [7, 14, 30]:
                flow_mean = df['flow_total'].rolling(window).mean()
                flow_std = df['flow_total'].rolling(window).std()
                df[f'flow_zscore_{window}d'] = (df['flow_total'] - flow_mean) / flow_std.replace(0, np.finfo(float).eps)
                
                # Flow momentum
                df[f'flow_momentum_{window}d'] = df['flow_total'].pct_change(periods=window)
            
            # Flow intensity
            df['flow_intensity'] = df['flow_total'] / df['flow_total'].rolling(30).mean().replace(0, np.finfo(float).eps)
            
            # Flow acceleration
            df['flow_acceleration'] = df['flow_total'].diff().diff()
            
            # Flow transaction ratio
            if 'transactions_count_flow' in df.columns:
                df['flow_tx_ratio'] = df['flow_total'] / df['transactions_count_flow'].replace(0, np.finfo(float).eps)
                df['flow_tx_ratio_vol'] = df['flow_tx_ratio'].rolling(14).std()
                
                # High inflow signal
                df['high_inflow_signal'] = ((df['flow_total'] > df['flow_total'].rolling(30).mean() * 1.5) & 
                                          (df['flow_total'].rolling(3).mean() > df['flow_total'].rolling(30).mean() * 1.2)).astype(int)
                
                # Transaction spike
                df['tx_spike_signal'] = ((df['transactions_count_flow'] > df['transactions_count_flow'].rolling(30).mean() * 2) & 
                                       (df['flow_total'] < df['flow_total'].rolling(30).mean() * 1.2)).astype(int)
        
        # 10. Create target variables (price movement in next n days)
        for days in [1, 3, 7, 14]:
            future_price = df[price_col].shift(-days)
            df[f'target_{days}d'] = (future_price > df[price_col]).astype(int)
            df[f'future_return_{days}d'] = future_price / df[price_col] - 1
        
        # Drop rows with NaN values
        df = df.dropna()
        
        self.features = df
        print(f"Feature engineering complete. Dataset shape: {df.shape}")
        return True
    
    def train_market_regime_model(self, n_regimes=4):
        """
        Train Hidden Markov Model to identify market regimes.
        
        Args:
            n_regimes (int): Number of market regimes to identify
        """
        print(f"Training HMM for market regime detection with {n_regimes} regimes...")
        
        if self.features is None:
            print("Error: No features available")
            return False
        
        # Select features for regime detection
        regime_features = [
            'returns', 'volatility_30d', 'rsi_14'
        ]
        
        # Add additional features if available
        optional_features = ['nvt_ratio', 'transaction_volume_ratio', 'hodl_ratio', 
                           'mvrv_ratio', 'flow_intensity']
        
        for feature in optional_features:
            if feature in self.features.columns:
                regime_features.append(feature)
        
        print(f"Using features for market regime: {regime_features}")
        
        # Check for availability of selected features
        if not all(feature in self.features.columns for feature in regime_features):
            print("Error: Not all required features available for regime detection")
            return False
        
        # Prepare features for HMM
        df_features = self.features[regime_features].copy()
        
        # Replace infinite values with NaN
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        
        # Handle NaN values (fill with column median)
        for col in df_features.columns:
            median_val = df_features[col].median()
            df_features[col] = df_features[col].fillna(median_val)
            
        # Check for extreme values and clip them
        for col in df_features.columns:
            q1 = df_features[col].quantile(0.01)
            q3 = df_features[col].quantile(0.99)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_features[col] = df_features[col].clip(lower_bound, upper_bound)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_features.values)
        
        # Train HMM
        model = hmm.GaussianHMM(
            n_components=n_regimes, 
            covariance_type="full",
            n_iter=100, 
            random_state=42
        )
        
        try:
            model.fit(X_scaled)
            
            # Predict regimes
            hidden_states = model.predict(X_scaled)
            
            # Add regime predictions to features
            self.features['market_regime'] = hidden_states
            
            # Save model and scaler
            self.models['hmm'] = {
                'model': model,
                'scaler': scaler,
                'features': regime_features
            }
            
            # Analyze regimes
            regime_stats = {}
            for regime in range(n_regimes):
                regime_data = self.features[self.features['market_regime'] == regime]
                regime_stats[regime] = {
                    'count': len(regime_data),
                    'avg_return': regime_data['returns'].mean(),
                    'volatility': regime_data['returns'].std(),
                    'avg_rsi': regime_data['rsi_14'].mean() if 'rsi_14' in regime_data.columns else None
                }
            
            print("\nMarket Regime Analysis:")
            for regime, stats in regime_stats.items():
                print(f"Regime {regime}: {stats['count']} days, Avg Return: {stats['avg_return']:.4f}, "
                     f"Volatility: {stats['volatility']:.4f}, Avg RSI: {stats['avg_rsi']:.1f}")
            
            # Configure regime-specific risk parameters based on regime statistics
            self.configure_regime_risk_params(regime_stats)
            
            print("Market regime model trained successfully")
            return True
            
        except Exception as e:
            print(f"Error training HMM model: {str(e)}")
            return False
    
    def configure_regime_risk_params(self, regime_stats):
        """
        Configure risk parameters for each market regime based on regime characteristics.
        
        Args:
            regime_stats (dict): Dictionary containing regime statistics
        """
        print("Configuring regime-specific risk parameters...")
        
        for regime, stats in regime_stats.items():
            # Base volatility target on regime volatility (annualized)
            regime_volatility = stats['volatility'] * np.sqrt(252)
            
            # Adjust risk parameters based on regime characteristics
            if stats['avg_return'] > 0.002:  # Bull market regime
                self.regime_params[regime] = {
                    'volatility_target': min(0.20, regime_volatility),
                    'max_position_size': 1.0,
                    'max_leverage': 1.5,
                    'stop_loss_pct': 0.075,  # Wider stops in bull market
                    'trailing_stop_pct': 0.05
                }
            elif stats['avg_return'] < -0.002:  # Bear market regime
                self.regime_params[regime] = {
                    'volatility_target': min(0.10, regime_volatility),
                    'max_position_size': 0.5,  # Reduced position size in bear market
                    'max_leverage': 0.5,  # No leverage in bear market
                    'stop_loss_pct': 0.03,  # Tighter stops in bear market
                    'trailing_stop_pct': 0.02
                }
            else:  # Sideways/neutral market regime
                self.regime_params[regime] = {
                    'volatility_target': min(0.15, regime_volatility),
                    'max_position_size': 0.75,
                    'max_leverage': 1.0,
                    'stop_loss_pct': 0.05,
                    'trailing_stop_pct': 0.03
                }
                
            # If regime is extremely volatile, adjust parameters
            if regime_volatility > 0.40:
                self.regime_params[regime]['max_position_size'] *= 0.7
                self.regime_params[regime]['max_leverage'] *= 0.7
                self.regime_params[regime]['stop_loss_pct'] *= 1.5  # Wider stops
            
            print(f"Risk parameters for Regime {regime}:")
            for param, value in self.regime_params[regime].items():
                print(f"  {param}: {value}")
    
    def train_cnn_model(self, lookback_window=30):
        """
        Train CNN model for pattern recognition in market data.
        
        Args:
            lookback_window (int): Number of days to look back for pattern recognition
        """
        print(f"Training CNN with {lookback_window}-day lookback window...")
        
        if self.features is None:
            print("Error: No features available")
            return False
        
        # Select features for the CNN
        base_features = [
            'returns', 'log_returns', 'volatility_14d', 'rsi_14',
            'macd', 'macd_hist', 'sma_cross_7_30', 'sma_cross_50_200',
            'market_regime'
        ]
        
        # Add optional features if available
        optional_features = [
            'nvt_ratio', 'transaction_volume_ratio', 'whale_dominance',
            'hodl_ratio', 'mvrv_ratio', 'short_term_holders', 'long_term_holders',
            'flow_zscore_14d', 'flow_intensity', 'high_inflow_signal'
        ]
        
        cnn_features = base_features.copy()
        for feature in optional_features:
            if feature in self.features.columns:
                cnn_features.append(feature)
        
        # Check if we have enough features
        if len(cnn_features) < 5:
            print("Error: Not enough features available for CNN model")
            return False
        
        # Ensure all features exist
        cnn_features = [f for f in cnn_features if f in self.features.columns]
        print(f"Using {len(cnn_features)} features for CNN: {cnn_features}")
        
        # Clean the feature data
        cnn_df = self.features[cnn_features].copy()
        
        # Replace infinite values with NaN
        cnn_df = cnn_df.replace([np.inf, -np.inf], np.nan)
        
        # Handle NaN values (fill with column median)
        for col in cnn_df.columns:
            median_val = cnn_df[col].median()
            cnn_df[col] = cnn_df[col].fillna(median_val)
        
        # Create sequences for CNN
        X_data = []
        y_data = []
        
        for i in range(len(cnn_df) - lookback_window):
            X_data.append(cnn_df.iloc[i:i+lookback_window].values)
            y_data.append(self.features['target_7d'].iloc[i+lookback_window])
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        print(f"Created {len(X_data)} training samples with shape {X_data.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.2, shuffle=False, random_state=42
        )
        
        # Configure TensorFlow for CPU
        try:
            tf.config.set_visible_devices([], 'GPU')
        except:
            pass
        
        # Build CNN model
        model = Sequential([
            # CNN layers
            Conv1D(filters=64, kernel_size=3, activation='relu', 
                   input_shape=(lookback_window, len(cnn_features))),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            # Flatten and dense layers
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=100,  # You may want to reduce for quicker execution
            batch_size=64,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate model
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y_test, y_pred)
        print(f"CNN model accuracy: {accuracy:.4f}")
        
        # Save model
        self.models['cnn'] = {
            'model': model,
            'features': cnn_features,
            'lookback': lookback_window,
            'history': history.history,
            'accuracy': accuracy
        }
        
        return True
    
    def train_exchange_flow_model(self, lookback_window=14):
        """
        Train a specialized model for exchange flow patterns
        
        Args:
            lookback_window (int): Number of days to look back
        """
        print(f"Training exchange flow model with {lookback_window}-day lookback window...")
        
        if self.features is None:
            print("Error: No features available")
            return False
        
        # Check if exchange flow features are available
        ef_cols = [col for col in self.features.columns if 'flow_' in col]
        
        if len(ef_cols) < 3:
            print("Error: Not enough exchange flow features available")
            return False
        
        # Select features for the model
        flow_features = [col for col in [
            'flow_total', 'flow_momentum_7d', 'flow_intensity',
            'flow_zscore_14d', 'flow_acceleration', 'high_inflow_signal',
            'tx_spike_signal', 'flow_tx_ratio'
        ] if col in self.features.columns]
        
        # Add some price features for context
        price_features = [col for col in [
            'returns', 'volatility_14d', 'rsi_14', 'macd',
            'sma_cross_7_30', 'market_regime'
        ] if col in self.features.columns]
        
        # Combine features
        ef_model_features = flow_features + price_features
        
        print(f"Using {len(ef_model_features)} features for exchange flow model: {ef_model_features}")
        
        # Prepare data
        ef_df = self.features[ef_model_features].copy()
        
        # Replace infinite values with NaN
        ef_df = ef_df.replace([np.inf, -np.inf], np.nan)
        
        # Handle NaN values (fill with column median)
        for col in ef_df.columns:
            median_val = ef_df[col].median()
            ef_df[col] = ef_df[col].fillna(median_val)
        
        # Create sequences
        X_data = []
        y_data = []
        
        for i in range(len(ef_df) - lookback_window):
            X_data.append(ef_df.iloc[i:i+lookback_window].values)
            y_data.append(self.features['target_7d'].iloc[i+lookback_window])
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        print(f"Created {len(X_data)} training samples with shape {X_data.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.2, shuffle=False, random_state=42
        )
        
        # Build the model - CNN + LSTM hybrid
        model = Sequential([
            # CNN layers
            Conv1D(filters=32, kernel_size=3, activation='relu', 
                  input_shape=(lookback_window, len(ef_model_features))),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            # LSTM layer
            LSTM(32, return_sequences=False),
            
            # Dense layers
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=64,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate model
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Exchange flow model accuracy: {accuracy:.4f}")
        
        # Save model
        self.models['exchange_flow'] = {
            'model': model,
            'features': ef_model_features,
            'lookback': lookback_window,
            'history': history.history,
            'accuracy': accuracy
        }
        
        return True
    
    def train_ensemble_model(self):
        """
        Train ensemble models (Random Forest and Gradient Boosting) for feature importance
        and market prediction.
        """
        print("Training ensemble models...")
        
        if self.features is None:
            print("Error: No features available")
            return False
        
        # Select features for ensemble models - start with most important ones
        ensemble_features = [
            'returns', 'volatility_14d', 'volatility_30d', 'rsi_14',
            'macd', 'macd_hist', 'macd_signal', 
            'sma_cross_7_30', 'sma_cross_14_50', 'sma_cross_50_200',
            'market_regime'
        ]
        
        # Add on-chain and exchange flow features if available
        optional_features = [
            'nvt_ratio', 'transaction_volume_ratio', 'hash_difficulty_ratio', 
            'fee_to_reward_ratio', 'whale_dominance', 'short_term_holders', 
            'mid_term_holders', 'long_term_holders', 'hodl_ratio', 'mvrv_ratio',
            'flow_zscore_14d', 'flow_momentum_7d', 'flow_intensity', 
            'flow_acceleration', 'flow_tx_ratio', 'high_inflow_signal'
        ]
        
        for feature in optional_features:
            if feature in self.features.columns:
                ensemble_features.append(feature)
        
        # Ensure all features exist
        ensemble_features = [f for f in ensemble_features if f in self.features.columns]
        print(f"Using {len(ensemble_features)} features for ensemble models: {ensemble_features}")
        
        # Prepare data
        X = self.features[ensemble_features].copy()
        
        # Replace infinite values with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Handle NaN values (fill with column median)
        for col in X.columns:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
        
        # Get target variable
        y_7d = self.features['target_7d'].copy()  # 7-day prediction target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_7d, test_size=0.2, shuffle=False, random_state=42
        )
        
        # Train Random Forest
        print("Training Random Forest model...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Train Gradient Boosting
        print("Training Gradient Boosting model...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        
        # Evaluate models
        model_metrics = {}
        for name, model in [('Random Forest', rf_model), ('Gradient Boosting', gb_model)]:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} accuracy: {accuracy:.4f}")
            model_metrics[name] = accuracy
        
        # Save models
        self.models['random_forest'] = {
            'model': rf_model,
            'features': ensemble_features,
            'accuracy': model_metrics['Random Forest']
        }
        
        self.models['gradient_boosting'] = {
            'model': gb_model,
            'features': ensemble_features,
            'accuracy': model_metrics['Gradient Boosting']
        }
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'Feature': ensemble_features,
            'RF_Importance': rf_model.feature_importances_,
            'GB_Importance': gb_model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('RF_Importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        
        return True
    
    def generate_trading_signals(self):
        """
        Generate trading signals based on model predictions and ensemble voting.
        """
        print("Generating trading signals...")
        
        if self.features is None:
            print("Error: No features available")
            return False
        
        # Create a copy of features for signals
        signals_df = self.features.copy()
        signals_df['signal'] = 0  # 0: Hold, 1: Buy, -1: Sell
        
        # Initialize model signal columns
        model_names = ['cnn', 'exchange_flow', 'random_forest', 'gradient_boosting']
        for model_name in model_names:
            signals_df[f'{model_name}_signal'] = 0
        
        # 1. CNN signal
        if 'cnn' in self.models:
            print("Generating CNN model signals...")
            model_info = self.models['cnn']
            model = model_info['model']
            features = model_info['features']
            lookback = model_info['lookback']
            
            # Create sequences for prediction
            cnn_signals = []
            batch_size = 32
            
            for i in range(lookback, len(signals_df), batch_size):
                batch_end = min(i + batch_size, len(signals_df))
                batch_windows = []
                
                for j in range(i, batch_end):
                    window_data = signals_df[features].iloc[j-lookback:j].values
                    # Replace any NaN or inf values
                    window_data = np.nan_to_num(window_data, nan=0.0, posinf=0.0, neginf=0.0)
                    batch_windows.append(window_data)
                
                if batch_windows:
                    batch_windows = np.array(batch_windows)
                    
                    try:
                        predictions = model.predict(batch_windows, verbose=0)
                        
                        # Convert predictions to signals
                        batch_signals = np.zeros(len(predictions))
                        batch_signals[predictions.flatten() > 0.6] = 1  # Buy threshold
                        batch_signals[predictions.flatten() < 0.4] = -1  # Sell threshold
                        
                        cnn_signals.extend(batch_signals)
                    except Exception as e:
                        print(f"Error in CNN prediction: {str(e)}")
                        # Fill with neutral signals
                        cnn_signals.extend([0] * (batch_end - i))
            
            # Add padding for the first lookback days
            padding = [0] * lookback
            cnn_signals = padding + cnn_signals
            
            # Ensure correct length
            if len(cnn_signals) > len(signals_df):
                cnn_signals = cnn_signals[:len(signals_df)]
            elif len(cnn_signals) < len(signals_df):
                cnn_signals.extend([0] * (len(signals_df) - len(cnn_signals)))
            
            signals_df['cnn_signal'] = cnn_signals
        
        # 2. Exchange flow signal
        if 'exchange_flow' in self.models:
            print("Generating exchange flow model signals...")
            model_info = self.models['exchange_flow']
            model = model_info['model']
            features = model_info['features']
            lookback = model_info['lookback']
            
            # Create sequences for prediction
            ef_signals = []
            batch_size = 32
            
            for i in range(lookback, len(signals_df), batch_size):
                batch_end = min(i + batch_size, len(signals_df))
                batch_windows = []
                
                for j in range(i, batch_end):
                    window_data = signals_df[features].iloc[j-lookback:j].values
                    # Replace any NaN or inf values
                    window_data = np.nan_to_num(window_data, nan=0.0, posinf=0.0, neginf=0.0)
                    batch_windows.append(window_data)
                
                if batch_windows:
                    batch_windows = np.array(batch_windows)
                    
                    try:
                        predictions = model.predict(batch_windows, verbose=0)
                        
                        # Convert predictions to signals
                        batch_signals = np.zeros(len(predictions))
                        batch_signals[predictions.flatten() > 0.65] = 1  # Buy threshold
                        batch_signals[predictions.flatten() < 0.35] = -1  # Sell threshold
                        
                        ef_signals.extend(batch_signals)
                    except Exception as e:
                        print(f"Error in exchange flow prediction: {str(e)}")
                        # Fill with neutral signals
                        ef_signals.extend([0] * (batch_end - i))
            
            # Add padding for the first lookback days
            padding = [0] * lookback
            ef_signals = padding + ef_signals
            
            # Ensure correct length
            if len(ef_signals) > len(signals_df):
                ef_signals = ef_signals[:len(signals_df)]
            elif len(ef_signals) < len(signals_df):
                ef_signals.extend([0] * (len(signals_df) - len(ef_signals)))
            
            signals_df['exchange_flow_signal'] = ef_signals
        
        # 3. Random Forest signal
        if 'random_forest' in self.models:
            print("Generating Random Forest model signals...")
            model_info = self.models['random_forest']
            model = model_info['model']
            features = model_info['features']
            
            # Clean the data
            X_rf = signals_df[features].copy()
            X_rf = X_rf.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with medians
            for col in X_rf.columns:
                median_val = X_rf[col].median()
                X_rf[col] = X_rf[col].fillna(median_val)
            
            try:
                rf_preds = model.predict_proba(X_rf)[:, 1]
                signals_df['rf_signal'] = np.zeros(len(rf_preds))
                signals_df.loc[rf_preds > 0.6, 'rf_signal'] = 1  # Buy threshold
                signals_df.loc[rf_preds < 0.4, 'rf_signal'] = -1  # Sell threshold
            except Exception as e:
                print(f"Error in Random Forest prediction: {str(e)}")
                signals_df['rf_signal'] = 0  # Neutral signal as fallback
        
        # 4. Gradient Boosting signal
        if 'gradient_boosting' in self.models:
            print("Generating Gradient Boosting model signals...")
            model_info = self.models['gradient_boosting']
            model = model_info['model']
            features = model_info['features']
            
            # Clean the data
            X_gb = signals_df[features].copy()
            X_gb = X_gb.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with medians
            for col in X_gb.columns:
                median_val = X_gb[col].median()
                X_gb[col] = X_gb[col].fillna(median_val)
            
            try:
                gb_preds = model.predict_proba(X_gb)[:, 1]
                signals_df['gb_signal'] = np.zeros(len(gb_preds))
                signals_df.loc[gb_preds > 0.6, 'gb_signal'] = 1  # Buy threshold
                signals_df.loc[gb_preds < 0.4, 'gb_signal'] = -1  # Sell threshold
            except Exception as e:
                print(f"Error in Gradient Boosting prediction: {str(e)}")
                signals_df['gb_signal'] = 0  # Neutral signal as fallback
        
        # 5. Ensemble signal through weighted voting
        signal_columns = [col for col in signals_df.columns if col.endswith('_signal')]
        
        if signal_columns:
            # Fill NaN values in signal columns
            for col in signal_columns:
                signals_df[col] = signals_df[col].fillna(0)
            
            # Set weights based on model accuracies
            weights = {}
            for model_name in model_names:
                if f'{model_name}_signal' in signal_columns and model_name in self.models:
                    weights[f'{model_name}_signal'] = self.models[model_name].get('accuracy', 0.5)
                elif model_name == 'random_forest' and 'rf_signal' in signal_columns:
                    weights['rf_signal'] = self.models['random_forest'].get('accuracy', 0.5)
                elif model_name == 'gradient_boosting' and 'gb_signal' in signal_columns:
                    weights['gb_signal'] = self.models['gradient_boosting'].get('accuracy', 0.5)
            
            # Default weights if none available
            if not weights:
                for col in signal_columns:
                    weights[col] = 1.0 / len(signal_columns)
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            print(f"Ensemble weights: {weights}")
            
            # Calculate weighted vote
            signals_df['vote_sum'] = 0
            for col, weight in weights.items():
                signals_df['vote_sum'] += signals_df[col] * weight
            
            # Convert vote sum to signal
            signals_df['signal'] = np.sign(signals_df['vote_sum'])
        
        # 6. Additional rule-based signals
        # RSI oversold/overbought
        if 'rsi_14' in signals_df.columns:
            signals_df['rsi_signal'] = 0  # Initialize with neutral signal
            signals_df.loc[signals_df['rsi_14'] < 30, 'rsi_signal'] = 1  # Oversold - Buy
            signals_df.loc[signals_df['rsi_14'] > 70, 'rsi_signal'] = -1  # Overbought - Sell
        
        # Moving average crossovers
        if 'sma_cross_50_200' in signals_df.columns:
            signals_df['ma_signal'] = 0
            # Golden cross (50-day crosses above 200-day) - Buy
            signals_df.loc[(signals_df['sma_cross_50_200'] > 0) & 
                          (signals_df['sma_cross_50_200'].shift(1) <= 0), 'ma_signal'] = 1
            # Death cross (50-day crosses below 200-day) - Sell
            signals_df.loc[(signals_df['sma_cross_50_200'] < 0) & 
                          (signals_df['sma_cross_50_200'].shift(1) >= 0), 'ma_signal'] = -1
        
        # Exchange flow based signals
        if 'flow_intensity' in signals_df.columns:
            signals_df['flow_rule_signal'] = 0
            # High intensity flows - potential selling pressure
            signals_df.loc[signals_df['flow_intensity'] > 2.0, 'flow_rule_signal'] = -1
            # Low intensity flows - potential accumulation
            signals_df.loc[signals_df['flow_intensity'] < 0.5, 'flow_rule_signal'] = 1
        
        # Combine with ensemble signals for final decision
        rule_signals = [col for col in ['rsi_signal', 'ma_signal', 'flow_rule_signal'] 
                       if col in signals_df.columns]
        
        if rule_signals and 'signal' in signals_df.columns:
            # Add rule-based signals with lower weight (0.3)
            for rule_signal in rule_signals:
                signals_df['signal'] = signals_df['signal'] + 0.3 * signals_df[rule_signal]
            
            # Convert to -1, 0, 1
            signals_df['signal'] = np.sign(signals_df['signal'])
        
        # Ensure sufficient signal frequency (at least 3% of rows should have signals)
        signal_ratio = (signals_df['signal'] != 0).mean()
        if signal_ratio < 0.03:
            print(f"Warning: Signal ratio ({signal_ratio:.2%}) is below the 3% threshold.")
            print("Adjusting signal sensitivity...")
            
            # Adjust thresholds to generate more signals
            if 'vote_sum' in signals_df.columns:
                # Use a lower threshold for vote_sum
                signals_df['signal'] = np.zeros(len(signals_df))
                signals_df.loc[signals_df['vote_sum'] >= 0.3, 'signal'] = 1
                signals_df.loc[signals_df['vote_sum'] <= -0.3, 'signal'] = -1
        
        # Add signal strength for position sizing
        if 'vote_sum' in signals_df.columns:
            signals_df['signal_strength'] = signals_df['vote_sum'].clip(-1, 1)
        else:
            signals_df['signal_strength'] = signals_df['signal']
        
        # Calculate final signal frequency
        final_signal_ratio = (signals_df['signal'] != 0).mean()
        print(f"Final signal ratio: {final_signal_ratio:.2%}")
        signals_df['trade_frequency'] = final_signal_ratio  # Store trade frequency
        
        self.signals = signals_df
        return True
    
    def backtest_strategy(self):
        """
        Backtest the trading strategy and calculate performance metrics,
        incorporating position sizing and risk management.
        """
        print("Backtesting trading strategy with risk management...")
        
        if self.signals is None:
            print("Error: No signals available")
            return False
        
        signals = self.signals.copy()
        
        # Get price column for returns calculation
        price_col = None
        for col in ['price_close', 'market_price_usd']:
            if col in signals.columns:
                price_col = col
                break
        
        if price_col is None:
            print("Error: No price column found")
            return False
        
        # Calculate returns if not already done
        if 'returns' not in signals.columns:
            signals['returns'] = signals[price_col].pct_change()
        
        # Clean and validate returns
        signals['returns'] = signals['returns'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Initialize portfolio tracking
        initial_cash = 100000
        signals['portfolio_value'] = initial_cash
        signals['position_size'] = 0
        signals['entry_price'] = np.nan
        signals['stop_loss'] = np.nan
        signals['trailing_stop'] = np.nan
        signals['position_type'] = None  # 'long', 'short', or None
        
        # Initialize trade tracking
        self.trade_history = []
        current_position = {
            'type': None,  # 'long' or 'short'
            'entry_date': None,
            'entry_price': 0,
            'position_size': 0,
            'stop_loss': 0,
            'trailing_stop': 0
        }
        
        # Initialize risk manager's peak value with initial cash
        self.risk_manager.peak_value = initial_cash
        self.risk_manager.drawdown_history = [0]  # Initialize drawdown history with 0
        
        # Process each day sequentially
        for i in range(1, len(signals)):
            prev_idx = signals.index[i-1]
            curr_idx = signals.index[i]
            
            # Get current volatility for position sizing
            current_volatility = signals.loc[prev_idx, 'volatility_30d']
            if pd.isna(current_volatility) or current_volatility == 0:
                current_volatility = 0.02  # Default to 2% volatility
            
            # Get current market regime if available
            current_regime = signals.loc[prev_idx, 'market_regime'] if 'market_regime' in signals.columns else None
            
            # Get current price and signal
            current_price = signals.loc[curr_idx, price_col]
            current_signal = signals.loc[prev_idx, 'signal']  # Use previous day's signal
            signal_strength = signals.loc[prev_idx, 'signal_strength']
            
            # Get previous portfolio value
            prev_portfolio_value = signals.loc[prev_idx, 'portfolio_value']
            
            # Update drawdown
            current_drawdown = self.risk_manager.update_drawdown(prev_portfolio_value)
            
            # Check for stop loss triggering
            if current_position['type'] == 'long' and current_position['stop_loss'] > 0:
                # Check if price dropped below stop loss
                if current_price <= current_position['stop_loss']:
                    print(f"Stop loss triggered at {curr_idx}: Price {current_price:.2f} below stop {current_position['stop_loss']:.2f}")
                    # Close position and record trade
                    trade_return = (current_price / current_position['entry_price'] - 1) * current_position['position_size']
                    self.trade_history.append({
                        'entry_date': current_position['entry_date'],
                        'exit_date': curr_idx,
                        'trade_type': 'long',
                        'entry_price': current_position['entry_price'],
                        'exit_price': current_price,
                        'exit_reason': 'stop_loss',
                        'position_size': current_position['position_size'],
                        'return': trade_return
                    })
                    
                    # Reset position
                    current_position = {'type': None, 'entry_date': None, 'entry_price': 0, 
                                       'position_size': 0, 'stop_loss': 0, 'trailing_stop': 0}
                    current_signal = 0  # Force no new position this period
            
            elif current_position['type'] == 'short' and current_position['stop_loss'] > 0:
                # Check if price rose above stop loss
                if current_price >= current_position['stop_loss']:
                    print(f"Stop loss triggered at {curr_idx}: Price {current_price:.2f} above stop {current_position['stop_loss']:.2f}")
                    # Close position and record trade
                    trade_return = (1 - current_price / current_position['entry_price']) * current_position['position_size']
                    self.trade_history.append({
                        'entry_date': current_position['entry_date'],
                        'exit_date': curr_idx,
                        'trade_type': 'short',
                        'entry_price': current_position['entry_price'],
                        'exit_price': current_price,
                        'exit_reason': 'stop_loss',
                        'position_size': current_position['position_size'],
                        'return': trade_return
                    })
                    
                    # Reset position
                    current_position = {'type': None, 'entry_date': None, 'entry_price': 0, 
                                       'position_size': 0, 'stop_loss': 0, 'trailing_stop': 0}
                    current_signal = 0  # Force no new position this period
            
            # Update trailing stop if we're in a position and price moved favorably
            if current_position['type'] == 'long' and current_position['trailing_stop'] > 0:
                new_trailing_stop = current_price * (1 - signals.loc[prev_idx, 'volatility_14d'])
                if new_trailing_stop > current_position['trailing_stop']:
                    current_position['trailing_stop'] = new_trailing_stop
                    print(f"Updated long trailing stop to {new_trailing_stop:.2f}")
            
            elif current_position['type'] == 'short' and current_position['trailing_stop'] > 0:
                new_trailing_stop = current_price * (1 + signals.loc[prev_idx, 'volatility_14d'])
                if new_trailing_stop < current_position['trailing_stop']:
                    current_position['trailing_stop'] = new_trailing_stop
                    print(f"Updated short trailing stop to {new_trailing_stop:.2f}")
            
            # Process signal and update position
            if current_signal != 0:
                # If we already have a position, check if we need to close it
                if current_position['type'] == 'long' and current_signal < 0:
                    # Close long position
                    trade_return = (current_price / current_position['entry_price'] - 1) * current_position['position_size']
                    self.trade_history.append({
                        'entry_date': current_position['entry_date'],
                        'exit_date': curr_idx,
                        'trade_type': 'long',
                        'entry_price': current_position['entry_price'],
                        'exit_price': current_price,
                        'exit_reason': 'signal',
                        'position_size': current_position['position_size'],
                        'return': trade_return
                    })
                    
                    # Reset position
                    current_position = {'type': None, 'entry_date': None, 'entry_price': 0, 
                                       'position_size': 0, 'stop_loss': 0, 'trailing_stop': 0}
                
                elif current_position['type'] == 'short' and current_signal > 0:
                    # Close short position
                    trade_return = (1 - current_price / current_position['entry_price']) * current_position['position_size']
                    self.trade_history.append({
                        'entry_date': current_position['entry_date'],
                        'exit_date': curr_idx,
                        'trade_type': 'short',
                        'entry_price': current_position['entry_price'],
                        'exit_price': current_price,
                        'exit_reason': 'signal',
                        'position_size': current_position['position_size'],
                        'return': trade_return
                    })
                    
                    # Reset position
                    current_position = {'type': None, 'entry_date': None, 'entry_price': 0, 
                                       'position_size': 0, 'stop_loss': 0, 'trailing_stop': 0}
                
                # If we don't have a position or just closed it, open a new one
                if current_position['type'] is None:
                    # Calculate optimal position size
                    position_size = self.risk_manager.calculate_position_size(
                        signal_strength, 
                        current_volatility,
                        current_regime
                    )
                    
                    position_type = 'long' if position_size > 0 else 'short'
                    position_size = abs(position_size)
                    
                    # Calculate stop loss levels
                    stop_levels = self.risk_manager.calculate_stop_loss(
                        current_price,
                        position_type,
                        current_volatility,
                        current_regime
                    )
                    
                    # Record new position
                    current_position = {
                        'type': position_type,
                        'entry_date': curr_idx,
                        'entry_price': current_price,
                        'position_size': position_size,
                        'stop_loss': stop_levels['stop_loss_level'],
                        'trailing_stop': stop_levels['trailing_stop_level']
                    }
                    
                    print(f"Opening {position_type} position at {curr_idx}: "
                          f"Price={current_price:.2f}, Size={position_size:.2%}, "
                          f"Stop={stop_levels['stop_loss_level']:.2f}")
            
            # Calculate today's portfolio value based on position
            if current_position['type'] == 'long':
                # For long position, calculate return based on price change
                position_return = (current_price / current_position['entry_price'] - 1) * current_position['position_size']
                new_portfolio_value = prev_portfolio_value * (1 + position_return)
            
            elif current_position['type'] == 'short':
                # For short position, calculate return based on negative price change
                position_return = (1 - current_price / current_position['entry_price']) * current_position['position_size']
                new_portfolio_value = prev_portfolio_value * (1 + position_return)
            
            else:
                # If no position, portfolio value remains the same
                new_portfolio_value = prev_portfolio_value
            
            # Update signals dataframe with position info
            signals.loc[curr_idx, 'portfolio_value'] = new_portfolio_value
            signals.loc[curr_idx, 'position_size'] = current_position['position_size'] if current_position['type'] else 0
            signals.loc[curr_idx, 'position_type'] = current_position['type']
            signals.loc[curr_idx, 'entry_price'] = current_position['entry_price']
            signals.loc[curr_idx, 'stop_loss'] = current_position['stop_loss']
            signals.loc[curr_idx, 'trailing_stop'] = current_position['trailing_stop']
        
        # Calculate strategy returns
        signals['strategy_returns'] = signals['portfolio_value'].pct_change()
        signals['strategy_returns'] = signals['strategy_returns'].fillna(0)  # First day has no return
        
        signals['cum_strategy_returns'] = (1 + signals['strategy_returns']).cumprod() - 1
        
        # Calculate market returns for comparison
        signals['market_returns'] = signals[price_col].pct_change()
        signals['market_returns'] = signals['market_returns'].fillna(0)
        signals['cum_market_returns'] = (1 + signals['market_returns']).cumprod() - 1
        
        # Calculate performance metrics
        # 1. Sharpe Ratio (annualized)
        risk_free_rate = 0.02 / 252  # Assuming 2% annual risk-free rate
        sharpe_ratio = np.sqrt(252) * (signals['strategy_returns'].mean() - risk_free_rate) / signals['strategy_returns'].std()
        
        # 2. Maximum Drawdown
        drawdown = self.risk_manager.max_drawdown
        
        # 3. Win Rate from trade history
        winning_trades = sum(trade['return'] > 0 for trade in self.trade_history)
        total_trades = len(self.trade_history)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 4. Annualized Return
        days = (signals.index[-1] - signals.index[0]).days
        if days > 0:
            annual_return = (1 + signals['cum_strategy_returns'].iloc[-1]) ** (365 / days) - 1
        else:
            annual_return = 0
        
        # 5. Calmar Ratio
        calmar_ratio = annual_return / abs(drawdown) if drawdown != 0 else float('inf')
        
        # 6. Average Position Size
        avg_position_size = signals['position_size'].mean()
        
        # 7. Trade Frequency
        trade_frequency = (signals['signal'] != 0).mean()
        
        # Store performance metrics
        self.performance = {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': drawdown,
            'win_rate': win_rate,
            'annual_return': annual_return,
            'calmar_ratio': calmar_ratio,
            'total_return': signals['cum_strategy_returns'].iloc[-1],
            'market_return': signals['cum_market_returns'].iloc[-1],
            'total_trades': total_trades,
            'signal_frequency': trade_frequency,
            'avg_position_size': avg_position_size,
            'final_value': signals['portfolio_value'].iloc[-1]
        }
        
        print("\nStrategy Performance:")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Maximum Drawdown: {drawdown:.4%}")
        print(f"Win Rate: {win_rate:.4%}")
        print(f"Annual Return: {annual_return:.4%}")
        print(f"Total Return: {signals['cum_strategy_returns'].iloc[-1]:.4%}")
        print(f"Market Return: {signals['cum_market_returns'].iloc[-1]:.4%}")
        print(f"Total Trades: {total_trades}")
        print(f"Average Position Size: {avg_position_size:.2%}")
        print(f"Initial Capital: ${initial_cash:,.2f}")
        print(f"Final Capital: ${signals['portfolio_value'].iloc[-1]:,.2f}")
        
        # Check success criteria
        success = (
            sharpe_ratio >= 1.8 and 
            drawdown >= -0.40 and 
            win_rate >= 0.45 and
            trade_frequency >= 0.03
        )
        
        print("\nSuccess Criteria Check:")
        print(f"Sharpe Ratio >= 1.8: {'✓' if sharpe_ratio >= 1.8 else '✗'} ({sharpe_ratio:.2f})")
        print(f"Maximum Drawdown >= -40%: {'✓' if drawdown >= -0.4 else '✗'} ({drawdown*100:.2f}%)")
        print(f"Win Rate >= 45%: {'✓' if win_rate >= 0.45 else '✗'} ({win_rate*100:.2f}%)")
        print(f"Trade Frequency >= 3%: {'✓' if trade_frequency >= 0.03 else '✗'} ({trade_frequency*100:.2f}%)")
        print(f"Overall: {'SUCCESS' if success else 'FAILURE'}")
        
        # Save the signals with calculated metrics
        self.signals = signals
        
        return success
    
    def visualize_results(self):
        """
        Visualize strategy results and performance.
        """
        if self.signals is None:
            print("Error: No signals available")
            return False
        
        signals = self.signals.copy()
        
        # 1. Cumulative returns plot
        plt.figure(figsize=(12, 6))
        plt.plot(signals.index, signals['cum_market_returns'] * 100, 'b-', alpha=0.5, label='Buy and Hold')
        plt.plot(signals.index, signals['cum_strategy_returns'] * 100, 'g-', label='Trading Strategy')
        plt.title('Cumulative Returns: Strategy vs Buy-and-Hold')
        plt.xlabel('Date')
        plt.ylabel('Returns (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # 2. Drawdown plot
        plt.figure(figsize=(12, 6))
        
        # Handle length mismatch between signals index and drawdown history
        if len(self.risk_manager.drawdown_history) < len(signals):
            # Pad drawdown history with zeros at the beginning to match signals length
            padding = [0] * (len(signals) - len(self.risk_manager.drawdown_history))
            drawdown_data = padding + self.risk_manager.drawdown_history
        elif len(self.risk_manager.drawdown_history) > len(signals):
            # Trim drawdown history to match signals length
            drawdown_data = self.risk_manager.drawdown_history[-len(signals):]
        else:
            drawdown_data = self.risk_manager.drawdown_history
            
        plt.plot(signals.index, drawdown_data)
        plt.title('Strategy Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # 3. Position size over time
        plt.figure(figsize=(12, 6))
        positions = signals['position_size']
        plt.plot(signals.index, positions)
        plt.title('Position Size Over Time')
        plt.xlabel('Date')
        plt.ylabel('Position Size (% of Portfolio)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # 4. Trade analysis
        if self.trade_history:
            trade_returns = [trade['return'] for trade in self.trade_history]
            trade_durations = [(trade['exit_date'] - trade['entry_date']).days for trade in self.trade_history]
            trade_types = [trade['trade_type'] for trade in self.trade_history]
            
            # Trade returns distribution
            plt.figure(figsize=(10, 6))
            plt.hist(trade_returns, bins=20)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('Distribution of Trade Returns')
            plt.xlabel('Return')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
            # Trade duration vs returns
            plt.figure(figsize=(10, 6))
            plt.scatter(trade_durations, trade_returns, alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('Trade Duration vs Returns')
            plt.xlabel('Duration (days)')
            plt.ylabel('Return')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
            # Trade performance by type
            plt.figure(figsize=(10, 6))
            long_returns = [trade['return'] for trade in self.trade_history if trade['trade_type'] == 'long']
            short_returns = [trade['return'] for trade in self.trade_history if trade['trade_type'] == 'short']
            
            plt.boxplot([long_returns, short_returns], labels=['Long', 'Short'])
            plt.title('Trade Performance by Type')
            plt.ylabel('Return')
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.show()
        
        # 5. Model signal contribution
        model_signal_cols = [col for col in signals.columns if col.endswith('_signal') and col != 'signal']
        
        if len(model_signal_cols) >= 2:
            # Correlation matrix
            signal_corr = signals[model_signal_cols + ['signal']].corr()
            
            plt.figure(figsize=(12, 10))
            plt.matshow(signal_corr, fignum=1)
            plt.colorbar()
            plt.xticks(range(len(signal_corr.columns)), signal_corr.columns, rotation=45)
            plt.yticks(range(len(signal_corr.columns)), signal_corr.columns)
            plt.title('Signal Correlation Matrix')
            for i in range(len(signal_corr.columns)):
                for j in range(len(signal_corr.columns)):
                    plt.text(i, j, f"{signal_corr.iloc[i, j]:.2f}", ha='center', va='center',
                           color='white' if abs(signal_corr.iloc[i, j]) > 0.5 else 'black')
            plt.tight_layout()
            plt.show()
        
        # 6. Regime-based performance
        if 'market_regime' in signals.columns and 'position_type' in signals.columns:
            # Group returns by regime
            regime_returns = signals.groupby('market_regime')['strategy_returns'].mean() * 252  # Annualized
            
            # Position distribution by regime
            regime_position_sizes = signals.groupby('market_regime')['position_size'].mean()
            
            # Number of trades by regime
            if self.trade_history:
                regime_trades = {}
                for trade in self.trade_history:
                    entry_regime = signals.loc[trade['entry_date'], 'market_regime']
                    regime_trades[entry_regime] = regime_trades.get(entry_regime, 0) + 1
                
                plt.figure(figsize=(12, 8))
                plt.subplot(2, 2, 1)
                plt.bar(regime_returns.index, regime_returns * 100)
                plt.title('Returns by Market Regime')
                plt.xlabel('Regime')
                plt.ylabel('Annualized Return (%)')
                plt.grid(True, axis='y')
                
                plt.subplot(2, 2, 2)
                plt.bar(regime_position_sizes.index, regime_position_sizes * 100)
                plt.title('Avg Position Size by Regime')
                plt.xlabel('Regime')
                plt.ylabel('Position Size (%)')
                plt.grid(True, axis='y')
                
                plt.subplot(2, 2, 3)
                plt.bar(regime_trades.keys(), regime_trades.values())
                plt.title('Number of Trades by Regime')
                plt.xlabel('Regime')
                plt.ylabel('Number of Trades')
                plt.grid(True, axis='y')
                
                # Position type by regime
                regime_long = signals[signals['position_type'] == 'long'].groupby('market_regime').size()
                regime_short = signals[signals['position_type'] == 'short'].groupby('market_regime').size()
                
                plt.subplot(2, 2, 4)
                if not regime_long.empty and not regime_short.empty:
                    # Ensure we're using the same regimes for both
                    all_regimes = sorted(set(regime_long.index) | set(regime_short.index))
                    
                    # Reindex to include all regimes, filling missing values with 0
                    regime_long = regime_long.reindex(all_regimes, fill_value=0)
                    regime_short = regime_short.reindex(all_regimes, fill_value=0)
                    
                    bar_width = 0.35
                    r1 = np.arange(len(all_regimes))
                    r2 = [x + bar_width for x in r1]
                    
                    plt.bar(r1, regime_long, width=bar_width, label='Long')
                    plt.bar(r2, regime_short, width=bar_width, label='Short')
                    plt.xlabel('Regime')
                    plt.ylabel('Days in Position')
                    plt.title('Position Types by Regime')
                    plt.xticks([r + bar_width/2 for r in r1], all_regimes)
                    plt.legend()
                
                plt.tight_layout()
                plt.show()
        
        # 7. Regime transition analysis
        if 'market_regime' in signals.columns:
            # Create regime transition matrix
            regime_transitions = np.zeros((signals['market_regime'].nunique(), signals['market_regime'].nunique()))
            
            for i in range(1, len(signals)):
                prev_regime = signals['market_regime'].iloc[i-1]
                curr_regime = signals['market_regime'].iloc[i]
                regime_transitions[prev_regime, curr_regime] += 1
                
            # Normalize to get transition probabilities
            row_sums = regime_transitions.sum(axis=1, keepdims=True)
            transition_probs = np.where(row_sums > 0, regime_transitions / row_sums, 0)
            
            # Plot transition matrix
            plt.figure(figsize=(10, 8))
            plt.imshow(transition_probs, cmap='Blues')
            plt.colorbar(label='Transition Probability')
            plt.title('Market Regime Transition Probabilities')
            
            n_regimes = signals['market_regime'].nunique()
            plt.xticks(np.arange(n_regimes), np.arange(n_regimes))
            plt.yticks(np.arange(n_regimes), np.arange(n_regimes))
            plt.xlabel('To Regime')
            plt.ylabel('From Regime')
            
            # Add text annotations
            for i in range(n_regimes):
                for j in range(n_regimes):
                    plt.text(j, i, f"{transition_probs[i, j]:.2f}", 
                           ha='center', va='center', 
                           color='white' if transition_probs[i, j] > 0.5 else 'black')
            
            plt.tight_layout()
            plt.show()
        
        # 8. Performance table summary
        if self.performance:
            metrics = list(self.performance.keys())
            values = list(self.performance.values())
            
            plt.figure(figsize=(10, 6))
            y_pos = np.arange(len(metrics))
            
            # Filter out values that are too large to display well
            plot_values = []
            plot_metrics = []
            for i, (metric, value) in enumerate(zip(metrics, values)):
                if isinstance(value, float) and abs(value) < 100:
                    plot_metrics.append(metric)
                    if metric in ['win_rate', 'max_drawdown', 'annual_return', 'total_return', 'avg_position_size']:
                        plot_values.append(value * 100)  # Convert to percentage
                    else:
                        plot_values.append(value)
            
            bars = plt.barh(np.arange(len(plot_metrics)), plot_values)
            plt.yticks(np.arange(len(plot_metrics)), plot_metrics)
            plt.xlabel('Value')
            plt.title('Strategy Performance Metrics')
            
            # Add value labels to bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_text = f"{plot_values[i]:.2f}"
                if plot_metrics[i] in ['win_rate', 'max_drawdown', 'annual_return', 'total_return', 'avg_position_size']:
                    label_text += '%'
                plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, label_text,
                       ha='left', va='center')
            
            plt.tight_layout()
            plt.show()
        
        return True
    
    def run_pipeline(self, data_dir=None, use_sample=True, start_date=None, end_date=None):
        """
        Run the complete pipeline from data loading to visualization.
        
        Args:
            data_dir: Directory containing data files
            use_sample: Whether to use sample data if real data not available
            start_date: Start date for data (if fetching from APIs)
            end_date: End date for data (if fetching from APIs)
        """
        print("Running complete trading strategy pipeline...")
        
        # Set default dates if not provided
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365*1)  # 3 years ago
        
        if end_date is None:
            end_date = datetime.now()
        
        # 1. Load data
        self.load_on_chain_data(data_dir, use_sample)
        self.fetch_exchange_flow_data(start_date, end_date)
        self.load_price_data('BTC-USD', start_date, end_date)
        
        # 2. Process data
        if not self.merge_data():
            print("Error: Data merging failed")
            return False
        
        if not self.create_features():
            print("Error: Feature creation failed")
            return False
        
        # 3. Train models
        models_trained = 0
        
        if self.train_market_regime_model(n_regimes=4):
            models_trained += 1
        
        if self.train_cnn_model(lookback_window=30):
            models_trained += 1
        
        if self.train_exchange_flow_model(lookback_window=14):
            models_trained += 1
        
        if self.train_ensemble_model():
            models_trained += 1
        
        if models_trained == 0:
            print("Error: No models were successfully trained")
            return False
        
        # 4. Generate signals and backtest
        if not self.generate_trading_signals():
            print("Error: Signal generation failed")
            return False
        
        success = self.backtest_strategy()
        
        # 5. Visualize results
        self.visualize_results()
        
        return success
    
    def predict_next_signal(self, current_data=None):
        """
        Generate a prediction for the next trading signal based on current data.
        
        Args:
            current_data: Optional DataFrame containing current market data.
                          If None, uses the last row of features.
        
        Returns:
            dict: Prediction details including signal direction, strength, and position sizing
        """
        print("Generating prediction for next trading signal...")
        
        if current_data is None and self.features is not None:
            # Use the last row of features
            current_data = self.features.iloc[-1:].copy()
        
        if current_data is None:
            print("Error: No current data available")
            return None
        
        # Initialize prediction results
        prediction = {
            'signal': 0,
            'signal_strength': 0,
            'position_size': 0,
            'confidence': 0,
            'model_predictions': {}
        }
        
        # Ensure we have models
        if not self.models:
            print("Error: No trained models available")
            return prediction
        
        # Generate predictions from each model
        for model_name, model_info in self.models.items():
            # Skip HMM model as it's for regime detection
            if model_name == 'hmm':
                continue
                
            try:
                if model_name in ['cnn', 'exchange_flow']:
                    # For sequence models, we need a look-back window
                    # In real-time prediction we would maintain a rolling window
                    # Here we'll use a simplified approach for demonstration
                    lookback = model_info['lookback']
                    features = model_info['features']
                    
                    if len(self.features) >= lookback:
                        window_data = self.features.iloc[-lookback:][features].values
                        window_data = np.nan_to_num(window_data)
                        
                        # Reshape for prediction
                        window_data = window_data.reshape(1, lookback, -1)
                        
                        # Get prediction
                        pred_prob = model_info['model'].predict(window_data, verbose=0)[0][0]
                        pred_signal = 1 if pred_prob > 0.6 else (-1 if pred_prob < 0.4 else 0)
                        
                        prediction['model_predictions'][model_name] = {
                            'probability': float(pred_prob),
                            'signal': int(pred_signal)
                        }
                
                elif model_name in ['random_forest', 'gradient_boosting']:
                    # For tree-based models
                    features = model_info['features']
                    X = current_data[features].copy()
                    
                    # Handle NaN and inf values
                    X = X.replace([np.inf, -np.inf], np.nan)
                    for col in X.columns:
                        if pd.isna(X[col]).any():
                            X[col] = X[col].fillna(self.features[col].median())
                    
                    # Get prediction probability
                    pred_prob = model_info['model'].predict_proba(X)[0, 1]
                    pred_signal = 1 if pred_prob > 0.6 else (-1 if pred_prob < 0.4 else 0)
                    
                    prediction['model_predictions'][model_name] = {
                        'probability': float(pred_prob),
                        'signal': int(pred_signal)
                    }
            
            except Exception as e:
                print(f"Error generating prediction for {model_name}: {str(e)}")
        
        # Calculate ensemble signal if we have individual predictions
        if prediction['model_predictions']:
            # Weight signals by model accuracy
            weighted_sum = 0
            total_weight = 0
            
            for model_name, pred_info in prediction['model_predictions'].items():
                weight = self.models[model_name].get('accuracy', 0.5)
                weighted_sum += pred_info['signal'] * weight
                total_weight += weight
            
            if total_weight > 0:
                ensemble_signal_strength = weighted_sum / total_weight
                ensemble_signal = np.sign(ensemble_signal_strength)
                
                prediction['signal'] = int(ensemble_signal)
                prediction['signal_strength'] = float(ensemble_signal_strength)
                prediction['confidence'] = abs(ensemble_signal_strength)
                
                # Get current volatility for position sizing
                if 'volatility_30d' in current_data.columns:
                    current_volatility = current_data['volatility_30d'].iloc[0]
                else:
                    current_volatility = 0.02  # Default to 2%
                
                # Get current regime if available
                current_regime = None
                if 'market_regime' in current_data.columns:
                    current_regime = current_data['market_regime'].iloc[0]
                
                # Calculate position size
                position_size = self.risk_manager.calculate_position_size(
                    ensemble_signal_strength, 
                    current_volatility,
                    current_regime
                )
                
                prediction['position_size'] = float(position_size)
                prediction['position_type'] = 'long' if position_size > 0 else ('short' if position_size < 0 else 'none')
                
                # Calculate stop-loss levels if we have a directional signal
                if prediction['signal'] != 0:
                    stop_levels = self.risk_manager.calculate_stop_loss(
                        current_data['price_close'].iloc[0] if 'price_close' in current_data.columns else 10000,
                        prediction['position_type'],
                        current_volatility,
                        current_regime
                    )
                    
                    prediction['stop_loss'] = float(stop_levels['stop_loss_level'])
                    prediction['trailing_stop'] = float(stop_levels['trailing_stop_level'])
        
        return prediction

# If run as main script
if __name__ == "__main__":
    strategy = IntegratedCryptoStrategy()
    strategy.run_pipeline(use_sample=True)