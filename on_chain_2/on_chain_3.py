import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from hmmlearn import hmm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling1D
from xgboost import XGBRegressor
import warnings
import copy
import random

# Optional imports for backtrader integration (if available)
try:
    import backtrader as bt
    import yfinance as yf
    BACKTRADER_AVAILABLE = True
except ImportError:
    BACKTRADER_AVAILABLE = False
    print("Backtrader and/or yfinance not available. Using built-in backtesting only.")

# Optional import for cvxpy optimization (if available)
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("CVXPY not available. Using simplified ensemble optimization.")

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

class IntegratedBitcoinStrategy:
    """
    An integrated self-learning trading strategy that leverages ML models and on-chain
    data to generate trading signals for Bitcoin. This strategy combines on-chain
    analytics with ensemble machine learning optimization.
    """
    
    def __init__(self):
        """Initialize the trading strategy."""
        self.data = {}
        self.merged_data = None
        self.features = None
        self.models = {}
        self.signals = None
        self.performance = {}
        self.ensemble_weights = None
        self.bt_strategies = {}
    
    def load_data(self, data_dir=None, ticker=None, start_date=None, end_date=None, use_sample_data=True):
        """
        Load data from multiple sources including blockchain.com and lookintobitcoin.
        Can also load market data via yfinance if ticker is specified.
        
        Args:
            data_dir (str): Directory containing on-chain data CSV files
            ticker (str): Ticker symbol (e.g., 'BTC-USD') for loading market data via yfinance
            start_date (datetime): Start date for market data
            end_date (datetime): End date for market data
            use_sample_data (bool): Whether to use sample data if no data source is provided
        """
        print("Loading data from multiple sources...")
        data_loaded = False
        
        # Load on-chain data if directory is provided
        if data_dir:
            # Load blockchain.com data
            try:
                self.data['blockchain_daily'] = pd.read_csv(f'{data_dir}/blockchain_dot_com_daily_data.csv')
                self.data['blockchain_hourly'] = pd.read_csv(f'{data_dir}/blockchain_dot_com_half_hourly_data.csv')
                
                # Load lookintobitcoin data
                self.data['bitcoin_daily'] = pd.read_csv(f'{data_dir}/look_into_bitcoin_daily_data.csv')
                self.data['address_balances'] = pd.read_csv(f'{data_dir}/look_into_bitcoin_address_balances_data.csv')
                self.data['hodl_waves'] = pd.read_csv(f'{data_dir}/look_into_bitcoin_hodl_waves_data.csv')
                self.data['realized_cap_hodl'] = pd.read_csv(f'{data_dir}/look_into_bitcoin_realised_cap_hodl_waves_data.csv')
                
                # Convert datetime columns
                for key, df in self.data.items():
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'])
                    elif 'Unnamed: 0' in df.columns and key == 'address_balances':
                        df['datetime'] = pd.to_datetime(df['Unnamed: 0'])
                        
                print("On-chain data loaded successfully")
                data_loaded = True
            except Exception as e:
                print(f"Error loading on-chain data: {str(e)}")
        
        # Load market data via yfinance if ticker is provided
        if ticker and BACKTRADER_AVAILABLE:
            try:
                if start_date is None:
                    start_date = datetime.now() - timedelta(days=5*365)  # 5 years by default
                if end_date is None:
                    end_date = datetime.now()
                
                print(f"Loading market data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                market_data = yf.download(ticker, start=start_date, end=end_date)
                
                # Standardize column names
                if isinstance(market_data.columns, pd.MultiIndex):
                    market_data.columns = [col[0].lower() for col in market_data.columns]
                else:
                    market_data.columns = [col.lower() for col in market_data.columns]
                
                self.data['market_data'] = market_data
                print("Market data loaded successfully")
                data_loaded = True
            except Exception as e:
                print(f"Error loading market data: {str(e)}")
        
        # If no data was loaded, create sample data
        if not data_loaded and use_sample_data:
            print("No data source provided. Using sample Bitcoin price data.")
            
            # Create sample date range
            dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
            
            # Create sample price data with some realistic patterns
            np.random.seed(42)  # For reproducibility
            
            # Start with a base price and add random walk
            base_price = 10000
            returns = np.random.normal(0.001, 0.03, len(dates))
            
            # Add a trend
            trend = np.linspace(0, 2, len(dates))
            returns = returns + trend * 0.001
            
            # Add some cyclical patterns
            cycle1 = np.sin(np.linspace(0, 10 * np.pi, len(dates))) * 0.02
            cycle2 = np.cos(np.linspace(0, 20 * np.pi, len(dates))) * 0.01
            returns = returns + cycle1 + cycle2
            
            # Calculate price from returns
            price = base_price * np.cumprod(1 + returns)
            
            # Add some jumps and crashes
            jump_points = np.random.choice(len(dates), 5, replace=False)
            for point in jump_points:
                jump_size = np.random.uniform(-0.15, 0.20)
                price[point:] = price[point:] * (1 + jump_size)
            
            # Create a DataFrame with OHLCV data
            market_data = pd.DataFrame({
                'open': price * np.random.uniform(0.98, 0.99, len(dates)),
                'high': price * np.random.uniform(1.01, 1.05, len(dates)),
                'low': price * np.random.uniform(0.95, 0.99, len(dates)),
                'close': price,
                'volume': np.random.lognormal(15, 1, len(dates))
            }, index=dates)
            
            # Add market data to the dataset
            self.data['market_data'] = market_data
            print(f"Sample data created with {len(dates)} days from {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
            data_loaded = True
            
        if not data_loaded:
            print("ERROR: No data loaded. Please provide a data directory or ticker symbol.")
            print("Run with use_sample_data=True to use sample data for testing purposes.")
            return False
            
        return True
    
    def merge_data(self):
        """
        Merge data from multiple sources and handle missing values.
        """
        print("Merging data from multiple sources...")
        
        if not self.data:
            print("Error: No data loaded. Please load data first.")
            return False
            
        # Check if on-chain data is available
        required_keys = ['blockchain_daily', 'bitcoin_daily']
        if all(key in self.data for key in required_keys):
            # Start with blockchain daily data
            blockchain_daily = self.data['blockchain_daily'].copy()
            blockchain_daily.set_index('datetime', inplace=True)
            
            # Merge with bitcoin daily data
            bitcoin_daily = self.data['bitcoin_daily'].copy()
            bitcoin_daily.set_index('datetime', inplace=True)
            merged = pd.merge(blockchain_daily, bitcoin_daily, 
                            left_index=True, right_index=True, 
                            how='inner', suffixes=('_bc', '_lib'))
            
            # Add address balance data if available
            if 'address_balances' in self.data:
                address_data = self.data['address_balances'].copy()
                address_data.set_index('datetime', inplace=True)
                address_data = address_data.drop('market_price_usd', axis=1, errors='ignore')  # Avoid duplicate columns
                merged = pd.merge(merged, address_data, left_index=True, right_index=True, how='inner')
            
            # Add HODL waves data if available
            if 'hodl_waves' in self.data:
                hodl_data = self.data['hodl_waves'].copy()
                hodl_data.set_index('datetime', inplace=True)
                hodl_data = hodl_data.drop('market_price_usd', axis=1, errors='ignore')  # Avoid duplicate columns
                merged = pd.merge(merged, hodl_data, left_index=True, right_index=True, how='inner')
            
            # Add realized cap HODL waves data if available
            if 'realized_cap_hodl' in self.data:
                real_hodl_data = self.data['realized_cap_hodl'].copy()
                real_hodl_data.set_index('datetime', inplace=True)
                real_hodl_data = real_hodl_data.drop('market_price_usd', axis=1, errors='ignore')  # Avoid duplicate columns
                merged = pd.merge(merged, real_hodl_data, 
                                left_index=True, right_index=True, 
                                how='inner', suffixes=('', '_realized'))
            
            # Handle missing values
            merged = merged.fillna(method='ffill').fillna(method='bfill')
            
            self.merged_data = merged
            print(f"On-chain data merged successfully. Shape: {merged.shape}")
            return True
        
        # If only market data is available
        elif 'market_data' in self.data:
            self.merged_data = self.data['market_data'].copy()
            print(f"Using market data only. Shape: {self.merged_data.shape}")
            return True
        
        else:
            print("No data available for merging. Please load data first.")
            return False
    
    def create_features(self):
        """
        Engineer features from raw data, including technical indicators,
        on-chain metrics, and market regime indicators.
        """
        print("Creating features...")
        
        if self.merged_data is None:
            print("Error: No merged data available. Run merge_data() first.")
            return
        
        df = self.merged_data.copy()
        
        # 1. Basic price features
        if 'market_price_usd_bc' in df.columns:
            # Use on-chain price data if available
            price_col = 'market_price_usd_bc'
        elif 'close' in df.columns:
            # Use market data if available
            price_col = 'close'
        else:
            # Try to find any column with 'price' or 'close' in it
            price_candidates = [col for col in df.columns if 'price' in col.lower() or 'close' in col.lower()]
            if price_candidates:
                price_col = price_candidates[0]
                print(f"Using {price_col} as price column")
            else:
                print("Error: No price column found in data")
                return
        
        # Calculate basic price features
        df['returns'] = df[price_col].pct_change()
        # Handle potential zero values before log calculation
        df['log_returns'] = np.log(df[price_col] / df[price_col].shift(1).replace(0, np.nan))
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
        rs = gain / loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 4. MACD (Moving Average Convergence Divergence)
        ema12 = df[price_col].ewm(span=12, adjust=False).mean()
        ema26 = df[price_col].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 5. Add Bollinger Bands
        rolling_mean = df[price_col].rolling(window=20).mean()
        rolling_std = df[price_col].rolling(window=20).std()
        df['bb_upper'] = rolling_mean + 2 * rolling_std
        df['bb_lower'] = rolling_mean - 2 * rolling_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / rolling_mean
        
        # 6. On-chain metrics (if available)
        if 'transaction_rate' in df.columns and 'average_block_size' in df.columns:
            df['transaction_volume_ratio'] = df['transaction_rate'] / df['average_block_size']
        
        if 'hash_rate' in df.columns and 'difficulty' in df.columns:
            df['hash_difficulty_ratio'] = df['hash_rate'] / df['difficulty']
        
        if 'total_transaction_fees' in df.columns and 'miners_revenue' in df.columns:
            df['fee_to_reward_ratio'] = df['total_transaction_fees'] / df['miners_revenue']
        
        if 'market_cap_usd_bc' in df.columns and 'transaction_rate' in df.columns:
            df['nvt_ratio'] = df['market_cap_usd_bc'] / df['transaction_rate']
        
        # 7. Address balance features (if available)
        balance_cols = ['addresses_with_1000_btc', 'addresses_with_100_btc',
                       'addresses_with_10_btc', 'addresses_with_1_btc',
                       'addresses_with_0.01_btc_x', 'addresses_with_0.01_btc_y']
        
        if all(col in df.columns for col in balance_cols):
            df['whale_dominance'] = (df['addresses_with_1000_btc'] + df['addresses_with_100_btc']) / (
                df['addresses_with_1000_btc'] + df['addresses_with_100_btc'] + 
                df['addresses_with_10_btc'] + df['addresses_with_1_btc'] + 
                df['addresses_with_0.01_btc_x'] + df['addresses_with_0.01_btc_y']
            )
        
        # 8. HODL wave metrics (if available)
        hodl_cols = ['24h', '1d_1w', '1w_1m', '1m_3m', '3m_6m', '6m_12m',
                    '1y_2y', '2y_3y', '3y_5y', '5y_7y', '7y_10y', '10y']
        
        if all(col in df.columns for col in hodl_cols):
            df['short_term_holders'] = df['24h'] + df['1d_1w'] + df['1w_1m'] + df['1m_3m']
            df['mid_term_holders'] = df['3m_6m'] + df['6m_12m'] + df['1y_2y']
            df['long_term_holders'] = df['2y_3y'] + df['3y_5y'] + df['5y_7y'] + df['7y_10y'] + df['10y']
            df['hodl_ratio'] = df['long_term_holders'] / df['short_term_holders']
        
        # 9. Market cap to realized cap ratio (MVRV)
        if 'realised_cap_usd' in df.columns:
            df['mvrv_ratio'] = df['market_cap_usd_bc'] / df['realised_cap_usd']
        
        # 10. Momentum indicators from ensemble.py
        df['momentum_5'] = df[price_col].pct_change(periods=5)
        
        # 11. Volume-based features (if volume data is available)
        volume_col = None
        if 'volume' in df.columns:
            volume_col = 'volume'
        elif 'transaction_count' in df.columns:
            volume_col = 'transaction_count'
            
        if volume_col:
            df['vol_5d_avg'] = df[volume_col].rolling(window=5).mean()
            df['vol_10d_avg'] = df[volume_col].rolling(window=10).mean()
            df['vol_ratio'] = df['vol_5d_avg'] / df['vol_10d_avg']
        
        # 12. Create target variables (price movement in next n days)
        for days in [1, 3, 7, 14]:
            future_price = df[price_col].shift(-days)
            df[f'target_{days}d'] = (future_price > df[price_col]).astype(int)
            df[f'future_ret_{days}d'] = df[price_col].pct_change(periods=days).shift(-days)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        self.features = df
        print(f"Feature engineering complete. Dataset shape: {df.shape}")
        
        # Show a preview of the features
        feature_subset = ['returns', 'volatility_14d', 'rsi_14', 'macd']
        
        # Add on-chain metrics if available
        if 'nvt_ratio' in df.columns:
            feature_subset.append('nvt_ratio')
        if 'whale_dominance' in df.columns:
            feature_subset.append('whale_dominance')
        if 'hodl_ratio' in df.columns:
            feature_subset.append('hodl_ratio')
            
        # Add target columns
        feature_subset.extend(['target_1d', 'target_7d'])
        
        print("\nFeature Preview:")
        print(df[feature_subset].tail(5))
    
    def split_data(self, train_pct=0.6, val_pct=0.2):
        """
        Split data into training, validation, and test sets
        
        Args:
            train_pct (float): Percentage of data to use for training
            val_pct (float): Percentage of data to use for validation
        
        Returns:
            train_data, val_data, test_data (pd.DataFrame): Split datasets
        """
        if self.features is None:
            print("Error: No feature data available. Run create_features() first.")
            return None, None, None
        
        df = self.features.copy()
        
        train_idx = int(len(df) * train_pct)
        val_idx = int(len(df) * (train_pct + val_pct))
        
        train_data = df.iloc[:train_idx].copy()
        val_data = df.iloc[train_idx:val_idx].copy()
        test_data = df.iloc[val_idx:].copy()
        
        print(f"Training set: {train_data.index.min()} to {train_data.index.max()} ({len(train_data)} samples)")
        print(f"Validation set: {val_data.index.min()} to {val_data.index.max()} ({len(val_data)} samples)")
        print(f"Test set: {test_data.index.min()} to {test_data.index.max()} ({len(test_data)} samples)")
        
        return train_data, val_data, test_data
    
    def train_market_regime_model(self, n_regimes=4):
        """
        Train Hidden Markov Model to identify market regimes.
        
        Args:
            n_regimes (int): Number of market regimes to identify
        """
        print(f"Training HMM for market regime detection with {n_regimes} regimes...")
        
        if self.features is None:
            print("Error: No feature data available. Run create_features() first.")
            return
        
        # Select features for regime detection
        regime_features = [
            'returns', 'volatility_30d', 'rsi_14', 
        ]
        
        # Add on-chain metrics if available
        if 'transaction_volume_ratio' in self.features.columns:
            regime_features.append('transaction_volume_ratio')
        if 'nvt_ratio' in self.features.columns:
            regime_features.append('nvt_ratio')
        if 'hodl_ratio' in self.features.columns:
            regime_features.append('hodl_ratio')
        
        # Check for infinite values and replace them
        df_features = self.features[regime_features].copy()
        
        # Replace infinite values with NaN
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        
        # Handle NaN values (fill with column median)
        for col in df_features.columns:
            median_val = df_features[col].median()
            df_features[col] = df_features[col].fillna(median_val)
            
        # Check for extreme values and clip them if necessary
        for col in df_features.columns:
            q1 = df_features[col].quantile(0.01)
            q3 = df_features[col].quantile(0.99)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_features[col] = df_features[col].clip(lower_bound, upper_bound)
        
        # Scale features
        X = df_features.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"Feature scaling completed successfully. Shape: {X_scaled.shape}")
        
        # Train HMM
        model = hmm.GaussianHMM(
            n_components=n_regimes, 
            covariance_type="full",
            n_iter=100, 
            random_state=42
        )
        model.fit(X_scaled)
        
        # Predict regimes
        hidden_states = model.predict(X_scaled)
        
        # Add regime predictions to features
        self.features['market_regime'] = hidden_states
        
        # Save model and scaler
        self.models['hmm'] = (model, scaler, regime_features)
        
        # Analyze regimes
        regime_stats = pd.DataFrame()
        for i in range(n_regimes):
            regime_data = self.features[self.features['market_regime'] == i]
            if 'returns' in regime_data.columns:
                regime_stats[f'Regime {i}'] = [
                    len(regime_data),
                    regime_data['returns'].mean() * 252,  # Annualized return
                    regime_data['volatility_30d'].mean() * np.sqrt(252),  # Annualized volatility
                    regime_data['rsi_14'].mean()
                ]
        
        regime_stats.index = ['Count', 'Ann. Return', 'Ann. Volatility', 'Avg RSI']
        print("\nMarket Regime Statistics:")
        print(regime_stats)
        
        print("Market regime model trained successfully")
    
    def train_cnn_model(self, lookback_window=30):
        """
        Train CNN model for pattern recognition in market data.
        
        Args:
            lookback_window (int): Number of days to look back for pattern recognition
        """
        print(f"Training CNN with {lookback_window}-day lookback window...")
        
        if self.features is None:
            print("Error: No feature data available. Run create_features() first.")
            return
        
        # Select features for the CNN
        cnn_features = [
            'returns', 'log_returns', 'volatility_14d', 'rsi_14',
            'macd', 'macd_hist', 'sma_cross_7_30', 'sma_cross_50_200',
        ]
        
        # Add market regime if available
        if 'market_regime' in self.features.columns:
            cnn_features.append('market_regime')
            
        # Add on-chain metrics if available
        if 'nvt_ratio' in self.features.columns:
            cnn_features.append('nvt_ratio')
        if 'short_term_holders' in self.features.columns:
            cnn_features.extend(['short_term_holders', 'long_term_holders'])
        
        # Clean the feature data first
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
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.2, shuffle=False, random_state=42
        )
        
        try:
            # Configure TensorFlow to use CPU for more stable execution
            tf.config.set_visible_devices([], 'GPU')
            print("TensorFlow configured to use CPU")
        except:
            print("Could not configure TensorFlow devices - continuing with default")
        
        # Build CNN model
        model = Sequential([
            Conv1D(filters=32, kernel_size=3, activation='relu', 
                   input_shape=(lookback_window, len(cnn_features))),
            MaxPooling1D(pool_size=2),
            Flatten(),
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
        
        # Train model
        print("Starting CNN model training")
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=100,  # Reduced for stability
            batch_size=64,
            verbose=1
        )
        
        # Evaluate model
        try:
            print("Evaluating CNN model...")
            y_pred_prob = model.predict(X_test, verbose=0, batch_size=32)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            accuracy = accuracy_score(y_test, y_pred)
            print(f"CNN model accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Error during model evaluation: {str(e)}")
            print("Continuing with untested model")
            accuracy = 0.0
        
        # Save model and features
        self.models['cnn'] = (model, cnn_features, lookback_window)
        
        return model, history
    
    def train_linear_model(self, X_train, y_train, X_val, y_val):
        """
        Train a linear regression model with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            best_model: Trained model
            best_score: Model validation score
        """
        print("Training Linear Regression model...")
        
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
                print(f"Linear model update: R² = {best_score:.4f}, Params = {best_params}")
        
        return best_model, best_score
    
    def train_xgboost_model(self, X_train, y_train, X_val, y_val):
        """
        Train an XGBoost model with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            best_model: Trained model
            best_score: Model validation score
        """
        print("Training XGBoost model...")
        
        pipeline = Pipeline([('xgb', XGBRegressor(random_state=42, verbosity=0))])
        
        param_grid = {
            'xgb__n_estimators': [100, 500],
            'xgb__learning_rate': [0.01, 0.1],
            'xgb__max_depth': [3, 5],
            'xgb__subsample': [0.8, 1.0]
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
                print(f"XGBoost update: R² = {best_score:.4f}, Params = {best_params}")
        
        return best_model, best_score
    
    def train_mlp_model(self, X_train, y_train, X_val, y_val):
        """
        Train a neural network (MLP) model with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            best_model: Trained model
            best_score: Model validation score
        """
        print("Training MLP model...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(random_state=42, max_iter=1000))
        ])
        
        param_grid = {
            'mlp__hidden_layer_sizes': [(64,), (128,)],
            'mlp__alpha': [1e-3, 1e-2],
            'mlp__learning_rate_init': [1e-3, 1e-2],
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
                print(f"MLP update: R² = {best_score:.4f}, Params = {best_params}")
        
        return best_model, best_score
    
    def train_ensemble_model(self):
        """
        Train ensemble models (Random Forest, Gradient Boosting, Linear, XGBoost, MLP)
        for feature importance and market prediction.
        """
        print("Training ensemble models...")
        
        # Get train/validation/test split
        train_data, val_data, test_data = self.split_data(train_pct=0.6, val_pct=0.2)
        
        # Select features for ensemble models
        ensemble_features = [
            'returns', 'volatility_14d', 'volatility_30d', 'rsi_14',
            'macd', 'macd_hist', 'macd_signal', 
            'sma_cross_7_30', 'sma_cross_14_50', 'sma_cross_50_200',
            'bb_width'
        ]
        
        # Add on-chain features if available
        additional_features = [
            'nvt_ratio', 'transaction_volume_ratio', 'hash_difficulty_ratio', 
            'fee_to_reward_ratio', 'whale_dominance',
            'short_term_holders', 'mid_term_holders', 'long_term_holders',
            'hodl_ratio', 'mvrv_ratio'
        ]
        
        for feature in additional_features:
            if feature in self.features.columns:
                ensemble_features.append(feature)
        
        # Add market regime if available
        if 'market_regime' in self.features.columns:
            ensemble_features.append('market_regime')
        
        # Add volume features if available
        if 'vol_ratio' in self.features.columns:
            ensemble_features.append('vol_ratio')
        
        print(f"Using {len(ensemble_features)} features: {ensemble_features}")
        
        # Prepare data and clean it
        X_train = train_data[ensemble_features].copy()
        X_val = val_data[ensemble_features].copy()
        X_test = test_data[ensemble_features].copy()
        
        for df in [X_train, X_val, X_test]:
            # Replace infinite values with NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Handle NaN values (fill with column median)
            for col in df.columns:
                median_val = df[col].median()
                if np.isnan(median_val):
                    # If median is also NaN, use 0 instead
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(median_val)
        
        # Prepare different target variables
        # Classification targets (price direction)
        y_class_train = train_data['target_7d'].copy()
        y_class_val = val_data['target_7d'].copy()
        y_class_test = test_data['target_7d'].copy()
        
        # Regression targets (future returns)
        y_reg_train = train_data['future_ret_7d'].copy()
        y_reg_val = val_data['future_ret_7d'].copy()
        y_reg_test = test_data['future_ret_7d'].copy()
        
        # 1. Train Random Forest (Classification)
        print("\n1. Training Random Forest Classifier...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_class_train)
        
        y_pred = rf_model.predict(X_val)
        accuracy = accuracy_score(y_class_val, y_pred)
        print(f"Random Forest Classifier accuracy: {accuracy:.4f}")
        
        # 2. Train Gradient Boosting (Classification)
        print("\n2. Training Gradient Boosting Classifier...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        gb_model.fit(X_train, y_class_train)
        
        y_pred = gb_model.predict(X_val)
        accuracy = accuracy_score(y_class_val, y_pred)
        print(f"Gradient Boosting Classifier accuracy: {accuracy:.4f}")
        
        # 3. Train Linear Regression (Regression)
        print("\n3. Training Linear Regression...")
        lr_model, lr_score = self.train_linear_model(
            X_train.values, y_reg_train.values, 
            X_val.values, y_reg_val.values
        )
        
        # 4. Train XGBoost (Regression)
        print("\n4. Training XGBoost Regressor...")
        xgb_model, xgb_score = self.train_xgboost_model(
            X_train.values, y_reg_train.values, 
            X_val.values, y_reg_val.values
        )
        
        # 5. Train MLP (Regression)
        print("\n5. Training MLP Regressor...")
        mlp_model, mlp_score = self.train_mlp_model(
            X_train.values, y_reg_train.values, 
            X_val.values, y_reg_val.values
        )
        
        # Save models and features
        self.models['random_forest'] = (rf_model, ensemble_features)
        self.models['gradient_boosting'] = (gb_model, ensemble_features)
        self.models['linear_regression'] = (lr_model, ensemble_features)
        self.models['xgboost'] = (xgb_model, ensemble_features)
        self.models['mlp'] = (mlp_model, ensemble_features)
        
        # Get feature importance from tree-based models
        feature_importance = pd.DataFrame({
            'Feature': ensemble_features,
            'RF_Importance': rf_model.feature_importances_,
            'GB_Importance': gb_model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('RF_Importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        
        # Optimize ensemble weights for regression models
        print("\nOptimizing ensemble weights for regression models...")
        reg_models = [lr_model, xgb_model, mlp_model]
        
        # Get predictions from each model
        val_predictions = np.column_stack([
            model.predict(X_val.values) for model in reg_models
        ])
        
        # Optimize weights
        weights = self.optimize_ensemble_weights(
            reg_models, X_val.values, y_reg_val.values, 
            alpha_l1=0.01, alpha_l2=0.01
        )
        
        self.ensemble_weights = weights
        print(f"Optimized weights: {weights}")
        
        # Test ensemble performance
        test_predictions = np.column_stack([
            model.predict(X_test.values) for model in reg_models
        ])
        ensemble_pred = test_predictions @ weights
        
        # Convert regression predictions to classification (direction)
        ensemble_class_pred = (ensemble_pred > 0).astype(int)
        ensemble_accuracy = accuracy_score(y_class_test, ensemble_class_pred)
        ensemble_mse = mean_squared_error(y_reg_test, ensemble_pred)
        ensemble_r2 = r2_score(y_reg_test, ensemble_pred)
        
        print(f"\nEnsemble Test Performance:")
        print(f"Accuracy (direction): {ensemble_accuracy:.4f}")
        print(f"MSE (returns): {ensemble_mse:.8f}")
        print(f"R² (returns): {ensemble_r2:.4f}")
        
        return {
            'rf_model': rf_model, 
            'gb_model': gb_model, 
            'lr_model': lr_model,
            'xgb_model': xgb_model, 
            'mlp_model': mlp_model,
            'feature_importance': feature_importance,
            'ensemble_weights': weights,
            'ensemble_accuracy': ensemble_accuracy,
            'ensemble_r2': ensemble_r2
        }
    
    def optimize_ensemble_weights(self, models, X_val, y_val, sum_to_1=True, 
                                  nonnegative=True, alpha_l1=0.0, alpha_l2=0.01):
        """
        Optimize ensemble weights for Sharpe ratio instead of R²
        
        Args:
            models: List of trained models
            X_val: Validation features
            y_val: Validation target values
            sum_to_1: Whether weights should sum to 1
            nonnegative: Whether weights should be non-negative
            alpha_l1: L1 regularization strength
            alpha_l2: L2 regularization strength
            
        Returns:
            weights: Optimized model weights
        """
        # Get predictions from each model
        predictions = np.column_stack([model.predict(X_val) for model in models])
        N, M = predictions.shape
        
        if CVXPY_AVAILABLE:
            try:
                # Create a grid of weight combinations to try
                if M <= 3:  # For small number of models, we can try a fine grid
                    grid_points = 11  # 0, 0.1, 0.2, ..., 1.0
                    weight_combinations = []
                    
                    if M == 2:
                        for w1 in np.linspace(0, 1, grid_points):
                            w2 = 1 - w1
                            weight_combinations.append([w1, w2])
                    elif M == 3:
                        for w1 in np.linspace(0, 1, grid_points):
                            for w2 in np.linspace(0, 1 - w1, grid_points):
                                w3 = 1 - w1 - w2
                                if w3 >= 0:
                                    weight_combinations.append([w1, w2, w3])
                    
                    # Evaluate Sharpe ratio for each combination
                    best_sharpe = -np.inf
                    best_weights = None
                    
                    for weights in weight_combinations:
                        # Calculate weighted predictions
                        y_pred = predictions @ np.array(weights)
                        
                        # Calculate returns (assuming predictions are returns)
                        returns = y_pred
                        
                        # Calculate Sharpe ratio
                        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
                        
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_weights = weights
                    
                    print(f"Grid search found weights with Sharpe ratio: {best_sharpe:.4f}")
                    return np.array(best_weights)
                
                else:
                    # Use CVXPY to optimize for Sharpe ratio
                    w = cp.Variable(M, nonneg=nonnegative)
                    
                    # Define constraints
                    constraints = []
                    if sum_to_1:
                        constraints.append(cp.sum(w) == 1)
                    
                    # Define returns
                    returns = predictions @ w
                    
                    # Objective: maximize Sharpe ratio (mean/std)
                    # Since CVXPY doesn't directly support fractional objectives,
                    # we'll use a proxy: maximize mean while constraining variance
                    target_risk = cp.Parameter(nonneg=True)
                    target_risk.value = 0.02  # Target annual volatility (adjust as needed)
                    
                    # Daily risk
                    daily_risk = target_risk / np.sqrt(252) 
                    
                    # Maximize return with constrained risk
                    objective = cp.Maximize(cp.mean(returns))
                    constraints.append(cp.std(returns) <= daily_risk)
                    
                    # Solve optimization problem
                    problem = cp.Problem(objective, constraints)
                    problem.solve(verbose=False)
                    
                    # Get optimal weights
                    w_opt = w.value
                    print("Optimized weights for constrained volatility using CVXPY")
                    return w_opt
            
            except Exception as e:
                print(f"Error in CVXPY optimization: {str(e)}")
                print("Using Sharpe ratio-based weights instead")
                w_opt = self._sharpe_based_weight_optimization(predictions, y_val)
        else:
            print("CVXPY not available, using Sharpe ratio-based weights instead")
            w_opt = self._sharpe_based_weight_optimization(predictions, y_val)
        
        # Calculate final prediction and Sharpe
        y_val_pred = predictions @ w_opt
        
        # Calculate Sharpe ratio (annualized)
        sharpe_val = np.mean(y_val_pred) / (np.std(y_val_pred) + 1e-8) * np.sqrt(252)
        print(f"Validation Sharpe ratio = {sharpe_val:.4f}")
        
        return w_opt

    def _sharpe_based_weight_optimization(self, predictions, y_val):
        """
        Simple ensemble weight optimization based on Sharpe ratios
        
        Args:
            predictions: Model predictions
            y_val: True values
            
        Returns:
            weights: Optimized weights
        """
        M = predictions.shape[1]
        
        # Calculate Sharpe ratio for each model
        sharpe_ratios = []
        for i in range(M):
            returns = predictions[:, i]
            # Avoid division by zero
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            # Handle negative Sharpe ratios
            sharpe_ratios.append(max(0.0001, sharpe))
        
        # Normalize weights to sum to 1
        weights = np.array(sharpe_ratios) / sum(sharpe_ratios) if sum(sharpe_ratios) > 0 else np.ones(M) / M
        
        return weights
    
    def generate_trading_signals(self):
        """
        Generate trading signals based on model predictions and ensemble voting,
        with improved risk management.
        """
        print("Generating trading signals with improved risk management...")
        
        if not self.models:
            print("Error: No trained models available. Train models first.")
            return
        
        # Create a copy of features for signals
        signals_df = self.features.copy()
        signals_df['signal'] = 0  # 0: Hold, 1: Buy, -1: Sell
        
        # 1. CNN signal
        if 'cnn' in self.models:
            self._generate_cnn_signals(signals_df)
        
        # 2. Random Forest signal
        if 'random_forest' in self.models:
            self._generate_random_forest_signals(signals_df)
        
        # 3. Gradient Boosting signal
        if 'gradient_boosting' in self.models:
            self._generate_gradient_boosting_signals(signals_df)
        
        # 4. Regression Ensemble signal (modified for volatility-based position sizing)
        if all(k in self.models for k in ['linear_regression', 'xgboost', 'mlp']) and self.ensemble_weights is not None:
            self._generate_regression_ensemble_signals(signals_df)
        
        # 5. Ensemble signal through voting
        signal_columns = [col for col in signals_df.columns if col.endswith('_signal')]
        if signal_columns:
            # Fill NaN values in signal columns with 0 (neutral signal)
            for col in signal_columns:
                signals_df[col] = signals_df[col].fillna(0)
            
            signals_df['vote_sum'] = signals_df[signal_columns].sum(axis=1)
            # Convert vote sum to signal: Buy if vote_sum > 0, Sell if vote_sum < 0, Hold otherwise
            signals_df['signal'] = np.sign(signals_df['vote_sum'])
        
        # 6. Enhanced rule-based signals with on-chain metrics
        self._add_rule_based_signals(signals_df)
        
        # 7. Apply regime-specific risk management
        self._apply_regime_specific_risk_management(signals_df)
        
        # Ensure sufficient signal frequency (at least 3% of rows should have signals)
        signal_ratio = (signals_df['signal'] != 0).mean()
        if signal_ratio < 0.03:
            print(f"Warning: Signal ratio ({signal_ratio:.2%}) is below the 3% threshold.")
            print("Adjusting signal sensitivity...")
            
            # Adjust thresholds to generate more signals
            if 'vote_sum' in signals_df.columns:
                # Use a lower threshold for vote_sum to generate more signals
                signals_df['signal'] = np.where(signals_df['vote_sum'] >= 0.5, 1, 
                                          np.where(signals_df['vote_sum'] <= -0.5, -1, 0))
        
        # Calculate final signal frequency
        final_signal_ratio = (signals_df['signal'] != 0).mean()
        print(f"Final signal ratio: {final_signal_ratio:.2%}")
        
        self.signals = signals_df
        return signals_df
    
    def _generate_cnn_signals(self, signals_df):
        """Generate signals from CNN model"""
        model, features, lookback = self.models['cnn']
        
        # Create sequences for each data point
        cnn_signals = []
        
        # Set batch size and add timeout to prevent hanging
        batch_size = 32  # Process predictions in batches
        print(f"Generating CNN signals for {len(signals_df) - lookback} data points...")
        
        # Process in batches to avoid memory issues
        for i in range(lookback, len(signals_df), batch_size):
            batch_end = min(i + batch_size, len(signals_df))
            batch_windows = []
            
            for j in range(i, batch_end):
                # Get the window data and check for problematic values
                window_data = signals_df[features].iloc[j-lookback:j].values
                # Replace any NaN or inf values with 0
                window_data = np.nan_to_num(window_data, nan=0.0, posinf=0.0, neginf=0.0)
                batch_windows.append(window_data)
            
            if batch_windows:
                batch_windows = np.array(batch_windows)
                
                try:
                    # Add timeout to model.predict
                    with tf.device('/CPU:0'):  # Force CPU execution which is more stable
                        predictions = model.predict(
                            batch_windows, 
                            verbose=0,  # Suppress the progress bar
                            batch_size=min(8, len(batch_windows))  # Small batch size
                        )
                    
                    # Convert predictions to signals
                    batch_signals = [1 if pred > 0.5 else -1 for pred in predictions.flatten()]
                    cnn_signals.extend(batch_signals)
                    
                except Exception as e:
                    print(f"Error in CNN prediction batch {i}-{batch_end}: {str(e)}")
                    # Fill with neutral signals on error
                    cnn_signals.extend([0] * (batch_end - i))
        
        print("CNN signal generation completed")
        
        # Add padding for the first lookback days
        padding = [0] * lookback
        cnn_signals = padding + cnn_signals
        
        # Ensure we have the right number of signals
        if len(cnn_signals) != len(signals_df):
            print(f"Warning: CNN signals length ({len(cnn_signals)}) doesn't match dataframe length ({len(signals_df)})")
            # Truncate or pad as necessary
            if len(cnn_signals) > len(signals_df):
                cnn_signals = cnn_signals[:len(signals_df)]
            else:
                cnn_signals.extend([0] * (len(signals_df) - len(cnn_signals)))
        
        signals_df['cnn_signal'] = cnn_signals
    
    def _generate_random_forest_signals(self, signals_df):
        """Generate signals from Random Forest model"""
        model, features = self.models['random_forest']
        
        # Clean the data before prediction
        X_rf = signals_df[features].copy()
        
        # Handle problematic values
        X_rf = X_rf.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with column medians
        for col in X_rf.columns:
            median_val = X_rf[col].median()
            if np.isnan(median_val):
                # If median is also NaN, use 0 instead
                X_rf[col] = X_rf[col].fillna(0)
            else:
                X_rf[col] = X_rf[col].fillna(median_val)
        
        # Final check for any remaining problematic values
        X_rf = np.nan_to_num(X_rf, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            print("Generating Random Forest predictions...")
            rf_preds = model.predict_proba(X_rf)[:, 1]
            signals_df['rf_signal'] = np.where(rf_preds > 0.6, 1, np.where(rf_preds < 0.4, -1, 0))
            print("Random Forest predictions completed")
        except Exception as e:
            print(f"Error in Random Forest prediction: {str(e)}")
            print("Using default neutral signals for RF model")
            signals_df['rf_signal'] = 0  # Neutral signal as fallback
    
    def _generate_gradient_boosting_signals(self, signals_df):
        """Generate signals from Gradient Boosting model"""
        model, features = self.models['gradient_boosting']
        
        # Clean the data before prediction
        X_gb = signals_df[features].copy()
        
        # Handle problematic values
        X_gb = X_gb.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with column medians
        for col in X_gb.columns:
            median_val = X_gb[col].median()
            if np.isnan(median_val):
                X_gb[col] = X_gb[col].fillna(0)
            else:
                X_gb[col] = X_gb[col].fillna(median_val)
        
        # Final check for any remaining problematic values
        X_gb = np.nan_to_num(X_gb, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            print("Generating Gradient Boosting predictions...")
            gb_preds = model.predict_proba(X_gb)[:, 1]
            signals_df['gb_signal'] = np.where(gb_preds > 0.6, 1, np.where(gb_preds < 0.4, -1, 0))
            print("Gradient Boosting predictions completed")
        except Exception as e:
            print(f"Error in Gradient Boosting prediction: {str(e)}")
            print("Using default neutral signals for GB model")
            signals_df['gb_signal'] = 0  # Neutral signal as fallback
    
    def _generate_regression_ensemble_signals(self, signals_df):
        """Generate signals from Regression Ensemble with volatility-based position sizing"""
        lr_model, features = self.models['linear_regression']
        xgb_model, _ = self.models['xgboost']
        mlp_model, _ = self.models['mlp']
        weights = self.ensemble_weights
        
        # Clean the data before prediction
        X_reg = signals_df[features].copy()
        
        # Handle problematic values
        X_reg = X_reg.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with column medians
        for col in X_reg.columns:
            median_val = X_reg[col].median()
            if np.isnan(median_val):
                X_reg[col] = X_reg[col].fillna(0)
            else:
                X_reg[col] = X_reg[col].fillna(median_val)
        
        # Final check for any remaining problematic values
        X_reg = np.nan_to_num(X_reg, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            print("Generating Regression Ensemble predictions...")
            # Get predictions from each model
            predictions = np.column_stack([
                lr_model.predict(X_reg), 
                xgb_model.predict(X_reg),
                mlp_model.predict(X_reg)
            ])
            
            # Apply ensemble weights
            ensemble_pred = predictions @ weights
            
            # Calculate prediction confidence (absolute value of prediction)
            prediction_confidence = np.abs(ensemble_pred)
            
            # Create signals with confidence level
            signals_df['ensemble_pred'] = ensemble_pred
            signals_df['prediction_confidence'] = prediction_confidence
            
            # Dynamic position sizing based on volatility
            if 'volatility_30d' in signals_df.columns:
                # Normalize volatility to get position size multiplier
                # Higher volatility = smaller position size
                vol_median = signals_df['volatility_30d'].median()
                vol_scaling = vol_median / signals_df['volatility_30d']
                
                # Cap the scaling factor to avoid extreme positions
                vol_scaling = vol_scaling.clip(0.25, 2.0)
                
                # Combine prediction confidence with volatility scaling
                signals_df['position_size'] = np.where(
                    ensemble_pred > 0.001,
                    vol_scaling * np.minimum(prediction_confidence * 5, 1.0),  # Limit max position
                    np.where(
                        ensemble_pred < -0.001,
                        -vol_scaling * np.minimum(prediction_confidence * 5, 1.0),  # Limit max position
                        0  # Neutral signal
                    )
                )
                
                # Convert to discrete signal for compatibility
                signals_df['reg_ensemble_signal'] = np.sign(signals_df['position_size'])
            else:
                # Fallback if volatility not available
                signals_df['reg_ensemble_signal'] = np.where(ensemble_pred > 0.001, 1, 
                                                           np.where(ensemble_pred < -0.001, -1, 0))
                signals_df['position_size'] = signals_df['reg_ensemble_signal']
            
            print("Regression Ensemble predictions completed")
        except Exception as e:
            print(f"Error in Regression Ensemble prediction: {str(e)}")
            print("Using default neutral signals for Regression Ensemble")
            signals_df['reg_ensemble_signal'] = 0  # Neutral signal as fallback
            signals_df['position_size'] = 0
    
    def _add_rule_based_signals(self, signals_df):
        """Add rule-based technical signals with enhanced on-chain metrics"""
        # RSI oversold/overbought
        if 'rsi_14' in signals_df.columns:
            # Make sure rsi_14 values are valid
            valid_rsi = signals_df['rsi_14'].copy()
            valid_rsi = valid_rsi.replace([np.inf, -np.inf], np.nan)
            valid_rsi = valid_rsi.fillna(50)  # Neutral RSI value
            signals_df['rsi_14'] = valid_rsi  # Replace with clean values
            
            signals_df['rsi_signal'] = 0  # Initialize with neutral signal
            signals_df.loc[signals_df['rsi_14'] < 30, 'rsi_signal'] = 1  # Oversold - Buy
            signals_df.loc[signals_df['rsi_14'] > 70, 'rsi_signal'] = -1  # Overbought - Sell
        
        # Moving average crossovers
        if 'sma_cross_50_200' in signals_df.columns:
            # Make sure sma_cross values are valid
            valid_cross = signals_df['sma_cross_50_200'].copy()
            valid_cross = valid_cross.replace([np.inf, -np.inf], np.nan)
            valid_cross = valid_cross.fillna(0)  # Neutral crossover value
            signals_df['sma_cross_50_200'] = valid_cross  # Replace with clean values
            
            signals_df['ma_signal'] = 0
            # Golden cross (50-day crosses above 200-day) - Buy
            signals_df.loc[(signals_df['sma_cross_50_200'] > 0) & 
                          (signals_df['sma_cross_50_200'].shift(1) <= 0), 'ma_signal'] = 1
            # Death cross (50-day crosses below 200-day) - Sell
            signals_df.loc[(signals_df['sma_cross_50_200'] < 0) & 
                          (signals_df['sma_cross_50_200'].shift(1) >= 0), 'ma_signal'] = -1
        
        # Add on-chain signals if available
        
        # NVT Ratio (high NVT = overvalued)
        if 'nvt_ratio' in signals_df.columns:
            signals_df['nvt_signal'] = 0
            
            # Calculate rolling Z-score of NVT ratio (more adaptive)
            window = 90  # 90-day window for calculating z-score
            rolling_mean = signals_df['nvt_ratio'].rolling(window=window).mean()
            rolling_std = signals_df['nvt_ratio'].rolling(window=window).std()
            
            # Handle the first window days where rolling stats aren't available
            global_mean = signals_df['nvt_ratio'].mean()
            global_std = signals_df['nvt_ratio'].std()
            rolling_mean.fillna(global_mean, inplace=True)
            rolling_std.fillna(global_std, inplace=True)
            
            # Calculate z-score
            nvt_z_score = (signals_df['nvt_ratio'] - rolling_mean) / rolling_std
            signals_df['nvt_z_score'] = nvt_z_score
            
            # Generate signals based on z-score
            signals_df.loc[nvt_z_score < -1.5, 'nvt_signal'] = 1  # Undervalued - Buy
            signals_df.loc[nvt_z_score > 1.5, 'nvt_signal'] = -1  # Overvalued - Sell
        
        # HODL Ratio (Long-term vs Short-term holders)
        if 'hodl_ratio' in signals_df.columns:
            signals_df['hodl_signal'] = 0
            
            # Calculate rolling Z-score of HODL ratio
            window = 90
            rolling_mean = signals_df['hodl_ratio'].rolling(window=window).mean()
            rolling_std = signals_df['hodl_ratio'].rolling(window=window).std()
            
            # Handle the first window days
            global_mean = signals_df['hodl_ratio'].mean()
            global_std = signals_df['hodl_ratio'].std()
            rolling_mean.fillna(global_mean, inplace=True)
            rolling_std.fillna(global_std, inplace=True)
            
            # Calculate z-score
            hodl_z_score = (signals_df['hodl_ratio'] - rolling_mean) / rolling_std
            signals_df['hodl_z_score'] = hodl_z_score
            
            # Generate signals based on z-score
            signals_df.loc[hodl_z_score > 1.0, 'hodl_signal'] = 1  # More long-term holders - Buy
            signals_df.loc[hodl_z_score < -1.0, 'hodl_signal'] = -1  # More short-term holders - Sell
        
        # MVRV Ratio (if available)
        if 'mvrv_ratio' in signals_df.columns:
            signals_df['mvrv_signal'] = 0
            
            # Calculate rolling Z-score of MVRV ratio
            window = 90
            rolling_mean = signals_df['mvrv_ratio'].rolling(window=window).mean()
            rolling_std = signals_df['mvrv_ratio'].rolling(window=window).std()
            
            # Handle the first window days
            global_mean = signals_df['mvrv_ratio'].mean()
            global_std = signals_df['mvrv_ratio'].std()
            rolling_mean.fillna(global_mean, inplace=True)
            rolling_std.fillna(global_std, inplace=True)
            
            # Calculate z-score
            mvrv_z_score = (signals_df['mvrv_ratio'] - rolling_mean) / rolling_std
            signals_df['mvrv_z_score'] = mvrv_z_score
            
            # Generate signals based on z-score
            signals_df.loc[mvrv_z_score < -1.0, 'mvrv_signal'] = 1  # Undervalued - Buy
            signals_df.loc[mvrv_z_score > 1.0, 'mvrv_signal'] = -1  # Overvalued - Sell
        
        # Market Regime-based signals
        if 'market_regime' in signals_df.columns:
            # Calculate regime-specific statistics
            regime_stats = {}
            for regime in signals_df['market_regime'].unique():
                regime_data = signals_df[signals_df['market_regime'] == regime]
                if 'returns' in regime_data.columns:
                    regime_stats[regime] = {
                        'mean_return': regime_data['returns'].mean(),
                        'volatility': regime_data['returns'].std(),
                        'sharpe': regime_data['returns'].mean() / regime_data['returns'].std() * np.sqrt(252) 
                        if regime_data['returns'].std() > 0 else 0
                    }
            
            # Generate regime-based signals
            signals_df['regime_signal'] = 0
            
            # Identify positive and negative regimes based on Sharpe ratio
            positive_regimes = [regime for regime, stats in regime_stats.items() 
                              if stats['sharpe'] > 0.5]  # Positive Sharpe regimes
            negative_regimes = [regime for regime, stats in regime_stats.items() 
                              if stats['sharpe'] < -0.5]  # Negative Sharpe regimes
            
            # Set signals based on current regime
            for regime in positive_regimes:
                signals_df.loc[signals_df['market_regime'] == regime, 'regime_signal'] = 1
            
            for regime in negative_regimes:
                signals_df.loc[signals_df['market_regime'] == regime, 'regime_signal'] = -1
        
        # Combine with ensemble signals for final decision
        rule_based_cols = ['rsi_signal', 'ma_signal', 'nvt_signal', 'hodl_signal', 
                          'mvrv_signal', 'regime_signal']
        rule_based_cols = [col for col in rule_based_cols if col in signals_df.columns]
        
        if rule_based_cols:
            # Calculate a weighted sum of rule-based signals
            signals_df['rule_based_signal'] = 0
            
            # Weights for different signal types
            weights = {
                'rsi_signal': 0.5,      # Technical
                'ma_signal': 0.5,       # Technical
                'nvt_signal': 1.0,      # On-chain (higher weight)
                'hodl_signal': 1.0,     # On-chain (higher weight)
                'mvrv_signal': 1.0,     # On-chain (higher weight)
                'regime_signal': 1.5    # Regime-based (highest weight)
            }
            
            # Apply weights to each signal type
            for col in rule_based_cols:
                if col in weights:
                    signals_df['rule_based_signal'] += signals_df[col] * weights[col]
                else:
                    signals_df['rule_based_signal'] += signals_df[col] * 0.5  # Default weight
            
            # Combine with existing model signals
            signals_df['signal'] = signals_df['signal'] + signals_df['rule_based_signal']
            signals_df['signal'] = np.sign(signals_df['signal'])  # Convert back to -1, 0, 1
    
    def _apply_regime_specific_risk_management(self, signals_df):
        """Apply different risk management parameters based on market regime"""
        if 'market_regime' not in signals_df.columns:
            return
        
        # Calculate regime-specific statistics
        regime_stats = {}
        for regime in signals_df['market_regime'].unique():
            regime_data = signals_df[signals_df['market_regime'] == regime]
            if 'returns' in regime_data.columns:
                regime_stats[regime] = {
                    'mean_return': regime_data['returns'].mean(),
                    'volatility': regime_data['returns'].std(),
                    'sharpe': regime_data['returns'].mean() / regime_data['returns'].std() * np.sqrt(252) 
                    if regime_data['returns'].std() > 0 else 0,
                    'count': len(regime_data)
                }
        
        # Classify regimes by risk level
        high_risk_regimes = []
        medium_risk_regimes = []
        low_risk_regimes = []
        
        for regime, stats in regime_stats.items():
            # Skip regimes with too few data points
            if stats['count'] < 10:
                continue
                
            if stats['volatility'] > 0.03:  # High daily volatility
                high_risk_regimes.append(regime)
            elif stats['volatility'] < 0.01:  # Low daily volatility
                low_risk_regimes.append(regime)
            else:
                medium_risk_regimes.append(regime)
        
        # Add risk classification to signal dataframe
        signals_df['regime_risk'] = 'medium'  # Default
        
        for regime in high_risk_regimes:
            signals_df.loc[signals_df['market_regime'] == regime, 'regime_risk'] = 'high'
        
        for regime in low_risk_regimes:
            signals_df.loc[signals_df['market_regime'] == regime, 'regime_risk'] = 'low'
        
        # Apply different position sizing based on regime risk
        if 'position_size' in signals_df.columns:
            # Adjust position sizing based on regime risk
            signals_df.loc[signals_df['regime_risk'] == 'high', 'position_size'] *= 0.5  # Reduce position in high risk
            signals_df.loc[signals_df['regime_risk'] == 'low', 'position_size'] *= 1.5  # Increase position in low risk
            
            # Cap position size at 1.0 (100%)
            signals_df['position_size'] = signals_df['position_size'].clip(-1.0, 1.0)
        else:
            # If no position_size column, create one based on signal and regime risk
            signals_df['position_size'] = signals_df['signal']
            signals_df.loc[signals_df['regime_risk'] == 'high', 'position_size'] *= 0.5
            signals_df.loc[signals_df['regime_risk'] == 'low', 'position_size'] *= 1.0  # No change for low risk
    
    def backtest_strategy(self, use_backtrader=False, initial_cash=100000):
        """
        Backtest the trading strategy and calculate performance metrics.
        
        Args:
            use_backtrader (bool): Whether to use backtrader for backtesting
            initial_cash (float): Initial capital for backtesting
            
        Returns:
            performance (dict): Strategy performance metrics
        """
        if self.signals is None:
            print("Error: Generate trading signals first")
            return
        
        if use_backtrader and BACKTRADER_AVAILABLE:
            return self._backtest_with_backtrader(initial_cash)
        else:
            return self._backtest_internal(initial_cash)
    
    def _backtest_internal(self, initial_cash=100000):
        """Internal backtesting implementation with stop-loss and take-profit"""
        print("Backtesting trading strategy using internal implementation...")

        signals = self.signals.copy()

        # Add required columns before using them
        signals['position'] = 0  # Add position column with default value 0
        signals['stop_loss'] = False
        signals['take_profit'] = False
        signals['active_position'] = False
        signals['entry_price'] = np.nan

        # Parameters
        stop_loss_pct = 0.05  # 5% stop loss
        take_profit_pct = 0.10  # 10% take profit

        # Get price column
        price_col = None
        for col in ['close', 'price', 'market_price_usd_bc']:
            if col in signals.columns:
                price_col = col
                break
            
        if price_col is None:
            print("Error: No price column found for stop-loss/take-profit")
            return self._backtest_internal_original(initial_cash)

        # Calculate position with stop-loss and take-profit
        position = 0
        entry_price = 0

        for i in range(1, len(signals)):
            # Get current price and signal
            current_price = signals[price_col].iloc[i]

            # If we have an active position, check stop-loss and take-profit
            if position != 0:
                # Calculate return since entry
                if entry_price > 0:
                    current_return = (current_price / entry_price - 1) * position

                    # Check stop-loss
                    if current_return < -stop_loss_pct:
                        signals.iloc[i, signals.columns.get_loc('stop_loss')] = True
                        position = 0
                        signals.iloc[i, signals.columns.get_loc('active_position')] = False
                        continue
                    
                    # Check take-profit
                    if current_return > take_profit_pct:
                        signals.iloc[i, signals.columns.get_loc('take_profit')] = True
                        position = 0
                        signals.iloc[i, signals.columns.get_loc('active_position')] = False
                        continue
                    
            # If no active position or no stop/take triggered, process the regular signal
            if 'position_size' in signals.columns:
                new_position = signals['position_size'].iloc[i]
            else:
                new_position = signals['signal'].iloc[i]

            # Only update position if it changed
            if new_position != position:
                position = new_position
                if position != 0:
                    entry_price = current_price
                    signals.iloc[i, signals.columns.get_loc('entry_price')] = entry_price
                    signals.iloc[i, signals.columns.get_loc('active_position')] = True

            # Save the current position
            signals.iloc[i, signals.columns.get_loc('position')] = position

        # Ensure 'returns' column exists
        if 'returns' not in signals.columns:
            print("Warning: 'returns' column not found. Calculating from price data...")
            price_candidates = [col for col in signals.columns if 'price' in col.lower() or 'close' in col.lower()]
            if price_candidates:
                price_col = price_candidates[0]
                signals['returns'] = signals[price_col].pct_change()
            else:
                print("Error: Price data not available. Cannot calculate returns.")
                return

        # Clean and validate returns to prevent errors
        signals['returns'] = signals['returns'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Calculate returns based on positions
        signals['strategy_returns'] = signals['position'].shift(1) * signals['returns']
        signals['strategy_returns'] = signals['strategy_returns'].fillna(0)  # First day has no position

        # Rest of the method remains the same
        try:
            # Calculate cumulative returns for buy-and-hold and strategy
            signals['cum_market_returns'] = (1 + signals['returns']).cumprod() - 1
            signals['cum_strategy_returns'] = (1 + signals['strategy_returns']).cumprod() - 1

            print("Cumulative returns calculated successfully")
        except Exception as e:
            print(f"Error calculating cumulative returns: {str(e)}")
            # Create dummy cumulative return columns 
            signals['cum_market_returns'] = signals['returns'].cumsum()
            signals['cum_strategy_returns'] = signals['strategy_returns'].cumsum()
            print("Using simplified cumulative returns calculation instead")

        try:
            # Calculate performance metrics
            # 1. Sharpe Ratio (annualized)
            risk_free_rate = 0.02 / 252  # Assuming 2% annual risk-free rate
            sharpe_ratio = np.sqrt(252) * (signals['strategy_returns'].mean() - risk_free_rate) / signals['strategy_returns'].std()

            # 2. Maximum Drawdown
            cum_returns = (1 + signals['strategy_returns']).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max - 1)
            max_drawdown = drawdown.min()

            # 3. Win Rate
            winning_trades = (signals['strategy_returns'] > 0).sum()
            total_trades = (signals['strategy_returns'] != 0).sum()
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            # 4. Annualized Return
            days = (signals.index[-1] - signals.index[0]).days
            if days > 0:
                annual_return = (1 + signals['cum_strategy_returns'].iloc[-1]) ** (365 / days) - 1
            else:
                annual_return = 0

            # 5. Calmar Ratio (annualized return / max drawdown)
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')

            # Store performance metrics
            self.performance = {
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown,
                'Win Rate': win_rate,
                'Annual Return': annual_return,
                'Calmar Ratio': calmar_ratio,
                'Total Trades': total_trades,
                'Signal Frequency': (signals['signal'] != 0).mean()
            }

            # Calculate buy-and-hold performance for comparison
            bh_annual_return = (1 + signals['cum_market_returns'].iloc[-1]) ** (365 / days) - 1 if days > 0 else 0

            print("\nStrategy Performance:")
            for metric, value in self.performance.items():
                print(f"{metric}: {value:.4f}")

            print(f"\nBuy-and-Hold Annual Return: {bh_annual_return:.4f}")
            print(f"Strategy Outperformance: {annual_return - bh_annual_return:.4f}")

        except Exception as e:
            print(f"Error calculating performance metrics: {str(e)}")
            self.performance = {
                'Sharpe Ratio': 0.0,
                'Max Drawdown': 0.0,
                'Win Rate': 0.0,
                'Annual Return': 0.0,
                'Calmar Ratio': 0.0,
                'Total Trades': 0,
                'Signal Frequency': 0.0
            }
            print("Using default performance metrics due to calculation error")

        # Save the signals with calculated metrics
        self.signals = signals

        return self.performance

    def _backtest_internal_original(self, initial_cash=100000):
        """Original internal backtesting implementation (fallback)"""
        print("Backtesting trading strategy using original internal implementation...")
        
        signals = self.signals.copy()
        
        # Calculate position (1: Long, 0: Cash, -1: Short)
        signals['position'] = signals['signal']
        
        # Check if 'returns' column exists
        if 'returns' not in signals.columns:
            print("Warning: 'returns' column not found. Calculating from price data...")
            price_candidates = [col for col in signals.columns if 'price' in col.lower() or 'close' in col.lower()]
            if price_candidates:
                price_col = price_candidates[0]
                signals['returns'] = signals[price_col].pct_change()
            else:
                print("Error: Price data not available. Cannot calculate returns.")
                return
        
        # Clean and validate returns to prevent errors
        signals['returns'] = signals['returns'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Calculate returns based on positions
        signals['strategy_returns'] = signals['position'].shift(1) * signals['returns']
        signals['strategy_returns'] = signals['strategy_returns'].fillna(0)  # First day has no position
        
        try:
            # Calculate cumulative returns for buy-and-hold and strategy
            signals['cum_market_returns'] = (1 + signals['returns']).cumprod() - 1
            signals['cum_strategy_returns'] = (1 + signals['strategy_returns']).cumprod() - 1
            
            print("Cumulative returns calculated successfully")
        except Exception as e:
            print(f"Error calculating cumulative returns: {str(e)}")
            # Create dummy cumulative return columns 
            signals['cum_market_returns'] = signals['returns'].cumsum()
            signals['cum_strategy_returns'] = signals['strategy_returns'].cumsum()
            print("Using simplified cumulative returns calculation instead")
        
        try:
            # Calculate performance metrics
            # 1. Sharpe Ratio (annualized)
            risk_free_rate = 0.02 / 252  # Assuming 2% annual risk-free rate
            sharpe_ratio = np.sqrt(252) * (signals['strategy_returns'].mean() - risk_free_rate) / signals['strategy_returns'].std()
            
            # 2. Maximum Drawdown
            cum_returns = (1 + signals['strategy_returns']).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max - 1)
            max_drawdown = drawdown.min()
            
            # 3. Win Rate
            winning_trades = (signals['strategy_returns'] > 0).sum()
            total_trades = (signals['strategy_returns'] != 0).sum()
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 4. Annualized Return
            days = (signals.index[-1] - signals.index[0]).days
            if days > 0:
                annual_return = (1 + signals['cum_strategy_returns'].iloc[-1]) ** (365 / days) - 1
            else:
                annual_return = 0
            
            # 5. Calmar Ratio (annualized return / max drawdown)
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
            
            # Store performance metrics
            self.performance = {
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown,
                'Win Rate': win_rate,
                'Annual Return': annual_return,
                'Calmar Ratio': calmar_ratio,
                'Total Trades': total_trades,
                'Signal Frequency': (signals['signal'] != 0).mean()
            }
            
            # Calculate buy-and-hold performance for comparison
            bh_annual_return = (1 + signals['cum_market_returns'].iloc[-1]) ** (365 / days) - 1 if days > 0 else 0
            
            print("\nStrategy Performance:")
            for metric, value in self.performance.items():
                print(f"{metric}: {value:.4f}")
            
            print(f"\nBuy-and-Hold Annual Return: {bh_annual_return:.4f}")
            print(f"Strategy Outperformance: {annual_return - bh_annual_return:.4f}")
            
        except Exception as e:
            print(f"Error calculating performance metrics: {str(e)}")
            self.performance = {
                'Sharpe Ratio': 0.0,
                'Max Drawdown': 0.0,
                'Win Rate': 0.0,
                'Annual Return': 0.0,
                'Calmar Ratio': 0.0,
                'Total Trades': 0,
                'Signal Frequency': 0.0
            }
            print("Using default performance metrics due to calculation error")
        
        # Save the signals with calculated metrics
        self.signals = signals
        
        return self.performance
    
    def _backtest_with_backtrader(self, initial_cash=100000):
        """
        Backtest using backtrader library
        
        Args:
            initial_cash (float): Initial capital for backtesting
            
        Returns:
            results_summary (dict): Backtest results summary
        """
        if not BACKTRADER_AVAILABLE:
            print("Error: backtrader not available. Please install backtrader first.")
            return self._backtest_internal(initial_cash)
        
        print("Backtesting trading strategy using backtrader...")
        
        # Get test data
        test_data = self.signals.copy()
        
        # Make sure we have the required columns for backtrader
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in test_data.columns:
                # Try to find alternative columns
                if col == 'close':
                    candidates = [c for c in test_data.columns if 'close' in c.lower() or 'price' in c.lower()]
                    if candidates:
                        test_data['close'] = test_data[candidates[0]]
                    else:
                        test_data['close'] = 100  # Default value
                elif col == 'open':
                    test_data['open'] = test_data['close'] * 0.99  # Default value
                elif col == 'high':
                    test_data['high'] = test_data['close'] * 1.01  # Default value
                elif col == 'low':
                    test_data['low'] = test_data['close'] * 0.99  # Default value
                elif col == 'volume':
                    test_data['volume'] = 1000000  # Default value
        
        # Create a backtrader cerebro engine
        cerebro = bt.Cerebro()
        
        # Define the MLSignalStrategy
        class MLSignalStrategy(bt.Strategy):
            params = (
                ('target_percent', 0.98),  # Target position size
                ('stop_loss_pct', 0.05),   # 5% stop loss
                ('take_profit_pct', 0.10),  # 10% take profit
            )
            
            def __init__(self):
                # Get signals from the parent class
                self.signals = test_data['signal']
                
                # Use position_size if available
                if 'position_size' in test_data.columns:
                    self.position_sizes = test_data['position_size']
                    self.use_position_sizes = True
                else:
                    self.use_position_sizes = False
                
                # Initialize variables
                self.order = None
                self.position_value = 0
                self.value_history_dates = []
                self.value_history_values = []
                self.entry_price = None
            
            def next(self):
                # Skip if an order is pending
                if self.order:
                    return
                
                current_date = self.data.datetime.date(0)
                
                # Find the signal for the current date
                try:
                    # Get the signal for the current date
                    current_signal = self.signals[self.data.datetime.datetime()]
                    if self.use_position_sizes:
                        position_size = self.position_sizes[self.data.datetime.datetime()]
                    else:
                        position_size = current_signal
                except:
                    try:
                        # Try with date instead of datetime
                        current_signal = self.signals[current_date]
                        if self.use_position_sizes:
                            position_size = self.position_sizes[current_date]
                        else:
                            position_size = current_signal
                    except:
                        current_signal = 0  # Default to hold
                        position_size = 0
                
                # Get current position
                current_position = self.getposition().size
                
                # Check stop-loss and take-profit for existing positions
                if current_position != 0 and self.entry_price is not None:
                    # Calculate current return
                    current_return = (self.data.close[0] / self.entry_price - 1)
                    if current_position > 0:
                        # Long position
                        if current_return < -self.p.stop_loss_pct:
                            # Stop loss hit for long position
                            print(f"{current_date} => STOP LOSS triggered at {current_return:.2%}")
                            self.order = self.order_target_percent(target=0.0)
                            self.entry_price = None
                            return
                        elif current_return > self.p.take_profit_pct:
                            # Take profit hit for long position
                            print(f"{current_date} => TAKE PROFIT triggered at {current_return:.2%}")
                            self.order = self.order_target_percent(target=0.0)
                            self.entry_price = None
                            return
                    elif current_position < 0:
                        # Short position (if enabled)
                        if current_return > self.p.stop_loss_pct:
                            # Stop loss hit for short position
                            print(f"{current_date} => STOP LOSS triggered at {current_return:.2%}")
                            self.order = self.order_target_percent(target=0.0)
                            self.entry_price = None
                            return
                        elif current_return < -self.p.take_profit_pct:
                            # Take profit hit for short position
                            print(f"{current_date} => TAKE PROFIT triggered at {current_return:.2%}")
                            self.order = self.order_target_percent(target=0.0)
                            self.entry_price = None
                            return
                
                # Execute trades based on signals and position sizing
                if self.use_position_sizes:
                    # Calculate target position based on position size
                    target_pct = abs(position_size) * self.p.target_percent
                    
                    # Check if position size direction changed
                    if (position_size > 0 and current_position <= 0) or \
                       (position_size < 0 and current_position >= 0) or \
                       (position_size == 0 and current_position != 0):
                        # Close current position and open new one
                        if position_size > 0:
                            self.order = self.order_target_percent(target=target_pct)
                            print(f"{current_date} => BUY signal (position size: {target_pct:.2%})")
                            self.entry_price = self.data.close[0]
                        elif position_size < 0:
                            # Short position if enabled
                            self.order = self.order_target_percent(target=-target_pct)
                            print(f"{current_date} => SELL signal (position size: {-target_pct:.2%})")
                            self.entry_price = self.data.close[0]
                        else:
                            self.order = self.order_target_percent(target=0.0)
                            print(f"{current_date} => CLOSE position")
                            self.entry_price = None
                else:
                    # Standard signal (1=buy, -1=sell, 0=hold)
                    if current_signal > 0 and current_position <= 0:
                        # Buy signal
                        self.order = self.order_target_percent(target=self.p.target_percent)
                        print(f"{current_date} => BUY signal")
                        self.entry_price = self.data.close[0]
                    elif current_signal < 0 and current_position > 0:
                        # Sell signal
                        self.order = self.order_target_percent(target=0.0)
                        print(f"{current_date} => SELL signal")
                        self.entry_price = None
                
                # Record portfolio value
                self.value_history_dates.append(current_date)
                self.value_history_values.append(self.broker.getvalue())
        
        # Add data feed
        data_feed = bt.feeds.PandasData(
            dataname=test_data,
            datetime=None,  # Use index as datetime
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=None
        )
        cerebro.adddata(data_feed)
        
        # Add ML signal strategy
        cerebro.addstrategy(MLSignalStrategy)
        
        # Add buy-and-hold strategy for comparison
        class BuyAndHoldStrategy(bt.Strategy):
            def __init__(self):
                self.value_history_dates = []
                self.value_history_values = []
            
            def next(self):
                # Buy on first day
                if len(self) == 1:
                    self.order_target_percent(target=0.98)
                    print(f"{self.data.datetime.date(0)} => BUY and HOLD")
                
                # Record portfolio value
                self.value_history_dates.append(self.data.datetime.date(0))
                self.value_history_values.append(self.broker.getvalue())
        
        # Create a separate cerebro for buy-and-hold strategy
        cerebro_bh = bt.Cerebro()
        cerebro_bh.adddata(data_feed)
        cerebro_bh.addstrategy(BuyAndHoldStrategy)
        
        # Set initial cash and commission
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.001)  # 0.1%
        
        cerebro_bh.broker.setcash(initial_cash)
        cerebro_bh.broker.setcommission(commission=0.001)  # 0.1%
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # Run backtest
        initial_value = cerebro.broker.getvalue()
        print(f"Initial portfolio value: ${initial_value:.2f}")
        
        ml_results = cerebro.run()
        bh_results = cerebro_bh.run()
        
        final_value = cerebro.broker.getvalue()
        print(f"Final portfolio value: ${final_value:.2f}")
        print(f"Return: {(final_value/initial_value - 1) * 100:.2f}%")
        
        # Get strategy instances
        ml_strategy = ml_results[0]
        bh_strategy = bh_results[0]
        
        # Save strategy objects for plotting
        self.bt_strategies = {
            'ml_strategy': ml_strategy,
            'bh_strategy': bh_strategy,
            'cerebro': cerebro,
            'cerebro_bh': cerebro_bh
        }
        
        # Calculate performance metrics for ML strategy
        ml_metrics = self._calculate_performance_metrics(ml_strategy)
        
        # Calculate performance metrics for buy-and-hold strategy
        bh_metrics = self._calculate_performance_metrics(bh_strategy)
        
        # Store performance in the class
        self.performance = ml_metrics
        
        # Create results summary
        results_summary = {
            'ML Strategy': ml_metrics,
            'Buy and Hold': bh_metrics
        }
        
        # Display performance metrics comparison
        print("\n=== Performance Metrics ===")
        metrics_table = pd.DataFrame({
            'ML Strategy': [
                f"{ml_metrics['total_return_pct']:.2f}%",
                f"{ml_metrics['annualized_return_pct']:.2f}%",
                f"{ml_metrics['sharpe_ratio']:.2f}",
                f"{ml_metrics['max_drawdown_pct']:.2f}%",
                f"{ml_metrics['calmar_ratio']:.2f}"
            ],
            'Buy and Hold': [
                f"{bh_metrics['total_return_pct']:.2f}%",
                f"{bh_metrics['annualized_return_pct']:.2f}%",
                f"{bh_metrics['sharpe_ratio']:.2f}",
                f"{bh_metrics['max_drawdown_pct']:.2f}%",
                f"{bh_metrics['calmar_ratio']:.2f}"
            ]
        }, index=['Total Return', 'Annualized Return', 'Sharpe Ratio', 'Maximum Drawdown', 'Calmar Ratio'])
        
        print(metrics_table)
        
        return results_summary
    
    def _calculate_performance_metrics(self, strategy):
        """
        Calculate key performance metrics for a strategy
        
        Args:
            strategy: Backtrader strategy instance
            
        Returns:
            metrics (dict): Dictionary of performance metrics
        """
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
            'total_return_pct': total_return_pct,
            'annualized_return_pct': ann_return_pct,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown_pct,
            'calmar_ratio': calmar_ratio
        }
    
    def visualize_results(self):
        """
        Visualize strategy results and performance.
        """
        if self.signals is None:
            print("Error: Generate trading signals first")
            return
        
        signals = self.signals.copy()
        
        # Check if required columns exist
        required_columns = ['cum_market_returns', 'cum_strategy_returns', 'strategy_returns']
        missing_columns = [col for col in required_columns if col not in signals.columns]
        
        if missing_columns:
            print(f"Missing required columns for visualization: {missing_columns}")
            print("Running backtest_strategy to generate necessary data...")
            # Try to run backtest to generate the missing columns
            self._backtest_internal()
            if self.signals is None:
                print("Error: Failed to generate signals. Cannot visualize results.")
                return
            signals = self.signals.copy()
        
        # Verify required columns again
        missing_columns = [col for col in required_columns if col not in signals.columns]
        if missing_columns:
            print(f"Still missing required columns after backtest: {missing_columns}")
            print("Cannot generate all visualizations. Showing partial results...")
        
        # Create figure for all plots
        plt.figure(figsize=(15, 20))
        
        # 1. Cumulative returns plot
        if 'cum_market_returns' in signals.columns and 'cum_strategy_returns' in signals.columns:
            plt.subplot(4, 1, 1)
            plt.plot(signals.index, signals['cum_market_returns'] * 100, label='Buy and Hold', color='blue', alpha=0.7)
            plt.plot(signals.index, signals['cum_strategy_returns'] * 100, label='Trading Strategy', color='green')
            plt.title('Cumulative Returns: Strategy vs Buy-and-Hold')
            plt.xlabel('Date')
            plt.ylabel('Returns (%)')
            plt.legend()
            plt.grid(True)
            
            print("Cumulative returns plot generated")
        
        # 2. Drawdown plot
        if 'strategy_returns' in signals.columns:
            try:
                plt.subplot(4, 1, 2)
                cum_returns = (1 + signals['strategy_returns']).cumprod()
                running_max = cum_returns.cummax()
                drawdown = (cum_returns / running_max - 1) * 100
                
                plt.plot(signals.index, drawdown)
                plt.title('Strategy Drawdown')
                plt.xlabel('Date')
                plt.ylabel('Drawdown (%)')
                plt.grid(True)
                
                print("Drawdown plot generated")
            except Exception as e:
                print(f"Error generating drawdown plot: {str(e)}")
        
        # 3. Signal distribution
        if 'signal' in signals.columns:
            try:
                plt.subplot(4, 1, 3)
                signals['signal'].value_counts().plot(kind='bar')
                plt.title('Distribution of Trading Signals')
                plt.xlabel('Signal (-1: Sell, 0: Hold, 1: Buy)')
                plt.ylabel('Frequency')
                plt.grid(True, axis='y')
                
                print("Signal distribution plot generated")
            except Exception as e:
                print(f"Error generating signal distribution plot: {str(e)}")
        
        # 4. Feature importance plot
        if 'random_forest' in self.models:
            try:
                plt.subplot(4, 1, 4)
                model, features = self.models['random_forest']
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
                
                print("Feature importance plot generated")
            except Exception as e:
                print(f"Error generating feature importance plot: {str(e)}")
        
        plt.tight_layout()
        plt.show()
        
        # 5. Market regimes
        if 'market_regime' in signals.columns:
            plt.figure(figsize=(15, 10))
            
            # Regime returns
            try:
                plt.subplot(2, 1, 1)
                regime_returns = signals.groupby('market_regime')['returns'].mean() * 252  # Annualized
                
                plt.bar(regime_returns.index, regime_returns * 100)
                plt.title('Average Annual Returns by Market Regime')
                plt.xlabel('Market Regime')
                plt.ylabel('Annualized Returns (%)')
                plt.grid(True, axis='y')
                
                print("Market regime returns plot generated")
            except Exception as e:
                print(f"Error generating market regime returns plot: {str(e)}")
            
            # Regime distribution over time
            try:
                plt.subplot(2, 1, 2)
                # Create a colormap for regimes
                n_regimes = signals['market_regime'].nunique()
                colors = plt.cm.viridis(np.linspace(0, 1, n_regimes))
                
                # Plot regime as line with color
                for regime in range(n_regimes):
                    regime_indices = signals.index[signals['market_regime'] == regime]
                    if len(regime_indices) > 0:
                        # Get a price column for plotting
                        if 'close' in signals.columns:
                            price = signals.loc[regime_indices, 'close']
                        else:
                            # Try to find any price column
                            price_candidates = [col for col in signals.columns if 'price' in col.lower() or 'close' in col.lower()]
                            if price_candidates:
                                price = signals.loc[regime_indices, price_candidates[0]]
                            else:
                                # Create a dummy price series
                                price = pd.Series(np.arange(len(regime_indices)), index=regime_indices)
                        
                        plt.plot(regime_indices, price, color=colors[regime], label=f'Regime {regime}', linewidth=2)
                
                plt.title('Market Regimes Over Time')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(True)
                
                print("Market regime distribution plot generated")
            except Exception as e:
                print(f"Error generating market regime distribution plot: {str(e)}")
            
            plt.tight_layout()
            plt.show()
        
        # 6. If backtrader was used, show portfolio comparison
        if hasattr(self, 'bt_strategies') and self.bt_strategies:
            try:
                ml_strategy = self.bt_strategies['ml_strategy']
                bh_strategy = self.bt_strategies['bh_strategy']
                
                plt.figure(figsize=(12, 6))
                plt.plot(ml_strategy.value_history_dates, ml_strategy.value_history_values, label='ML Ensemble Strategy')
                plt.plot(bh_strategy.value_history_dates, bh_strategy.value_history_values, label='Buy and Hold Strategy')
                plt.xlabel('Date')
                plt.ylabel('Portfolio Value ($)')
                plt.title('Strategy Performance Comparison')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                
                print("Backtrader portfolio comparison plot generated")
                
                # Also show the backtrader plots
                try:
                    self.bt_strategies['cerebro'].plot(style='candlestick')
                    print("Backtrader ML strategy plot generated")
                except Exception as e:
                    print(f"Error generating backtrader ML strategy plot: {str(e)}")
            except Exception as e:
                print(f"Error generating backtrader portfolio comparison: {str(e)}")
        
        print("Visualization completed")


class EnhancedOnChainAnalysis:
    """
    Enhanced on-chain analysis module with improved validation and metric reporting.
    Addresses issues with zero-value metrics and improves data processing.
    """
    
    def __init__(self, parent_strategy):
        """Initialize the on-chain analysis module with validation tracking."""
        self.parent = parent_strategy
        self.onchain_features = None
        self.models = {}
        self.onchain_signals = None
        self.onchain_regimes = None
        
        # Track validation metrics explicitly
        self.validation_metrics = {
            'hmm': {},
            'cnn': {},
            'rules': {}
        }
        
        # Data quality tracking
        self.data_quality = {
            'missing_values': 0,
            'inf_values': 0,
            'processed_features': 0
        }
        
        # Model performance comparisons
        self.model_comparisons = None
    
    def extract_onchain_features(self):
        """
        Extract and validate on-chain features with improved data quality checks.
        """
        if self.parent.merged_data is None:
            print("Error: No merged data available. Run merge_data() first.")
            return False
            
        print("Extracting and validating on-chain features...")
        df = self.parent.merged_data.copy()
        
        # Data quality check - log original data shape and nulls
        print(f"Original data shape: {df.shape}")
        total_cells = df.shape[0] * df.shape[1]
        null_cells = df.isnull().sum().sum()
        print(f"Missing values before processing: {null_cells} ({null_cells/total_cells:.2%} of data)")
        
        # Check if we have any on-chain data at all
        expected_onchain_indicators = [
            'transaction_rate', 'hash_rate', 'average_block_size', 
            'miners_revenue', 'total_transaction_fees', 'difficulty',
            'cost_per_transaction', 'market_cap_usd_bc'
        ]
        
        expected_address_indicators = [
            'addresses_with_1000_btc', 'addresses_with_100_btc',
            'addresses_with_10_btc', 'addresses_with_1_btc',
            'addresses_with_0.01_btc'
        ]
        
        expected_hodl_indicators = [
            '24h', '1d_1w', '1w_1m', '1m_3m', '3m_6m', '6m_12m',
            '1y_2y', '2y_3y', '3y_5y', '5y_7y', '7y_10y', '10y'
        ]
        
        # Create normalized column names to handle variations in naming
        def normalize_column_name(col):
            col = col.lower()
            # Handle common variations in naming
            col = col.replace('addresses_with_', '')
            col = col.replace('addresses_balance_', '')
            col = col.replace('address_count_', '')
            col = col.replace('_btc', '')
            col = col.replace('_bitcoin', '')
            return col
        
        # Create mapping of normalized names to actual column names
        column_mapping = {normalize_column_name(col): col for col in df.columns}
        
        # Check what on-chain data we have available with more flexible matching
        avail_onchain = []
        for indicator in expected_onchain_indicators:
            norm_indicator = normalize_column_name(indicator)
            if norm_indicator in column_mapping:
                avail_onchain.append(column_mapping[norm_indicator])
            elif indicator in df.columns:
                avail_onchain.append(indicator)
        
        avail_address = []
        for indicator in expected_address_indicators:
            norm_indicator = normalize_column_name(indicator)
            if norm_indicator in column_mapping:
                avail_address.append(column_mapping[norm_indicator])
            elif indicator in df.columns:
                avail_address.append(indicator)
                
        avail_hodl = []
        for indicator in expected_hodl_indicators:
            norm_indicator = normalize_column_name(indicator)
            if norm_indicator in column_mapping:
                avail_hodl.append(column_mapping[norm_indicator])
            elif indicator in df.columns:
                avail_hodl.append(indicator)
        
        # Log what data was found
        print(f"Found {len(avail_onchain)} on-chain metrics, {len(avail_address)} address metrics, "
              f"and {len(avail_hodl)} HODL metrics")
        
        if not avail_onchain and not avail_address and not avail_hodl:
            print("Warning: No on-chain data found. Will create synthetic features for testing.")
            return self._create_synthetic_features(df)
        
        # Create on-chain features dataframe
        onchain_df = pd.DataFrame(index=df.index)
        
        # 1. Basic on-chain metrics (with more advanced calculations)
        if 'transaction_rate' in df.columns and 'average_block_size' in df.columns:
            # Transaction efficiency (higher means more tx in less block space)
            onchain_df['tx_efficiency'] = df['transaction_rate'] / df['average_block_size']
            
            # Transaction rate momentum features
            for window in [7, 14, 30]:
                # Calculate rate of change in transaction rate
                onchain_df[f'tx_rate_change_{window}d'] = df['transaction_rate'].pct_change(periods=window)
                
                # Volatility of transaction rate
                onchain_df[f'tx_rate_volatility_{window}d'] = df['transaction_rate'].pct_change().rolling(window).std()
            
            # Transaction rate Z-score (compared to its recent history)
            window = min(90, len(df) // 4)  # Adaptive window size
            rolling_mean = df['transaction_rate'].rolling(window=window).mean()
            rolling_std = df['transaction_rate'].rolling(window=window).std()
            onchain_df['tx_rate_zscore'] = (df['transaction_rate'] - rolling_mean) / rolling_std
            
            # Data validation - log transaction features
            print(f"Created transaction features: {onchain_df.columns.tolist()}")
        else:
            print("Warning: Transaction rate or block size data not found")
        
        # 2. Miner related metrics
        if 'hash_rate' in df.columns and 'difficulty' in df.columns:
            # Hash to difficulty ratio (miner efficiency)
            onchain_df['hash_difficulty_ratio'] = df['hash_rate'] / df['difficulty']
            
            # Mining difficulty growth rate 
            for window in [7, 14, 30]:
                growth_col = f'difficulty_growth_{window}d'
                onchain_df[growth_col] = df['difficulty'].pct_change(periods=window)
                
                # Validate data range
                valid_range = onchain_df[growth_col].dropna().between(-0.5, 0.5).mean()
                print(f"{growth_col} has {valid_range:.2%} values in valid range")
            
            # Difficulty adjustment trend (smoothed growth rate)
            onchain_df['diff_adj_trend'] = df['difficulty'].pct_change().rolling(14).mean()
        else:
            print("Warning: Hash rate or difficulty data not found")
        
        # 3. MVRV and NVT metrics
        if 'market_cap_usd_bc' in df.columns:
            # Calculate NVT if transaction metrics are available
            if 'transaction_rate' in df.columns:
                # NVT Ratio: Network Value to Transactions Ratio
                onchain_df['nvt_ratio'] = df['market_cap_usd_bc'] / df['transaction_rate']
                
                # Validate NVT ratio range
                nvt_mean = onchain_df['nvt_ratio'].mean()
                nvt_std = onchain_df['nvt_ratio'].std()
                print(f"NVT Ratio - Mean: {nvt_mean:.2f}, Std: {nvt_std:.2f}")
                
                # NVT Signal (90-day MA of daily NVT Ratio - smoother signal)
                onchain_df['nvt_signal'] = onchain_df['nvt_ratio'].rolling(90).mean()
            
            # Calculate MVRV if realized cap is available
            if 'realised_cap_usd' in df.columns:
                # MVRV Ratio: Market Value to Realized Value
                onchain_df['mvrv_ratio'] = df['market_cap_usd_bc'] / df['realised_cap_usd']
                
                # Validate MVRV ratio range
                mvrv_mean = onchain_df['mvrv_ratio'].mean()
                mvrv_std = onchain_df['mvrv_ratio'].std()
                print(f"MVRV Ratio - Mean: {mvrv_mean:.2f}, Std: {mvrv_std:.2f}")
        else:
            print("Warning: Market cap data not found")
        
        # 4. Advanced Address Metrics
        if len(avail_address) > 2:  # Need at least 3 categories for meaningful metrics
            # Extract addresses into specific classes with data validation
            address_data = {}
            
            # Map address indicators to categories
            for col in avail_address:
                if '1000' in col:
                    address_data['whale'] = df[col]
                elif '100' in col:
                    address_data['large'] = df[col]
                elif '10' in col and not '100' in col:
                    address_data['medium'] = df[col]
                elif '1' in col and not '10' in col:
                    address_data['small'] = df[col]
                elif '0.01' in col:
                    address_data['micro'] = df[col]
            
            # Calculate address metrics if we have enough data
            if len(address_data) >= 3:
                # Validate address data
                for category, series in address_data.items():
                    print(f"{category} addresses - Mean: {series.mean():.2f}, Missing: {series.isnull().mean():.2%}")
                
                # Calculate whale concentration if possible
                if 'whale' in address_data and 'large' in address_data:
                    # Total whales
                    total_whales = address_data['whale'] + address_data.get('large', 0)
                    onchain_df['whale_addresses'] = total_whales
                    
                    # Calculate total addresses with meaningful amounts
                    total_addresses = sum(series for series in address_data.values())
                    
                    # Whale dominance (what percentage of addresses are whales)
                    onchain_df['whale_dominance'] = total_whales / total_addresses
        else:
            print("Warning: Insufficient address data found")
        
        # 5. HODL Wave Metrics with data validation
        if len(avail_hodl) >= 4:  # Need at least 4 timeframes for meaningful HODL metrics
            # Group HODL waves into timeframe categories
            hodl_groups = {
                'short_term': [col for col in avail_hodl if any(x in col.lower() for x in ['24h', '1d', '1w', '1m', '3m'])],
                'mid_term': [col for col in avail_hodl if any(x in col.lower() for x in ['6m', '12m', '1y'])],
                'long_term': [col for col in avail_hodl if any(x in col.lower() for x in ['2y', '3y', '5y', '7y', '10y'])]
            }
            
            # Log what was found in each group
            for term, cols in hodl_groups.items():
                print(f"{term} HODL indicators: {cols}")
            
            # Calculate holder percentages
            for term, cols in hodl_groups.items():
                if cols:
                    onchain_df[f'{term}_holders'] = sum(df[col] for col in cols).fillna(0)
            
            # Calculate holder ratios if we have the necessary data
            if 'short_term_holders' in onchain_df.columns and 'long_term_holders' in onchain_df.columns:
                # Validate holder metrics first
                st_mean = onchain_df['short_term_holders'].mean()
                lt_mean = onchain_df['long_term_holders'].mean()
                print(f"Short-term holders mean: {st_mean:.4f}, Long-term holders mean: {lt_mean:.4f}")
                
                # Long-term to short-term holder ratio (higher means more HODLing)
                # Add small epsilon to avoid division by zero
                onchain_df['hodl_ratio'] = onchain_df['long_term_holders'] / (onchain_df['short_term_holders'] + 1e-10)
                
                # Check and report HODL ratio range
                hodl_mean = onchain_df['hodl_ratio'].mean()
                hodl_median = onchain_df['hodl_ratio'].median()
                print(f"HODL ratio - Mean: {hodl_mean:.2f}, Median: {hodl_median:.2f}")
                
                # If HODL ratio is extremely high or low, it might indicate data issues
                if hodl_mean > 10 or hodl_mean < 0.1:
                    print("Warning: HODL ratio may be miscalculated - check input data")
                    # Apply reasonable bounds
                    onchain_df['hodl_ratio'] = onchain_df['hodl_ratio'].clip(0.1, 10)
                
                # Change in HODL ratio - rate of change with appropriate window size
                window_size = min(30, len(onchain_df) // 10)  # Adaptive window
                onchain_df['hodl_ratio_change'] = onchain_df['hodl_ratio'].pct_change(window_size)
        else:
            print("Warning: Insufficient HODL wave data found")
        
        # 6. Synthetic features - combine metrics for more robust indicators
        if 'mvrv_ratio' in onchain_df.columns and 'hodl_ratio' in onchain_df.columns:
            # Market cycle indicator
            onchain_df['market_cycle'] = onchain_df['mvrv_ratio'] / onchain_df['hodl_ratio']
            
            # Normalize to 0-1 range for cycle phase
            q05 = onchain_df['market_cycle'].quantile(0.05)
            q95 = onchain_df['market_cycle'].quantile(0.95)
            range_mc = q95 - q05
            
            if range_mc > 0:
                onchain_df['cycle_phase'] = ((onchain_df['market_cycle'] - q05) / range_mc).clip(0, 1)
                
                # Check cycle phase distribution (should be somewhat uniform)
                phase_bins = [0, 0.25, 0.5, 0.75, 1.0]
                phase_dist = pd.cut(onchain_df['cycle_phase'], bins=phase_bins).value_counts(normalize=True)
                print(f"Cycle phase distribution:\n{phase_dist}")
        
        # 7. On-chain sentiment indicator with data validation
        sentiment_columns = [
            col for col in onchain_df.columns if any(
                term in col for term in ['ratio', 'growth', 'change', 'dominance', 'zscore']
            )
        ]
        
        if len(sentiment_columns) >= 3:
            print(f"Using {len(sentiment_columns)} metrics for sentiment calculation")
            
            # Normalize sentiment components individually for better data quality
            normalized_components = []
            
            for col in sentiment_columns:
                series = onchain_df[col].copy()
                # Replace infinities and NaNs
                series = series.replace([np.inf, -np.inf], np.nan)
                
                # Check for NaN percentage
                nan_pct = series.isnull().mean()
                if nan_pct > 0.1:  # More than 10% missing
                    print(f"Warning: {col} has {nan_pct:.1%} missing values - skipping for sentiment")
                    continue
                
                # Fill NaNs with 0 (neutral)
                series = series.fillna(0)
                
                # Normalize based on percentiles instead of min/max for robustness
                q10 = series.quantile(0.10)
                q90 = series.quantile(0.90)
                
                if q90 > q10:
                    normalized = (series - q10) / (q90 - q10)
                    # Rescale to -1 to 1 
                    normalized = (normalized * 2 - 1).clip(-1, 1)
                    normalized_components.append(normalized)
            
            if normalized_components:
                # Combine into sentiment score with equal weighting
                onchain_df['onchain_sentiment'] = sum(normalized_components) / len(normalized_components)
                
                # Validate sentiment distribution
                sentiment_mean = onchain_df['onchain_sentiment'].mean()
                sentiment_std = onchain_df['onchain_sentiment'].std()
                print(f"Sentiment indicator - Mean: {sentiment_mean:.2f}, Std: {sentiment_std:.2f}")
                
                # Smooth sentiment for less noise with appropriate window
                smooth_window = min(7, len(onchain_df) // 20)
                onchain_df['onchain_sentiment_ma'] = onchain_df['onchain_sentiment'].rolling(smooth_window).mean().fillna(0)
        
        # Clean up the dataframe and handle missing values
        # Count infinities before replacing
        inf_count = (np.isinf(onchain_df.values) | np.isneginf(onchain_df.values)).sum()
        self.data_quality['inf_values'] = inf_count
        
        # Replace infinities with NaN for proper handling
        onchain_df = onchain_df.replace([np.inf, -np.inf], np.nan)
        
        # Count missing values
        missing_count = onchain_df.isnull().sum().sum()
        self.data_quality['missing_values'] = missing_count
        print(f"Data quality - Infinities: {inf_count}, Missing values: {missing_count}")
        
        # For dataset with many NaN values, use more aggressive filling strategy
        if missing_count / (onchain_df.shape[0] * onchain_df.shape[1]) > 0.1:
            print("High missing value rate detected - using more aggressive filling strategy")
            # Forward fill with larger limit for trend data
            onchain_df = onchain_df.fillna(method='ffill', limit=30)
            # Then backfill for remaining gaps
            onchain_df = onchain_df.fillna(method='bfill', limit=30)
            # Fill any remaining NaNs with column median
            for col in onchain_df.columns:
                onchain_df[col] = onchain_df[col].fillna(onchain_df[col].median())
        else:
            # Standard fill strategy for cleaner data
            onchain_df = onchain_df.fillna(method='ffill', limit=7).fillna(method='bfill', limit=7)
            
            # Fill any remaining NaNs with column median
            for col in onchain_df.columns:
                onchain_df[col] = onchain_df[col].fillna(onchain_df[col].median())
        
        # Store processed feature count
        self.data_quality['processed_features'] = len(onchain_df.columns)
        
        # Final data validation
        if onchain_df.isnull().sum().sum() > 0:
            print("Warning: There are still NaN values after cleaning!")
            print(onchain_df.isnull().sum()[onchain_df.isnull().sum() > 0])
            
            # Last-ditch effort - fill remaining NaNs with 0
            onchain_df = onchain_df.fillna(0)
        
        # Store the on-chain features
        self.onchain_features = onchain_df
        
        # Report final feature set
        print(f"Final on-chain feature set: {len(onchain_df.columns)} features created")
        print(f"Sample of features: {onchain_df.columns[:10].tolist()}...")
        
        # Integrate features into parent strategy if available
        self._integrate_features_with_parent()
        
        return True
    
    def _create_synthetic_features(self, df):
        """
        Create synthetic on-chain features when real data is not available.
        This is mostly for testing the pipeline when on-chain data is missing.
        """
        print("WARNING: Creating synthetic on-chain features for testing purposes only.")
        
        # Create synthetic date range matching original data
        onchain_df = pd.DataFrame(index=df.index)
        
        # Find price column for synthetic feature correlation
        price_col = None
        if 'market_price_usd_bc' in df.columns:
            price_col = 'market_price_usd_bc'
        elif 'close' in df.columns:
            price_col = 'close'
        else:
            price_candidates = [col for col in df.columns if 'price' in col.lower() or 'close' in col.lower()]
            if price_candidates:
                price_col = price_candidates[0]
        
        if price_col is not None:
            base_price = df[price_col]
        else:
            # Create completely synthetic price
            np.random.seed(42)
            base_price = 10000 * (1 + np.random.normal(0, 0.02, len(df.index)).cumsum())
            base_price = pd.Series(base_price, index=df.index)
        
        # Create synthetic on-chain metrics with various correlations to price
        # 1. High positive correlation
        onchain_df['synthetic_nvt_ratio'] = base_price / 1000 * (1 + np.random.normal(0, 0.3, len(df.index)))
        onchain_df['synthetic_mvrv_ratio'] = 1.5 + 2 * (base_price / base_price.median()) + np.random.normal(0, 0.5, len(df.index))
        
        # 2. Moderate correlation with lag
        onchain_df['synthetic_hodl_ratio'] = (base_price.shift(30) / base_price.median() + np.random.normal(0, 0.7, len(df.index))).fillna(1.0)
        
        # 3. Metrics with cyclical patterns
        t = np.linspace(0, 10 * np.pi, len(df.index))
        onchain_df['synthetic_short_term_holders'] = 0.4 + 0.2 * np.sin(t) + 0.1 * np.random.normal(0, 1, len(df.index))
        onchain_df['synthetic_long_term_holders'] = 0.6 - 0.2 * np.sin(t) + 0.1 * np.random.normal(0, 1, len(df.index))
        
        # Ensure positive values
        onchain_df['synthetic_short_term_holders'] = onchain_df['synthetic_short_term_holders'].clip(0.1, 0.9)
        onchain_df['synthetic_long_term_holders'] = onchain_df['synthetic_long_term_holders'].clip(0.1, 0.9)
        
        # 4. Sentiment indicator as composite
        onchain_df['synthetic_sentiment'] = 0.4 * np.sin(t / 2) + 0.3 * (base_price.pct_change(30).fillna(0)) + 0.3 * np.random.normal(0, 0.5, len(df.index))
        onchain_df['synthetic_sentiment'] = onchain_df['synthetic_sentiment'].clip(-1, 1)
        
        # 5. Cycle phase indicator
        cycle = ((t % (2 * np.pi)) / (2 * np.pi)).clip(0, 0.99)  # 0 to 0.99 cycle phase
        onchain_df['synthetic_cycle_phase'] = cycle
        
        # Clean up and store synthetic features
        print(f"Created {len(onchain_df.columns)} synthetic features for testing")
        self.onchain_features = onchain_df
        
        # Integrate features into parent strategy if available
        self._integrate_features_with_parent()
        
        return True
    
    def _integrate_features_with_parent(self):
        """Helper to integrate on-chain features with parent strategy"""
        if self.parent.features is not None and self.onchain_features is not None:
            # Make sure indices match
            common_idx = self.parent.features.index.intersection(self.onchain_features.index)
            
            # Only proceed if we have matching data
            if len(common_idx) > 0:
                for col in self.onchain_features.columns:
                    self.parent.features[f'onchain_{col}'] = self.onchain_features[col]
                
                print(f"Added {len(self.onchain_features.columns)} on-chain features to main feature set")
                print(f"Common data points: {len(common_idx)} of {len(self.parent.features)} ({len(common_idx)/len(self.parent.features):.1%})")
            else:
                print("Warning: On-chain features couldn't be integrated due to mismatched indices")
    
    def train_onchain_regime_hmm(self, n_regimes=5):
        """
        Train a dedicated Hidden Markov Model to identify on-chain specific regimes.
        Includes robust validation and error handling.
        """
        if self.onchain_features is None:
            print("Error: No on-chain features available. Run extract_onchain_features() first.")
            return
            
        print(f"Training HMM for on-chain regime detection with {n_regimes} regimes...")
        
        # Select key on-chain metrics with validation
        all_features = self.onchain_features.columns.tolist()
        
        # Categorize features by type
        feature_categories = {
            'valuations': [col for col in all_features if any(x in col for x in ['ratio', 'cap', 'value', 'nvt', 'mvrv'])],
            'holders': [col for col in all_features if any(x in col for x in ['holders', 'hodl', 'dominance', 'addresses'])],
            'blockchain': [col for col in all_features if any(x in col for x in ['hash', 'difficulty', 'fee', 'transaction'])],
            'sentiment': [col for col in all_features if any(x in col for x in ['sentiment', 'cycle', 'phase'])]
        }
        
        # Report available metrics by category
        for category, features in feature_categories.items():
            print(f"{category}: {len(features)} features available")
            if features:
                print(f"  Sample: {features[:3]}")
        
        # For regime detection, select diverse indicators from each category
        regime_features = []
        # Select up to 3 features from each category
        for category, features in feature_categories.items():
            for feature in features[:3]:  # Take up to 3 from each
                if feature not in regime_features:
                    regime_features.append(feature)
        
        # Ensure we have sufficient features
        if len(regime_features) < 3:
            print("Error: Not enough on-chain features for regime detection")
            print("Adding synthetic features for testing purposes")
            
            # Add synthetic features for testing
            for i in range(3 - len(regime_features)):
                synthetic_feature = f"synthetic_feature_{i}"
                self.onchain_features[synthetic_feature] = np.random.normal(0, 1, len(self.onchain_features))
                regime_features.append(synthetic_feature)
            
        print(f"Using {len(regime_features)} features for on-chain regime detection: {regime_features}")
        
        # Get features for HMM with validation
        df_features = self.onchain_features[regime_features].copy()
        
        # Report data quality
        for col in df_features.columns:
            null_pct = df_features[col].isnull().mean()
            if null_pct > 0:
                print(f"Warning: {col} has {null_pct:.1%} missing values")
        
        # Replace infinities and handle NaNs
        orig_shape = df_features.shape
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values with median of each column
        for col in df_features.columns:
            median_val = df_features[col].median()
            df_features[col] = df_features[col].fillna(median_val)
        
        # Verify no missing values remain
        missing_count = df_features.isnull().sum().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} missing values remain after filling")
            # Last resort fill with 0
            df_features = df_features.fillna(0)
        
        # Clip extreme values to prevent HMM instability
        for col in df_features.columns:
            q1 = df_features[col].quantile(0.01)
            q3 = df_features[col].quantile(0.99)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            orig_extreme = ((df_features[col] < lower_bound) | (df_features[col] > upper_bound)).sum()
            if orig_extreme > 0:
                print(f"Clipping {orig_extreme} extreme values in {col}")
            df_features[col] = df_features[col].clip(lower_bound, upper_bound)
        
        # Scale features
        from sklearn.preprocessing import StandardScaler, RobustScaler
        # Use RobustScaler for better outlier handling
        scaler = RobustScaler()  
        X_scaled = scaler.fit_transform(df_features.values)
        
        print(f"Feature preprocessing complete. Scaled shape: {X_scaled.shape}")
        
        # Track distribution statistics for validation
        feature_means = X_scaled.mean(axis=0)
        feature_stds = X_scaled.std(axis=0)
        print("Scaled feature statistics:")
        for i, feature in enumerate(regime_features):
            print(f"  {feature}: mean={feature_means[i]:.3f}, std={feature_stds[i]:.3f}")
        
        # Try multiple random initializations to find best model
        best_model = None
        best_score = float('-inf')
        best_states = None
        
        for seed in range(3):  # Try 3 random initializations
            try:
                # Train HMM with specified random seed
                from hmmlearn import hmm
                model = hmm.GaussianHMM(
                    n_components=n_regimes, 
                    covariance_type="full",
                    n_iter=250,  # More iterations for better convergence 
                    random_state=seed*42,
                    tol=0.001,  # Tighter tolerance
                    verbose=True  # Enable verbose output for debugging
                )
                
                model.fit(X_scaled)
                
                # Evaluate model using BIC score
                # For HMM, lower BIC is better
                n_params = n_regimes * (n_regimes - 1)  # Transition parameters
                n_params += n_regimes * X_scaled.shape[1]  # Mean parameters
                n_params += n_regimes * X_scaled.shape[1] * (X_scaled.shape[1] + 1) // 2  # Covariance parameters
                
                bic = -2 * model.score(X_scaled) + n_params * np.log(X_scaled.shape[0])
                
                # For comparison, we want higher scores
                score = -bic
                
                print(f"Seed {seed} - model score: {score:.2f}")
                
                # Keep the best model
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_states = model.predict(X_scaled)
                    print(f"New best model found with seed {seed}")
            
            except Exception as e:
                print(f"Error in HMM training with seed {seed}: {str(e)}")
                print("Trying different initialization...")
        
        if best_model is None:
            print("All HMM training attempts failed. Falling back to KMeans for regime detection.")
            return self._fallback_regime_detection(df_features, n_regimes)
        
        print(f"Best HMM model score: {best_score:.2f}")
        
        # Predict regimes with the best model
        hidden_states = best_states
        
        # Store validation metrics
        self.validation_metrics['hmm']['model_score'] = best_score
        
        # Create DataFrame with regime classifications
        self.onchain_regimes = pd.DataFrame({
            'onchain_regime': hidden_states
        }, index=self.onchain_features.index)
        
        # Validate regime distribution
        regime_counts = np.bincount(hidden_states)
        for regime in range(n_regimes):
            count = regime_counts[regime] if regime < len(regime_counts) else 0
            print(f"Regime {regime}: {count} samples ({count/len(hidden_states):.1%})")
            
            # Check for severe imbalance
            if count < 10:  # Severely underrepresented regime
                print(f"Warning: Regime {regime} has very few samples")
        
        # Add to on-chain features
        self.onchain_features['onchain_regime'] = hidden_states
        
        # Add to parent features if available
        if self.parent.features is not None:
            common_idx = self.parent.features.index.intersection(self.onchain_features.index)
            if len(common_idx) > 0:
                self.parent.features['onchain_regime'] = pd.Series(
                    index=self.parent.features.index,
                    data=np.nan
                )
                self.parent.features.loc[common_idx, 'onchain_regime'] = self.onchain_regimes.loc[common_idx, 'onchain_regime']
        
        # Save model and scaler
        self.models['hmm'] = (best_model, scaler, regime_features)
        
        # Analyze regimes
        self._analyze_onchain_regimes(n_regimes)
        
        # Return model data for validation
        return best_model, hidden_states
    
    def _fallback_regime_detection(self, features_df, n_clusters):
        """
        Fallback to KMeans clustering if HMM fails.
        """
        from sklearn.cluster import KMeans
        
        print("Using KMeans clustering as fallback regime detection method")
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df)
        
        # Find optimal number of clusters if necessary
        if n_clusters > len(features_df) // 20:
            # If too many clusters for data size, reduce
            n_clusters = max(2, len(features_df) // 20)
            print(f"Adjusted number of regimes to {n_clusters} based on data size")
        
        # Train KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Create DataFrame with regime classifications
        self.onchain_regimes = pd.DataFrame({
            'onchain_regime': clusters
        }, index=features_df.index)
        
        # Add to on-chain features
        self.onchain_features['onchain_regime'] = clusters
        
        # Add to parent features if available
        if self.parent.features is not None:
            common_idx = self.parent.features.index.intersection(self.onchain_features.index)
            if len(common_idx) > 0:
                self.parent.features['onchain_regime'] = pd.Series(
                    index=self.parent.features.index,
                    data=np.nan
                )
                self.parent.features.loc[common_idx, 'onchain_regime'] = self.onchain_regimes.loc[common_idx, 'onchain_regime']
        
        # Save model (use KMeans as the model)
        self.models['kmeans'] = (kmeans, scaler, features_df.columns.tolist())
        
        # Analyze regimes
        self._analyze_onchain_regimes(n_clusters)
        
        # Store validation metrics
        self.validation_metrics['hmm']['model_score'] = kmeans.inertia_
        self.validation_metrics['hmm']['algorithm'] = 'kmeans'
        
        return kmeans, clusters
    
    def _analyze_onchain_regimes(self, n_regimes):
        """
        Analyze regime characteristics with robust validation checks.
        """
        if self.onchain_regimes is None or self.onchain_features is None:
            print("Error: No regimes or features to analyze")
            return
            
        print("\nAnalyzing on-chain regime characteristics...")
        
        # Create a summary table for each regime
        regime_summary = pd.DataFrame(index=range(n_regimes))
        
        # Enhance with descriptive feature calculations per regime
        descriptive_metrics = {}
        for regime in range(n_regimes):
            # Get data for this regime
            regime_data = self.onchain_features[self.onchain_regimes['onchain_regime'] == regime]
            
            # Skip if no data (could happen if a regime is empty)
            if len(regime_data) == 0:
                print(f"Warning: Regime {regime} has no data points!")
                continue
            
            # Basic count statistics
            regime_summary.loc[regime, 'Count'] = len(regime_data)
            regime_summary.loc[regime, 'Percent'] = len(regime_data) / len(self.onchain_regimes) * 100
            
            # For each feature, compute mean value
            for feature in self.onchain_features.columns:
                if feature == 'onchain_regime':
                    continue
                    
                feature_mean = regime_data[feature].mean()
                descriptive_metrics[(regime, feature)] = feature_mean
        
        # Get the top 5 most distinctive features for each regime
        # (comparing each regime's feature mean to global feature mean)
        for regime in range(n_regimes):
            if regime_summary.loc[regime, 'Count'] == 0:
                continue
                
            feature_diffs = {}
            for feature in self.onchain_features.columns:
                if feature == 'onchain_regime':
                    continue
                
                # Get global and regime means
                global_mean = self.onchain_features[feature].mean()
                regime_mean = descriptive_metrics.get((regime, feature), 0)
                
                # Calculate z-score of difference 
                global_std = self.onchain_features[feature].std()
                if global_std > 0:
                    z_diff = (regime_mean - global_mean) / global_std
                    feature_diffs[feature] = z_diff
            
            # Sort features by absolute z-score difference
            distinctive_features = sorted(
                feature_diffs.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:5]
            
            # Add distinctive features to summary
            feature_desc = ", ".join([
                f"{feature} ({z_score:+.2f}σ)" 
                for feature, z_score in distinctive_features
            ])
            regime_summary.loc[regime, 'Distinctive Features'] = feature_desc
        
        # Add price performance information if available
        price_data = None
        returns_data = None
        
        if self.parent.features is not None:
            # Look for price column in parent features
            price_cols = [col for col in self.parent.features.columns if 'price' in col.lower() or 'close' in col.lower()]
            if price_cols:
                price_data = self.parent.features[price_cols[0]]
                
            # Look for returns column
            if 'returns' in self.parent.features.columns:
                returns_data = self.parent.features['returns']
        
        if price_data is not None and returns_data is not None:
            # Align price data with regime data
            common_idx = self.onchain_regimes.index.intersection(price_data.index)
            
            # Check if there's enough overlapping data
            if len(common_idx) > 0.5 * len(self.onchain_regimes):  # At least 50% overlap
                print(f"Found price data overlapping with {len(common_idx)} out of {len(self.onchain_regimes)} regime data points")
                
                price_aligned = price_data.loc[common_idx]
                returns_aligned = returns_data.loc[common_idx]
                regimes_aligned = self.onchain_regimes.loc[common_idx, 'onchain_regime']
                
                # Calculate price performance for each regime
                has_valid_returns = False
                for regime in range(n_regimes):
                    # Get data for this regime
                    regime_dates = common_idx[regimes_aligned == regime]
                    
                    if len(regime_dates) == 0:
                        continue
                        
                    regime_returns = returns_aligned.loc[regime_dates]
                    
                    # Validate return data (check for NaN or extreme values)
                    valid_returns = regime_returns.dropna()
                    
                    if len(valid_returns) < 5:  # Need at least 5 data points
                        print(f"Warning: Not enough valid return data for regime {regime}")
                        continue
                        
                    # Exclude extreme outliers for more robust metrics
                    q1 = valid_returns.quantile(0.01)
                    q99 = valid_returns.quantile(0.99)
                    valid_returns = valid_returns[(valid_returns >= q1) & (valid_returns <= q99)]
                    
                    if len(valid_returns) < 5:
                        print(f"Warning: Not enough non-outlier returns for regime {regime}")
                        continue
                    
                    # Calculate performance metrics
                    daily_return = valid_returns.mean() * 100  # As percentage
                    annual_return = valid_returns.mean() * 252 * 100  # Annualized percentage
                    volatility = valid_returns.std() * np.sqrt(252) * 100  # Annualized percentage
                    
                    # Sharpe ratio (annualized)
                    sharpe = annual_return / volatility if volatility > 0 else 0
                    
                    # Store metrics
                    regime_summary.loc[regime, 'Daily Return (%)'] = daily_return
                    regime_summary.loc[regime, 'Ann. Return (%)'] = annual_return
                    regime_summary.loc[regime, 'Volatility (%)'] = volatility
                    regime_summary.loc[regime, 'Sharpe Ratio'] = sharpe
                    
                    has_valid_returns = True
                
                if has_valid_returns:
                    print("\nRegime return statistics validated successfully")
                else:
                    print("\nWarning: Could not calculate valid return statistics for any regime")
            else:
                print(f"Warning: Insufficient overlap between price and regime data ({len(common_idx)} of {len(self.onchain_regimes)} points)")
        else:
            print("Note: Price/return data not available for regime analysis")
        
        # Store regime summary
        self.regime_summary = regime_summary
        
        # Determine regime types based on metrics (if available)
        if 'Sharpe Ratio' in regime_summary.columns and 'Ann. Return (%)' in regime_summary.columns:
            # Define regime categories
            regime_types = {}
            
            for regime in range(n_regimes):
                if pd.isna(regime_summary.loc[regime, 'Sharpe Ratio']):
                    regime_types[regime] = "Unknown"
                    continue
                    
                sharpe = regime_summary.loc[regime, 'Sharpe Ratio']
                annual_return = regime_summary.loc[regime, 'Ann. Return (%)']
                volatility = regime_summary.loc[regime, 'Volatility (%)']
                
                if annual_return > 50 and sharpe > 1.0:
                    regime_types[regime] = "Strong Bull"
                elif annual_return > 20 and sharpe > 0.5:
                    regime_types[regime] = "Moderate Bull"
                elif annual_return < -30 and sharpe < -0.5:
                    regime_types[regime] = "Strong Bear"
                elif annual_return < -15 and sharpe < -0.3:
                    regime_types[regime] = "Moderate Bear"
                elif -15 <= annual_return <= 15 and volatility < 50:
                    regime_types[regime] = "Low Volatility Consolidation"
                elif -15 <= annual_return <= 15 and volatility >= 50:
                    regime_types[regime] = "High Volatility Consolidation"
                else:
                    regime_types[regime] = "Mixed/Neutral"
            
            # Add regime types to summary
            regime_summary['Regime Type'] = pd.Series(regime_types)
            
            print("\nRegime classification:")
            for regime, regime_type in regime_types.items():
                print(f"Regime {regime}: {regime_type}")
        
        # Print the final regime summary
        pd.set_option('display.float_format', '{:.2f}'.format)
        pd.set_option('display.width', 100)
        print("\nOn-chain Regime Characteristics:")
        print(regime_summary)
    
    def train_onchain_cnn(self, lookback_window=14, validation_split=0.2):
        """
        Train a dedicated CNN model for on-chain pattern recognition with comprehensive validation.

        Args:
            lookback_window (int): Number of days to look back for pattern recognition
            validation_split (float): Percentage of data to use for validation
        """
        if self.onchain_features is None:
            print("Error: No on-chain features available. Run extract_onchain_features() first.")
            return

        print(f"Training dedicated CNN for on-chain pattern recognition with {lookback_window}-day window...")

        # Check if we have price target data in parent features
        if self.parent.features is None or 'target_7d' not in self.parent.features.columns:
            print("Error: No target data available in parent features. Creating synthetic targets for testing.")

            # Create synthetic targets for testing if parent features are missing
            timestamps = self.onchain_features.index

            # Synthetic binary target (roughly balanced)
            synthetic_targets = pd.Series(
                index=timestamps,
                data=(np.sin(np.linspace(0, 10*np.pi, len(timestamps))) > 0).astype(int)
            )

            # Store in parent features
            if self.parent.features is None:
                self.parent.features = pd.DataFrame(index=timestamps)

            self.parent.features['target_7d'] = synthetic_targets

            print("Created synthetic target data for testing")

        # Select features with validation
        all_features = self.onchain_features.columns.tolist()

        # Remove the regime column (outcome of another model)
        if 'onchain_regime' in all_features:
            all_features.remove('onchain_regime')

        # Categorize features - FIX: Define categories in steps, not all at once
        # Define primary features first
        primary_features = [col for col in all_features if any(
            x in col for x in ['ratio', 'sentiment', 'hodl', 'mvrv', 'nvt', 'cycle']
        )]

        # Then define secondary features based on what's not in primary
        secondary_features = [col for col in all_features if col not in primary_features]

        # Now create the feature categories dictionary
        feature_categories = {
            'primary': primary_features,
            'secondary': secondary_features
        }

        # Choose features based on data quality
        onchain_cnn_features = []

        # Add primary features (most important ones)
        primary_limit = min(8, len(feature_categories['primary']))
        onchain_cnn_features.extend(feature_categories['primary'][:primary_limit])

        # Add some secondary features if needed
        if len(onchain_cnn_features) < 5 and feature_categories['secondary']:
            secondary_limit = min(5 - len(onchain_cnn_features), len(feature_categories['secondary']))
            onchain_cnn_features.extend(feature_categories['secondary'][:secondary_limit])

        # Ensure we have enough features
        min_features = 5
        if len(onchain_cnn_features) < min_features:
            print(f"Warning: Not enough features for CNN ({len(onchain_cnn_features)} < {min_features})")

            # Create synthetic features if needed
            for i in range(min_features - len(onchain_cnn_features)):
                col_name = f"synthetic_feature_{i}"
                self.onchain_features[col_name] = np.random.normal(0, 1, len(self.onchain_features))
                onchain_cnn_features.append(col_name)

            print(f"Added {min_features - len(onchain_cnn_features)} synthetic features")

        # Report the final feature set
        print(f"Using {len(onchain_cnn_features)} features for CNN: {onchain_cnn_features}")

        # Align on-chain features with target data
        common_idx = self.onchain_features.index.intersection(self.parent.features.index)

        # Ensure enough data for training
        min_samples = lookback_window + 30  # Minimum is window size plus some extra for training
        if len(common_idx) < min_samples:
            print(f"Error: Not enough aligned data points. Found {len(common_idx)}, need at least {min_samples}.")
            print("Creating synthetic data for testing purposes...")

            # Create synthetic data by replicating existing data with noise
            if len(common_idx) > 0:
                # Use existing data as base
                base_features = self.onchain_features.loc[common_idx, onchain_cnn_features]
                base_targets = self.parent.features.loc[common_idx, 'target_7d']

                # Replicate with noise until we have enough
                num_replications = int(np.ceil(min_samples / len(common_idx)))

                extended_features = []
                extended_targets = []

                for i in range(num_replications):
                    # Add noise to features
                    noise_factor = 0.1 * (i + 1)  # Increase noise with each replication
                    noisy_features = base_features.copy()

                    for col in noisy_features.columns:
                        std = noisy_features[col].std()
                        noisy_features[col] += np.random.normal(0, noise_factor * std, len(noisy_features))

                    extended_features.append(noisy_features)

                    # Add noise to targets (flip some labels)
                    noisy_targets = base_targets.copy()
                    flip_mask = np.random.random(len(noisy_targets)) < (0.1 * (i + 1))
                    noisy_targets[flip_mask] = 1 - noisy_targets[flip_mask]

                    extended_targets.append(noisy_targets)

                # Combine replicated data
                X_features = pd.concat(extended_features)
                y_target = pd.concat(extended_targets)

                # Truncate to required size
                X_features = X_features.iloc[:min_samples]
                y_target = y_target.iloc[:min_samples]

                print(f"Created synthetic dataset with {len(X_features)} samples")
            else:
                # Create completely synthetic data
                X_features = pd.DataFrame(
                    np.random.normal(0, 1, (min_samples, len(onchain_cnn_features))),
                    columns=onchain_cnn_features
                )

                # Create synthetic binary targets (balanced)
                y_target = pd.Series(np.random.randint(0, 2, min_samples))

                print(f"Created fully synthetic dataset with {len(X_features)} samples")
        else:
            # Use actual data
            print(f"Found {len(common_idx)} aligned data points for CNN training")
            X_features = self.onchain_features.loc[common_idx, onchain_cnn_features]
            y_target = self.parent.features.loc[common_idx, 'target_7d']

        # Clean the data
        X_features = X_features.replace([np.inf, -np.inf], np.nan)

        # Check for NaN values
        nan_cols = X_features.columns[X_features.isnull().any()].tolist()
        if nan_cols:
            print(f"Warning: NaN values found in columns: {nan_cols}")

            # Handle NaN values (fill with column median)
            for col in X_features.columns:
                X_features[col] = X_features[col].fillna(X_features[col].median())

        # Final check for any remaining NaNs
        if X_features.isnull().sum().sum() > 0:
            print("Warning: NaN values remain after cleaning. Filling with 0.")
            X_features = X_features.fillna(0)

        # Create sequences for CNN
        X_data = []
        y_data = []

        for i in range(len(X_features) - lookback_window):
            # Get the window and ensure it's clean
            window = X_features.iloc[i:i+lookback_window].values
            window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

            # Add to dataset
            X_data.append(window)
            y_data.append(y_target.iloc[i+lookback_window])

        X_data = np.array(X_data)
        y_data = np.array(y_data)

        # Validate dataset balance
        class_balance = np.mean(y_data)
        print(f"Target class balance: {class_balance:.2%} positive samples")

        if class_balance < 0.1 or class_balance > 0.9:
            print("Warning: Highly imbalanced dataset may lead to biased model")

        # Check data shapes
        print(f"CNN Input shape: {X_data.shape}, Target shape: {y_data.shape}")

        # Split data with validation
        try:
            from sklearn.model_selection import train_test_split

            # Use stratified split to maintain class balance
            X_train, X_test, y_train, y_test = train_test_split(
                X_data, y_data, 
                test_size=0.2,
                shuffle=True,  # Shuffling for better generalization
                stratify=y_data,  # Maintain class distribution
                random_state=42
            )

            print(f"Training set: {X_train.shape[0]} samples")
            print(f"Test set: {X_test.shape[0]} samples")

            # Verify split maintained class balance
            train_balance = np.mean(y_train)
            test_balance = np.mean(y_test)
            print(f"Training set balance: {train_balance:.2%} positive")
            print(f"Test set balance: {test_balance:.2%} positive")

        except Exception as e:
            print(f"Error in train-test split: {str(e)}")
            print("Using simple temporal split instead")

            # Simple temporal split
            split_idx = int(len(X_data) * 0.8)
            X_train, X_test = X_data[:split_idx], X_data[split_idx:]
            y_train, y_test = y_data[:split_idx], y_data[split_idx:]

        try:
            # Configure TensorFlow with error handling
            import tensorflow as tf

            try:
                # Use CPU for more stable execution
                tf.config.set_visible_devices([], 'GPU')
                print("TensorFlow configured to use CPU")
            except Exception as tf_config_error:
                print(f"Warning in TF config: {str(tf_config_error)}")
                print("Continuing with default TensorFlow configuration")

            # Reset TensorFlow session and clear memory
            tf.keras.backend.clear_session()

            # Adaptive model size based on data
            if len(X_train) < 100:
                print("Small dataset detected, using simpler model architecture")
                model = tf.keras.Sequential([
                    # Simplified architecture
                    tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', 
                                    input_shape=(lookback_window, len(onchain_cnn_features))),
                    tf.keras.layers.GlobalAveragePooling1D(),
                    tf.keras.layers.Dense(8, activation='relu'),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                batch_size = 8
                epochs = 50
            else:
                # Regular model for larger datasets
                model = tf.keras.Sequential([
                    # First conv layer with more filters to capture on-chain patterns
                    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', 
                                    input_shape=(lookback_window, len(onchain_cnn_features))),
                    tf.keras.layers.BatchNormalization(),  # Normalization for stability
                    tf.keras.layers.MaxPooling1D(pool_size=2),

                    # Second conv layer for higher-level pattern detection
                    tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling1D(pool_size=2),

                    # Global pooling to reduce dimensions
                    tf.keras.layers.GlobalAveragePooling1D(),

                    # Fully connected layers
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dropout(0.3),  # Higher dropout for better generalization
                    tf.keras.layers.Dense(16, activation='relu'),
                    tf.keras.layers.Dropout(0.2),

                    # Output layer
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                batch_size = 32
                epochs = 100

            # Use reduced learning rate for more stable training
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

            # Compile model with binary classification metrics
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=[
                    'accuracy', 
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')
                ]
            )

            # Model summary for validation
            model.summary()

            # Early stopping to prevent overfitting
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            )

            # Learning rate scheduler
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=0.0001
            )

            # Class weights for imbalanced data
            class_weight = None
            if class_balance < 0.3 or class_balance > 0.7:
                # Calculate class weights inversely proportional to class frequencies
                n_negative = np.sum(1 - y_train)
                n_positive = np.sum(y_train)
                total = n_negative + n_positive

                class_weight = {
                    0: total / (2 * n_negative) if n_negative > 0 else 1.0,
                    1: total / (2 * n_positive) if n_positive > 0 else 1.0
                }
                print(f"Using class weights: {class_weight}")

            # Train model with early stopping
            print("Starting on-chain CNN model training...")
            history = model.fit(
                X_train, y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, lr_scheduler],
                class_weight=class_weight,
                verbose=1
            )

            # Evaluate model on test set
            print("Evaluating on-chain CNN model on test set...")
            test_results = model.evaluate(X_test, y_test, verbose=0)

            # Print test metrics
            metric_names = ['loss', 'accuracy', 'auc', 'precision', 'recall']
            test_metrics = {name: value for name, value in zip(metric_names, test_results)}

            print("\nTest set metrics:")
            for metric, value in test_metrics.items():
                print(f"{metric}: {value:.4f}")

            # Store validation metrics
            self.validation_metrics['cnn'] = test_metrics

            # Calculate detailed metrics on test set
            y_pred_prob = model.predict(X_test, verbose=0, batch_size=32)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()

            # Get full classification metrics
            from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
            from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # ROC AUC
            try:
                auc = roc_auc_score(y_test, y_pred_prob)
            except Exception as auc_error:
                print(f"AUC calculation error: {str(auc_error)}")
                auc = 0.5  # Default if calculation fails

            # Precision, Recall, F1
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            # Print metrics
            print(f"\nDetailed test metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"AUC: {auc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

            # Add additional metrics to validation_metrics
            self.validation_metrics['cnn'].update({
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            })

            # Save model and features
            self.models['cnn'] = (model, onchain_cnn_features, lookback_window)

            # Return detailed info for validation
            return model, history, test_metrics

        except Exception as e:
            print(f"Error in CNN model training: {str(e)}")
            import traceback
            traceback.print_exc()

            # Record the error
            self.validation_metrics['cnn']['error'] = str(e)

            print("Falling back to logistic regression model for testing")
            return self._train_fallback_model(X_train, y_train, X_test, y_test)
    
    def _train_fallback_model(self, X_train, y_train, X_test, y_test):
        """Fallback to simpler model if CNN fails"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        print("Training logistic regression as fallback model")
        
        # Flatten the 3D input for use with sklearn
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Create and train pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        pipeline.fit(X_train_flat, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test_flat)
        y_pred_prob = pipeline.predict_proba(X_test_flat)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_pred_prob)
        except:
            auc = 0.5
        
        print(f"Fallback model - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        # Store in validation metrics
        self.validation_metrics['cnn'] = {
            'model_type': 'logistic_regression',
            'accuracy': accuracy,
            'auc': auc
        }
        
        # Store model
        self.models['logistic_regression'] = (pipeline, None, None)
        
        class MockHistory:
            def __init__(self):
                self.history = {'accuracy': [accuracy], 'val_accuracy': [accuracy]}
                
        return pipeline, MockHistory(), self.validation_metrics['cnn']
    
    def generate_onchain_signals(self):
        """
        Generate trading signals from on-chain data with enhanced validation.
        """
        if self.onchain_features is None:
            print("Error: No on-chain features available.")
            return None
            
        print("Generating validated on-chain trading signals...")
        
        # Create a copy of features for signals
        signals_df = self.onchain_features.copy()
        signals_df['onchain_signal'] = 0  # Initialize neutral signal
        
        # Count signals generated by each method
        signal_counts = {
            'cnn': 0,
            'regime': 0,
            'rule': 0,
            'cycle': 0
        }
        
        # 1. Generate CNN-based signals with validation
        if 'cnn' in self.models:
            cnn_signals = self._generate_onchain_cnn_signals(signals_df)
            signal_counts['cnn'] = (signals_df['cnn_signal'] != 0).sum()
        
        # 2. Generate regime-based signals
        if 'onchain_regime' in signals_df.columns:
            regime_signals = self._generate_regime_based_signals(signals_df)
            signal_counts['regime'] = (signals_df['regime_signal'] != 0).sum()
        
        # 3. Generate rule-based signals
        rule_signals = self._generate_rule_based_signals(signals_df)
        signal_counts['rule'] = (signals_df['rule_signal'] != 0).sum()
        
        # 4. Generate cycle-based signals if available
        if 'cycle_phase' in signals_df.columns:
            cycle_signals = self._generate_cycle_based_signals(signals_df)
            signal_counts['cycle'] = (signals_df['cycle_signal'] != 0).sum()
        
        # Log initial signal distribution
        signal_columns = [col for col in signals_df.columns if col.endswith('_signal')]
        if signal_columns:
            print("\nSignal generation statistics:")
            for signal_type, count in signal_counts.items():
                total = len(signals_df)
                if count > 0:
                    print(f"{signal_type} signals: {count} ({count/total:.1%} of data points)")
        
        # 5. Generate combined signal with validation
        if signal_columns:
            # Calculate weighted sum of signals
            signals_df['onchain_signal_raw'] = 0
            
            # Weights for different signal types with validation-based adjustment
            # Default weights:
            weights = {
                'cnn_signal': 1.0,       # CNN predictions
                'regime_signal': 1.0,     # Regime-based
                'rule_signal': 0.7,       # Rule-based
                'cycle_signal': 0.8,      # Cycle-based
                'nvt_signal': 0.5,        # NVT-based
                'mvrv_signal': 0.5,       # MVRV-based
                'hodl_signal': 0.5,       # HODL-based
                'whale_signal': 0.4,      # Whale activity
                'mining_signal': 0.3      # Mining activity
            }
            
            # Adjust CNN weight based on validation metrics if available
            if 'cnn' in self.validation_metrics and 'accuracy' in self.validation_metrics['cnn']:
                cnn_accuracy = self.validation_metrics['cnn']['accuracy']
                
                # Scale weight based on accuracy (higher accuracy = more weight)
                # Baseline is 0.5 accuracy (random guessing)
                accuracy_boost = (cnn_accuracy - 0.5) * 2  # Normalized to [-1, 1] range
                
                if accuracy_boost > 0:
                    # Increase weight for good models
                    weights['cnn_signal'] = 1.0 + accuracy_boost  # Scales to [1.0, 2.0]
                    print(f"CNN model accuracy: {cnn_accuracy:.4f} - Weight adjusted to {weights['cnn_signal']:.2f}")
                else:
                    # Decrease weight for poor models
                    weights['cnn_signal'] = max(0.2, 1.0 + accuracy_boost)  # Minimum 0.2 weight
                    print(f"CNN model accuracy: {cnn_accuracy:.4f} - Weight reduced to {weights['cnn_signal']:.2f}")
            
            # Apply weights to each signal type
            for col in signal_columns:
                # Clean the signal column first
                signals_df[col] = signals_df[col].fillna(0).replace([np.inf, -np.inf], 0)
                
                # Apply weight
                weight = weights.get(col, 0.5)  # Default weight of 0.5
                signals_df['onchain_signal_raw'] += signals_df[col] * weight
            
            # Convert to final signal (-1 to 1)
            signals_df['onchain_signal'] = np.sign(signals_df['onchain_signal_raw'])
            
            # Validate signal distribution
            signal_dist = signals_df['onchain_signal'].value_counts(normalize=True)
            print("\nFinal signal distribution:")
            for signal, freq in signal_dist.items():
                print(f"Signal {signal}: {freq:.1%}")
            
            # Warn if signals are extremely imbalanced
            if (signals_df['onchain_signal'] != 0).mean() < 0.05:
                print("Warning: Very few active signals generated (<5% of data points)")
                
                # If too few signals, try to generate more by lowering threshold
                if (signals_df['onchain_signal'] != 0).mean() < 0.03:
                    print("Adjusting threshold to generate more signals")
                    signals_df['onchain_signal'] = np.sign(signals_df['onchain_signal_raw'].clip(-0.3, 0.3))
                    
                    # Recheck distribution
                    signal_dist = signals_df['onchain_signal'].value_counts(normalize=True)
                    print("Adjusted signal distribution:")
                    for signal, freq in signal_dist.items():
                        print(f"Signal {signal}: {freq:.1%}")
        
        # Store the signals
        self.onchain_signals = signals_df
        
        # Calculate signal agreement metrics
        if len(signal_columns) >= 2:
            # Calculate pairwise agreement between signal sources
            agreement_matrix = {}
            for i, col1 in enumerate(signal_columns):
                for col2 in signal_columns[i+1:]:
                    # Calculate agreement (both same sign or both zero)
                    agreement = (np.sign(signals_df[col1]) == np.sign(signals_df[col2])).mean()
                    agreement_matrix[(col1, col2)] = agreement
            
            # Print agreement statistics
            print("\nSignal agreement matrix:")
            for (col1, col2), agreement in agreement_matrix.items():
                print(f"{col1} vs {col2}: {agreement:.2%} agreement")
        
        # Integrate with parent strategy
        self._integrate_signals_with_parent()
        
        return signals_df
    
    def _generate_onchain_cnn_signals(self, signals_df):
        """Generate validated signals from on-chain CNN model"""
        if 'cnn' not in self.models and 'logistic_regression' not in self.models:
            print("No CNN or fallback model available for signal generation")
            signals_df['cnn_signal'] = 0
            return False
            
        # Determine which model to use
        if 'cnn' in self.models:
            model, features, lookback = self.models['cnn']
            model_type = 'CNN'
        else:
            model, features, lookback = self.models['logistic_regression']
            model_type = 'LogisticRegression'
            # For LogisticRegression, we need to adjust the lookback
            lookback = X_train_flat.shape[1] // len(features) if features else 14
        
        print(f"Generating signals with {model_type} model (lookback={lookback})")
        
        # Initialize signal column
        signals_df['cnn_signal'] = 0
        
        if model_type == 'CNN':
            # Process with standard CNN approach
            X_features = signals_df[features].copy()
            
            # Clean the data
            X_features = X_features.replace([np.inf, -np.inf], np.nan)
            
            # Handle NaN values with column median
            for col in X_features.columns:
                if X_features[col].isnull().any():
                    X_features[col] = X_features[col].fillna(X_features[col].median())
            
            # Process in manageable batches
            batch_size = 32
            signal_count = 0
            
            for i in range(lookback, len(X_features), batch_size):
                batch_end = min(i + batch_size, len(X_features))
                batch_windows = []
                batch_indices = []
                
                for j in range(i, batch_end):
                    if j >= lookback:
                        # Get window data
                        window_data = X_features.iloc[j-lookback:j].values
                        # Handle problematic values
                        window_data = np.nan_to_num(window_data, nan=0.0, posinf=0.0, neginf=0.0)
                        batch_windows.append(window_data)
                        batch_indices.append(j)
                
                if batch_windows:
                    try:
                        # Convert to numpy array
                        batch_windows = np.array(batch_windows)
                        
                        # Get predictions
                        predictions = model.predict(batch_windows, verbose=0, batch_size=32)
                        
                        # Convert to signals with confidence threshold
                        # Use asymmetric thresholds for better precision/recall balance
                        # From validation metrics, determine optimal thresholds
                        if 'precision' in self.validation_metrics['cnn'] and 'recall' in self.validation_metrics['cnn']:
                            precision = self.validation_metrics['cnn']['precision']
                            recall = self.validation_metrics['cnn']['recall']
                            
                            # Adjust thresholds based on precision/recall balance
                            if precision > 0.7 and recall < 0.3:
                                # High precision, low recall: Loosen positive threshold
                                pos_threshold = 0.6
                                neg_threshold = 0.3
                                print("Using precision-optimized thresholds")
                            elif precision < 0.3 and recall > 0.7:
                                # Low precision, high recall: Tighten positive threshold
                                pos_threshold = 0.7
                                neg_threshold = 0.4
                                print("Using recall-balanced thresholds")
                            else:
                                # Balanced metrics: Use standard thresholds
                                pos_threshold = 0.65
                                neg_threshold = 0.35
                        else:
                            # Default thresholds
                            pos_threshold = 0.65
                            neg_threshold = 0.35
                        
                        batch_signals = [
                            1 if pred > pos_threshold else 
                            (-1 if pred < neg_threshold else 0) 
                            for pred in predictions.flatten()
                        ]
                        
                        # Store signals
                        for idx, signal in zip(batch_indices, batch_signals):
                            signals_df.iloc[idx, signals_df.columns.get_loc('cnn_signal')] = signal
                            if signal != 0:
                                signal_count += 1
                        
                    except Exception as e:
                        print(f"Error in batch prediction: {str(e)}")
                        # Skip this batch on error
            
            print(f"CNN signal generation completed. Generated {signal_count} non-zero signals.")
            
            return True
            
        else:
            # For logistic regression, handle differently due to flattened input
            print("Using logistic regression for signal generation")
            # Implementation depends on the specific structure of the fallback model
            # This is a simplified version - adapt as needed
            return False
    
    def _generate_regime_based_signals(self, signals_df):
        """Generate signals based on on-chain regime classification with validation"""
        if 'onchain_regime' not in signals_df.columns:
            print("No regime data available for signal generation")
            return False
            
        print("Generating regime-based signals...")
        
        signals_df['regime_signal'] = 0  # Initialize neutral signal
        
        # Identify positive and negative regimes based on validation
        positive_regimes = []
        negative_regimes = []
        
        # Check if we have regime summary with performance data
        if hasattr(self, 'regime_summary'):
            # Use regime summary for more robust signal generation
            for regime in range(len(self.regime_summary)):
                # Skip regimes with no data
                if 'Count' not in self.regime_summary.columns or pd.isna(self.regime_summary.loc[regime, 'Count']):
                    continue
                    
                # Skip regimes with very few samples
                if self.regime_summary.loc[regime, 'Count'] < 10:
                    print(f"Skipping regime {regime} due to insufficient data")
                    continue
                
                # Check return characteristics if available
                if 'Ann. Return (%)' in self.regime_summary.columns and 'Sharpe Ratio' in self.regime_summary.columns:
                    ann_return = self.regime_summary.loc[regime, 'Ann. Return (%)']
                    sharpe = self.regime_summary.loc[regime, 'Sharpe Ratio']
                    
                    # Only use regimes with non-NaN values
                    if pd.isna(ann_return) or pd.isna(sharpe):
                        continue
                    
                    # Use more conservative thresholds based on validation
                    # Strong positive regimes
                    if ann_return > 50 and sharpe > 0.7:
                        positive_regimes.append(regime)
                        print(f"Regime {regime}: Strong positive (Return: {ann_return:.1f}%, Sharpe: {sharpe:.1f})")
                    # Moderate positive with good reliability
                    elif ann_return > 30 and sharpe > 0.5:
                        positive_regimes.append(regime)
                        print(f"Regime {regime}: Moderate positive (Return: {ann_return:.1f}%, Sharpe: {sharpe:.1f})")
                    # Strong negative regimes
                    elif ann_return < -40 and sharpe < -0.5:
                        negative_regimes.append(regime)
                        print(f"Regime {regime}: Strong negative (Return: {ann_return:.1f}%, Sharpe: {sharpe:.1f})")
                    # Moderate negative with good reliability
                    elif ann_return < -25 and sharpe < -0.4:
                        negative_regimes.append(regime)
                        print(f"Regime {regime}: Moderate negative (Return: {ann_return:.1f}%, Sharpe: {sharpe:.1f})")
                    else:
                        print(f"Regime {regime}: Neutral (Return: {ann_return:.1f}%, Sharpe: {sharpe:.1f})")
                        
                # If no return data available, check regime type
                elif 'Regime Type' in self.regime_summary.columns:
                    regime_type = self.regime_summary.loc[regime, 'Regime Type']
                    if regime_type == "Strong Bull" or regime_type == "Moderate Bull":
                        positive_regimes.append(regime)
                        print(f"Regime {regime}: Positive ({regime_type})")
                    elif regime_type == "Strong Bear" or regime_type == "Moderate Bear":
                        negative_regimes.append(regime)
                        print(f"Regime {regime}: Negative ({regime_type})")
                    else:
                        print(f"Regime {regime}: Neutral ({regime_type})")
        else:
            # If no regime summary, estimate based on metrics
            # This is a fallback approach with less validation
            print("No regime summary available. Using on-chain metrics as proxy.")
            
            for regime in signals_df['onchain_regime'].unique():
                regime_data = signals_df[signals_df['onchain_regime'] == regime]
                
                # Check sentiment if available
                if 'onchain_sentiment' in regime_data.columns:
                    avg_sentiment = regime_data['onchain_sentiment'].mean()
                    if avg_sentiment > 0.3:
                        positive_regimes.append(regime)
                        print(f"Regime {regime}: Positive (sentiment={avg_sentiment:.2f})")
                    elif avg_sentiment < -0.3:
                        negative_regimes.append(regime)
                        print(f"Regime {regime}: Negative (sentiment={avg_sentiment:.2f})")
                        
                # Check cycle phase if available
                elif 'cycle_phase' in regime_data.columns:
                    avg_phase = regime_data['cycle_phase'].mean()
                    # Early cycle tends to be bullish
                    if avg_phase < 0.3:
                        positive_regimes.append(regime)
                        print(f"Regime {regime}: Positive (cycle phase={avg_phase:.2f})")
                    # Late cycle tends to be bearish
                    elif avg_phase > 0.7:
                        negative_regimes.append(regime)
                        print(f"Regime {regime}: Negative (cycle phase={avg_phase:.2f})")
        
        # Apply signals based on regime classification
        for regime in positive_regimes:
            signals_df.loc[signals_df['onchain_regime'] == regime, 'regime_signal'] = 1
            
        for regime in negative_regimes:
            signals_df.loc[signals_df['onchain_regime'] == regime, 'regime_signal'] = -1
            
        # Validate signal distribution
        signal_counts = signals_df['regime_signal'].value_counts()
        total = len(signals_df)
        
        print("Regime signal distribution:")
        for signal, count in signal_counts.items():
            print(f"Signal {signal}: {count} ({count/total:.1%})")
            
        return True
    
    def _generate_rule_based_signals(self, signals_df):
        """Generate signals based on validated rule-based on-chain indicators"""
        print("Generating rule-based on-chain signals...")
        
        signals_df['rule_signal'] = 0  # Initialize neutral signal
        
        # Create individual signal components
        signal_components = {}
        
        # Track rule performance for validation
        rule_performance = {
            'rules_checked': 0,
            'rules_applied': 0,
            'signal_distribution': {}
        }
        
        # 1. NVT Ratio based signal
        if 'nvt_ratio' in signals_df.columns:
            rule_performance['rules_checked'] += 1
            signals_df['nvt_signal'] = 0
            
            # Validate NVT ratio range first
            nvt_stats = signals_df['nvt_ratio'].describe()
            
            # Check if NVT ratio is in reasonable range
            if nvt_stats['50%'] > 0 and nvt_stats['std'] > 0:
                print(f"NVT Ratio - Median: {nvt_stats['50%']:.1f}, Std: {nvt_stats['std']:.1f}")
                
                # Calculate rolling Z-score with adaptive window size
                window = min(90, len(signals_df) // 4)  # Adaptive window size
                rolling_mean = signals_df['nvt_ratio'].rolling(window=window).mean()
                rolling_std = signals_df['nvt_ratio'].rolling(window=window).std().clip(0.001)  # Prevent zero division
                
                # Handle the first window days
                global_mean = signals_df['nvt_ratio'].mean()
                global_std = signals_df['nvt_ratio'].std()
                rolling_mean.fillna(global_mean, inplace=True)
                rolling_std.fillna(global_std, inplace=True)
                
                # Calculate z-score
                nvt_z_score = (signals_df['nvt_ratio'] - rolling_mean) / rolling_std
                
                # Generate signals based on validated z-score thresholds
                z_min = nvt_z_score.quantile(0.05)
                z_max = nvt_z_score.quantile(0.95)
                
                # Adjust thresholds based on distribution
                neg_threshold = max(-2.0, z_min * 0.8)  # Don't go below -2.0
                pos_threshold = min(2.0, z_max * 0.8)  # Don't go above 2.0
                
                signals_df.loc[nvt_z_score < neg_threshold, 'nvt_signal'] = 1     # Undervalued - Buy
                signals_df.loc[nvt_z_score > pos_threshold, 'nvt_signal'] = -1    # Overvalued - Sell
                
                # Log rule performance
                rule_performance['rules_applied'] += 1
                signal_counts = signals_df['nvt_signal'].value_counts()
                total = len(signals_df)
                
                print(f"NVT signal - Negative threshold: {neg_threshold:.2f}, Positive threshold: {pos_threshold:.2f}")
                print(f"NVT signal distribution: {signal_counts.to_dict()}")
                
                rule_performance['signal_distribution']['nvt'] = {
                    'buy': (signals_df['nvt_signal'] == 1).sum() / total,
                    'sell': (signals_df['nvt_signal'] == -1).sum() / total,
                    'neutral': (signals_df['nvt_signal'] == 0).sum() / total
                }
            else:
                print(f"NVT Ratio validation failed - data out of normal range")
        
        # 2. MVRV Ratio based signal
        if 'mvrv_ratio' in signals_df.columns:
            rule_performance['rules_checked'] += 1
            signals_df['mvrv_signal'] = 0
            
            # Validate MVRV ratio first
            mvrv_stats = signals_df['mvrv_ratio'].describe()
            
            # Check if MVRV ratio is in reasonable range
            if 0.8 < mvrv_stats['50%'] < 5:
                print(f"MVRV Ratio - Median: {mvrv_stats['50%']:.1f}, Std: {mvrv_stats['std']:.1f}")
                
                # Historical analysis suggests specific MVRV thresholds for Bitcoin
                # Validate with percentile analysis
                mvrv_1p = signals_df['mvrv_ratio'].quantile(0.01)
                mvrv_99p = signals_df['mvrv_ratio'].quantile(0.99)
                
                # Adjust thresholds based on data range
                buy_threshold = max(0.8, mvrv_1p * 1.1)   # Historical bottom ~0.8
                sell_threshold = min(3.5, mvrv_99p * 0.9)  # Historical top ~3.5
                
                # Strong thresholds for more conviction
                strong_buy = max(0.7, mvrv_1p)
                strong_sell = min(4.0, mvrv_99p)
                
                # Generate signals
                signals_df.loc[signals_df['mvrv_ratio'] < buy_threshold, 'mvrv_signal'] = 1     # Buy zone
                signals_df.loc[signals_df['mvrv_ratio'] < strong_buy, 'mvrv_signal'] = 2        # Strong buy
                signals_df.loc[signals_df['mvrv_ratio'] > sell_threshold, 'mvrv_signal'] = -1   # Sell zone
                signals_df.loc[signals_df['mvrv_ratio'] > strong_sell, 'mvrv_signal'] = -2      # Strong sell
                
                # Normalize to -1 to 1 range
                signals_df['mvrv_signal'] = signals_df['mvrv_signal'] / 2
                
                # Log rule performance
                rule_performance['rules_applied'] += 1
                signal_counts = signals_df['mvrv_signal'].value_counts()
                total = len(signals_df)
                
                print(f"MVRV thresholds - Buy: {buy_threshold:.2f}, Sell: {sell_threshold:.2f}")
                print(f"MVRV signal distribution: {signal_counts.to_dict()}")
                
                rule_performance['signal_distribution']['mvrv'] = {
                    'buy': (signals_df['mvrv_signal'] > 0).sum() / total,
                    'sell': (signals_df['mvrv_signal'] < 0).sum() / total,
                    'neutral': (signals_df['mvrv_signal'] == 0).sum() / total
                }
            else:
                print(f"MVRV Ratio validation failed - data out of normal range")
        
        # 3. HODL Ratio based signal with validation
        if 'hodl_ratio' in signals_df.columns:
            rule_performance['rules_checked'] += 1
            signals_df['hodl_signal'] = 0
            
            # Validate HODL ratio range
            hodl_stats = signals_df['hodl_ratio'].describe()
            
            # Check if HODL ratio is in reasonable range (typically between 0.5 and 5)
            if 0.2 < hodl_stats['50%'] < 10 and hodl_stats['std'] > 0:
                print(f"HODL Ratio - Median: {hodl_stats['50%']:.2f}, Std: {hodl_stats['std']:.2f}")

            # Calculate rolling Z-score of HODL ratio
                window = min(90, len(signals_df) // 4)  # Adaptive window size
                rolling_mean = signals_df['hodl_ratio'].rolling(window=window).mean()
                rolling_std = signals_df['hodl_ratio'].rolling(window=window).std().clip(0.001)  # Prevent zero division
                
                # Handle the first window days
                global_mean = signals_df['hodl_ratio'].mean()
                global_std = signals_df['hodl_ratio'].std()
                rolling_mean.fillna(global_mean, inplace=True)
                rolling_std.fillna(global_std, inplace=True)
                
                # Calculate z-score
                hodl_z_score = (signals_df['hodl_ratio'] - rolling_mean) / rolling_std
                
                # Validate z-score distribution
                z_min = hodl_z_score.quantile(0.05)
                z_max = hodl_z_score.quantile(0.95)
                
                # Adjust thresholds based on distribution
                neg_threshold = max(-1.5, z_min * 0.8)
                pos_threshold = min(1.5, z_max * 0.8)
                
                # Generate signals based on Z-score and validation
                signals_df.loc[hodl_z_score < neg_threshold, 'hodl_signal'] = -0.5  # Low HODL ratio (potential distribution)
                signals_df.loc[hodl_z_score > pos_threshold, 'hodl_signal'] = 0.5    # High HODL ratio (potential accumulation)
                
                # Check for HODL ratio change if available for trend confirmation
                if 'hodl_ratio_change' in signals_df.columns:
                    # Verify the column is valid
                    if signals_df['hodl_ratio_change'].notna().mean() > 0.8:  # At least 80% valid values
                        # Strengthen signal if HODL ratio is increasing (more hodling)
                        change_threshold = signals_df['hodl_ratio_change'].quantile(0.8)
                        mask = (signals_df['hodl_ratio_change'] > change_threshold) & (signals_df['hodl_signal'] > 0)
                        signals_df.loc[mask, 'hodl_signal'] = 1.0
                        
                        # Strengthen signal if HODL ratio is decreasing rapidly (distribution)
                        change_threshold_neg = signals_df['hodl_ratio_change'].quantile(0.2)
                        mask = (signals_df['hodl_ratio_change'] < change_threshold_neg) & (signals_df['hodl_signal'] < 0)
                        signals_df.loc[mask, 'hodl_signal'] = -1.0
                
                # Log rule performance
                rule_performance['rules_applied'] += 1
                signal_counts = signals_df['hodl_signal'].value_counts()
                total = len(signals_df)
                
                print(f"HODL signal thresholds - Negative: {neg_threshold:.2f}, Positive: {pos_threshold:.2f}")
                print(f"HODL signal distribution: {signal_counts.to_dict()}")
                
                rule_performance['signal_distribution']['hodl'] = {
                    'buy': (signals_df['hodl_signal'] > 0).sum() / total,
                    'sell': (signals_df['hodl_signal'] < 0).sum() / total,
                    'neutral': (signals_df['hodl_signal'] == 0).sum() / total
                }
            else:
                print(f"HODL Ratio validation failed - data out of normal range")
        
        # 4. Whale activity based signal with validation
        if 'whale_dominance' in signals_df.columns:
            rule_performance['rules_checked'] += 1
            signals_df['whale_signal'] = 0
            
            # Validate whale dominance range
            whale_stats = signals_df['whale_dominance'].describe()
            
            # Check if whale dominance is in reasonable range (typically a small percentage)
            if 0 < whale_stats['mean'] < 0.3 and whale_stats['std'] > 0:
                print(f"Whale Dominance - Mean: {whale_stats['mean']:.4f}, Std: {whale_stats['std']:.4f}")
                
                # Calculate rolling Z-score with adaptive window
                window = min(90, len(signals_df) // 4)
                rolling_mean = signals_df['whale_dominance'].rolling(window=window).mean()
                rolling_std = signals_df['whale_dominance'].rolling(window=window).std().clip(0.001)
                
                # Handle the first window days
                global_mean = signals_df['whale_dominance'].mean()
                global_std = signals_df['whale_dominance'].std()
                rolling_mean.fillna(global_mean, inplace=True)
                rolling_std.fillna(global_std, inplace=True)
                
                # Calculate z-score
                whale_z_score = (signals_df['whale_dominance'] - rolling_mean) / rolling_std
                
                # Validate z-score distribution
                z_min = whale_z_score.quantile(0.05)
                z_max = whale_z_score.quantile(0.95)
                
                # Adjust thresholds based on distribution
                pos_threshold = min(2.0, z_max * 0.8)
                neg_threshold = max(-2.0, z_min * 0.8)
                
                # Generate signals based on Z-score
                signals_df.loc[whale_z_score > pos_threshold, 'whale_signal'] = 1    # Increased whale accumulation
                signals_df.loc[whale_z_score < neg_threshold, 'whale_signal'] = -1   # Decreased whale dominance
                
                # Log rule performance
                rule_performance['rules_applied'] += 1
                signal_counts = signals_df['whale_signal'].value_counts()
                total = len(signals_df)
                
                print(f"Whale signal thresholds - Negative: {neg_threshold:.2f}, Positive: {pos_threshold:.2f}")
                print(f"Whale signal distribution: {signal_counts.to_dict()}")
                
                rule_performance['signal_distribution']['whale'] = {
                    'buy': (signals_df['whale_signal'] > 0).sum() / total,
                    'sell': (signals_df['whale_signal'] < 0).sum() / total,
                    'neutral': (signals_df['whale_signal'] == 0).sum() / total
                }
            else:
                print(f"Whale Dominance validation failed - data out of normal range")
        
        # 5. Cycle phase based signal with validation
        if 'cycle_phase' in signals_df.columns:
            rule_performance['rules_checked'] += 1
            signals_df['cycle_signal'] = 0
            
            # Validate cycle phase range (should be 0-1)
            cycle_stats = signals_df['cycle_phase'].describe()
            
            # Check if cycle phase is in valid range
            if 0 <= cycle_stats['min'] <= cycle_stats['max'] <= 1:
                print(f"Cycle Phase - Mean: {cycle_stats['mean']:.2f}, Range: [{cycle_stats['min']:.2f}, {cycle_stats['max']:.2f}]")
                
                # Adjust thresholds based on distribution
                early_threshold = signals_df['cycle_phase'].quantile(0.25)
                late_threshold = signals_df['cycle_phase'].quantile(0.75)
                
                # Generate signals based on cycle phase
                signals_df.loc[signals_df['cycle_phase'] < early_threshold, 'cycle_signal'] = 1      # Early cycle - Buy
                signals_df.loc[signals_df['cycle_phase'] > late_threshold, 'cycle_signal'] = -1     # Late cycle - Sell
                
                # Check cycle momentum if available for trend confirmation
                if 'cycle_momentum' in signals_df.columns:
                    if signals_df['cycle_momentum'].notna().mean() > 0.8:  # At least 80% valid values
                        # Get distribution for validation
                        momentum_high = signals_df['cycle_momentum'].quantile(0.8)
                        momentum_low = signals_df['cycle_momentum'].quantile(0.2)
                        
                        # If in mid-cycle but momentum is strong positive, generate buy signal
                        mask = (signals_df['cycle_phase'].between(early_threshold, 0.5)) & (signals_df['cycle_momentum'] > momentum_high)
                        signals_df.loc[mask, 'cycle_signal'] = 0.5  # Moderate buy
                        
                        # If in mid-cycle but momentum is strong negative, generate sell signal
                        mask = (signals_df['cycle_phase'].between(0.5, late_threshold)) & (signals_df['cycle_momentum'] < momentum_low)
                        signals_df.loc[mask, 'cycle_signal'] = -0.5  # Moderate sell
                
                # Log rule performance
                rule_performance['rules_applied'] += 1
                signal_counts = signals_df['cycle_signal'].value_counts()
                total = len(signals_df)
                
                print(f"Cycle signal thresholds - Early: {early_threshold:.2f}, Late: {late_threshold:.2f}")
                print(f"Cycle signal distribution: {signal_counts.to_dict()}")
                
                rule_performance['signal_distribution']['cycle'] = {
                    'buy': (signals_df['cycle_signal'] > 0).sum() / total,
                    'sell': (signals_df['cycle_signal'] < 0).sum() / total,
                    'neutral': (signals_df['cycle_signal'] == 0).sum() / total
                }
            else:
                print(f"Cycle Phase validation failed - data out of expected range [0,1]")
        
        # 6. Combine rule-based signals with weighted approach
        rule_cols = [col for col in signals_df.columns if col.endswith('_signal') and col != 'cnn_signal' and col != 'regime_signal']
        avail_rule_cols = [col for col in rule_cols if col != 'rule_signal']
        
        if avail_rule_cols:
            # Ensure all signal columns have 0 for NaN values
            for col in avail_rule_cols:
                signals_df[col] = signals_df[col].fillna(0)
            
            # Get weight adjustment based on validation results
            # More reliable rules get more weight
            weights = {}
            
            # Default weights
            default_weights = {
                'nvt_signal': 1.0,
                'mvrv_signal': 1.0,
                'hodl_signal': 0.8,
                'whale_signal': 0.7,
                'mining_signal': 0.5,
                'cycle_signal': 0.8
            }
            
            # Adjust weights based on signal distribution
            for col in avail_rule_cols:
                base_weight = default_weights.get(col, 0.5)
                
                # Extract rule type from column name
                rule_type = col.replace('_signal', '')
                
                # Get distribution data if available
                if (rule_type in rule_performance['signal_distribution'] and 
                    'buy' in rule_performance['signal_distribution'][rule_type]):
                    
                    dist = rule_performance['signal_distribution'][rule_type]
                    
                    # Calculate signal frequency (% of non-neutral signals)
                    signal_freq = dist['buy'] + dist['sell']
                    
                    # Penalize rules that generate too few signals
                    if signal_freq < 0.05:  # Less than 5% signals
                        weight_adj = 0.8  # 20% penalty
                        print(f"{col} generates few signals ({signal_freq:.1%}) - reducing weight by 20%")
                    # Penalize rules that generate too many signals
                    elif signal_freq > 0.4:  # More than 40% signals
                        weight_adj = 0.9  # 10% penalty
                        print(f"{col} generates many signals ({signal_freq:.1%}) - reducing weight by 10%")
                    else:
                        weight_adj = 1.0  # No adjustment
                        
                    # Final weight with adjustment
                    weights[col] = base_weight * weight_adj
                else:
                    weights[col] = base_weight
            
            # Report weights
            print("\nRule weights after validation:")
            for col, weight in weights.items():
                print(f"  {col}: {weight:.2f}")
            
            # Calculate weighted average of rule-based signals
            signals_df['rule_signal_raw'] = 0
            weight_sum = 0
            
            for col in avail_rule_cols:
                weight = weights.get(col, 0.5)
                signals_df['rule_signal_raw'] += signals_df[col] * weight
                weight_sum += weight
            
            if weight_sum > 0:
                # Normalize by dividing by sum of weights
                signals_df['rule_signal_raw'] /= weight_sum
                
                # Convert to discrete signal (-1, 0, 1) with adaptive thresholds
                # Calculate distribution for determining thresholds
                q25 = signals_df['rule_signal_raw'].quantile(0.25)
                q75 = signals_df['rule_signal_raw'].quantile(0.75)
                
                # Set thresholds to generate reasonable number of signals
                buy_threshold = max(0.2, min(0.3, q75 * 0.8))
                sell_threshold = min(-0.2, max(-0.3, q25 * 0.8))
                
                # Apply thresholds
                signals_df['rule_signal'] = np.where(
                    signals_df['rule_signal_raw'] > buy_threshold, 1,
                    np.where(signals_df['rule_signal_raw'] < sell_threshold, -1, 0)
                )
                
                # Report signal distribution
                signal_counts = signals_df['rule_signal'].value_counts()
                total = len(signals_df)
                
                print(f"\nRule signal thresholds - Buy: {buy_threshold:.2f}, Sell: {sell_threshold:.2f}")
                print(f"Rule signal distribution: {signal_counts.to_dict()}")
                
                # Log overall rule performance
                rule_performance['buy_signals'] = (signals_df['rule_signal'] == 1).sum() / total
                rule_performance['sell_signals'] = (signals_df['rule_signal'] == -1).sum() / total
                rule_performance['neutral_signals'] = (signals_df['rule_signal'] == 0).sum() / total
            else:
                print("No valid rules available for signal generation")
        
        # Store rule performance for evaluation
        self.validation_metrics['rules'] = rule_performance
        
        return True
    
    def _generate_cycle_based_signals(self, signals_df):
        """Generate signals based on market cycle analysis with validation"""
        if 'cycle_phase' not in signals_df.columns:
            print("No cycle phase data available for signal generation")
            return False
            
        print("Generating cycle-based signals...")
        
        # Already handled in rule-based signals
        # This is kept as a separate method for compatibility
        return True
    
    def _integrate_signals_with_parent(self):
        """Integrate on-chain signals with parent strategy signals"""
        if self.onchain_signals is None:
            print("No on-chain signals available for integration")
            return False
            
        if not hasattr(self.parent, 'signals') or self.parent.signals is None:
            print("Parent strategy has no signals for integration")
            return False
        
        print("Integrating on-chain signals with parent strategy...")
        
        # Align indices
        parent_idx = self.parent.signals.index
        onchain_idx = self.onchain_signals.index
        common_idx = parent_idx.intersection(onchain_idx)
        
        if len(common_idx) == 0:
            print("Error: No common dates between parent and on-chain signals.")
            return False
            
        print(f"Found {len(common_idx)} common dates for signal integration ({len(common_idx)/len(parent_idx):.1%} of parent data)")
        
        # Add on-chain signal to parent signals if not already present
        if 'onchain_signal' not in self.parent.signals.columns:
            self.parent.signals.loc[common_idx, 'onchain_signal'] = self.onchain_signals.loc[common_idx, 'onchain_signal']
            
        # Add on-chain regime if available
        if 'onchain_regime' in self.onchain_signals.columns:
            self.parent.signals.loc[common_idx, 'onchain_regime'] = self.onchain_signals.loc[common_idx, 'onchain_regime']
        
        # Add individual signal components for detailed analysis
        component_columns = [col for col in self.onchain_signals.columns if col.endswith('_signal') and col != 'onchain_signal']
        for col in component_columns:
            self.parent.signals.loc[common_idx, col] = self.onchain_signals.loc[common_idx, col]
        
        # Add signal strength for position sizing
        if 'onchain_signal_raw' in self.onchain_signals.columns:
            self.parent.signals.loc[common_idx, 'onchain_signal_strength'] = self.onchain_signals.loc[common_idx, 'onchain_signal_raw']
        
        # Create enhanced signal by combining parent and on-chain signals
        self._create_enhanced_signals(common_idx)
        
        return True
    
    def _create_enhanced_signals(self, common_idx):
        """Create enhanced signals by combining parent and on-chain signals"""
        parent_signals = self.parent.signals
        
        # Get original parent signal
        if 'signal' not in parent_signals.columns:
            print("Warning: Parent signals don't have a 'signal' column.")
            parent_signals['signal'] = 0
            
        # Get on-chain signal
        onchain_signal = self.onchain_signals.loc[common_idx, 'onchain_signal']
        
        # Get validation metrics for weighting
        # Default weights: 60% parent, 40% on-chain
        parent_weight = 0.6
        onchain_weight = 0.4
        
        # Adjust weights based on validation metrics if available
        if 'cnn' in self.validation_metrics and 'accuracy' in self.validation_metrics['cnn']:
            cnn_accuracy = self.validation_metrics['cnn']['accuracy']
            
            # Only increase on-chain weight if CNN is performing well
            if cnn_accuracy > 0.6:  # Better than random + margin
                # Scale weight up to 50% based on accuracy
                accuracy_boost = (cnn_accuracy - 0.5) * 2  # 0.6 -> 0.2, 0.7 -> 0.4
                onchain_weight = min(0.5, 0.4 + accuracy_boost * 0.1)
                parent_weight = 1 - onchain_weight
                print(f"Increasing on-chain weight to {onchain_weight:.2f} based on CNN accuracy of {cnn_accuracy:.2f}")
        
        # Create enhanced signal with weighted combination
        parent_signals.loc[common_idx, 'enhanced_signal'] = np.sign(
            parent_signals.loc[common_idx, 'signal'] * parent_weight +
            onchain_signal * onchain_weight
        )
        
        # Enhanced position sizing
        if 'position_size' in parent_signals.columns:
            # Get original position size
            orig_position = parent_signals['position_size']
            
            # Only adjust if on-chain signal strength is available
            if 'onchain_signal_strength' in self.onchain_signals.columns:
                onchain_strength = self.onchain_signals.loc[common_idx, 'onchain_signal_strength']
                
                # Scale the multiplier based on strength and agreement
                # Strong agreement = increase position, strong disagreement = reduce position
                agreement = np.sign(orig_position.loc[common_idx]) == np.sign(onchain_signal)
                
                # Base multiplier (1.0 = no change)
                base_mult = 1.0
                
                # Create conditional multipliers
                agreement_boost = 0.2  # Max 20% boost on agreement
                disagreement_penalty = 0.2  # Max 20% reduction on disagreement
                
                # Calculate final multiplier
                multiplier = np.where(
                    agreement,
                    base_mult + agreement_boost * np.abs(onchain_strength).clip(0, 1),  # Increase on agreement
                    base_mult - disagreement_penalty * np.abs(onchain_strength).clip(0, 1)  # Decrease on disagreement
                )
                
                # Apply multiplier to position size
                parent_signals.loc[common_idx, 'enhanced_position_size'] = (
                    orig_position.loc[common_idx] * multiplier
                )
                
                # Ensure positions stay within bounds (-1 to 1)
                parent_signals['enhanced_position_size'] = parent_signals['enhanced_position_size'].clip(-1, 1)
                
                # Validate position sizing
                orig_mean = abs(orig_position.loc[common_idx]).mean()
                enhanced_mean = abs(parent_signals.loc[common_idx, 'enhanced_position_size']).mean()
                
                print(f"Position sizing: Original={orig_mean:.2f}, Enhanced={enhanced_mean:.2f}")
            else:
                # Simple approach if signal strength unavailable
                parent_signals.loc[common_idx, 'enhanced_position_size'] = orig_position.loc[common_idx]
        
        # Apply on-chain regime-specific risk management if available
        if 'onchain_regime' in self.onchain_signals.columns:
            self._apply_onchain_regime_risk_adjustment(parent_signals, common_idx)
        
        # Calculate and report signal changes
        if 'signal' in parent_signals.columns and 'enhanced_signal' in parent_signals.columns:
            agreement_rate = (parent_signals.loc[common_idx, 'signal'] == 
                             parent_signals.loc[common_idx, 'enhanced_signal']).mean()
            
            # Count signal changes
            signal_changes = (parent_signals.loc[common_idx, 'signal'] != 
                             parent_signals.loc[common_idx, 'enhanced_signal']).sum()
            
            print(f"Signal agreement rate: {agreement_rate:.2f}")
            print(f"Number of signal changes: {signal_changes} ({signal_changes/len(common_idx):.1%} of samples)")
    
    def _apply_onchain_regime_risk_adjustment(self, signals_df, common_idx):
        """Apply risk adjustments based on on-chain regimes with validation"""
        if 'onchain_regime' not in signals_df.columns or 'enhanced_position_size' not in signals_df.columns:
            return
            
        # Check if we have regime summary with performance data
        if hasattr(self, 'regime_summary') and 'Sharpe Ratio' in self.regime_summary.columns:
            # Classify regimes based on historical performance with validation
            high_return_regimes = []
            high_risk_regimes = []
            
            for regime in range(len(self.regime_summary)):
                # Skip regimes with invalid data
                if ('Ann. Return (%)' not in self.regime_summary.columns or 
                    pd.isna(self.regime_summary.loc[regime, 'Ann. Return (%)'])):
                    continue
                    
                # Get performance metrics
                ann_return = self.regime_summary.loc[regime, 'Ann. Return (%)']
                volatility = self.regime_summary.loc[regime, 'Volatility (%)'] if 'Volatility (%)' in self.regime_summary.columns else None
                sharpe = self.regime_summary.loc[regime, 'Sharpe Ratio'] if 'Sharpe Ratio' in self.regime_summary.columns else None
                
                # Validate metrics
                if pd.isna(ann_return) or (sharpe is not None and pd.isna(sharpe)):
                    continue
                
                # High return and good Sharpe = increase position
                if ann_return > 50 and (sharpe is None or sharpe > 0.7):
                    high_return_regimes.append(regime)
                    print(f"Regime {regime}: High return regime - increasing positions")
                
                # High volatility and poor Sharpe = reduce position
                if volatility is not None and volatility > 80 and (sharpe is None or sharpe < 0.5):
                    high_risk_regimes.append(regime)
                    print(f"Regime {regime}: High risk regime - reducing positions")
            
            # Apply position size adjustments
            for regime in high_return_regimes:
                mask = (signals_df['onchain_regime'] == regime) & signals_df.index.isin(common_idx)
                if mask.any():
                    # Increase position by 20% in high return regimes
                    signals_df.loc[mask, 'enhanced_position_size'] *= 1.2
                    print(f"Adjusted {mask.sum()} positions in regime {regime} (high return)")
            
            for regime in high_risk_regimes:
                mask = (signals_df['onchain_regime'] == regime) & signals_df.index.isin(common_idx)
                if mask.any():
                    # Reduce position by 20% in high risk regimes
                    signals_df.loc[mask, 'enhanced_position_size'] *= 0.8
                    print(f"Adjusted {mask.sum()} positions in regime {regime} (high risk)")
            
            # Ensure positions stay within bounds
            signals_df['enhanced_position_size'] = signals_df['enhanced_position_size'].clip(-1, 1)
            
            # Validate position adjustments
            if 'position_size' in signals_df.columns:
                original_pos = signals_df.loc[common_idx, 'position_size'].abs().mean()
                enhanced_pos = signals_df.loc[common_idx, 'enhanced_position_size'].abs().mean()
                print(f"Position size after regime adjustment: Original={original_pos:.2f}, Enhanced={enhanced_pos:.2f}")
        else:
            # If no performance data, use sentiment for basic adjustment
            if 'onchain_sentiment' in self.onchain_features.columns:
                # Calculate average sentiment by regime
                regime_sentiment = {}
                
                for regime in self.onchain_features['onchain_regime'].unique():
                    regime_data = self.onchain_features[self.onchain_features['onchain_regime'] == regime]
                    if 'onchain_sentiment' in regime_data.columns:
                        avg_sentiment = regime_data['onchain_sentiment'].mean()
                        if not pd.isna(avg_sentiment):
                            regime_sentiment[regime] = avg_sentiment
                
                # Apply basic position adjustments based on sentiment
                for regime, sentiment in regime_sentiment.items():
                    mask = (signals_df['onchain_regime'] == regime) & signals_df.index.isin(common_idx)
                    
                    if mask.any():
                        if sentiment > 0.3:
                            # Positive sentiment regime - increase positions slightly
                            signals_df.loc[mask, 'enhanced_position_size'] *= 1.1
                            print(f"Adjusted {mask.sum()} positions in regime {regime} (positive sentiment)")
                        elif sentiment < -0.3:
                            # Negative sentiment regime - decrease positions slightly
                            signals_df.loc[mask, 'enhanced_position_size'] *= 0.9
                            print(f"Adjusted {mask.sum()} positions in regime {regime} (negative sentiment)")
                
                # Ensure positions stay within bounds
                signals_df['enhanced_position_size'] = signals_df['enhanced_position_size'].clip(-1, 1)
    
    def validate_model_performance(self):
        """
        Validate and compare model performance across different components.
        This method should be called after backtesting to analyze results.
        """
        if not hasattr(self.parent, 'signals') or self.parent.signals is None:
            print("No signals available for validation analysis")
            return
            
        print("\n" + "="*50)
        print("Validating On-Chain Model Performance")
        print("="*50)
        
        signals = self.parent.signals
        
        # Make sure we have the necessary columns
        if 'returns' not in signals.columns:
            print("Returns data not available for model validation")
            return
            
        # Get aligned data for validation
        if 'onchain_signal' not in signals.columns:
            print("On-chain signals not found in parent signals")
            return
            
        # Create validation dataframe
        valid_mask = signals['onchain_signal'].notna()
        validation_df = signals[valid_mask].copy()
        
        if len(validation_df) == 0:
            print("No valid on-chain signals found for validation")
            return
            
        print(f"Validating on-chain models with {len(validation_df)} data points")
        
        # Get signal components for validation
        signal_components = [col for col in validation_df.columns if col.endswith('_signal') and col != 'signal']
        
        # Calculate accuracy and returns metrics for each signal component
        model_performance = {}
        
        # Reference metrics
        if 'cum_strategy_returns' in validation_df.columns and 'cum_market_returns' in validation_df.columns:
            final_strategy = validation_df['cum_strategy_returns'].iloc[-1]
            final_market = validation_df['cum_market_returns'].iloc[-1]
            
            print(f"Parent strategy return: {final_strategy:.4f}")
            print(f"Market return: {final_market:.4f}")
            
            model_performance['parent_strategy'] = {
                'total_return': final_strategy,
                'vs_market': final_strategy - final_market
            }
        
        # Enhanced strategy performance if available
        if 'enhanced_signal' in validation_df.columns and 'returns' in validation_df.columns:
            # Calculate returns based on enhanced signals
            validation_df['enhanced_returns'] = validation_df['enhanced_signal'].shift(1) * validation_df['returns']
            validation_df['enhanced_cum_returns'] = (1 + validation_df['enhanced_returns']).cumprod() - 1
            
            # Calculate metrics
            enhanced_return = validation_df['enhanced_cum_returns'].iloc[-1]
            
            # Calculate risk and performance metrics
            from sklearn.metrics import accuracy_score
            
            # Direction accuracy (did the signal predict the correct direction?)
            pred_mask = validation_df['enhanced_signal'].shift(1) != 0  # Only consider actual predictions
            if pred_mask.sum() > 0:
                actual_direction = (validation_df.loc[pred_mask, 'returns'] > 0).astype(int)
                pred_direction = (validation_df.loc[pred_mask, 'enhanced_signal'].shift(1) > 0).astype(int)
                accuracy = accuracy_score(actual_direction, pred_direction)
            else:
                accuracy = np.nan
            
            # Sharpe ratio
            if validation_df['enhanced_returns'].std() > 0:
                sharpe = validation_df['enhanced_returns'].mean() / validation_df['enhanced_returns'].std() * np.sqrt(252)
            else:
                sharpe = np.nan
            
            # Store metrics
            model_performance['enhanced_strategy'] = {
                'total_return': enhanced_return,
                'vs_market': enhanced_return - final_market if 'final_market' in locals() else np.nan,
                'accuracy': accuracy,
                'sharpe': sharpe
            }
            
            print(f"Enhanced strategy return: {enhanced_return:.4f}, Accuracy: {accuracy:.4f}, Sharpe: {sharpe:.4f}")
        
        # Validate individual signal components
        for component in signal_components:
            # Skip if not enough signals
            if (validation_df[component] != 0).sum() < 10:
                print(f"Skipping {component} - insufficient signals")
                continue
                
            # Calculate returns based on this signal component
            validation_df[f'{component}_returns'] = validation_df[component].shift(1) * validation_df['returns']
            validation_df[f'{component}_cum_returns'] = (1 + validation_df[f'{component}_returns']).cumprod() - 1
            
            # Calculate metrics
            component_return = validation_df[f'{component}_cum_returns'].iloc[-1]
            
            # Direction accuracy
            pred_mask = validation_df[component].shift(1) != 0
            if pred_mask.sum() > 0:
                actual_direction = (validation_df.loc[pred_mask, 'returns'] > 0).astype(int)
                pred_direction = (validation_df.loc[pred_mask, component].shift(1) > 0).astype(int)
                accuracy = accuracy_score(actual_direction, pred_direction)
            else:
                accuracy = np.nan
            
            # Sharpe ratio
            if validation_df[f'{component}_returns'].std() > 0:
                sharpe = (validation_df[f'{component}_returns'].mean() / 
                         validation_df[f'{component}_returns'].std() * np.sqrt(252))
            else:
                sharpe = np.nan
            
            # Store metrics
            model_performance[component] = {
                'total_return': component_return,
                'vs_market': component_return - final_market if 'final_market' in locals() else np.nan,
                'accuracy': accuracy,
                'sharpe': sharpe,
                'signal_count': (validation_df[component] != 0).sum(),
                'signal_ratio': (validation_df[component] != 0).mean()
            }
        
        # Create comparison table
        model_comparison = pd.DataFrame(model_performance).T
        
        # Sort by total return
        if 'total_return' in model_comparison.columns:
            model_comparison = model_comparison.sort_values('total_return', ascending=False)
        
        # Format for display
        pd.set_option('display.float_format', '{:.4f}'.format)
        print("\nOn-Chain Model Performance Comparison:")
        print(model_comparison)
        
        # Store for reference
        self.model_comparisons = model_comparison
        
        # Highlight best performing models
        if len(model_comparison) > 1:
            # Find best performing model components
            if 'total_return' in model_comparison.columns:
                best_return = model_comparison['total_return'].max()
                best_model = model_comparison['total_return'].idxmax()
                print(f"\nBest performing model: {best_model} (Return: {best_return:.4f})")
            
            if 'accuracy' in model_comparison.columns:
                best_accuracy = model_comparison['accuracy'].max()
                best_model = model_comparison['accuracy'].idxmax()
                print(f"Most accurate model: {best_model} (Accuracy: {best_accuracy:.4f})")
            
            if 'sharpe' in model_comparison.columns:
                best_sharpe = model_comparison['sharpe'].max()
                best_model = model_comparison['sharpe'].idxmax()
                print(f"Highest Sharpe ratio: {best_model} (Sharpe: {best_sharpe:.4f})")
        
        return model_comparison


# Function to add enhanced on-chain analysis to strategy
def add_enhanced_onchain_analysis(strategy):
    """
    Add enhanced on-chain analysis with robust validation to an existing IntegratedBitcoinStrategy instance.
    
    Args:
        strategy: IntegratedBitcoinStrategy instance
        
    Returns:
        onchain_analyzer: EnhancedOnChainAnalysis instance
    """
    # Create on-chain analyzer instance
    onchain_analyzer = EnhancedOnChainAnalysis(strategy)
    
    # Extract specialized on-chain features with validation
    features_extracted = onchain_analyzer.extract_onchain_features()
    
    if not features_extracted:
        print("Warning: Could not extract on-chain features. Continuing with limited functionality.")
    
    # Train on-chain specific models if features available
    if onchain_analyzer.onchain_features is not None:
        # Train on-chain regime HMM
        onchain_analyzer.train_onchain_regime_hmm(n_regimes=5)
        
        # Train on-chain CNN with validation
        onchain_analyzer.train_onchain_cnn(lookback_window=14, validation_split=0.2)
        
        # Generate on-chain signals
        onchain_analyzer.generate_onchain_signals()
    
    # Store the analyzer in the strategy
    strategy.onchain_analyzer = onchain_analyzer
    
    # Modify run_backtest to use enhanced signals
    original_backtest = strategy.backtest_strategy
    
    def enhanced_backtest(use_backtrader=False, initial_cash=100000):
        """Enhanced backtest that validates and uses on-chain enhanced signals"""
        if hasattr(strategy, 'signals') and strategy.signals is not None:
            # Store original signals before potentially modifying
            original_signal = None
            original_position = None
            
            # Check if enhanced signals are available
            if 'enhanced_signal' in strategy.signals.columns:
                # Store original signal for comparison
                original_signal = strategy.signals['signal'].copy()
                if 'position_size' in strategy.signals.columns:
                    original_position = strategy.signals['position_size'].copy()
                
                print("\nRunning backtest with enhanced on-chain signals")
                
                # Use enhanced signal
                strategy.signals['signal'] = strategy.signals['enhanced_signal']
                if 'enhanced_position_size' in strategy.signals.columns and original_position is not None:
                    strategy.signals['position_size'] = strategy.signals['enhanced_position_size']
                
                # Run backtest
                enhanced_result = original_backtest(use_backtrader, initial_cash)
                
                # Store enhanced results for comparison
                enhanced_performance = strategy.performance.copy() if hasattr(strategy, 'performance') else {}
                
                # Restore original signal and run again for direct comparison
                print("\nRunning baseline backtest with original signals for comparison")
                strategy.signals['signal'] = original_signal
                if original_position is not None:
                    strategy.signals['position_size'] = original_position
                
                # Run backtest with original signals
                original_result = original_backtest(use_backtrader, initial_cash)
                
                # Store original performance for comparison
                original_performance = strategy.performance.copy() if hasattr(strategy, 'performance') else {}
                
                # Compare results
                print("\n" + "="*50)
                print("Performance Comparison: Original vs Enhanced Strategy")
                print("="*50)
                
                if enhanced_performance and original_performance:
                    # Create comparison dataframe
                    comparison_metrics = ['Sharpe Ratio', 'Max Drawdown', 'Win Rate', 
                                         'Annual Return', 'Calmar Ratio', 'Total Trades']
                    
                    comparison = pd.DataFrame({
                        'Original Strategy': [original_performance.get(m, np.nan) for m in comparison_metrics],
                        'Enhanced Strategy': [enhanced_performance.get(m, np.nan) for m in comparison_metrics]
                    }, index=comparison_metrics)
                    
                    # Display comparison
                    pd.set_option('display.float_format', '{:.4f}'.format)
                    print(comparison)
                    
                    # Highlight key improvements
                    improved_metrics = []
                    for metric in comparison_metrics:
                        if metric in enhanced_performance and metric in original_performance:
                            # For drawdown, lower is better
                            if metric == 'Max Drawdown':
                                if enhanced_performance[metric] < original_performance[metric]:
                                    improved_metrics.append(metric)
                            # For other metrics, higher is better
                            elif enhanced_performance[metric] > original_performance[metric]:
                                improved_metrics.append(metric)
                    
                    if improved_metrics:
                        print(f"\nEnhanced strategy showed improvement in: {', '.join(improved_metrics)}")
                    else:
                        print("\nEnhanced strategy did not show clear improvements over the original")
                
                # Validate model performance
                if hasattr(strategy, 'onchain_analyzer'):
                    strategy.onchain_analyzer.validate_model_performance()
                
                # Leave with enhanced signals active if they're better
                if (enhanced_performance.get('Sharpe Ratio', 0) > original_performance.get('Sharpe Ratio', 0) or
                    enhanced_performance.get('Annual Return', 0) > original_performance.get('Annual Return', 0)):
                    print("\nRestoring enhanced signals as they showed better performance")
                    strategy.signals['signal'] = strategy.signals['enhanced_signal']
                    if 'enhanced_position_size' in strategy.signals.columns and original_position is not None:
                        strategy.signals['position_size'] = strategy.signals['enhanced_position_size']
                    strategy.performance = enhanced_performance
                    return enhanced_result
                else:
                    print("\nKeeping original signals as they showed better performance")
                    strategy.performance = original_performance
                    return original_result
        
        # Fall back to original backtest if enhanced signals not available
        return original_backtest(use_backtrader, initial_cash)
    
    # Replace backtest method with enhanced version
    strategy.backtest_strategy = enhanced_backtest
    
    # Enhance visualization to show on-chain indicators with validation
    original_visualize = strategy.visualize_results
    
    def enhanced_visualize():
        """Enhanced visualization with on-chain validation metrics"""
        # Call original visualization first
        original_visualize()
        
        # Add on-chain specific visualizations if data available
        if onchain_analyzer.onchain_features is not None and onchain_analyzer.onchain_signals is not None:
            import matplotlib.pyplot as plt
            
            # Create figure for on-chain visualization
            plt.figure(figsize=(15, 20))
            
            # Plot count = 0 to track subplot position
            plot_count = 0
            
            # 1. Compare original vs on-chain enhanced signals
            if hasattr(strategy, 'signals') and 'original_signal' in strategy.signals.columns and 'enhanced_signal' in strategy.signals.columns:
                plot_count += 1
                plt.subplot(5, 1, plot_count)
                
                # Get common date range
                dates = strategy.signals.index
                
                # Plot signals
                plt.plot(dates, strategy.signals['original_signal'], label='Original Signal', color='blue', alpha=0.7)
                plt.plot(dates, strategy.signals['enhanced_signal'], label='On-Chain Enhanced Signal', color='green')
                
                # If we have price data, plot it on secondary axis
                if 'close' in strategy.signals.columns or any('price' in col.lower() for col in strategy.signals.columns):
                    price_col = 'close' if 'close' in strategy.signals.columns else [col for col in strategy.signals.columns if 'price' in col.lower()][0]
                    
                    ax1 = plt.gca()
                    ax2 = ax1.twinx()
                    ax2.plot(dates, strategy.signals[price_col], color='gray', alpha=0.3)
                    ax2.set_ylabel('Price')
                
                plt.title('Trading Signals: Original vs On-Chain Enhanced')
                plt.xlabel('Date')
                plt.ylabel('Signal (-1: Sell, 0: Hold, 1: Buy)')
                plt.legend()
                plt.grid(True)
            
            # 2. On-chain regimes visualization with validation
            if 'onchain_regime' in onchain_analyzer.onchain_features.columns:
                plot_count += 1
                plt.subplot(5, 1, plot_count)
                
                # Create a colormap for regimes
                n_regimes = onchain_analyzer.onchain_features['onchain_regime'].nunique()
                colors = plt.cm.viridis(np.linspace(0, 1, n_regimes))
                
                # Get price data if available
                price_data = None
                if hasattr(strategy, 'signals'):
                    if 'close' in strategy.signals.columns:
                        price_data = strategy.signals['close']
                    elif any('price' in col.lower() for col in strategy.signals.columns):
                        price_col = [col for col in strategy.signals.columns if 'price' in col.lower()][0]
                        price_data = strategy.signals[price_col]
                
                if price_data is not None:
                    # Get data for vertical span highlighting
                    min_price = price_data.min()
                    max_price = price_data.max()
                    price_range = max_price - min_price
                    
                    # Plot regime as background color regions with validation
                    regime_info = []
                    
                    for regime in range(n_regimes):
                        # Get regime periods
                        regime_mask = onchain_analyzer.onchain_features['onchain_regime'] == regime
                        if not regime_mask.any():
                            continue
                            
                        # Find contiguous periods
                        regime_periods = []
                        regime_indices = onchain_analyzer.onchain_features.index[regime_mask]
                        
                        if len(regime_indices) > 0:
                            # Get regime description if available
                            regime_desc = ""
                            if hasattr(onchain_analyzer, 'regime_summary'):
                                if 'Regime Type' in onchain_analyzer.regime_summary.columns:
                                    regime_type = onchain_analyzer.regime_summary.loc[regime, 'Regime Type']
                                    if not pd.isna(regime_type):
                                        regime_desc = f": {regime_type}"
                                elif 'Ann. Return (%)' in onchain_analyzer.regime_summary.columns:
                                    ret = onchain_analyzer.regime_summary.loc[regime, 'Ann. Return (%)']
                                    if not pd.isna(ret):
                                        regime_desc = f": {ret:.1f}%"
                            
                            # Store for legend
                            regime_info.append(f"Regime {regime}{regime_desc}")
                            
                            # Plot price during this regime
                            common_indices = regime_indices.intersection(price_data.index)
                            if len(common_indices) > 0:
                                plt.plot(common_indices, price_data.loc[common_indices], 
                                        color=colors[regime], label=f'Regime {regime}{regime_desc}', linewidth=2)
                    
                    plt.title('On-Chain Market Regimes with Validation')
                    plt.xlabel('Date')
                    plt.ylabel('Price')
                    plt.legend()
                    plt.grid(True)
                else:
                    # Plot regime distribution if price not available
                    regime_counts = onchain_analyzer.onchain_features['onchain_regime'].value_counts()
                    plt.bar(regime_counts.index, regime_counts.values, color=colors)
                    plt.title('On-Chain Regime Distribution')
                    plt.xlabel('Regime')
                    plt.ylabel('Frequency')
                    plt.xticks(regime_counts.index)
                    plt.grid(True, axis='y')
            
            # 3. On-chain signal performance visualization
            if hasattr(onchain_analyzer, 'model_comparisons') and onchain_analyzer.model_comparisons is not None:
                plot_count += 1
                plt.subplot(5, 1, plot_count)
                
                # Extract performance metrics for plotting
                model_comp = onchain_analyzer.model_comparisons
                
                # Plot performance comparison
                if 'total_return' in model_comp.columns:
                    returns = model_comp['total_return'].sort_values(ascending=False)
                    
                    # Create bar chart
                    bars = plt.bar(returns.index, returns.values * 100)
                    
                    # Color coding for returns
                    for i, bar in enumerate(bars):
                        if returns.values[i] > 0:
                            bar.set_color('green')
                        else:
                            bar.set_color('red')
                    
                    plt.title('Performance Comparison of On-Chain Signal Components')
                    plt.xlabel('Signal Component')
                    plt.ylabel('Total Return (%)')
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, axis='y')
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{height:.1f}%', ha='center', va='bottom', rotation=0)
            
            # 4. Key on-chain metrics visualization with validation
            if onchain_analyzer.onchain_features is not None:
                # Select key metrics to plot based on validation
                key_metrics = []
                
                # Use model performance to select top metrics
                if hasattr(onchain_analyzer, 'model_comparisons') and onchain_analyzer.model_comparisons is not None:
                    # Get top performing signal components
                    top_signals = onchain_analyzer.model_comparisons.index[:5]
                    for signal in top_signals:
                        if signal.endswith('_signal'):
                            # Extract base feature from signal name
                            base_feature = signal.replace('_signal', '')
                            if base_feature in onchain_analyzer.onchain_features.columns:
                                key_metrics.append(base_feature)
                
                # Add standard metrics if needed
                standard_metrics = ['mvrv_ratio', 'nvt_ratio', 'hodl_ratio', 'onchain_sentiment', 'cycle_phase']
                for metric in standard_metrics:
                    if metric not in key_metrics and metric in onchain_analyzer.onchain_features.columns:
                        key_metrics.append(metric)
                
                # Limit to top 5 metrics
                key_metrics = key_metrics[:5]
                
                if key_metrics:
                    plot_count += 1
                    plt.subplot(5, 1, plot_count)
                    
                    # Plot each metric with normalization for comparison
                    for metric in key_metrics:
                        # Get the data and normalize for visual comparison
                        data = onchain_analyzer.onchain_features[metric]
                        normalized = (data - data.min()) / (data.max() - data.min())
                        plt.plot(onchain_analyzer.onchain_features.index, normalized, label=metric)
                    
                    plt.title('Key Validated On-Chain Metrics (Normalized)')
                    plt.xlabel('Date')
                    plt.ylabel('Normalized Value')
                    plt.legend()
                    plt.grid(True)
            
            # 5. Validation metrics summary
            plot_count += 1
            plt.subplot(5, 1, plot_count)
            
            # Create validation summary
            validation_metrics = []
            
            # CNN model performance
            if 'cnn' in onchain_analyzer.validation_metrics:
                cnn_metrics = onchain_analyzer.validation_metrics['cnn']
                if 'accuracy' in cnn_metrics:
                    validation_metrics.append(('CNN Accuracy', cnn_metrics['accuracy'] * 100))
                if 'auc' in cnn_metrics:
                    validation_metrics.append(('CNN AUC', cnn_metrics['auc'] * 100))
                if 'precision' in cnn_metrics:
                    validation_metrics.append(('CNN Precision', cnn_metrics['precision'] * 100))
                if 'recall' in cnn_metrics:
                    validation_metrics.append(('CNN Recall', cnn_metrics['recall'] * 100))
            
            # Regime model performance
            if hasattr(onchain_analyzer, 'regime_summary'):
                if 'Sharpe Ratio' in onchain_analyzer.regime_summary.columns:
                    # Calculate average positive regime Sharpe
                    positive_sharpes = onchain_analyzer.regime_summary['Sharpe Ratio'][onchain_analyzer.regime_summary['Sharpe Ratio'] > 0]
                    if len(positive_sharpes) > 0:
                        validation_metrics.append(('Avg Pos. Regime Sharpe', positive_sharpes.mean()))
            
            # Rule-based signals
            if 'rules' in onchain_analyzer.validation_metrics:
                rule_metrics = onchain_analyzer.validation_metrics['rules']
                if ('rules_applied' in rule_metrics and 'rules_checked' in rule_metrics and 
                    rule_metrics['rules_checked'] > 0):  # Added check for non-zero denominator
                    validation_metrics.append(('Rule Success Rate', 
                                             rule_metrics['rules_applied'] / rule_metrics['rules_checked'] * 100))
            
            # Enhanced vs original performance
            if hasattr(strategy, 'performance') and hasattr(strategy, 'signals'):
                if 'original_signal' in strategy.signals.columns and 'enhanced_signal' in strategy.signals.columns:
                    # Calculate agreement rate
                    agreement = (strategy.signals['original_signal'] == strategy.signals['enhanced_signal']).mean() * 100
                    validation_metrics.append(('Signal Agreement Rate', agreement))
            
            # Plot validation metrics as horizontal bar chart
            if validation_metrics:
                metrics = [m[0] for m in validation_metrics]
                values = [m[1] for m in validation_metrics]
                
                # Create horizontal bar chart
                bars = plt.barh(metrics, values)
                
                # Color based on values
                for i, bar in enumerate(bars):
                    if values[i] > 80:  # Excellent
                        bar.set_color('darkgreen')
                    elif values[i] > 60:  # Good
                        bar.set_color('green')
                    elif values[i] > 40:  # Fair
                        bar.set_color('lightgreen')
                    else:  # Poor
                        bar.set_color('orange')
                
                plt.title('On-Chain Analysis Validation Metrics')
                plt.xlabel('Score')
                plt.grid(True, axis='x')
                
                # Add value labels
                for i, v in enumerate(values):
                    plt.text(v + 0.5, i, f'{v:.1f}', va='center')
                
                # Set reasonable x-axis limit
                plt.xlim(0, max(values) * 1.1)
            
            plt.tight_layout()
            plt.show()
    
    # Replace visualization method with enhanced version
    strategy.visualize_results = enhanced_visualize
    
    return onchain_analyzer


# Main execution flow with validated on-chain analysis
def run_integrated_strategy(data_dir=None, ticker=None, start_date=None, end_date=None, 
                           use_backtrader=False, initial_cash=100000, use_sample_data=True,
                           use_enhanced_onchain=True):
    """
    Execute the integrated trading strategy with validated on-chain analysis.
    
    Args:
        data_dir (str): Directory containing on-chain data CSV files
        ticker (str): Ticker symbol (e.g., 'BTC-USD') for loading market data via yfinance
        start_date (datetime): Start date for market data
        end_date (datetime): End date for market data
        use_backtrader (bool): Whether to use backtrader for backtesting
        initial_cash (float): Initial capital for backtesting
        use_sample_data (bool): Whether to use sample data if no data source is provided
        use_enhanced_onchain (bool): Whether to use enhanced on-chain analysis
        
    Returns:
        strategy: Trained strategy object
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    
    # Initialize strategy
    strategy = IntegratedBitcoinStrategy()
    
    # Data processing
    data_loaded = strategy.load_data(data_dir=data_dir, ticker=ticker, 
                                    start_date=start_date, end_date=end_date,
                                    use_sample_data=use_sample_data)
    
    if not data_loaded:
        print("Error: No data loaded. Exiting.")
        return None
    
    data_merged = strategy.merge_data()
    if not data_merged:
        print("Error: Could not merge data. Exiting.")
        return None
    
    strategy.create_features()
    
    if strategy.features is None:
        print("Error: Failed to create features. Exiting.")
        return None
    
    print("\n" + "="*50)
    print("Data successfully loaded and processed.")
    print(f"Timeframe: {strategy.features.index[0]} to {strategy.features.index[-1]}")
    print(f"Number of samples: {len(strategy.features)}")
    print("="*50 + "\n")
    
    # Add enhanced on-chain analysis if requested
    onchain_analyzer = None
    if use_enhanced_onchain:
        print("\n" + "="*50)
        print("Initializing Enhanced On-Chain Analysis with Validation...")
        print("="*50 + "\n")
        onchain_analyzer = add_enhanced_onchain_analysis(strategy)
    
    # Train market regime model
    strategy.train_market_regime_model(n_regimes=4)
    
    # Train CNN model for pattern recognition
    strategy.train_cnn_model(lookback_window=30)
    
    # Train ensemble models
    strategy.train_ensemble_model()
    
    # Generate signals and backtest
    strategy.generate_trading_signals()
    strategy.backtest_strategy(use_backtrader=use_backtrader, initial_cash=initial_cash)
    strategy.visualize_results()
    
    print("\n" + "="*50)
    print("Strategy Analysis Completed Successfully")
    print("="*50 + "\n")
    
    return strategy

if __name__ == "__main__":
    # Example usage with on-chain data
    on_chain_dir = "path/to/bitcoin/onchain/data"
    strategy = run_integrated_strategy(
        data_dir=on_chain_dir,
        start_date=datetime(2018, 1, 1),
        end_date=datetime(2023, 1, 1),
        use_enhanced_onchain=True
    )
    
    # Example usage with market data only (still with enhanced on-chain analysis)
    strategy = run_integrated_strategy(
        ticker="BTC-USD", 
        start_date=datetime(2018, 1, 1), 
        end_date=datetime.now(),
        use_backtrader=True
    )
    
    # Simple example with sample data
    strategy = run_integrated_strategy()