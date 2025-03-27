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
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
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

# Main execution flow
def run_integrated_strategy(data_dir=None, ticker=None, start_date=None, end_date=None, 
                           use_backtrader=False, initial_cash=100000, use_sample_data=True):
    """
    Execute the complete integrated trading strategy.
    
    Args:
        data_dir (str): Directory containing on-chain data CSV files
        ticker (str): Ticker symbol (e.g., 'BTC-USD') for loading market data via yfinance
        start_date (datetime): Start date for market data
        end_date (datetime): End date for market data
        use_backtrader (bool): Whether to use backtrader for backtesting
        initial_cash (float): Initial capital for backtesting
        use_sample_data (bool): Whether to use sample data if no data source is provided
        
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
    
    # Train market regime model first
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
    on_chain_dir = r"C:\Users\User\Downloads\archive"
    strategy = run_integrated_strategy(data_dir=on_chain_dir)
    
    # Example usage with market data only
    strategy = run_integrated_strategy(ticker="BTC-USD", start_date=datetime(2018, 1, 1), 
                                      end_date=datetime.now(), use_backtrader=True)
    
    # Simple example (will use example data)
    strategy = run_integrated_strategy()
