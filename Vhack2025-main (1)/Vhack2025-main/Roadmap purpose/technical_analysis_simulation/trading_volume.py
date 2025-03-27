import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import os
from pathlib import Path
import logging
import json
import itertools
from typing import Tuple, Optional, Dict, Any, List, Union
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
TICKER = "BTC-USD"
PERIOD = "7y"
OUTPUT_DIR = Path.home() / "Downloads" / "TradingSystem"
TRADING_CAPITAL = 10000  # Starting capital in USD

class TradingSystem:
    """Advanced trading system for cryptocurrency analysis and trading."""
    
    def __init__(self, ticker: str = TICKER, period: str = PERIOD, 
                capital: float = TRADING_CAPITAL, output_dir: Path = OUTPUT_DIR):
        """
        Initialize the trading system.
        
        Args:
            ticker: Asset ticker symbol
            period: Time period for analysis
            capital: Initial trading capital
            output_dir: Directory for output files
        """
        self.ticker = ticker
        self.period = period
        self.capital = capital
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.raw_data = None
        self.data = None
        self.strategy_results = {}
        self.best_strategy = None
        
        # Set up plot style
        self._setup_plot_style()
        
    def _setup_plot_style(self):
        """Configure the plot style for consistent visualization."""
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.figsize'] = (14, 7)
        plt.rcParams['font.size'] = 12
        
        # Support Chinese characters if needed
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Show all columns in pandas DataFrames
        pd.set_option('display.max_columns', None)
    
    def download_data(self):
        """Download financial data for the specified ticker and period."""
        try:
            logger.info(f"Downloading data for {self.ticker} over {self.period}")
            stock_data = yf.download(self.ticker, period=self.period, group_by='column')
            
            # Flatten MultiIndex columns for simpler access
            stock_data.columns = stock_data.columns.get_level_values(0)
            
            # Basic data validation
            if len(stock_data) == 0:
                raise ValueError(f"No data downloaded for {self.ticker}")
                
            logger.info(f"Successfully downloaded {len(stock_data)} rows of data")
            self.raw_data = stock_data
            self.data = stock_data.copy()
            
            # Save raw data
            self._save_data(self.raw_data, f"{self.ticker}_raw_data.csv")
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise
    
    def _save_data(self, data: pd.DataFrame, filename: str) -> Path:
        """
        Save DataFrame to CSV file.
        
        Args:
            data: DataFrame to save
            filename: Filename to save to
            
        Returns:
            Path to saved file
        """
        try:
            filepath = self.output_dir / filename
            data.to_csv(filepath)
            logger.info(f"Data saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise
            
    def load_data(self, filepath: Path) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with loaded data
        """
        try:
            logger.info(f"Loading data from {filepath}")
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            self.raw_data = data
            self.data = data.copy()
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_data(self):
        """Calculate all technical indicators and prepare data for analysis."""
        if self.data is None:
            raise ValueError("No data available. Please download or load data first.")
            
        logger.info("Calculating technical indicators...")
        
        # Basic metrics
        self._calculate_basic_metrics()
        
        # Price-based indicators
        self._calculate_moving_averages()
        self._calculate_rsi()
        self._calculate_macd()
        self._calculate_bollinger_bands()
        self._calculate_atr()
        self._calculate_adx()
        self._calculate_ichimoku_cloud()
        self._calculate_stochastic_oscillator()
        
        # Volume-based indicators
        self._calculate_obv()
        self._calculate_mfi()
        self._calculate_vwap()
        
        # Market regime indicators
        self._detect_volatility_regime()
        self._detect_trend_regime()
        
        # Support/Resistance levels
        self._calculate_pivot_points()
        
        # Save the enhanced data
        self._save_data(self.data, f"{self.ticker}_enhanced_data.csv")
        
        logger.info("Data preparation completed")
        return self.data
    
    def _calculate_basic_metrics(self):
        """Calculate basic price metrics."""
        self.data['pre_close'] = self.data['Close'].shift(1)
        self.data['change'] = self.data['Close'] - self.data['pre_close']
        self.data['pct_chg'] = (self.data['change'] / self.data['pre_close'] * 100).round(2)
        self.data['log_return'] = np.log(self.data['Close'] / self.data['pre_close'])
        self.data['amount'] = self.data['Volume'] * self.data['Close']
        self.data['ts'] = self.data.index.astype(np.int64) // 10**9
        self.data['code'] = self.ticker
    
    def _calculate_moving_averages(self):
        """Calculate various moving averages for price and volume."""
        # Price moving averages (short, medium, long term)
        for period in [5, 10, 20, 50, 100, 200]:
            self.data[f'ma{period}'] = self.data['Close'].rolling(window=period).mean()
        
        # Exponential moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            self.data[f'ema{period}'] = self.data['Close'].ewm(span=period, adjust=False).mean()
        
        # Volume moving averages
        for period in [5, 10, 20, 50]:
            self.data[f'vol_ma{period}'] = self.data['Volume'].rolling(window=period).mean()
        
        # Volume ratio to 5-day MA
        self.data['vol_ratio'] = self.data['Volume'] / self.data['vol_ma5']
        
        # MA cross signals
        self.data['ma_cross_20_50'] = np.where(
            self.data['ma20'] > self.data['ma50'], 1, -1
        )
        self.data['ma_cross_50_200'] = np.where(
            self.data['ma50'] > self.data['ma200'], 1, -1
        )
    
    def _calculate_rsi(self, periods: List[int] = [6, 14, 28]):
        """
        Calculate Relative Strength Index (RSI) for multiple periods.
        
        Args:
            periods: List of periods for RSI calculation
        """
        for period in periods:
            delta = self.data['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
            
            rs = avg_gain / avg_loss
            self.data[f'RSI{period}'] = 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
        """
        # Calculate MACD line
        fast_ema = self.data['Close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = self.data['Close'].ewm(span=slow_period, adjust=False).mean()
        self.data['MACD_line'] = fast_ema - slow_ema
        
        # Calculate signal line
        self.data['MACD_signal'] = self.data['MACD_line'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        self.data['MACD_hist'] = self.data['MACD_line'] - self.data['MACD_signal']
        
        # Calculate MACD crossover signal
        self.data['MACD_cross'] = np.where(
            self.data['MACD_line'] > self.data['MACD_signal'], 1, 
            np.where(self.data['MACD_line'] < self.data['MACD_signal'], -1, 0)
        )
    
    def _calculate_bollinger_bands(self, window: int = 20, num_std: float = 2.0):
        """
        Calculate Bollinger Bands.
        
        Args:
            window: Moving average window
            num_std: Number of standard deviations for the bands
        """
        rolling_mean = self.data['Close'].rolling(window=window).mean()
        rolling_std = self.data['Close'].rolling(window=window).std()
        
        self.data['BB_middle'] = rolling_mean
        self.data['BB_upper'] = rolling_mean + (rolling_std * num_std)
        self.data['BB_lower'] = rolling_mean - (rolling_std * num_std)
        
        # Calculate bandwidth and %B indicators
        self.data['BB_bandwidth'] = (self.data['BB_upper'] - self.data['BB_lower']) / self.data['BB_middle']
        self.data['BB_pct_b'] = (self.data['Close'] - self.data['BB_lower']) / (self.data['BB_upper'] - self.data['BB_lower'])
    
    def _calculate_atr(self, period: int = 14):
        """
        Calculate Average True Range (ATR).
        
        Args:
            period: Period for ATR calculation
        """
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        
        # Calculate true range
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        # Calculate ATR
        self.data['ATR'] = true_range.rolling(window=period).mean()
        
        # Normalized ATR (ATR as percentage of price)
        self.data['ATR_pct'] = self.data['ATR'] / self.data['Close'] * 100
    
    def _calculate_adx(self, period: int = 14):
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            period: Period for ADX calculation
        """
        # Calculate +DM and -DM
        high_diff = self.data['High'].diff()
        low_diff = self.data['Low'].diff()
        
        pos_dm = np.where(
            (high_diff > 0) & (high_diff > low_diff.abs()),
            high_diff,
            0
        )
        neg_dm = np.where(
            (low_diff < 0) & (low_diff.abs() > high_diff),
            low_diff.abs(),
            0
        )
        
        # True Range
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate smoothed values using Wilder's smoothing
        tr14 = true_range.rolling(window=period).sum()
        pos_dm14 = pd.Series(pos_dm).rolling(window=period).sum()
        neg_dm14 = pd.Series(neg_dm).rolling(window=period).sum()
        
        # Calculate +DI and -DI
        pdi14 = 100 * (pos_dm14 / tr14)
        ndi14 = 100 * (neg_dm14 / tr14)
        
        # Calculate DX
        dx = 100 * np.abs(pdi14 - ndi14) / (pdi14 + ndi14)
        
        # Calculate ADX
        self.data['ADX'] = dx.rolling(window=period).mean()
        self.data['+DI'] = pdi14
        self.data['-DI'] = ndi14
    
    def _calculate_ichimoku_cloud(self):
        """Calculate Ichimoku Cloud components."""
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = self.data['High'].rolling(window=9).max()
        period9_low = self.data['Low'].rolling(window=9).min()
        self.data['tenkan_sen'] = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = self.data['High'].rolling(window=26).max()
        period26_low = self.data['Low'].rolling(window=26).min()
        self.data['kijun_sen'] = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        self.data['senkou_span_a'] = ((self.data['tenkan_sen'] + self.data['kijun_sen']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        period52_high = self.data['High'].rolling(window=52).max()
        period52_low = self.data['Low'].rolling(window=52).min()
        self.data['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Current closing price, shifted backwards
        self.data['chikou_span'] = self.data['Close'].shift(-26)
        
        # Ichimoku signal: Above cloud (bullish), Below cloud (bearish)
        self.data['cloud_top'] = self.data[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        self.data['cloud_bottom'] = self.data[['senkou_span_a', 'senkou_span_b']].min(axis=1)
        self.data['above_cloud'] = np.where(self.data['Close'] > self.data['cloud_top'], 1, 0)
        self.data['below_cloud'] = np.where(self.data['Close'] < self.data['cloud_bottom'], 1, 0)
        self.data['in_cloud'] = np.where(
            (self.data['Close'] <= self.data['cloud_top']) & 
            (self.data['Close'] >= self.data['cloud_bottom']), 
            1, 0
        )
    
    def _calculate_stochastic_oscillator(self, k_period: int = 14, d_period: int = 3):
        """
        Calculate Stochastic Oscillator.
        
        Args:
            k_period: Period for %K calculation
            d_period: Period for %D calculation
        """
        # Calculate %K
        low_min = self.data['Low'].rolling(window=k_period).min()
        high_max = self.data['High'].rolling(window=k_period).max()
        
        self.data['stoch_%K'] = 100 * ((self.data['Close'] - low_min) / (high_max - low_min))
        
        # Calculate %D (3-day SMA of %K)
        self.data['stoch_%D'] = self.data['stoch_%K'].rolling(window=d_period).mean()
        
        # Stochastic crossover signal
        self.data['stoch_cross'] = np.where(
            self.data['stoch_%K'] > self.data['stoch_%D'], 1, 
            np.where(self.data['stoch_%K'] < self.data['stoch_%D'], -1, 0)
        )
    
    def _calculate_obv(self):
        """Calculate On-Balance Volume (OBV)."""
        # Initialize OBV column
        self.data['OBV'] = 0
        
        # Loop-based calculation for OBV
        for i in range(1, len(self.data)):
            if self.data['Close'].iloc[i] > self.data['Close'].iloc[i-1]:
                self.data.loc[self.data.index[i], 'OBV'] = (
                    self.data.loc[self.data.index[i-1], 'OBV'] + 
                    self.data.loc[self.data.index[i], 'Volume']
                )
            elif self.data['Close'].iloc[i] < self.data['Close'].iloc[i-1]:
                self.data.loc[self.data.index[i], 'OBV'] = (
                    self.data.loc[self.data.index[i-1], 'OBV'] - 
                    self.data.loc[self.data.index[i], 'Volume']
                )
            else:
                self.data.loc[self.data.index[i], 'OBV'] = self.data.loc[self.data.index[i-1], 'OBV']
                
        # OBV 20-day EMA for signal line
        self.data['OBV_ema'] = self.data['OBV'].ewm(span=20, adjust=False).mean()
        
        # OBV signal
        self.data['OBV_signal'] = np.where(
            self.data['OBV'] > self.data['OBV_ema'], 1, 
            np.where(self.data['OBV'] < self.data['OBV_ema'], -1, 0)
        )
    
    def _calculate_mfi(self, period: int = 14):
        """
        Calculate Money Flow Index (MFI).
        
        Args:
            period: Period for MFI calculation
        """
        # Calculate typical price
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        
        # Calculate money flow
        money_flow = typical_price * self.data['Volume']
        
        # Initialize positive and negative flow columns
        positive_flow = pd.Series(0, index=self.data.index)
        negative_flow = pd.Series(0, index=self.data.index)
        
        # Calculate positive and negative money flow
        for i in range(1, len(self.data)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                negative_flow.iloc[i] = money_flow.iloc[i]
        
        # Calculate sums over the period
        positive_flow_sum = positive_flow.rolling(window=period).sum()
        negative_flow_sum = negative_flow.rolling(window=period).sum()
        
        # Calculate money ratio, avoiding division by zero
        money_ratio = np.where(
            negative_flow_sum > 0,
            positive_flow_sum / negative_flow_sum,
            100  # Default value when negative flow sum is zero
        )
        
        # Calculate MFI
        self.data['MFI'] = 100 - (100 / (1 + money_ratio))
    
    def _calculate_vwap(self, reset_period: str = 'D'):
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            reset_period: Period to reset VWAP calculation ('D' for daily)
        """
        # Create a group key for the reset period
        self.data['group'] = self.data.index.to_period(reset_period)
        
        # Calculate VWAP for each period
        self.data['vwap_numerator'] = self.data['Close'] * self.data['Volume']
        self.data['vwap_denominator'] = self.data['Volume']
        
        # Group by the reset period and calculate cumulative sums
        cum_num = self.data.groupby('group')['vwap_numerator'].cumsum()
        cum_den = self.data.groupby('group')['vwap_denominator'].cumsum()
        
        # Calculate VWAP and avoid division by zero
        self.data['VWAP'] = np.where(cum_den != 0, cum_num / cum_den, np.nan)
        
        # Calculate VWAP signal: Above VWAP (bullish), Below VWAP (bearish)
        self.data['VWAP_signal'] = np.where(
            self.data['Close'] > self.data['VWAP'], 1, 
            np.where(self.data['Close'] < self.data['VWAP'], -1, 0)
        )
        
        # Clean up intermediate columns
        self.data = self.data.drop(['group', 'vwap_numerator', 'vwap_denominator'], axis=1)
    
    def _detect_volatility_regime(self, window: int = 20):
        """
        Detect market volatility regime.
        
        Args:
            window: Window for volatility calculation
        """
        # Calculate historical volatility
        self.data['hist_vol'] = self.data['log_return'].rolling(window=window).std() * np.sqrt(252)
        
        # Calculate average volatility and volatility of volatility
        avg_vol = self.data['hist_vol'].rolling(window=252).mean().shift(1)
        vol_of_vol = self.data['hist_vol'].rolling(window=252).std().shift(1)
        
        # Define volatility regimes
        self.data['vol_regime'] = np.where(
            self.data['hist_vol'] > avg_vol + vol_of_vol, 'high',
            np.where(self.data['hist_vol'] < avg_vol - vol_of_vol, 'low', 'normal')
        )
        
        # Convert regime to numeric indicators for strategy use
        self.data['high_vol_regime'] = np.where(self.data['vol_regime'] == 'high', 1, 0)
        self.data['normal_vol_regime'] = np.where(self.data['vol_regime'] == 'normal', 1, 0)
        self.data['low_vol_regime'] = np.where(self.data['vol_regime'] == 'low', 1, 0)
    
    def _detect_trend_regime(self, short_window: int = 20, long_window: int = 100):
        """
        Detect market trend regime.
        
        Args:
            short_window: Short-term window for trend detection
            long_window: Long-term window for trend detection
        """
        # Calculate price slopes
        short_slope = self._calculate_slope(self.data['Close'], short_window)
        long_slope = self._calculate_slope(self.data['Close'], long_window)
        
        # Define trend regimes
        self.data['short_trend'] = np.where(short_slope > 0, 'up', 'down')
        self.data['long_trend'] = np.where(long_slope > 0, 'up', 'down')
        
        # Combine trends
        self.data['trend_regime'] = np.where(
            (self.data['short_trend'] == 'up') & (self.data['long_trend'] == 'up'), 'strong_up',
            np.where((self.data['short_trend'] == 'down') & (self.data['long_trend'] == 'down'), 'strong_down',
            np.where((self.data['short_trend'] == 'up') & (self.data['long_trend'] == 'down'), 'weak_up', 'weak_down'))
        )
        
        # Convert regime to numeric indicators for strategy use
        self.data['strong_up_trend'] = np.where(self.data['trend_regime'] == 'strong_up', 1, 0)
        self.data['weak_up_trend'] = np.where(self.data['trend_regime'] == 'weak_up', 1, 0)
        self.data['weak_down_trend'] = np.where(self.data['trend_regime'] == 'weak_down', 1, 0)
        self.data['strong_down_trend'] = np.where(self.data['trend_regime'] == 'strong_down', 1, 0)
    
    def _calculate_slope(self, series: pd.Series, window: int) -> pd.Series:
        """
        Calculate the slope of a time series over a rolling window.
        
        Args:
            series: Time series data
            window: Window size for slope calculation
            
        Returns:
            Series with slope values
        """
        slopes = pd.Series(index=series.index, dtype=float)
        
        for i in range(window - 1, len(series)):
            y = series.iloc[i-window+1:i+1].values
            x = np.arange(window)
            slope, _, _, _, _ = stats.linregress(x, y)
            slopes.iloc[i] = slope
            
        return slopes
    
    def _calculate_pivot_points(self, method: str = 'standard'):
        """
        Calculate pivot points for support and resistance levels.
        
        Args:
            method: Method for pivot point calculation ('standard', 'fibonacci', 'camarilla')
        """
        # Get previous day's high, low, and close
        prev_high = self.data['High'].shift(1)
        prev_low = self.data['Low'].shift(1)
        prev_close = self.data['Close'].shift(1)
        
        if method == 'standard':
            # Calculate pivot point
            pivot = (prev_high + prev_low + prev_close) / 3
            
            # Calculate support levels
            s1 = 2 * pivot - prev_high
            s2 = pivot - (prev_high - prev_low)
            s3 = s1 - (prev_high - prev_low)
            
            # Calculate resistance levels
            r1 = 2 * pivot - prev_low
            r2 = pivot + (prev_high - prev_low)
            r3 = r1 + (prev_high - prev_low)
            
        elif method == 'fibonacci':
            # Calculate pivot point
            pivot = (prev_high + prev_low + prev_close) / 3
            
            # Calculate support levels
            s1 = pivot - 0.382 * (prev_high - prev_low)
            s2 = pivot - 0.618 * (prev_high - prev_low)
            s3 = pivot - (prev_high - prev_low)
            
            # Calculate resistance levels
            r1 = pivot + 0.382 * (prev_high - prev_low)
            r2 = pivot + 0.618 * (prev_high - prev_low)
            r3 = pivot + (prev_high - prev_low)
            
        elif method == 'camarilla':
            # Calculate support levels
            s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
            s2 = prev_close - (prev_high - prev_low) * 1.1 / 6
            s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
            
            # Calculate resistance levels
            r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
            r2 = prev_close + (prev_high - prev_low) * 1.1 / 6
            r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
            
            # Set pivot to previous close for consistency
            pivot = prev_close
        
        # Store pivot points in DataFrame
        self.data['pivot'] = pivot
        self.data['s1'] = s1
        self.data['s2'] = s2
        self.data['s3'] = s3
        self.data['r1'] = r1
        self.data['r2'] = r2
        self.data['r3'] = r3
        
        # Calculate proximity to support/resistance (for strategy use)
        self.data['near_support'] = np.where(
            (self.data['Close'] >= self.data['s1'] * 0.99) & 
            (self.data['Close'] <= self.data['s1'] * 1.01), 1, 0
        )
        self.data['near_resistance'] = np.where(
            (self.data['Close'] >= self.data['r1'] * 0.99) & 
            (self.data['Close'] <= self.data['r1'] * 1.01), 1, 0
        )
    
    def run_strategy(self, strategy_name: str, **params):
        """
        Run a specified trading strategy with the given parameters.
        
        Args:
            strategy_name: Name of the strategy to run
            **params: Strategy-specific parameters
        
        Returns:
            DataFrame with strategy results
        """
        if self.data is None:
            raise ValueError("No data available. Please prepare data first.")
            
        strategy_functions = {
            'ma_crossover': self._run_ma_crossover_strategy,
            'macd': self._run_macd_strategy,
            'rsi': self._run_rsi_strategy,
            'bollinger': self._run_bollinger_strategy,
            'ichimoku': self._run_ichimoku_strategy,
            'volume': self._run_volume_strategy,
            'combined': self._run_combined_strategy,
            'ml': self._run_ml_strategy,
            'breakout': self._run_breakout_strategy,
            'pivot': self._run_pivot_strategy,
            'mean_reversion': self._run_mean_reversion_strategy
        }
        
        if strategy_name not in strategy_functions:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        logger.info(f"Running {strategy_name} strategy with parameters: {params}")
        
        # Run the selected strategy
        strategy_data = strategy_functions[strategy_name](**params)
        
        # Calculate and store performance metrics
        metrics = self._calculate_strategy_metrics(strategy_data)
        
        # Save the strategy results
        self.strategy_results[strategy_name] = {
            'data': strategy_data,
            'metrics': metrics,
            'params': params
        }
        
        # Save strategy data to CSV
        self._save_data(strategy_data, f"{self.ticker}_{strategy_name}_strategy.csv")
        
        return strategy_data, metrics
    
    def _run_ma_crossover_strategy(self, fast_ma: str = 'ma20', slow_ma: str = 'ma50', 
                                  transaction_cost: float = 0.001, stop_loss: float = 0.05,
                                  trailing_stop: bool = False):
        """
        Implement a moving average crossover trading strategy.
        
        Args:
            fast_ma: Column name of the fast moving average
            slow_ma: Column name of the slow moving average
            transaction_cost: Transaction cost as a decimal
            stop_loss: Stop loss level as a decimal
            trailing_stop: Whether to use trailing stop loss
            
        Returns:
            DataFrame with strategy performance metrics
        """
        df = self.data.copy()
        
        # Generate signals: 1 for buy (fast_ma > slow_ma), -1 for sell (fast_ma < slow_ma)
        df['Signal'] = 0
        df.loc[df[fast_ma] > df[slow_ma], 'Signal'] = 1
        df.loc[df[fast_ma] < df[slow_ma], 'Signal'] = -1
        
        # Delay the signal by 1 day to simulate trading at the next day's open
        df['Trade_Signal'] = df['Signal'].shift(1)
        
        # Add features for risk management
        self._apply_risk_management(df, 'Trade_Signal', stop_loss, trailing_stop)
        
        # Calculate daily market returns (percentage change in close price)
        df['Market_Return'] = df['Close'].pct_change()
        
        # Calculate position changes (for transaction costs)
        df['Position_Change'] = df['Trade_Signal'].diff().abs()
        
        # Calculate strategy returns with transaction costs
        df['Strategy_Return'] = df['Trade_Signal'] * df['Market_Return'] - transaction_cost * df['Position_Change']
        
        # Calculate cumulative returns
        df['Cumulative_Market'] = (1 + df['Market_Return'].fillna(0)).cumprod()
        df['Cumulative_Strategy'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
        
        # For plotting: track when trades occur (signal changes)
        df['Trade_Change'] = df['Trade_Signal'].diff()
        
        # Calculate equity curve and drawdown
        df['Equity'] = self.capital * df['Cumulative_Strategy']
        df['Peak'] = df['Equity'].cummax()
        df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
        
        # Strategy-specific characteristics
        df['Strategy'] = 'MA Crossover'
        df['Strategy_Params'] = f"Fast: {fast_ma}, Slow: {slow_ma}"
        
        return df
    
    def _run_macd_strategy(self, fast_period: int = 12, slow_period: int = 26, 
                          signal_period: int = 9, transaction_cost: float = 0.001,
                          stop_loss: float = 0.05, trailing_stop: bool = False):
        """
        Implement a MACD trading strategy.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            transaction_cost: Transaction cost as a decimal
            stop_loss: Stop loss level as a decimal
            trailing_stop: Whether to use trailing stop loss
            
        Returns:
            DataFrame with strategy performance metrics
        """
        df = self.data.copy()
        
        # Ensure MACD is calculated
        if 'MACD_cross' not in df.columns:
            # Calculate MACD
            fast_ema = df['Close'].ewm(span=fast_period, adjust=False).mean()
            slow_ema = df['Close'].ewm(span=slow_period, adjust=False).mean()
            df['MACD_line'] = fast_ema - slow_ema
            df['MACD_signal'] = df['MACD_line'].ewm(span=signal_period, adjust=False).mean()
            df['MACD_hist'] = df['MACD_line'] - df['MACD_signal']
            df['MACD_cross'] = np.where(
                df['MACD_line'] > df['MACD_signal'], 1, 
                np.where(df['MACD_line'] < df['MACD_signal'], -1, 0)
            )
        
        # Generate trade signals from MACD crossover
        df['Trade_Signal'] = df['MACD_cross'].shift(1)
        
        # Add features for risk management
        self._apply_risk_management(df, 'Trade_Signal', stop_loss, trailing_stop)
        
        # Calculate daily market returns (percentage change in close price)
        df['Market_Return'] = df['Close'].pct_change()
        
        # Calculate position changes (for transaction costs)
        df['Position_Change'] = df['Trade_Signal'].diff().abs()
        
        # Calculate strategy returns with transaction costs
        df['Strategy_Return'] = df['Trade_Signal'] * df['Market_Return'] - transaction_cost * df['Position_Change']
        
        # Calculate cumulative returns
        df['Cumulative_Market'] = (1 + df['Market_Return'].fillna(0)).cumprod()
        df['Cumulative_Strategy'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
        
        # For plotting: track when trades occur (signal changes)
        df['Trade_Change'] = df['Trade_Signal'].diff()
        
        # Calculate equity curve and drawdown
        df['Equity'] = self.capital * df['Cumulative_Strategy']
        df['Peak'] = df['Equity'].cummax()
        df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
        
        # Strategy-specific characteristics
        df['Strategy'] = 'MACD'
        df['Strategy_Params'] = f"Fast: {fast_period}, Slow: {slow_period}, Signal: {signal_period}"
        
        return df
    
    def _run_rsi_strategy(self, rsi_period: int = 14, overbought: int = 70, 
                         oversold: int = 30, transaction_cost: float = 0.001,
                         stop_loss: float = 0.05, trailing_stop: bool = False):
        """
        Implement an RSI trading strategy.
        
        Args:
            rsi_period: Period for RSI calculation
            overbought: Overbought threshold
            oversold: Oversold threshold
            transaction_cost: Transaction cost as a decimal
            stop_loss: Stop loss level as a decimal
            trailing_stop: Whether to use trailing stop loss
            
        Returns:
            DataFrame with strategy performance metrics
        """
        df = self.data.copy()
        
        # Ensure RSI is calculated
        rsi_col = f'RSI{rsi_period}'
        if rsi_col not in df.columns:
            self._calculate_rsi([rsi_period])
            df = self.data.copy()
        
        # Generate signals: 1 for buy (RSI < oversold), -1 for sell (RSI > overbought)
        df['RSI_Signal'] = 0
        
        # Entry signals
        df.loc[df[rsi_col] < oversold, 'RSI_Signal'] = 1  # Buy when oversold
        df.loc[df[rsi_col] > overbought, 'RSI_Signal'] = -1  # Sell when overbought
        
        # Apply trend filter (optional)
        trend_filter = df['ma50'] > df['ma200']  # Only buy in uptrend
        df.loc[~trend_filter & (df['RSI_Signal'] == 1), 'RSI_Signal'] = 0  # Cancel buy signals in downtrend
        
        # Implement position management
        df['Position'] = 0
        position = 0
        
        for i in range(1, len(df)):
            if df['RSI_Signal'].iloc[i] == 1 and position <= 0:
                position = 1  # Enter long
            elif df['RSI_Signal'].iloc[i] == -1 and position >= 0:
                position = -1  # Enter short
            # If no signal, maintain previous position
            df.loc[df.index[i], 'Position'] = position
        
        # Delay the signal by 1 day to simulate trading at the next day's open
        df['Trade_Signal'] = df['Position'].shift(1)
        
        # Add features for risk management
        self._apply_risk_management(df, 'Trade_Signal', stop_loss, trailing_stop)
        
        # Calculate daily market returns (percentage change in close price)
        df['Market_Return'] = df['Close'].pct_change()
        
        # Calculate position changes (for transaction costs)
        df['Position_Change'] = df['Trade_Signal'].diff().abs()
        
        # Calculate strategy returns with transaction costs
        df['Strategy_Return'] = df['Trade_Signal'] * df['Market_Return'] - transaction_cost * df['Position_Change']
        
        # Calculate cumulative returns
        df['Cumulative_Market'] = (1 + df['Market_Return'].fillna(0)).cumprod()
        df['Cumulative_Strategy'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
        
        # For plotting: track when trades occur (signal changes)
        df['Trade_Change'] = df['Trade_Signal'].diff()
        
        # Calculate equity curve and drawdown
        df['Equity'] = self.capital * df['Cumulative_Strategy']
        df['Peak'] = df['Equity'].cummax()
        df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
        
        # Strategy-specific characteristics
        df['Strategy'] = 'RSI'
        df['Strategy_Params'] = f"Period: {rsi_period}, Overbought: {overbought}, Oversold: {oversold}"
        
        return df
    
    def _run_bollinger_strategy(self, window: int = 20, num_std: float = 2.0,
                               transaction_cost: float = 0.001, stop_loss: float = 0.05,
                               trailing_stop: bool = False):
        """
        Implement a Bollinger Bands trading strategy.
        
        Args:
            window: Moving average window
            num_std: Number of standard deviations for the bands
            transaction_cost: Transaction cost as a decimal
            stop_loss: Stop loss level as a decimal
            trailing_stop: Whether to use trailing stop loss
            
        Returns:
            DataFrame with strategy performance metrics
        """
        df = self.data.copy()
        
        # Ensure Bollinger Bands are calculated
        if 'BB_lower' not in df.columns:
            self._calculate_bollinger_bands(window, num_std)
            df = self.data.copy()
        
        # Generate signals
        df['BB_Signal'] = 0
        
        # Buy when price touches lower band and sell when it touches upper band
        df.loc[df['Close'] <= df['BB_lower'], 'BB_Signal'] = 1
        df.loc[df['Close'] >= df['BB_upper'], 'BB_Signal'] = -1
        
        # Apply additional filters based on trend
        # Only take buy signals in uptrend and sell signals in downtrend
        df.loc[(df['BB_Signal'] == 1) & (df['Close'] < df['ma50']), 'BB_Signal'] = 0
        df.loc[(df['BB_Signal'] == -1) & (df['Close'] > df['ma50']), 'BB_Signal'] = 0
        
        # Position management
        df['Position'] = 0
        position = 0
        
        for i in range(1, len(df)):
            if df['BB_Signal'].iloc[i] == 1 and position <= 0:
                position = 1  # Enter long
            elif df['BB_Signal'].iloc[i] == -1 and position >= 0:
                position = -1  # Enter short
            # If no signal, maintain previous position
            df.loc[df.index[i], 'Position'] = position
        
        # Delay the signal by 1 day to simulate trading at the next day's open
        df['Trade_Signal'] = df['Position'].shift(1)
        
        # Add features for risk management
        self._apply_risk_management(df, 'Trade_Signal', stop_loss, trailing_stop)
        
        # Calculate daily market returns (percentage change in close price)
        df['Market_Return'] = df['Close'].pct_change()
        
        # Calculate position changes (for transaction costs)
        df['Position_Change'] = df['Trade_Signal'].diff().abs()
        
        # Calculate strategy returns with transaction costs
        df['Strategy_Return'] = df['Trade_Signal'] * df['Market_Return'] - transaction_cost * df['Position_Change']
        
        # Calculate cumulative returns
        df['Cumulative_Market'] = (1 + df['Market_Return'].fillna(0)).cumprod()
        df['Cumulative_Strategy'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
        
        # For plotting: track when trades occur (signal changes)
        df['Trade_Change'] = df['Trade_Signal'].diff()
        
        # Calculate equity curve and drawdown
        df['Equity'] = self.capital * df['Cumulative_Strategy']
        df['Peak'] = df['Equity'].cummax()
        df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
        
        # Strategy-specific characteristics
        df['Strategy'] = 'Bollinger Bands'
        df['Strategy_Params'] = f"Window: {window}, StdDev: {num_std}"
        
        return df
    
    def _run_ichimoku_strategy(self, transaction_cost: float = 0.001,
                              stop_loss: float = 0.05, trailing_stop: bool = False):
        """
        Implement an Ichimoku Cloud trading strategy.
        
        Args:
            transaction_cost: Transaction cost as a decimal
            stop_loss: Stop loss level as a decimal
            trailing_stop: Whether to use trailing stop loss
            
        Returns:
            DataFrame with strategy performance metrics
        """
        df = self.data.copy()
        
        # Ensure Ichimoku components are calculated
        if 'tenkan_sen' not in df.columns:
            self._calculate_ichimoku_cloud()
            df = self.data.copy()
        
        # Generate signals based on multiple Ichimoku components
        df['Ichimoku_Signal'] = 0
        
        # TK Cross (Tenkan-sen crosses above Kijun-sen)
        tk_cross_up = (df['tenkan_sen'] > df['kijun_sen']) & (df['tenkan_sen'].shift(1) <= df['kijun_sen'].shift(1))
        tk_cross_down = (df['tenkan_sen'] < df['kijun_sen']) & (df['tenkan_sen'].shift(1) >= df['kijun_sen'].shift(1))
        
        # Price above/below the cloud
        price_above_cloud = df['Close'] > df['cloud_top']
        price_below_cloud = df['Close'] < df['cloud_bottom']
        
        # Chikou span confirming the trend
        chikou_confirming_uptrend = df['chikou_span'].shift(26) > df['Close'].shift(52)
        chikou_confirming_downtrend = df['chikou_span'].shift(26) < df['Close'].shift(52)
        
        # Define buy and sell conditions
        buy_condition = tk_cross_up & price_above_cloud & chikou_confirming_uptrend
        sell_condition = tk_cross_down & price_below_cloud & chikou_confirming_downtrend
        
        df.loc[buy_condition, 'Ichimoku_Signal'] = 1
        df.loc[sell_condition, 'Ichimoku_Signal'] = -1
        
        # Position management
        df['Position'] = 0
        position = 0
        
        for i in range(1, len(df)):
            if df['Ichimoku_Signal'].iloc[i] == 1 and position <= 0:
                position = 1  # Enter long
            elif df['Ichimoku_Signal'].iloc[i] == -1 and position >= 0:
                position = -1  # Enter short
            # If no signal, maintain previous position
            df.loc[df.index[i], 'Position'] = position
        
        # Delay the signal by 1 day to simulate trading at the next day's open
        df['Trade_Signal'] = df['Position'].shift(1)
        
        # Add features for risk management
        self._apply_risk_management(df, 'Trade_Signal', stop_loss, trailing_stop)
        
        # Calculate daily market returns (percentage change in close price)
        df['Market_Return'] = df['Close'].pct_change()
        
        # Calculate position changes (for transaction costs)
        df['Position_Change'] = df['Trade_Signal'].diff().abs()
        
        # Calculate strategy returns with transaction costs
        df['Strategy_Return'] = df['Trade_Signal'] * df['Market_Return'] - transaction_cost * df['Position_Change']
        
        # Calculate cumulative returns
        df['Cumulative_Market'] = (1 + df['Market_Return'].fillna(0)).cumprod()
        df['Cumulative_Strategy'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
        
        # For plotting: track when trades occur (signal changes)
        df['Trade_Change'] = df['Trade_Signal'].diff()
        
        # Calculate equity curve and drawdown
        df['Equity'] = self.capital * df['Cumulative_Strategy']
        df['Peak'] = df['Equity'].cummax()
        df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
        
        # Strategy-specific characteristics
        df['Strategy'] = 'Ichimoku'
        df['Strategy_Params'] = "Standard Ichimoku Settings"
        
        return df
    
    def _run_volume_strategy(self, obv_window: int = 20, vwap_filter: bool = True,
                            transaction_cost: float = 0.001, stop_loss: float = 0.05,
                            trailing_stop: bool = False):
        """
        Implement a volume-based trading strategy.
        
        Args:
            obv_window: Window for OBV EMA
            vwap_filter: Whether to use VWAP as a filter
            transaction_cost: Transaction cost as a decimal
            stop_loss: Stop loss level as a decimal
            trailing_stop: Whether to use trailing stop loss
            
        Returns:
            DataFrame with strategy performance metrics
        """
        df = self.data.copy()
        
        # Ensure volume indicators are calculated
        if 'OBV' not in df.columns:
            self._calculate_obv()
        if 'VWAP' not in df.columns:
            self._calculate_vwap()
        df = self.data.copy()
        
        # Calculate OBV change rate
        df['OBV_change'] = df['OBV'].pct_change(5)
        
        # Generate volume signals
        df['Volume_Signal'] = 0
        
        # Buy when OBV rising strongly and price above VWAP (bullish volume)
        # Sell when OBV falling strongly and price below VWAP (bearish volume)
        if vwap_filter:
            df.loc[(df['OBV_change'] > 0.05) & (df['Close'] > df['VWAP']), 'Volume_Signal'] = 1
            df.loc[(df['OBV_change'] < -0.05) & (df['Close'] < df['VWAP']), 'Volume_Signal'] = -1
        else:
            df.loc[df['OBV_change'] > 0.05, 'Volume_Signal'] = 1
            df.loc[df['OBV_change'] < -0.05, 'Volume_Signal'] = -1
        
        # Add volume spike detection
        volume_ratio = df['Volume'] / df['vol_ma20']
        df.loc[(volume_ratio > 2) & (df['Close'] > df['ma20']), 'Volume_Signal'] = 1  # Bullish volume spike
        df.loc[(volume_ratio > 2) & (df['Close'] < df['ma20']), 'Volume_Signal'] = -1  # Bearish volume spike
        
        # Position management
        df['Position'] = 0
        position = 0
        
        for i in range(1, len(df)):
            if df['Volume_Signal'].iloc[i] == 1 and position <= 0:
                position = 1  # Enter long
            elif df['Volume_Signal'].iloc[i] == -1 and position >= 0:
                position = -1  # Enter short
            # If no signal, maintain previous position
            df.loc[df.index[i], 'Position'] = position
        
        # Delay the signal by 1 day to simulate trading at the next day's open
        df['Trade_Signal'] = df['Position'].shift(1)
        
        # Add features for risk management
        self._apply_risk_management(df, 'Trade_Signal', stop_loss, trailing_stop)
        
        # Calculate daily market returns (percentage change in close price)
        df['Market_Return'] = df['Close'].pct_change()
        
        # Calculate position changes (for transaction costs)
        df['Position_Change'] = df['Trade_Signal'].diff().abs()
        
        # Calculate strategy returns with transaction costs
        df['Strategy_Return'] = df['Trade_Signal'] * df['Market_Return'] - transaction_cost * df['Position_Change']
        
        # Calculate cumulative returns
        df['Cumulative_Market'] = (1 + df['Market_Return'].fillna(0)).cumprod()
        df['Cumulative_Strategy'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
        
        # For plotting: track when trades occur (signal changes)
        df['Trade_Change'] = df['Trade_Signal'].diff()
        
        # Calculate equity curve and drawdown
        df['Equity'] = self.capital * df['Cumulative_Strategy']
        df['Peak'] = df['Equity'].cummax()
        df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
        
        # Strategy-specific characteristics
        df['Strategy'] = 'Volume'
        df['Strategy_Params'] = f"OBV Window: {obv_window}, VWAP Filter: {vwap_filter}"
        
        return df
    
    def _run_combined_strategy(self, weights: Dict[str, float] = None, 
                              transaction_cost: float = 0.001, stop_loss: float = 0.05, 
                              trailing_stop: bool = False):
        """
        Implement a combined trading strategy using multiple signals.
        
        Args:
            weights: Dictionary with signal weights
            transaction_cost: Transaction cost as a decimal
            stop_loss: Stop loss level as a decimal
            trailing_stop: Whether to use trailing stop loss
            
        Returns:
            DataFrame with strategy performance metrics
        """
        df = self.data.copy()
        
        # Default weights if not provided
        if weights is None:
            weights = {
                'ma_cross': 0.2,
                'macd': 0.2,
                'rsi': 0.2,
                'adx': 0.1,
                'volume': 0.1,
                'bb': 0.1,
                'ichimoku': 0.1
            }
        
        # Ensure all required indicators are calculated
        if 'ma_cross_20_50' not in df.columns:
            self._calculate_moving_averages()
        if 'MACD_cross' not in df.columns:
            self._calculate_macd()
        if 'RSI14' not in df.columns:
            self._calculate_rsi()
        if 'ADX' not in df.columns:
            self._calculate_adx()
        if 'OBV_signal' not in df.columns:
            self._calculate_obv()
        if 'BB_pct_b' not in df.columns:
            self._calculate_bollinger_bands()
        
        df = self.data.copy()
        
        # Generate individual signals
        
        # MA cross signal
        ma_signal = df['ma_cross_20_50']
        
        # MACD signal
        macd_signal = df['MACD_cross']
        
        # RSI signal
        rsi_signal = np.where(
            df['RSI14'] < 30, 1,
            np.where(df['RSI14'] > 70, -1, 0)
        )
        
        # ADX signal (trend strength)
        adx_signal = np.where(
            (df['ADX'] > 25) & (df['+DI'] > df['-DI']), 1,
            np.where((df['ADX'] > 25) & (df['+DI'] < df['-DI']), -1, 0)
        )
        
        # Volume signal
        volume_signal = df['OBV_signal']
        
        # Bollinger Bands signal
        bb_signal = np.where(
            df['BB_pct_b'] < 0.1, 1,
            np.where(df['BB_pct_b'] > 0.9, -1, 0)
        )
        
        # Ichimoku signal
        ichimoku_signal = np.where(
            df['above_cloud'] == 1, 1,
            np.where(df['below_cloud'] == 1, -1, 0)
        )
        
        # Combine signals with weights
        df['Combined_Signal'] = (
            weights.get('ma_cross', 0) * ma_signal +
            weights.get('macd', 0) * macd_signal +
            weights.get('rsi', 0) * rsi_signal +
            weights.get('adx', 0) * adx_signal +
            weights.get('volume', 0) * volume_signal +
            weights.get('bb', 0) * bb_signal +
            weights.get('ichimoku', 0) * ichimoku_signal
        )
        
        # Convert to position: positive -> long, negative -> short, zero -> flat
        df['Position'] = np.sign(df['Combined_Signal'])
        
        # Delay the signal by 1 day to simulate trading at the next day's open
        df['Trade_Signal'] = df['Position'].shift(1)
        
        # Add features for risk management
        self._apply_risk_management(df, 'Trade_Signal', stop_loss, trailing_stop)
        
        # Calculate daily market returns (percentage change in close price)
        df['Market_Return'] = df['Close'].pct_change()
        
        # Calculate position changes (for transaction costs)
        df['Position_Change'] = df['Trade_Signal'].diff().abs()
        
        # Calculate strategy returns with transaction costs
        df['Strategy_Return'] = df['Trade_Signal'] * df['Market_Return'] - transaction_cost * df['Position_Change']
        
        # Apply position sizing based on signal strength
        signal_strength = df['Combined_Signal'].abs() / df['Combined_Signal'].abs().max()
        df['Sized_Return'] = df['Trade_Signal'] * df['Market_Return'] * signal_strength - transaction_cost * df['Position_Change']
        
        # Calculate cumulative returns
        df['Cumulative_Market'] = (1 + df['Market_Return'].fillna(0)).cumprod()
        df['Cumulative_Strategy'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
        df['Cumulative_Sized'] = (1 + df['Sized_Return'].fillna(0)).cumprod()
        
        # For plotting: track when trades occur (signal changes)
        df['Trade_Change'] = df['Trade_Signal'].diff()
        
        # Calculate equity curve and drawdown
        df['Equity'] = self.capital * df['Cumulative_Strategy']
        df['Peak'] = df['Equity'].cummax()
        df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
        
        # Strategy-specific characteristics
        df['Strategy'] = 'Combined'
        df['Strategy_Params'] = f"Weights: {weights}"
        
        return df
    
    def _run_ml_strategy(self, lookback: int = 20, train_size: float = 0.7,
                        transaction_cost: float = 0.001, stop_loss: float = 0.05,
                        trailing_stop: bool = False):
        """
        Implement a machine learning-based trading strategy.
        
        Args:
            lookback: Lookback period for features
            train_size: Training set size (as a fraction)
            transaction_cost: Transaction cost as a decimal
            stop_loss: Stop loss level as a decimal
            trailing_stop: Whether to use trailing stop loss
            
        Returns:
            DataFrame with strategy performance metrics
        """
        df = self.data.copy()
        
        # Ensure all necessary features are calculated
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(method='ffill')
        
        # Prepare features
        feature_cols = [
            'ma_cross_20_50', 'ma_cross_50_200', 'vol_ratio',
            'RSI14', 'MACD_line', 'MACD_signal', 'BB_pct_b',
            'ATR_pct', 'ADX', '+DI', '-DI', 'MFI', 'OBV_signal'
        ]
        
        # Create target: 1 if price increases over next n days, -1 if decreases
        forward_returns = df['Close'].pct_change(5).shift(-5)
        df['target'] = np.where(forward_returns > 0, 1, -1)
        
        # Drop NaN values
        df_ml = df.dropna()
        
        if len(df_ml) < 100:
            logger.warning("Not enough data for ML strategy after removing NaNs")
            return df
        
        # Split into training and testing sets
        train_size_idx = int(len(df_ml) * train_size)
        train = df_ml.iloc[:train_size_idx]
        test = df_ml.iloc[train_size_idx:]
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[feature_cols])
        X_test = scaler.transform(test[feature_cols])
        
        y_train = train['target']
        y_test = test['target']
        
        # Train a simple linear model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Convert predictions to signals
        train['ML_Signal'] = np.sign(train_pred)
        test['ML_Signal'] = np.sign(test_pred)
        
        # Combine predictions
        df_ml.loc[train.index, 'ML_Signal'] = train['ML_Signal']
        df_ml.loc[test.index, 'ML_Signal'] = test['ML_Signal']
        
        # Merge back with original dataframe
        df.loc[df_ml.index, 'ML_Signal'] = df_ml['ML_Signal']
        df['ML_Signal'] = df['ML_Signal'].fillna(0)
        
        # Delay the signal by 1 day to simulate trading at the next day's open
        df['Trade_Signal'] = df['ML_Signal'].shift(1)
        
        # Add features for risk management
        self._apply_risk_management(df, 'Trade_Signal', stop_loss, trailing_stop)
        
        # Calculate daily market returns (percentage change in close price)
        df['Market_Return'] = df['Close'].pct_change()
        
        # Calculate position changes (for transaction costs)
        df['Position_Change'] = df['Trade_Signal'].diff().abs()
        
        # Calculate strategy returns with transaction costs
        df['Strategy_Return'] = df['Trade_Signal'] * df['Market_Return'] - transaction_cost * df['Position_Change']
        
        # Calculate cumulative returns
        df['Cumulative_Market'] = (1 + df['Market_Return'].fillna(0)).cumprod()
        df['Cumulative_Strategy'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
        
        # For plotting: track when trades occur (signal changes)
        df['Trade_Change'] = df['Trade_Signal'].diff()
        
        # Calculate equity curve and drawdown
        df['Equity'] = self.capital * df['Cumulative_Strategy']
        df['Peak'] = df['Equity'].cummax()
        df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
        
        # Strategy-specific characteristics
        df['Strategy'] = 'Machine Learning'
        df['Strategy_Params'] = f"Lookback: {lookback}, Train Size: {train_size}"
        
        return df
    
    def _run_breakout_strategy(self, breakout_period: int = 20, confirmation_days: int = 3,
                               transaction_cost: float = 0.001, stop_loss: float = 0.05,
                               trailing_stop: bool = False):
        """
        Implement a breakout trading strategy.
        
        Args:
            breakout_period: Period for calculating high/low range
            confirmation_days: Number of days to confirm breakout
            transaction_cost: Transaction cost as a decimal
            stop_loss: Stop loss level as a decimal
            trailing_stop: Whether to use trailing stop loss
            
        Returns:
            DataFrame with strategy performance metrics
        """
        df = self.data.copy()
        
        # Calculate rolling high and low
        df['rolling_high'] = df['High'].rolling(window=breakout_period).max()
        df['rolling_low'] = df['Low'].rolling(window=breakout_period).min()
        
        # Identify breakouts
        df['breakout_up'] = df['Close'] > df['rolling_high'].shift(1)
        df['breakout_down'] = df['Close'] < df['rolling_low'].shift(1)
        
        # Confirm breakouts with volume
        df['volume_confirm'] = df['Volume'] > df['Volume'].rolling(window=breakout_period).mean()
        
        # Generate signals
        df['Breakout_Signal'] = 0
        
        # Upward breakout with volume confirmation
        breakout_up_confirmed = df['breakout_up'] & df['volume_confirm']
        # Downward breakout with volume confirmation
        breakout_down_confirmed = df['breakout_down'] & df['volume_confirm']
        
        # Use rolling confirmation
        for i in range(len(df)):
            if i < confirmation_days:
                continue
                
            # Check if breakout has been confirmed for 'confirmation_days'
            up_confirmed = True
            down_confirmed = True
            
            for j in range(confirmation_days):
                up_confirmed = up_confirmed and df['breakout_up'].iloc[i-j]
                down_confirmed = down_confirmed and df['breakout_down'].iloc[i-j]
            
            if up_confirmed:
                df.loc[df.index[i], 'Breakout_Signal'] = 1
            elif down_confirmed:
                df.loc[df.index[i], 'Breakout_Signal'] = -1
        
        # Position management
        df['Position'] = 0
        position = 0
        
        for i in range(1, len(df)):
            if df['Breakout_Signal'].iloc[i] == 1 and position <= 0:
                position = 1  # Enter long
            elif df['Breakout_Signal'].iloc[i] == -1 and position >= 0:
                position = -1  # Enter short
            # If no signal, maintain previous position
            df.loc[df.index[i], 'Position'] = position
        
        # Delay the signal by 1 day to simulate trading at the next day's open
        df['Trade_Signal'] = df['Position'].shift(1)
        
        # Add features for risk management
        self._apply_risk_management(df, 'Trade_Signal', stop_loss, trailing_stop)
        
        # Calculate daily market returns (percentage change in close price)
        df['Market_Return'] = df['Close'].pct_change()
        
        # Calculate position changes (for transaction costs)
        df['Position_Change'] = df['Trade_Signal'].diff().abs()
        
        # Calculate strategy returns with transaction costs
        df['Strategy_Return'] = df['Trade_Signal'] * df['Market_Return'] - transaction_cost * df['Position_Change']
        
        # Calculate cumulative returns
        df['Cumulative_Market'] = (1 + df['Market_Return'].fillna(0)).cumprod()
        df['Cumulative_Strategy'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
        
        # For plotting: track when trades occur (signal changes)
        df['Trade_Change'] = df['Trade_Signal'].diff()
        
        # Calculate equity curve and drawdown
        df['Equity'] = self.capital * df['Cumulative_Strategy']
        df['Peak'] = df['Equity'].cummax()
        df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
        
        # Strategy-specific characteristics
        df['Strategy'] = 'Breakout'
        df['Strategy_Params'] = f"Period: {breakout_period}, Confirmation: {confirmation_days}"
        
        return df
    
    def _run_pivot_strategy(self, transaction_cost: float = 0.001, stop_loss: float = 0.05,
                           trailing_stop: bool = False):
        """
        Implement a pivot point trading strategy.
        
        Args:
            transaction_cost: Transaction cost as a decimal
            stop_loss: Stop loss level as a decimal
            trailing_stop: Whether to use trailing stop loss
            
        Returns:
            DataFrame with strategy performance metrics
        """
        df = self.data.copy()
        
        # Ensure pivot points are calculated
        if 'pivot' not in df.columns:
            self._calculate_pivot_points()
            df = self.data.copy()
        
        # Generate signals
        df['Pivot_Signal'] = 0
        
        # Buy when price bounces from support, sell when it bounces from resistance
        support_bounce = (df['Low'] <= df['s1']) & (df['Close'] > df['s1']) & (df['Close'] > df['Open'])
        resistance_bounce = (df['High'] >= df['r1']) & (df['Close'] < df['r1']) & (df['Close'] < df['Open'])
        
        df.loc[support_bounce, 'Pivot_Signal'] = 1
        df.loc[resistance_bounce, 'Pivot_Signal'] = -1
        
        # Add trend filter
        trend_filter = df['ma50'] > df['ma200']  # Uptrend
        df.loc[~trend_filter & (df['Pivot_Signal'] == 1), 'Pivot_Signal'] = 0  # Cancel buy in downtrend
        df.loc[trend_filter & (df['Pivot_Signal'] == -1), 'Pivot_Signal'] = 0  # Cancel sell in uptrend
        
        # Position management
        df['Position'] = 0
        position = 0
        
        for i in range(1, len(df)):
            if df['Pivot_Signal'].iloc[i] == 1 and position <= 0:
                position = 1  # Enter long
            elif df['Pivot_Signal'].iloc[i] == -1 and position >= 0:
                position = -1  # Enter short
            # If no signal, maintain previous position
            df.loc[df.index[i], 'Position'] = position
        
        # Delay the signal by 1 day to simulate trading at the next day's open
        df['Trade_Signal'] = df['Position'].shift(1)
        
        # Add features for risk management
        self._apply_risk_management(df, 'Trade_Signal', stop_loss, trailing_stop)
        
        # Calculate daily market returns (percentage change in close price)
        df['Market_Return'] = df['Close'].pct_change()
        
        # Calculate position changes (for transaction costs)
        df['Position_Change'] = df['Trade_Signal'].diff().abs()
        
        # Calculate strategy returns with transaction costs
        df['Strategy_Return'] = df['Trade_Signal'] * df['Market_Return'] - transaction_cost * df['Position_Change']
        
        # Calculate cumulative returns
        df['Cumulative_Market'] = (1 + df['Market_Return'].fillna(0)).cumprod()
        df['Cumulative_Strategy'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
        
        # For plotting: track when trades occur (signal changes)
        df['Trade_Change'] = df['Trade_Signal'].diff()
        
        # Calculate equity curve and drawdown
        df['Equity'] = self.capital * df['Cumulative_Strategy']
        df['Peak'] = df['Equity'].cummax()
        df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
        
        # Strategy-specific characteristics
        df['Strategy'] = 'Pivot Points'
        df['Strategy_Params'] = "Standard Pivot Points"
        
        return df
    
    def _run_mean_reversion_strategy(self, window: int = 20, std_dev: float = 2.0,
                                    transaction_cost: float = 0.001, stop_loss: float = 0.05,
                                    trailing_stop: bool = False):
        """
        Implement a mean reversion trading strategy.
        
        Args:
            window: Window for calculating mean and standard deviation
            std_dev: Number of standard deviations for overbought/oversold
            transaction_cost: Transaction cost as a decimal
            stop_loss: Stop loss level as a decimal
            trailing_stop: Whether to use trailing stop loss
            
        Returns:
            DataFrame with strategy performance metrics
        """
        df = self.data.copy()
        
        # Calculate z-score
        df['rolling_mean'] = df['Close'].rolling(window=window).mean()
        df['rolling_std'] = df['Close'].rolling(window=window).std()
        df['z_score'] = (df['Close'] - df['rolling_mean']) / df['rolling_std']
        
        # Generate signals
        df['MR_Signal'] = 0
        
        # Buy when oversold, sell when overbought
        df.loc[df['z_score'] < -std_dev, 'MR_Signal'] = 1  # Oversold
        df.loc[df['z_score'] > std_dev, 'MR_Signal'] = -1  # Overbought
        
        # Add volatility filter: avoid trading in high volatility
        if 'vol_regime' in df.columns:
            df.loc[df['vol_regime'] == 'high', 'MR_Signal'] = 0
        
        # Position management
        df['Position'] = 0
        position = 0
        
        for i in range(1, len(df)):
            if df['MR_Signal'].iloc[i] == 1 and position <= 0:
                position = 1  # Enter long
            elif df['MR_Signal'].iloc[i] == -1 and position >= 0:
                position = -1  # Enter short
            # If no signal, maintain previous position
            df.loc[df.index[i], 'Position'] = position
        
        # Delay the signal by 1 day to simulate trading at the next day's open
        df['Trade_Signal'] = df['Position'].shift(1)
        
        # Add features for risk management
        self._apply_risk_management(df, 'Trade_Signal', stop_loss, trailing_stop)
        
        # Calculate daily market returns (percentage change in close price)
        df['Market_Return'] = df['Close'].pct_change()
        
        # Calculate position changes (for transaction costs)
        df['Position_Change'] = df['Trade_Signal'].diff().abs()
        
        # Calculate strategy returns with transaction costs
        df['Strategy_Return'] = df['Trade_Signal'] * df['Market_Return'] - transaction_cost * df['Position_Change']
        
        # Calculate cumulative returns
        df['Cumulative_Market'] = (1 + df['Market_Return'].fillna(0)).cumprod()
        df['Cumulative_Strategy'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
        
        # For plotting: track when trades occur (signal changes)
        df['Trade_Change'] = df['Trade_Signal'].diff()
        
        # Calculate equity curve and drawdown
        df['Equity'] = self.capital * df['Cumulative_Strategy']
        df['Peak'] = df['Equity'].cummax()
        df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
        
        # Strategy-specific characteristics
        df['Strategy'] = 'Mean Reversion'
        df['Strategy_Params'] = f"Window: {window}, StdDev: {std_dev}"
        
        return df
    
    def _apply_risk_management(self, df: pd.DataFrame, signal_col: str, 
                              stop_loss: float, trailing_stop: bool):
        """
        Apply risk management rules to a strategy.
        
        Args:
            df: DataFrame with strategy data
            signal_col: Column name for the trade signal
            stop_loss: Stop loss level as a decimal
            trailing_stop: Whether to use trailing stop loss
        """
        if stop_loss <= 0:
            return
        
        # Create entry price and stop loss columns
        df['entry_price'] = np.nan
        df['stop_loss_price'] = np.nan
        
        # Initialize position and entry price
        position = 0
        entry_price = 0
        stop_price = 0
        
        # Track highest/lowest price since entry for trailing stop
        high_since_entry = 0
        low_since_entry = float('inf')
        
        for i in range(1, len(df)):
            prev_signal = df[signal_col].iloc[i-1]
            curr_signal = df[signal_col].iloc[i]
            curr_price = df['Close'].iloc[i]
            
            # Position changed
            if curr_signal != prev_signal:
                # New position
                position = curr_signal
                
                if position != 0:
                    # Set entry price
                    entry_price = curr_price
                    
                    # Set initial stop loss
                    if position > 0:  # Long position
                        stop_price = entry_price * (1 - stop_loss)
                        high_since_entry = curr_price
                    else:  # Short position
                        stop_price = entry_price * (1 + stop_loss)
                        low_since_entry = curr_price
            
            # Update trailing stop if active
            if trailing_stop and position != 0:
                if position > 0:  # Long position
                    high_since_entry = max(high_since_entry, curr_price)
                    new_stop = high_since_entry * (1 - stop_loss)
                    stop_price = max(stop_price, new_stop)
                else:  # Short position
                    low_since_entry = min(low_since_entry, curr_price)
                    new_stop = low_since_entry * (1 + stop_loss)
                    stop_price = min(stop_price, new_stop)
            
            # Check if stop loss was hit
            if position > 0 and curr_price <= stop_price:  # Long position stop hit
                df.loc[df.index[i], signal_col] = 0  # Exit position
                position = 0
            elif position < 0 and curr_price >= stop_price:  # Short position stop hit
                df.loc[df.index[i], signal_col] = 0  # Exit position
                position = 0
            
            # Store entry and stop loss prices
            if position != 0:
                df.loc[df.index[i], 'entry_price'] = entry_price
                df.loc[df.index[i], 'stop_loss_price'] = stop_price
    
    def _calculate_strategy_metrics(self, strategy_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate performance metrics for a trading strategy.
        
        Args:
            strategy_data: DataFrame with strategy data
            
        Returns:
            Dictionary with performance metrics
        """
        # Select valid data (drop NaN values)
        valid_data = strategy_data.dropna(subset=['Strategy_Return'])
        
        # Ensure we have enough data
        if len(valid_data) < 30:
            logger.warning("Not enough data for reliable performance metrics")
            return {}
        
        # Calculate time period
        total_days = (valid_data.index[-1] - valid_data.index[0]).days
        years = total_days / 365.25
        
        # Calculate total returns
        total_return_strategy = valid_data['Cumulative_Strategy'].iloc[-1] - 1
        total_return_market = valid_data['Cumulative_Market'].iloc[-1] - 1
        
        # Calculate annualized returns
        annual_return_strategy = (1 + total_return_strategy) ** (1 / years) - 1
        annual_return_market = (1 + total_return_market) ** (1 / years) - 1
        
        # Calculate volatility (standard deviation)
        daily_vol_strategy = valid_data['Strategy_Return'].std()
        annual_vol_strategy = daily_vol_strategy * np.sqrt(252)  # Assuming 252 trading days per year
        
        daily_vol_market = valid_data['Market_Return'].std()
        annual_vol_market = daily_vol_market * np.sqrt(252)
        
        # Calculate risk-adjusted metrics
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        
        # Sharpe Ratio
        excess_return_strategy = annual_return_strategy - risk_free_rate
        sharpe_ratio_strategy = excess_return_strategy / annual_vol_strategy if annual_vol_strategy != 0 else 0
        
        excess_return_market = annual_return_market - risk_free_rate
        sharpe_ratio_market = excess_return_market / annual_vol_market if annual_vol_market != 0 else 0
        
        # Sortino Ratio (downside risk only)
        downside_returns_strategy = valid_data['Strategy_Return'][valid_data['Strategy_Return'] < 0]
        downside_vol_strategy = downside_returns_strategy.std() * np.sqrt(252)
        sortino_ratio_strategy = excess_return_strategy / downside_vol_strategy if downside_vol_strategy != 0 else 0
        
        downside_returns_market = valid_data['Market_Return'][valid_data['Market_Return'] < 0]
        downside_vol_market = downside_returns_market.std() * np.sqrt(252)
        sortino_ratio_market = excess_return_market / downside_vol_market if downside_vol_market != 0 else 0
        
        # Calculate maximum drawdown
        drawdown_strategy = valid_data['Drawdown']
        max_drawdown_strategy = drawdown_strategy.min()
        
        # Calculate drawdown for market
        peak_market = valid_data['Cumulative_Market'].cummax()
        drawdown_market = (valid_data['Cumulative_Market'] - peak_market) / peak_market
        max_drawdown_market = drawdown_market.min()
        
        # Calmar Ratio (return/max drawdown)
        calmar_ratio_strategy = -annual_return_strategy / max_drawdown_strategy if max_drawdown_strategy != 0 else 0
        calmar_ratio_market = -annual_return_market / max_drawdown_market if max_drawdown_market != 0 else 0
        
        # Calculate winning trades statistics
        trade_changes = valid_data['Trade_Change'].dropna()
        trade_entries = trade_changes[trade_changes != 0].index
        
        trades = []
        current_position = 0
        entry_date = None
        entry_price = 0
        
        # Extract trade statistics
        for i in range(len(valid_data)):
            date = valid_data.index[i]
            if date in trade_entries:
                signal = valid_data['Trade_Signal'].iloc[i]
                price = valid_data['Close'].iloc[i]
                
                # Position changed
                if signal != current_position:
                    # Close previous position
                    if current_position != 0 and entry_date is not None:
                        exit_date = date
                        exit_price = price
                        pnl = (exit_price - entry_price) / entry_price if current_position > 0 else (entry_price - exit_price) / entry_price
                        duration = (exit_date - entry_date).days
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'direction': 'long' if current_position > 0 else 'short',
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'duration': duration
                        })
                    
                    # Open new position
                    current_position = signal
                    if current_position != 0:
                        entry_date = date
                        entry_price = price
        
        # Close final position if open
        if current_position != 0 and entry_date is not None:
            exit_date = valid_data.index[-1]
            exit_price = valid_data['Close'].iloc[-1]
            pnl = (exit_price - entry_price) / entry_price if current_position > 0 else (entry_price - exit_price) / entry_price
            duration = (exit_date - entry_date).days
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'direction': 'long' if current_position > 0 else 'short',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'duration': duration
            })
        
        # Calculate trade statistics
        if trades:
            trades_df = pd.DataFrame(trades)
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] <= 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            avg_winning_trade = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_losing_trade = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
            
            profit_factor = (
                abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum()) / 
                abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
                if trades_df[trades_df['pnl'] <= 0]['pnl'].sum() != 0 else float('inf')
            )
            
            avg_trade_duration = trades_df['duration'].mean()
        else:
            total_trades = 0
            winning_trades = 0
            losing_trades = 0
            win_rate = 0
            avg_winning_trade = 0
            avg_losing_trade = 0
            profit_factor = 0
            avg_trade_duration = 0
        
        # Create results dictionary
        metrics = {
            'start_date': valid_data.index[0],
            'end_date': valid_data.index[-1],
            'total_days': len(valid_data),
            'years': years,
            'strategy': {
                'total_return': total_return_strategy,
                'annual_return': annual_return_strategy,
                'annual_volatility': annual_vol_strategy,
                'sharpe_ratio': sharpe_ratio_strategy,
                'sortino_ratio': sortino_ratio_strategy,
                'max_drawdown': max_drawdown_strategy,
                'calmar_ratio': calmar_ratio_strategy,
                'final_equity': self.capital * valid_data['Cumulative_Strategy'].iloc[-1]
            },
            'market': {
                'total_return': total_return_market,
                'annual_return': annual_return_market,
                'annual_volatility': annual_vol_market,
                'sharpe_ratio': sharpe_ratio_market,
                'sortino_ratio': sortino_ratio_market,
                'max_drawdown': max_drawdown_market,
                'calmar_ratio': calmar_ratio_market,
                'final_equity': self.capital * valid_data['Cumulative_Market'].iloc[-1]
            },
            'trade_stats': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_winning_trade': avg_winning_trade,
                'avg_losing_trade': avg_losing_trade,
                'profit_factor': profit_factor,
                'avg_trade_duration': avg_trade_duration
            }
        }
        
        return metrics
    
    def optimize_strategy(self, strategy_name: str, param_grid: Dict[str, List], 
                         metric: str = 'sharpe_ratio'):
        """
        Optimize a strategy by grid search over parameters.
        
        Args:
            strategy_name: Name of the strategy to optimize
            param_grid: Dictionary with parameter names and lists of values
            metric: Metric to optimize
            
        Returns:
            Tuple of (best parameters, best metric value)
        """
        if not self.data.empty:
            logger.info(f"Optimizing {strategy_name} strategy")
            
            # Generate all parameter combinations
            param_keys = list(param_grid.keys())
            param_values = list(param_grid.values())
            param_combinations = list(itertools.product(*param_values))
            
            best_params = None
            best_metric_value = -float('inf')
            
            for params in param_combinations:
                param_dict = dict(zip(param_keys, params))
                logger.info(f"Testing parameters: {param_dict}")
                
                # Run strategy with current parameters
                _, metrics = self.run_strategy(strategy_name, **param_dict)
                
                # Check if this is the best so far
                if 'strategy' in metrics and metric in metrics['strategy']:
                    metric_value = metrics['strategy'][metric]

                     # For drawdown, lower is better
                    if metric == 'max_drawdown':
                        metric_value = -metric_value

                    if metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_params = param_dict
                        
                        # Store best strategy
                        self.best_strategy = {
                            'name': strategy_name,
                            'params': best_params,
                            'metrics': metrics,
                            'data': self.strategy_results[strategy_name]['data']
                        }
            
            logger.info(f"Optimization complete. Best parameters: {best_params}")
            return best_params, best_metric_value
        else:
            logger.error("No data available for optimization")
            return None, None
    
    def compare_strategies(self, strategies: List[str] = None) -> pd.DataFrame:
        """
        Compare the performance of multiple strategies.
        
        Args:
            strategies: List of strategy names to compare (all if None)
            
        Returns:
            DataFrame with comparison metrics
        """
        if not self.strategy_results:
            logger.error("No strategy results available")
            return pd.DataFrame()
        
        # Use all strategies if none specified
        if strategies is None:
            strategies = list(self.strategy_results.keys())
        
        comparison = []
        
        for strategy in strategies:
            if strategy in self.strategy_results:
                metrics = self.strategy_results[strategy]['metrics']
                params = self.strategy_results[strategy]['params']
                
                # Create a row for this strategy
                row = {
                    'Strategy': strategy,
                    'Parameters': str(params),
                    'Total Return': metrics['strategy']['total_return'],
                    'Annual Return': metrics['strategy']['annual_return'],
                    'Volatility': metrics['strategy']['annual_volatility'],
                    'Sharpe Ratio': metrics['strategy']['sharpe_ratio'],
                    'Sortino Ratio': metrics['strategy']['sortino_ratio'],
                    'Max Drawdown': metrics['strategy']['max_drawdown'],
                    'Calmar Ratio': metrics['strategy']['calmar_ratio'],
                    'Win Rate': metrics['trade_stats']['win_rate'],
                    'Profit Factor': metrics['trade_stats']['profit_factor'],
                    'Total Trades': metrics['trade_stats']['total_trades']
                }
                comparison.append(row)
            else:
                logger.warning(f"Strategy {strategy} not found in results")
        
        if comparison:
            # Convert to DataFrame and sort by Sharpe Ratio
            comparison_df = pd.DataFrame(comparison)
            comparison_df = comparison_df.sort_values('Sharpe Ratio', ascending=False)
            return comparison_df
        else:
            return pd.DataFrame()
    
    def plot_equity_curves(self, strategies: List[str] = None, save_path: Optional[Path] = None):
        """
        Plot equity curves for multiple strategies.
        
        Args:
            strategies: List of strategy names to plot (all if None)
            save_path: Path to save the chart
        """
        if not self.strategy_results:
            logger.error("No strategy results available")
            return
        
        # Use all strategies if none specified
        if strategies is None:
            strategies = list(self.strategy_results.keys())
        
        plt.figure(figsize=(14, 8))
        
        # Plot market return as benchmark
        for strategy in strategies:
            if strategy in self.strategy_results:
                data = self.strategy_results[strategy]['data']
                plt.plot(data.index, data['Equity'], label=f"{strategy} Strategy")
        
        # Plot market equity for comparison
        if strategies and strategies[0] in self.strategy_results:
            market_data = self.strategy_results[strategies[0]]['data']
            plt.plot(market_data.index, self.capital * market_data['Cumulative_Market'], 
                     'k--', label='Buy & Hold')
        
        plt.title('Strategy Equity Curves Comparison', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Account Value ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.gcf().autofmt_xdate()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Equity curves chart saved to {save_path}")
        
        plt.close()
    
    def plot_drawdowns(self, strategies: List[str] = None, save_path: Optional[Path] = None):
        """
        Plot drawdowns for multiple strategies.
        
        Args:
            strategies: List of strategy names to plot (all if None)
            save_path: Path to save the chart
        """
        if not self.strategy_results:
            logger.error("No strategy results available")
            return
        
        # Use all strategies if none specified
        if strategies is None:
            strategies = list(self.strategy_results.keys())
        
        plt.figure(figsize=(14, 8))
        
        for strategy in strategies:
            if strategy in self.strategy_results:
                data = self.strategy_results[strategy]['data']
                plt.plot(data.index, data['Drawdown'] * 100, label=f"{strategy} Strategy")
        
        # Plot market drawdown for comparison
        if strategies and strategies[0] in self.strategy_results:
            market_data = self.strategy_results[strategies[0]]['data']
            peak_market = market_data['Cumulative_Market'].cummax()
            drawdown_market = (market_data['Cumulative_Market'] - peak_market) / peak_market
            plt.plot(market_data.index, drawdown_market * 100, 'k--', label='Buy & Hold')
        
        plt.title('Strategy Drawdowns Comparison', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.gcf().autofmt_xdate()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Drawdowns chart saved to {save_path}")
        
        plt.close()
    
    def plot_returns_distribution(self, strategies: List[str] = None, save_path: Optional[Path] = None):
        """
        Plot the distribution of returns for multiple strategies.
        
        Args:
            strategies: List of strategy names to plot (all if None)
            save_path: Path to save the chart
        """
        if not self.strategy_results:
            logger.error("No strategy results available")
            return
        
        # Use all strategies if none specified
        if strategies is None:
            strategies = list(self.strategy_results.keys())
        
        plt.figure(figsize=(14, 8))
        
        for strategy in strategies:
            if strategy in self.strategy_results:
                data = self.strategy_results[strategy]['data']
                returns = data['Strategy_Return'].dropna() * 100  # Convert to percentage
                sns.kdeplot(returns, label=f"{strategy} Strategy")
        
        # Plot market returns for comparison
        if strategies and strategies[0] in self.strategy_results:
            market_data = self.strategy_results[strategies[0]]['data']
            market_returns = market_data['Market_Return'].dropna() * 100  # Convert to percentage
            sns.kdeplot(market_returns, linestyle='--', label='Buy & Hold')
        
        plt.title('Distribution of Daily Returns', fontsize=16)
        plt.xlabel('Daily Return (%)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Returns distribution chart saved to {save_path}")
        
        plt.close()
    
    def plot_monthly_returns(self, strategy_name: str, save_path: Optional[Path] = None):
        """
        Plot monthly returns heatmap for a strategy.
        
        Args:
            strategy_name: Name of the strategy to plot
            save_path: Path to save the chart
        """
        if strategy_name not in self.strategy_results:
            logger.error(f"Strategy {strategy_name} not found in results")
            return
        
        data = self.strategy_results[strategy_name]['data']
        
        # Resample returns to monthly frequency
        monthly_returns = data['Strategy_Return'].dropna().resample('M').apply(
            lambda x: (1 + x).prod() - 1
        ) * 100  # Convert to percentage
        
        # Create a DataFrame with year and month as indices
        returns_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        # Pivot the data for the heatmap
        heatmap_data = returns_df.pivot_table(
            index='Year', columns='Month', values='Return'
        )
        
        # Plot heatmap
        plt.figure(figsize=(14, 10))
        ax = sns.heatmap(
            heatmap_data, 
            annot=True, 
            fmt=".2f", 
            cmap="RdYlGn", 
            center=0, 
            linewidths=1,
            cbar_kws={'label': 'Monthly Return (%)'}
        )
        
        # Set labels
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(months, rotation=45)
        
        plt.title(f'Monthly Returns for {strategy_name} Strategy (%)', fontsize=16)
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Monthly returns chart saved to {save_path}")
        
        plt.close()
    
    def plot_technical_analysis(self, days: int = 180, save_path: Optional[Path] = None):
        """
        Plot a comprehensive technical analysis chart.
        
        Args:
            days: Number of days to plot
            save_path: Path to save the chart
        """
        if self.data is None:
            logger.error("No data available for plotting")
            return
        
        # Get the most recent days
        plot_data = self.data.iloc[-days:].copy()
        
        # Create subplots
        fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True, 
                                 gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        
        # 1. Price and moving averages
        axes[0].plot(plot_data.index, plot_data['Close'], label='Close', linewidth=1.5)
        axes[0].plot(plot_data.index, plot_data['ma20'], label='MA20', linewidth=1)
        axes[0].plot(plot_data.index, plot_data['ma50'], label='MA50', linewidth=1)
        axes[0].plot(plot_data.index, plot_data['ma200'], label='MA200', linewidth=1)
        
        # Plot Bollinger Bands if available
        if 'BB_upper' in plot_data.columns:
            axes[0].plot(plot_data.index, plot_data['BB_upper'], 'r--', label='BB Upper', alpha=0.6)
            axes[0].plot(plot_data.index, plot_data['BB_lower'], 'r--', label='BB Lower', alpha=0.6)
        
        # Plot Ichimoku Cloud if available
        if 'senkou_span_a' in plot_data.columns:
            axes[0].plot(plot_data.index, plot_data['senkou_span_a'], 'g--', label='Span A', alpha=0.6)
            axes[0].plot(plot_data.index, plot_data['senkou_span_b'], 'r--', label='Span B', alpha=0.6)
            
            # Fill between the spans to form the cloud
            axes[0].fill_between(
                plot_data.index, 
                plot_data['senkou_span_a'], 
                plot_data['senkou_span_b'],
                where=plot_data['senkou_span_a'] >= plot_data['senkou_span_b'],
                color='green', alpha=0.2
            )
            axes[0].fill_between(
                plot_data.index, 
                plot_data['senkou_span_a'], 
                plot_data['senkou_span_b'],
                where=plot_data['senkou_span_a'] < plot_data['senkou_span_b'],
                color='red', alpha=0.2
            )
        
        axes[0].set_title(f'{self.ticker} Price and Technical Indicators', fontsize=16)
        axes[0].set_ylabel('Price (USD)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper left')
        
        # 2. Volume and volume moving averages
        axes[1].bar(plot_data.index, plot_data['Volume'], label='Volume', alpha=0.6)
        axes[1].plot(plot_data.index, plot_data['vol_ma5'], label='Vol MA5', linewidth=1.5)
        axes[1].plot(plot_data.index, plot_data['vol_ma20'], label='Vol MA20', linewidth=1.5)
        axes[1].set_ylabel('Volume', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper left')
        
        # 3. RSI
        if 'RSI14' in plot_data.columns:
            axes[2].plot(plot_data.index, plot_data['RSI14'], label='RSI(14)', color='purple')
            axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.5)
            axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.5)
            axes[2].axhline(y=50, color='k', linestyle='-', alpha=0.2)
            axes[2].set_ylim(0, 100)
            axes[2].set_ylabel('RSI', fontsize=12)
            axes[2].grid(True, alpha=0.3)
        
        # 4. MACD
        if 'MACD_line' in plot_data.columns:
            axes[3].plot(plot_data.index, plot_data['MACD_line'], label='MACD', color='blue')
            axes[3].plot(plot_data.index, plot_data['MACD_signal'], label='Signal', color='red')
            
            # Plot MACD histogram
            for i in range(len(plot_data)):
                if i > 0:
                    if plot_data['MACD_hist'].iloc[i] >= 0:
                        color = 'g' if plot_data['MACD_hist'].iloc[i] > plot_data['MACD_hist'].iloc[i-1] else 'lime'
                    else:
                        color = 'r' if plot_data['MACD_hist'].iloc[i] < plot_data['MACD_hist'].iloc[i-1] else 'tomato'
                    
                    axes[3].bar(plot_data.index[i], plot_data['MACD_hist'].iloc[i], color=color, alpha=0.5)
            
            axes[3].axhline(y=0, color='k', linestyle='-', alpha=0.2)
            axes[3].set_ylabel('MACD', fontsize=12)
            axes[3].grid(True, alpha=0.3)
            axes[3].legend(loc='upper left')
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Technical analysis chart saved to {save_path}")
        
        plt.close()
    
    def save_results(self, filename: str = 'trading_system_results.json'):
        """
        Save strategy results to a JSON file.
        
        Args:
            filename: Name of the file to save
        """
        results = {}
        
        # Extract and store metrics for each strategy
        for strategy_name, strategy_data in self.strategy_results.items():
            results[strategy_name] = {
                'params': strategy_data['params'],
                'metrics': strategy_data['metrics']
            }
        
        # Save to file
        try:
            filepath = self.output_dir / filename
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def load_results(self, filename: str = 'trading_system_results.json'):
        """
        Load strategy results from a JSON file.
        
        Args:
            filename: Name of the file to load
        """
        try:
            filepath = self.output_dir / filename
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Results loaded from {filepath}")
            return results
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return {}
    
    def print_strategy_summary(self, strategy_name: str):
        """
        Print a summary of a strategy's performance.
        
        Args:
            strategy_name: Name of the strategy to summarize
        """
        if strategy_name not in self.strategy_results:
            logger.error(f"Strategy {strategy_name} not found in results")
            return
        
        metrics = self.strategy_results[strategy_name]['metrics']
        params = self.strategy_results[strategy_name]['params']
        
        print("\n" + "="*80)
        print(f"STRATEGY PERFORMANCE SUMMARY: {strategy_name}")
        print("="*80)
        print(f"Evaluation Period: {metrics['start_date']} to {metrics['end_date']}")
        print(f"Total Days: {metrics['total_days']} ({metrics['years']:.2f} years)")
        print(f"Strategy Parameters: {params}")
        print("\n--- PERFORMANCE METRICS ---")
        print(f"Initial Capital: ${self.capital:,.2f}")
        print(f"Final Equity: ${metrics['strategy']['final_equity']:,.2f}")
        print(f"Total Return: {metrics['strategy']['total_return']:,.2%}")
        print(f"Annual Return: {metrics['strategy']['annual_return']:,.2%}")
        print(f"Annual Volatility: {metrics['strategy']['annual_volatility']:,.2%}")
        print(f"Sharpe Ratio: {metrics['strategy']['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['strategy']['sortino_ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['strategy']['max_drawdown']:,.2%}")
        print(f"Calmar Ratio: {metrics['strategy']['calmar_ratio']:.2f}")
        print("\n--- TRADE STATISTICS ---")
        print(f"Total Trades: {metrics['trade_stats']['total_trades']}")
        print(f"Winning Trades: {metrics['trade_stats']['winning_trades']} ({metrics['trade_stats']['win_rate']:,.2%})")
        print(f"Losing Trades: {metrics['trade_stats']['losing_trades']} ({1-metrics['trade_stats']['win_rate']:,.2%})")
        print(f"Average Winning Trade: {metrics['trade_stats']['avg_winning_trade']:,.2%}")
        print(f"Average Losing Trade: {metrics['trade_stats']['avg_losing_trade']:,.2%}")
        print(f"Profit Factor: {metrics['trade_stats']['profit_factor']:.2f}")
        print(f"Average Trade Duration: {metrics['trade_stats']['avg_trade_duration']:.1f} days")
        print("\n--- MARKET COMPARISON ---")
        print(f"Market Total Return: {metrics['market']['total_return']:,.2%}")
        print(f"Market Annual Return: {metrics['market']['annual_return']:,.2%}")
        print(f"Market Sharpe Ratio: {metrics['market']['sharpe_ratio']:.2f}")
        print(f"Market Maximum Drawdown: {metrics['market']['max_drawdown']:,.2%}")
        print(f"Strategy Outperformance: {metrics['strategy']['annual_return']-metrics['market']['annual_return']:,.2%} per year")
        print("="*80)


def run_demo():
    """Run a demonstration of the trading system."""
    # Initialize trading system
    ts = TradingSystem(ticker="BTC-USD", period="5y", capital=10000)
    
    # Download and prepare data
    ts.download_data()
    ts.prepare_data()
    
    # Save technical analysis chart
    ts.plot_technical_analysis(days=365, save_path=ts.output_dir / "Stock_Technical_Analysis.png")
    
    # Run different strategies
    strategies_to_run = [
        ('ma_crossover', {'fast_ma': 'ma20', 'slow_ma': 'ma50'}),
        ('macd', {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}),
        ('rsi', {'rsi_period': 14, 'overbought': 70, 'oversold': 30}),
        ('bollinger', {'window': 20, 'num_std': 2.0}),
        ('breakout', {'breakout_period': 20, 'confirmation_days': 2}),
        ('mean_reversion', {'window': 20, 'std_dev': 2.0}),
        ('combined', {})
    ]
    
    for strategy_name, params in strategies_to_run:
        ts.run_strategy(strategy_name, **params)
        ts.print_strategy_summary(strategy_name)
    
    # Compare strategies
    comparison = ts.compare_strategies()
    print("\nSTRATEGY COMPARISON:")
    print(comparison.to_string())
    
    # Plot equity curves
    ts.plot_equity_curves(save_path=ts.output_dir / "Stock_Equity_Curves.png")
    
    # Plot drawdowns
    ts.plot_drawdowns(save_path=ts.output_dir / "Stock_Drawdowns.png")
    
    # Plot returns distribution
    ts.plot_returns_distribution(save_path=ts.output_dir / "Stock_Returns_Distribution.png")
    
    # Plot monthly returns for the best strategy
    best_strategy = comparison.iloc[0]['Strategy']
    ts.plot_monthly_returns(best_strategy, save_path=ts.output_dir / f"This Stock _{best_strategy}_Monthly_Returns.png")
    
    # Define strategy-specific parameter grids for optimization
    optimization_grids = {
        'ma_crossover': {
            'fast_ma': ['ma5', 'ma10', 'ma20', 'ma50'],
            'slow_ma': ['ma50', 'ma100', 'ma200'],
            'stop_loss': [0.03, 0.05, 0.07],
            'trailing_stop': [True, False]
        },
        'macd': {
            'fast_period': [8, 12, 16],
            'slow_period': [21, 26, 30],
            'signal_period': [7, 9, 11],
            'stop_loss': [0.03, 0.05, 0.07]
        },
        'rsi': {
            'rsi_period': [9, 14, 21],
            'overbought': [65, 70, 75],
            'oversold': [25, 30, 35],
            'stop_loss': [0.03, 0.05, 0.07]
        },
        'bollinger': {
            'window': [15, 20, 25],
            'num_std': [1.5, 2.0, 2.5],
            'stop_loss': [0.03, 0.05, 0.07]
        },
        'breakout': {
            'breakout_period': [15, 20, 25, 30],
            'confirmation_days': [1, 2, 3],
            'stop_loss': [0.03, 0.05, 0.07]
        },
        'mean_reversion': {
            'window': [15, 20, 25],
            'std_dev': [1.5, 2.0, 2.5],
            'stop_loss': [0.03, 0.05, 0.07]
        },
        'combined': {
            'weights': [
                {'ma_cross': 0.3, 'macd': 0.3, 'rsi': 0.4},
                {'ma_cross': 0.4, 'macd': 0.2, 'rsi': 0.4},
                {'ma_cross': 0.2, 'macd': 0.4, 'rsi': 0.4}
            ],
            'stop_loss': [0.03, 0.05, 0.07]
        }
    }
    
    # Get appropriate parameter grid for the best strategy
    if best_strategy in optimization_grids:
        param_grid = optimization_grids[best_strategy]
        
        print(f"\nOptimizing {best_strategy} strategy with appropriate parameters...")
        best_params, best_value = ts.optimize_strategy(best_strategy, param_grid, 'sharpe_ratio')
        
        print(f"Best parameters: {best_params}")
        print(f"Best Sharpe ratio: {best_value:.2f}")
    else:
        print(f"\nNo optimization grid defined for {best_strategy} strategy.")
    
    # Save results
    ts.save_results()
    
    print(f"\nAll results and charts saved to {ts.output_dir}")


def optimize_strategy_safely(ts, strategy_name):
    """Safely optimize a strategy with appropriate parameters."""
    # Define strategy-specific parameter grids
    optimization_grids = {
        'ma_crossover': {
            'fast_ma': ['ma5', 'ma10', 'ma20', 'ma50'],
            'slow_ma': ['ma50', 'ma100', 'ma200'],
            'stop_loss': [0.03, 0.05, 0.07],
            'trailing_stop': [True, False]
        },
        'macd': {
            'fast_period': [8, 12, 16],
            'slow_period': [21, 26, 30],
            'signal_period': [7, 9, 11],
            'stop_loss': [0.03, 0.05, 0.07]
        },
        'rsi': {
            'rsi_period': [9, 14, 21],
            'overbought': [65, 70, 75],
            'oversold': [25, 30, 35],
            'stop_loss': [0.03, 0.05, 0.07]
        },
        'bollinger': {
            'window': [15, 20, 25],
            'num_std': [1.5, 2.0, 2.5],
            'stop_loss': [0.03, 0.05, 0.07]
        },
        'breakout': {
            'breakout_period': [15, 20, 25, 30],
            'confirmation_days': [1, 2, 3],
            'stop_loss': [0.03, 0.05, 0.07]
        },
        'mean_reversion': {
            'window': [15, 20, 25],
            'std_dev': [1.5, 2.0, 2.5],
            'stop_loss': [0.03, 0.05, 0.07]
        },
        'combined': {
            'weights': [
                {'ma_cross': 0.3, 'macd': 0.3, 'rsi': 0.4},
                {'ma_cross': 0.4, 'macd': 0.2, 'rsi': 0.4},
                {'ma_cross': 0.2, 'macd': 0.4, 'rsi': 0.4}
            ],
            'stop_loss': [0.03, 0.05, 0.07]
        }
    }
    
    # Check if we have a parameter grid for this strategy
    if strategy_name in optimization_grids:
        param_grid = optimization_grids[strategy_name]
        
        print(f"\nOptimizing {strategy_name} strategy with appropriate parameters...")
        return ts.optimize_strategy(strategy_name, param_grid, 'sharpe_ratio')
    else:
        print(f"\nNo optimization grid defined for {strategy_name} strategy.")
        return None, None


if __name__ == "__main__":
    run_demo()