import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Global transaction fee (0.1%)
TRANSACTION_FEE = 0.001

class TradingEnvironment:
    def __init__(self, trading_system, data, initial_cash=10000, max_position=0.5):
        """
        Simulated trading environment for DQN training.
        
        Args:
            trading_system (AITradingSystem): Instance of the trading system
            data (np.ndarray): Scaled feature data
            initial_cash (float): Starting cash balance
            max_position (float): Maximum percentage of portfolio to hold in Bitcoin
        """
        self.trading_system = trading_system
        self.data = data
        self.initial_cash = initial_cash
        self.max_position = max_position
        self.reset()

    def reset(self):
        """Reset the environment to initial state."""
        self.cash = self.initial_cash
        self.bitcoin_held = 0
        self.current_step = self.trading_system.seq_length
        self.portfolio_value = self.cash
        return self._get_state()

    def _get_state(self):
        """Get the current state for the DQN model."""
        price_history = self.data[self.current_step - self.trading_system.seq_length:self.current_step]
        sentiment_score = np.random.uniform(-1, 1)  # Simulate sentiment (replace with real data)
        predicted_price = self.trading_system.predict_price(price_history)
        state = np.concatenate([price_history.flatten(), [sentiment_score, predicted_price, self.cash, self.bitcoin_held]])
        return state

    def step(self, action):
        """Execute an action and return the next state, reward, and done flag."""
        # Using 'Close' price which is at index 0 in our scaled data
        current_price_scaled = self.data[self.current_step, 0]
        current_price = self.trading_system.price_scaler.inverse_transform([[current_price_scaled]])[0, 0]
        
        if action == 0:  # Sell 10% of held Bitcoin
            sell_amount = 0.1 * self.bitcoin_held
            self.cash += sell_amount * current_price * (1 - TRANSACTION_FEE)
            self.bitcoin_held -= sell_amount
        elif action == 2:  # Buy with 10% of cash, respecting max position
            cash_to_spend = 0.1 * self.cash
            buy_amount = cash_to_spend / current_price
            potential_position = (self.bitcoin_held + buy_amount) * current_price / self.portfolio_value
            if potential_position <= self.max_position:
                self.bitcoin_held += buy_amount
                self.cash -= cash_to_spend * (1 + TRANSACTION_FEE)
        
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
            reward = 0
            next_state = None
        else:
            next_price_scaled = self.data[self.current_step, 0]
            next_price = self.trading_system.price_scaler.inverse_transform([[next_price_scaled]])[0, 0]
            next_portfolio_value = self.cash + self.bitcoin_held * next_price
            reward = next_portfolio_value - self.portfolio_value
            self.portfolio_value = next_portfolio_value
            done = False
            next_state = self._get_state()
        return next_state, reward, done

class AITradingSystem:
    def __init__(self, ticker='BTC-USD', start_date='2020-01-01', end_date='2024-03-11', seq_length=60):
        self.ticker = ticker
        self.seq_length = seq_length
        self.raw_data = yf.download(ticker, start=start_date, end=end_date)
        
        # Add technical indicators if they don't already exist in the data
        self.add_technical_indicators()
        
        if len(self.raw_data) < self.seq_length + 1:
            raise ValueError(f"Insufficient data. Need at least {self.seq_length + 1} data points.")
        
        self.prepare_data()
        self.create_models()
        self.env = TradingEnvironment(self, self.scaled_features)

    def add_technical_indicators(self):
        """
        Check if technical indicators exist in the data, if not, calculate them.
        Assumes the imported data might already have these columns.
        """
        # Only add indicators if they don't already exist
        if 'SMA20' not in self.raw_data.columns:
            # Calculate SMAs
            self.raw_data['SMA20'] = self.raw_data['Close'].rolling(window=20).mean()
            self.raw_data['SMA50'] = self.raw_data['Close'].rolling(window=50).mean()
            self.raw_data['SMA200'] = self.raw_data['Close'].rolling(window=200).mean()
            
            # Calculate RSI
            period = 14
            delta = self.raw_data['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
            rs = avg_gain / avg_loss
            self.raw_data['RSI14'] = 100 - (100 / (1 + rs))
            
            # Calculate Bollinger Bands
            rolling_mean = self.raw_data['Close'].rolling(window=20).mean()
            rolling_std = self.raw_data['Close'].rolling(window=20).std()
            self.raw_data['BB_middle'] = rolling_mean
            self.raw_data['BB_upper'] = rolling_mean + (rolling_std * 2)
            self.raw_data['BB_lower'] = rolling_mean - (rolling_std * 2)
        
        # Drop NaN values that may result from calculating indicators
        self.raw_data = self.raw_data.dropna()

    def prepare_data(self):
        """Prepare and preprocess financial data with technical indicators."""
        # Select all relevant features for scaling
        feature_columns = ['Close', 'High', 'Low', 'Open', 'Volume', 
                          'SMA20', 'SMA50', 'SMA200', 'RSI14', 
                          'BB_middle', 'BB_upper', 'BB_lower']
        
        self.feature_scaler = MinMaxScaler()
        features = self.raw_data[feature_columns].values
        self.scaled_features = self.feature_scaler.fit_transform(features)
        
        # Create separate scaler just for the Close price for predictions
        self.price_scaler = MinMaxScaler()
        self.scaled_prices = self.price_scaler.fit_transform(self.raw_data[['Close']].values)
        
        # Create sequences for training
        self.X, self.y = self.create_sequences(self.scaled_features, self.scaled_prices)

    def create_sequences(self, data, targets):
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i + self.seq_length])
            y.append(targets[i + self.seq_length, 0])
        return np.array(X), np.array(y)

    def create_models(self):
        """Create predictive and decision-making models."""
        self.predictive_model = self.create_predictive_model()
        self.lstm_model = self.create_lstm_model()
        self.dqn_model = self.create_dqn_model()

    def create_predictive_model(self):
        """CNN-based model for price prediction with updated input shape."""
        num_features = self.scaled_features.shape[1]  # Number of features (now 12 instead of 4)
        
        model = keras.Sequential([
            keras.layers.Input(shape=(self.seq_length, num_features)),
            keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
            keras.layers.GlobalMaxPooling1D(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(self.X, self.y, epochs=2, batch_size=32, validation_split=0.2, verbose=0)
        return model

    def create_lstm_model(self):
        """LSTM model for time-series forecasting with updated input shape."""
        num_features = self.scaled_features.shape[1]  # Updated number of features
        
        model = keras.Sequential([
            keras.layers.Input(shape=(self.seq_length, num_features)),
            keras.layers.LSTM(64, activation='tanh', return_sequences=True),
            keras.layers.LSTM(32, activation='tanh'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(self.X, self.y, epochs=2, batch_size=32, validation_split=0.2, verbose=0)
        return model

    def create_dqn_model(self):
        """DQN model for trading decisions with updated input dimensions."""
        num_features = self.scaled_features.shape[1]
        input_dim = self.seq_length * num_features + 4  # Flattened features + sentiment + predicted_price + cash + bitcoin_held
        
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(3, activation='linear')  # Buy, hold, sell
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def predict_price(self, input_sequence):
        """Predict future price using ensemble of models."""
        try:
            if input_sequence.ndim == 2:
                input_sequence = input_sequence[np.newaxis, :, :]
            pred_nn = self.predictive_model.predict(input_sequence, verbose=0)[0, 0]
            pred_lstm = self.lstm_model.predict(input_sequence, verbose=0)[0, 0]
            combined_pred = (pred_nn + pred_lstm) / 2
            return self.price_scaler.inverse_transform([[combined_pred]])[0, 0]
        except Exception as e:
            print(f"Error in predict_price: {e}")
            return None

    def train_dqn(self, num_episodes=1, batch_size=32, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        """Train the DQN model using reinforcement learning."""
        replay_buffer = []
        epsilon = epsilon_start
        target_model = tf.keras.models.clone_model(self.dqn_model)
        target_model.set_weights(self.dqn_model.get_weights())

        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                if np.random.rand() < epsilon:
                    action = np.random.choice(3)
                else:
                    q_values = self.dqn_model.predict(state[np.newaxis, :], verbose=0)
                    action = np.argmax(q_values[0])

                next_state, reward, done = self.env.step(action)
                total_reward += reward

                if next_state is not None:
                    replay_buffer.append((state, action, reward, next_state, done))
                    state = next_state

                if len(replay_buffer) > batch_size:
                    batch = np.random.choice(len(replay_buffer), batch_size, replace=False)
                    states, actions, rewards, next_states, dones = zip(*[replay_buffer[i] for i in batch])
                    states = np.array(states)
                    next_states = np.array(next_states)
                    q_values_next = target_model.predict(next_states, verbose=0)
                    targets = rewards + gamma * np.max(q_values_next, axis=1) * (1 - np.array(dones))

                    q_values = self.dqn_model.predict(states, verbose=0)
                    for i, a in enumerate(actions):
                        q_values[i, a] = targets[i]

                    self.dqn_model.fit(states, q_values, epochs=1, verbose=0)

            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            if episode % 10 == 0:
                target_model.set_weights(self.dqn_model.get_weights())
                print(f"Episode {episode}, Total Reward: {total_reward}")

        print("DQN training completed.")
        
    def load_from_csv(self, csv_path):
        """
        Load data from a CSV file with technical indicators already calculated.
        """
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            print(f"Loaded data from {csv_path} with {len(df)} rows")
            print(f"Columns: {df.columns.tolist()}")
            
            # If the CSV has our technical indicators, use it directly
            self.raw_data = df
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
            
    # 修复评估交易策略性能的方法
    def evaluate_performance(self, start_date='2020-03-10', end_date='2021-02-23'):
        """
        评估交易策略在指定时间段内的表现
        
        Args:
            start_date (str): 评估起始日期
            end_date (str): 评估结束日期
            
        Returns:
            dict: 性能指标
        """
        try:
            # 过滤评估期间的数据
            eval_data = self.raw_data.loc[start_date:end_date].copy()
            
            # 初始化投资组合价值
            initial_cash = 10000
            initial_btc_price = eval_data.iloc[0]['Close']
            initial_btc_amount = initial_cash / initial_btc_price
            
            # 模拟交易
            cash = initial_cash
            btc_held = 0
            portfolio_values = []
            actions_taken = []
            
            # 创建跟踪每日表现的DataFrame
            performance_df = pd.DataFrame(index=eval_data.index)
            performance_df['Close'] = eval_data['Close']
            performance_df['Portfolio_Value'] = 0.0  # 初始化列
            
            # 获取评估期间的缩放数据
            feature_columns = ['Close', 'High', 'Low', 'Open', 'Volume', 
                            'SMA20', 'SMA50', 'SMA200', 'RSI14', 
                            'BB_middle', 'BB_upper', 'BB_lower']
            
            # 使用相同的缩放器缩放评估数据
            eval_features = eval_data[feature_columns].values
            scaled_eval_data = self.feature_scaler.transform(eval_features)
            
            # 确保有足够的数据用于预测
            if len(scaled_eval_data) <= self.seq_length:
                raise ValueError(f"评估数据不足: 需要>{self.seq_length}个数据点，但只有{len(scaled_eval_data)}个")
            
            # 逐日模拟交易
            for i in range(self.seq_length, len(scaled_eval_data)):
                # 获取当前状态
                price_history = scaled_eval_data[i-self.seq_length:i]
                sentiment_score = np.random.uniform(-1, 1)  # 模拟情绪
                predicted_price = self.predict_price(price_history)
                
                # 使用eval_data的索引
                current_idx = eval_data.index[i]
                current_price = eval_data.loc[current_idx, 'Close']
                current_portfolio_value = cash + btc_held * current_price
                
                # 准备DQN的状态
                state = np.concatenate([price_history.flatten(), 
                                    [sentiment_score, predicted_price, cash, btc_held]])
                
                # 从DQN获取行动
                q_values = self.dqn_model.predict(state[np.newaxis, :], verbose=0)
                action = np.argmax(q_values[0])
                actions_taken.append(action)
                
                # 执行行动
                if action == 0:  # 卖出10%持有的比特币
                    sell_amount = 0.1 * btc_held
                    cash += sell_amount * current_price * (1 - TRANSACTION_FEE)
                    btc_held -= sell_amount
                elif action == 2:  # 用10%的现金买入
                    cash_to_spend = 0.1 * cash
                    buy_amount = cash_to_spend / current_price
                    potential_position = (btc_held + buy_amount) * current_price / current_portfolio_value
                    if potential_position <= 0.5:  # 最大仓位50%
                        btc_held += buy_amount
                        cash -= cash_to_spend * (1 + TRANSACTION_FEE)
                
                # 记录投资组合价值
                portfolio_value = cash + btc_held * current_price
                portfolio_values.append(portfolio_value)
                performance_df.at[current_idx, 'Portfolio_Value'] = portfolio_value
            
            # 计算买入持有价值
            performance_df['Buy_Hold_Value'] = initial_btc_amount * performance_df['Close']
            
            # 填充可能的空值
            performance_df['Portfolio_Value'].fillna(method='ffill', inplace=True)
            performance_df = performance_df.dropna()
            
            # 计算每日回报率
            performance_df['Strategy_Return'] = performance_df['Portfolio_Value'].pct_change()
            performance_df['Buy_Hold_Return'] = performance_df['Buy_Hold_Value'].pct_change()
            
            # 计算性能指标
            trading_days = len(performance_df)
            
            # 策略指标
            strategy_returns = performance_df['Strategy_Return'].dropna()
            annualized_return = (1 + strategy_returns.mean()) ** 252 - 1
            annualized_volatility = strategy_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
            
            # 最大回撤计算
            cumulative_returns = (1 + strategy_returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdowns = (cumulative_returns / running_max) - 1
            max_drawdown = drawdowns.min()
            
            # 买入持有指标
            bh_returns = performance_df['Buy_Hold_Return'].dropna()
            bh_annualized_return = (1 + bh_returns.mean()) ** 252 - 1
            bh_annualized_volatility = bh_returns.std() * np.sqrt(252)
            bh_sharpe_ratio = bh_annualized_return / bh_annualized_volatility if bh_annualized_volatility != 0 else 0
            
            # 买入持有回撤
            bh_cumulative_returns = (1 + bh_returns).cumprod()
            bh_running_max = bh_cumulative_returns.cummax()
            bh_drawdowns = (bh_cumulative_returns / bh_running_max) - 1
            bh_max_drawdown = bh_drawdowns.min()
            
            # 整合结果
            metrics = {
                'evaluation_period': f"{start_date} to {end_date}",
                'trading_days': trading_days,
                'strategy': {
                    'annualized_return': annualized_return * 100,
                    'annualized_volatility': annualized_volatility * 100,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown * 100
                },
                'buy_hold': {
                    'annualized_return': bh_annualized_return * 100,
                    'annualized_volatility': bh_annualized_volatility * 100,
                    'sharpe_ratio': bh_sharpe_ratio,
                    'max_drawdown': bh_max_drawdown * 100
                },
                'actions_count': {
                    'buy': actions_taken.count(2),
                    'hold': actions_taken.count(1),
                    'sell': actions_taken.count(0)
                }
            }
            
            # 打印性能总结
            print(f"评估时间段: {start_date} 至 {end_date}")
            print(f"总交易天数: {trading_days}")
            print("--- 策略表现 ---")
            print(f"年化收益率: {metrics['strategy']['annualized_return']:.2f}%")
            print(f"年化波动率: {metrics['strategy']['annualized_volatility']:.2f}%")
            print(f"夏普比率: {metrics['strategy']['sharpe_ratio']:.2f}")
            print(f"最大回撤: {metrics['strategy']['max_drawdown']:.2f}%")
            print("--- 基准表现 (Buy & Hold) ---")
            print(f"年化收益率: {metrics['buy_hold']['annualized_return']:.2f}%")
            print(f"年化波动率: {metrics['buy_hold']['annualized_volatility']:.2f}%")
            print(f"夏普比率: {metrics['buy_hold']['sharpe_ratio']:.2f}")
            print(f"最大回撤: {metrics['buy_hold']['max_drawdown']:.2f}%")
            
            return metrics, performance_df
            
        except Exception as e:
            print(f"评估性能时出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    # 新增方法：保存模型
    def save_model(self, model_path='model.keras'):
        """
        将训练好的DQN模型保存到文件
        
        Args:
            model_path (str): 保存模型的路径
        """
        try:
            self.dqn_model.save(model_path)
            print(f"模型已保存到 {model_path}")
            return True
        except Exception as e:
            print(f"保存模型时出错: {e}")
            return False
    
    # 新增方法：保存缩放器
    def save_scalers(self, scaler_path='scaler.json'):
        """
        将特征和价格缩放器保存到JSON文件
        
        Args:
            scaler_path (str): 保存缩放器的路径
        """
        try:
            # 提取缩放器的最小值和最大值
            feature_scaler_data = {
                'data_min': self.feature_scaler.data_min_.tolist(),
                'data_max': self.feature_scaler.data_max_.tolist(),
                'data_range': self.feature_scaler.data_range_.tolist(),
            }
            
            price_scaler_data = {
                'data_min': self.price_scaler.data_min_.tolist(),
                'data_max': self.price_scaler.data_max_.tolist(),
                'data_range': self.price_scaler.data_range_.tolist(),
            }
            
            # 合并缩放器
            scaler_data = {
                'feature_scaler': feature_scaler_data,
                'price_scaler': price_scaler_data
            }
            
            # 保存到JSON
            with open(scaler_path, 'w') as f:
                json.dump(scaler_data, f)
                
            print(f"缩放器已保存到 {scaler_path}")
            return True
        except Exception as e:
            print(f"保存缩放器时出错: {e}")
            return False
    
    # 新增方法：加载模型
    def load_model(self, model_path='model.keras'):
        """
        从文件加载训练好的DQN模型
        
        Args:
            model_path (str): 保存的模型路径
        """
        try:
            self.dqn_model = keras.models.load_model(model_path)
            print(f"已从 {model_path} 加载模型")
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return False
    
    # 新增方法：加载缩放器
    def load_scalers(self, scaler_path='scaler.json'):
        """
        从JSON文件加载特征和价格缩放器
        
        Args:
            scaler_path (str): 保存的缩放器路径
        """
        try:
            # 从JSON加载
            with open(scaler_path, 'r') as f:
                scaler_data = json.load(f)
            
            # 重新创建特征缩放器
            feature_scaler = MinMaxScaler()
            feature_scaler.data_min_ = np.array(scaler_data['feature_scaler']['data_min'])
            feature_scaler.data_max_ = np.array(scaler_data['feature_scaler']['data_max'])
            feature_scaler.data_range_ = np.array(scaler_data['feature_scaler']['data_range'])
            self.feature_scaler = feature_scaler
            
            # 重新创建价格缩放器
            price_scaler = MinMaxScaler()
            price_scaler.data_min_ = np.array(scaler_data['price_scaler']['data_min'])
            price_scaler.data_max_ = np.array(scaler_data['price_scaler']['data_max'])
            price_scaler.data_range_ = np.array(scaler_data['price_scaler']['data_range'])
            self.price_scaler = price_scaler
            
            print(f"已从 {scaler_path} 加载缩放器")
            return True
        except Exception as e:
            print(f"加载缩放器时出错: {e}")
            return False

def main():
    try:
        # 创建交易系统
        trading_system = AITradingSystem()
        
        # 可选：从CSV加载预先计算好指标的数据
        # trading_system.load_from_csv("C:/Users/User/Downloads/btc_technical_indicators.csv")
        
        # 训练系统
        trading_system.train_dqn(num_episodes=1)  # 训练1个回合
        
        # 评估系统
        metrics, performance_df = trading_system.evaluate_performance('2020-03-10', '2021-02-23')
        
        # 保存模型和缩放器
        trading_system.save_model('model.keras')
        trading_system.save_scalers('scaler.json')
        
        # 使用训练好的DQN模型模拟交易
        state = trading_system.env.reset()
        done = False
        actions = []
        while not done:
            q_values = trading_system.dqn_model.predict(state[np.newaxis, :], verbose=0)
            action = np.argmax(q_values[0])
            actions.append(action)
            state, _, done = trading_system.env.step(action)
        
        print(f"模拟交易行动: {actions}")
        print(f"最终投资组合价值: ${trading_system.env.portfolio_value:.2f}")
    except Exception as e:
        print(f"主函数中出错: {e}")

if __name__ == "__main__":
    main()