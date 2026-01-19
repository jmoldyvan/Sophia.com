"""
Gold Futures Reinforcement Learning Trading Agent
Implements Q-Learning and Deep Q-Network (DQN) for automated trading
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, namedtuple
import random
import warnings
warnings.filterwarnings('ignore')

import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Configuration
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def load_and_preprocess_data(filepath):
    """Load and preprocess gold futures data with technical indicators."""
    print("=" * 70)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.replace('\ufeff', '')
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    
    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = df[col].str.replace(',', '').astype(float)
    
    def parse_volume(vol):
        if vol == '-' or pd.isna(vol):
            return np.nan
        vol = str(vol).replace(',', '')
        if 'K' in vol:
            return float(vol.replace('K', '')) * 1000
        elif 'M' in vol:
            return float(vol.replace('M', '')) * 1000000
        return float(vol)
    
    df['Volume'] = df['Vol.'].apply(parse_volume)
    df['Change_Pct'] = df['Change %'].str.replace('%', '').astype(float)
    df = df.drop(['Vol.', 'Change %'], axis=1)
    df = df.sort_values('Date').reset_index(drop=True)
    df['Volume'] = df['Volume'].fillna(method='ffill').fillna(method='bfill')
    
    # Technical Indicators
    df['SMA_5'] = df['Price'].rolling(window=5).mean()
    df['SMA_10'] = df['Price'].rolling(window=10).mean()
    df['SMA_20'] = df['Price'].rolling(window=20).mean()
    
    df['Price_to_SMA5'] = df['Price'] / df['SMA_5'] - 1
    df['Price_to_SMA10'] = df['Price'] / df['SMA_10'] - 1
    df['Price_to_SMA20'] = df['Price'] / df['SMA_20'] - 1
    
    # RSI
    delta = df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_normalized'] = df['RSI'] / 100
    
    # MACD
    df['EMA_12'] = df['Price'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_normalized'] = df['MACD'] / df['Price']
    
    # Volatility and Returns
    df['Returns'] = df['Price'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=10).std()
    
    # Bollinger Bands
    bb_middle = df['Price'].rolling(window=20).mean()
    bb_std = df['Price'].rolling(window=20).std()
    df['BB_position'] = (df['Price'] - bb_middle) / (2 * bb_std)
    
    # Momentum
    df['Momentum_5'] = df['Price'].pct_change(periods=5)
    df['Momentum_10'] = df['Price'].pct_change(periods=10)
    
    df = df.dropna().reset_index(drop=True)
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Price range: ${df['Price'].min():.2f} - ${df['Price'].max():.2f}")
    
    return df


# =============================================================================
# TRADING ENVIRONMENT
# =============================================================================

class GoldTradingEnv(gym.Env):
    """
    Custom Gymnasium environment for gold futures trading.
    
    State: Technical indicators + position + unrealized P&L
    Actions: 0=Hold, 1=Buy, 2=Sell
    Reward: Percentage change in portfolio value
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, df, initial_balance=10000, transaction_cost=0.001, window_size=10):
        super(GoldTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        
        self.feature_cols = [
            'Price_to_SMA5', 'Price_to_SMA10', 'Price_to_SMA20',
            'RSI_normalized', 'MACD_normalized', 'BB_position',
            'Momentum_5', 'Momentum_10', 'Volatility', 'Returns'
        ]
        
        self.features = np.clip(self.df[self.feature_cols].values, -3, 3)
        self.prices = self.df['Price'].values
        self.dates = self.df['Date'].values
        
        self.action_space = spaces.Discrete(3)
        state_size = len(self.feature_cols) + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0
        
        self.balance_history = [self.initial_balance]
        self.position_history = [0]
        self.action_history = []
        
        return self._get_state(), {}
    
    def _get_state(self):
        features = self.features[self.current_step]
        position_indicator = 1.0 if self.position > 0 else 0.0
        
        if self.position > 0:
            unrealized_pnl = (self.prices[self.current_step] - self.entry_price) / self.entry_price
        else:
            unrealized_pnl = 0.0
        
        return np.concatenate([features, [position_indicator, unrealized_pnl]]).astype(np.float32)
    
    def step(self, action):
        current_price = self.prices[self.current_step]
        prev_portfolio_value = self._get_portfolio_value()
        
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            cost = current_price * (1 + self.transaction_cost)
            self.position = self.balance / cost
            self.entry_price = current_price
            self.balance = 0
            self.total_trades += 1
                
        elif action == 2 and self.position > 0:  # Sell
            proceeds = self.position * current_price * (1 - self.transaction_cost)
            profit = proceeds - (self.position * self.entry_price)
            if profit > 0:
                self.winning_trades += 1
            self.total_profit += profit
            self.balance = proceeds
            self.position = 0
            self.entry_price = 0
        
        self.current_step += 1
        new_portfolio_value = self._get_portfolio_value()
        
        # Reward
        reward = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value * 100
        if action == 0 and self.position == 0:
            reward -= 0.01
        
        self.balance_history.append(new_portfolio_value)
        self.position_history.append(1 if self.position > 0 else 0)
        self.action_history.append(action)
        
        terminated = self.current_step >= len(self.prices) - 1
        
        # Force sell at end
        if terminated and self.position > 0:
            proceeds = self.position * self.prices[self.current_step] * (1 - self.transaction_cost)
            profit = proceeds - (self.position * self.entry_price)
            if profit > 0:
                self.winning_trades += 1
            self.total_profit += profit
            self.balance = proceeds
            self.position = 0
        
        info = {'portfolio_value': new_portfolio_value, 'total_trades': self.total_trades}
        return self._get_state(), reward, terminated, False, info
    
    def _get_portfolio_value(self):
        return self.balance + self.position * self.prices[self.current_step]
    
    def get_performance_metrics(self):
        final_value = self._get_portfolio_value()
        total_return = (final_value - self.initial_balance) / self.initial_balance * 100
        
        returns = np.diff(self.balance_history) / self.balance_history[:-1]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        peak = np.maximum.accumulate(self.balance_history)
        drawdown = (peak - self.balance_history) / peak
        max_drawdown = np.max(drawdown) * 100
        
        win_rate = self.winning_trades / max(self.total_trades, 1) * 100
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate
        }


# =============================================================================
# Q-LEARNING AGENT
# =============================================================================

class QLearningAgent:
    """Tabular Q-Learning agent with discretized states."""
    
    def __init__(self, state_bins=10, n_actions=3, learning_rate=0.1,
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_bins = state_bins
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
        self.training_rewards = []
        self.epsilon_history = []
    
    def discretize_state(self, state):
        discretized = []
        for val in state:
            val = np.clip(val, -2, 2)
            bin_idx = int((val + 2) / 4 * self.state_bins)
            bin_idx = min(bin_idx, self.state_bins - 1)
            discretized.append(bin_idx)
        return tuple(discretized)
    
    def get_q_values(self, state):
        discrete_state = self.discretize_state(state)
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.n_actions)
        return self.q_table[discrete_state]
    
    def choose_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.get_q_values(state))
    
    def update(self, state, action, reward, next_state, done):
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.n_actions)
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(self.n_actions)
        
        current_q = self.q_table[discrete_state][action]
        target_q = reward if done else reward + self.gamma * np.max(self.q_table[discrete_next_state])
        self.q_table[discrete_state][action] += self.lr * (target_q - current_q)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# =============================================================================
# DEEP Q-NETWORK AGENT
# =============================================================================

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(device)
        actions = torch.LongTensor(np.array([e.action for e in experiences])).to(device)
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).to(device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(device)
        dones = torch.FloatTensor(np.array([e.done for e in experiences])).to(device)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNetwork(nn.Module):
    """Neural network for Q-value approximation."""
    
    def __init__(self, state_size, action_size, hidden_sizes=[128, 64, 32]):
        super(DQNetwork, self).__init__()
        
        layers = []
        prev_size = state_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, action_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """Deep Q-Network agent with experience replay and target network."""
    
    def __init__(self, state_size, action_size, learning_rate=0.001,
                 discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01, buffer_size=10000, batch_size=64, target_update_freq=10):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.policy_net = DQNetwork(state_size, action_size).to(device)
        self.target_net = DQNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        
        self.training_losses = []
        self.training_rewards = []
        self.epsilon_history = []
        self.update_count = 0
    
    def choose_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            return self.policy_net(state_tensor).argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_q_learning(env, agent, n_episodes=500, verbose=True):
    """Train Q-Learning agent."""
    print("\n" + "=" * 70)
    print("TRAINING Q-LEARNING AGENT")
    print("=" * 70)
    
    episode_rewards = []
    episode_returns = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        agent.decay_epsilon()
        metrics = env.get_performance_metrics()
        episode_rewards.append(total_reward)
        episode_returns.append(metrics['total_return'])
        agent.training_rewards.append(total_reward)
        agent.epsilon_history.append(agent.epsilon)
        
        if verbose and (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {np.mean(episode_rewards[-50:]):.2f} | "
                  f"Avg Return: {np.mean(episode_returns[-50:]):.2f}% | "
                  f"Epsilon: {agent.epsilon:.4f}")
    
    return {'episode_rewards': episode_rewards, 'episode_returns': episode_returns,
            'epsilon_history': agent.epsilon_history}


def train_dqn(env, agent, n_episodes=300, verbose=True):
    """Train DQN agent."""
    print("\n" + "=" * 70)
    print("TRAINING DEEP Q-NETWORK AGENT")
    print("=" * 70)
    
    episode_rewards = []
    episode_returns = []
    episode_losses = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        total_loss = 0
        loss_count = 0
        done = False
        
        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_experience(state, action, reward, next_state, done)
            loss = agent.train_step()
            
            if loss is not None:
                total_loss += loss
                loss_count += 1
            
            state = next_state
            total_reward += reward
        
        agent.decay_epsilon()
        metrics = env.get_performance_metrics()
        episode_rewards.append(total_reward)
        episode_returns.append(metrics['total_return'])
        avg_loss = total_loss / max(loss_count, 1)
        episode_losses.append(avg_loss)
        
        agent.training_rewards.append(total_reward)
        agent.epsilon_history.append(agent.epsilon)
        agent.training_losses.append(avg_loss)
        
        if verbose and (episode + 1) % 30 == 0:
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {np.mean(episode_rewards[-30:]):.2f} | "
                  f"Avg Return: {np.mean(episode_returns[-30:]):.2f}% | "
                  f"Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.4f}")
    
    return {'episode_rewards': episode_rewards, 'episode_returns': episode_returns,
            'episode_losses': episode_losses, 'epsilon_history': agent.epsilon_history}


def evaluate_agent(env, agent, agent_type='DQN'):
    """Evaluate trained agent on test data."""
    print(f"\n{'=' * 70}")
    print(f"EVALUATING {agent_type} AGENT")
    print("=" * 70)
    
    state, _ = env.reset()
    done = False
    
    while not done:
        action = agent.choose_action(state, training=False)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    metrics = env.get_performance_metrics()
    action_counts = np.bincount(env.action_history, minlength=3)
    action_pcts = action_counts / len(env.action_history) * 100
    
    print(f"\n{agent_type} Agent Performance:")
    print(f"  Initial Balance:   $10,000.00")
    print(f"  Final Value:       ${metrics['final_value']:,.2f}")
    print(f"  Total Return:      {metrics['total_return']:.2f}%")
    print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown:      {metrics['max_drawdown']:.2f}%")
    print(f"  Total Trades:      {metrics['total_trades']}")
    print(f"  Win Rate:          {metrics['win_rate']:.1f}%")
    print(f"\nAction Distribution: Hold: {action_pcts[0]:.1f}% | Buy: {action_pcts[1]:.1f}% | Sell: {action_pcts[2]:.1f}%")
    
    return metrics, env


def run_buy_and_hold(env):
    """Run buy-and-hold baseline strategy."""
    state, _ = env.reset()
    env.step(1)  # Buy
    done = False
    while not done:
        state, reward, terminated, truncated, info = env.step(0)  # Hold
        done = terminated or truncated
    return env.get_performance_metrics()


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_training_plots(q_history, dqn_history, save_path='rl_training_results.png'):
    """Create training visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Q-Learning rewards
    ax1 = axes[0, 0]
    ax1.plot(q_history['episode_rewards'], alpha=0.3, color='blue')
    ax1.plot(pd.Series(q_history['episode_rewards']).rolling(50).mean(), color='blue', linewidth=2)
    ax1.set_title('Q-Learning: Episode Rewards', fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    
    # DQN rewards
    ax2 = axes[0, 1]
    ax2.plot(dqn_history['episode_rewards'], alpha=0.3, color='green')
    ax2.plot(pd.Series(dqn_history['episode_rewards']).rolling(30).mean(), color='green', linewidth=2)
    ax2.set_title('DQN: Episode Rewards', fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.grid(True, alpha=0.3)
    
    # Returns comparison
    ax3 = axes[0, 2]
    ax3.plot(q_history['episode_returns'], alpha=0.5, label='Q-Learning', color='blue')
    ax3.plot(dqn_history['episode_returns'], alpha=0.5, label='DQN', color='green')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Portfolio Returns', fontweight='bold')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Return (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Epsilon decay
    ax4 = axes[1, 0]
    ax4.plot(q_history['epsilon_history'], label='Q-Learning', color='blue')
    ax4.plot(dqn_history['epsilon_history'], label='DQN', color='green')
    ax4.set_title('Epsilon Decay', fontweight='bold')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Epsilon')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # DQN Loss
    ax5 = axes[1, 1]
    ax5.plot(dqn_history['episode_losses'], color='red', alpha=0.5)
    ax5.plot(pd.Series(dqn_history['episode_losses']).rolling(30).mean(), color='red', linewidth=2)
    ax5.set_title('DQN: Training Loss', fontweight='bold')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Loss')
    ax5.grid(True, alpha=0.3)
    
    # Learning progress
    ax6 = axes[1, 2]
    ax6.plot(pd.Series(q_history['episode_returns']).rolling(50).mean(), label='Q-Learning', color='blue', linewidth=2)
    ax6.plot(pd.Series(dqn_history['episode_returns']).rolling(30).mean(), label='DQN', color='green', linewidth=2)
    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax6.set_title('Learning Progress (Rolling Avg)', fontweight='bold')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Avg Return (%)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    return fig


def create_trading_plots(env, df, agent_type, save_path='rl_trading_results.png'):
    """Create trading results visualization."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    
    start_idx = env.window_size
    dates = df['Date'].iloc[start_idx:start_idx + len(env.balance_history)].values
    prices = df['Price'].iloc[start_idx:start_idx + len(env.balance_history)].values
    
    # Portfolio value
    ax1 = axes[0, 0]
    ax1.plot(dates, env.balance_history, color='green', linewidth=2)
    ax1.axhline(y=env.initial_balance, color='gray', linestyle='--')
    ax1.set_title(f'{agent_type}: Portfolio Value', fontweight='bold')
    ax1.set_ylabel('Value ($)')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Trading signals
    ax2 = axes[0, 1]
    ax2.plot(dates, prices, color='gold', linewidth=1.5)
    actions = np.array(env.action_history)
    buy_idx = np.where(actions == 1)[0]
    sell_idx = np.where(actions == 2)[0]
    if len(buy_idx) > 0:
        ax2.scatter(dates[buy_idx], prices[buy_idx], color='green', marker='^', s=100, label='Buy', zorder=5)
    if len(sell_idx) > 0:
        ax2.scatter(dates[sell_idx], prices[sell_idx], color='red', marker='v', s=100, label='Sell', zorder=5)
    ax2.set_title(f'{agent_type}: Trading Signals', fontweight='bold')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Position
    ax3 = axes[1, 0]
    ax3.fill_between(dates, env.position_history, alpha=0.5, color='blue', step='pre')
    ax3.set_title(f'{agent_type}: Position', fontweight='bold')
    ax3.set_ylabel('Position (0=None, 1=Long)')
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Returns comparison
    ax4 = axes[1, 1]
    portfolio_returns = (np.array(env.balance_history) - env.initial_balance) / env.initial_balance * 100
    price_returns = (prices - prices[0]) / prices[0] * 100
    ax4.plot(dates, portfolio_returns, label=f'{agent_type}', color='green', linewidth=2)
    ax4.plot(dates, price_returns, label='Buy & Hold', color='orange', linewidth=2)
    ax4.axhline(y=0, color='gray', linestyle='--')
    ax4.set_title('Cumulative Returns', fontweight='bold')
    ax4.set_ylabel('Return (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Action distribution
    ax5 = axes[2, 0]
    action_counts = np.bincount(env.action_history, minlength=3)
    bars = ax5.bar(['Hold', 'Buy', 'Sell'], action_counts, color=['gray', 'green', 'red'])
    ax5.set_title(f'{agent_type}: Action Distribution', fontweight='bold')
    ax5.set_ylabel('Count')
    for bar, count in zip(bars, action_counts):
        ax5.annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Drawdown
    ax6 = axes[2, 1]
    peak = np.maximum.accumulate(env.balance_history)
    drawdown = (np.array(env.balance_history) - peak) / peak * 100
    ax6.fill_between(dates, drawdown, 0, alpha=0.5, color='red')
    ax6.set_title(f'{agent_type}: Drawdown', fontweight='bold')
    ax6.set_ylabel('Drawdown (%)')
    ax6.grid(True, alpha=0.3)
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    return fig


def create_comparison_chart(q_metrics, dqn_metrics, bh_metrics, save_path='rl_comparison.png'):
    """Create strategy comparison chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Metrics comparison
    ax1 = axes[0]
    metrics_names = ['Total Return (%)', 'Sharpe Ratio', 'Win Rate (%)']
    x = np.arange(len(metrics_names))
    width = 0.25
    
    q_vals = [q_metrics['total_return'], q_metrics['sharpe_ratio'], q_metrics['win_rate']]
    dqn_vals = [dqn_metrics['total_return'], dqn_metrics['sharpe_ratio'], dqn_metrics['win_rate']]
    bh_vals = [bh_metrics['total_return'], bh_metrics['sharpe_ratio'], bh_metrics.get('win_rate', 0)]
    
    ax1.bar(x - width, q_vals, width, label='Q-Learning', color='blue', alpha=0.7)
    ax1.bar(x, dqn_vals, width, label='DQN', color='green', alpha=0.7)
    ax1.bar(x + width, bh_vals, width, label='Buy & Hold', color='orange', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names)
    ax1.set_title('Performance Metrics', fontweight='bold')
    ax1.legend()
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Final values
    ax2 = axes[1]
    strategies = ['Q-Learning', 'DQN', 'Buy & Hold']
    final_vals = [q_metrics['final_value'], dqn_metrics['final_value'], bh_metrics['final_value']]
    bars = ax2.bar(strategies, final_vals, color=['blue', 'green', 'orange'], alpha=0.7)
    ax2.axhline(y=10000, color='gray', linestyle='--', label='Initial')
    ax2.set_title('Final Portfolio Value', fontweight='bold')
    ax2.set_ylabel('Value ($)')
    ax2.legend()
    for bar, val in zip(bars, final_vals):
        ax2.annotate(f'${val:,.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the complete RL trading pipeline."""
    
    print("\n" + "=" * 70)
    print("   GOLD FUTURES RL TRADING AGENT")
    print("   Reinforcement Learning for Automated Trading")
    print("=" * 70)
    print(f"\nUsing device: {device}")
    
    # Load data
    df = load_and_preprocess_data('Gold Futures Historical Data.csv')
    
    # Train/test split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)
    
    print(f"\nTraining samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    
    # Create environments
    train_env = GoldTradingEnv(train_df, initial_balance=10000)
    test_env = GoldTradingEnv(test_df, initial_balance=10000)
    
    state_size = train_env.observation_space.shape[0]
    action_size = train_env.action_space.n
    
    print(f"\nState size: {state_size}")
    print(f"Action size: {action_size} (Hold, Buy, Sell)")
    
    # Initialize agents
    print("\n" + "=" * 70)
    print("INITIALIZING AGENTS")
    print("=" * 70)
    
    q_agent = QLearningAgent(state_bins=15, n_actions=action_size)
    print("\nQ-Learning Agent: bins=15, lr=0.1, gamma=0.95")
    
    dqn_agent = DQNAgent(state_size=state_size, action_size=action_size)
    print(f"DQN Agent: {state_size}->128->64->32->{action_size}, lr=0.001, gamma=0.99")
    
    # Train agents
    q_history = train_q_learning(train_env, q_agent, n_episodes=500)
    dqn_history = train_dqn(train_env, dqn_agent, n_episodes=300)
    
    # Evaluate on test data
    print("\n" + "=" * 70)
    print("EVALUATION ON TEST DATA")
    print("=" * 70)
    
    q_metrics, q_env = evaluate_agent(test_env, q_agent, 'Q-Learning')
    
    test_env_dqn = GoldTradingEnv(test_df, initial_balance=10000)
    dqn_metrics, dqn_env = evaluate_agent(test_env_dqn, dqn_agent, 'DQN')
    
    bh_env = GoldTradingEnv(test_df, initial_balance=10000)
    bh_metrics = run_buy_and_hold(bh_env)
    
    print(f"\nBuy-and-Hold Baseline:")
    print(f"  Final Value:  ${bh_metrics['final_value']:,.2f}")
    print(f"  Total Return: {bh_metrics['total_return']:.2f}%")
    
    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    create_training_plots(q_history, dqn_history)
    
    best_agent = 'DQN' if dqn_metrics['total_return'] > q_metrics['total_return'] else 'Q-Learning'
    best_env = dqn_env if best_agent == 'DQN' else q_env
    create_trading_plots(best_env, test_df, best_agent)
    
    create_comparison_chart(q_metrics, dqn_metrics, bh_metrics)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print("\n┌─────────────────┬──────────────┬──────────────┬──────────────┐")
    print("│     Metric      │  Q-Learning  │     DQN      │  Buy & Hold  │")
    print("├─────────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ Final Value ($) │ {q_metrics['final_value']:>12,.2f} │ {dqn_metrics['final_value']:>12,.2f} │ {bh_metrics['final_value']:>12,.2f} │")
    print(f"│ Total Return (%)│ {q_metrics['total_return']:>12.2f} │ {dqn_metrics['total_return']:>12.2f} │ {bh_metrics['total_return']:>12.2f} │")
    print(f"│ Sharpe Ratio    │ {q_metrics['sharpe_ratio']:>12.4f} │ {dqn_metrics['sharpe_ratio']:>12.4f} │ {bh_metrics['sharpe_ratio']:>12.4f} │")
    print(f"│ Max Drawdown (%)│ {q_metrics['max_drawdown']:>12.2f} │ {dqn_metrics['max_drawdown']:>12.2f} │ {bh_metrics['max_drawdown']:>12.2f} │")
    print(f"│ Total Trades    │ {q_metrics['total_trades']:>12} │ {dqn_metrics['total_trades']:>12} │ {'1':>12} │")
    print(f"│ Win Rate (%)    │ {q_metrics['win_rate']:>12.1f} │ {dqn_metrics['win_rate']:>12.1f} │ {'N/A':>12} │")
    print("└─────────────────┴──────────────┴──────────────┴──────────────┘")
    
    print(f"\n🏆 Best Performing Agent: {best_agent}")
    
    print("\n" + "=" * 70)
    print("COMPLETE - Generated Files:")
    print("  • rl_training_results.png")
    print("  • rl_trading_results.png")
    print("  • rl_comparison.png")
    print("=" * 70)
    
    return {
        'q_agent': q_agent, 'dqn_agent': dqn_agent,
        'q_metrics': q_metrics, 'dqn_metrics': dqn_metrics, 'bh_metrics': bh_metrics
    }


if __name__ == "__main__":
    results = main()