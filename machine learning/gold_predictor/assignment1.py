"""
GoldPredict
=====================================================
This application predicts gold futures prices using historical data
and multiple machine learning algorithms including:
- Support Vector Regression (SVR)
- Random Forest Regressor
- Neural Network (MLPRegressor)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)

# Configure plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_clean_data(filepath):
    """
    Load and clean the gold futures historical data.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe with proper data types
    """
    print("=" * 60)
    print("STEP 1: LOADING AND CLEANING DATA")
    print("=" * 60)
    
    # Load the data
    df = pd.read_csv(filepath)
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Clean column names (remove BOM if present)
    df.columns = df.columns.str.replace('\ufeff', '')
    
    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    
    # Clean numeric columns (remove commas and convert to float)
    numeric_cols = ['Price', 'Open', 'High', 'Low']
    for col in numeric_cols:
        df[col] = df[col].str.replace(',', '').astype(float)
    
    # Clean Volume column (handle 'K' suffix and '-' values)
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
    
    # Clean Change % column
    df['Change_Pct'] = df['Change %'].str.replace('%', '').astype(float)
    
    # Drop original columns we've transformed
    df = df.drop(['Vol.', 'Change %'], axis=1)
    
    # Sort by date (oldest first for time series)
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Fill missing volumes with forward/backward fill
    df['Volume'] = df['Volume'].fillna(method='ffill').fillna(method='bfill')
    
    print(f"\nCleaned data shape: {df.shape}")
    print(f"\nDate range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nBasic statistics:\n{df.describe()}")
    
    return df


# =============================================================================
# SECTION 2: FEATURE ENGINEERING
# =============================================================================

def create_technical_indicators(df):
    """
    Create technical indicators as features for the ML models.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Clean dataframe with OHLCV data
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added technical indicator features
    """
    print("\n" + "=" * 60)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 60)
    
    df = df.copy()
    
    # 1. Moving Averages
    df['SMA_5'] = df['Price'].rolling(window=5).mean()
    df['SMA_10'] = df['Price'].rolling(window=10).mean()
    df['SMA_20'] = df['Price'].rolling(window=20).mean()
    df['SMA_50'] = df['Price'].rolling(window=50).mean()
    
    # 2. Exponential Moving Averages
    df['EMA_12'] = df['Price'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Price'].ewm(span=26, adjust=False).mean()
    
    # 3. MACD (Moving Average Convergence Divergence)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 4. RSI (Relative Strength Index)
    delta = df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 5. Bollinger Bands
    df['BB_Middle'] = df['Price'].rolling(window=20).mean()
    bb_std = df['Price'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # 6. Price-based features
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Change'] = df['Price'].diff()
    df['Price_Pct_Change'] = df['Price'].pct_change() * 100
    
    # 7. Volatility (Standard deviation of returns)
    df['Volatility_5'] = df['Price_Pct_Change'].rolling(window=5).std()
    df['Volatility_20'] = df['Price_Pct_Change'].rolling(window=20).std()
    
    # 8. Momentum indicators
    df['Momentum_5'] = df['Price'] - df['Price'].shift(5)
    df['Momentum_10'] = df['Price'] - df['Price'].shift(10)
    df['ROC_5'] = ((df['Price'] - df['Price'].shift(5)) / df['Price'].shift(5)) * 100
    
    # 9. Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Price'].shift())
    low_close = np.abs(df['Low'] - df['Price'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean()
    
    # 10. Lag features (previous day prices)
    for lag in [1, 2, 3, 5, 7]:
        df[f'Price_Lag_{lag}'] = df['Price'].shift(lag)
    
    # 11. Time-based features
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Year'] = df['Date'].dt.year
    
    # 12. Target variable: Next day's price
    df['Target'] = df['Price'].shift(-1)
    
    # Drop rows with NaN values (due to rolling calculations)
    initial_rows = len(df)
    df = df.dropna()
    final_rows = len(df)
    
    print(f"\nCreated {len(df.columns) - 8} technical indicator features")
    print(f"Rows removed due to NaN: {initial_rows - final_rows}")
    print(f"Final dataset shape: {df.shape}")
    print(f"\nFeatures created:")
    
    features = [col for col in df.columns if col not in ['Date', 'Target']]
    for i, feat in enumerate(features, 1):
        print(f"  {i:2d}. {feat}")
    
    return df


# =============================================================================
# SECTION 3: DATA PREPARATION FOR MODELING
# =============================================================================

def prepare_data_for_modeling(df, test_size=0.2):
    """
    Prepare features and target for machine learning models.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with features and target
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    print("\n" + "=" * 60)
    print("STEP 3: DATA PREPARATION FOR MODELING")
    print("=" * 60)
    
    # Define feature columns (exclude Date and Target)
    feature_cols = [col for col in df.columns if col not in ['Date', 'Target']]
    
    X = df[feature_cols].values
    y = df['Target'].values
    
    # For time series, we use a chronological split (not random)
    split_idx = int(len(X) * (1 - test_size))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"\nTraining period: {df['Date'].iloc[0]} to {df['Date'].iloc[split_idx-1]}")
    print(f"Testing period: {df['Date'].iloc[split_idx]} to {df['Date'].iloc[-1]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler


# =============================================================================
# SECTION 4: MODEL TRAINING AND EVALUATION
# =============================================================================

def evaluate_model(y_true, y_pred, model_name):
    """
    Calculate and display evaluation metrics for a model.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
    
    print(f"\n{model_name} Results:")
    print(f"  - Mean Squared Error (MSE):      {mse:.4f}")
    print(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  - Mean Absolute Error (MAE):      {mae:.4f}")
    print(f"  - R-squared (R²):                 {r2:.4f}")
    print(f"  - Mean Absolute % Error (MAPE):   {mape:.2f}%")
    
    return metrics


def train_svr_model(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Support Vector Regression model.
    """
    print("\n" + "-" * 50)
    print("Training Support Vector Regression (SVR)...")
    print("-" * 50)
    
    # SVR with RBF kernel
    svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    svr.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = svr.predict(X_train)
    y_pred_test = svr.predict(X_test)
    
    print("\nTraining Performance:")
    evaluate_model(y_train, y_pred_train, "SVR (Train)")
    
    print("\nTest Performance:")
    metrics = evaluate_model(y_test, y_pred_test, "SVR (Test)")
    
    return svr, y_pred_test, metrics


def train_random_forest_model(X_train, X_test, y_train, y_test, feature_names):
    """
    Train and evaluate Random Forest Regression model.
    """
    print("\n" + "-" * 50)
    print("Training Random Forest Regressor...")
    print("-" * 50)
    
    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    
    print("\nTraining Performance:")
    evaluate_model(y_train, y_pred_train, "Random Forest (Train)")
    
    print("\nTest Performance:")
    metrics = evaluate_model(y_test, y_pred_test, "Random Forest (Test)")
    
    # Feature importance
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    for i, row in importance.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    return rf, y_pred_test, metrics, importance


def train_neural_network_model(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Neural Network (MLP) Regression model.
    """
    print("\n" + "-" * 50)
    print("Training Neural Network (MLPRegressor)...")
    print("-" * 50)
    
    # Neural Network
    nn = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    nn.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = nn.predict(X_train)
    y_pred_test = nn.predict(X_test)
    
    print("\nTraining Performance:")
    evaluate_model(y_train, y_pred_train, "Neural Network (Train)")
    
    print("\nTest Performance:")
    metrics = evaluate_model(y_test, y_pred_test, "Neural Network (Test)")
    
    print(f"\nNeural Network converged in {nn.n_iter_} iterations")
    
    return nn, y_pred_test, metrics


# =============================================================================
# SECTION 5: VISUALIZATION
# =============================================================================

def create_visualizations(df, y_test, predictions_dict, feature_importance, test_dates):
    """
    Create comprehensive visualizations of the data and model results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe with all data
    y_test : array-like
        Actual test values
    predictions_dict : dict
        Dictionary of model predictions
    feature_importance : pd.DataFrame
        Feature importance from Random Forest
    test_dates : pd.Series
        Dates corresponding to test data
    """
    print("\n" + "=" * 60)
    print("STEP 5: CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Historical Gold Price with Moving Averages
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(df['Date'], df['Price'], label='Gold Price', linewidth=1.5, color='gold')
    ax1.plot(df['Date'], df['SMA_20'], label='SMA 20', linewidth=1, alpha=0.7)
    ax1.plot(df['Date'], df['SMA_50'], label='SMA 50', linewidth=1, alpha=0.7)
    ax1.fill_between(df['Date'], df['BB_Lower'], df['BB_Upper'], alpha=0.2, label='Bollinger Bands')
    ax1.set_title('Gold Futures Price with Technical Indicators', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Price Distribution
    ax2 = fig.add_subplot(4, 2, 2)
    ax2.hist(df['Price'], bins=50, color='gold', edgecolor='black', alpha=0.7)
    ax2.axvline(df['Price'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df["Price"].mean():.2f}')
    ax2.axvline(df['Price'].median(), color='blue', linestyle='--', linewidth=2, label=f'Median: ${df["Price"].median():.2f}')
    ax2.set_title('Gold Price Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Price (USD)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Model Predictions Comparison
    ax3 = fig.add_subplot(4, 2, 3)
    ax3.plot(test_dates, y_test, label='Actual Price', linewidth=2, color='black')
    colors = ['blue', 'green', 'red']
    for (name, pred), color in zip(predictions_dict.items(), colors):
        ax3.plot(test_dates, pred, label=f'{name} Prediction', linewidth=1.5, alpha=0.8, color=color)
    ax3.set_title('Model Predictions vs Actual Prices', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Price (USD)')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Prediction Errors Comparison
    ax4 = fig.add_subplot(4, 2, 4)
    for (name, pred), color in zip(predictions_dict.items(), colors):
        errors = y_test - pred
        ax4.plot(test_dates, errors, label=f'{name} Error', linewidth=1, alpha=0.7, color=color)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_title('Prediction Errors Over Time', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Error (USD)')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # 5. Feature Importance (Top 15)
    ax5 = fig.add_subplot(4, 2, 5)
    top_features = feature_importance.head(15)
    bars = ax5.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
    ax5.set_yticks(range(len(top_features)))
    ax5.set_yticklabels(top_features['Feature'])
    ax5.invert_yaxis()
    ax5.set_title('Top 15 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Importance')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # 6. Scatter Plot: Actual vs Predicted (Best Model)
    ax6 = fig.add_subplot(4, 2, 6)
    best_model = 'Random Forest'  # Usually performs best
    ax6.scatter(y_test, predictions_dict[best_model], alpha=0.5, color='steelblue', edgecolor='none')
    min_val, max_val = min(y_test.min(), predictions_dict[best_model].min()), max(y_test.max(), predictions_dict[best_model].max())
    ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax6.set_title(f'{best_model}: Actual vs Predicted', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Actual Price (USD)')
    ax6.set_ylabel('Predicted Price (USD)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. RSI and MACD Indicators
    ax7 = fig.add_subplot(4, 2, 7)
    ax7_twin = ax7.twinx()
    ax7.plot(df['Date'], df['RSI'], label='RSI', color='purple', linewidth=1)
    ax7.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
    ax7.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
    ax7.set_ylabel('RSI', color='purple')
    ax7.set_ylim(0, 100)
    
    ax7_twin.bar(df['Date'], df['MACD_Hist'], alpha=0.3, color='gray', label='MACD Histogram')
    ax7_twin.set_ylabel('MACD Histogram', color='gray')
    
    ax7.set_title('RSI and MACD Indicators', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Date')
    ax7.legend(loc='upper left')
    ax7.grid(True, alpha=0.3)
    
    # 8. Model Performance Comparison
    ax8 = fig.add_subplot(4, 2, 8)
    metrics_data = []
    for name, pred in predictions_dict.items():
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        metrics_data.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'R²': r2})
    
    metrics_df = pd.DataFrame(metrics_data)
    x = np.arange(len(metrics_df))
    width = 0.25
    
    bars1 = ax8.bar(x - width, metrics_df['RMSE'], width, label='RMSE', color='indianred')
    bars2 = ax8.bar(x, metrics_df['MAE'], width, label='MAE', color='steelblue')
    
    ax8.set_ylabel('Error (USD)')
    ax8.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels(metrics_df['Model'])
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Add R² values as text
    for i, r2 in enumerate(metrics_df['R²']):
        ax8.annotate(f'R²={r2:.4f}', xy=(i, metrics_df['RMSE'].iloc[i] + 5), 
                    ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('gold_price_analysis.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to 'gold_price_analysis.png'")
    
    return fig


def create_correlation_heatmap(df):
    """
    Create a correlation heatmap for the features.
    """
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdYlBu_r', 
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
    print("Correlation heatmap saved to 'correlation_heatmap.png'")
    
    return fig


# =============================================================================
# SECTION 6: MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to execute the gold price prediction pipeline.
    """
    print("\n" + "=" * 70)
    print("   GOLD FUTURES PRICE PREDICTION USING MACHINE LEARNING")
    print("=" * 70)
    print("\nThis application uses historical gold futures data to predict")
    print("future prices using multiple machine learning algorithms.\n")
    
    # 1. Load and clean data
    df = load_and_clean_data('Gold Futures Historical Data.csv')
    
    # 2. Create technical indicators
    df_features = create_technical_indicators(df)
    
    # Store test dates before splitting
    test_size = 0.2
    split_idx = int(len(df_features) * (1 - test_size))
    test_dates = df_features['Date'].iloc[split_idx:].reset_index(drop=True)
    
    # 3. Prepare data for modeling
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data_for_modeling(
        df_features, test_size=test_size
    )
    
    # 4. Train and evaluate models
    print("\n" + "=" * 60)
    print("STEP 4: MODEL TRAINING AND EVALUATION")
    print("=" * 60)
    
    # Dictionary to store predictions and metrics
    predictions = {}
    all_metrics = []
    
    # Train SVR
    svr_model, svr_pred, svr_metrics = train_svr_model(X_train, X_test, y_train, y_test)
    predictions['SVR'] = svr_pred
    all_metrics.append(svr_metrics)
    
    # Train Random Forest
    rf_model, rf_pred, rf_metrics, feature_importance = train_random_forest_model(
        X_train, X_test, y_train, y_test, feature_names
    )
    predictions['Random Forest'] = rf_pred
    all_metrics.append(rf_metrics)
    
    # Train Neural Network
    nn_model, nn_pred, nn_metrics = train_neural_network_model(X_train, X_test, y_train, y_test)
    predictions['Neural Network'] = nn_pred
    all_metrics.append(nn_metrics)
    
    # 5. Summary comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    
    metrics_df = pd.DataFrame(all_metrics)
    print("\n" + metrics_df.to_string(index=False))
    
    # Find best model
    best_model_idx = metrics_df['RMSE'].idxmin()
    best_model_name = metrics_df.iloc[best_model_idx]['Model'].replace(' (Test)', '')
    print(f"\n🏆 Best Performing Model: {best_model_name}")
    print(f"   - RMSE: ${metrics_df.iloc[best_model_idx]['RMSE']:.2f}")
    print(f"   - R²: {metrics_df.iloc[best_model_idx]['R2']:.4f}")
    
    # 6. Create visualizations
    create_visualizations(df_features, y_test, predictions, feature_importance, test_dates)
    create_correlation_heatmap(df_features)
    
    # 7. Sample predictions
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS (Last 10 Test Days)")
    print("=" * 60)
    
    sample_df = pd.DataFrame({
        'Date': test_dates.tail(10).values,
        'Actual': y_test[-10:],
        'SVR': np.round(svr_pred[-10:], 2),
        'RF': np.round(rf_pred[-10:], 2),
        'NN': np.round(nn_pred[-10:], 2)
    })
    print("\n" + sample_df.to_string(index=False))
    
    # 8. Future prediction example
    print("\n" + "=" * 60)
    print("FUTURE PRICE PREDICTION")
    print("=" * 60)
    
    # Use the best model to predict the next day
    last_features = X_test[-1:].reshape(1, -1)
    
    if best_model_name == 'Random Forest':
        next_pred = rf_model.predict(last_features)[0]
    elif best_model_name == 'Neural Network':
        next_pred = nn_model.predict(last_features)[0]
    else:
        next_pred = svr_model.predict(last_features)[0]
    
    last_actual = y_test[-1]
    print(f"\nLast known price: ${last_actual:.2f}")
    print(f"Predicted next day price ({best_model_name}): ${next_pred:.2f}")
    print(f"Predicted change: ${next_pred - last_actual:.2f} ({((next_pred - last_actual) / last_actual * 100):.2f}%)")
    
    print("\n" + "=" * 70)
    print("   ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - gold_price_analysis.png (Main visualization)")
    print("  - correlation_heatmap.png (Feature correlations)")
    
    return df_features, predictions, metrics_df, feature_importance


if __name__ == "__main__":
    # Run the main analysis
    df_features, predictions, metrics_df, feature_importance = main()