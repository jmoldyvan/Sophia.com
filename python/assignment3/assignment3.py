"""
Housing Price Prediction with Linear Regression
================================================

This program predicts housing prices based on house size (square footage)
using linear regression. It demonstrates data handling, visualization,
machine learning model training, and interactive predictions.

How it works:
1. Loads housing data from a CSV file using pandas
2. Displays basic statistics about the dataset
3. Visualizes the data with a scatter plot
4. Trains a linear regression model using scikit-learn
5. Displays model metrics (coefficient, intercept, R² score)
6. Shows the regression line fitted to the data
7. Allows users to input house sizes and get price predictions

Required libraries: pandas, matplotlib, scikit-learn
Install with: pip install pandas matplotlib scikit-learn

Author: Student
Date: 2024
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sys


def load_data(filepath):
    """
    Load housing data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        pandas.DataFrame: The loaded dataset, or None if loading fails
    """
    try:
        data = pd.read_csv(filepath)
        print(f"[OK] Data loaded successfully from '{filepath}'")
        print(f"     Rows: {len(data)}, Columns: {len(data.columns)}")
        return data
    
    except FileNotFoundError:
        print(f"[ERROR] File not found: '{filepath}'")
        print("        Please ensure the CSV file exists in the specified location.")
        return None
    
    except pd.errors.EmptyDataError:
        print(f"[ERROR] The file '{filepath}' is empty.")
        return None
    
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return None


def explore_data(data):
    """
    Display basic statistics and information about the dataset.
    
    Args:
        data: pandas DataFrame containing the housing data
    """
    print("\n" + "=" * 55)
    print("              DATA EXPLORATION")
    print("=" * 55)
    
    # Display first few rows
    print("\nFirst 5 rows of the dataset:")
    print("-" * 40)
    print(data.head().to_string(index=False))
    
    # Display basic statistics
    print("\n" + "-" * 40)
    print("Statistical Summary:")
    print("-" * 40)
    
    # Get column names (handle different naming conventions)
    size_col = data.columns[0]  # First column is size
    price_col = data.columns[1]  # Second column is price
    
    # Size statistics
    print(f"\nHouse Size ({size_col}):")
    print(f"  Count:    {data[size_col].count()}")
    print(f"  Mean:     {data[size_col].mean():,.2f} sqft")
    print(f"  Std Dev:  {data[size_col].std():,.2f} sqft")
    print(f"  Min:      {data[size_col].min():,.2f} sqft")
    print(f"  Max:      {data[size_col].max():,.2f} sqft")
    
    # Price statistics
    print(f"\nHouse Price ({price_col}):")
    print(f"  Count:    {data[price_col].count()}")
    print(f"  Mean:     ${data[price_col].mean():,.2f}")
    print(f"  Std Dev:  ${data[price_col].std():,.2f}")
    print(f"  Min:      ${data[price_col].min():,.2f}")
    print(f"  Max:      ${data[price_col].max():,.2f}")


def visualize_data(data, save_path="housing_scatter.png"):
    """
    Create a scatter plot showing the relationship between house size and price.
    
    Args:
        data: pandas DataFrame containing the housing data
        save_path: Path to save the plot image
    """
    print("\nGenerating scatter plot...")
    
    try:
        # Get column names
        size_col = data.columns[0]
        price_col = data.columns[1]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        plt.scatter(data[size_col], data[price_col], 
                   color='blue', alpha=0.6, edgecolors='black', s=80)
        
        # Labels and title
        plt.xlabel('House Size (sqft)', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.title('Housing Prices vs. House Size', fontsize=14, fontweight='bold')
        
        # Format y-axis to show dollar amounts
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
        )
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"[OK] Scatter plot saved to '{save_path}'")
        
    except Exception as e:
        print(f"[ERROR] Failed to create visualization: {e}")


def train_model(data):
    """
    Train a linear regression model on the housing data.
    
    Args:
        data: pandas DataFrame containing the housing data
    
    Returns:
        tuple: (trained model, X values, y values) or (None, None, None) if failed
    """
    print("\n" + "=" * 55)
    print("              MODEL TRAINING")
    print("=" * 55)
    
    try:
        # Get column names
        size_col = data.columns[0]
        price_col = data.columns[1]
        
        # Prepare features (X) and target (y)
        # reshape(-1, 1) converts 1D array to 2D array required by sklearn
        X = data[size_col].values.reshape(-1, 1)
        y = data[price_col].values
        
        # Create and train the model
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate predictions for R² score
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        # Display model parameters
        print("\nModel Training Complete!")
        print("-" * 40)
        print(f"\nModel Parameters:")
        print(f"  Coefficient (slope):  {model.coef_[0]:,.4f}")
        print(f"  Intercept:            ${model.intercept_:,.2f}")
        print(f"  R² Score:             {r2:.4f} ({r2*100:.2f}%)")
        
        # Interpret the model
        print("\nInterpretation:")
        print(f"  For every 1 sqft increase in size,")
        print(f"  the price increases by ${model.coef_[0]:,.2f}")
        print(f"\n  The model explains {r2*100:.1f}% of the variance in prices.")
        
        return model, X, y
    
    except Exception as e:
        print(f"[ERROR] Failed to train model: {e}")
        return None, None, None


def visualize_model(data, model, X, y, save_path="housing_regression.png"):
    """
    Create a scatter plot with the regression line overlaid.
    
    Args:
        data: pandas DataFrame containing the housing data
        model: Trained LinearRegression model
        X: Feature values (house sizes)
        y: Target values (prices)
        save_path: Path to save the plot image
    """
    print("\nGenerating regression plot...")
    
    try:
        # Get column names
        size_col = data.columns[0]
        price_col = data.columns[1]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Scatter plot of actual data
        plt.scatter(X, y, color='blue', alpha=0.6, 
                   edgecolors='black', s=80, label='Actual Data')
        
        # Regression line
        y_pred = model.predict(X)
        plt.plot(X, y_pred, color='red', linewidth=2, 
                label=f'Regression Line (R² = {r2_score(y, y_pred):.4f})')
        
        # Labels and title
        plt.xlabel('House Size (sqft)', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.title('Linear Regression: Housing Prices vs. House Size', 
                 fontsize=14, fontweight='bold')
        
        # Format y-axis
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
        )
        
        # Add legend and grid
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add equation annotation
        equation = f'Price = {model.coef_[0]:,.2f} × Size + ${model.intercept_:,.2f}'
        plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"[OK] Regression plot saved to '{save_path}'")
        
    except Exception as e:
        print(f"[ERROR] Failed to create visualization: {e}")

def predict_price(model):
    """
    Prompt user for house size and predict the price.
    
    Args:
        model: Trained LinearRegression model
    
    Returns:
        bool: True to continue predicting, False to exit
    """
    print("\n" + "-" * 40)
    
    user_input = input("Enter house size in sqft (or 'quit' to exit): ").strip()
    
    # Check for exit
    if user_input.lower() in ['quit', 'q', 'exit']:
        return False
    
    try:
        # Validate input
        size = float(user_input)
        
        # Check for negative or unreasonable values
        if size <= 0:
            print("[!] House size must be a positive number.")
            return True
        
        if size > 50000:
            print("[!] Warning: That's an unusually large house size!")
        
        # Make prediction
        predicted_price = model.predict([[size]])[0]
        
        # Display result
        print("\n  +-------------------------------+")
        print(f"  |  House Size:  {size:>10,.0f} sqft |")
        print(f"  |  Est. Price:  ${predicted_price:>10,.2f} |")
        print("  +-------------------------------+")
        
        return True
    
    except ValueError:
        print("[!] Invalid input. Please enter a numeric value.")
        return True


def main():
    """
    Main function that orchestrates the housing price prediction program.
    """
    # Display header
    print("\n" + "=" * 55)
    print("   HOUSING PRICE PREDICTION WITH LINEAR REGRESSION")
    print("=" * 55)
    
    # Step 1: Load the data
    print("\n[Step 1] Loading Dataset...")
    data = load_data("housing_data.csv")
    
    if data is None:
        print("\nExiting program due to data loading error.")
        sys.exit(1)
    
    # Step 2: Explore the data
    print("\n[Step 2] Exploring Data...")
    explore_data(data)
    
    # Step 3: Visualize the raw data
    print("\n[Step 3] Visualizing Data...")
    visualize_data(data)
    
    # Step 4: Train the model
    print("\n[Step 4] Training Model...")
    model, X, y = train_model(data)
    
    if model is None:
        print("\nExiting program due to model training error.")
        sys.exit(1)
    
    # Step 5: Visualize the model fit
    print("\n[Step 5] Visualizing Model Fit...")
    visualize_model(data, model, X, y)
    
    # Step 6: Interactive predictions
    print("\n" + "=" * 55)
    print("              PRICE PREDICTIONS")
    print("=" * 55)
    print("\nEnter house sizes to get price predictions.")
    print("Type 'quit' to exit the program.")
    
    while predict_price(model):
        pass
    
    # Goodbye message
    print("\n" + "=" * 55)
    print("Thank you for using the Housing Price Predictor!")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()