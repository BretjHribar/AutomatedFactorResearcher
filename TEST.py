from StockReturnDecoder import StockReturnDecoder
import pandas as pd
import numpy as np

# Example data (dimensions chosen for clarity)
num_alphas = 100
num_stocks = 2000
time_steps = 10
np.random.seed(42)  # For reproducibility
alpha_expected_returns = np.random.rand(num_alphas, time_steps)  # (N x d)
stock_positions = np.random.rand(num_alphas, num_stocks, time_steps)  # (N x M x d)

# Initialize the decoder
decoder = StockReturnDecoder()

# Decode stock returns
stock_returns = decoder.decode_stock_returns(
    alpha_expected_returns, stock_positions, num_components=0
)

# Print dimensions for verification
print("Alpha Expected Returns:", alpha_expected_returns.shape)  # (N x d)
print("Stock Positions:", stock_positions.shape)  # (N x M x d)
print("Stock Returns:", stock_returns.shape)  # (M,)
print("Stock Returns:", stock_returns)
# Output:
# Alpha Expected Returns: (5, 4)
# Stock Positions: (5, 3, 4)
# Stock Returns: (3,)