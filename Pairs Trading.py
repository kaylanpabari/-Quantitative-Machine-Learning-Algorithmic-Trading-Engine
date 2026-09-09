import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from itertools import combinations

# =====================================================================
# 1. KALMAN FILTER FOR DYNAMIC HEDGE RATIO
# =====================================================================
def get_dynamic_hedge_ratio(x: pd.Series, y: pd.Series) -> np.ndarray:
    """
    Tracks the dynamic hedge ratio (beta) using a 2D online Kalman Filter.
    State vector: [beta, alpha]^T
    """
    n = len(x)
    beta = np.zeros(n)
    
    # Initialize state space variables
    delta = 1e-5
    trans_cov = (delta / (1 - delta)) * np.eye(2)
    obs_cov = 1.0
    
    state_mean = np.zeros(2)
    state_cov = np.ones((2, 2))
    
    for t in range(n):
        # Observation matrix [x_t, 1]
        F = np.array([x.iloc[t], 1.0])
        
        # 1. Predict state
        # state_mean = state_mean (random walk model)
        state_cov = state_cov + trans_cov
        
        # 2. Compute Innovation
        y_pred = np.dot(F, state_mean)
        innovation = y.iloc[t] - y_pred
        innovation_cov = np.dot(F, np.dot(state_cov, F)) + obs_cov
        
        # 3. Kalman Gain & State Update
        kalman_gain = np.dot(state_cov, F) / innovation_cov
        state_mean = state_mean + kalman_gain * innovation
        state_cov = state_cov - np.outer(kalman_gain, np.dot(F, state_cov))
        
        beta[t] = state_mean[0]
        
    return beta

# =====================================================================
# 2. ORNSTEIN-UHLENBECK HALF-LIFE OF MEAN REVERSION
# =====================================================================
def calculate_half_life(spread: pd.Series) -> int:
    """
    Estimates the half-life of mean reversion using an Ornstein-Uhlenbeck process:
    d(Spread) = lambda * (Spread - mean) * dt + sigma * dW
    """
    spread_clean = spread.dropna()
    spread_lag = spread_clean.shift(1).dropna()
    spread_diff = spread_clean.diff().dropna()
    
    # Align indices
    aligned_lag = spread_lag.loc[spread_diff.index]
    
    X = sm.add_constant(aligned_lag)
    model = sm.OLS(spread_diff, X).fit()
    
    lambda_param = model.params.iloc[1]
    
    # Half-life tau = -ln(2) / lambda
    if lambda_param < 0:
        half_life = -np.log(2) / lambda_param
        return max(int(round(half_life)), 5)  # Enforce minimum bound of 5 days
    else:
        return 30  # Default fallback if non-stationary drift occurs

# =====================================================================
# 3. UNIVERSE PAIR SCANNER
# =====================================================================
def find_best_cointegrated_pair(data: pd.DataFrame, p_cutoff=0.05):
    """
    Scans a dataframe of asset prices for the strongest cointegrated pair.
    """
    pairs = []
    tickers = data.columns
    for t1, t2 in combinations(tickers, 2):
        score, pvalue, _ = coint(data[t1], data[t2])
        if pvalue < p_cutoff:
            pairs.append((t1, t2, pvalue))
            
    if not pairs:
        raise ValueError("No cointegrated pairs found under the selected p-value cutoff.")
        
    pairs.sort(key=lambda x: x[2]) # Sort by lowest p-value
    return pairs[0]

# =====================================================================
# MAIN EXECUTION & BACKTEST PIPELINE
# =====================================================================
if __name__ == "__main__":
    # --- Step A: Load Universe Data ---
    universe_tickers = ["XOM", "CVX", "COP", "EOG", "SLB"]
    print(f"1. Downloading historical prices for universe: {universe_tickers}...")
    
    raw_data = yf.download(universe_tickers, start="2021-01-01", end="2024-01-01")["Close"]
    data = raw_data.dropna()

    # --- Step B: Scan for Top Cointegrated Pair ---
    ticker_1, ticker_2, p_val = find_best_cointegrated_pair(data, p_cutoff=0.05)
    print(f"   -> Top Cointegrated Pair Found: {ticker_1} & {ticker_2} (p-value: {p_val:.4f})")

    # --- Step C: Compute Dynamic Kalman Hedge Ratio & Spread ---
    S1 = data[ticker_1]
    S2 = data[ticker_2]
    
    dynamic_beta = get_dynamic_hedge_ratio(S2, S1)
    df = pd.DataFrame({"S1": S1, "S2": S2, "Beta": dynamic_beta}, index=data.index)
    
    # Dynamic Spread = S1 - Beta_t * S2
    df["Spread"] = df["S1"] - (df["Beta"] * df["S2"])

    # --- Step D: Estimate Optimal OU Lookback Window ---
    optimal_window = calculate_half_life(df["Spread"])
    print(f"   -> Calculated OU Half-Life Lookback Window: {optimal_window} days")

    # --- Step E: Calculate Dynamic Z-Score ---
    df["Rolling_Mean"] = df["Spread"].rolling(window=optimal_window).mean()
    df["Rolling_Std"] = df["Spread"].rolling(window=optimal_window).std()
    df["Z_Score"] = (df["Spread"] - df["Rolling_Mean"]) / df["Rolling_Std"]

    # --- Step F: Apply Entry, Exit, and Stop-Loss Rules ---
    entry_threshold = 1.5
    exit_threshold = 0.1
    stop_loss_threshold = 3.5
    
    df["Positions"] = 0
    
    # Signal Generation
    df.loc[df["Z_Score"] < -entry_threshold, "Positions"] = 1    # Long Spread
    df.loc[df["Z_Score"] > entry_threshold, "Positions"] = -1    # Short Spread
    df.loc[df["Z_Score"].abs() > stop_loss_threshold, "Positions"] = 0  # Emergency Stop-Loss
    df.loc[df["Z_Score"].abs() < exit_threshold, "Positions"] = 0       # Take-Profit Exit
    
    # Forward fill positions between entry and exit/stop-loss
    df["Positions"] = df["Positions"].replace(0, np.nan).ffill().fillna(0)

    # --- Step G: Compute Returns and Microstructure Costs ---
    # Spread return
    df["Spread_Return"] = df["Spread"].diff()
    
    # Gross strategy returns
    df["Gross_Return"] = df["Positions"].shift(1) * df["Spread_Return"]
    
    # Portfolio Capitalization Normalization
    portfolio_base = df["S1"].iloc[0] + (df["Beta"].iloc[0] * df["S2"].iloc[0])
    df["Normalized_Gross_Return"] = df["Gross_Return"] / portfolio_base
    
    # Transaction Friction: 10 bps per trade turnover
    transaction_cost_bps = 0.0010
    trades = df["Positions"].diff().abs()
    df["Friction_Cost"] = trades * transaction_cost_bps
    
    # Net Returns
    df["Net_Strategy_Return"] = df["Normalized_Gross_Return"] - df["Friction_Cost"]
    df["Cumulative_Returns"] = (1 + df["Net_Strategy_Return"]).cumprod()

    # --- Step H: Risk and Performance Analytics ---
    total_net_return = df["Cumulative_Returns"].iloc[-1] - 1
    annualized_sharpe = (df["Net_Strategy_Return"].mean() / df["Net_Strategy_Return"].std()) * np.sqrt(252)
    
    # Maximum Drawdown calculation
    cumulative_max = df["Cumulative_Returns"].cummax()
    drawdown = (df["Cumulative_Returns"] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()

    print("\n=====================================================")
    print("           STRATEGY PERFORMANCE REPORT               ")
    print("=====================================================")
    print(f"Selected Pair              : {ticker_1} / {ticker_2}")
    print(f"Average Dynamic Beta       : {df['Beta'].mean():.4f}")
    print(f"Total Net Return           : {total_net_return * 100:.2f}%")
    print(f"Net Annualized Sharpe      : {annualized_sharpe:.2f}")
    print(f"Maximum Drawdown           : {max_drawdown * 100:.2f}%")
    print("=====================================================")