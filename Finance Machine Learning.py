import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================================
# 1. FETCH DATA & ENGINEER TECHNICAL FEATURES
# =====================================================================
def get_engineered_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    # Technical Indicators
    df['Return_1D'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Return_5D'] = np.log(df['Close'] / df['Close'].shift(5))
    df['Return_10D'] = np.log(df['Close'] / df['Close'].shift(10))

    df['Vol_10D'] = df['Return_1D'].rolling(10).std()
    df['Vol_30D'] = df['Return_1D'].rolling(30).std()

    df['SMA_10'] = df['Close'] / df['Close'].rolling(10).mean() - 1
    df['SMA_50'] = df['Close'] / df['Close'].rolling(50).mean() - 1

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # Target: 1 if return 5 days ahead is positive, else 0
    future_return = np.log(df['Close'].shift(-5) / df['Close'])
    df['Target'] = np.where(future_return > 0, 1, 0)

    return df.dropna()

# =====================================================================
# 2. MAIN EXECUTION PIPELINE
# =====================================================================
if __name__ == "__main__":
    ticker = "AAPL"
    print(f"1. Fetching and preparing data for {ticker}...")
    df = get_engineered_data(ticker, start="2018-01-01", end="2024-01-01")

    features = [
        'Return_1D', 'Return_5D', 'Return_10D', 
        'Vol_10D', 'Vol_30D', 'SMA_10', 'SMA_50', 
        'RSI_14', 'Volume_Ratio'
    ]
    
    # Chronological Train / Test Split (Prevent Data Leakage)
    split_idx = int(len(df) * 0.80)
    X_train, X_test = df[features].iloc[:split_idx], df[features].iloc[split_idx:]
    y_train, y_test = df['Target'].iloc[:split_idx], df['Target'].iloc[split_idx:]

    print("2. Training XGBoost Model...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    # --- Predictions ---
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > 0.50).astype(int)

    # =====================================================================
    # 3. ORIGINAL STATISTICS & METRICS
    # =====================================================================
    print("\n" + "="*53)
    print("           CLASSIFICATION METRICS (ML)               ")
    print("="*53)
    print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))

    # --- Financial Backtest on Test Set ---
    test_df = df.iloc[len(X_train):].copy()
    
    # Generate strategy signals: 1 for Long, -1 for Short
    test_df['Signal'] = np.where(y_pred == 1, 1, -1)
    
    # Returns
    test_df['Strategy_Return'] = test_df['Signal'].shift(1) * test_df['Return_1D']
    test_df['Buy_Hold_Return'] = test_df['Return_1D']

    # Cumulative Returns
    test_df['Strategy_Cum'] = np.exp(test_df['Strategy_Return'].cumsum())
    test_df['Buy_Hold_Cum'] = np.exp(test_df['Buy_Hold_Return'].cumsum())

    # Risk Metrics
    strat_sharpe = (test_df['Strategy_Return'].mean() / test_df['Strategy_Return'].std()) * np.sqrt(252)
    bh_sharpe = (test_df['Buy_Hold_Return'].mean() / test_df['Buy_Hold_Return'].std()) * np.sqrt(252)

    print("="*53)
    print("            FINANCIAL BACKTEST REPORT                ")
    print("="*53)
    print(f"Strategy Sharpe Ratio : {strat_sharpe:.2f}")
    print(f"Buy & Hold Sharpe     : {bh_sharpe:.2f}")
    print(f"Total Strategy Return : {(test_df['Strategy_Cum'].iloc[-1] - 1) * 100:.2f}%")
    print(f"Total Buy & Hold Return: {(test_df['Buy_Hold_Cum'].iloc[-1] - 1) * 100:.2f}%")
    print("="*53 + "\n")

    # =====================================================================
    # 4. SHAP EXPLAINABILITY VISUALIZATIONS
    # =====================================================================
    print("3. Computing SHAP explanations...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # --- Plot A: Global Feature Importance Summary ---
    print("   -> Rendering SHAP Summary Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"SHAP Feature Importance Summary ({ticker})", fontsize=14)
    plt.tight_layout()
    plt.show()

    # --- Plot B: Single Latest Signal Explanation ---
    latest_idx = -1
    latest_date = X_test.index[latest_idx].strftime('%Y-%m-%d')
    prob_up = y_pred_prob[latest_idx]

    print(f"   -> Rendering Waterfall Plot for latest signal ({latest_date})...")
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[latest_idx], show=False)
    plt.title(f"SHAP Breakdown for Signal on {latest_date} (Prob UP: {prob_up*100:.1f}%)", fontsize=14)
    plt.tight_layout()
    plt.show()