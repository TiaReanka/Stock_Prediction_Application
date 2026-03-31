"""
Stock Market Predictor using LSTM Neural Network
================================================
Predicts next-day closing price using historical OHLCV data + technical indicators.

Requirements:
    pip install yfinance pandas numpy scikit-learn tensorflow matplotlib

Usage:
    py -3.11 stock_predictor.py
"""

import os
import logging
import warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"]     = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]      = "3"
os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# ─────────────────────────────────────────────
# CONFIG  — only edit this section
# ─────────────────────────────────────────────
TICKER        = "AAPL"        # Stock ticker  e.g. "TSLA", "MSFT", "GOOGL"
START_DATE    = "2024-01-01"  # Recent 15 months keeps price range consistent
END_DATE      = "2026-03-29"  # Keep close to today
LOOKBACK      = 20            # Trading days of history per input sequence
TEST_SPLIT    = 0.15          # Fraction of data held out for testing
EPOCHS        = 200
BATCH_SIZE    = 8             # Small batch size for small dataset
LSTM_UNITS    = [64, 32]
DROPOUT       = 0.2
LEARNING_RATE = 0.001


# ─────────────────────────────────────────────
# 1. DATA FETCHING
# ─────────────────────────────────────────────
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    print(f"\n📥  Fetching data for {ticker}  ({start} -> {end}) ...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for '{ticker}'. Check the ticker symbol.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    print(f"    {len(df)} trading days loaded.")
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    df["SMA_5"]       = close.rolling(5).mean()
    df["SMA_10"]      = close.rolling(10).mean()
    df["SMA_20"]      = close.rolling(20).mean()
    df["EMA_9"]       = close.ewm(span=9,  adjust=False).mean()
    df["EMA_12"]      = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"]      = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    delta         = close.diff()
    gain          = delta.clip(lower=0).rolling(14).mean()
    loss          = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"]     = 100 - (100 / (1 + gain / (loss + 1e-10)))

    bb_mid         = close.rolling(20).mean()
    bb_std         = close.rolling(20).std()
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std
    df["BB_Pos"]   = (close - df["BB_Lower"]) / ((df["BB_Upper"] - df["BB_Lower"]) + 1e-10)

    obv = [0]
    cv  = close.values
    vv  = volume.values
    for i in range(1, len(cv)):
        if cv[i] > cv[i - 1]:
            obv.append(obv[-1] + vv[i])
        elif cv[i] < cv[i - 1]:
            obv.append(obv[-1] - vv[i])
        else:
            obv.append(obv[-1])
    df["OBV"]         = obv
    df["Return"]      = close.pct_change()
    df["Return_5"]    = close.pct_change(5)
    df["Volatility"]  = df["Return"].rolling(5).std()
    df["High_Low_Pct"]= (df["High"].astype(float) - df["Low"].astype(float)) / close

    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# 3. DATASET PREPARATION
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_5", "SMA_10", "SMA_20",
    "EMA_9", "MACD", "MACD_Signal",
    "RSI", "BB_Upper", "BB_Lower", "BB_Pos",
    "OBV", "Return", "Return_5",
    "Volatility", "High_Low_Pct",
]
TARGET_COL = "Close"


def build_sequences(data: np.ndarray, target_idx: int, lookback: int):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback: i])
        y.append(data[i, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_data(df: pd.DataFrame, lookback: int, test_split: float):
    features   = df[FEATURE_COLS].astype(float).values
    target_idx = FEATURE_COLS.index(TARGET_COL)
    split_row  = int(len(features) * (1 - test_split))

    scaler = MinMaxScaler()
    scaler.fit(features[:split_row])
    scaled = scaler.transform(features)

    target_scaler = MinMaxScaler()
    target_scaler.fit(features[:split_row, target_idx].reshape(-1, 1))

    X, y  = build_sequences(scaled, target_idx, lookback)
    split = int(len(X) * (1 - test_split))

    return (
        X[:split], X[split:],
        y[:split], y[split:],
        scaler, target_scaler,
        df.index[lookback:],
    )


# ─────────────────────────────────────────────
# 4. MODEL
# ─────────────────────────────────────────────
def build_model(input_shape, lstm_units, dropout, lr) -> Sequential:
    model = Sequential()
    for i, units in enumerate(lstm_units):
        return_seq = i < len(lstm_units) - 1
        kwargs = dict(return_sequences=return_seq)
        if i == 0:
            kwargs["input_shape"] = input_shape
        model.add(LSTM(units, **kwargs))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr), loss="mse", metrics=["mae"])
    model.summary()
    return model


# ─────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────
def train_model(model, X_train, y_train, epochs, batch_size):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=25,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=10, min_lr=1e-6, verbose=1),
    ]
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────
def evaluate(y_true, y_pred, label="Test"):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100)
    r2   = r2_score(y_true, y_pred)
    print(f"\n📊  {label} Metrics")
    print(f"    MAE  : ${mae:.2f}")
    print(f"    RMSE : ${rmse:.2f}")
    print(f"    MAPE : {mape:.2f}%")
    print(f"    R2   : {r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


# ─────────────────────────────────────────────
# 7. VISUALISATION
# ─────────────────────────────────────────────
def plot_results(dates, y_true, y_pred, ticker, history, split_idx):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{ticker} - LSTM Stock Price Predictor",
                 fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(dates[:split_idx], y_true[:split_idx],
            label="Train (actual)", alpha=0.5, color="steelblue")
    ax.plot(dates[split_idx:], y_true[split_idx:],
            label="Test (actual)", color="steelblue")
    ax.plot(dates[split_idx:], y_pred,
            label="Predicted", color="tomato", linestyle="--")
    ax.axvline(dates[split_idx], color="gray", linestyle=":", linewidth=1.5)
    ax.set_title("Actual vs Predicted Close Price")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(dates[split_idx:], y_true[split_idx:],
            label="Actual", color="steelblue")
    ax.plot(dates[split_idx:], y_pred,
            label="Predicted", color="tomato", linestyle="--")
    ax.fill_between(dates[split_idx:], y_true[split_idx:], y_pred,
                    alpha=0.15, color="tomato")
    ax.set_title("Test Period: Actual vs Predicted")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(history.history["loss"],     label="Train Loss")
    ax.plot(history.history["val_loss"], label="Val Loss")
    ax.set_title("Training Loss (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    residuals = y_true[split_idx:] - y_pred
    ax.scatter(dates[split_idx:], residuals, alpha=0.4, s=10, color="purple")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Prediction Residuals (Test Set)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Error (USD)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("stock_prediction.png", dpi=150, bbox_inches="tight")
    print("\n📈  Chart saved to stock_prediction.png")
    plt.show()


# ─────────────────────────────────────────────
# 8. NEXT-DAY PREDICTION
# ─────────────────────────────────────────────
def predict_next_day(model, df, scaler, target_scaler, lookback):
    features   = df[FEATURE_COLS].astype(float).values
    scaled     = scaler.transform(features)
    last_seq   = scaled[-lookback:].reshape(1, lookback, len(FEATURE_COLS))
    pred_s     = float(model.predict(last_seq, verbose=0)[0, 0])
    pred_price = float(target_scaler.inverse_transform([[pred_s]])[0, 0])
    last_close = float(df["Close"].values[-1])
    change     = (pred_price - last_close) / last_close * 100
    direction  = "UP" if change > 0 else "DOWN"
    print(f"\n🔮  Next-Day Prediction for {TICKER}")
    print(f"    Last Close : ${last_close:.2f}")
    print(f"    Predicted  : ${pred_price:.2f}  {direction} {abs(change):.2f}%")
    return pred_price


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
class StockPredictor:
    def __init__(self, ticker=TICKER, start=START_DATE, end=END_DATE,
                 lookback=LOOKBACK, test_split=TEST_SPLIT):
        self.ticker     = ticker
        self.start      = start
        self.end        = end
        self.lookback   = lookback
        self.test_split = test_split

    def run(self):
        df = fetch_data(self.ticker, self.start, self.end)
        df = add_technical_indicators(df)

        X_train, X_test, y_train, y_test, scaler, target_scaler, dates = \
            prepare_data(df, self.lookback, self.test_split)
        print(f"\nTraining samples : {len(X_train)}")
        print(f"Testing  samples : {len(X_test)}")

        model = build_model(
            input_shape=(self.lookback, X_train.shape[2]),
            lstm_units=LSTM_UNITS,
            dropout=DROPOUT,
            lr=LEARNING_RATE,
        )
        history = train_model(model, X_train, y_train, EPOCHS, BATCH_SIZE)

        y_pred_scaled  = model.predict(X_test,  verbose=0).flatten()
        y_train_scaled = model.predict(X_train, verbose=0).flatten()

        y_pred         = target_scaler.inverse_transform(
                             y_pred_scaled.reshape(-1, 1)).flatten()
        y_true         = target_scaler.inverse_transform(
                             y_test.reshape(-1, 1)).flatten()
        y_train_actual = target_scaler.inverse_transform(
                             y_train.reshape(-1, 1)).flatten()

        evaluate(y_true, y_pred, label="Test")

        all_actual = np.concatenate([y_train_actual, y_true])
        plot_results(dates, all_actual, y_pred,
                     self.ticker, history, len(y_train_actual))

        predict_next_day(model, df, scaler, target_scaler, self.lookback)

        return model, scaler, target_scaler


if __name__ == "__main__":
    predictor = StockPredictor(ticker=TICKER)
    predictor.run()