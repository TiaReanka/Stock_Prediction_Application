📈 Stock Market Predictor — LSTM Neural Network
A machine learning project that predicts next-day stock closing prices using a Long Short-Term Memory (LSTM) neural network, trained on historical OHLCV data and technical indicators.
________________________________________
🚀 Live Demo
Run the predictor on any stock:
py -3.11 stock_predictor.py
Output example:
📥  Fetching data for AAPL  (2024-01-01 -> 2026-03-29) ...
    561 trading days loaded.

🔮  Next-Day Prediction for AAPL
    Last Close : $248.80
    Predicted  : $246.29  DOWN 1.01%
________________________________________
📋 Table of Contents
•	Overview
•	Features
•	Installation
•	Usage
•	Project Structure
•	How It Works
•	Tools & Technologies
•	AI Assistance
•	SOLID & DRY Principles
•	Results
•	Disclaimer

Overview
This project builds a complete end-to-end stock price prediction pipeline using deep learning. It automatically downloads historical stock data from Yahoo Finance, engineers 20+ technical indicators as features, trains a stacked LSTM model, evaluates its performance, and outputs a next-day price prediction with direction signal.
The project was built as a learning exercise in Python, machine learning, and software engineering principles — developed iteratively from scratch through real debugging and hands-on problem solving.

✨ Features
•	Automatic data fetching from Yahoo Finance using yfinance
•	20+ engineered features including RSI, MACD, Bollinger Bands, OBV, moving averages, and volatility metrics
•	Stacked LSTM architecture with Batch Normalisation and Dropout for regularisation
•	Data leakage prevention — scalers are fit only on training data
•	Early stopping & learning rate scheduling for optimal training
•	4-panel visualisation — full price history, test period zoom, training loss, and residuals
•	Next-day price prediction with percentage change and direction signal
•	Clean, warning-free output with all TensorFlow noise suppressed
•	Configurable — change ticker, dates, and model settings in one place

🛠 Installation
Prerequisites
•	Python 3.11 (recommended)
•	Visual Studio 2022 with Python workload, or any Python IDE
Step 1 — Clone the repository
git clone https://github.com/yourusername/stock-predictor.git
cd stock-predictor
Step 2 — Install dependencies
pip install yfinance pandas numpy scikit-learn tensorflow matplotlib
Step 3 — Run the predictor
py -3.11 stock_predictor.py

📖 Usage
Change the stock ticker
Open stock_predictor.py and edit the CONFIG section at the top:
TICKER     = "TSLA"        # Change to any valid ticker
START_DATE = "2024-01-01"
END_DATE   = "2026-03-29"
Supported tickers
Any ticker available on Yahoo Finance, for example:
•	AAPL — Apple
•	TSLA — Tesla
•	MSFT — Microsoft
•	GOOGL — Alphabet
•	NVDA — NVIDIA
•	AMZN — Amazon
Output
After training, the script produces:
•	Console metrics: MAE, RMSE, MAPE, R²
•	stock_prediction.png — 4-panel chart saved in the project folder
•	Next-day price prediction printed to console

📁 Project Structure
Stock_Prediction_Application/
│
├── stock_predictor.py                          # Main script — all logic lives here
├── stock_prediction.png                        # Generated chart (auto-created on run)
├── Stock_Prediction_Application/               # Python project folder
├── Stock_Prediction_Application.sln            # Microsoft Visual Studio solution file
└── README.md                                   # This file

⚙️ How It Works
1. Data Fetching
Historical OHLCV (Open, High, Low, Close, Volume) data is downloaded directly from Yahoo Finance using the yfinance library.
2. Feature Engineering
20+ technical indicators are calculated from raw price data:
Feature	Description
SMA 5/10/20	Simple Moving Averages
EMA 9/12/26	Exponential Moving Averages
MACD + Signal	Momentum indicator
RSI (14)	Relative Strength Index
Bollinger Bands	Volatility bands + position
OBV	On-Balance Volume
Daily Returns	1-day and 5-day % change
Volatility	5-day rolling standard deviation
High-Low %	Daily range as fraction of close
3. Data Preparation
•	Data is split 85% training / 15% testing
•	MinMaxScaler is fit only on training data to prevent data leakage
•	Sequences of 20 trading days are built as LSTM inputs
4. Model Architecture
Input (20 days × 20 features)
    ↓
LSTM (64 units) + BatchNorm + Dropout (0.2)
    ↓
LSTM (32 units) + BatchNorm + Dropout (0.2)
    ↓
Dense (32, ReLU)
    ↓
Dense (1) → Predicted Close Price
5. Training
•	Loss function: Mean Squared Error (MSE)
•	Optimiser: Adam (lr=0.001)
•	Early stopping with 25-epoch patience
•	Learning rate halved when validation loss plateaus
6. Prediction
The last 20 days of real data are fed into the trained model to generate the next-day closing price prediction.

🧰 Tools & Technologies
Languages & Runtime
•	Python 3.11 — chosen for its stability and broad ML library support
Libraries
Library	Purpose
TensorFlow / Keras	Building and training the LSTM neural network
yfinance	Fetching historical stock data from Yahoo Finance
pandas	Data manipulation and time series handling
numpy	Numerical computation and array operations
scikit-learn	Data scaling (MinMaxScaler) and evaluation metrics
matplotlib	Visualising predictions, loss curves, and residuals
Development Environment
•	Visual Studio 2022 — IDE used for development, debugging, and running the project
•	Python Environments panel — used to manage the Python 3.11 interpreter within VS 2022
•	Git & GitHub — version control and repository hosting
Model
•	LSTM (Long Short-Term Memory) — a type of recurrent neural network well-suited to sequential time series data, capable of learning long-term dependencies in stock price patterns

🤖 AI Assistance
This project was built with the assistance of Claude (Anthropic), an AI assistant, throughout the development process.
How AI was used
•	Code generation — the initial LSTM architecture, feature engineering pipeline, and training loop were generated with Claude as a starting point
•	Debugging — multiple runtime errors were diagnosed and fixed collaboratively, including ValueError from pandas Series ambiguity, TypeError from float conversion, and data leakage in the scaler setup
•	Iterative improvement — model hyperparameters (lookback window, LSTM units, dropout, batch size, learning rate) were tuned across multiple runs based on observed metrics
•	Environment setup — Claude guided the setup process including configuring PATH variables, installing pip packages, and configuring Visual Studio 2022
•	Explaining concepts — concepts like data leakage, MinMaxScaler fitting, LSTM sequence building, and R² interpretation were explained throughout
What was learnt through the process
•	Understanding what each error meant before being able to fix it
•	Evaluating whether model results were acceptable or needed improvement
•	Making configuration decisions about dates, architecture, and hyperparameters
•	Debugging environment issues (PATH variables, Python version conflicts, pip scoping)

🏗 SOLID & DRY Principles
This project was structured with core software engineering principles in mind, even within a single-file Python script.
DRY — Don't Repeat Yourself
The DRY principle states that every piece of logic should have a single authoritative definition. Repetition creates maintenance problems — change one instance and forget another and bugs appear.
Applied in this project:
•	CONFIG block at the top — all tunable parameters (TICKER, START_DATE, LOOKBACK, LSTM_UNITS etc.) are defined once. Changing the ticker or date range requires editing one line, not hunting through the code.
•	FEATURE_COLS list — the list of feature column names is defined once and reused across prepare_data(), predict_next_day(), and add_technical_indicators(). No column name is hardcoded in multiple places.
•	build_sequences() — the sliding window sequence logic is written once and called from prepare_data(), rather than being duplicated for train and test sets separately.
•	evaluate() — metric calculation (MAE, RMSE, MAPE, R²) is defined once and reusable for any set of predictions.
SOLID Principles
SOLID is a set of five object-oriented design principles that make software easier to maintain and extend:
S — Single Responsibility Principle: Each function or class should have one job and one reason to change.
Every function in this project has a clearly defined, single responsibility:
Function	Single Responsibility
fetch_data()	Download raw data from Yahoo Finance only
add_technical_indicators()	Engineer features only
prepare_data()	Scale data and build sequences only
build_model()	Define model architecture only
train_model()	Run the training loop only
evaluate()	Calculate and print metrics only
plot_results()	Generate visualisations only
predict_next_day()	Generate the next-day forecast only
No function does more than one job. If the charting library changes, only plot_results() needs updating. If the data source changes, only fetch_data() needs updating.

O — Open/Closed Principle: Software should be open for extension but closed for modification.
The StockPredictor class accepts parameters at initialisation (ticker, start, end, lookback, test_split), allowing different configurations without modifying the core class logic. You can extend behaviour by subclassing StockPredictor or passing different arguments — the internals stay untouched.

L — Liskov Substitution Principle: Subtypes should be substitutable for their base types.
The modular function design means any function can be swapped out for an improved version with the same interface. For example, build_model() could be replaced with a Transformer-based model function as long as it accepts input_shape, lstm_units, dropout, and lr and returns a compiled Keras model — the rest of the pipeline continues to work unchanged.

I — Interface Segregation Principle: No component should be forced to depend on methods it does not use.
Each function takes only the inputs it needs. evaluate() only needs y_true and y_pred. plot_results() only needs dates, actuals, predictions, ticker, history, and split index. No function receives a large object and pulls out what it needs — dependencies are explicit and minimal.

D — Dependency Inversion Principle: High-level modules should not depend on low-level modules. Both should depend on abstractions.
The StockPredictor.run() method orchestrates the pipeline by calling high-level functions (fetch_data, build_model, train_model etc.) without depending on their internal implementations. The model training step does not care whether the data came from Yahoo Finance or a CSV file — it only depends on the prepared numpy arrays passed to it.

📊 Results
Performance on AAPL (2024-01-01 to 2026-03-29):
Metric	Value
MAE	$13.21
RMSE	$16.00
MAPE	4.88%
Last Close	$248.80
Predicted	$246.29
Difference	$2.51
The model predicted within $2.51 of the actual closing price on its final run — approximately 1% error.

⚠️ Disclaimer
This project is for educational purposes only. Stock prices are inherently unpredictable and no machine learning model should be used to make real financial decisions. Past price patterns do not guarantee future results. Always consult a qualified financial advisor before making investment decisions.


