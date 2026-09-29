# =============================================================================
# LSTM to predict market DIRECTION (up/down) - leakage-free version
# -----------------------------------------------------------------------------
# What this version fixes compared to a typical tutorial LSTM:
#   1. TEMPORAL SPLIT BEFORE SCALING. The scaler is fitted only on the training
#      set. The test set is transformed with that same scaler. Without this,
#      the model "sees" the future through the normalisation (lookahead bias).
#   2. STATIONARY FEATURES. I use returns, not price levels. Predicting the
#      price level makes the model copy the last value ("tomorrow = today"),
#      which gives nice-looking but useless metrics.
#   3. TARGET = DIRECTION, not level. Binary classification: is tomorrow's
#      return positive (1) or not (0)?
#   4. MANDATORY BASELINES. A model is only worth something if it beats
#      (a) always predicting the majority class and (b) persistence
#      (tomorrow = today's direction). Without a baseline, an accuracy
#      number is impossible to interpret.
#
# How to run: in Google Colab, click Runtime -> Run all
#             (or locally: pip install -r requirements.txt && python lstm_direction.py)
# =============================================================================

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import random

# -----------------------------------------------------------------------------
# Reproducibility (note: on GPU it may not be 100% deterministic)
# -----------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# -----------------------------------------------------------------------------
# 1) Parameters
# -----------------------------------------------------------------------------
TICKER    = "SPY"
START     = "2010-01-01"
END       = None          # None = up to today
SEQ_LEN   = 60            # days of history in each window
TEST_SIZE = 0.20          # final fraction of the series kept for testing (out-of-sample)
BATCH     = 32
EPOCHS    = 60

# -----------------------------------------------------------------------------
# 2) Download
# -----------------------------------------------------------------------------
df = yf.download(TICKER, start=START, end=END, progress=False, auto_adjust=True)
if df.empty:
    raise SystemExit(f"No data for {TICKER}.")

# yfinance can return MultiIndex columns (e.g. ('Close','SPY')). Flatten them:
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[["Close", "Volume"]].dropna()
print(f"Data downloaded: {df.shape[0]} days  ({df.index.min().date()} -> {df.index.max().date()})")

# -----------------------------------------------------------------------------
# 3) Feature engineering - everything is stationary (returns, not levels)
# -----------------------------------------------------------------------------
feat = pd.DataFrame(index=df.index)
feat["log_ret"]   = np.log(df["Close"] / df["Close"].shift(1))      # daily return
feat["mom_5"]     = feat["log_ret"].rolling(5).mean()               # momentum (5d)
feat["vol_10"]    = feat["log_ret"].rolling(10).std()               # volatility (10d)
feat["vol_chg"]   = np.log(df["Volume"] / df["Volume"].shift(1))    # volume change
feat["ret_abs_5"] = feat["log_ret"].abs().rolling(5).mean()         # average return size (5d)

FEATURES = ["log_ret", "mom_5", "vol_10", "vol_chg", "ret_abs_5"]

# TARGET: direction of TOMORROW's return (1 = positive, 0 = otherwise)
feat["target"] = (feat["log_ret"].shift(-1) > 0).astype(int)

# TODAY's direction (used for the persistence baseline)
feat["dir_today"] = (feat["log_ret"] > 0).astype(int)

feat = feat.dropna().copy()   # remove NaNs from the rolling windows (start) and shift(-1) (end)

# -----------------------------------------------------------------------------
# 4) TEMPORAL split first, scale after (the golden rule against leakage)
# -----------------------------------------------------------------------------
split_idx = int((1 - TEST_SIZE) * len(feat))
train_df  = feat.iloc[:split_idx]
test_df   = feat.iloc[split_idx:]

scaler = StandardScaler()                                  # StandardScaler over MinMax for returns (more robust to outliers)
train_scaled = scaler.fit_transform(train_df[FEATURES])   # fit ONLY on the training set
test_scaled  = scaler.transform(test_df[FEATURES])        # transform the test set with the training scaler

train_target = train_df["target"].values
test_target  = test_df["target"].values
test_dir_today = test_df["dir_today"].values              # for the persistence baseline

# -----------------------------------------------------------------------------
# 5) Building sequences (causal: window up to day t -> direction of day t+1)
#    Note: windows are created inside each set. The first SEQ_LEN-1 samples of
#    each side are lost - negligible and, above all, ZERO leakage.
# -----------------------------------------------------------------------------
def make_sequences(X2d, y1d, seq_len):
    X, y = [], []
    for i in range(seq_len - 1, len(X2d)):
        X.append(X2d[i - seq_len + 1 : i + 1])
        y.append(y1d[i])
    return np.array(X), np.array(y)

X_train, y_train = make_sequences(train_scaled, train_target, SEQ_LEN)
X_test,  y_test  = make_sequences(test_scaled,  test_target,  SEQ_LEN)

# Persistence aligned with y_test (same loss of SEQ_LEN-1 samples at the start)
persist_pred = test_dir_today[SEQ_LEN - 1:]

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Target distribution (train): up={y_train.mean():.1%}  down={1 - y_train.mean():.1%}")

# -----------------------------------------------------------------------------
# 6) Model - binary classification LSTM (modest architecture to limit overfitting)
# -----------------------------------------------------------------------------
model = Sequential([
    Input(shape=(SEQ_LEN, len(FEATURES))),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid"),            # probability of "up"
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

# Class weights (the market goes up slightly more often than it goes down)
cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
class_weight = {0: cw[0], 1: cw[1]}

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
]

# -----------------------------------------------------------------------------
# 7) Training
#    validation_split uses the TAIL of the training set (Keras does not shuffle
#    here), so validation always comes after the training data in time.
#    This is the correct approach for time series.
# -----------------------------------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1,
)

# -----------------------------------------------------------------------------
# 8) Evaluation + BASELINES (the core of the honest methodology)
# -----------------------------------------------------------------------------
proba = model.predict(X_test).flatten()
lstm_pred = (proba > 0.5).astype(int)

acc_lstm     = accuracy_score(y_test, lstm_pred)
majority_cls = int(round(y_train.mean()))
acc_majority = accuracy_score(y_test, np.full_like(y_test, majority_cls))
acc_persist  = accuracy_score(y_test, persist_pred)

print("\n" + "=" * 60)
print("RESULTS (out-of-sample)")
print("=" * 60)
print(f"  LSTM ................. {acc_lstm:.4f}")
print(f"  Majority baseline .... {acc_majority:.4f}   (always predict class '{majority_cls}')")
print(f"  Persistence baseline . {acc_persist:.4f}   (tomorrow = today's direction)")
best_baseline = max(acc_majority, acc_persist)
edge = acc_lstm - best_baseline
print("-" * 60)
print(f"  LSTM edge over the best baseline: {edge:+.4f}  ({edge*100:+.2f} p.p.)")

# Honest conclusion
print("\nInterpretation:")
if edge > 0.02:
    print("  The LSTM beats the best baseline by more than 2 p.p. Promising result,")
    print("  but I should confirm it with walk-forward validation and multiple seeds.")
elif edge > 0.0:
    print("  The LSTM is only slightly above the baseline. This is within statistical")
    print("  noise, so it is NOT a demonstrable edge. Expected result for index direction.")
else:
    print("  The LSTM does NOT beat the baseline. This is the expected result: the daily")
    print("  direction of a liquid index is essentially unpredictable. The value of this")
    print("  project is the leakage-proof METHODOLOGY, not the final number.")
print("=" * 60)

# -----------------------------------------------------------------------------
# 9) Charts
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

# (a) Training curves
axes[0].plot(history.history["loss"], label="train")
axes[0].plot(history.history["val_loss"], label="validation")
axes[0].set_title("Loss (binary crossentropy)")
axes[0].set_xlabel("epoch"); axes[0].legend()

# (b) Accuracy
axes[1].plot(history.history["accuracy"], label="train")
axes[1].plot(history.history["val_accuracy"], label="validation")
axes[1].axhline(0.5, ls="--", c="grey", label="chance (50%)")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("epoch"); axes[1].legend()

# (c) Confusion matrix
cm = confusion_matrix(y_test, lstm_pred)
ConfusionMatrixDisplay(cm, display_labels=["down", "up"]).plot(ax=axes[2], colorbar=False)
axes[2].set_title(f"Confusion matrix (acc={acc_lstm:.3f})")

plt.tight_layout()
plt.savefig("results.png", dpi=120, bbox_inches="tight")
print("\n[chart saved as results.png]")
