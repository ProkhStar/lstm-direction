# LSTM Direction — Predicting Market Direction, and Showing It Can't Be Done

An LSTM model that predicts the daily direction (up/down) of the S&P 500 index (SPY), built with one central concern: eliminating every form of data leakage that makes a tutorial-style LSTM look deceptively good. The goal was not to get an impressive number, but to build a test honest enough to trust the result, even when that result is negative.

## Result

**The LSTM does not beat trivial baselines.** It finished 6 percentage points below the best baseline:

```
Strategy                Accuracy (out-of-sample)
--------------------------------------------------
LSTM                        0.5144
Majority baseline           0.5744   (always predict "up")
Persistence baseline        0.5196   (tomorrow = today's direction)
--------------------------------------------------
LSTM edge:                  -6.0 p.p.
```

More telling than the accuracy itself: the training loss plateaued at ~0.693, which is exactly `ln(2)`, the binary cross-entropy of pure chance predictions. No matter how long it trained, the model never found any predictive structure. This is not a bug; it is the mathematical signature of a signal that does not exist.

This is the correct and expected result. The daily direction of a liquid, efficient index is essentially unpredictable from its own price and volume history. The value of this project is not the final number, but the leakage-proof methodology that makes it possible to state this with confidence, instead of reporting an accuracy inflated by a silent error.

## Leakage-proof methodology

What sets this project apart from a tutorial LSTM is the set of decisions that eliminate lookahead bias, the error that makes a model appear to predict the future when it is actually seeing it:

- **Temporal split before scaling.** The `StandardScaler` is fitted only on the training set; the test set is transformed using the training parameters. The classic mistake of scaling the whole series before splitting lets the model "see" statistics from the future through the normalisation. Here, that does not happen.
- **Stationary features.** All features are returns or return-derived (momentum, volatility, volume change), never price levels. Predicting the price level makes the model copy the last value ("tomorrow = today") and show spectacular but completely useless metrics.
- **Direction target, not level target.** Honest binary classification: is tomorrow's return positive or not? This avoids the "predict the price" trap, which disguises persistence as predictive ability.
- **Mandatory baselines.** An accuracy figure on its own is uninterpretable. The model is compared against two trivial references: always predicting the majority class, and persistence (tomorrow follows today's direction). A model is only worth something if it beats both, and this one does not.
- **Causal sequence construction.** Each window uses only data up to day `t` to predict `t+1`, and sequences are built within each set (train/test) separately, with no contamination between them. Validation uses the temporal tail of the training set, keeping it strictly later than the training data.

## Stack

Python, TensorFlow/Keras (LSTM), scikit-learn, yfinance, NumPy/pandas, matplotlib.

## How to run

```
pip install -r requirements.txt
python lstm_direction.py
```
