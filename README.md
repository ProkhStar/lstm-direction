# LSTM Direction: Can an LSTM predict the market direction?

In this project I trained an LSTM to predict whether the S&P 500 (SPY) goes up or down the next day. Most LSTM tutorials show great results because of data leakage, so I spent most of my time making sure my test was honest. I wanted a result I could actually trust, even if it was a bad one.

## Result

The LSTM did not beat simple baselines. It ended 6 percentage points below the best one.

```
Strategy                Accuracy (out-of-sample)
--------------------------------------------------
LSTM                        0.5144
Majority baseline           0.5744   (always predict "up")
Persistence baseline        0.5196   (tomorrow = today's direction)
--------------------------------------------------
LSTM edge:                  -6.0 p.p.
```

Something I found interesting is that the training loss got stuck at about 0.693. That is ln(2), which is the binary cross-entropy you get when you just guess randomly. So the model never learned anything useful, however long I trained it.

I think this makes sense. The daily direction of a big, liquid index is very hard to predict using only its own past prices and volume. For me the important part of the project is not the accuracy, it is that I can say this with confidence because I avoided the usual mistakes.

## How I avoided data leakage

- **Split first, scale after.** I fit the `StandardScaler` only on the training data and then applied it to the test data. If you scale the whole dataset before splitting, the model indirectly sees information from the future.
- **Stationary features.** I only used returns and things calculated from returns (momentum, volatility, volume change). I did not use raw prices, because a model that predicts price just copies yesterday's value and looks great for no real reason.
- **Predicting direction, not price.** The target is simply whether tomorrow's return is positive or not.
- **Baselines.** An accuracy number alone doesn't mean much, so I compared the LSTM with two simple rules: always predict the most common class, and assume tomorrow moves like today. A model is only useful if it beats both, and mine didn't.
- **Sequences built correctly.** Each window only uses data up to day `t` to predict day `t+1`. I built the sequences separately for train and test so nothing mixes between them. The validation set is the last part of the training data, so it always comes after the data used to train.

## Tools

Python, TensorFlow/Keras, scikit-learn, yfinance, NumPy, pandas, matplotlib.

## How to run

Open the notebook in Google Colab and click Runtime → Run all.
