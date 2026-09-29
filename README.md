# Predicting SPY's daily direction with an LSTM

I wanted to see if an LSTM could predict whether SPY goes up or down the
next day. My main focus was doing it without data leakage and comparing
the model against simple baselines, so I could trust the result even if
it was bad.

It was bad. The LSTM got 47.9% accuracy on the test set, while just
predicting "up" every day got 56.3%.

| Model                          | Test accuracy |
|--------------------------------|---------------|
| LSTM                           | 47.9%         |
| Always predict "up"            | 56.3%         |
| Persistence (tomorrow = today) | 51.7%         |

<img width="2038" height="528" alt="results" src="https://github.com/user-attachments/assets/dacff24e-9495-4ed0-9272-371c03626f97" />

The validation accuracy stays around 50% for the whole training, so the
model never really learned anything useful.

## What I did to avoid leakage

- Used returns, volatility and volume changes instead of raw prices
- Split train/test by date first (80/20), and only then fitted the scaler,
  on the training data only
- Built the 60-day windows separately inside train and test, so no window
  crosses the split
- Used the end of the training period as validation, without shuffling

## Why it lost to "always up"

I used balanced class weights, which probably made the model predict
"down" too often. The test period (roughly 2023 to 2026) was mostly a
rising market, so those "down" predictions were costly.

## Limitations

Single train/test split, single seed, and the data goes up to the day you
run it, so the numbers change slightly each run.


