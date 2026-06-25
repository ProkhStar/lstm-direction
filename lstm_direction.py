# =============================================================================
# LSTM para previsao de DIRECAO do mercado (sobe/desce) — versao metodologicamente correta
# -----------------------------------------------------------------------------
# O que esta versao corrige face a uma LSTM "de tutorial":
#   1. SPLIT TEMPORAL ANTES DE ESCALAR. O scaler e treinado (fit) apenas no
#      conjunto de treino. O teste e transformado com esse scaler. Sem isto,
#      o modelo "ve" o futuro atraves da normalizacao (lookahead bias).
#   2. FEATURES ESTACIONARIAS. Usamos retornos (nao niveis de preco). Prever o
#      nivel do preco leva o modelo a copiar o ultimo valor ("amanha = hoje"),
#      o que da metricas bonitas e inuteis.
#   3. TARGET = DIRECAO, nao nivel. Classificacao binaria: o retorno de amanha
#      e positivo (1) ou nao (0).
#   4. BASELINES OBRIGATORIAS. Um modelo so "vale" se bater (a) prever sempre a
#      classe maioritaria e (b) persistencia (amanha = direcao de hoje).
#      Sem baseline, qualquer accuracy e impossivel de interpretar.
#
# Como correr: pip install -r requirements.txt && python lstm_direction.py
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
# Reprodutibilidade (nota: com GPU pode nao ser 100% deterministico)
# -----------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# -----------------------------------------------------------------------------
# 1) Parametros
# -----------------------------------------------------------------------------
TICKER    = "SPY"
START     = "2010-01-01"
END       = None          # None = ate hoje
SEQ_LEN   = 60            # dias de historico que entram em cada janela
TEST_SIZE = 0.20          # fracao final da serie reservada para teste (out-of-sample)
BATCH     = 32
EPOCHS    = 60

# -----------------------------------------------------------------------------
# 2) Download
# -----------------------------------------------------------------------------
df = yf.download(TICKER, start=START, end=END, progress=False, auto_adjust=True)
if df.empty:
    raise SystemExit(f"Sem dados para {TICKER}.")

# yfinance pode devolver colunas MultiIndex (ex.: ('Close','SPY')). Achatar:
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[["Close", "Volume"]].dropna()
print(f"Dados obtidos: {df.shape[0]} dias  ({df.index.min().date()} -> {df.index.max().date()})")

# -----------------------------------------------------------------------------
# 3) Feature engineering — TUDO estacionario (retornos, nao niveis)
# -----------------------------------------------------------------------------
feat = pd.DataFrame(index=df.index)
feat["log_ret"]   = np.log(df["Close"] / df["Close"].shift(1))      # retorno do dia
feat["mom_5"]     = feat["log_ret"].rolling(5).mean()               # momentum (5d)
feat["vol_10"]    = feat["log_ret"].rolling(10).std()               # volatilidade (10d)
feat["vol_chg"]   = np.log(df["Volume"] / df["Volume"].shift(1))    # variacao de volume
feat["ret_abs_5"] = feat["log_ret"].abs().rolling(5).mean()         # magnitude media (5d)

FEATURES = ["log_ret", "mom_5", "vol_10", "vol_chg", "ret_abs_5"]

# TARGET: direcao do retorno de AMANHA (1 = positivo, 0 = caso contrario)
feat["target"] = (feat["log_ret"].shift(-1) > 0).astype(int)

# Direcao de HOJE (serve para a baseline de persistencia)
feat["dir_today"] = (feat["log_ret"] > 0).astype(int)

feat = feat.dropna().copy()   # remove NaNs do rolling (inicio) e do shift(-1) (fim)

# -----------------------------------------------------------------------------
# 4) Split TEMPORAL primeiro, escalar depois (a regra de ouro contra leakage)
# -----------------------------------------------------------------------------
split_idx = int((1 - TEST_SIZE) * len(feat))
train_df  = feat.iloc[:split_idx]
test_df   = feat.iloc[split_idx:]

scaler = StandardScaler()                                  # StandardScaler > MinMax para retornos (robusto a outliers)
train_scaled = scaler.fit_transform(train_df[FEATURES])   # fit SO no treino
test_scaled  = scaler.transform(test_df[FEATURES])        # transform no teste com o scaler do treino

train_target = train_df["target"].values
test_target  = test_df["target"].values
test_dir_today = test_df["dir_today"].values              # para a baseline de persistencia

# -----------------------------------------------------------------------------
# 5) Construcao de sequencias (causal: janela ate ao dia t -> direcao de t+1)
#    Nota: criamos as janelas dentro de cada conjunto. Perdem-se as primeiras
#    SEQ_LEN-1 amostras de cada lado — negligivel e, sobretudo, ZERO leakage.
# -----------------------------------------------------------------------------
def make_sequences(X2d, y1d, seq_len):
    X, y = [], []
    for i in range(seq_len - 1, len(X2d)):
        X.append(X2d[i - seq_len + 1 : i + 1])
        y.append(y1d[i])
    return np.array(X), np.array(y)

X_train, y_train = make_sequences(train_scaled, train_target, SEQ_LEN)
X_test,  y_test  = make_sequences(test_scaled,  test_target,  SEQ_LEN)

# Persistencia alinhada com y_test (mesma perda de SEQ_LEN-1 no inicio)
persist_pred = test_dir_today[SEQ_LEN - 1:]

print(f"Treino: {X_train.shape}  |  Teste: {X_test.shape}")
print(f"Distribuicao do target (treino): sobe={y_train.mean():.1%}  desce={1 - y_train.mean():.1%}")

# -----------------------------------------------------------------------------
# 6) Modelo — LSTM de classificacao binaria (arquitetura modesta p/ limitar overfit)
# -----------------------------------------------------------------------------
model = Sequential([
    Input(shape=(SEQ_LEN, len(FEATURES))),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid"),            # probabilidade de "sobe"
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

# Pesos de classe (o mercado sobe ligeiramente mais vezes do que desce)
cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
class_weight = {0: cw[0], 1: cw[1]}

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
]

# -----------------------------------------------------------------------------
# 7) Treino
#    validation_split usa a CAUDA do treino (Keras nao baralha aqui), o que
#    mantem a validacao temporalmente posterior ao treino. Correto para series.
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
# 8) Avaliacao + BASELINES (o coracao da honestidade metodologica)
# -----------------------------------------------------------------------------
proba = model.predict(X_test).flatten()
lstm_pred = (proba > 0.5).astype(int)

acc_lstm     = accuracy_score(y_test, lstm_pred)
majority_cls = int(round(y_train.mean()))
acc_majority = accuracy_score(y_test, np.full_like(y_test, majority_cls))
acc_persist  = accuracy_score(y_test, persist_pred)

print("\n" + "=" * 60)
print("RESULTADOS (out-of-sample)")
print("=" * 60)
print(f"  LSTM ............... {acc_lstm:.4f}")
print(f"  Baseline maioria ... {acc_majority:.4f}   (prever sempre a classe '{majority_cls}')")
print(f"  Baseline persist. .. {acc_persist:.4f}   (amanha = direcao de hoje)")
best_baseline = max(acc_majority, acc_persist)
edge = acc_lstm - best_baseline
print("-" * 60)
print(f"  Edge da LSTM sobre a melhor baseline: {edge:+.4f}  ({edge*100:+.2f} p.p.)")

# Veredicto honesto
print("\nLeitura:")
if edge > 0.02:
    print("  A LSTM bate a melhor baseline em >2 p.p. Resultado promissor — mas")
    print("  confirma com walk-forward e multiplas seeds antes de acreditar.")
elif edge > 0.0:
    print("  A LSTM esta marginalmente acima da baseline. Dentro do ruido estatistico:")
    print("  NAO e um edge demonstravel. Resultado honesto e esperado para direcao do indice.")
else:
    print("  A LSTM NAO bate a baseline. E o resultado correto e esperado: a direcao")
    print("  diaria de um indice liquido e essencialmente imprevisivel. O valor deste")
    print("  exercicio e a METODOLOGIA a prova de leakage, nao o numero final.")
print("=" * 60)

# -----------------------------------------------------------------------------
# 9) Graficos
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

# (a) Curvas de treino
axes[0].plot(history.history["loss"], label="treino")
axes[0].plot(history.history["val_loss"], label="validacao")
axes[0].set_title("Loss (binary crossentropy)")
axes[0].set_xlabel("epoca"); axes[0].legend()

# (b) Accuracy de treino
axes[1].plot(history.history["accuracy"], label="treino")
axes[1].plot(history.history["val_accuracy"], label="validacao")
axes[1].axhline(0.5, ls="--", c="grey", label="acaso (50%)")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("epoca"); axes[1].legend()

# (c) Matriz de confusao
cm = confusion_matrix(y_test, lstm_pred)
ConfusionMatrixDisplay(cm, display_labels=["desce", "sobe"]).plot(ax=axes[2], colorbar=False)
axes[2].set_title(f"Matriz de confusao (acc={acc_lstm:.3f})")

plt.tight_layout()
plt.savefig("resultados.png", dpi=120, bbox_inches="tight")
print("\n[grafico guardado em resultados.png]")
