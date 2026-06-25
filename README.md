# LSTM Direction — Prever a direção do mercado, e provar que não dá

Modelo LSTM para prever a direção diária (sobe/desce) do índice S&P 500 (SPY), construído com uma preocupação central: **eliminar todas as formas de leakage que tornam uma LSTM de tutorial enganadoramente boa**. O objetivo não foi obter um número bonito, mas montar um teste honesto o suficiente para se acreditar no resultado — mesmo quando o resultado é negativo.

## Resultado

A LSTM **não supera baselines triviais**. Ficou abaixo da melhor baseline por 6 pontos percentuais:

```
Estrategia             Accuracy (out-of-sample)
--------------------------------------------------
LSTM                       0.5144
Baseline maioria           0.5744   (prever sempre "sobe")
Baseline persistencia      0.5196   (amanha = direcao de hoje)
--------------------------------------------------
Edge da LSTM:              -6.0 p.p.
```

Mais revelador do que a accuracy: a **loss de treino estabilizou em ~0.693**, que é exatamente `ln(2)` — o valor da entropia cruzada binária para previsões de puro acaso. O modelo, por mais que treinasse, nunca encontrou estrutura preditiva. Não é um bug: é a assinatura matemática de um sinal que não existe.

Este é o resultado correto e esperado. **A direção diária de um índice líquido e eficiente é essencialmente imprevisível** a partir do seu próprio histórico de preços e volume. O valor do projeto não é o número final — é a metodologia à prova de leakage que permite afirmá-lo com confiança, em vez de reportar uma accuracy inflacionada por um erro silencioso.

## A metodologia à prova de leakage

O que distingue este projeto de uma LSTM de tutorial é o conjunto de decisões que eliminam o lookahead bias — o erro que faz um modelo parecer prever o futuro quando na verdade está a vê-lo:

**Split temporal antes de escalar.** O `StandardScaler` é ajustado (`fit`) apenas no conjunto de treino; o teste é transformado com os parâmetros do treino. O erro clássico — escalar a série toda antes de dividir — deixa o modelo "ver" estatísticas do futuro através da normalização. Aqui isso não acontece.

**Features estacionárias.** Todas as features são retornos ou derivadas de retornos (momentum, volatilidade, variação de volume), nunca níveis de preço. Prever o nível do preço leva o modelo a copiar o último valor ("amanhã = hoje") e a exibir métricas espetaculares e completamente inúteis.

**Target de direção, não de nível.** Classificação binária honesta: o retorno de amanhã é positivo ou não. Sem a armadilha de "prever o preço" que mascara a persistência como capacidade preditiva.

**Baselines obrigatórias.** Uma accuracy isolada é ininterpretável. O modelo é comparado contra duas referências triviais: prever sempre a classe maioritária, e persistência (amanhã segue a direção de hoje). Um modelo só "vale" se bater ambas — e este não bate.

**Construção causal das sequências.** Cada janela usa apenas dados até ao dia `t` para prever `t+1`, e as sequências são formadas dentro de cada conjunto (treino/teste) separadamente, sem qualquer contaminação entre eles. A validação usa a cauda temporal do treino, mantendo-a posterior aos dados de treino.

## Stack

Python, TensorFlow/Keras (LSTM), scikit-learn, yfinance, NumPy/pandas, matplotlib.

## Como correr

```bash
pip install -r requirements.txt
python lstm_direction.py
```

## Nota

Este é um projeto companheiro de um estudo mais amplo sobre extração de sinal em dados financeiros públicos. A conclusão — que a direção diária não é previsível a partir de dados de preço/volume — motivou, noutro projeto, uma abordagem por reforço focada em *gestão de risco* em vez de previsão de direção.

---
*Projeto de investigação. Não constitui aconselhamento financeiro.*
