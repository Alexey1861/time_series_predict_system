import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler
matplotlib.use('Agg')  # Использовать рендеринг без GUI


def lstm_model(df, future):
    data = df[['<CLOSE>']].copy()

    data.columns = ['close']

    data['exp_close'] = data['close'].ewm(alpha=0.1).mean()

    scaler = StandardScaler()
    scaler.fit(data)

    data = pd.DataFrame(scaler.transform(data))
    data.columns = ['close', 'exp_close']

    X_train, X_test, y_train = make_dataset(data, 100, future)

    model = keras.models.Sequential([
        keras.layers.LSTM(units=100, return_sequences=True),
        keras.layers.LSTM(units=100, return_sequences=False),
        keras.layers.Dense(units=1)
    ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=50
    )

    test_pred = pd.Series(model.predict(X_test).reshape(-1)) * scaler.scale_[1] + scaler.mean_[1]

    test_pred.index = [i for i in range(data.shape[0] - 1, data.shape[0] - 1 + test_pred.shape[0])]

    fig, ax = plt.subplots()

    ax.plot(data[data.index > data.shape[0] - 100]['close'] * scaler.scale_[0] + scaler.mean_[0], color='blue', label='Цена закрытия')
    ax.plot(data[data.index > data.shape[0] - 100]['exp_close'] * scaler.scale_[1] + scaler.mean_[1], color='yellow',
            label='Экспоненциальная скользящая средняя')
    ax.plot(test_pred, color='red', label=f'Предсказание модели на {future} дней')

    ax.legend()
    ax.set_title('Котировки акций')
    ax.set_xlabel('Временная метка')
    ax.set_ylabel('Цена закрытия')

    return test_pred, fig


def make_dataset(df, sequence, future):
    sequence -= 1

    data = df.copy()
    data = data.reset_index(drop=True)
    data.exp_close = data.exp_close.apply(lambda x: [x])

    X_train = []
    for i in range(data.shape[0] - sequence - future):
        X_train.append(data.loc[i: i + sequence, 'exp_close'].to_list())

    X_test = []
    for i in range(data.shape[0] - sequence - future, data.shape[0] - sequence):
        X_test.append(data.loc[i: i + sequence, 'exp_close'].to_list())

    y_train = []
    for i in range(sequence + future, data.shape[0]):
        y_train.append(data.loc[i, 'exp_close'])

    return np.array(X_train), np.array(X_test), np.array(y_train)
