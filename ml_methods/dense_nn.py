import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import StandardScaler


def dense_nn_model(df, future):
    data = df[['<CLOSE>']].copy()

    data.columns = ['close']
    data['exp_close'] = data['close'].ewm(alpha=0.1).mean()

    scaler = StandardScaler()
    scaler.fit(data)

    data = pd.DataFrame(scaler.transform(data))
    data.columns = ['close', 'exp_close']

    data['target'] = data.exp_close.shift(-future)

    test_data = data[data.target.isna()]

    train_data = data.dropna()

    model = keras.Sequential([
        keras.layers.Dense(units=1000, activation='relu'),
        keras.layers.Dense(units=1000, activation='relu'),
        keras.layers.Dense(units=1000, activation='relu'),
        keras.layers.Dense(units=1000, activation='relu'),
        keras.layers.Dense(units=1000, activation='relu'),
        keras.layers.Dense(units=1)
    ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )

    history = model.fit(
        train_data['exp_close'],
        train_data['target'],
        epochs=20
    )

    pred_test = pd.Series(model.predict(test_data['exp_close']).reshape(-1)) * scaler.scale_[1] + scaler.mean_[1]

    pred_test.index = [i for i in range(data.shape[0] - 1, data.shape[0] - 1 + pred_test.shape[0])]

    fig, ax = plt.subplots()

    ax.plot(data[data.index > data.shape[0] - 100]['close'] * scaler.scale_[0] + scaler.mean_[0], color='blue',
            label='Цена закрытия')
    ax.plot(data[data.index > data.shape[0] - 100]['exp_close'] * scaler.scale_[1] + scaler.mean_[1], color='yellow',
            label='Экспоненциальная скользящая средняя')
    ax.plot(pred_test, color='red', label=f'Предсказание модели на {future} дней')

    ax.legend()
    ax.set_title('Котировки акций')
    ax.set_xlabel('Временная метка')
    ax.set_ylabel('Цена закрытия')

    return pred_test, fig
