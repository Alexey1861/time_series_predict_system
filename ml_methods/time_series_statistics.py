import pandas as pd
import matplotlib.pyplot as plt


def get_statistics(data):
    data = data[['<CLOSE>']]

    data.columns = ['close']

    data['exp_close'] = data['close'].ewm(alpha=0.1).mean()

    fig, ax = plt.subplots()

    ax.plot(data['close'], color='blue', label='Цена закрытия')
    ax.plot(data['exp_close'], color='red', label='Экспоненциальное скользящее среднее')

    ax.legend()
    ax.set_title('Котировки акций')
    ax.set_xlabel('Временная метка')
    ax.set_ylabel('Цена закрытия')

    return data['close'].describe(), fig
