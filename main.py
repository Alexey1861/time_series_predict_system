from flask import Flask, render_template, request, send_file
from ml_methods.time_series_statistics import get_statistics
from ml_methods.lstm import lstm_model
from ml_methods.cnn import cnn_model
from ml_methods.dense_nn import dense_nn_model
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analysis', methods=['POST'])
def analysis():
    file = request.files['file']
    future = int(request.form['future'])

    data = pd.read_csv(file, sep=';')

    statistics, graphic = get_statistics(data)
    main_graphic = save_graph_to_png(graphic)

    dense_pred_data, linear_graphic = dense_nn_model(data, future)
    main_dense_graphic = save_graph_to_png(linear_graphic)
    dense_pred_data.to_csv('predict_data/dense.csv', index=False)

    cnn_pred_data, cnn_graphic = cnn_model(data, future)
    main_cnn_graphic = save_graph_to_png(cnn_graphic)
    cnn_pred_data.to_csv('predict_data/cnn.csv', index=False)

    lstm_pred_data, lstm_graphic = lstm_model(data, future)
    main_lstm_graphic = save_graph_to_png(lstm_graphic)
    lstm_pred_data.to_csv('predict_data/lstm.csv', index=False)

    return render_template(
        'analysis.html',
        data=statistics,
        main_graphic=main_graphic,
        main_dense_graphic=main_dense_graphic,
        main_cnn_graphic=main_cnn_graphic,
        main_lstm_graphic=main_lstm_graphic
    )


@app.route('/download_<filename>')
def download(filename):
    try:
        return send_file(f'predict_data/{filename}.csv', as_attachment=True)
    except Exception as e:
        return str(e), 404


def save_graph_to_png(figure):
    # Сохраняем график в буфер
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    # Кодируем в base64
    graphic = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return graphic


if __name__ == '__main__':
    app.run(debug=True)
