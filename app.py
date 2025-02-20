from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import os
from datetime import datetime, timedelta
import shutil

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/charts'

CRYPTOS = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'DOGE-USD': 'Dogecoin'
}

def process_crypto(crypto, prediction_days):
    
    
    model = load_model(f'models/{crypto.replace("-", "_")}_model.keras')
    
    
    end = datetime.now()
    start = end - timedelta(days=365*15)
    data = yf.download(crypto, start=start, end=end)
    
    closing_price = data[['Close']].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_data = scaler.fit_transform(closing_price.dropna())
    
    
    last_100_days = scaled_data[-100:].reshape(1, -1, 1)
    future_predictions = []
    for _ in range(prediction_days):
        pred = model.predict(last_100_days)
        
        future_predictions.append(scaler.inverse_transform(pred)[0][0])
        
        last_100_days = np.append(last_100_days[:, 1:, :], pred.reshape(1, 1, -1), axis=1)
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    chart_files = {}
    
    # Price History Chart
    plt.figure(figsize=(10, 6))
    plt.plot(closing_price.index, closing_price['Close'], label='Price History')
    
    
    plt.title(f'{CRYPTOS[crypto]} Price History')
    plt.xlabel('Date')
    
    plt.ylabel('Price (USD)')
    
    history_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{crypto}_history.png')
    plt.savefig(history_path)
    plt.close()
    chart_files['history'] = f'charts/{crypto}_history.png'
    
    # Prediction Chart
    dates = [datetime.now() + timedelta(days=i) for i in range(1, prediction_days + 1)]
    plt.figure(figsize=(10, 6))
    
    plt.plot(dates, future_predictions, marker='o', color='purple')
    
    plt.title(f'{CRYPTOS[crypto]} {prediction_days}-Day Prediction')
    
    plt.xlabel('Date')
    plt.ylabel('Predicted Price (USD)')
    
    plt.xticks(rotation=45)
    prediction_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{crypto}_prediction.png')
    
    plt.savefig(prediction_path)
    plt.close()
    chart_files['prediction'] = f'charts/{crypto}_prediction.png'
    
    return {
        'name': CRYPTOS[crypto],
        'charts': chart_files,
        'predictions': future_predictions
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        crypto = request.form.get('crypto')
        
        prediction_days = int(request.form.get('days', 10))
        return redirect(url_for('results', crypto=crypto, days=prediction_days))
    
    return render_template('index.html', cryptos=CRYPTOS)

@app.route('/results')
def results():
    crypto = request.args.get('crypto')
    
    
    if crypto not in CRYPTOS:
        
        return redirect(url_for('index'))
    
    prediction_days = int(request.args.get('days', 10))
    analysis = process_crypto(crypto, prediction_days)
    
    now = datetime.now()
    
    
    predictions = [float(p) for p in analysis['predictions']]
    
    labels = [(now + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(prediction_days)]
    
    return render_template(
        'results.html',
        analysis=analysis,
        now=now,
        prediction_days=prediction_days,
        timedelta=timedelta,
        labels=labels,
        data=predictions
    )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
