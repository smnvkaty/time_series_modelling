import pandas as pd 

import numpy as np 
from tqdm import tqdm
from copy import deepcopy
from itertools import product


import statsmodels.api as sm

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, coint
from sklearn.metrics import mean_absolute_error
from warnings import filterwarnings
filterwarnings('ignore')


import os


# Генерируем индексы для окон разных типов
def win_cv(win_size, win_step, data_size, test_size, win_type):
    
    cur = 0
    while cur < data_size - win_size:
        if win_type == 'sliding':
            yield (cur, cur + win_size, win_size - test_size)
        if win_type == 'expanding':
            yield (0, cur + win_size, cur + win_size - test_size)
        cur += win_step


    if win_type == 'sliding':
        yield (data_size - win_size, data_size, win_size - test_size)
    if win_type == 'expanding':
        yield (0, data_size, data_size - test_size)

def mae(y_true, y_pred): 
    y_true, y_pred = np.array([y_true]), np.array([y_pred])
    return np.mean(np.abs((y_true - y_pred)))


# Проводим кросс-валидацию на окнах и возвращаем пары гиперпараметры + ошибка для заявленной метрики
def cross_val_window(data, windows, model_type, metric, params, maps):
    metrics = []

    for win in windows:
        sub_par = deepcopy(params)
        feature_map = deepcopy(maps[0])

        pred_xgb = None
        pred_rf = None
        if model_type in ('boosting', 'ensemble'):
            X = data.iloc[win[0]:win[1], feature_map].copy()
            y = data.iloc[win[0]:win[1], 0].copy()
            X_train, y_train, X_test, y_test = X.iloc[:win[2],:], y.iloc[:win[2]], X.iloc[win[2]:,:], y.iloc[win[2]:]

            par = sub_par.pop()
            xgb = GradientBoostingRegressor(    
                                                **par,
                                                random_state=0
                                                )
            xgb.fit(X_train, y_train)
            pred_xgb = np.array([xgb.predict(X_test).item()])
            prediction = pred_xgb
        
        if model_type in ('random_forest', 'ensemble'):
            X = data.iloc[win[0]:win[1], feature_map].copy()
            y = data.iloc[win[0]:win[1], 0].copy()
            X_train, y_train, X_test, y_test = X.iloc[:win[2],:], y.iloc[:win[2]], X.iloc[win[2]:,:], y.iloc[win[2]:]

            par = sub_par.pop()
            rf = RandomForestRegressor(
                                        **par,
                                        random_state=0,
                                        n_jobs=os.cpu_count()-1
                                        )
            rf.fit(X_train, y_train)
            pred_rf = np.array([rf.predict(X_test).item()])
            prediction = pred_rf
        
 
        if model_type == 'ensemble' and (pred_xgb is not None) and (pred_rf is not None):

            prediction = np.array([np.mean([pred_xgb, pred_rf])])

        metrics.append(metric(prediction, y_test)) 
    # Возвращаем ошибку
    return np.mean(metrics)


#Блок для эконометрических моделей 

#тест Дики-Фуллера для проверки стационарности ряда
def adf_pval(series):
    return adfuller(series.dropna())[1]

#тест Энгла-Гренджера для проверки коинтеграции рядов (для ARIMAX)
def engle_granger(y, x):
    return coint(y, x)[1]

#проводим тесты, чтобы понять, берем ли мы ряды в уровнях или в разницах
def test_stationarity_and_cointegration(df, target_col, exog_cols, alpha=0.05, model_type = None):
    stationary = {}
    for col in [target_col] + exog_cols:
        pval = adf_pval(df[col])
        if pval < alpha:
          stationary[col] = 'levels'
        else:
          stationary[col] = 'diff'
        print(f"ADF test {col}: p-value={pval:.4f} -> {'стац.' if pval < alpha else 'нестац.'}")
    if (len(set(stationary.values())) == 1) and (set(stationary.values()) == 'levels'):
        return 'levels'  # всё стационарно

    # Проверим коинтеграцию
    if model_type == 'ARIMAX':
      cointegrated = any(engle_granger(df[target_col], df[col]) < alpha for col in exog_cols)
      print("Engle-Granger тест:", "коинтеграция обнаружена" if cointegrated else "коинтеграции нет")
      return 'levels' if cointegrated else stationary
    else:
       return stationary

#собираем финальный датасет: в уровнях или в разницах
def prepare_data(df, mode, target_col, exog_cols):
    df = df.copy()
    if mode == 'levels':
      return df
    elif mode[target_col] == 'diff':
        df[target_col] = df[target_col].diff()
        for col in exog_cols:
          if mode[col] == 'diff':
            df[col] = df[col].diff()
    df = df.dropna()
    return df

#подбираем p, q для ARIMA и ARIMAX по информационному критерию BIC
def try_arima_models(df, target_col, exog_cols=None, max_p=5, max_q=5, test_size=0.2, criterion='bic', target_type = None):
    results = []
    y = df[target_col]
    X = df[exog_cols] if exog_cols else None
    test_size = int(test_size * len(df))
    y_train, y_test = y[:-test_size], y[-test_size:]
    X_train, X_test = (X[:-test_size], X[-test_size:]) if X is not None else (None, None)

    for p, q in product(range(max_p+1), range(max_q+1)):
        if p == 0 and q == 0:
            continue
        try:
            if target_type == 'diff':
              model = SARIMAX(endog=y_train, exog=X_train, order=(p, 0, q), enforce_stationarity = False)
              fit = model.fit()
              pred = fit.predict(start=y_test.index[0], end=y_test.index[-1], exog=X_test)
              ic = getattr(fit, criterion)
              last_actual = df[target_col].iloc[-test_size - 1]
              forecast_level = pred.cumsum() + last_actual
              mae = mean_absolute_error(y_test, forecast_level)
            else:
              model = SARIMAX(endog=y_train, exog=X_train, order=(p, 0, q), enforce_stationarity = False)
              fit = model.fit()
              pred = fit.predict(start=y_test.index[0], end=y_test.index[-1], exog=X_test)
              forecast_level = pred
              ic = getattr(fit, criterion)
              mae = mean_absolute_error(y_test, pred)

            results.append({'order': (p, 0, q), 'ic': ic, 'mae': mae, 'model': fit, 'predicts': pred})
        except:
            continue

    if not results:
        raise ValueError("Ни одна модель не обучилась.")
    # сначала по критерию, потом по MAE
    best_models = sorted(results, key=lambda x: x['ic'])[:3]
    best = sorted(best_models, key=lambda x: x['mae'])[0]
    return best
