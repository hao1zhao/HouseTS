import numpy as np

EPS = 1e-8
def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))

def LOG_RMSE(pred, true):
    return np.sqrt(np.mean((np.log1p(true) - np.log1p(pred)) ** 2))
    
def SMAPE(pred, true):
    return np.mean(2 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + EPS))

def metric(pred, true, with_logrmse=False):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    smape = SMAPE(pred, true)
    if with_logrmse:
        lrmse = LOG_RMSE(pred, true)
        return mae, mse, rmse, mape, mspe, smape, lrmse
    return mae, mse, rmse, mape, mspe, smape