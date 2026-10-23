"""Прогноз спроса по SKU и рекомендации по закупке.

Модель: Ridge-регрессия на лагах (1, 2, 3, 12 мес.) и месяце года, рекурсивный прогноз.
Качество: MAPE на отложенных последних 6 месяцах. Страховой запас: z * sigma * sqrt(срок поставки).
"""
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

Z = 1.65  # уровень сервиса ~95%


def _features(hist, month):
    m = np.zeros(12)
    m[month - 1] = 1
    return np.concatenate([[hist[-1], hist[-2], hist[-3], hist[-12]], m])


def _fit(s: pd.Series) -> Ridge:
    v = s.values.astype(float)
    X = [_features(v[:i], s.index[i].month) for i in range(12, len(v))]
    return Ridge(alpha=1.0).fit(np.array(X), v[12:])


def _predict(model, s: pd.Series, steps: int):
    hist, d, out = list(s.values.astype(float)), s.index[-1], []
    for _ in range(steps):
        d = d + pd.offsets.MonthBegin(1)
        p = max(0.0, float(model.predict([_features(hist, d.month)])[0]))
        hist.append(p)
        out.append(p)
    return out


def mape(actual, pred) -> float:
    a, p = np.array(actual, float), np.array(pred, float)
    mask = a > 0
    return float(np.mean(np.abs(a[mask] - p[mask]) / a[mask]) * 100)


def forecast_sku(series: pd.Series, horizon: int = 6, holdout: int = 6):
    """Возвращает (прогноз на horizon мес., MAPE на holdout, sigma ошибки)."""
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]
    back = _predict(_fit(train), train, holdout)
    sigma = float(np.std(test.values - np.array(back)))
    return _predict(_fit(series), series, horizon), mape(test.values, back), sigma


def purchase_plan(sales: pd.DataFrame, inventory: pd.DataFrame, horizon: int = 6) -> pd.DataFrame:
    sales = sales.assign(month=pd.to_datetime(sales["month"]))
    rows = []
    for _, r in inventory.iterrows():
        s = sales[sales.sku == r.sku].set_index("month")["qty"].asfreq("MS").fillna(0)
        fc, err, sigma = forecast_sku(s, horizon)
        lead = int(r.lead_months)
        safety = Z * sigma * math.sqrt(lead)
        need = sum(fc[:lead]) + safety
        order = max(0, math.ceil(need - r.stock))
        if order > 0:
            status = "ЗАКАЗАТЬ СЕЙЧАС"
        elif r.stock > sum(fc[:horizon]) * 1.3:
            status = "ИЗЛИШЕК"
        else:
            status = "ОК"
        rows.append({"sku": r.sku, "name": r["name"], "stock": int(r.stock), "lead_months": lead,
                     "forecast_next_month": round(fc[0]), f"forecast_{horizon}m": round(sum(fc)),
                     "safety_stock": round(safety), "order_qty": order,
                     "mape_%": round(err, 1), "status": status})
    return pd.DataFrame(rows)
