import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROM_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
OUT_JSON = os.environ.get("BENCHMARK_JSON", os.path.join("data", "metric_benchmark.json"))
OUT_CSV = os.environ.get("BENCHMARK_CSV", os.path.join("data", "metric_benchmark_points.csv"))
SECONDS = int(os.environ.get("BENCHMARK_SECONDS", "900"))
STEP = int(os.environ.get("BENCHMARK_STEP", "5"))
QUERY_TIMEOUT = int(os.environ.get("BENCHMARK_QUERY_TIMEOUT", "30"))
QUERY_RETRIES = int(os.environ.get("BENCHMARK_QUERY_RETRIES", "3"))


FEATURES = [
    "anomaly_memory_target_bytes",
    "anomaly_cpu_percent",
    "anomaly_latency_ms",
    "anomaly_error_rate",
]


def query_range(metric: str, seconds: int, step: int) -> Tuple[List[float], List[float]]:
    end = int(pd.Timestamp.utcnow().timestamp())
    start = end - seconds
    url = f"{PROM_URL}/api/v1/query_range"
    params = {"query": metric, "start": str(start), "end": str(end), "step": str(step)}
    last_err = None
    payload = None
    for _ in range(max(1, QUERY_RETRIES)):
        try:
            r = requests.get(url, params=params, timeout=QUERY_TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            break
        except Exception as e:
            last_err = e
            time.sleep(1)
    if payload is None:
        raise RuntimeError(f"Prometheus query_range failed for {metric}: {last_err}")
    result = payload.get("data", {}).get("result", [])
    if not result:
        return [], []
    values = result[0].get("values", [])
    ts = [float(v[0]) for v in values]
    vals = [float(v[1]) for v in values]
    return ts, vals


def build_frame(seconds: int, step: int) -> pd.DataFrame:
    base_ts: List[float] = []
    cols: Dict[str, List[float]] = {}
    for i, metric in enumerate(FEATURES):
        ts, vals = query_range(metric, seconds=seconds, step=step)
        if i == 0:
            base_ts = ts
        # hizalama: min uzunlukta kes
        if not base_ts or not vals:
            continue
        n = min(len(base_ts), len(vals))
        base_ts = base_ts[:n]
        vals = vals[:n]
        for k in list(cols.keys()):
            cols[k] = cols[k][:n]
        cols[metric] = vals

    if not base_ts or not cols:
        return pd.DataFrame()

    df = pd.DataFrame({"ts": base_ts})
    for c, v in cols.items():
        df[c] = v

    # Ground-truth proxy (demo için): kurallara göre label
    df["label"] = (
        (df["anomaly_memory_target_bytes"] > 200_000_000)
        | (df["anomaly_cpu_percent"] > 5)
        | (df["anomaly_latency_ms"] > 400)
        | (df["anomaly_error_rate"] > 0.05)
    ).astype(int)
    return df


def eval_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def run_zscore(df: pd.DataFrame) -> Dict[str, float]:
    # Univariate: her feature için zscore>3 ise anomaly; OR ile birleştir
    zflags = np.zeros(len(df), dtype=int)
    for c in FEATURES:
        x = df[c].values
        mu, std = x.mean(), x.std() + 1e-9
        z = np.abs((x - mu) / std)
        zflags = np.maximum(zflags, (z > 3.0).astype(int))
    return eval_binary(df["label"].values, zflags)


def run_isolation_forest(df: pd.DataFrame) -> Dict[str, float]:
    X = df[FEATURES].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # contamination demo label oranına yakın seçilir
    contam = float(max(0.01, min(0.4, df["label"].mean() + 1e-6)))
    model = IsolationForest(n_estimators=200, contamination=contam, random_state=42)
    pred = model.fit_predict(Xs)  # -1 anomaly
    y_pred = (pred == -1).astype(int)
    return eval_binary(df["label"].values, y_pred)


def run_autoencoder(df: pd.DataFrame) -> Dict[str, float]:
    """
    PyTorch varsa AE, yoksa simple fallback (reconstruction by PCA-like mean model).
    """
    X = df[FEATURES].values.astype(np.float32)
    y = df["label"].values.astype(int)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except Exception:
        # fallback: basit reconstruction error (mean vector)
        mu = Xs.mean(axis=0, keepdims=True)
        err = ((Xs - mu) ** 2).mean(axis=1)
        thr = np.quantile(err, 0.85)
        y_pred = (err > thr).astype(int)
        return eval_binary(y, y_pred)

    device = "cpu"
    Xt = torch.tensor(Xs, dtype=torch.float32, device=device)

    class AE(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(d, 8), nn.ReLU(), nn.Linear(8, 3))
            self.dec = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, d))

        def forward(self, z):
            return self.dec(self.enc(z))

    model = AE(Xs.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # sadece normal kabul edilen örneklerle eğit
    normal_idx = np.where(y == 0)[0]
    if len(normal_idx) < 10:
        normal_idx = np.arange(len(y))
    Xn = Xt[normal_idx]

    for _ in range(40):
        opt.zero_grad()
        rec = model(Xn)
        loss = loss_fn(rec, Xn)
        loss.backward()
        opt.step()

    with torch.no_grad():
        rec_all = model(Xt).cpu().numpy()
    err = ((rec_all - Xs) ** 2).mean(axis=1)
    thr = np.quantile(err, 0.85)
    y_pred = (err > thr).astype(int)
    return eval_binary(y, y_pred)


def main() -> None:
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    df = build_frame(seconds=SECONDS, step=STEP)
    if df.empty or len(df) < 20:
        payload = {"status": "insufficient_data", "rows": int(len(df))}
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(json.dumps(payload, ensure_ascii=False))
        return

    df.to_csv(OUT_CSV, index=False)

    report = {
        "rows": int(len(df)),
        "label_anomaly_ratio": float(df["label"].mean()),
        "zscore_univariate": run_zscore(df),
        "isolation_forest_multivariate": run_isolation_forest(df),
        "autoencoder": run_autoencoder(df),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

