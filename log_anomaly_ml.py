import os
import time
import threading
from collections import deque

import numpy as np
from prometheus_client import Gauge, start_http_server

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest


LOG_PATH = os.environ.get("LOG_PATH", "/data/app.log")
METRICS_PORT = int(os.environ.get("METRICS_PORT", "8200"))

BASELINE_LOGS = int(os.environ.get("BASELINE_LOGS", "300"))
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "200"))
ANOMALY_PERCENTILE = float(os.environ.get("ANOMALY_PERCENTILE", "5"))  # düşük skor = anomali


log_anomaly_score = Gauge("log_anomaly_score", "Anomaly score (lower means more anomalous)")
log_anomaly_flag = Gauge("log_anomaly_flag", "1 if current log line is anomalous else 0")
log_anomaly_rate = Gauge("log_anomaly_rate", "Fraction of anomalous logs in recent window (0-1)")


def _tail_f(path: str):
    """
    Yeni log satırlarını kuyruklar.
    İlk açılışta dosyanın sonuna gidip sadece yeni satırları dinler.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.05)
                continue
            yield line.rstrip("\n")


def main() -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    # Model objeleri
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
    clf: IsolationForest | None = None
    threshold: float | None = None

    baseline_lines: list[str] = []
    recent_flags: deque[int] = deque(maxlen=WINDOW_SIZE)
    current_line_count = 0

    # Ensitif: Prometheus exporter
    start_http_server(METRICS_PORT)

    # Baseline toplama: yeni satırları saymaya başlıyoruz.
    for line in _tail_f(LOG_PATH):
        current_line_count += 1
        if clf is None:
            # Baseline'i deterministik tutmak için sadece "normal" satırları topluyoruz.
            # (Jeneratör normalde level=INFO status=200 üretir; anomaly penceresinde level=ERROR olur.)
            if "level=INFO" in line and "status=200" in line:
                baseline_lines.append(line)
            # Baseline'i dolduralım ve modeli eğitelim.
            if len(baseline_lines) >= BASELINE_LOGS:
                X = vectorizer.fit_transform(baseline_lines)
                clf = IsolationForest(
                    n_estimators=200,
                    contamination="auto",
                    random_state=42,
                )
                clf.fit(X)
                # score_samples: daha yüksek daha normal; burada "anomalı"ları düşük skorlardan ayırıyoruz.
                base_scores = clf.score_samples(X)
                cutoff_index = max(0, int(np.floor(len(base_scores) * (ANOMALY_PERCENTILE / 100.0))))
                base_scores_sorted = np.sort(base_scores)
                threshold = float(base_scores_sorted[cutoff_index])
                # baseline window'unu boş bırak.
                for _ in range(WINDOW_SIZE):
                    recent_flags.append(0)
            continue

        # Model hazır: score hesapla
        X = vectorizer.transform([line])
        assert clf is not None and threshold is not None
        score = float(clf.score_samples(X)[0])
        # Anomali: score threshold'un altında ise.
        is_anomaly = int(score < threshold)
        recent_flags.append(is_anomaly)

        log_anomaly_score.set(score)
        log_anomaly_flag.set(is_anomaly)
        log_anomaly_rate.set(sum(recent_flags) / len(recent_flags))


if __name__ == "__main__":
    main()

