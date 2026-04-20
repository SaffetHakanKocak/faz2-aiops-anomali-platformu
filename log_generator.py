import os
import time
import random
from datetime import datetime, timezone


LOG_PATH = os.environ.get("LOG_PATH", "/data/app.log")
CYCLE_PERIOD_SEC = float(os.environ.get("CYCLE_PERIOD_SEC", "30"))
ANOMALY_START_SEC = float(os.environ.get("ANOMALY_START_SEC", "20"))
ANOMALY_END_SEC = float(os.environ.get("ANOMALY_END_SEC", "30"))
LOG_INTERVAL_SEC = float(os.environ.get("LOG_INTERVAL_SEC", "0.2"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _make_normal(i: int) -> str:
    req_id = f"r-{i:06d}"
    latency = random.randint(20, 220)
    size_kb = random.randint(5, 120)
    return (
        f"{_now_iso()} level=INFO req_id={req_id} method=GET route=/api/items "
        f"status=200 latency_ms={latency} size_kb={size_kb} bytes=1024"
    )


def _make_anomaly(i: int) -> str:
    req_id = f"r-{i:06d}"
    latency = random.randint(250, 1200)
    # Loglarda anomaliyi belirgin kılan sabit token'lar.
    exc = random.choice(
        ["OutOfMemoryError: Java heap space", "NullPointerException", "TimeoutError", "DiskFullError"]
    )
    status = random.choice([500, 502, 503])
    return (
        f"{_now_iso()} level=ERROR req_id={req_id} method=GET route=/api/items "
        f"status={status} latency_ms={latency} exc={exc}"
    )


def main() -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    # Dosyayı append ile açıyoruz.
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        start = time.time()
        i = 0
        while True:
            elapsed = time.time() - start
            in_cycle = elapsed % CYCLE_PERIOD_SEC
            in_anomaly = ANOMALY_START_SEC <= in_cycle < ANOMALY_END_SEC

            line = _make_anomaly(i) if in_anomaly else _make_normal(i)
            f.write(line + "\n")
            f.flush()

            i += 1
            time.sleep(LOG_INTERVAL_SEC)


if __name__ == "__main__":
    main()

