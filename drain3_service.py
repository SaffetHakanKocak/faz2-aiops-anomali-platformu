import json
import os
import time
from datetime import datetime, timezone

import requests
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from prometheus_client import Gauge, start_http_server


LOKI_URL = os.environ.get("LOKI_URL", "http://loki:3100").rstrip("/")
LOKI_QUERY = os.environ.get("LOKI_QUERY", '{service="log_generator"}')
METRICS_PORT = int(os.environ.get("METRICS_PORT", "8300"))
POLL_SEC = float(os.environ.get("POLL_SEC", "2"))
LOOKBACK_SEC = int(os.environ.get("LOOKBACK_SEC", "60"))

STATE_PATH = os.environ.get("STATE_PATH", "/data/drain3_state.bin")
TEMPLATES_PATH = os.environ.get("TEMPLATES_PATH", "/data/drain3_templates.jsonl")


drain3_total_templates = Gauge("drain3_total_templates", "Total distinct Drain3 templates observed")
drain3_new_template = Gauge("drain3_new_template", "1 if last processed line created a new template else 0")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def loki_query_range(start_ns: int, end_ns: int, limit: int = 2000):
    url = f"{LOKI_URL}/loki/api/v1/query_range"
    params = {
        "query": LOKI_QUERY,
        "start": str(start_ns),
        "end": str(end_ns),
        "limit": str(limit),
        "direction": "forward",
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def main() -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)

    cfg = TemplateMinerConfig()
    # drain3 sürümleri arasında config API farklılık gösterebiliyor.
    # En yalın şekilde snapshot ayarlarını burada veriyoruz.
    cfg.snapshot_interval_minutes = 1
    cfg.snapshot_compress_state = False
    cfg.snapshot_dir = os.path.dirname(STATE_PATH)
    cfg.snapshot_prefix = os.path.basename(STATE_PATH)

    miner = TemplateMiner(config=cfg)
    start_http_server(METRICS_PORT)

    last_end_ns = int((time.time() - LOOKBACK_SEC) * 1e9)

    while True:
        end_ns = int(time.time() * 1e9)
        start_ns = last_end_ns

        try:
            payload = loki_query_range(start_ns, end_ns)
            result = payload.get("data", {}).get("result", [])
        except Exception:
            time.sleep(POLL_SEC)
            continue

        # Loki: result -> stream list; each has values [[ts, line], ...]
        events = []
        for stream in result:
            for ts, line in stream.get("values", []):
                try:
                    ts_i = int(ts)
                except Exception:
                    ts_i = None
                events.append((ts_i, line))

        events.sort(key=lambda x: x[0] or 0)

        for ts_i, line in events:
            if ts_i is not None:
                last_end_ns = max(last_end_ns, ts_i + 1)

            res = miner.add_log_message(line)
            cluster = res.get("cluster_id")
            change_type = res.get("change_type")  # cluster_created / none / cluster_id_changed
            is_new = 1 if change_type == "cluster_created" else 0

            drain3_new_template.set(is_new)
            drain3_total_templates.set(len(miner.drain.clusters))

            if is_new:
                # Yeni template'i history'e yaz
                try:
                    record = {
                        "ts": now_iso(),
                        "cluster_id": cluster,
                        "template": res.get("template_mined"),
                        "size": res.get("cluster_size"),
                    }
                    with open(TEMPLATES_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception:
                    pass

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()

