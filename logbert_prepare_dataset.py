import json
import os
import re
import sqlite3
from typing import Optional

import pandas as pd


DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
ANNOTATIONS_DB = os.path.join(DATA_DIR, "annotations.db")
TEMPLATES_PATH = os.path.join(DATA_DIR, "drain3_templates.jsonl")
RAW_LOG_PATH = os.path.join(DATA_DIR, "app.log")
OUT_CSV = os.path.join(DATA_DIR, "logbert_dataset.csv")


def read_template_annotations() -> dict[str, int]:
    """
    cluster_id -> label(0 normal, 1 anomaly)
    """
    if not os.path.exists(ANNOTATIONS_DB):
        return {}
    conn = sqlite3.connect(ANNOTATIONS_DB)
    try:
        rows = conn.execute("SELECT cluster_id, label FROM template_annotations").fetchall()
    finally:
        conn.close()

    out: dict[str, int] = {}
    for cid, lbl in rows:
        out[str(cid)] = 1 if lbl == "anomaly" else 0
    return out


def read_templates() -> dict[str, str]:
    """
    cluster_id -> template text (latest seen)
    """
    if not os.path.exists(TEMPLATES_PATH):
        return {}
    out: dict[str, str] = {}
    with open(TEMPLATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            cid = str(rec.get("cluster_id"))
            tpl = str(rec.get("template", ""))
            if cid and tpl:
                out[cid] = tpl
    return out


def heuristic_label(log_line: str) -> int:
    """
    Yalın pseudo-label:
    - status>=500 veya level=ERROR => anomaly (1)
    - aksi => normal (0)
    """
    if "level=ERROR" in log_line:
        return 1
    m = re.search(r"status=(\d+)", log_line)
    if m:
        try:
            status = int(m.group(1))
            if status >= 500:
                return 1
        except Exception:
            pass
    return 0


def read_raw_logs(limit: int = 5000) -> list[str]:
    if not os.path.exists(RAW_LOG_PATH):
        return []
    with open(RAW_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    lines = [ln.strip() for ln in lines if ln.strip()]
    return lines[-limit:]


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    ann = read_template_annotations()
    templates = read_templates()
    logs = read_raw_logs()

    rows = []

    # 1) Template-annotation verisini direkt supervised örnek yap
    for cid, tpl in templates.items():
        if cid in ann:
            rows.append({"text": tpl, "label": ann[cid], "source": "template_annotation"})

    # 2) Raw log satırlarından pseudo-label ile ek veri
    for ln in logs:
        rows.append({"text": ln, "label": heuristic_label(ln), "source": "raw_heuristic"})

    if not rows:
        # Boş dataset yazıp çık.
        pd.DataFrame(columns=["text", "label", "source"]).to_csv(OUT_CSV, index=False)
        print(f"Boş dataset yazıldı: {OUT_CSV}")
        return

    df = pd.DataFrame(rows)
    # Duplicate satırları temizle
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Dataset yazıldı: {OUT_CSV} satır={len(df)} anomaly={(df['label']==1).sum()} normal={(df['label']==0).sum()}")


if __name__ == "__main__":
    main()

