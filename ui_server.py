from __future__ import annotations

import os
from datetime import datetime, timedelta

import requests
from fastapi import FastAPI, HTTPException
from fastapi import Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from annotation_store import AnnotationStore

PORT = int(os.environ.get("PORT", "4000"))
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://prometheus:9090").rstrip("/")
ALERTMANAGER_URL = os.environ.get("ALERTMANAGER_URL", "http://alertmanager:9093").rstrip("/")
HISTORY_PATH = os.environ.get("HISTORY_PATH", "/data/alert_history.jsonl")
ANNOTATIONS_DB = os.environ.get("ANNOTATIONS_DB", "/data/annotations.db")
TEMPLATES_PATH = os.environ.get("TEMPLATES_PATH", "/data/drain3_templates.jsonl")


app = FastAPI(title="Faz2 Anomali X UI")

ui_dir = os.path.join(os.path.dirname(__file__), "ui")
app.mount("/ui", StaticFiles(directory=os.path.join(ui_dir)), name="ui")


INDEX_PATH = os.path.join(ui_dir, "index.html")
store = AnnotationStore(ANNOTATIONS_DB)


def _read_index() -> str:
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="index.html bulunamadı")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _read_index()


@app.get("/api/alerts")
def get_alerts() -> JSONResponse:
    try:
        r = requests.get(f"{ALERTMANAGER_URL}/api/v2/alerts", timeout=5)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Alertmanager ulaşılamadı: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Alertmanager hata: {r.status_code} {r.text}")
    # Alertmanager /api/v2/alerts -> liste döner; UI'nin beklediği formata sarıyoruz.
    alerts = r.json()
    return JSONResponse({"alerts": alerts})


@app.get("/api/alert_history")
def get_alert_history(limit: int = 30) -> JSONResponse:
    """
    alert_receiver'ın yazdığı JSONL dosyasından son alert olaylarını okur.
    """
    try:
        limit = max(1, int(limit))
    except Exception:
        limit = 30

    if not os.path.exists(HISTORY_PATH):
        return JSONResponse({"events": []})

    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History okunamadı: {e}")

    tail = lines[-limit:]
    import json as _json

    events: list[dict] = []
    for line in tail:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(_json.loads(line))
        except Exception:
            continue

    return JSONResponse({"events": events})


@app.get("/api/templates")
def get_templates(limit: int = 50) -> JSONResponse:
    """
    Drain3'ün yazdığı JSONL template dosyasından son template kayıtlarını döner.
    Aynı cluster_id için en son template'i tutar.
    """
    if not os.path.exists(TEMPLATES_PATH):
        return JSONResponse({"templates": []})
    try:
        limit = max(1, int(limit))
    except Exception:
        limit = 50

    import json as _json

    try:
        with open(TEMPLATES_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Templates okunamadı: {e}")

    # Son N satırdan başlayıp cluster_id'leri uniqleyelim
    templates_by_id: dict[int, dict] = {}
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rec = _json.loads(line)
        except Exception:
            continue
        cid = rec.get("cluster_id")
        if cid is None:
            continue
        try:
            cid = int(cid)
        except Exception:
            continue
        if cid not in templates_by_id:
            templates_by_id[cid] = rec
        if len(templates_by_id) >= limit:
            break

    out = list(templates_by_id.values())
    out.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return JSONResponse({"templates": out})


@app.get("/api/annotations")
def list_annotations() -> JSONResponse:
    return JSONResponse({"annotations": store.list_all()})


@app.get("/api/annotations/{cluster_id}")
def get_annotation(cluster_id: int) -> JSONResponse:
    rec = store.get(cluster_id)
    return JSONResponse({"annotation": rec})


@app.post("/api/annotations/{cluster_id}")
def upsert_annotation(
    cluster_id: int,
    payload: dict = Body(...),
) -> JSONResponse:
    label = payload.get("label")
    note = payload.get("note")
    if label not in ("normal", "anomaly"):
        raise HTTPException(status_code=400, detail="label normal|anomaly olmalı")
    updated_at = datetime.utcnow().isoformat() + "Z"
    store.upsert(int(cluster_id), label, note, updated_at)
    return JSONResponse({"ok": True})


@app.get("/api/series")
def get_series(metric: str, seconds: int = 300, step: int = 5) -> JSONResponse:
    """
    Prometheus query_range ile zaman serisi döndürür.
    Frontend: {labels: [...], values: [...]}
    """
    end = datetime.utcnow()
    start = end - timedelta(seconds=seconds)
    start_s = start.timestamp()
    end_s = end.timestamp()

    query = metric
    params = {
        "query": query,
        "start": f"{start_s}",
        "end": f"{end_s}",
        "step": str(step),
    }
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/query_range", params=params, timeout=5)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Prometheus ulaşılamadı: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Prometheus hata: {r.status_code} {r.text}")

    payload = r.json()
    if payload.get("status") != "success":
        raise HTTPException(status_code=502, detail=f"Prometheus başarısız: {payload}")

    # Tek seri bekliyoruz (label setine göre birden fazla dönebilir).
    result = payload.get("data", {}).get("result", [])
    if not result:
        return JSONResponse({"timestamps": [], "values": []})

    series = result[0]
    values = series.get("values", [])

    timestamps: list[float] = []
    numeric: list[float] = []
    for ts_str, val_str in values:
        ts = float(ts_str)
        timestamps.append(ts)
        try:
            numeric.append(float(val_str))
        except Exception:
            numeric.append(0.0)

    # Labels'i tarayıcıda, kullanıcının yerel saat dilimine göre formatlıyoruz.
    return JSONResponse({"timestamps": timestamps, "values": numeric})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ui_server:app", host="0.0.0.0", port=PORT, reload=False)

