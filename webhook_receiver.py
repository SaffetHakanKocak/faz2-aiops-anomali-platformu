import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime


PORT = int(os.environ.get("PORT", "9001"))
HISTORY_PATH = os.environ.get("HISTORY_PATH", "/data/alert_history.jsonl")


class Handler(BaseHTTPRequestHandler):
    def _append_history(self, payload: dict) -> None:
        """
        Alertmanager webhook POST'larında gelen alertleri history dosyasına yazar.
        Format: JSONL (her satır bir olay)
        """
        if not isinstance(payload, dict):
            return
        alerts = payload.get("alerts", [])
        if not isinstance(alerts, list) or not alerts:
            return

        received_at = datetime.utcnow().isoformat() + "Z"
        try:
            os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        except Exception:
            # directory açılamazsa sessiz geçiyoruz (demo yine çalışır)
            return

        lines: list[str] = []
        for a in alerts:
            if not isinstance(a, dict):
                continue
            status_obj = a.get("status")
            if isinstance(status_obj, str):
                status = status_obj
            elif isinstance(status_obj, dict):
                status = status_obj.get("state")
            else:
                status = None
            rec = {
                "receivedAt": received_at,
                "startsAt": a.get("startsAt"),
                "endsAt": a.get("endsAt"),
                "alertname": (a.get("labels") or {}).get("alertname"),
                "severity": (a.get("labels") or {}).get("severity"),
                "status": status,
                "summary": (a.get("annotations") or {}).get("summary"),
            }
            lines.append(json.dumps(rec, ensure_ascii=False))

        if not lines:
            return

        # Basit append: yarış olasılığı düşük (tek kullanıcı demo). Kilit yok.
        try:
            with open(HISTORY_PATH, "a", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
        except Exception:
            return

    def _send(self, code: int, body: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b""
        text = raw.decode("utf-8", errors="replace")

        ts = datetime.utcnow().isoformat() + "Z"
        try:
            payload = json.loads(text) if text.strip().startswith("{") else text
        except Exception:
            payload = text

        # Alertmanager'ın somut geldiğini görmek için basit çıktı basıyoruz.
        preview = text[:500].replace("\n", "\\n")
        print(f"[{ts}] POST {self.path} len={length} preview={preview}", flush=True)

        # History'e yaz.
        if isinstance(payload, dict):
            self._append_history(payload)

        self._send(200, "ok")

    def log_message(self, fmt, *args) -> None:
        # Varsayılan HTTP sunucu logunu bastır.
        return


def main() -> None:
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Webhook receiver listening on :{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

