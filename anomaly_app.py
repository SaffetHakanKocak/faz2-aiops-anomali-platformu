import os
import random
import threading
import time

import psutil
from prometheus_client import Gauge, start_http_server


METRICS_PORT = int(os.environ.get("METRICS_PORT", "8000"))

# Zaman çizelgesi: önce düşük yük, sonra yük yükselt, sonra bir süre yüksek kal.
MEM_LOW_MB = int(os.environ.get("MEM_LOW_MB", "60"))
MEM_HIGH_MB = int(os.environ.get("MEM_HIGH_MB", "320"))
MEM_HIGH_AFTER_SEC = int(os.environ.get("MEM_HIGH_AFTER_SEC", "15"))
MEM_HIGH_DURATION_SEC = int(os.environ.get("MEM_HIGH_DURATION_SEC", "25"))

CPU_HIGH_AFTER_SEC = int(os.environ.get("CPU_HIGH_AFTER_SEC", "15"))
CPU_HIGH_DURATION_SEC = int(os.environ.get("CPU_HIGH_DURATION_SEC", "25"))

DEMO_ALWAYS_ON = os.environ.get("DEMO_ALWAYS_ON", "false").lower() in ("1", "true", "yes")
DEMO_MODE = os.environ.get("DEMO_MODE", "window").lower()  # window|cycle|always_on
RANDOM_SPIKE_PROB = float(os.environ.get("RANDOM_SPIKE_PROB", "0.08"))

# cycle demo: her CYCLE_PERIOD_SEC içinde kısa süreler low/high yap
CYCLE_PERIOD_SEC = int(os.environ.get("CYCLE_PERIOD_SEC", "60"))

# cycle dilimleri (sec)
# 0 - MEM_LOW
# MEM_HIGH_START - MEM_HIGH_END: memory yükselir
# CPU_HIGH_START - CPU_HIGH_END: cpu yükselir
MEM_HIGH_START = int(os.environ.get("MEM_HIGH_START", "10"))
MEM_HIGH_END = int(os.environ.get("MEM_HIGH_END", "25"))
CPU_HIGH_START = int(os.environ.get("CPU_HIGH_START", "25"))
CPU_HIGH_END = int(os.environ.get("CPU_HIGH_END", "40"))


anomaly_memory_rss_bytes = Gauge("anomaly_memory_rss_bytes", "Process RSS (bytes)")
anomaly_memory_target_bytes = Gauge(
    "anomaly_memory_target_bytes",
    "Demo hedef bellek (low/high pencerelerine göre)",
)
anomaly_cpu_percent = Gauge("anomaly_cpu_percent", "Process CPU percent (0-100)")
anomaly_latency_ms = Gauge("anomaly_latency_ms", "Demo request latency (ms)")
anomaly_error_rate = Gauge("anomaly_error_rate", "Demo error rate (0-1)")


def memory_worker(boom_event: threading.Event, stop_event: threading.Event) -> None:
    """
    Basit bellek yükleyici.
    - Düşük dönemde ~MEM_LOW_MB tut
    - boom_event set olunca ~MEM_HIGH_MB tut
    """
    low_b = MEM_LOW_MB * 1024 * 1024
    high_b = MEM_HIGH_MB * 1024 * 1024
    chunk = 1024 * 1024  # 1MB

    holder: list[bytearray] = []
    target_b = low_b

    while not stop_event.is_set():
        target_b = high_b if boom_event.is_set() else low_b

        # Mevcut holder boyutunu target'a yaklaştır.
        want_chunks = max(1, target_b // chunk)
        cur_chunks = len(holder)

        if cur_chunks < want_chunks:
            for _ in range(want_chunks - cur_chunks):
                holder.append(bytearray(chunk))
        elif cur_chunks > want_chunks:
            del holder[want_chunks:]

        time.sleep(0.2)


def cpu_worker(boom_event: threading.Event, stop_event: threading.Event) -> None:
    """
    Basit CPU yükleyici.
    boom_event set iken sürekli busy loop, değilken kısa uyku.
    """
    while not stop_event.is_set():
        if boom_event.is_set():
            # Zaman bazlı busy loop: container içinde cpu_percent'in deterministik artmasını sağlar.
            x = 0
            t_end = time.time() + 0.25  # yaklaşık 250ms CPU yükü
            while time.time() < t_end:
                x = (x + 1) % 1_000_000_007
        else:
            time.sleep(0.15)


def main() -> None:
    proc = psutil.Process(os.getpid())

    stop_event = threading.Event()
    mem_event = threading.Event()
    cpu_event = threading.Event()

    low_b = MEM_LOW_MB * 1024 * 1024
    high_b = MEM_HIGH_MB * 1024 * 1024

    t_mem = threading.Thread(target=memory_worker, args=(mem_event, stop_event), daemon=True)
    t_cpu = threading.Thread(target=cpu_worker, args=(cpu_event, stop_event), daemon=True)
    t_mem.start()
    t_cpu.start()

    # Prometheus metrik yayınını başlat.
    start_http_server(METRICS_PORT)

    # psutil ilk cpu_percent çağrısında 0 dönebilir; sonra düzgünleşir.
    proc.cpu_percent(interval=None)

    start_ts = time.time()
    last = start_ts

    while True:
        now = time.time()
        elapsed = now - start_ts

        if DEMO_ALWAYS_ON or DEMO_MODE == "always_on":
            mem_event.set()
            cpu_event.set()
        elif DEMO_MODE == "cycle":
            # Hem RSS hem CPU için periyodik düşük/yüksek pencereler.
            in_cycle = elapsed % CYCLE_PERIOD_SEC
            mem_high = MEM_HIGH_START <= in_cycle < MEM_HIGH_END
            cpu_high = CPU_HIGH_START <= in_cycle < CPU_HIGH_END
            if mem_high:
                mem_event.set()
            else:
                mem_event.clear()

            if cpu_high:
                cpu_event.set()
            else:
                cpu_event.clear()
        else:
            # window (orijinal): başlangıçtan itibaren tek seferlik pencereler
            mem_boom = MEM_HIGH_AFTER_SEC <= elapsed < (MEM_HIGH_AFTER_SEC + MEM_HIGH_DURATION_SEC)
            cpu_boom = CPU_HIGH_AFTER_SEC <= elapsed < (CPU_HIGH_AFTER_SEC + CPU_HIGH_DURATION_SEC)
            if mem_boom:
                mem_event.set()
            else:
                mem_event.clear()
            if cpu_boom:
                cpu_event.set()
            else:
                cpu_event.clear()

        # Her ~1s metrik güncelle
        if now - last >= 1.0:
            rss = proc.memory_info().rss
            cpu_pct = proc.cpu_percent(interval=None)

            # Gerçekçi görünüm için periyodik desene küçük rastgele oynamalar ekle.
            mem_target = high_b if mem_event.is_set() else low_b
            if mem_event.is_set():
                # High dönemde hedefi her ölçümde biraz farklılaştır.
                mem_target = int(random.uniform(0.85, 1.15) * high_b)
            elif random.random() < RANDOM_SPIKE_PROB:
                # Low dönemde de ara sıra kısa spike.
                mem_target = int(random.uniform(1.5, 2.2) * low_b)

            latency_base = 850.0 if cpu_event.is_set() else 110.0
            latency_ms = max(20.0, latency_base + random.uniform(-120.0, 180.0))

            error_base = 0.12 if mem_event.is_set() else 0.01
            error_rate = min(1.0, max(0.0, error_base + random.uniform(-0.02, 0.04)))

            anomaly_memory_rss_bytes.set(rss)
            anomaly_memory_target_bytes.set(float(mem_target))
            anomaly_cpu_percent.set(float(cpu_pct))
            anomaly_latency_ms.set(float(latency_ms))
            anomaly_error_rate.set(float(error_rate))

            last = now

        time.sleep(0.1)


if __name__ == "__main__":
    main()

