"""Stop a run cleanly before it takes the machine's memory.

A daemon thread checks the process RSS and, on Apple MPS, the driver's
allocated memory; above the limit it prints a message and exits with code 3.
"""

import os
import subprocess
import threading
import time


def rss_gb(pid=None):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid or os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return int(out or 0) / 1e6


def mps_gb():
    try:
        import torch
        if torch.backends.mps.is_available():
            return torch.mps.driver_allocated_memory() / 1e9
    except Exception:
        pass
    return 0.0


def start(limit_gb, interval=0.5, log=print):
    def watch():
        while True:
            r, g = rss_gb(), mps_gb()
            if r + g > limit_gb:
                log(f"memcap: stopping, rss={r:.2f} GB + mps={g:.2f} GB > {limit_gb} GB")
                os._exit(3)
            time.sleep(interval)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    return t
