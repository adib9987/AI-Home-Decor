import json
import os
import time
from typing import Dict, Any

LOG_FILE = os.path.join(os.path.dirname(__file__), "telemetry.log")

def log_event(event: Dict[str, Any]) -> None:
    """Save simple telemetry events (e.g., user interactions or AI decisions)."""
    event.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        # Fail silently if writing telemetry fails
        print(f"[Telemetry] Failed to log event: {e}")
