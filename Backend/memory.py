# Backend/memory.py
from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List, Optional

# -------- configuration (env overrides optional) --------
HERE = os.path.dirname(os.path.abspath(__file__))
PROTOS_DIR = os.getenv("PLANNER_PROTOS_DIR", os.path.join(HERE, "prototypes"))
# MODE: "rolling" keeps only the most recent MAX_PROTOS, "first" keeps first N and then ignores new saves
MODE = os.getenv("PLANNER_PROTOS_MODE", "rolling").lower()
MAX_PROTOS = int(os.getenv("PLANNER_PROTOS_MAX", "20"))

# ensure folder exists
os.makedirs(PROTOS_DIR, exist_ok=True)

def _list_proto_files() -> List[str]:
    files = []
    try:
        for name in os.listdir(PROTOS_DIR):
            if name.lower().endswith(".json"):
                files.append(os.path.join(PROTOS_DIR, name))
    except FileNotFoundError:
        pass
    # sort by modified time ascending (oldest first)
    files.sort(key=lambda p: os.path.getmtime(p))
    return files

def count_prototypes() -> int:
    return len(_list_proto_files())

def list_prototypes(latest_first: bool = True, limit: Optional[int] = None) -> List[str]:
    files = _list_proto_files()
    files = list(reversed(files)) if latest_first else files
    if limit is not None:
        files = files[:limit]
    return files

def _atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # atomic on Windows/NTFS & POSIX

def _next_filename() -> str:
    # timestamp + counter to avoid collisions
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # count existing to add a suffix
    num = count_prototypes() + 1
    name = f"proto_{ts}_{num:04d}.json"
    return os.path.join(PROTOS_DIR, name)

def save_final(
    room: Dict[str, Any],
    items: List[Dict[str, Any]],
    placements: List[Dict[str, Any]],
    constraints: Dict[str, Any],
    penalties: Dict[str, Any],
) -> str:
    """
    Save a single finalized layout to Backend/prototypes as JSON.
    Honors MODE ('rolling' or 'first') and MAX_PROTOS cap.
    Returns the saved file path (or existing path if skipped in 'first' mode after cap).
    """
    # enforce "first" mode cap
    if MODE == "first" and count_prototypes() >= MAX_PROTOS:
        # do not save more; return most recent as a no-op indication
        files = list_prototypes(latest_first=True, limit=1)
        return files[0] if files else ""

    payload = {
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "room": room,
        "items": items,
        "placements": placements,
        "constraints": constraints,
        "penalties": penalties,
    }
    path = _next_filename()
    _atomic_write_json(path, payload)

    # enforce "rolling" cap (delete oldest beyond MAX_PROTOS)
    if MODE == "rolling":
        files = _list_proto_files()
        excess = len(files) - MAX_PROTOS
        for i in range(excess):
            try:
                os.remove(files[i])  # remove oldest first
            except FileNotFoundError:
                pass

    return path

def load_first_n(n: int) -> List[Dict[str, Any]]:
    """
    Load up to n most recent prototypes (latest first).
    Returns list of prototype dicts.
    """
    out: List[Dict[str, Any]] = []
    for path in list_prototypes(latest_first=True, limit=n):
        try:
            with open(path, "r", encoding="utf-8") as f:
                out.append(json.load(f))
        except Exception:
            # skip corrupted entries
            continue
    return out
