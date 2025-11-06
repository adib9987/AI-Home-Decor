import json
import os
from typing import Dict, Any

WEIGHTS_FILE = os.getenv("PLANNER_WEIGHTS_PATH", os.path.join(os.path.dirname(__file__), "weights.json"))

DEFAULT_WEIGHTS: Dict[str, float] = {
    "compactness": 0.5,
    "tv_viewing_distance": 2.0,
    "walkway": 3.0,
}

def load_weights() -> Dict[str, float]:
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {**DEFAULT_WEIGHTS, **{k: float(v) for k, v in data.items()}}
        except Exception:
            pass
    return DEFAULT_WEIGHTS.copy()

def save_weights(w: Dict[str, float]) -> None:
    try:
        with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(w, f)
    except Exception as e:
        print(f"[weights] Failed to save weights: {e}")

def features_from_context(room: Dict[str, Any], items: list[Dict[str, Any]]) -> Dict[str, float]:
    aspect = room.get("width_cm", 1) / max(1, room.get("height_cm", 1))
    n_doors = len(room.get("doors", []))
    n_seats = sum(1 for it in items if it.get("type") in {"sofa", "chair", "sectional"})
    return {"aspect": float(aspect), "n_doors": float(n_doors), "n_seats": float(n_seats)}

def personalize(weights: Dict[str, float], ctx: Dict[str, float]) -> Dict[str, float]:
    # small context tweaks; keep stable
    w = dict(weights)
    if ctx.get("aspect", 1.0) > 1.3:
        w["walkway"] = w.get("walkway", 3.0) * 1.05
    if ctx.get("n_doors", 0) >= 2:
        w["walkway"] = w.get("walkway", 3.0) * 1.08
    return w

# ------------------- New: update from penalties -------------------
def update_from_penalties(w: Dict[str,float], pens: Dict[str,float], eta: float = 0.20) -> Dict[str,float]:
    """
    Learning rule: move weights toward the user's accepted trade-off.
    - Normalize inverse penalties as a "preference signal": pref_k = 1/(pen_k+eps)
    - New weights = (1-eta)*w + eta * (C * pref_norm), where C preserves total magnitude.
    """
    eps = 1e-6
    pref = {k: 1.0 / (max(pens.get(k, 0.0), 0.0) + eps) for k in w.keys()}
    s = sum(pref.values()) or eps
    pref_norm = {k: v / s for k, v in pref.items()}

    # keep overall weight mass similar to current
    total_w = sum(max(0.0, v) for v in w.values())
    if total_w <= 0: total_w = sum(DEFAULT_WEIGHTS.values())

    target = {k: total_w * pref_norm.get(k, 0.0) for k in w.keys()}

    new_w = {}
    for k in w.keys():
        val = (1 - eta) * w[k] + eta * target[k]
        new_w[k] = float(max(0.05, min(10.0, val)))  # clamp to sane range
    return new_w
