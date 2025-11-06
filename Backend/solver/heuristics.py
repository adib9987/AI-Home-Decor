# Backend/solver/heuristics.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import math
import statistics

# ------------- small utils -------------
def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def _room_size(room: Any) -> tuple[int, int]:
    """Supports Pydantic models (attrs) or plain dicts."""
    if hasattr(room, "width_cm") and hasattr(room, "height_cm"):
        return int(room.width_cm), int(room.height_cm)
    return int(room["width_cm"]), int(room["height_cm"])

def _as_dict_or_obj_get(s: Any, key: str, default: Any = None) -> Any:
    """Read a field from dict OR Pydantic object seamlessly."""
    if isinstance(s, dict):
        return s.get(key, default)
    return getattr(s, key, default)

# ------------- geometry helpers -------------
def rect_overlap(a: Dict[str, int], b: Dict[str, int]) -> int:
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ox = max(0, min(ax2, bx2) - max(ax1, bx1))
    oy = max(0, min(ay2, by2) - max(ay1, by1))
    return ox * oy

def any_overlap(placements: List[Dict[str, int]]) -> bool:
    n = len(placements)
    for i in range(n):
        for j in range(i + 1, n):
            if rect_overlap(placements[i], placements[j]) > 0:
                return True
    return False

def clamp_to_room(p: Dict[str, int], room: Any) -> None:
    W, H = _room_size(room)
    p["x"] = _clamp(p["x"], 0, max(0, W - p["w"]))
    p["y"] = _clamp(p["y"], 0, max(0, H - p["h"]))

# ---------- penalties helpers ----------
def _overlap_area_sum(placements: List[Dict[str,int]]) -> float:
    n = len(placements); s = 0.0
    for i in range(n):
        for j in range(i+1, n):
            s += rect_overlap(placements[i], placements[j])
    return s

def _centerline_penalty(placements: List[Dict[str,int]], room: Any) -> float:
    """Penalize low variance of y-centers (everything on one horizontal line)."""
    if len(placements) <= 2:
        return 0.0
    cy = [p["y"] + p["h"]/2.0 for p in placements]
    try:
        var = statistics.pvariance(cy)
    except Exception:
        return 0.0
    H = room.height_cm if hasattr(room, "height_cm") else room["height_cm"]
    if H <= 0:
        return 0.0
    norm_var = var / (H*H)  # 0..~0.25
    return float(max(0.0, 0.0005 - norm_var) * 1000.0)  # tune threshold/scale as needed

def _dist_to_walls(p: Dict[str,int], W: int, H: int) -> Tuple[float,float,float,float]:
    cx = p["x"] + p["w"]/2.0
    cy = p["y"] + p["h"]/2.0
    left = cx
    right = W - cx
    top = cy
    bottom = H - cy
    return left, right, top, bottom

def _prefer_wall_penalty(placements: List[Dict[str,int]], room: Any, constraints: Any) -> float:
    W, H = _room_size(room)
    id2p = {p["id"]: p for p in placements}
    if hasattr(constraints, "soft"):
        soft = constraints.soft or []
    elif isinstance(constraints, dict):
        soft = constraints.get("soft", []) or []
    else:
        soft = []
    pen = 0.0
    for s in soft:
        t = _as_dict_or_obj_get(s, "type")
        if t != "prefer_wall":
            continue
        obj = _as_dict_or_obj_get(s, "object")
        side = _as_dict_or_obj_get(s, "side")
        if obj not in id2p or side not in {"left","right","top","bottom"}:
            continue
        p = id2p[obj]
        left, right, top, bottom = _dist_to_walls(p, W, H)
        d = {"left": left, "right": right, "top": top, "bottom": bottom}[side]
        scale = (W if side in ("left","right") else H)
        pen += float(d / max(1.0, scale)) * 200.0  # scale factor
    return pen

def snap_to_walls(placements: List[Dict[str,int]], room: Any, constraints: Any, max_snap: int = 20) -> None:
    """If 'prefer_wall' soft rule exists, nudge toward that wall by up to max_snap cm."""
    W, H = _room_size(room)
    if hasattr(constraints, "soft"):
        soft = constraints.soft or []
    elif isinstance(constraints, dict):
        soft = constraints.get("soft", []) or []
    else:
        soft = []
    id2p = {p["id"]: p for p in placements}
    for s in soft:
        t = _as_dict_or_obj_get(s, "type")
        if t != "prefer_wall":
            continue
        obj = _as_dict_or_obj_get(s, "object")
        side = _as_dict_or_obj_get(s, "side")
        if obj not in id2p or side not in {"left","right","top","bottom"}:
            continue
        p = id2p[obj]
        if side == "left":
            p["x"] = max(0, p["x"] - max_snap)
        elif side == "right":
            p["x"] = min(W - p["w"], p["x"] + max_snap)
        elif side == "top":
            p["y"] = max(0, p["y"] - max_snap)
        elif side == "bottom":
            p["y"] = min(H - p["h"], p["y"] + max_snap)

# ------------- penalties (main) -------------
def compute_penalties(room: Any, placements: List[Dict[str, Any]], constraints: Any) -> Dict[str, float]:
    """
    Return a dict of penalty components; lower is better.
    Supports constraints as dicts or Pydantic models.
    """
    W, H = _room_size(room)

    # Compactness (encourages filling space)
    used = sum(int(p["w"]) * int(p["h"]) for p in placements)
    compactness = max(0.0, (W * H - used) / max(1, W * H)) * 1000.0

    # Walkway: penalize centers too close (<60cm)
    walkway = 0.0
    centers = [(p["x"] + p["w"] / 2.0, p["y"] + p["h"] / 2.0) for p in placements]
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(centers[i][0] - centers[j][0])
            dy = abs(centers[i][1] - centers[j][1])
            d = math.hypot(dx, dy)
            if d < 60:
                walkway += (60 - d)

    # TV viewing distance (soft)
    tv_view = 0.0
    if hasattr(constraints, "soft"):
        soft_list = constraints.soft or []
    elif isinstance(constraints, dict):
        soft_list = constraints.get("soft", []) or []
    else:
        soft_list = []

    id2p = {p["id"]: p for p in placements}
    for s in soft_list:
        stype = _as_dict_or_obj_get(s, "type")
        if stype == "tv_viewing_distance":
            subj_id = _as_dict_or_obj_get(s, "subject")
            obj_id  = _as_dict_or_obj_get(s, "object")
            if subj_id in id2p and obj_id in id2p:
                subj = id2p[subj_id]; obj = id2p[obj_id]
                sx = subj["x"] + subj["w"] / 2.0; sy = subj["y"] + subj["h"] / 2.0
                ox = obj["x"] + obj["w"] / 2.0;  oy = obj["y"] + obj["h"] / 2.0
                d = math.hypot(sx - ox, sy - oy)
                dmin = float(_as_dict_or_obj_get(s, "min_cm", 180.0) or 180.0)
                dmax = float(_as_dict_or_obj_get(s, "max_cm", 350.0) or 350.0)
                if d < dmin: tv_view += (dmin - d)
                if d > dmax: tv_view += (d - dmax)

    # Additional penalties
    overlap = _overlap_area_sum(placements) / max(1.0, W*H) * 1000.0
    centerline = _centerline_penalty(placements, room)
    wall_pref = _prefer_wall_penalty(placements, room, constraints)

    return {
        "compactness": float(compactness),
        "walkway": float(walkway),
        "tv_viewing_distance": float(tv_view),
        "overlap": float(overlap),
        "centerline": float(centerline),
        "prefer_wall": float(wall_pref),
    }

# ------------- HARD no-overlap enforcer -------------
def enforce_no_overlap(
    placements: List[Dict[str, int]],
    room: Any,
    margin: int = 6,
    max_iters: int = 400
) -> None:
    """
    Deterministically push overlapping rectangles apart until there is no overlap.
    Keeps everything inside the room. margin adds spacing.
    """
    W, H = _room_size(room)

    def _separate(a: Dict[str, int], b: Dict[str, int]) -> bool:
        ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
        bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
        ox = min(ax2, bx2) - max(ax1, bx1)
        oy = min(ay2, by2) - max(ay1, by1)
        if ox <= 0 or oy <= 0:
            return False  # no overlap

        # push along least penetration axis
        if ox < oy:
            shift = int(ox // 2 + margin // 2)
            if ax1 < bx1:
                a["x"] -= shift
                b["x"] += shift
            else:
                a["x"] += shift
                b["x"] -= shift
        else:
            shift = int(oy // 2 + margin // 2)
            if ay1 < by1:
                a["y"] -= shift
                b["y"] += shift
            else:
                a["y"] += shift
                b["y"] -= shift

        clamp_to_room(a, {"width_cm": W, "height_cm": H})
        clamp_to_room(b, {"width_cm": W, "height_cm": H})
        return True

    k = 0
    changed = True
    while changed and k < max_iters:
        changed = False
        for i in range(len(placements)):
            for j in range(i + 1, len(placements)):
                if rect_overlap(placements[i], placements[j]) > 0:
                    if _separate(placements[i], placements[j]):
                        changed = True
        k += 1

# ------------- de-stack fallback (fresh model helper) -------------
def spread_if_stacked(placements: List[Dict[str,int]], room: Any) -> None:
    """
    If most pairs overlap (fresh model piles everything at center),
    spread items along a line to give the enforcer a head start.
    """
    W, H = _room_size(room)
    if not placements: return
    n = len(placements)
    if n < 2: return

    overlaps = 0
    total_pairs = n*(n-1)//2
    for i in range(n):
        for j in range(i+1, n):
            if rect_overlap(placements[i], placements[j]) > 0:
                overlaps += 1
    if not total_pairs or overlaps/total_pairs < 0.6:
        return  # not badly stacked

    # spread along a horizontal line at mid-height
    y_mid = max(0, min(H - 1, H // 2))
    gap = max(40, W // (n + 1))
    x = gap // 2
    for p in placements:
        p["y"] = max(0, min(int(y_mid - p["h"] // 2), H - p["h"]))
        p["x"] = max(0, min(int(x), W - p["w"]))
        x += gap

# ------------- deterministic packing fallback -------------
def pack_fallback(placements: List[Dict[str,int]], room: Any, h_gap: int = 20, v_gap: int = 20) -> None:
    """
    Deterministic grid/flow packing inside the room:
    lays items left→right with h_gap, wraps to next row with v_gap.
    Mutates placements in place.
    """
    W, H = _room_size(room)
    x, y, row_h = 0, 0, 0
    for p in placements:
        w, h = p["w"], p["h"]
        if w <= 0 or h <= 0:
            continue
        # wrap to next row if needed
        if x + w > W:
            x = 0
            y = y + row_h + v_gap
            row_h = 0
        # clamp Y if we run out of room (best-effort)
        if y + h > H:
            y = max(0, min(H - h, y))
        p["x"] = max(0, min(W - w, x))
        p["y"] = max(0, min(H - h, y))
        x = p["x"] + w + h_gap
        row_h = max(row_h, h)

# ------------- one-call geometry fixer -------------
def repair(
    placements: List[Dict[str, int]],
    room: Any,
    constraints: Any | None = None
) -> None:
    """
    Keep items inside room; if constraints request no_overlap, enforce it.
    Works with dict or Pydantic constraints.
    """
    W, H = _room_size(room)
    for p in placements:
        clamp_to_room(p, {"width_cm": W, "height_cm": H})

    # If request contains hard no-overlap, enforce now.
    want_hard = False
    if constraints is not None:
        if hasattr(constraints, "hard"):
            hard_list = constraints.hard or []
        elif isinstance(constraints, dict):
            hard_list = constraints.get("hard", []) or []
        else:
            hard_list = []
        for h in hard_list:
            htype = _as_dict_or_obj_get(h, "type")
            if htype == "no_overlap":
                want_hard = True
                break
    if want_hard:
        enforce_no_overlap(placements, {"width_cm": W, "height_cm": H})

def ensure_separated(
    placements: List[Dict[str,int]],
    room: Any,
    constraints: Any | None = None,
    margin: int = 8
) -> None:
    """
    Repair → de-stack → enforce no-overlap; if still overlapping,
    apply pack_fallback then enforce again.
    """
    repair(placements, room, constraints=constraints)
    spread_if_stacked(placements, room)
    enforce_no_overlap(placements, room, margin=margin, max_iters=400)
    if any_overlap(placements):
        pack_fallback(placements, room)
        enforce_no_overlap(placements, room, margin=margin, max_iters=400)
