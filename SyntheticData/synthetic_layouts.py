import os
import json
import math
import time
import random
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

# ---------- Simple data classes (standalone; no project imports) ----------
@dataclass
class Room:
    width_cm: int
    height_cm: int

@dataclass
class ItemSpec:
    id: str
    type: str
    min_w: int
    max_w: int
    min_h: int
    max_h: int
    rotation: int = 0

# ---------- Geometry helpers ----------
def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def rect_overlap(a: Dict[str, int], b: Dict[str, int]) -> int:
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ox = max(0, min(ax2, bx2) - max(ax1, bx1))
    oy = max(0, min(ay2, by2) - max(ay1, by1))
    return ox * oy

def any_overlap(placements: List[Dict[str,int]]) -> bool:
    for i in range(len(placements)):
        for j in range(i+1, len(placements)):
            if rect_overlap(placements[i], placements[j]) > 0:
                return True
    return False

def enforce_no_overlap(placements: List[Dict[str,int]], room: Room, margin: int = 6, max_iters: int = 400) -> None:
    W, H = room.width_cm, room.height_cm
    def clamp_p(p):
        p["x"] = clamp(p["x"], 0, max(0, W - p["w"]))
        p["y"] = clamp(p["y"], 0, max(0, H - p["h"]))

    def separate(a, b):
        ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
        bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
        ox = min(ax2, bx2) - max(ax1, bx1)
        oy = min(ay2, by2) - max(ay1, by1)
        if ox <= 0 or oy <= 0:
            return False
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
        clamp_p(a); clamp_p(b)
        return True

    it = 0
    changed = True
    while changed and it < max_iters:
        changed = False
        for i in range(len(placements)):
            for j in range(i+1, len(placements)):
                if rect_overlap(placements[i], placements[j]) > 0:
                    if separate(placements[i], placements[j]):
                        changed = True
        it += 1

def pack_fallback(placements: List[Dict[str,int]], room: Room, h_gap: int = 20, v_gap: int = 20) -> None:
    """Left→right packing with wrap; simple, valid, non-overlapping."""
    W, H = room.width_cm, room.height_cm
    x, y, row_h = 0, 0, 0
    for p in placements:
        w, h = p["w"], p["h"]
        if x + w > W:
            x = 0
            y = y + row_h + v_gap
            row_h = 0
        if y + h > H:
            y = max(0, min(H - h, y))
        p["x"] = max(0, min(W - w, x))
        p["y"] = max(0, min(H - h, y))
        x = p["x"] + w + h_gap
        row_h = max(row_h, h)

# ---------- Penalties (align with your backend’s scoring) ----------
def compute_penalties(room: Room, placements: List[Dict[str,Any]], constraints: Dict[str,Any]) -> Dict[str,float]:
    W, H = room.width_cm, room.height_cm
    used = sum(p["w"]*p["h"] for p in placements)
    compactness = max(0.0, (W*H - used)/max(1, W*H)) * 1000.0

    # walkway: penalize centers too close
    walkway = 0.0
    centers = [(p["x"] + p["w"]/2.0, p["y"] + p["h"]/2.0) for p in placements]
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dx = abs(centers[i][0] - centers[j][0])
            dy = abs(centers[i][1] - centers[j][1])
            d = math.hypot(dx, dy)
            if d < 60:
                walkway += (60 - d)

    # tv view soft constraint
    id2p = {p["id"]: p for p in placements}
    tv_view = 0.0
    soft = constraints.get("soft", [])
    for s in soft:
        if s.get("type") == "tv_viewing_distance":
            subj = id2p.get(s.get("subject"))
            obj  = id2p.get(s.get("object"))
            if subj and obj:
                sx = subj["x"] + subj["w"]/2.0; sy = subj["y"] + subj["h"]/2.0
                ox = obj["x"] + obj["w"]/2.0;  oy = obj["y"] + obj["h"]/2.0
                d = math.hypot(sx-ox, sy-oy)
                dmin = float(s.get("min_cm", 180.0)); dmax = float(s.get("max_cm", 350.0))
                if d < dmin: tv_view += (dmin - d)
                if d > dmax: tv_view += (d - dmax)

    # overlap + simple "centerline" variance
    overlap_area = 0.0
    for i in range(len(placements)):
        for j in range(i+1, len(placements)):
            overlap_area += rect_overlap(placements[i], placements[j])
    overlap = overlap_area / max(1.0, W*H) * 1000.0

    ys = [p["y"] + p["h"]/2.0 for p in placements]
    var_y = 0.0
    if len(ys) > 2:
        m = sum(ys)/len(ys)
        var_y = sum((y - m)**2 for y in ys)/len(ys)
    centerline = max(0.0, 0.0005 - var_y/(H*H)) * 1000.0

    # prefer_wall soft
    def dist_to_walls(p):
        cx = p["x"] + p["w"]/2.0
        cy = p["y"] + p["h"]/2.0
        left = cx; right = W - cx; top = cy; bottom = H - cy
        return {"left": left, "right": right, "top": top, "bottom": bottom}

    prefer = 0.0
    for s in soft:
        if s.get("type") == "prefer_wall":
            obj = id2p.get(s.get("object"))
            side = s.get("side")
            if obj and side in {"left","right","top","bottom"}:
                d = dist_to_walls(obj)[side]
                scale = W if side in ("left","right") else H
                prefer += (d/max(1.0, scale))*200.0

    return {
        "compactness": float(compactness),
        "walkway": float(walkway),
        "tv_viewing_distance": float(tv_view),
        "overlap": float(overlap),
        "centerline": float(centerline),
        "prefer_wall": float(prefer),
    }

# ---------- Content presets ----------
def living_room_items() -> List[ItemSpec]:
    return [
        ItemSpec("sofa1", "sofa", 160, 200, 80, 100),
        ItemSpec("tv", "tv", 110, 140, 60, 90),
        ItemSpec("coffee", "table", 80, 120, 50, 70),
        ItemSpec("plant1", "plant", 40, 60, 40, 60),
        ItemSpec("cabinet1", "cabinet", 100, 140, 40, 60),
    ]

def kitchen_items() -> List[ItemSpec]:
    return [
        ItemSpec("island", "island", 120, 180, 60, 100),
        ItemSpec("fridge", "fridge", 70, 90, 70, 90),
        ItemSpec("stove", "stove", 70, 90, 70, 90),
        ItemSpec("sink", "sink", 80, 120, 60, 80),
        ItemSpec("dining", "table", 120, 180, 80, 120),
    ]

def make_room(preset: str) -> Room:
    if preset == "living":
        return Room(width_cm=random.randint(360, 520), height_cm=random.randint(280, 400))
    else:
        return Room(width_cm=random.randint(320, 460), height_cm=random.randint(260, 360))

def make_items(preset: str) -> List[ItemSpec]:
    return living_room_items() if preset == "living" else kitchen_items()

def make_constraints(preset: str) -> Dict[str,Any]:
    soft = []
    hard = [{"type": "no_overlap"}]
    if preset == "living":
        soft.append({"type":"tv_viewing_distance", "subject":"sofa1", "object":"tv", "min_cm":180, "max_cm":350})
        soft.append({"type":"prefer_wall", "object":"tv", "side": random.choice(["top","bottom"])})
        soft.append({"type":"prefer_wall", "object":"cabinet1", "side": random.choice(["left","right"])})
    else:
        soft.append({"type":"prefer_wall", "object":"fridge", "side": random.choice(["left","right"])})
        soft.append({"type":"prefer_wall", "object":"stove", "side": random.choice(["top","bottom"])})
    return {"hard": hard, "soft": soft}

# ---------- Layout synthesis ----------
def random_size(spec: ItemSpec) -> Tuple[int,int]:
    w = random.randint(spec.min_w, spec.max_w)
    h = random.randint(spec.min_h, spec.max_h)
    return w, h

def jitter(p: Dict[str,int], room: Room, max_shift: int = 30) -> None:
    p["x"] = clamp(p["x"] + random.randint(-max_shift, max_shift), 0, room.width_cm - p["w"])
    p["y"] = clamp(p["y"] + random.randint(-max_shift, max_shift), 0, room.height_cm - p["h"])

def synthesize_layout(preset: str) -> Dict[str,Any]:
    room = make_room(preset)
    specs = make_items(preset)
    constraints = make_constraints(preset)

    # create items + initial sizes
    items = []
    placements = []
    for spec in specs:
        w, h = random_size(spec)
        items.append({
            "id": spec.id,
            "type": spec.type,
            "min_w": spec.min_w, "max_w": spec.max_w,
            "min_h": spec.min_h, "max_h": spec.max_h,
            "rotation": 0,
        })
        placements.append({"id": spec.id, "x": 0, "y": 0, "w": w, "h": h, "rotation": 0})

    # initial pack + jitter + separate
    pack_fallback(placements, room, h_gap=25, v_gap=25)
    for p in placements:
        jitter(p, room, max_shift=20)
    enforce_no_overlap(placements, room, margin=8, max_iters=300)

    # optional: small heuristic improvements (nudges to walls for 'prefer_wall')
    W, H = room.width_cm, room.height_cm
    for s in constraints.get("soft", []):
        if s.get("type") == "prefer_wall":
            obj = s.get("object"); side = s.get("side")
            for p in placements:
                if p["id"] == obj:
                    if side == "left":   p["x"] = max(0, p["x"] - 20)
                    if side == "right":  p["x"] = min(W - p["w"], p["x"] + 20)
                    if side == "top":    p["y"] = max(0, p["y"] - 20)
                    if side == "bottom": p["y"] = min(H - p["h"], p["y"] + 20)
    enforce_no_overlap(placements, room, margin=8, max_iters=200)

    penalties = compute_penalties(room, placements, constraints)

    proto = {
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "room": asdict(room),
        "items": items,
        "placements": placements,
        "constraints": constraints,
        "penalties": penalties,
    }
    return proto

# ---------- IO ----------
def write_jsonl(records: List[Dict[str,Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_prototypes(records: List[Dict[str,Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i, r in enumerate(records, start=1):
        name = f"proto_synth_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_{i:04d}.json"
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of synthetic layouts to generate")
    ap.add_argument("--preset", choices=["living","kitchen"], default="living")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_jsonl", type=str, default="Backend/data/gnn_synth.jsonl")
    ap.add_argument("--write_prototypes", action="store_true", help="also write into Backend/prototypes/")
    ap.add_argument("--protos_dir", type=str, default="Backend/prototypes")
    args = ap.parse_args()

    random.seed(args.seed)

    print(f"[synth] generating {args.n} layouts preset={args.preset}")
    recs = [synthesize_layout(args.preset) for _ in range(args.n)]
    write_jsonl(recs, args.out_jsonl)
    print(f"[synth] wrote JSONL dataset to {args.out_jsonl} ({len(recs)} rows)")

    if args.write_prototypes:
        write_prototypes(recs, args.protos_dir)
        print(f"[synth] wrote {len(recs)} prototype JSON files to {args.protos_dir}")

if __name__ == "__main__":
    main()