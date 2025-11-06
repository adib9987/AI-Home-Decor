from __future__ import annotations
import os, json, math, random
from typing import Any, Dict, List, Tuple, Optional

# ---- safe torch import ----
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None

_THIS_DIR = os.path.dirname(__file__)
MODEL_PATH   = os.getenv("PLANNER_MODEL_PATH",   os.path.join(_THIS_DIR, "layout_model.pt"))
VOCAB_PATH   = os.getenv("PLANNER_VOCAB_PATH",   os.path.join(_THIS_DIR, "type_vocab.json"))
DATASET_PATH = os.getenv("PLANNER_DATASET_PATH", os.path.join(_THIS_DIR, "finals_history.jsonl"))

# ---------------- Vocab helpers ----------------
def _load_vocab() -> Dict[str, int]:
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # reserve 0 for <UNK>
    return {"<UNK>": 0}

def _save_vocab(v: Dict[str, int]) -> None:
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(v, f)

def _get_or_add_type_for_training(t: Optional[str], vocab: Dict[str, int]) -> Tuple[int, Dict[str, int], bool]:
    """During TRAINING we allow adding new types; returns (index, vocab, grew)."""
    t = t or "<UNK>"
    grew = False
    if t not in vocab:
        vocab[t] = max(vocab.values()) + 1 if vocab else 1
        grew = True
    return vocab[t], vocab, grew

def _lookup_type_for_predict(t: Optional[str], vocab: Dict[str, int]) -> int:
    """During PREDICT we DO NOT grow vocab; unknown -> <UNK>(0)."""
    t = t or "<UNK>"
    return vocab.get(t, 0)

# ---------------- Model ----------------
class ItemNet(nn.Module):
    """
    Predicts (cx, cy, w, h, rot_prob) in [0,1] for ONE item using:
      room (2), bounds (4), allow_rotate (1), type_embedding (E)
    """
    def __init__(self, type_vocab_size: int, type_emb_dim: int = 16, hidden: int = 128):
        super().__init__()
        self.type_emb_dim = type_emb_dim
        self.emb = nn.Embedding(num_embeddings=max(1, type_vocab_size), embedding_dim=type_emb_dim)  # 0..V-1
        in_dim = 2 + 4 + 1 + type_emb_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.head_xy = nn.Linear(hidden, 2)
        self.head_wh = nn.Linear(hidden, 2)
        self.head_rot = nn.Linear(hidden, 1)

    def forward(self, feats: torch.Tensor, type_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e = self.emb(type_idx.clamp(min=0))  # (N, E)
        x = torch.cat([feats, e], dim=-1)
        h = self.fc(x)
        xy  = torch.sigmoid(self.head_xy(h))
        wh  = torch.sigmoid(self.head_wh(h))
        rot = torch.sigmoid(self.head_rot(h))
        return xy, wh, rot

    def grow_embedding(self, new_vocab_size: int) -> None:
        """Resize embedding if vocab grew; keep old weights."""
        old = self.emb
        old_vocab, emb_dim = old.num_embeddings, old.embedding_dim
        if new_vocab_size <= old_vocab:
            return
        new_emb = nn.Embedding(new_vocab_size, emb_dim)
        with torch.no_grad():
            new_emb.weight[:old_vocab].copy_(old.weight)
            nn.init.normal_(new_emb.weight[old_vocab:], mean=0.0, std=0.02)
        self.emb = new_emb


# ---------------- Utils ----------------
def _ensure_torch():
    if torch is None:
        raise RuntimeError("PyTorch not installed. Run: pip install torch --index-url https://download.pytorch.org/whl/cpu")

def _room_norm(w: int, h: int) -> Tuple[float, float, float]:
    scale = float(max(w, h, 1))
    return w/scale, h/scale, scale

def _bounds_norm(x: int, scale: float) -> float:
    return max(0.0, min(1.0, x/scale))

def _pack_item_feat(room: Dict[str,int], it: Any) -> Tuple[List[float], int]:
    rw, rh, scale = _room_norm(room["width_cm"], room["height_cm"])
    f = [
        rw, rh,
        _bounds_norm(it.min_w, scale), _bounds_norm(it.max_w, scale),
        _bounds_norm(it.min_h, scale), _bounds_norm(it.max_h, scale),
        1.0 if (90 in getattr(it, "allow_rotate", []) or 90 in getattr(it, "allow_rotate", [0])) else 0.0,
    ]
    return f, int(scale)

def _pack_target(room: Dict[str,int], p: Dict[str,int], scale: float) -> List[float]:
    cx = (p["x"] + p["w"]/2.0) / scale
    cy = (p["y"] + p["h"]/2.0) / scale
    ww = p["w"] / scale
    hh = p["h"] / scale
    rot = 1.0 if (p.get("rotation",0) % 180) == 90 else 0.0
    return [cx, cy, ww, hh, rot]

def _align_by_id(items: List[Any], placements: List[Dict[str,Any]]) -> List[Tuple[Any, Dict[str,Any]]]:
    id2p = {p["id"]: p for p in placements}
    pairs: List[Tuple[Any, Dict[str,Any]]] = []
    for it in items:
        p = id2p.get(it.id)
        if p: pairs.append((it, p))
    return pairs


# ---------------- Wrapper ----------------
class LayoutModel:
    def __init__(self):
        _ensure_torch()
        self.device = torch.device("cpu")
        self.vocab = _load_vocab()  # includes <UNK>:0
        vocab_size = max(1, max(self.vocab.values()) + 1)
        self.net = ItemNet(type_vocab_size=vocab_size).to(self.device)
        if os.path.exists(MODEL_PATH):
            try:
                self.net.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            except Exception:
                pass
        self.net.train()

    def save(self):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(self.net.state_dict(), MODEL_PATH)
        _save_vocab(self.vocab)

    # -------- training over a dataset (prototypes/history) --------
    def train_dataset(self, dataset: List[Dict[str,Any]], epochs: int = 30, lr: float = 1e-3, batch_size: int = 64):
        if not dataset:
            return

        # Build batches + allow vocab to GROW here
        items_feats, items_type_idx, targets_xy, targets_wh, targets_rot = [], [], [], [], []
        grew_any = False

        for rec in dataset:
            room = rec["room"]
            items = rec["items"]
            placements = rec["placements"]

            # items are dicts in stored dataset
            class DItem:
                def __init__(self, d):
                    self.id = d["id"]; self.type = d.get("type")
                    self.min_w = d.get("min_w", d.get("w", 60))
                    self.max_w = d.get("max_w", d.get("w", 120))
                    self.min_h = d.get("min_h", d.get("h", 40))
                    self.max_h = d.get("max_h", d.get("h", 120))
                    self.allow_rotate = d.get("allow_rotate", [0,90])

            ditems = [DItem(d) for d in items]
            pairs = _align_by_id(ditems, placements)
            _, _, scale = _room_norm(room["width_cm"], room["height_cm"])

            for it, p in pairs:
                f, _ = _pack_item_feat(room, it)
                t = _pack_target(room, p, scale)
                idx, self.vocab, grew = _get_or_add_type_for_training(getattr(it, "type", None), self.vocab)
                grew_any |= grew

                items_feats.append(f)
                items_type_idx.append(idx)
                targets_xy.append(t[:2])
                targets_wh.append(t[2:4])
                targets_rot.append([t[4]])

        if not items_feats:
            return

        # If vocab grew, resize embedding BEFORE constructing tensors/optimizer
        if grew_any:
            new_vocab_size = max(1, max(self.vocab.values()) + 1)
            self.net.grow_embedding(new_vocab_size)

        X   = torch.tensor(items_feats,   dtype=torch.float32, device=self.device)
        Txy = torch.tensor(targets_xy,    dtype=torch.float32, device=self.device)
        Twh = torch.tensor(targets_wh,    dtype=torch.float32, device=self.device)
        Tro = torch.tensor(targets_rot,   dtype=torch.float32, device=self.device)
        Tix = torch.tensor(items_type_idx,dtype=torch.long,    device=self.device)

        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        n = X.shape[0]
        for ep in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                xy, wh, ro = self.net(X[idx], Tix[idx])
                loss = F.mse_loss(xy, Txy[idx]) + F.mse_loss(wh, Twh[idx]) + F.binary_cross_entropy(ro, Tro[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            # simple LR decay
            if (ep+1) % 10 == 0:
                for g in opt.param_groups:
                    g["lr"] *= 0.5

        self.save()

    def update_one(self, room: Dict[str,Any], items: List[Dict[str,Any]], placements: List[Dict[str,Any]], epochs: int = 10, lr: float = 1e-3):
        rec = {"room": room, "items": items, "placements": placements}
        self.train_dataset([rec], epochs=epochs, lr=lr, batch_size=64)
        # append to history
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        with open(DATASET_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    @torch.no_grad()
    def predict(self, room: Any, items: List[Any]) -> List[Dict[str,int]]:
        """
        PREDICT does NOT grow vocab. Unknown types map to <UNK>=0.
        This avoids 'index out of range' during inference.
        """
        self.net.eval()
        room_d = room.model_dump() if hasattr(room, "model_dump") else dict(room)
        rw, rh, scale = _room_norm(room_d["width_cm"], room_d["height_cm"])

        feats, type_idx, meta = [], [], []
        for it in items:
            f, _ = _pack_item_feat(room_d, it)
            idx = _lookup_type_for_predict(getattr(it, "type", None), self.vocab)
            feats.append(f); type_idx.append(idx)
            meta.append({
                "id": it.id,
                "type": getattr(it, "type", None),
                "min_w": it.min_w, "max_w": it.max_w,
                "min_h": it.min_h, "max_h": it.max_h,
            })

        if not feats:
            return []

        X   = torch.tensor(feats,    dtype=torch.float32)
        Tix = torch.tensor(type_idx, dtype=torch.long)
        xy, wh, ro = self.net(X, Tix)
        xy = xy.numpy(); wh = wh.numpy(); ro = ro.numpy()

        out: List[Dict[str,int]] = []
        W = room_d["width_cm"]; H = room_d["height_cm"]
        for i, m in enumerate(meta):
            cx = max(0.0, min(1.0, xy[i,0])) * max(W, H)
            cy = max(0.0, min(1.0, xy[i,1])) * max(W, H)
            ww = max(0.0, min(1.0, wh[i,0])) * max(W, H)
            hh = max(0.0, min(1.0, wh[i,1])) * max(W, H)

            ww = max(m["min_w"], min(m["max_w"], int(round(ww))))
            hh = max(m["min_h"], min(m["max_h"], int(round(hh))))

            x = int(round(cx - ww/2.0))
            y = int(round(cy - hh/2.0))
            x = max(0, min(x, W - ww))
            y = max(0, min(y, H - hh))

            rot = 90 if ro[i,0] >= 0.5 else 0
            out.append({"id": m["id"], "type": m["type"], "x": x, "y": y, "w": ww, "h": hh, "rotation": rot})

        self.net.train()
        return out


# -------- public helpers --------
def load_model() -> LayoutModel:
    return LayoutModel()

def retrain_from_history_and_prototypes(prototypes: List[Dict[str,Any]], max_lines: int = 500):
    _ensure_torch()
    ds: List[Dict[str,Any]] = []
    for rec in prototypes:
        ds.append({"room": rec["room"], "items": rec["items"], "placements": rec["placements"]})

    if os.path.exists(DATASET_PATH):
        try:
            with open(DATASET_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()[-max_lines:]
            for ln in lines:
                try:
                    r = json.loads(ln)
                    ds.append({"room": r["room"], "items": r["items"], "placements": r["placements"]})
                except Exception:
                    pass
        except Exception:
            pass

    if not ds:
        return

    model = load_model()
    epochs = 40 if len(ds) <= 20 else 20
    model.train_dataset(ds, epochs=epochs, lr=1e-3, batch_size=64)
    model.save()
