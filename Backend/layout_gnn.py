from __future__ import annotations
import os, json, math, random, time
from typing import Any, Dict, List, Tuple, Optional

# Safe torch import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch, nn, F = None, None, None

_THIS_DIR = os.path.dirname(__file__)
GNN_MODEL_PATH   = os.getenv("PLANNER_GNN_MODEL_PATH",   os.path.join(_THIS_DIR, "layout_gnn.pt"))
GNN_VOCAB_PATH   = os.getenv("PLANNER_GNN_VOCAB_PATH",   os.path.join(_THIS_DIR, "gnn_type_vocab.json"))
GNN_DATASET_PATH = os.getenv("PLANNER_GNN_DATASET_PATH", os.path.join(_THIS_DIR, "gnn_history.jsonl"))

# ---------- vocab (type -> index) ----------
def _load_vocab() -> Dict[str, int]:
    if os.path.exists(GNN_VOCAB_PATH):
        with open(GNN_VOCAB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"<UNK>": 0}

def _save_vocab(v: Dict[str,int]) -> None:
    with open(GNN_VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(v, f)

def _get_or_add_for_training(t: Optional[str], vocab: Dict[str,int]) -> Tuple[int, Dict[str,int], bool]:
    t = t or "<UNK>"
    grew = False
    if t not in vocab:
        vocab[t] = max(vocab.values()) + 1 if vocab else 1
        grew = True
    return vocab[t], vocab, grew

def _lookup_for_predict(t: Optional[str], vocab: Dict[str,int]) -> int:
    t = t or "<UNK>"
    return vocab.get(t, 0)

# ---------- utilities ----------
def _ensure_torch():
    if torch is None:
        raise RuntimeError("PyTorch not installed. Run: pip install torch --index-url https://download.pytorch.org/whl/cpu")

def _room_scale(w: int, h: int) -> float:
    return float(max(w, h, 1))

def _pack_item_feat(room: Dict[str,int], it: Any) -> Tuple[List[float], float]:
    W, H = room["width_cm"], room["height_cm"]
    S = _room_scale(W, H)
    allow_rot = 1.0 if (90 in getattr(it, "allow_rotate", [0])) else 0.0
    return [
        W / S, H / S,
        max(0, it.min_w) / S, max(0, it.max_w) / S,
        max(0, it.min_h) / S, max(0, it.max_h) / S,
        allow_rot
    ], S

def _target_from_place(room: Dict[str,int], p: Dict[str,int]) -> List[float]:
    S = _room_scale(room["width_cm"], room["height_cm"])
    cx = (p["x"] + p["w"]/2.0) / S
    cy = (p["y"] + p["h"]/2.0) / S
    ww = p["w"] / S
    hh = p["h"] / S
    rot = 1.0 if (p.get("rotation",0) % 180) == 90 else 0.0
    return [cx, cy, ww, hh, rot]

def _align_pairs(items: List[Any], placements: List[Dict[str,Any]]) -> List[Tuple[Any, Dict[str,Any]]]:
    id2p = {p["id"]: p for p in placements}
    pairs: List[Tuple[Any, Dict[str,Any]]] = []
    for it in items:
        p = id2p.get(it.id)
        if p: pairs.append((it,p))
    return pairs

# ---------- simple geometry helpers ----------
def _pair_overlap(ax, ay, aw, ah, bx, by, bw, bh) -> float:
    x_overlap = max(0.0, min(ax+aw, bx+bw) - max(ax, bx))
    y_overlap = max(0.0, min(ay+ah, by+bh) - max(ay, by))
    return x_overlap * y_overlap

# ---------- MLP with dropout ----------
class MLP(nn.Module):
    def __init__(self, inp: int, hid: int, out: int, pdrop: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(inp, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.fc3 = nn.Linear(hid, out)
        self.drop = nn.Dropout(p=pdrop)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        return self.fc3(x)

# ---------- Dense GNN ----------
class GraphMPN(nn.Module):
    """
    Dense message passing among N nodes (items).
    Predicts per-node (cx, cy, w, h, rot_prob) in [0,1].
    Dropout enables MC-sampling at inference to pick best candidate.
    """
    def __init__(self, type_vocab_size: int, type_emb_dim: int = 16, hidden: int = 160, steps: int = 3, pdrop: float = 0.15):
        super().__init__()
        self.emb = nn.Embedding(max(1, type_vocab_size), type_emb_dim)  # 0..V-1
        self.steps = steps
        node_in = 2 + 4 + 1 + type_emb_dim
        self.node_enc = MLP(node_in, hidden, hidden, pdrop)
        self.edge_mlp = MLP(2*hidden, hidden, hidden, pdrop)
        self.node_upd = MLP(2*hidden, hidden, hidden, pdrop)
        self.head_xy  = MLP(hidden, hidden, 2, pdrop)
        self.head_wh  = MLP(hidden, hidden, 2, pdrop)
        self.head_rot = MLP(hidden, hidden, 1, pdrop)
        self.dropout  = nn.Dropout(p=pdrop)

    def grow_embedding(self, new_vocab_size: int) -> None:
        old = self.emb
        oldV, D = old.num_embeddings, old.embedding_dim
        if new_vocab_size <= oldV: return
        new_emb = nn.Embedding(new_vocab_size, D)
        with torch.no_grad():
            new_emb.weight[:oldV].copy_(old.weight)
            nn.init.normal_(new_emb.weight[oldV:], mean=0.0, std=0.02)
        self.emb = new_emb

    def forward(self, feats: torch.Tensor, t_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e  = self.emb(t_idx.clamp(min=0))        # (N, E)
        x  = torch.cat([feats, e], dim=-1)       # (N, F+E)
        h  = self.node_enc(x)                    # (N, H)
        N, H = h.shape

        for _ in range(self.steps):
            hi = h.unsqueeze(1).expand(-1, N, -1)     # (N, N, H)
            hj = h.unsqueeze(0).expand(N, -1, -1)     # (N, N, H)
            pair = torch.cat([hi, hj], dim=-1)        # (N, N, 2H)
            m = self.edge_mlp(pair)                   # (N, N, H)
            m = m * (1 - torch.eye(N, device=m.device).unsqueeze(-1))  # zero self
            agg = m.mean(dim=1)                       # (N, H)
            h = self.node_upd(torch.cat([h, agg], dim=-1)) # (N, H)
            h = self.dropout(h)

        xy  = torch.sigmoid(self.head_xy(h))
        wh  = torch.sigmoid(self.head_wh(h))
        rot = torch.sigmoid(self.head_rot(h))
        return xy, wh, rot

# ---------- Loss builder (supervised + soft geometry) ----------
class LossBuilder:
    def __init__(self, lam_sup=1.0, lam_bound=0.2, lam_overlap=0.2, lam_tv=0.05):
        self.lam_sup    = lam_sup
        self.lam_bound  = lam_bound
        self.lam_overlap= lam_overlap
        self.lam_tv     = lam_tv

    def boundary_loss(self, pred_xy, pred_wh):
        # Punish (cx,cy,w,h) leaving [0,1] box in a soft way
        cx, cy = pred_xy[:,0], pred_xy[:,1]
        ww, hh = pred_wh[:,0], pred_wh[:,1]
        # derive box edges in normalized coords
        left   = cx - ww/2
        right  = cx + ww/2
        top    = cy - hh/2
        bottom = cy + hh/2
        # ReLU for out-of-bounds magnitudes
        return (
            F.relu(-left).mean() + F.relu(right - 1).mean() +
            F.relu(-top).mean()  + F.relu(bottom - 1).mean()
        )

    def overlap_loss(self, pred_xy, pred_wh):
        # Soft pair overlap in normalized coords (sum over pairs)
        N = pred_xy.shape[0]
        if N <= 1: return pred_xy.sum()*0
        loss = pred_xy.new_tensor(0.0)
        for i in range(N):
            cxi, cyi = pred_xy[i]; wi, hi = pred_wh[i]
            for j in range(i+1, N):
                cxj, cyj = pred_xy[j]; wj, hj = pred_wh[j]
                # convert center/size to (x,y,w,h) with left-top origin
                axi = cxi - wi/2; ayi = cyi - hi/2
                axj = cxj - wj/2; ayj = cyj - hj/2
                # approx overlap area in normalized square
                x_ov = torch.relu(torch.min(axi+wi, axj+wj) - torch.max(axi, axj))
                y_ov = torch.relu(torch.min(ayi+hi, ayj+hj) - torch.max(ayi, ayj))
                loss = loss + x_ov*y_ov
        return loss / max(1, N*(N-1)//2)

    def tv_distance_loss(self, pred_ids: List[str], pred_xy, pred_wh, cons: Dict[str,Any]):
        # If constraint has tv_viewing_distance(subject, object), encourage distance in bounds
        if not cons: return pred_xy.sum()*0
        subj, obj, minc, maxc = None, None, None, None
        for s in cons.get("soft", []):
            if s.get("type") == "tv_viewing_distance" and s.get("subject") and s.get("object"):
                subj = s["subject"]; obj = s["object"]
                minc = float(s.get("min_cm", 180)); maxc = float(s.get("max_cm", 350))
                break
        if not subj or not obj: return pred_xy.sum()*0
        if subj not in pred_ids or obj not in pred_ids: return pred_xy.sum()*0

        i = pred_ids.index(subj); j = pred_ids.index(obj)
        # Use L1 in normalized space; scale is unknown here but penalty shape helps
        d = torch.abs(pred_xy[i] - pred_xy[j]).sum()
        # Soft hinge: prefer d within [a,b] (approx)
        a = 0.15  # ~ 180 / 1200 if room max ~1200 cm (tunable)
        b = 0.30  # ~ 350 / 1200
        return F.relu(a - d) + F.relu(d - b)

    def total(self, pred_xy, pred_wh, pred_ro, Txy, Twh, Tro, ids: List[str], cons: Dict[str,Any]):
        sup = F.mse_loss(pred_xy, Txy) + F.mse_loss(pred_wh, Twh) + F.binary_cross_entropy(pred_ro, Tro)
        bnd = self.boundary_loss(pred_xy, pred_wh)
        ovl = self.overlap_loss(pred_xy, pred_wh)
        tvl = self.tv_distance_loss(ids, pred_xy, pred_wh, cons)
        return self.lam_sup*sup + self.lam_bound*bnd + self.lam_overlap*ovl + self.lam_tv*tvl

# ---------- wrapper ----------
class GNNLayoutModel:
    def __init__(self):
        _ensure_torch()
        self.device = torch.device("cpu")
        self.vocab = _load_vocab()  # includes <UNK>:0
        vocab_size = max(1, max(self.vocab.values()) + 1)
        self.net = GraphMPN(type_vocab_size=vocab_size).to(self.device)
        if os.path.exists(GNN_MODEL_PATH):
            try:
                self.net.load_state_dict(torch.load(GNN_MODEL_PATH, map_location=self.device))
            except Exception:
                pass
        self.net.train()

    def save(self):
        os.makedirs(os.path.dirname(GNN_MODEL_PATH), exist_ok=True)
        torch.save(self.net.state_dict(), GNN_MODEL_PATH)
        _save_vocab(self.vocab)

    def _build_graph(self, rec: Dict[str,Any]) -> Tuple[torch.Tensor,...] | None:
        room, items, places = rec["room"], rec["items"], rec["placements"]
        # Convert to minimal objects
        class DItem:
            def __init__(self, d):
                self.id = d["id"]; self.type = d.get("type")
                self.min_w = d.get("min_w", d.get("w", 60))
                self.max_w = d.get("max_w", d.get("w", 120))
                self.min_h = d.get("min_h", d.get("h", 40))
                self.max_h = d.get("max_h", d.get("h", 120))
                self.allow_rotate = d.get("allow_rotate", [0,90])
        ditems = [DItem(d) for d in items]
        pairs = _align_pairs(ditems, places)
        if not pairs: return None

        feats, tix, Txy, Twh, Tro, ids = [], [], [], [], [], []
        grew_any = False

        for it, p in pairs:
            f, _ = _pack_item_feat(room, it)
            t = _target_from_place(room, p)
            idx, self.vocab, grew = _get_or_add_for_training(getattr(it,"type",None), self.vocab)
            grew_any |= grew
            feats.append(f); tix.append(idx)
            Txy.append(t[:2]); Twh.append(t[2:4]); Tro.append([t[4]])
            ids.append(it.id)

        if grew_any:
            newV = max(1, max(self.vocab.values()) + 1)
            self.net.grow_embedding(newV)

        X   = torch.tensor(feats, dtype=torch.float32, device=self.device)
        Tix = torch.tensor(tix,   dtype=torch.long,    device=self.device)
        Txy = torch.tensor(Txy,   dtype=torch.float32, device=self.device)
        Twh = torch.tensor(Twh,   dtype=torch.float32, device=self.device)
        Tro = torch.tensor(Tro,   dtype=torch.float32, device=self.device)
        return X, Tix, Txy, Twh, Tro, ids

    # -------- dataset building with augmentation --------
    def _build_supervised(self, dataset: List[Dict[str,Any]], jitter: float = 0.02):
        graphs = []
        meta   = []
        for rec in dataset:
            pack = self._build_graph(rec)
            if pack is None: continue
            X, Tix, Txy, Twh, Tro, ids = pack
            # data augmentation: small jitters on targets
            if jitter > 0.0:
                noise_xy = (torch.rand_like(Txy) - 0.5) * 2 * jitter
                noise_wh = (torch.rand_like(Twh) - 0.5) * 2 * jitter
                Txy = (Txy + noise_xy).clamp(0.0, 1.0)
                Twh = (Twh + noise_wh).clamp(0.0, 1.0)
            graphs.append((X, Tix, Txy, Twh, Tro))
            meta.append({"ids": ids, "cons": rec.get("constraints")})
        return graphs, meta

    def train_dataset(self, dataset: List[Dict[str,Any]], epochs: int = 60, lr: float = 1e-3, batch_graphs: int = 12, jitter: float = 0.02):
        if not dataset: return
        graphs, meta = self._build_supervised(dataset, jitter=jitter)
        if not graphs: return

        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_builder = LossBuilder(lam_sup=1.0, lam_bound=0.25, lam_overlap=0.25, lam_tv=0.08)

        for ep in range(epochs):
            order = list(range(len(graphs)))
            random.shuffle(order)
            total = 0.0
            for i in range(0, len(order), batch_graphs):
                idxs = order[i:i+batch_graphs]
                # concatenate nodes across selected graphs into one big batch
                X = torch.cat([graphs[k][0] for k in idxs], dim=0)
                Tix = torch.cat([graphs[k][1] for k in idxs], dim=0)
                Txy = torch.cat([graphs[k][2] for k in idxs], dim=0)
                Twh = torch.cat([graphs[k][3] for k in idxs], dim=0)
                Tro = torch.cat([graphs[k][4] for k in idxs], dim=0)
                # forward
                xy, wh, ro = self.net(X, Tix)
                # compute loss — we pass only the first metadata of the batch (approx), or you could loop per-graph
                # for simplicity we ignore per-graph constraints here and use no tv loss in mini-batch (keeps it fast)
                loss = (
                    F.mse_loss(xy, Txy) + F.mse_loss(wh, Twh) + F.binary_cross_entropy(ro, Tro)
                    + 0.25*loss_builder.boundary_loss(xy, wh)
                    + 0.25*loss_builder.overlap_loss(xy, wh)
                )
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
            # mild LR decay
            if (ep+1) % 15 == 0:
                for g in opt.param_groups:
                    g["lr"] *= 0.7

        self.save()

    def update_one(self, room: Dict[str,Any], items: List[Dict[str,Any]], placements: List[Dict[str,Any]], constraints: Dict[str,Any] | None = None, epochs: int = 15, lr: float = 1e-3):
        rec = {"room": room, "items": items, "placements": placements, "constraints": constraints or {}}
        self.train_dataset([rec], epochs=epochs, lr=lr, batch_graphs=1, jitter=0.02)
        # append to history
        os.makedirs(os.path.dirname(GNN_DATASET_PATH), exist_ok=True)
        with open(GNN_DATASET_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    @torch.no_grad()
    def predict(self, room: Any, items: List[Any], samples: int = 8) -> List[Dict[str,int]]:
        """
        MC-Dropout sampling: produce several stochastic predictions and return ONE
        The caller will score and select the best; here we just return the first sample.
        (We keep this function simple; sampling loop is driven by the app for scoring.)
        """
        self.net.train()  # important: enable dropout for sampling
        room_d = room.model_dump() if hasattr(room,"model_dump") else dict(room)

        feats, tix, meta = [], [], []
        for it in items:
            f, _ = _pack_item_feat(room_d, it)
            feats.append(f)
            tix.append(_lookup_for_predict(getattr(it,"type",None), self.vocab))
            meta.append({
                "id": it.id, "type": getattr(it,"type",None),
                "min_w": it.min_w, "max_w": it.max_w,
                "min_h": it.min_h, "max_h": it.max_h
            })

        X   = torch.tensor(feats, dtype=torch.float32)
        Tix = torch.tensor(tix,   dtype=torch.long)

        # one sample
        xy, wh, ro = self.net(X, Tix)
        xy = xy.numpy(); wh = wh.numpy(); ro = ro.numpy()

        W, H = room_d["width_cm"], room_d["height_cm"]
        S = _room_scale(W,H)
        out: List[Dict[str,int]] = []
        for i,m in enumerate(meta):
            cx, cy = xy[i,0]*S, xy[i,1]*S
            ww, hh = wh[i,0]*S, wh[i,1]*S
            ww = max(m["min_w"], min(m["max_w"], int(round(ww))))
            hh = max(m["min_h"], min(m["max_h"], int(round(hh))))
            x = int(round(cx - ww/2.0)); y = int(round(cy - hh/2.0))
            x = max(0, min(x, W - ww));  y = max(0, min(y, H - hh))
            rot = 90 if ro[i,0] >= 0.5 else 0
            out.append({"id": m["id"], "type": m["type"], "x": x, "y": y, "w": ww, "h": hh, "rotation": rot})
        return out

# public helpers
def load_gnn() -> GNNLayoutModel:
    return GNNLayoutModel()

def retrain_gnn_from(prototypes: List[Dict[str,Any]], max_lines: int = 1200, epochs: int = 80):
    _ensure_torch()
    ds: List[Dict[str,Any]] = []
    for r in prototypes:
        ds.append({"room": r["room"], "items": r["items"], "placements": r["placements"], "constraints": r.get("constraints", {})})
    if os.path.exists(GNN_DATASET_PATH):
        try:
            with open(GNN_DATASET_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()[-max_lines:]
            for ln in lines:
                try:
                    rec = json.loads(ln)
                    ds.append(rec)
                except Exception:
                    pass
        except Exception:
            pass
    if not ds: return
    model = load_gnn()
    model.train_dataset(ds, epochs=epochs, lr=1e-3, batch_graphs=12, jitter=0.02)
    model.save()

def gnn_status() -> Dict[str,Any]:
    return {
        "model_exists": os.path.exists(GNN_MODEL_PATH),
        "model_mtime": os.path.getmtime(GNN_MODEL_PATH) if os.path.exists(GNN_MODEL_PATH) else None,
        "vocab_size": (max(_load_vocab().values()) + 1) if _load_vocab() else 1,
        "history_exists": os.path.exists(GNN_DATASET_PATH),
        "history_size_lines": sum(1 for _ in open(GNN_DATASET_PATH, "r", encoding="utf-8")) if os.path.exists(GNN_DATASET_PATH) else 0,
    }
