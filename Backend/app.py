from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

# ---- project imports ----
from .models import SolveRequest, Room, Item, Constraints
from .solver.heuristics import (
    compute_penalties,
    ensure_separated,
    snap_to_walls,
)
from .weights import (
    load_weights, save_weights, personalize, features_from_context, update_from_penalties
)
from .memory import load_first_n, save_final, count_prototypes
from .layout_gnn import load_gnn, retrain_gnn_from, gnn_status

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="AI Room Planner Backend",
              version="4.4.0 (GNN + safety fallback packing)")

# CORS (adjust your frontend origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def log_event(event: Dict[str, Any]) -> None:
    event.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print(f"[event] {event}")

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "room-planner-backend", "mode": "gnn-first"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/routes")
def list_routes():
    """Quick route lister to debug 404s."""
    return [
        {"path": r.path, "methods": sorted(list(r.methods or [])), "name": r.name}
        for r in app.routes
        if isinstance(r, APIRoute)
    ]

# -----------------------------------------------------------------------------
# Layout solving (GNN-first, sample-and-rank by penalties)
# -----------------------------------------------------------------------------
@app.post("/solve-layout")
def solve_layout(
    req: SolveRequest,
    samples: int = Query(12, ge=1, le=32),
    hard_no_overlap: bool = Query(True, description="Kept for compatibility; geometry is always enforced by ensure_separated"),
):
    """
    Predict placements with a GNN, generate several stochastic samples (MC Dropout),
    fix geometry robustly, then score by penalties (weighted) and pick the best.
    """
    # 1) personalize weights (used only for scoring penalties)
    base_w = load_weights()
    ctx = features_from_context(req.room.model_dump(), [it.model_dump() for it in req.items])
    used_w = personalize(base_w, ctx)

    # 2) model
    model = load_gnn()

    # 3) MC-Dropout sampling, pick best candidate by penalties
    best = None
    best_score = float("inf")

    for _ in range(samples):
        cand = model.predict(req.room, req.items, samples=1)

        # tiny wall nudge based on soft constraints (optional)
        snap_to_walls(cand, req.room, req.constraints, max_snap=25)

        # guarantee usable geometry regardless of model freshness
        ensure_separated(cand, req.room, constraints=req.constraints, margin=8)

        pens = compute_penalties(req.room, cand, req.constraints)
        score = sum(used_w.get(k, 0.0) * v for k, v in pens.items())
        if score < best_score:
            best_score, best = score, (cand, pens)

    placements, pens = best
    log_event({
        "event": "solve",
        "method": "gnn",
        "samples": samples,
        "score": best_score,
        "penalties": pens
    })
    return {
        "status": "ok",
        "method": "gnn",
        "placements": placements,
        "score": best_score,
        "penalties": pens,
        "weights_used": used_w,
        "samples": samples,
        "hard_no_overlap": True,  # always enforced via ensure_separated
    }

# -----------------------------------------------------------------------------
# Finalize (save prototype + quick online update)
# -----------------------------------------------------------------------------
def _finalize_impl(payload: Dict[str, Any]) -> Dict[str, Any]:
    accepted = bool(payload.get("accepted", False))
    if not accepted:
        return {"ok": True, "accepted": False}

    room = Room(**payload["room"])
    items = [Item(**obj) for obj in payload["items"]]
    cons = Constraints(**payload["constraints"])
    placements: List[Dict[str, Any]] = [dict(p) for p in payload["placements"]]

    # Robust geometry fix before accepting/saving
    ensure_separated(placements, room, constraints=cons, margin=8)

    pens = compute_penalties(room, placements, cons)

    # Save prototype (rolling up to 20; see Backend/memory.py)
    saved_path = save_final(
        room.model_dump(),
        [it.model_dump() for it in items],
        placements,
        cons.model_dump(),
        pens,
    )
    proto_count = count_prototypes()

    # Update penalty weights (optional shaping)
    w = load_weights()
    new_w = update_from_penalties(w, pens, eta=0.25)
    save_weights(new_w)

    # Quick online GNN update from this single example
    try:
        gnn = load_gnn()
        gnn.update_one(
            room.model_dump(),
            [it.model_dump() for it in items],
            placements,
            constraints=cons.model_dump(),
            epochs=18,
            lr=1e-3,
        )
        gnn_updated = True
    except Exception as e:
        log_event({"event": "gnn_update_error", "error": str(e)})
        gnn_updated = False

    log_event({
        "event": "finalize",
        "accepted": True,
        "gnn_updated": gnn_updated,
        "penalties": pens,
        "saved_path": saved_path,
        "prototype_count": proto_count
    })
    return {
        "ok": True,
        "accepted": True,
        "weights": new_w,
        "penalties": pens,
        "gnn_updated": gnn_updated,
        "saved_path": saved_path,
        "prototype_count": proto_count
    }

# Primary path (frontend should POST here)
@app.post("/finalize")
def finalize(payload: Dict[str, Any]):
    return _finalize_impl(payload)

# Alias path in case your frontend prefixes /api
@app.post("/api/finalize")
def finalize_alias(payload: Dict[str, Any]):
    return _finalize_impl(payload)

# -----------------------------------------------------------------------------
# Admin endpoints
# -----------------------------------------------------------------------------
@app.post("/admin/retrain-gnn")
def retrain_gnn(epochs: int = 80):
    """
    Offline retrain on saved prototypes + any internal history the model keeps.
    Increase epochs to learn more deeply from your examples.
    """
    protos = load_first_n(5)
    retrain_gnn_from(protos, epochs=epochs)
    return {"ok": True, "retrained": True, "epochs": epochs}

@app.get("/admin/gnn-status")
def admin_gnn_status():
    """
    Reports whether a model exists, vocab size, and how many history lines exist.
    Use this to verify that saving + training actually produce a model file.
    """
    return gnn_status()

@app.post("/admin/clear-gnn")
def clear_gnn():
    """
    Remove GNN model, vocab, and dataset files. Starts from scratch afterwards.
    """
    from .layout_gnn import GNN_MODEL_PATH, GNN_VOCAB_PATH, GNN_DATASET_PATH
    for p in [GNN_MODEL_PATH, GNN_VOCAB_PATH, GNN_DATASET_PATH]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    return {"ok": True, "cleared": "gnn_model_vocab_history"}
