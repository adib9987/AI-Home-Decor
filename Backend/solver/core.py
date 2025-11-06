
from ortools.sat.python import cp_model
from ..models import Room, Item, Constraints
from .variables import build_variables
from ..constraints.registry import HARD_REGISTRY, SOFT_REGISTRY


def solve(room: Room, items: list[Item], constraints: Constraints, time_limit_s: float = 2.0):
    m = cp_model.CpModel()

    # 1) Variables
    V = build_variables(m, room, items)

    # 2) Hard constraints
    # Always apply core three first
    for name in ["stay_inside_room", "no_overlap", "keepouts_respected"]:
        fn = HARD_REGISTRY.get(name)
        if fn:
            fn(m, V, room, None)

    # Convert relations into hard rules (if provided)
    hard_rules = list(constraints.hard)
    for rel in constraints.relations:
        # Create a simple proxy carrying rel fields
        hard_rules.append(type("HardProxy", (), {**rel.model_dump(), "type": rel.type}))

    for rule in hard_rules:
        fn = HARD_REGISTRY.get(rule.type)
        if fn and rule.type not in {"stay_inside_room", "no_overlap", "keepouts_respected"}:
            fn(m, V, room, rule)

    # 3) Soft constraints → objective
    soft_terms = []
    labels = []
    for s in constraints.soft:
        fn = SOFT_REGISTRY.get(s.type)
        if not fn:
            continue
        expr, label = fn(m, V, room, s)
        w = int(max(0.0, s.weight) * 100)
        if w > 0:
            term = m.NewIntVar(0, 10**9, f"term_{label}")
            m.AddMultiplicationEquality(term, [expr, w])
            soft_terms.append(term)
            labels.append(label)

    if soft_terms:
        total = m.NewIntVar(0, 10**9, "objective_total")
        m.Add(total == sum(soft_terms))
        m.Minimize(total)

    # 4) Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    status = solver.Solve(m)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"placements": [], "score": 0.0, "status": "infeasible"}

    # 5) Extract placements
    out = []
    for sid, v in V.items():
        rot = solver.Value(v["r"])
        out.append({
            "id": sid,
            "x": solver.Value(v["x"]),
            "y": solver.Value(v["y"]),
            "w": solver.Value(v["ew"]),
            "h": solver.Value(v["eh"]),
            "rotation": 90 if rot == 1 else 0,
        })

    return {
        "placements": out,
        "score": 1.0,
        "status": "ok",
        "objective_breakdown": {l: 0 for l in labels},
    }
