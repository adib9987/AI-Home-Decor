from ortools.sat.python import cp_model
from ..models import Item, Room

def build_variables(m: cp_model.CpModel, room: Room, items: list[Item]):
    V = {}
    for it in items:
        x = m.NewIntVar(0, room.width_cm, f"x_{it.id}")
        y = m.NewIntVar(0, room.height_cm, f"y_{it.id}")
        w = m.NewIntVar(it.min_w, it.max_w, f"w_{it.id}")
        h = m.NewIntVar(it.min_h, it.max_h, f"h_{it.id}")

        # rotation allowed -> binary var, else fixed 0
        r = m.NewIntVar(0, 1 if set(it.allow_rotate) != {0} else 0, f"r_{it.id}")

        # effective width/height after rotation
        ew = m.NewIntVar(0, max(it.max_w, it.max_h), f"ew_{it.id}")
        eh = m.NewIntVar(0, max(it.max_w, it.max_h), f"eh_{it.id}")

        # if not rotated: ew=w, eh=h ; if rotated: ew=h, eh=w
        m.Add(ew == w).OnlyEnforceIf(r.Not())
        m.Add(eh == h).OnlyEnforceIf(r.Not())
        m.Add(ew == h).OnlyEnforceIf(r)
        m.Add(eh == w).OnlyEnforceIf(r)

        V[it.id] = {"x": x, "y": y, "w": w, "h": h, "r": r, "ew": ew, "eh": eh}
    return V
