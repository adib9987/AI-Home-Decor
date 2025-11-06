from __future__ import annotations
from constraints.registry import register_soft


@register_soft("tv_viewing_distance")
def tv_viewing_distance(m, V, room, rule):
a = rule.subject; b = rule.object
minc = rule.min_cm or 180
maxc = rule.max_cm or 350


cx_a = m.NewIntVar(0, room.width_cm, f"cx_{a}")
cy_a = m.NewIntVar(0, room.height_cm, f"cy_{a}")
cx_b = m.NewIntVar(0, room.width_cm, f"cx_{b}")
cy_b = m.NewIntVar(0, room.height_cm, f"cy_{b}")
m.Add(cx_a == V[a]["x"] + V[a]["ew"] // 2)
m.Add(cy_a == V[a]["y"] + V[a]["eh"] // 2)
m.Add(cx_b == V[b]["x"] + V[b]["ew"] // 2)
m.Add(cy_b == V[b]["y"] + V[b]["eh"] // 2)


dx = m.NewIntVar(-room.width_cm, room.width_cm, f"dx_{a}_{b}")
dy = m.NewIntVar(-room.height_cm, room.height_cm, f"dy_{a}_{b}")
m.Add(dx == cx_a - cx_b); m.Add(dy == cy_a - cy_b)
adx = m.NewIntVar(0, room.width_cm, f"adx_{a}_{b}")
ady = m.NewIntVar(0, room.height_cm, f"ady_{a}_{b}")
m.AddAbsEquality(adx, dx); m.AddAbsEquality(ady, dy)
l1 = m.NewIntVar(0, room.width_cm + room.height_cm, f"l1_{a}_{b}")
m.Add(l1 == adx + ady)


below = m.NewIntVar(0, room.width_cm + room.height_cm, f"below_{a}_{b}")
above = m.NewIntVar(0, room.width_cm + room.height_cm, f"above_{a}_{b}")
m.Add(below >= minc - l1); m.Add(below >= 0)
m.Add(above >= l1 - maxc); m.Add(above >= 0)


penalty = m.NewIntVar(0, 2*(room.width_cm + room.height_cm), f"tv_penalty_{a}_{b}")
m.Add(penalty == below + above)
return penalty, "tv_viewing_distance"