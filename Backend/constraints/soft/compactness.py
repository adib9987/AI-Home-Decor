from __future__ import annotations
from constraints.registry import register_soft


@register_soft("compactness")
def compactness(m, V, room, rule):
max_right = m.NewIntVar(0, room.width_cm, "max_right")
max_down = m.NewIntVar(0, room.height_cm, "max_down")
m.AddMaxEquality(max_right, [V[s]["x"] + V[s]["ew"] for s in V])
m.AddMaxEquality(max_down, [V[s]["y"] + V[s]["eh"] for s in V])
total = m.NewIntVar(0, room.width_cm + room.height_cm, "total_extent")
m.Add(total == max_right + max_down)
return total, "compactness"