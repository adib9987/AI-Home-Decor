from __future__ import annotations
from constraints.registry import register_hard


@register_hard("near_wall")
def near_wall(m, V, room, rule):
sid = rule.subject
if rule.wall == "west": m.Add(V[sid]["x"] <= 5)
if rule.wall == "east": m.Add(V[sid]["x"] + V[sid]["ew"] >= room.width_cm - 5)
if rule.wall == "north":m.Add(V[sid]["y"] <= 5)
if rule.wall == "south":m.Add(V[sid]["y"] + V[sid]["eh"] >= room.height_cm - 5)


@register_hard("anchor_corner")
def anchor_corner(m, V, room, rule):
sid = rule.subject; D = rule.max_distance_cm or 80
if rule.corner == "NW": m.Add(V[sid]["x"] <= D); m.Add(V[sid]["y"] <= D)
if rule.corner == "NE": m.Add(room.width_cm - (V[sid]["x"]+V[sid]["ew"]) <= D); m.Add(V[sid]["y"] <= D)
if rule.corner == "SW": m.Add(V[sid]["x"] <= D); m.Add(room.height_cm - (V[sid]["y"]+V[sid]["eh"]) <= D)
if rule.corner == "SE": m.Add(room.width_cm - (V[sid]["x"]+V[sid]["ew"]) <= D); m.Add(room.height_cm - (V[sid]["y"]+V[sid]["eh"]) <= D)


@register_hard("faces")
def faces(m, V, room, rule):
a,b = rule.subject, rule.object
cy_a = m.NewIntVar(0, room.height_cm, f"cy_{a}")
cy_b = m.NewIntVar(0, room.height_cm, f"cy_{b}")
m.Add(cy_a == V[a]["y"] + V[a]["eh"] // 2)
m.Add(cy_b == V[b]["y"] + V[b]["eh"] // 2)
diff = m.NewIntVar(-room.height_cm, room.height_cm, f"diffy_{a}_{b}")
m.Add(diff == cy_a - cy_b)
absd = m.NewIntVar(0, room.height_cm, f"absdy_{a}_{b}")
m.AddAbsEquality(absd, diff)
m.Add(absd <= 40)


@register_hard("distance_between")
def distance_between(m, V, room, rule):
a,b = rule.a or rule.subject, rule.b or rule.object
minc = rule.min_cm or 0
maxc = rule.max_cm or (room.width_cm + room.height_cm)


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
m.Add(dx == cx_a - cx_b)
m.Add(dy == cy_a - cy_b)
adx = m.NewIntVar(0, room.width_cm, f"adx_{a}_{b}")
ady = m.NewIntVar(0, room.height_cm, f"ady_{a}_{b}")
m.AddAbsEquality(adx, dx)
m.AddAbsEquality(ady, dy)
l1 = m.NewIntVar(0, room.width_cm + room.height_cm, f"l1_{a}_{b}")
m.Add(l1 == adx + ady)
m.Add(l1 >= minc)
m.Add(l1 <= maxc)