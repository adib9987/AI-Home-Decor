from __future__ import annotations
from constraints.registry import register_hard


@register_hard("no_overlap")
def no_overlap(m, V, room, rule):
ids = list(V.keys())
for i in range(len(ids)):
for j in range(i+1, len(ids)):
a,b = ids[i], ids[j]
left = m.NewBoolVar(f"{a}_left_{b}")
right = m.NewBoolVar(f"{a}_right_{b}")
above = m.NewBoolVar(f"{a}_above_{b}")
below = m.NewBoolVar(f"{a}_below_{b}")
m.AddBoolOr([left,right,above,below])
m.Add(V[a]["x"] + V[a]["ew"] <= V[b]["x"]).OnlyEnforceIf(left)
m.Add(V[b]["x"] + V[b]["ew"] <= V[a]["x"]).OnlyEnforceIf(right)
m.Add(V[a]["y"] + V[a]["eh"] <= V[b]["y"]).OnlyEnforceIf(above)
m.Add(V[b]["y"] + V[b]["eh"] <= V[a]["y"]).OnlyEnforceIf(below)