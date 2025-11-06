from __future__ import annotations
from constraints.registry import register_hard


@register_hard("keepouts_respected")
def keepouts_respected(m, V, room, rule):
blocked = [*room.doors, *room.windows, *room.keepouts]
for sid, v in V.items():
for k, blk in enumerate(blocked):
left = m.NewBoolVar(f"{sid}_blk{k}_left")
right = m.NewBoolVar(f"{sid}_blk{k}_right")
above = m.NewBoolVar(f"{sid}_blk{k}_above")
below = m.NewBoolVar(f"{sid}_blk{k}_below")
m.AddBoolOr([left,right,above,below])
m.Add(v["x"] + v["ew"] <= blk.x).OnlyEnforceIf(left)
m.Add(blk.x + blk.w <= v["x"]).OnlyEnforceIf(right)
m.Add(v["y"] + v["eh"] <= blk.y).OnlyEnforceIf(above)
m.Add(blk.y + blk.h <= v["y"]).OnlyEnforceIf(below)