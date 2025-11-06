from __future__ import annotations
from constraints.registry import register_hard


@register_hard("stay_inside_room")
def stay_inside_room(m, V, room, rule):
for sid, v in V.items():
m.Add(v["x"] >= 0)
m.Add(v["y"] >= 0)
m.Add(v["x"] + v["ew"] <= room.width_cm)
m.Add(v["y"] + v["eh"] <= room.height_cm)