from __future__ import annotations
from constraints.registry import register_soft


@register_soft("walkway")
def walkway(m, V, room, rule):
# Placeholder zero-penalty. Replace with real corridor/gap logic later.
zero = m.NewIntVar(0, 0, "walkway_zero")
return zero, "walkway"