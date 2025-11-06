from __future__ import annotations
from typing import List, Dict, Tuple
from math import exp
from random import random, seed
from .heuristics import random_initial, score_total, mutate
from ..models import Room, Item, Constraints

def sa_solve(room: Room, items: List[Item], cons: Constraints, weights: Dict[str,float],
             iters=2500, t0=2000.0, t_end=1.0, rng_seed: int | None = None,
             target_acc: float = 0.25
             ) -> Tuple[List[Dict], float]:
    """
    Simulated Annealing with adaptive temperature based on acceptance ratio.
    Keeps acceptance rate near target_acc (default 25%).
    Reheats if stuck; cools faster if too random.
    """
    if rng_seed is not None:
        seed(rng_seed)

    cur = random_initial(room, items)
    cur_score = score_total(room, cur, cons, weights)
    best, best_score = cur, cur_score

    accepted = 0
    window = 0

    for i in range(1, iters + 1):
        # Temperature schedule
        t = t0 * ((t_end / t0) ** (i / iters))

        # Create a neighbor
        nxt = mutate(room, items, cur)
        s = score_total(room, nxt, cons, weights)
        delta = s - cur_score

        # Acceptance rule
        if delta <= 0 or random() < exp(-delta / max(1.0, t)):
            cur, cur_score = nxt, s
            if s < best_score:
                best, best_score = nxt, s
            accepted += 1

        window += 1

        # Every 200 iterations, adapt the "temperature intensity" t0
        if window == 200:
            acc_ratio = accepted / window
            if acc_ratio < target_acc * 0.6:
                # too few acceptances → too cold → reheat slightly
                t0 *= 1.10
            elif acc_ratio > target_acc * 1.4:
                # too many acceptances → too hot → cool a bit
                t0 *= 0.95
            # reset counters
            accepted = 0
            window = 0

    return best, best_score
