from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from random import random, randint, seed

from .heuristics import random_initial, score_total, mutate, repair, mimic_distance
from ..models import Room, Item, Constraints

def _tournament(pop, fit, k=3):
    idxs = [randint(0, len(pop) - 1) for _ in range(k)]
    return min(idxs, key=lambda i: fit[i])  # lower is better

def _noisy_copy(room: Room, items: List[Item], placements: List[Dict], copies: int = 6, noise_steps: int = 1) -> List[List[Dict]]:
    out: List[List[Dict]] = []
    for _ in range(copies):
        cur = [dict(p) for p in placements]
        for __ in range(noise_steps):  # tiny noise
            cur = mutate(room, items, cur)
        repair(cur, room)
        out.append(cur)
    return out

def ga_solve(
    room: Room,
    items: List[Item],
    cons: Constraints,
    weights: Dict[str, float],
    pop_size: int = 40,
    gens: int = 80,
    cx_rate: float = 0.30,
    mut_rate: float = 0.25,
    elitism: int = 8,
    rng_seed: Optional[int] = None,
    warm_starts: Optional[List[List[Dict]]] = None,
    refs_for_mimic: Optional[List[List[Dict]]] = None,
    lambda_mimic: float = 0.9,          # strength of mimic objective
    freeze_gens: int = 10,              # early gens: minimal change to stay close
) -> Tuple[List[Dict], float]:
    """
    GA with mimic objective:
    fitness = score_total(...) + lambda_mimic * mimic_distance(...)
    Early generations are "frozen" (low cx/mutation) to keep layouts similar to exemplars.
    """
    if rng_seed is not None:
        seed(rng_seed)

    # --- init population: ONLY from warm starts if provided ---
    pop: List[List[Dict]] = []
    if warm_starts and len(warm_starts) > 0:
        per_seed = max(4, pop_size // max(1, len(warm_starts)))
        for seed_sol in warm_starts:
            pop.extend(_noisy_copy(room, items, seed_sol, copies=per_seed, noise_steps=0))
    while len(pop) < pop_size:
        pop.append(random_initial(room, items))
    pop = pop[:pop_size]

    best_so_far = float("inf")
    stall = 0
    cur_mut_rate = float(mut_rate)
    cur_cx_rate  = float(cx_rate)

    def _fitness(indiv: List[Dict]) -> float:
        base = score_total(room, indiv, cons, weights)
        mimic = mimic_distance(room, items, indiv, refs_for_mimic or [])
        return base + lambda_mimic * mimic

    for g in range(gens):
        fit = [_fitness(s) for s in pop]

        # elitism
        elite_pairs = sorted(zip(pop, fit), key=lambda t: t[1])[:elitism]
        elites = [p for p, _ in elite_pairs]

        # selection
        parents = [pop[_tournament(pop, fit, k=3)] for __ in range(pop_size)]

        # "freeze" early generations by damping rates
        freeze_factor = 0.2 if g < freeze_gens else 1.0
        eff_cx = cur_cx_rate * freeze_factor
        eff_mut = cur_mut_rate * freeze_factor

        # crossover
        kids: List[List[Dict]] = []
        for i in range(0, pop_size, 2):
            a = [dict(p) for p in parents[i]]
            b = [dict(p) for p in parents[(i + 1) % pop_size]]
            if random() < eff_cx and len(a) == len(b) and len(a) >= 2:
                cut = randint(1, len(a) - 1)
                c1 = a[:cut] + b[cut:]
                c2 = b[:cut] + a[cut:]
            else:
                c1, c2 = a, b
            kids.extend([c1, c2])

        # mutation + repair
        next_pop: List[List[Dict]] = []
        for k in kids[: pop_size - elitism]:
            if random() < eff_mut:
                k = mutate(room, items, k)
            repair(k, room)
            next_pop.append(k)

        next_pop.extend(elites)
        pop = next_pop

        # adapt on stall (after freeze window)
        fit = [_fitness(s) for s in pop]
        gen_best = min(fit)
        if gen_best + 1e-6 < best_so_far:
            best_so_far = gen_best
            stall = 0
        else:
            stall += 1

        if g >= freeze_gens:
            if stall >= 6:
                cur_mut_rate = min(0.6, cur_mut_rate * 1.15)    # explore a bit more
                cur_cx_rate  = min(0.6, cur_cx_rate  * 1.10)
            else:
                cur_mut_rate = max(0.20, cur_mut_rate * 0.995)  # cool slowly
                cur_cx_rate  = max(0.20, cur_cx_rate  * 0.997)

    # final best (by mimic-aware fitness)
    fit = [_fitness(s) for s in pop]
    best_idx = min(range(len(pop)), key=lambda i: fit[i])
    best = pop[best_idx]
    return best, fit[best_idx]
