from typing import Dict, Callable, Any, Tuple

# Hard constraint function signature:
#   fn(model, V, room, rule) -> None
RegistryFn = Callable[[Any, dict, Any, Any], None]

# Soft constraint function signature:
#   fn(model, V, room, rule) -> (IntVar, str)
SoftFn = Callable[[Any, dict, Any, Any], Tuple[Any, str]]

HARD_REGISTRY: Dict[str, RegistryFn] = {}
SOFT_REGISTRY: Dict[str, SoftFn] = {}

def register_hard(name: str):
    def deco(fn: RegistryFn):
        HARD_REGISTRY[name] = fn
        return fn
    return deco

def register_soft(name: str):
    def deco(fn: SoftFn):
        SOFT_REGISTRY[name] = fn
        return fn
    return deco
