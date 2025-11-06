# Local stub: no OpenAI. We just return default constraints and a warning.
from typing import Dict, Any, List, Tuple
from ..models import Constraints

def nlp_to_constraints(text: str, room: Dict[str, Any], items: List[Dict[str, Any]]
) -> Tuple[Constraints, Dict[str, Any]]:
    return Constraints(), {"warnings": ["NLP disabled (no OpenAI). Using default constraints only."]}
