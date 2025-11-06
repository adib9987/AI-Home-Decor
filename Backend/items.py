from __future__ import annotations
from typing import List
from models import Item


def default_items() -> List[Item]:
return [
Item(id='sofa1', type='sofa', min_w=180, max_w=240, min_h=80, max_h=100, allow_rotate=[0,90], clearance_cm=10),
Item(id='tv1', type='tv', min_w=120, max_w=150, min_h=30, max_h=40, allow_rotate=[0,90], clearance_cm=0),
Item(id='table1',type='table',min_w=100, max_w=140, min_h=70, max_h=90, allow_rotate=[0], clearance_cm=10),
Item(id='plant1',type='plant',min_w=40, max_w=50, min_h=40, max_h=50, allow_rotate=[0], clearance_cm=0),
]