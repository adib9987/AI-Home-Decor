
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Rect(BaseModel):
    x: int
    y: int
    w: int
    h: int

class Room(BaseModel):
    width_cm: int
    height_cm: int
    doors: List[Rect] = []
    windows: List[Rect] = []
    keepouts: List[Rect] = []

class Item(BaseModel):
    id: str
    type: str
    min_w: int
    max_w: int
    min_h: int
    max_h: int
    allow_rotate: List[int] = Field(default_factory=lambda: [0, 90])
    clearance_cm: int = 0
    aspect_lock: bool = False

Wall = Literal["north", "east", "south", "west"]
Corner = Literal["NW", "NE", "SW", "SE"]

class Relation(BaseModel):
    type: str
    subject: Optional[str] = None
    object: Optional[str] = None
    a: Optional[str] = None
    b: Optional[str] = None
    wall: Optional[Wall] = None
    corner: Optional[Corner] = None
    min_cm: Optional[int] = None
    max_cm: Optional[int] = None
    max_distance_cm: Optional[int] = None
    axis: Optional[Literal["x", "y"]] = None
    notes: Optional[str] = None

class HardRule(BaseModel):
    type: Literal[
        "no_overlap",
        "stay_inside_room",
        "keepouts_respected",
        "keep_clearances",
        "near_wall",
        "faces",
        "distance_between",
        "anchor_corner",
    ]
    subject: Optional[str] = None
    object: Optional[str] = None
    a: Optional[str] = None
    b: Optional[str] = None
    wall: Optional[Wall] = None
    corner: Optional[Corner] = None
    min_cm: Optional[int] = None
    max_cm: Optional[int] = None
    max_distance_cm: Optional[int] = None

class SoftRule(BaseModel):
    type: Literal["compactness", "tv_viewing_distance", "walkway"]
    weight: float = 1.0
    min_cm: Optional[int] = None
    max_cm: Optional[int] = None
    target_cm: Optional[int] = None
    subject: Optional[str] = None
    object: Optional[str] = None

class Constraints(BaseModel):
    hard: List[HardRule] = Field(
        default_factory=lambda: [
            HardRule(type="stay_inside_room"),
            HardRule(type="no_overlap"),
            HardRule(type="keepouts_respected"),
        ]
    )
    soft: List[SoftRule] = Field(
        default_factory=lambda: [SoftRule(type="compactness", weight=0.5)]
    )
    relations: List[Relation] = Field(default_factory=list)

class SolveRequest(BaseModel):
    room: Room
    items: List[Item]
    constraints: Constraints
