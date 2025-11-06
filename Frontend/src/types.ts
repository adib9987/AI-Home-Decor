export type Rect = { x:number; y:number; w:number; h:number }
export type Room = { width_cm:number; height_cm:number; doors:Rect[]; windows:Rect[]; keepouts:Rect[] }


export type Item = {
id:string; type:string;
min_w:number; max_w:number; min_h:number; max_h:number;
allow_rotate:number[]; clearance_cm:number; aspect_lock?:boolean;
}


export type Relation = {
type:string; subject?:string; object?:string; a?:string; b?:string;
wall?:"north"|"east"|"south"|"west"; corner?:"NW"|"NE"|"SW"|"SE";
min_cm?:number; max_cm?:number; max_distance_cm?:number; axis?:"x"|"y"; notes?:string;
}


export type HardRule = {
type: "no_overlap"|"stay_inside_room"|"keepouts_respected"|"keep_clearances"|"near_wall"|"faces"|"distance_between"|"anchor_corner"
subject?:string; object?:string; a?:string; b?:string; wall?:"north"|"east"|"south"|"west"; corner?:"NW"|"NE"|"SW"|"SE"; min_cm?:number; max_cm?:number; max_distance_cm?:number;
}


export type SoftRule = {
type: "compactness"|"tv_viewing_distance"|"walkway";
weight:number; min_cm?:number; max_cm?:number; target_cm?:number; subject?:string; object?:string;
}


export type Constraints = { hard:HardRule[]; soft:SoftRule[]; relations:Relation[] }
export type Placement = { id:string; x:number; y:number; w:number; h:number; rotation:number }
export type SolveResponse = { placements:Placement[]; score:number; status:string; objective_breakdown?:Record<string,number> }