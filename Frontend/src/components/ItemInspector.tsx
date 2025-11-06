import React from 'react'
import type { Item } from '../types'
export default function ItemInspector({items}:{items:Item[]}){
return (
<div>
<h3>Items</h3>
{items.map((it)=>(
<div key={it.id} style={{border:'1px solid #eee', padding:8, marginBottom:8}}>
<strong>{it.id}</strong> <em>{it.type}</em>
<div>W: {it.min_w}-{it.max_w} H: {it.min_h}-{it.max_h} rot: [{it.allow_rotate.join(',')}]</div>
</div>
))}
</div>
)
}