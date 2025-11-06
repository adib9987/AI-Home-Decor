import React from "react"
import type { Item } from "../types"

type Props = {
  catalog: Item[]
}

export default function Palette({ catalog }: Props) {
  return (
    <div style={{display:'flex', flexDirection:'column', gap:8}}>
      <h3 style={{margin:'8px 0'}}>Objects</h3>
      {catalog.map((it) => (
        <div key={it.id}
             draggable
             onDragStart={(e) => {
               e.dataTransfer.setData("application/json", JSON.stringify(it))
             }}
             style={{
               border:'1px solid #ddd',
               padding:8,
               borderRadius:6,
               cursor:'grab',
               display:'flex',
               justifyContent:'space-between',
               alignItems:'center',
               background:'#fff'
             }}>
          <div>
            <strong>{it.type}</strong>
            <div style={{fontSize:12, opacity:0.7}}>
              {it.min_w}–{it.max_w} × {it.min_h}–{it.max_h} cm
            </div>
          </div>
          <span style={{fontSize:12, opacity:0.6}}>drag</span>
        </div>
      ))}
    </div>
  )
}
