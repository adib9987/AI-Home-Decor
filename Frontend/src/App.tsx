import React, { useEffect, useMemo, useState } from "react"
import RoomCanvas from "./components/RoomCanvas"
import Palette from "./components/Palette"
import { solveLayout, finalize } from "./api"
import type { Room, Item, Placement, Constraints } from "./types"

const DEFAULT_CONSTRAINTS: Constraints = {
  hard: [{ type: "stay_inside_room" }, { type: "no_overlap" }, { type: "keepouts_respected" }],
  soft: [
    { type: "compactness", weight: 0.6 },
    { type: "tv_viewing_distance", weight: 2.0, subject: "sofa1", object: "tv1", min_cm: 180, max_cm: 350 },
    { type: "walkway", weight: 3.0 }
  ],
  relations: []
}

export default function App() {
  const [room, setRoom] = useState<Room>({
    width_cm: 520, height_cm: 380,
    doors: [{ x: 0, y: 160, w: 90, h: 5 }],
    windows: [{ x: 300, y: 0, w: 120, h: 5 }],
    keepouts: []
  })

  const catalog: Item[] = useMemo(() => [
    { id: "tmpl-sofa", type: "sofa",  min_w: 180, max_w: 240, min_h: 80,  max_h: 100, allow_rotate: [0, 90], clearance_cm: 10 },
    { id: "tmpl-tv",   type: "tv",    min_w: 120, max_w: 150, min_h: 30,  max_h: 40,  allow_rotate: [0],     clearance_cm: 0  },
    { id: "tmpl-table",type: "table", min_w: 100, max_w: 140, min_h: 70,  max_h: 90,  allow_rotate: [0],     clearance_cm: 10 },
    { id: "tmpl-chair",type: "chair", min_w:  60, max_w:  80, min_h: 60,  max_h: 80,  allow_rotate: [0, 90], clearance_cm:  5 },
    { id: "tmpl-plant",type: "plant", min_w:  40, max_w:  50, min_h: 40,  max_h: 50,  allow_rotate: [0],     clearance_cm:  0 },
  ], [])

  const [items, setItems] = useState<Item[]>([
    { id: "sofa1",  type: "sofa",  min_w: 180, max_w: 240, min_h: 80,  max_h: 100, allow_rotate: [0, 90], clearance_cm: 10 },
    { id: "tv1",    type: "tv",    min_w: 120, max_w: 150, min_h: 30,  max_h: 40,  allow_rotate: [0],     clearance_cm: 0  },
    { id: "table1", type: "table", min_w: 100, max_w: 140, min_h: 70,  max_h: 90,  allow_rotate: [0],     clearance_cm: 10 },
  ])
  const [constraints] = useState<Constraints>(DEFAULT_CONSTRAINTS)
  const [placements, setPlacements] = useState<Placement[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const clamp = (v:number, lo:number, hi:number) => Math.max(lo, Math.min(hi, v))
  function nextId(prefix: string) {
    let n = 1
    const ids = new Set(items.map(i => i.id).concat(placements.map(p => p.id)))
    while (ids.has(`${prefix}${n}`)) n++
    return `${prefix}${n}`
  }

  // --- Drag-move -------------------------------------------------------------
  function movePlacement(id: string, x: number, y: number) {
    setPlacements(prev =>
      prev.map(p => p.id === id
        ? { ...p, x: clamp(x, 0, room.width_cm - p.w), y: clamp(y, 0, room.height_cm - p.h) }
        : p
      )
    )
  }

  // --- Drop new --------------------------------------------------------------
  function dropNew(x: number, y: number, json: string) {
    try {
      const tmpl: Item = JSON.parse(json)
      const id = nextId(tmpl.type)
      const instance: Item = { ...tmpl, id }
      setItems(prev => [...prev, instance])

      const w = tmpl.min_w, h = tmpl.min_h
      const nx = clamp(x, 0, room.width_cm - w)
      const ny = clamp(y, 0, room.height_cm - h)
      setPlacements(prev => [...prev, { id, x: nx, y: ny, w, h, rotation: 0 }])
      setSelectedId(id)
    } catch (e) {
      console.error("Bad drop payload", e)
    }
  }

  // --- Rotate ---------------------------------------------------------------
  function rotatePlacement(id: string) {
    const item = items.find(i => i.id === id)
    if (!item || !item.allow_rotate.includes(90)) return
    setPlacements(prev => prev.map(p => {
      if (p.id !== id) return p
      const rotated = p.rotation === 90 ? 0 : 90
      let nw = p.h, nh = p.w
      if (p.x + nw > room.width_cm)  nw = room.width_cm - p.x
      if (p.y + nh > room.height_cm) nh = room.height_cm - p.y
      nw = clamp(nw, item.min_w, item.max_w)
      nh = clamp(nh, item.min_h, item.max_h)
      return { ...p, w: nw, h: nh, rotation: rotated }
    }))
  }

  // --- Resize with bounds ----------------------------------------------------
  function resizePlacement(id: string, w: number, h: number) {
    const item = items.find(i => i.id === id)
    if (!item) return
    setPlacements(prev => prev.map(p => {
      if (p.id !== id) return p
      let nw = clamp(w, item.min_w, item.max_w)
      let nh = clamp(h, item.min_h, item.max_h)
      nw = clamp(nw, 10, room.width_cm - p.x)
      nh = clamp(nh, 10, room.height_cm - p.y)
      return { ...p, w: nw, h: nh }
    }))
  }

  // --- Delete ---------------------------------------------------------------
  function deletePlacement(id: string) {
    setPlacements(prev => prev.filter(p => p.id !== id))
    setItems(prev => prev.filter(i => i.id !== id))
    if (selectedId === id) setSelectedId(null)
  }

  // --- Keyboard shortcuts ----------------------------------------------------
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (!selectedId) return
      const step = e.shiftKey ? 10 : 1
      if (["ArrowUp","ArrowDown","ArrowLeft","ArrowRight"].includes(e.key)) e.preventDefault()

      if (e.key === "r" || e.key === "R") {
        rotatePlacement(selectedId)
      } else if (e.key === "Delete" || e.key === "Backspace") {
        deletePlacement(selectedId)
      } else if (e.key === "ArrowLeft") {
        const p = placements.find(p => p.id === selectedId); if (!p) return
        movePlacement(selectedId, p.x - step, p.y)
      } else if (e.key === "ArrowRight") {
        const p = placements.find(p => p.id === selectedId); if (!p) return
        movePlacement(selectedId, p.x + step, p.y)
      } else if (e.key === "ArrowUp") {
        const p = placements.find(p => p.id === selectedId); if (!p) return
        movePlacement(selectedId, p.x, p.y - step)
      } else if (e.key === "ArrowDown") {
        const p = placements.find(p => p.id === selectedId); if (!p) return
        movePlacement(selectedId, p.x, p.y + step)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [selectedId, placements])

  // --- Auto-arrange & save ---------------------------------------------------
  async function autoArrange(method: "ga" | "sa") {
    setBusy(true)
    try {
      const res = await solveLayout(room, items, constraints, method)
      setPlacements(res.placements || [])
      setSelectedId(null)
    } finally { setBusy(false) }
  }
  async function saveAsFinal() {
    await finalize({ accepted: true, room, items, constraints, placements })
    alert("Saved as final and weights updated.")
  }

  // --- Room size (keep items inside) ----------------------------------------
  function updateRoomSize(wcm: number, hcm: number) {
    wcm = Math.max(100, Math.round(wcm))
    hcm = Math.max(100, Math.round(hcm))
    setRoom(r => ({ ...r, width_cm: wcm, height_cm: hcm }))
    setPlacements(prev => prev.map(p => ({
      ...p,
      x: clamp(p.x, 0, wcm - p.w),
      y: clamp(p.y, 0, hcm - p.h)
    })))
  }

  return (
    <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 16, padding: 16, fontFamily: "Inter, system-ui, sans-serif" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <h2>AI Room Planner (Local GA/SA)</h2>

        <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:8}}>
          <label>Width (cm)
            <input type="number" value={room.width_cm} onChange={e => updateRoomSize(Number(e.target.value), room.height_cm)} style={inp}/>
          </label>
          <label>Height (cm)
            <input type="number" value={room.height_cm} onChange={e => updateRoomSize(room.width_cm, Number(e.target.value))} style={inp}/>
          </label>
        </div>

        <div style={{ display: "flex", gap: 8 }}>
          <button disabled={busy} onClick={() => autoArrange("ga")}>Auto-arrange (GA)</button>
          <button disabled={busy} onClick={() => autoArrange("sa")}>Auto-arrange (SA)</button>
        </div>
        <button onClick={saveAsFinal}>Save as Final (learn)</button>
        {busy && <div>Solving...</div>}

        <Palette catalog={catalog} />
      </div>

      <div>
        <RoomCanvas
          room={room}
          placements={placements}
          onMove={movePlacement}
          onDropNew={dropNew}
          onRotate={rotatePlacement}
          onDelete={deletePlacement}
          onResize={resizePlacement}
          onSelect={setSelectedId}
          selectedId={selectedId}
        />
        <div style={{marginTop:8, fontSize:12, opacity:0.7}}>
          Tip: drag items into the room • drag to move • double-click or press <b>R</b> to rotate • <b>Del</b> to delete • use corner handles to resize • hold <b>Shift</b> for faster arrow nudges
        </div>
      </div>
    </div>
  )
}

const inp: React.CSSProperties = { width:'100%', padding:'6px 8px', border:'1px solid #ccc', borderRadius:6 }
