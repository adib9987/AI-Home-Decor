import React, { useRef, useState } from "react"
import type { Placement, Room } from "../types"

const PX = 1.0

type Props = {
  room: Room
  placements: Placement[]
  onMove?: (id: string, x: number, y: number) => void
  onDropNew?: (x: number, y: number, clientDataJSON: string) => void
  onRotate?: (id: string) => void
  onDelete?: (id: string) => void
  onResize?: (id: string, w: number, h: number) => void
  onSelect?: (id: string | null) => void
  selectedId?: string | null
}

export default function RoomCanvas({
  room, placements, onMove, onDropNew, onRotate, onDelete, onResize, onSelect, selectedId
}: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const [dragId, setDragId] = useState<string | null>(null)
  const [offset, setOffset] = useState({ dx: 0, dy: 0 })
  const [resizeState, setResizeState] = useState<null | {
    id: string, corner: "nw" | "ne" | "sw" | "se",
    startW: number, startH: number, startPX: number, startPY: number
  }>(null)

  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

  // --- Drag to move ----------------------------------------------------------
  function onPointerDown(e: React.PointerEvent, p: Placement) {
    const box = ref.current!.getBoundingClientRect()
    const x = (e.clientX - box.left) / PX
    const y = (e.clientY - box.top) / PX
    setDragId(p.id)
    onSelect?.(p.id)
    setOffset({ dx: x - p.x, dy: y - p.y })
    ;(e.target as Element).setPointerCapture(e.pointerId)
  }
  function onPointerMove(e: React.PointerEvent) {
    if (resizeState && onResize) {
      const box = ref.current!.getBoundingClientRect()
      const mx = (e.clientX - box.left) / PX
      const my = (e.clientY - box.top) / PX
      const { id, corner, startW, startH, startPX, startPY } = resizeState

      let nx = startPX, ny = startPY, nw = startW, nh = startH

      if (corner === "se") {
        nw = clamp(Math.round(mx - startPX), 10, room.width_cm - startPX)
        nh = clamp(Math.round(my - startPY), 10, room.height_cm - startPY)
      } else if (corner === "sw") {
        nw = clamp(Math.round(startW + (startPX - mx)), 10, room.width_cm)
        nx = clamp(Math.round(mx), 0, startPX + startW)
        nh = clamp(Math.round(my - startPY), 10, room.height_cm - startPY)
      } else if (corner === "ne") {
        nh = clamp(Math.round(startH + (startPY - my)), 10, room.height_cm)
        ny = clamp(Math.round(my), 0, startPY + startH)
        nw = clamp(Math.round(mx - startPX), 10, room.width_cm - startPX)
      } else if (corner === "nw") {
        nw = clamp(Math.round(startW + (startPX - mx)), 10, room.width_cm)
        nh = clamp(Math.round(startH + (startPY - my)), 10, room.height_cm)
        nx = clamp(Math.round(mx), 0, startPX + startW)
        ny = clamp(Math.round(my), 0, startPY + startH)
      }

      onMove?.(id, nx, ny)
      onResize?.(id, nw, nh)
      return
    }

    if (!dragId || !onMove) return
    const box = ref.current!.getBoundingClientRect()
    const x = (e.clientX - box.left) / PX
    const y = (e.clientY - box.top) / PX
    const nx = clamp(Math.round(x - offset.dx), 0, room.width_cm)
    const ny = clamp(Math.round(y - offset.dy), 0, room.height_cm)
    onMove(dragId, nx, ny)
  }
  function onPointerUp() {
    setDragId(null)
    setResizeState(null)
  }

  // --- Drop new --------------------------------------------------------------
  function handleDrop(e: React.DragEvent) {
    if (!onDropNew) return
    e.preventDefault()
    const box = ref.current!.getBoundingClientRect()
    const x = Math.round((e.clientX - box.left) / PX)
    const y = Math.round((e.clientY - box.top) / PX)
    const payload = e.dataTransfer.getData("application/json")
    if (payload) onDropNew(x, y, payload)
  }

  // --- Begin resize ----------------------------------------------------------
  function beginResize(e: React.PointerEvent, p: Placement, corner: "nw" | "ne" | "sw" | "se") {
    e.stopPropagation()
    setResizeState({
      id: p.id, corner,
      startW: p.w, startH: p.h, startPX: p.x, startPY: p.y
    })
    ;(e.target as Element).setPointerCapture(e.pointerId)
  }

  return (
    <div
      ref={ref}
      style={{
        position: "relative",
        border: "1px solid #ccc",
        width: room.width_cm * PX,
        height: room.height_cm * PX,
        background: "#fafafa",
        userSelect: "none",
        borderRadius: 8,
        outline: "none",
      }}
      tabIndex={0}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      onDoubleClick={() => { if (selectedId && onRotate) onRotate(selectedId) }}
      onClick={() => onSelect?.(null)}
    >
      {[...room.doors, ...room.windows, ...room.keepouts].map((r, i) => (
        <div key={"blk" + i} style={{
          position: "absolute", left: r.x * PX, top: r.y * PX, width: r.w * PX, height: r.h * PX, background: "#ddd"
        }} />
      ))}

      {placements.map((p) => {
        const isSel = p.id === selectedId
        return (
          <div
            key={p.id}
            onPointerDown={(e) => onPointerDown(e, p)}
            onClick={(e) => { e.stopPropagation(); onSelect?.(p.id) }}
            onDoubleClick={(e) => { e.stopPropagation(); onRotate?.(p.id) }}
            style={{
              position: "absolute",
              left: p.x * PX,
              top: p.y * PX,
              width: p.w * PX,
              height: p.h * PX,
              border: isSel ? "2px solid #1976d2" : "1px solid #333",
              background: "rgba(0,0,0,0.04)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 12,
              cursor: "grab",
              boxShadow: isSel ? "0 0 0 2px rgba(25,118,210,0.2)" : undefined,
            }}
            title="Drag to move • Double-click to rotate"
          >
            {p.id}{p.rotation ? ` (${p.rotation}°)` : ""}

            {isSel && (
              <>
                <Handle pos="nw" onPointerDown={(e)=>beginResize(e,p,"nw")} />
                <Handle pos="ne" onPointerDown={(e)=>beginResize(e,p,"ne")} />
                <Handle pos="sw" onPointerDown={(e)=>beginResize(e,p,"sw")} />
                <Handle pos="se" onPointerDown={(e)=>beginResize(e,p,"se")} />

                <div style={{ position: "absolute", right: -8, top: -8, display: "flex", gap: 6, zIndex: 2 }}>
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); onRotate?.(p.id) }}
                    style={btnStyle}
                    title="Rotate 90°"
                  >⟳</button>
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); onDelete?.(p.id) }}
                    style={btnStyle}
                    title="Delete"
                  >✕</button>
                </div>
              </>
            )}
          </div>
        )
      })}
    </div>
  )
}

function Handle({ pos, onPointerDown }: { pos: "nw" | "ne" | "sw" | "se"; onPointerDown: (e: React.PointerEvent) => void }) {
  const style: React.CSSProperties = {
    position: "absolute",
    width: 10, height: 10, background: "#1976d2", borderRadius: 2,
    cursor:
      pos === "nw" ? "nwse-resize" :
      pos === "se" ? "nwse-resize" :
      pos === "ne" ? "nesw-resize" : "nesw-resize",
  }
  if (pos === "nw") { style.left = -5; style.top = -5 }
  if (pos === "ne") { style.right = -5; style.top = -5 }
  if (pos === "sw") { style.left = -5; style.bottom = -5 }
  if (pos === "se") { style.right = -5; style.bottom = -5 }
  return <div style={style} onPointerDown={onPointerDown} />
}

const btnStyle: React.CSSProperties = {
  border: "1px solid #aaa", background: "#fff", borderRadius: 6, width: 24, height: 24, cursor: "pointer"
}
