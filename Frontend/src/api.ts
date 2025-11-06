import type { Room, Item, Constraints, SolveResponse } from './types'
const BASE = 'http://127.0.0.1:8000'

export async function solveLayout(room:Room, items:Item[], constraints:Constraints, method:"ga"|"sa"="ga") {
  const r = await fetch(`${BASE}/solve-layout?method=${method}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ room, items, constraints })
  })
  return r.json() as Promise<SolveResponse>
}

export async function finalize(payload:any) {
  const r = await fetch(`${BASE}/finalize`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  return r.json()
}
