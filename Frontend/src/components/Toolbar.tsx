import React from 'react'
export default function Toolbar({onSolve}:{onSolve:()=>void}){ return (<div style={{display:'flex',gap:8}}><button onClick={onSolve}>Auto-arrange</button></div>) }