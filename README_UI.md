# SmartScan AFM Dashboard UI

A self-contained React dashboard that simulates the SmartScan adaptive AFM workflow for demos and judging.

## What's inside
- Component: `smartscan_dashboard.tsx` (Tailwind + lucide-react only)
- Built-in simulated data for traditional vs adaptive scanning
- Interactive controls (start/pause/reset, mode, speed)

## Prerequisites
- Node 18+ and npm/yarn
- Project already using React + Tailwind
- Install icons if needed: `npm install lucide-react`

## Quick start
1) Place `smartscan_dashboard.tsx` in your source (e.g., `src/`).
2) Render it (example):
   ```tsx
   import SmartScanDashboard from "./smartscan_dashboard";
   export default function App() {
     return <SmartScanDashboard />;
   }
   ```
3) Install deps (once): `npm install`
4) Run dev server:
   - Vite: `npm run dev`
   - CRA: `npm start`
5) Open the app (default):
   - Vite: http://localhost:5173
   - CRA: http://localhost:3000

## Controls & tips
- Modes: Traditional | Adaptive | Side-by-Side
- Speed: 1x/2x/5x/10x for faster demos
- Start/Pause/Reset for live simulation loops
- Works offline (all data simulated); best viewed on desktop/dark theme
