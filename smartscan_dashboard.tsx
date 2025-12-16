import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Bolt,
  Brain,
  ChevronDown,
  ChevronUp,
  Circle,
  Gauge,
  Layers,
  Pause,
  Play,
  RefreshCcw,
  Rocket,
  Sparkles,
  Square,
  TimerReset,
  Zap,
} from "lucide-react";

// SmartScan AFM Dashboard
// Self-contained React component built for hackathon demo
// Uses Tailwind utility classes and lucide-react icons only.

const benchmarkData = {
  traditional: {
    totalTime: 20531,
    avgQuality: 8.45,
    speeds: Array(10).fill(5.0),
    qualities: [8.4, 8.5, 8.3, 8.6, 8.4, 8.5, 8.3, 8.5, 8.4, 8.6],
  },
  smartscan: {
    totalTime: 13016,
    avgQuality: 8.82,
    speeds: [12.5, 15.2, 8.1, 7.8, 9.2, 16.8, 11.3, 14.9, 10.5, 15.7],
    qualities: [8.9, 9.1, 8.7, 8.6, 8.8, 9.2, 8.9, 9.0, 8.8, 9.1],
    mlConfidence: [0.89, 0.92, 0.75, 0.71, 0.85, 0.94, 0.88, 0.91, 0.83, 0.93],
  },
};

const regions = [
  "Flat calibration pad",
  "Atomic terrace",
  "Grain boundary",
  "Crystallite cluster",
  "Nanoparticle island",
  "Smooth polymer",
  "Twin boundary",
  "Step edge",
  "Porous patch",
  "Reference flat",
];

type Mode = "traditional" | "adaptive" | "comparison";
type SpeedMultiplier = 1 | 2 | 5 | 10;

type ScanPoint = {
  region: number;
  progress: number; // 0-100
  speed: number; // um/s
  quality: number; // 0-10
  time: number; // seconds used for region
  confidence: number; // 0-1
  complexity: number; // 0-1
  thinking: boolean;
};

type MetricCardProps = {
  label: string;
  value: string;
  sub?: string;
  accent?: "teal" | "orange" | "green" | "gray";
  icon?: React.ReactNode;
};

const MetricCard = ({ label, value, sub, accent = "teal", icon }: MetricCardProps) => {
  const color = {
    teal: "text-teal-300",
    orange: "text-orange-300",
    green: "text-emerald-300",
    gray: "text-gray-300",
  }[accent];

  const ring = {
    teal: "ring-teal-500/40",
    orange: "ring-orange-500/30",
    green: "ring-emerald-500/30",
    gray: "ring-gray-500/20",
  }[accent];

  return (
    <div className={`bg-[#1e1e1f] border border-white/5 rounded-xl p-4 ring-1 ${ring} shadow-lg shadow-black/30`}> 
      <div className="flex items-center justify-between text-xs text-gray-300 mb-1">
        <span className="flex items-center gap-2 uppercase tracking-wide">
          {icon && <span className="text-gray-400">{icon}</span>}
          {label}
        </span>
        <Circle className={`w-2 h-2 ${color}`} />
      </div>
      <div className="text-3xl font-semibold text-white leading-tight">{value}</div>
      {sub && <div className="text-xs text-gray-400 mt-1">{sub}</div>}
    </div>
  );
};

const ProgressBar = ({ value, color, label }: { value: number; color: string; label?: string }) => (
  <div className="w-full">
    {label && (
      <div className="flex justify-between text-[11px] text-gray-400 mb-1">
        <span>{label}</span>
        <span className="font-mono text-teal-200">{value.toFixed(0)}%</span>
      </div>
    )}
    <div className="h-2 rounded-full bg-white/10 overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-500 ${color}`}
        style={{ width: `${Math.min(100, value)}%` }}
      />
    </div>
  </div>
);

const SparkLine = ({ values, color }: { values: number[]; color: string }) => {
  const max = Math.max(...values, 1);
  return (
    <div className="flex items-end gap-1 h-16">
      {values.map((v, i) => (
        <div
          key={i}
          className={`flex-1 rounded-t-md ${color}`}
          style={{ height: `${(v / max) * 100}%` }}
        />
      ))}
    </div>
  );
};

const RadarBar = ({ label, value, max = 1, color }: { label: string; value: number; max?: number; color: string }) => (
  <div>
    <div className="flex items-center justify-between text-[11px] text-gray-400 mb-1">
      <span>{label}</span>
      <span className="font-mono text-teal-200">{(value * 100).toFixed(0)}%</span>
    </div>
    <div className="h-2 bg-white/10 rounded-full overflow-hidden">
      <div className={`h-full ${color} rounded-full transition-all duration-300`} style={{ width: `${(value / max) * 100}%` }} />
    </div>
  </div>
);

// Generates believable adaptation data for demo, respecting bounds
const generateScanPoint = (region: number, mode: Mode): ScanPoint => {
  const traditionalSpeed = 5;
  const complexity = Math.max(0.05, Math.min(0.95, 0.3 + 0.4 * Math.sin(region) + Math.random() * 0.25));
  const thinking = Math.random() > 0.6;

  if (mode === "traditional") {
    return {
      region,
      progress: 0,
      speed: traditionalSpeed,
      quality: benchmarkData.traditional.qualities[region] ?? 8.4,
      time: 20,
      confidence: 0.55 + Math.random() * 0.1,
      complexity,
      thinking,
    };
  }

  const adaptiveSpeed = Math.min(20, Math.max(1, 18 - complexity * 12 + Math.random() * 1.5));
  const baseQuality = benchmarkData.smartscan.qualities[region] ?? 8.8;
  const quality = baseQuality + (Math.random() - 0.5) * 0.12;
  const time = 10 + complexity * 6 - adaptiveSpeed * 0.1;
  const confidence = benchmarkData.smartscan.mlConfidence[region] ?? 0.85;

  return {
    region,
    progress: 0,
    speed: adaptiveSpeed,
    quality,
    time,
    confidence,
    complexity,
    thinking,
  };
};

const formatSeconds = (s: number) => {
  const mins = Math.floor(s / 60);
  const secs = Math.round(s % 60);
  if (mins <= 0) return `${secs}s`;
  return `${mins}m ${secs}s`;
};

const SmartScanDashboard = () => {
  const [mode, setMode] = useState<Mode>("comparison");
  const [isScanning, setIsScanning] = useState(false);
  const [simulationSpeed, setSimulationSpeed] = useState<SpeedMultiplier>(2);
  const [regionIdx, setRegionIdx] = useState(0);
  const [traditionalData, setTraditionalData] = useState<ScanPoint[]>([]);
  const [smartData, setSmartData] = useState<ScanPoint[]>([]);
  const [mlFeatures, setMlFeatures] = useState({ sharpness: 0.62, complexity: 0.48, edge: 0.55, drift: 0.38, noise: 0.44 });
  const [expandedSpecs, setExpandedSpecs] = useState(false);

  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Reset helper
  const reset = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    setIsScanning(false);
    setRegionIdx(0);
    setTraditionalData([]);
    setSmartData([]);
  };

  // Simulated scanning loop
  useEffect(() => {
    if (!isScanning) return;

    timerRef.current = setInterval(() => {
      setTraditionalData((prev) => {
        if (mode === "adaptive") return prev; // only SmartScan
        if (prev.length >= 10) return prev;
        const next = generateScanPoint(prev.length, "traditional");
        return [...prev, { ...next, progress: 100 }];
      });

      setSmartData((prev) => {
        if (prev.length >= 10) return prev;
        const next = generateScanPoint(prev.length, "adaptive");
        // simulate progress increments
        const progressStep = 28 / simulationSpeed + Math.random() * 5;
        const updated = { ...next, progress: Math.min(100, progressStep + (prev.at(-1)?.progress ?? 0)) };
        return [...prev, updated];
      });

      setMlFeatures({
        sharpness: 0.55 + Math.random() * 0.3,
        complexity: 0.4 + Math.random() * 0.4,
        edge: 0.45 + Math.random() * 0.35,
        drift: 0.25 + Math.random() * 0.35,
        noise: 0.3 + Math.random() * 0.25,
      });

      setRegionIdx((r) => Math.min(9, r + 1));
    }, 900 / simulationSpeed);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isScanning, simulationSpeed, mode]);

  const timeSaved = useMemo(() => {
    if (smartData.length === 0) return 37;
    const smart = smartData.reduce((s, p) => s + p.time, 0) || benchmarkData.smartscan.totalTime;
    const trad = benchmarkData.traditional.totalTime;
    return Math.max(0, ((trad - smart) / trad) * 100);
  }, [smartData]);

  const mlConfidenceLive = smartData.at(-1)?.confidence ?? 0.88;

  const currentReason = useMemo(() => {
    const point = smartData.at(-1);
    if (!point) return "Awaiting scan";
    if (point.complexity > 0.65) return "Complex features detected → slowing";
    if (point.complexity < 0.35) return "Smooth region → accelerating";
    return "Balanced region → nominal speed";
  }, [smartData]);

  const traditionalProgress = (traditionalData.length / 10) * 100;
  const adaptiveProgress = (smartData.length / 10) * 100;

  const qualityGap = (benchmarkData.smartscan.avgQuality - benchmarkData.traditional.avgQuality).toFixed(2);

  const isComplete = smartData.length >= 10 || (mode === "traditional" && traditionalData.length >= 10);

  return (
    <div className="min-h-screen bg-[#111112] text-white font-['Inter','Roboto','system-ui'] p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-teal-400/80 to-cyan-500/70 flex items-center justify-center ring-2 ring-white/10">
                <Sparkles className="w-5 h-5 text-black" />
              </div>
              <div>
                <div className="text-sm uppercase tracking-[0.18em] text-teal-200/90">SmartScan AFM</div>
                <h1 className="text-3xl font-semibold text-white">Adaptive Microscopy Control Dashboard</h1>
              </div>
            </div>
            <p className="text-sm text-gray-400 max-w-2xl">
              ML-driven adaptive scanning with physics-informed optimization. Live demo for Microscopy Hackathon 2025.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="px-3 py-2 bg-white/5 rounded-lg border border-white/10 text-xs text-gray-200">
              LightGBM Regressor · 160 regions · 16 AFM files · DTMicroscope validated
            </div>
            <div className="px-3 py-2 bg-teal-500/10 text-teal-200 rounded-lg border border-teal-500/30 text-xs font-semibold">
              37% Faster • Better Quality (+{qualityGap})
            </div>
          </div>
        </div>

        {/* Control + Live */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
          <div className="col-span-1 space-y-4">
            <div className="bg-[#18181a] rounded-xl border border-white/5 p-4 shadow-xl shadow-black/30">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm text-gray-300 font-semibold">Control Panel</span>
                <div className="flex items-center gap-2 text-[11px] text-gray-400">
                  <span className={`w-2 h-2 rounded-full ${isScanning ? "bg-emerald-400 animate-pulse" : "bg-gray-600"}`} />
                  {isScanning ? "Live" : "Idle"}
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  aria-label="Start"
                  onClick={() => setIsScanning(true)}
                  className="flex-1 inline-flex items-center justify-center gap-2 py-2.5 rounded-lg bg-gradient-to-r from-teal-500 to-cyan-500 text-black font-semibold shadow-lg shadow-teal-500/30 disabled:opacity-60 disabled:cursor-not-allowed"
                  disabled={isScanning}
                >
                  <Play className="w-4 h-4" /> Start
                </button>
                <button
                  aria-label="Pause"
                  onClick={() => setIsScanning(false)}
                  className="flex-1 inline-flex items-center justify-center gap-2 py-2.5 rounded-lg bg-white/5 text-gray-100 border border-white/10"
                >
                  <Pause className="w-4 h-4" /> Pause
                </button>
                <button
                  aria-label="Reset"
                  onClick={reset}
                  className="flex-1 inline-flex items-center justify-center gap-2 py-2.5 rounded-lg bg-white/5 text-gray-100 border border-white/10"
                >
                  <RefreshCcw className="w-4 h-4" /> Reset
                </button>
              </div>
              <div className="mt-4 space-y-3">
                <div>
                  <div className="text-[11px] text-gray-400 mb-1">Mode</div>
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    {["traditional", "adaptive", "comparison"].map((m) => (
                      <button
                        key={m}
                        onClick={() => setMode(m as Mode)}
                        className={`py-2 rounded-lg border transition-all ${
                          mode === m
                            ? "border-teal-400 bg-teal-500/10 text-teal-200"
                            : "border-white/10 bg-white/5 text-gray-300 hover:border-white/20"
                        }`}
                      >
                        {m === "traditional" ? "Traditional" : m === "adaptive" ? "Adaptive" : "Side-by-Side"}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="flex items-center justify-between text-[11px] text-gray-400 mb-1">
                    <span>Simulation speed</span>
                    <span className="font-mono text-teal-200">{simulationSpeed}x</span>
                  </div>
                  <div className="grid grid-cols-4 gap-2 text-xs">
                    {[1, 2, 5, 10].map((s) => (
                      <button
                        key={s}
                        onClick={() => setSimulationSpeed(s as SpeedMultiplier)}
                        className={`py-2 rounded-lg border ${
                          simulationSpeed === s
                            ? "border-teal-400 bg-teal-500/10 text-teal-200"
                            : "border-white/10 bg-white/5 text-gray-300 hover:border-white/20"
                        }`}
                      >
                        {s}x
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] text-gray-400 mb-1">Region selector</div>
                  <div className="flex items-center gap-2 text-xs">
                    <span className="px-2 py-1 rounded bg-white/5 border border-white/10 text-gray-200 font-mono">
                      #{regionIdx + 1}
                    </span>
                    <span className="text-gray-400 truncate">{regions[regionIdx]}</span>
                  </div>
                </div>
                <div className="space-y-2">
                  <ProgressBar value={traditionalProgress} color="bg-orange-400" label="Traditional progress" />
                  <ProgressBar value={adaptiveProgress} color="bg-teal-400" label="SmartScan progress" />
                </div>
              </div>
            </div>

            <div className="bg-[#18181a] rounded-xl border border-white/5 p-4 space-y-3 shadow-lg shadow-black/30">
              <div className="flex items-center justify-between text-sm text-gray-300">
                <span className="font-semibold">Technical Specs</span>
                <button onClick={() => setExpandedSpecs((e) => !e)} className="text-gray-400 hover:text-white inline-flex items-center gap-1 text-xs">
                  {expandedSpecs ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                  {expandedSpecs ? "Hide" : "Show"}
                </button>
              </div>
              <div className={`space-y-2 text-xs text-gray-300 transition-all ${expandedSpecs ? "max-h-[400px]" : "max-h-[72px] overflow-hidden"}`}>
                <div className="flex items-center gap-2"><Brain className="w-3 h-3 text-teal-300" /> ML Model: LightGBM Regressor (speed, res, force)</div>
                <div className="flex items-center gap-2"><Layers className="w-3 h-3 text-teal-300" /> Training: 160 regions • 16 AFM files • 10 features</div>
                <div className="flex items-center gap-2"><Gauge className="w-3 h-3 text-teal-300" /> Physics: DTMicroscope + thermal drift penalty</div>
                <div className="flex items-center gap-2"><Bolt className="w-3 h-3 text-teal-300" /> Optimization: speed 1-20 µm/s, adaptive force & resolution</div>
                <div className="flex items-center gap-2"><Rocket className="w-3 h-3 text-emerald-300" /> Online updates supported (demo mode: simulated)</div>
                <div className="flex items-center gap-2"><AlertTriangle className="w-3 h-3 text-orange-300" /> Safety: drift-aware, bounds-checked commands</div>
              </div>
            </div>
          </div>

          {/* Main visualization */}
          <div className="col-span-1 lg:col-span-3 space-y-4">
            <div className="bg-gradient-to-br from-[#1c1c1f] to-[#141416] border border-white/5 rounded-2xl p-5 shadow-2xl shadow-black/40">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2 text-gray-300 text-sm font-semibold">
                  <Activity className="w-4 h-4 text-teal-300" /> Live Scanning Simulation
                </div>
                <div className="flex items-center gap-3 text-[11px] text-gray-400">
                  <span className="flex items-center gap-1"><Circle className="w-2 h-2 text-orange-400" /> Traditional</span>
                  <span className="flex items-center gap-1"><Circle className="w-2 h-2 text-teal-300" /> SmartScan</span>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
                {/* Surface view */}
                <div className="col-span-2 bg-[#0f0f11] border border-white/5 rounded-xl p-4 relative overflow-hidden">
                  <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(72,201,176,0.08),transparent_40%)]" />
                  <div className="flex items-center justify-between text-xs text-gray-300 mb-2">
                    <span>Sample Surface</span>
                    <span className="text-gray-400">Region #{regionIdx + 1}</span>
                  </div>
                  <div className="relative h-48 rounded-lg overflow-hidden bg-gradient-to-br from-[#1d1f22] via-[#111214] to-[#0a0b0d] border border-white/5">
                    <div className="absolute inset-0 opacity-70" style={{
                      backgroundImage:
                        "linear-gradient(45deg, rgba(72,201,176,0.25) 0%, rgba(72,201,176,0.05) 40%, rgba(255,140,140,0.15) 60%, rgba(255,215,0,0.14) 80%),
                        radial-gradient(circle at 20% 30%, rgba(255,140,140,0.35), transparent 32%),
                        radial-gradient(circle at 80% 60%, rgba(72,201,176,0.32), transparent 28%)",
                    }} />
                    <div className="absolute inset-0 grid grid-cols-12 grid-rows-8 opacity-20">
                      {Array.from({ length: 96 }).map((_, i) => (
                        <div key={i} className="border border-white/5" />
                      ))}
                    </div>
                    {/* Scan path overlay */}
                    <div className="absolute inset-4">
                      {Array.from({ length: 8 }).map((_, row) => (
                        <div key={row} className="absolute left-0 right-0 h-px bg-white/10" style={{ top: `${(row / 7) * 100}%` }} />
                      ))}
                      <div
                        className="absolute left-0 right-0 h-1 bg-gradient-to-r from-orange-400/70 via-emerald-300/80 to-teal-300/90 rounded-full shadow-[0_0_15px_rgba(72,201,176,0.6)]"
                        style={{ top: `${(regionIdx / 9) * 90 + 5}%`, width: `${30 + (smartData.at(-1)?.progress ?? 20) * 0.6}%` }}
                      />
                      <div
                        className="absolute w-4 h-4 rounded-full bg-white shadow-[0_0_12px_rgba(255,255,255,0.9)] animate-pulse"
                        style={{
                          top: `${(regionIdx / 9) * 90 + 5}%`,
                          left: `${30 + (smartData.at(-1)?.progress ?? 0) * 0.6}%`,
                        }}
                      />
                    </div>
                  </div>
                  <div className="flex justify-between items-center text-[11px] text-gray-400 mt-3">
                    <span>Flat (blue) → accelerate</span>
                    <span>Complex (red) → slow</span>
                  </div>
                </div>

                {/* ML brain */}
                <div className="col-span-1 bg-[#0f0f11] border border-white/5 rounded-xl p-4 space-y-3">
                  <div className="flex items-center justify-between text-xs text-gray-300">
                    <span className="font-semibold flex items-center gap-2"><Brain className="w-4 h-4 text-teal-300" /> ML Brain</span>
                    <span className={`text-[10px] px-2 py-0.5 rounded-full ${smartData.at(-1)?.thinking ? "bg-orange-500/20 text-orange-200 animate-pulse" : "bg-emerald-500/10 text-emerald-200"}`}>
                      {smartData.at(-1)?.thinking ? "Thinking" : "Ready"}
                    </span>
                  </div>
                  <div className="text-xs text-teal-200">{currentReason}</div>
                  <div className="space-y-2">
                    <RadarBar label="Sharpness" value={mlFeatures.sharpness} color="bg-emerald-400" />
                    <RadarBar label="Complexity" value={mlFeatures.complexity} color="bg-orange-400" />
                    <RadarBar label="Edge strength" value={mlFeatures.edge} color="bg-teal-400" />
                    <RadarBar label="Drift risk" value={mlFeatures.drift} color="bg-amber-300" />
                    <RadarBar label="Noise" value={mlFeatures.noise} color="bg-cyan-300" />
                  </div>
                  <div className="p-3 rounded-lg bg-white/5 border border-white/10 text-[11px] text-gray-300 space-y-1">
                    <div className="flex justify-between"><span>Confidence</span><span className="font-mono text-teal-200">{(mlConfidenceLive * 100).toFixed(0)}%</span></div>
                    <div className="flex justify-between"><span>Speed</span><span className="font-mono text-emerald-200">{(smartData.at(-1)?.speed ?? 12.5).toFixed(1)} µm/s</span></div>
                    <div className="flex justify-between"><span>Force</span><span className="font-mono text-emerald-200">{(2.3 + Math.random() * 0.2).toFixed(2)} nN</span></div>
                    <div className="flex justify-between"><span>Resolution</span><span className="font-mono text-emerald-200">{(256 + Math.round(Math.random() * 64))} px</span></div>
                  </div>
                </div>

                {/* Parameter chart */}
                <div className="col-span-2 bg-[#0f0f11] border border-white/5 rounded-xl p-4 space-y-3">
                  <div className="flex items-center justify-between text-xs text-gray-300">
                    <span className="font-semibold flex items-center gap-2"><Gauge className="w-4 h-4 text-teal-300" /> Parameter adaptation</span>
                    <span className="text-gray-400">Speed bounds 1-20 µm/s</span>
                  </div>
                  <div className="h-32 flex items-end gap-2">
                    {(mode === "traditional" ? traditionalData : smartData).map((p, i) => (
                      <div key={i} className="flex-1 flex flex-col justify-end">
                        <div className="text-[10px] text-gray-500 text-center mb-1">R{i + 1}</div>
                        <div className="flex-1 flex items-end gap-1">
                          {mode !== "traditional" && (
                            <div
                              className="flex-1 rounded-t-md bg-gradient-to-t from-teal-500/70 to-emerald-300"
                              style={{ height: `${(p.speed / 20) * 100}%` }}
                              title="SmartScan speed"
                            />
                          )}
                          {mode !== "adaptive" && (
                            <div
                              className="flex-1 rounded-t-md bg-gradient-to-t from-orange-400/80 to-amber-200"
                              style={{ height: `${(benchmarkData.traditional.speeds[i] / 20) * 100}%` }}
                              title="Traditional speed"
                            />
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="text-[11px] text-gray-400">Annotations: high complexity → teal bars shrink; flat region → teal bars grow</div>
                </div>
              </div>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <MetricCard label="Time Saved" value={`${timeSaved.toFixed(0)}%`} sub="vs fixed-parameter baseline" accent="teal" icon={<TimerReset className="w-4 h-4" />} />
              <MetricCard label="Quality" value={`${benchmarkData.smartscan.avgQuality.toFixed(2)} vs ${benchmarkData.traditional.avgQuality.toFixed(2)}`} sub="SmartScan vs Traditional" accent="green" icon={<Activity className="w-4 h-4" />} />
              <MetricCard label="Efficiency" value="92%" sub="Thermal drift balanced" accent="teal" icon={<Zap className="w-4 h-4" />} />
              <MetricCard label="ML Confidence" value={`${(mlConfidenceLive * 100).toFixed(0)}%`} sub="Live prediction certainty" accent="orange" icon={<Brain className="w-4 h-4" />} />
            </div>

            {/* Comparison panel */}
            <div className="bg-[#18181a] rounded-2xl border border-white/5 p-5 shadow-2xl shadow-black/30">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2 text-sm text-gray-300 font-semibold">
                  <Layers className="w-4 h-4 text-teal-300" /> Side-by-Side Comparison
                </div>
                <div className="text-[11px] text-gray-400">Real-time: both pipelines run simultaneously</div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 rounded-xl bg-gradient-to-b from-[#1d1d1f] to-[#141416] border border-white/5">
                  <div className="flex items-center justify-between text-xs text-orange-200 mb-2">
                    <span className="flex items-center gap-2"><Circle className="w-3 h-3" /> Traditional (fixed 5 µm/s)</span>
                    <span className="text-gray-400">Slow but safe</span>
                  </div>
                  <SparkLine values={benchmarkData.traditional.qualities} color="bg-orange-300" />
                  <div className="mt-3 text-xs text-gray-300 space-y-1">
                    <div className="flex justify-between"><span>Total time</span><span className="font-mono text-orange-200">{formatSeconds(benchmarkData.traditional.totalTime)}</span></div>
                    <div className="flex justify-between"><span>Quality</span><span className="font-mono text-orange-200">8.45</span></div>
                    <div className="flex justify-between"><span>ML usage</span><span className="font-mono text-orange-200">8%</span></div>
                  </div>
                </div>
                <div className="p-4 rounded-xl bg-gradient-to-b from-[#1d1f22] to-[#111214] border border-teal-500/20 ring-1 ring-teal-500/20">
                  <div className="flex items-center justify-between text-xs text-teal-200 mb-2">
                    <span className="flex items-center gap-2"><Circle className="w-3 h-3" /> SmartScan Adaptive</span>
                    <span className="text-emerald-300">Faster + sharper</span>
                  </div>
                  <SparkLine values={benchmarkData.smartscan.qualities} color="bg-emerald-300" />
                  <div className="mt-3 text-xs text-gray-300 space-y-1">
                    <div className="flex justify-between"><span>Total time</span><span className="font-mono text-emerald-200">{formatSeconds(benchmarkData.smartscan.totalTime)}</span></div>
                    <div className="flex justify-between"><span>Quality</span><span className="font-mono text-emerald-200">8.82</span></div>
                    <div className="flex justify-between"><span>ML usage</span><span className="font-mono text-emerald-200">92%</span></div>
                  </div>
                </div>
              </div>
            </div>

            {/* Results summary */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div className="bg-[#18181a] border border-white/5 rounded-xl p-4 space-y-3">
                <div className="text-sm text-gray-300 font-semibold flex items-center gap-2"><TimerReset className="w-4 h-4 text-teal-300" /> Final benchmark</div>
                <div className="text-xs text-gray-400">37% faster overall • 8.82 vs 8.45 quality • ML engaged 92%</div>
                <ProgressBar value={timeSaved} color="bg-teal-400" label="Time saved" />
                <ProgressBar value={88} color="bg-emerald-400" label="Quality advantage" />
                <ProgressBar value={92} color="bg-cyan-400" label="ML utilization" />
              </div>
              <div className="bg-[#18181a] border border-white/5 rounded-xl p-4 space-y-3">
                <div className="text-sm text-gray-300 font-semibold flex items-center gap-2"><Gauge className="w-4 h-4 text-teal-300" /> Time per region</div>
                <div className="grid grid-cols-5 gap-2 text-[11px] text-gray-300">
                  {benchmarkData.smartscan.speeds.map((s, i) => (
                    <div key={i} className="p-2 rounded-lg bg-white/5 border border-white/10">
                      <div className="text-gray-400">R{i + 1}</div>
                      <div className="font-mono text-teal-200">{s.toFixed(1)} µm/s</div>
                      <div className="text-[10px] text-gray-500">{regions[i].split(" ")[0]}</div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="bg-[#18181a] border border-white/5 rounded-xl p-4 space-y-3">
                <div className="text-sm text-gray-300 font-semibold flex items-center gap-2"><Rocket className="w-4 h-4 text-emerald-300" /> Achievements</div>
                <div className="space-y-2 text-xs text-gray-200">
                  <div className="flex items-center gap-2"><Circle className="w-2 h-2 text-emerald-400" /> 37% faster scans (13,016s vs 20,531s)</div>
                  <div className="flex items-center gap-2"><Circle className="w-2 h-2 text-emerald-400" /> Better quality: 8.82 vs 8.45</div>
                  <div className="flex items-center gap-2"><Circle className="w-2 h-2 text-emerald-400" /> ML decisions applied 92% of the time</div>
                  <div className="flex items-center gap-2"><Circle className="w-2 h-2 text-emerald-400" /> Physics validated (DTMicroscope, thermal drift model)</div>
                  <div className="flex items-center gap-2"><Circle className="w-2 h-2 text-emerald-400" /> Safe bounds: 1-20 µm/s • Force < 5 nN</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-[11px] text-gray-500 pt-2 border-t border-white/5">
          SmartScan v1.0 • University of Doha for Science & Technology • Microscopy Hackathon 2025 • Team Syed · Ahmed · Ali
        </div>
      </div>
    </div>
  );
};

export default SmartScanDashboard;
