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
    TimerReset,
    Zap,
} from "lucide-react";

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
    progress: number;
    speed: number;
    quality: number;
    time: number;
    confidence: number;
    complexity: number;
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

const SectionCard = ({ title, children }: { title: string; children: React.ReactNode }) => (
    <div className="bg-[#18181a] border border-white/5 rounded-xl p-4 shadow-xl shadow-black/30">
        <div className="text-sm text-gray-300 font-semibold mb-3">{title}</div>
        {children}
    </div>
);

const Sparkline = ({
    data,
    color = "#76d7c4",
    height = 64,
    width = 300,
    dashedAt,
}: {
    data: number[];
    color?: string;
    height?: number;
    width?: number;
    dashedAt?: number;
}) => {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const norm = (v: number) => (max === min ? height / 2 : height - ((v - min) / (max - min)) * height);
    const points = data.map((v, i) => `${(i / (data.length - 1)) * width},${norm(v)}`).join(" ");
    return (
        <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
            {dashedAt !== undefined && (
                <line
                    x1={0}
                    y1={norm(dashedAt)}
                    x2={width}
                    y2={norm(dashedAt)}
                    stroke="#9ca3af"
                    strokeDasharray="6 4"
                    opacity={0.5}
                />
            )}
            <polyline points={points} fill="none" stroke={color} strokeWidth={2} />
        </svg>
    );
};

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

const SmartScanDashboard = () => {
    const [mode, setMode] = useState<Mode>("comparison");
    const [isScanning, setIsScanning] = useState(false);
    const [simulationSpeed, setSimulationSpeed] = useState<SpeedMultiplier>(2);
    const [regionIdx, setRegionIdx] = useState(0);
    const [traditionalData, setTraditionalData] = useState<ScanPoint[]>([]);
    const [smartData, setSmartData] = useState<ScanPoint[]>([]);
    const [expandedSpecs, setExpandedSpecs] = useState(false);

    const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

    const reset = () => {
        if (timerRef.current) clearInterval(timerRef.current);
        setIsScanning(false);
        setRegionIdx(0);
        setTraditionalData([]);
        setSmartData([]);
    };

    useEffect(() => {
        if (!isScanning) return;

        timerRef.current = setInterval(() => {
            setTraditionalData((prev) => {
                if (mode === "adaptive") return prev;
                if (prev.length >= 10) return prev;
                const next = generateScanPoint(prev.length, "traditional");
                return [...prev, { ...next, progress: 100 }];
            });

            setSmartData((prev) => {
                if (prev.length >= 10) return prev;
                const next = generateScanPoint(prev.length, "adaptive");
                const progressStep = 28 / simulationSpeed + Math.random() * 5;
                const updated = { ...next, progress: Math.min(100, progressStep + (prev.at(-1)?.progress ?? 0)) };
                return [...prev, updated];
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

    const traditionalProgress = (traditionalData.length / 10) * 100;
    const adaptiveProgress = (smartData.length / 10) * 100;

    const qualityGap = (benchmarkData.smartscan.avgQuality - benchmarkData.traditional.avgQuality).toFixed(2);

    return (
        <div className="min-h-screen bg-[#111112] text-white font-['Inter','Roboto','system-ui'] p-6">
            <div className="max-w-7xl mx-auto space-y-6">
                <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
                    <div className="space-y-1">
                        <div className="flex items-center gap-3">
                            <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-teal-400/80 to-cyan-500/70 flex items-center justify-center ring-2 ring-white/10">
                                <Sparkles className="w-5 h-5 text-black" />
                            </div>
                            <div>
                                <div className="text-sm uppercase tracking-[0.18em] text-teal-200/90">SmartScan AFM</div>
                                <h1 className="text-3xl font-semibold text-white">Real-Time Adaptive Scanning</h1>
                            </div>
                        </div>
                        <p className="text-sm text-gray-400 max-w-2xl">
                            ML-driven adaptive scanning with physics-informed optimization.
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
                                                className={`py-2 rounded-lg border transition-all ${mode === m
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
                                                className={`py-2 rounded-lg border ${simulationSpeed === s
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
                                <div className="flex items-center gap-2"><Brain className="w-3 h-3 text-teal-300" /> LightGBM Regressor</div>
                                <div className="flex items-center gap-2"><Layers className="w-3 h-3 text-teal-300" /> 160 regions • 16 AFM files</div>
                                <div className="flex items-center gap-2"><Gauge className="w-3 h-3 text-teal-300" /> DTMicroscope physics</div>
                                <div className="flex items-center gap-2"><Bolt className="w-3 h-3 text-teal-300" /> Speed 1–20 µm/s bounds</div>
                                <div className="flex items-center gap-2"><AlertTriangle className="w-3 h-3 text-orange-300" /> Safety checks</div>
                            </div>
                        </div>
                    </div>

                    <div className="col-span-1 lg:col-span-3 space-y-4">
                        <SectionCard title="A) Scan Speed">
                            <div className="space-y-3">
                                <div className="text-[11px] text-gray-400">Total Scan Time</div>
                                <div className="flex items-center gap-3">
                                    <div className="flex-1">
                                        <div className="text-[11px] text-gray-400 mb-1">Traditional</div>
                                        <div className="h-3 bg-white/10 rounded">
                                            <div className="h-3 rounded bg-[#ff8c8c]" style={{ width: "100%" }} />
                                        </div>
                                    </div>
                                    <div className="flex-1">
                                        <div className="text-[11px] text-gray-400 mb-1">SmartScan</div>
                                        <div className="h-3 bg-white/10 rounded">
                                            <div
                                                className="h-3 rounded"
                                                style={{
                                                    width: `${Math.max(20, 100 - timeSaved)}%`,
                                                    backgroundColor: "#76d7c4",
                                                }}
                                            />
                                        </div>
                                    </div>
                                    <div className="text-teal-200 text-xs font-semibold">{timeSaved.toFixed(0)}% faster</div>
                                </div>
                            </div>
                        </SectionCard>

                        <SectionCard title="B) Image Quality">
                            <div className="grid grid-cols-2 gap-4">
                                <MetricCard label="Traditional" value={benchmarkData.traditional.avgQuality.toFixed(2)} accent="orange" />
                                <MetricCard label="SmartScan" value={benchmarkData.smartscan.avgQuality.toFixed(2)} accent="teal" />
                            </div>
                            <div className="text-[11px] text-gray-400 mt-3">Higher is better</div>
                        </SectionCard>

                        <SectionCard title="C) Time per Region">
                            <div className="flex items-center gap-4">
                                <Sparkline
                                    data={(smartData.length ? smartData : Array.from({ length: 10 }, (_, i) => generateScanPoint(i, "adaptive"))).map((p) => p.time)}
                                    color="#76d7c4"
                                    width={500}
                                    height={80}
                                />
                            </div>
                            <div className="text-[11px] text-gray-400 mt-2">SmartScan maintains ~steady times with spikes on complex regions</div>
                        </SectionCard>

                        <SectionCard title="D) Quality Throughout Scan">
                            <div className="flex items-center gap-4">
                                <Sparkline
                                    data={(traditionalData.length ? traditionalData : Array.from({ length: 10 }, (_, i) => generateScanPoint(i, "traditional"))).map((p) => p.quality)}
                                    color="#ff8c8c"
                                    width={500}
                                    height={80}
                                />
                                <Sparkline
                                    data={(smartData.length ? smartData : Array.from({ length: 10 }, (_, i) => generateScanPoint(i, "adaptive"))).map((p) => p.quality)}
                                    color="#76d7c4"
                                    width={500}
                                    height={80}
                                    dashedAt={benchmarkData.smartscan.avgQuality}
                                />
                            </div>
                        </SectionCard>

                        <SectionCard title="E) Real-Time Parameter Adaptation">
                            <Sparkline
                                data={(smartData.length ? smartData : Array.from({ length: 10 }, (_, i) => generateScanPoint(i, "adaptive"))).map((p) => p.speed)}
                                color="#76d7c4"
                                width={500}
                                height={80}
                            />
                            <div className="text-[11px] text-gray-400 mt-2">Speed adapts to region complexity</div>
                        </SectionCard>

                        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                            <MetricCard label="Time Saved" value={`${timeSaved.toFixed(0)}%`} sub="vs Traditional" accent="teal" icon={<TimerReset className="w-4 h-4" />} />
                            <MetricCard label="Quality Score" value={`${benchmarkData.smartscan.avgQuality.toFixed(2)}`} sub={`vs ${benchmarkData.traditional.avgQuality.toFixed(2)} baseline`} accent="green" icon={<Activity className="w-4 h-4" />} />
                            <MetricCard label="Efficiency" value="92%" sub="Drift balanced" accent="teal" icon={<Zap className="w-4 h-4" />} />
                            <MetricCard label="Total Speed" value="37%" sub="Improvement" accent="orange" icon={<Rocket className="w-4 h-4" />} />
                        </div>

                        <div className="bg-[#18181a] border border-white/5 rounded-xl p-5 flex flex-col md:flex-row items-center justify-between gap-4">
                            <div className="flex items-center gap-3">
                                <div className="h-10 w-10 bg-teal-500/10 rounded-full flex items-center justify-center border border-teal-500/20">
                                    <Sparkles className="w-5 h-5 text-teal-400" />
                                </div>
                                <div>
                                    <div className="text-sm font-semibold text-white">Hackathon Achievement</div>
                                    <div className="text-xs text-gray-400">Physics-informed reinforcement learning</div>
                                </div>
                            </div>
                            <div className="flex gap-6 text-xs text-gray-300">
                                <span className="flex items-center gap-2"><Circle className="w-1.5 h-1.5 text-emerald-400" /> 160 Regions</span>
                                <span className="flex items-center gap-2"><Circle className="w-1.5 h-1.5 text-emerald-400" /> 16 Datasets</span>
                                <span className="flex items-center gap-2"><Circle className="w-1.5 h-1.5 text-emerald-400" /> LightGBM</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="text-center text-[10px] text-gray-600 pt-8 pb-4">
                    SmartScan v1.0 • Microscopy Hackathon 2025
                </div>
            </div>
        </div>
    );
};

export default SmartScanDashboard;
