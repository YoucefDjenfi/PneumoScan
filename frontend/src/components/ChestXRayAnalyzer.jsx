import React, { useState, useRef, useEffect, useCallback } from 'react';

/* ─────────────────────────────────────────────────────────────────────────────
   FONTS & GLOBAL STYLES
   Source: original (kept verbatim) + slider/range styles added
───────────────────────────────────────────────────────────────────────────── */
(() => {
  if (document.getElementById('cxa-styles')) return; // hot-reload guard
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href =
    'https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap';
  document.head.appendChild(link);

  const s = document.createElement('style');
  s.id = 'cxa-styles';
  s.textContent = `
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html, body, #root { height: 100%; }
    body { font-family: 'DM Sans', sans-serif; background: #0f1117; color: #e2e8f0;
           -webkit-font-smoothing: antialiased; }
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }

    @keyframes fadeUp   { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
    @keyframes fadeIn   { from{opacity:0} to{opacity:1} }
    @keyframes fadeOut  { from{opacity:1;transform:translateY(0)} to{opacity:0;transform:translateY(-8px)} }
    @keyframes spin     { to{transform:rotate(360deg)} }
    @keyframes sweep    { 0%{top:0%;opacity:.85} 90%{top:100%;opacity:.85} 100%{top:100%;opacity:0} }
    @keyframes pulse-dot{ 0%,100%{opacity:1} 50%{opacity:.25} }
    @keyframes shimmer  { 0%{background-position:200% 0} 100%{background-position:-200% 0} }
    @keyframes toastIn  { from{opacity:0;transform:translateX(16px)} to{opacity:1;transform:translateX(0)} }
    @keyframes toastOut { from{opacity:1;transform:translateX(0)} to{opacity:0;transform:translateX(16px)} }
    @keyframes countUp  { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }

    .fade-up  { animation: fadeUp  .4s cubic-bezier(.4,0,.2,1) forwards; }
    .fade-in  { animation: fadeIn  .3s ease forwards; }
    .mono     { font-family: 'JetBrains Mono', monospace; }

    /* ── Sidebar ── */
    .nav-item {
      display:flex; flex-direction:column; align-items:center; gap:5px;
      padding:11px 0; cursor:pointer; border-radius:10px;
      color:#4b5563; font-size:9.5px; font-weight:600;
      letter-spacing:.05em; text-transform:uppercase;
      transition: color .15s, background .15s; width:100%;
    }
    .nav-item:hover  { color:#94a3b8; background:rgba(148,163,184,.07); }
    .nav-item.active { color:#38bdf8; background:rgba(56,189,248,.1); }

    /* ── Toolbar ── */
    .tool-btn {
      display:flex; align-items:center; gap:6px;
      padding:5px 11px; border-radius:7px;
      color:#94a3b8; font-size:12px; font-weight:500;
      cursor:pointer; transition:background .13s, color .13s;
      border:none; background:transparent; font-family:'DM Sans',sans-serif;
    }
    .tool-btn:hover  { background:rgba(148,163,184,.09); color:#e2e8f0; }
    .tool-btn.active { background:rgba(56,189,248,.14); color:#38bdf8; }

    /* ── Upload zone ── */
    .upload-zone {
      border:1.5px dashed #1e2d3d; border-radius:16px;
      transition:border-color .2s, background .2s; cursor:pointer;
    }
    .upload-zone:hover, .upload-zone.drag-over {
      border-color:#0ea5e9; background:rgba(14,165,233,.03);
    }

    /* ── Cards ── */
    .card    { background:#1a2032; border-radius:14px; border:1px solid rgba(255,255,255,.055); }
    .card-sm { background:#1e2840; border-radius:10px; border:1px solid rgba(255,255,255,.045); }

    /* ── Sweep scanline ── */
    .sweep-line {
      position:absolute; left:0; right:0; height:2px; pointer-events:none; z-index:10;
      background:linear-gradient(90deg,transparent 0%,rgba(56,189,248,.8) 40%,
                 rgba(255,255,255,.9) 50%,rgba(56,189,248,.8) 60%,transparent 100%);
      animation:sweep 1.8s linear infinite;
    }

    /* ── Progress bar ── */
    .progress-animated {
      background: linear-gradient(90deg, #0ea5e9 0%, #38bdf8 50%, #0ea5e9 100%);
      background-size: 200% 100%;
      animation: shimmer 1.4s linear infinite;
    }

    /* ── Badges ── */
    .badge-normal    { background:rgba(34,197,94,.13);  color:#4ade80; border:1px solid rgba(34,197,94,.22); }
    .badge-pneumonia { background:rgba(239,68,68,.13);  color:#f87171; border:1px solid rgba(239,68,68,.22); }

    /* ── Step rows ── */
    .step-row { display:flex; align-items:flex-start; gap:9px; margin-bottom:9px; }
    .step-circle {
      width:18px; height:18px; border-radius:50%; flex-shrink:0; margin-top:1px;
      display:flex; align-items:center; justify-content:center;
      transition: background .2s, border-color .2s;
    }

    /* ── Misc ── */
    .divider { border:none; border-top:1px solid rgba(255,255,255,.055); }

    .tag {
      display:inline-flex; align-items:center;
      padding:2px 8px; border-radius:99px;
      font-size:10.5px; font-weight:600; letter-spacing:.02em;
    }

    .search-input {
      background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.08);
      border-radius:9px; color:#e2e8f0; outline:none;
      font-family:'DM Sans',sans-serif; transition:border-color .15s;
    }
    .search-input::placeholder { color:#3d4f63; }
    .search-input:focus { border-color:rgba(14,165,233,.45); background:rgba(14,165,233,.04); }

    .new-patient-btn {
      background:#0ea5e9; border:none; cursor:pointer; color:white;
      font-family:'DM Sans',sans-serif; font-weight:600; font-size:13px;
      border-radius:9px; padding:8px 16px;
      display:flex; align-items:center; gap:6px;
      transition:background .15s, box-shadow .15s; white-space:nowrap;
    }
    .new-patient-btn:hover { background:#38bdf8; box-shadow:0 2px 16px rgba(14,165,233,.35); }

    .stat-card {
      background:#1a2032; border-radius:12px;
      border:1px solid rgba(255,255,255,.055); padding:14px 16px;
      transition:border-color .2s, transform .2s;
    }
    .stat-card:hover { border-color:rgba(255,255,255,.1); transform:translateY(-1px); }
    .stat-card.flash { border-color:rgba(56,189,248,.5); }

    /* ── Toast system (from original) ── */
    .toast-wrap {
      position:fixed; bottom:28px; right:28px; z-index:9999;
      display:flex; flex-direction:column; gap:10px; pointer-events:none;
    }
    .toast {
      display:flex; align-items:center; gap:10px;
      background:#1e2840; border:1px solid rgba(255,255,255,.1);
      border-radius:10px; padding:10px 16px;
      font-size:13px; font-weight:500; color:#e2e8f0;
      box-shadow:0 8px 32px rgba(0,0,0,.45);
      pointer-events:all;
    }
    .toast.entering { animation: toastIn  .25s cubic-bezier(.4,0,.2,1) forwards; }
    .toast.leaving  { animation: toastOut .25s cubic-bezier(.4,0,.2,1) forwards; }

    /* ── Count-up flash ── */
    .count-flash { animation: countUp .35s ease forwards; }

    /* ── Range inputs (shared style) ── */
    input[type=range] {
      -webkit-appearance: none; appearance: none;
      width: 100%; height: 4px; border-radius: 2px;
      background: rgba(255,255,255,.08); outline: none; cursor: pointer;
    }
    input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none; appearance: none;
      width: 14px; height: 14px; border-radius: 50%;
      background: #0ea5e9; cursor: pointer;
      box-shadow: 0 0 0 2px rgba(14,165,233,.25);
      transition: box-shadow .15s;
    }
    input[type=range]::-webkit-slider-thumb:hover {
      box-shadow: 0 0 0 4px rgba(14,165,233,.35);
    }
    input[type=range]:disabled { opacity: .35; cursor: not-allowed; }
    input[type=range]:disabled::-webkit-slider-thumb { cursor: not-allowed; }

    /* ── Crosshair cursor on canvas wrapper ── */
    .canvas-viewport { cursor: crosshair; }
  `;
  document.head.appendChild(s);
})();

/* ─────────────────────────────────────────────────────────────────────────────
   ICONS  (original set + info/error kept for toast system)
───────────────────────────────────────────────────────────────────────────── */
const Ic = ({ d, size = 18, stroke = 'currentColor', fill = 'none', sw = 1.75 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke={stroke}
    strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
    {(Array.isArray(d) ? d : [d]).map((p, i) => <path key={i} d={p} />)}
  </svg>
);

const IC = {
  home:     'M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z',
  patients: ['M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2', 'M23 21v-2a4 4 0 00-3-3.87', 'M16 3.13a4 4 0 010 7.75'],
  chart:    ['M18 20V10', 'M12 20V4', 'M6 20v-6'],
  settings: ['M12 15a3 3 0 100-6 3 3 0 000 6z', 'M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z'],
  upload:   ['M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4', 'M17 8l-5-5-5 5', 'M12 3v12'],
  pan:      ['M5 9l-3 3 3 3', 'M9 5l3-3 3 3', 'M15 19l-3 3-3-3', 'M19 9l3 3-3 3', 'M2 12h20', 'M12 2v20'],
  zoom:     ['M11 19a8 8 0 100-16 8 8 0 000 16z', 'M21 21l-4.35-4.35', 'M11 8v6', 'M8 11h6'],
  measure:  ['M2 12h20', 'M12 2v20'],
  invert:   ['M12 2a10 10 0 100 20', 'M12 2v20'],
  search:   ['M21 21l-4.35-4.35', 'M17 11A6 6 0 115 11a6 6 0 0112 0'],
  bell:     ['M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9', 'M13.73 21a2 2 0 01-3.46 0'],
  plus:     ['M12 5v14', 'M5 12h14'],
  reset:    ['M1 4v6h6', 'M3.51 15a9 9 0 102.13-9.36L1 10'],
  xray:     ['M8 3H5a2 2 0 00-2 2v3', 'M21 8V5a2 2 0 00-2-2h-3', 'M3 16v3a2 2 0 002 2h3', 'M16 21h3a2 2 0 002-2v-3', 'M7 12h10', 'M12 7v10'],
  activity: 'M22 12h-4l-3 9L9 3l-3 9H2',
  check:    'M20 6L9 17l-5-5',
  warn:     ['M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z', 'M12 9v4', 'M12 17h.01'],
  clock:    ['M12 22a10 10 0 100-20 10 10 0 000 20', 'M12 6v6l4 2'],
  info:     ['M12 22a10 10 0 100-20 10 10 0 000 20', 'M12 16v-4', 'M12 8h.01'],
  error:    ['M12 22a10 10 0 100-20 10 10 0 000 20', 'M15 9l-6 6', 'M9 9l6 6'],
  contrast: ['M12 2a10 10 0 100 20', 'M12 2v20', 'M12 7a5 5 0 010 10'],
};

/* ─────────────────────────────────────────────────────────────────────────────
   HEATMAP COLOR MAP  (jet-like: blue→cyan→green→yellow→red)
───────────────────────────────────────────────────────────────────────────── */
function colorMap(v) {
  let r, g, b;
  if      (v < .25) { const t = v / .25;         r = 0;               g = Math.floor(255 * t); b = 255; }
  else if (v < .5)  { const t = (v - .25) / .25;  r = 0;               g = 255; b = Math.floor(255 * (1 - t)); }
  else if (v < .75) { const t = (v - .5)  / .25;  r = Math.floor(255 * t); g = 255; b = 0; }
  else              { const t = (v - .75) / .25;  r = 255; g = Math.floor(255 * (1 - t)); b = 0; }
  return [r, g, b];
}

/* ─────────────────────────────────────────────────────────────────────────────
   CONSTANTS
───────────────────────────────────────────────────────────────────────────── */
const STATUS_STEPS = [
  { msg: 'Initialising inference pipeline',       pct: 5  },
  { msg: 'Preprocessing image tensor (224×224)',  pct: 22 },
  { msg: 'Running ResNet-34 forward pass',         pct: 48 },
  { msg: 'Generating Grad-CAM activation map',    pct: 72 },
  { msg: 'Computing confidence scores',            pct: 90 },
];

const CANVAS_SIZE = 512;

/* ─────────────────────────────────────────────────────────────────────────────
   TOAST SYSTEM  (from original — proper enter/leave animation)
───────────────────────────────────────────────────────────────────────────── */
let _toastId   = 0;
let _setToasts = null;

function toast(message, type = 'info', duration = 2800) {
  if (!_setToasts) return;
  const id = ++_toastId;
  _setToasts(prev => [...prev, { id, message, type, leaving: false }]);
  setTimeout(() => {
    _setToasts(prev => prev.map(t => t.id === id ? { ...t, leaving: true } : t));
    setTimeout(() => _setToasts(prev => prev.filter(t => t.id !== id)), 280);
  }, duration);
}

function ToastProvider() {
  const [toasts, setToasts] = useState([]);
  _setToasts = setToasts;
  const ICON  = { info: IC.info,  error: IC.error,  success: IC.check };
  const COLOR = { info: '#38bdf8', error: '#f87171', success: '#4ade80' };
  return (
    <div className="toast-wrap">
      {toasts.map(t => (
        <div key={t.id} className={`toast ${t.leaving ? 'leaving' : 'entering'}`}>
          <Ic d={ICON[t.type] || IC.info} size={16} stroke={COLOR[t.type] || '#38bdf8'} sw={2} />
          {t.message}
        </div>
      ))}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   SIDEBAR  (original — toast on inactive items restored)
───────────────────────────────────────────────────────────────────────────── */
function Sidebar() {
  const items = [
    { id: 'home',     label: 'Dashboard', icon: 'home'     },
    { id: 'patients', label: 'Patients',  icon: 'patients' },
    { id: 'chart',    label: 'Analytics', icon: 'chart'    },
    { id: 'settings', label: 'Settings',  icon: 'settings' },
  ];
  return (
    <aside style={{
      width: 72, flexShrink: 0, background: '#13192a',
      borderRight: '1px solid rgba(255,255,255,.055)',
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      paddingTop: 18, paddingBottom: 18, gap: 2,
    }}>
      <div style={{
        width: 40, height: 40, borderRadius: 11, marginBottom: 26, flexShrink: 0,
        background: 'linear-gradient(135deg,#0ea5e9 0%,#6366f1 100%)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        boxShadow: '0 4px 16px rgba(14,165,233,.32)',
      }}>
        <Ic d={IC.xray} size={21} stroke="#fff" sw={2} />
      </div>
      {items.map(item => (
        <div key={item.id}
          className={`nav-item ${item.id === 'home' ? 'active' : ''}`}
          style={{ width: 58 }}
          onClick={() => item.id !== 'home' && toast('Feature coming soon', 'info')}
        >
          <Ic d={IC[item.icon]} size={19} />
          <span>{item.label}</span>
        </div>
      ))}
    </aside>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   TOP HEADER  (original — unchanged)
───────────────────────────────────────────────────────────────────────────── */
function TopHeader() {
  return (
    <header style={{
      height: 60, background: '#13192a',
      borderBottom: '1px solid rgba(255,255,255,.055)',
      display: 'flex', alignItems: 'center',
      paddingLeft: 24, paddingRight: 20, gap: 14, flexShrink: 0,
    }}>
      <nav style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 13, color: '#4b5563', flexShrink: 0 }}>
        <span>Radiology</span>
        <Ic d="M9 18l6-6-6-6" size={14} stroke="#374151" />
        <span style={{ color: '#e2e8f0', fontWeight: 500 }}>AI Workstation</span>
      </nav>

      <div style={{ flex: 1, maxWidth: 380, position: 'relative' }}>
        <div style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }}>
          <Ic d={IC.search} size={14} stroke="#3d4f63" />
        </div>
        <input className="search-input"
          placeholder="Search patients, accession numbers…"
          style={{ width: '100%', padding: '8px 12px 8px 32px', fontSize: 13 }}
        />
      </div>

      <div style={{ flex: 1 }} />

      <div style={{
        width: 36, height: 36, borderRadius: 9, cursor: 'pointer',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'rgba(255,255,255,.045)', border: '1px solid rgba(255,255,255,.07)',
        position: 'relative',
      }}>
        <Ic d={IC.bell} size={17} stroke="#64748b" />
        <div style={{
          position: 'absolute', top: 8, right: 8, width: 6, height: 6,
          borderRadius: '50%', background: '#ef4444', border: '1.5px solid #13192a',
          animation: 'pulse-dot 2.5s ease-in-out infinite',
        }} />
      </div>

      <div style={{ width: 1, height: 28, background: 'rgba(255,255,255,.07)' }} />

      <div style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer' }}>
        <div style={{
          width: 34, height: 34, borderRadius: 9, flexShrink: 0,
          background: 'linear-gradient(135deg,#6366f1,#0ea5e9)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 12, fontWeight: 700, color: '#fff',
        }}>DJ</div>
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: '#f1f5f9', lineHeight: 1.25 }}>Dr. Djenfi</div>
          <div style={{ fontSize: 10.5, color: '#4b5563', lineHeight: 1 }}>Radiologist</div>
        </div>
      </div>

      <button className="new-patient-btn"
        onClick={() => toast('Patient registration coming soon', 'info')}>
        <Ic d={IC.plus} size={15} stroke="#fff" sw={2.5} />
        New Patient
      </button>
    </header>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   RIGHT PANEL
   Merges: original layout + latest's Clinical AI Overlays section
   New:    Image Contrast (Windowing) slider
───────────────────────────────────────────────────────────────────────────── */
function RightPanel({
  state, result, progress, statusIdx,
  showGradCam, setShowGradCam,
  heatmapOpacity, setHeatmapOpacity,
  heatmapBlur,    setHeatmapBlur,
  windowContrast, setWindowContrast,
}) {
  const isPneu = result?.diagnosis === 'Pneumonia';
  const pct    = result ? result.probability * 100 : 0;

  const SectionLabel = ({ children }) => (
    <div style={{
      fontSize: 10.5, fontWeight: 700, color: '#374151',
      letterSpacing: '.07em', textTransform: 'uppercase', marginBottom: 10,
    }}>
      {children}
    </div>
  );

  const DataRow = ({ label, value, mono }) => (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
      <span style={{ fontSize: 12, color: '#4b5563' }}>{label}</span>
      <span style={{
        fontSize: 12, fontWeight: 500, color: '#94a3b8',
        fontFamily: mono ? "'JetBrains Mono',monospace" : undefined,
      }}>{value}</span>
    </div>
  );

  const SliderRow = ({ label, value, display, min, max, step, onChange, disabled }) => (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#94a3b8', marginBottom: 8 }}>
        <span>{label}</span>
        <span className="mono">{display}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        disabled={!!disabled}
      />
    </div>
  );

  const canEdit = state === 'result';

  return (
    <aside style={{
      width: 296, flexShrink: 0, background: '#13192a',
      borderLeft: '1px solid rgba(255,255,255,.055)',
      display: 'flex', flexDirection: 'column',
      overflowY: 'auto', padding: '20px 16px', gap: 18,
    }}>

      {/* ── Patient Demographics ── */}
      <section>
        <SectionLabel>Patient Demographics</SectionLabel>
        <div className="card" style={{ padding: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
            <div style={{
              width: 42, height: 42, borderRadius: 10, flexShrink: 0,
              background: 'linear-gradient(135deg,#1e3a5f,#0c4a6e)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 15, fontWeight: 700, color: '#7dd3fc',
            }}>JD</div>
            <div>
              <div style={{ fontSize: 14.5, fontWeight: 600, color: '#f1f5f9', lineHeight: 1.2 }}>John Doe</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 5 }}>
                <span className="tag" style={{ background: 'rgba(56,189,248,.1)',  color: '#38bdf8' }}>In Review</span>
                <span className="tag" style={{ background: 'rgba(99,102,241,.1)', color: '#a5b4fc' }}>Chest PA</span>
              </div>
            </div>
          </div>
          <hr className="divider" style={{ marginBottom: 12 }} />
          <DataRow label="Patient ID"  value="#84729"       mono />
          <DataRow label="Age"         value="45 years" />
          <DataRow label="Sex"         value="Male" />
          <DataRow label="DOB"         value="12 Mar 1980" />
          <DataRow label="Referring"   value="Dr. Ahmed" />
          <DataRow label="Study Date"  value={new Date().toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })} />
          <DataRow label="Accession"   value="CXR-2025-0847" mono />
          <DataRow label="Priority"    value="Routine" />
        </div>
      </section>

      {/* ── AI Inference Engine ── */}
      <section>
        <SectionLabel>AI Inference Engine</SectionLabel>
        <div className="card" style={{ padding: 16 }}>
          {/* Status pill */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14,
            padding: '8px 10px', borderRadius: 8,
            background: state === 'result' ? 'rgba(34,197,94,.07)' : state === 'idle' ? 'rgba(255,255,255,.03)' : 'rgba(245,158,11,.06)',
            border: `1px solid ${state === 'result' ? 'rgba(34,197,94,.15)' : state === 'idle' ? 'rgba(255,255,255,.06)' : 'rgba(245,158,11,.15)'}`,
          }}>
            <div style={{
              width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
              background: state === 'result' ? '#22c55e' : state === 'idle' ? '#374151' : '#f59e0b',
              animation: state === 'thinking' ? 'pulse-dot 1.1s ease-in-out infinite' : 'none',
              boxShadow:  state === 'thinking' ? '0 0 7px rgba(245,158,11,.7)' : 'none',
            }} />
            <span style={{
              fontSize: 12, fontWeight: 500,
              color: state === 'result' ? '#4ade80' : state === 'idle' ? '#4b5563' : '#fbbf24',
            }}>
              {state === 'idle'        ? 'Standby — awaiting input'
               : state === 'processing' ? 'Connecting to API…'
               : state === 'thinking'   ? (STATUS_STEPS[statusIdx]?.msg ?? '') + '…'
               : 'Analysis complete'}
            </span>
          </div>

          {/* Progress bar + step tracker */}
          {(state === 'thinking' || state === 'result') && (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                <span style={{ fontSize: 11, color: '#374151' }}>Progress</span>
                <span className="mono" style={{ fontSize: 11, color: '#64748b' }}>
                  {state === 'result' ? '100' : Math.round(progress)}%
                </span>
              </div>
              <div style={{ background: 'rgba(255,255,255,.06)', borderRadius: 99, height: 5, overflow: 'hidden', marginBottom: 14 }}>
                <div
                  className={state === 'thinking' ? 'progress-animated' : ''}
                  style={{
                    height: '100%', borderRadius: 99,
                    width: `${state === 'result' ? 100 : progress}%`,
                    background: state === 'result' ? '#22c55e' : undefined,
                    transition: state === 'result' ? 'width .5s ease' : 'width .12s linear',
                  }}
                />
              </div>

              {STATUS_STEPS.map((step, i) => {
                const done    = state === 'result' || i < statusIdx;
                const current = state === 'thinking' && i === statusIdx;
                return (
                  <div key={i} className="step-row">
                    <div className="step-circle" style={{
                      background: done ? 'rgba(34,197,94,.16)' : current ? 'rgba(245,158,11,.14)' : 'rgba(255,255,255,.04)',
                      border: `1px solid ${done ? 'rgba(34,197,94,.35)' : current ? 'rgba(245,158,11,.35)' : 'rgba(255,255,255,.07)'}`,
                    }}>
                      {done
                        ? <Ic d={IC.check} size={9} stroke="#22c55e" sw={2.5} />
                        : current
                          ? <div style={{ width: 5, height: 5, borderRadius: '50%', background: '#f59e0b', animation: 'pulse-dot 1s ease-in-out infinite' }} />
                          : null}
                    </div>
                    <span style={{
                      fontSize: 11.5, lineHeight: 1.45,
                      color: done ? '#374151' : current ? '#e2e8f0' : '#2d3748',
                      fontWeight: current ? 500 : 400,
                    }}>{step.msg}</span>
                  </div>
                );
              })}
            </>
          )}
        </div>
      </section>

      {/* ── Diagnostic Result Card ── */}
      {state === 'result' && result && (
        <section className="fade-up">
          <SectionLabel>Diagnostic Result</SectionLabel>
          <div className="card" style={{
            padding: 18,
            borderLeft: `3px solid ${isPneu ? '#ef4444' : '#22c55e'}`,
            boxShadow: isPneu
              ? '0 4px 28px rgba(239,68,68,.1),0 1px 3px rgba(0,0,0,.3)'
              : '0 4px 28px rgba(34,197,94,.09),0 1px 3px rgba(0,0,0,.3)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{
                  width: 38, height: 38, borderRadius: 9,
                  background: isPneu ? 'rgba(239,68,68,.13)' : 'rgba(34,197,94,.11)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  <Ic d={isPneu ? IC.warn : IC.check} size={19} stroke={isPneu ? '#f87171' : '#4ade80'} sw={2} />
                </div>
                <div>
                  <div style={{ fontSize: 18, fontWeight: 700, lineHeight: 1, color: isPneu ? '#f87171' : '#4ade80' }}>
                    {result.diagnosis}
                  </div>
                  <div style={{ fontSize: 11, color: '#4b5563', marginTop: 3 }}>
                    {isPneu ? 'Pathology Detected' : 'No Pathology Detected'}
                  </div>
                </div>
              </div>
              <span className={`tag ${isPneu ? 'badge-pneumonia' : 'badge-normal'}`}>
                {isPneu ? '⚠ Alert' : '✓ Clear'}
              </span>
            </div>

            <hr className="divider" style={{ marginBottom: 14 }} />

            {/* Confidence segmented bar */}
            <div style={{ marginBottom: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                <span style={{ fontSize: 12, color: '#4b5563' }}>Confidence Score</span>
                <span className="mono" style={{ fontSize: 13, fontWeight: 600, color: isPneu ? '#f87171' : '#4ade80' }}>
                  {pct.toFixed(2)}%
                </span>
              </div>
              <div style={{ display: 'flex', gap: 2.5 }}>
                {Array.from({ length: 20 }).map((_, i) => (
                  <div key={i} style={{
                    flex: 1, height: 7, borderRadius: 3,
                    background: (i + 1) * 5 <= pct ? (isPneu ? '#ef4444' : '#22c55e') : 'rgba(255,255,255,.06)',
                    transition: `background .04s ${i * 35}ms`,
                  }} />
                ))}
              </div>
            </div>

            {/* Class probability breakdown */}
            {[
              { label: 'Pneumonia', val: result.probability,      color: '#ef4444' },
              { label: 'Normal',    val: 1 - result.probability,  color: '#22c55e' },
            ].map(({ label, val, color }) => (
              <div key={label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 7 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                  <div style={{ width: 9, height: 9, borderRadius: 3, background: color, flexShrink: 0 }} />
                  <span style={{ fontSize: 12, color: '#64748b' }}>{label}</span>
                </div>
                <span className="mono" style={{ fontSize: 12, color: '#94a3b8' }}>{(val * 100).toFixed(2)}%</span>
              </div>
            ))}

            <div style={{
              marginTop: 12, padding: '9px 11px', borderRadius: 9,
              background: 'rgba(255,255,255,.03)', border: '1px solid rgba(255,255,255,.06)',
              fontSize: 11, color: '#4b5563', lineHeight: 1.55,
            }}>
              ⓘ AI-assisted analysis only. Must be reviewed by a licensed radiologist before clinical use.
            </div>
          </div>
        </section>
      )}

      {/* ── Clinical AI Overlays (from latest) ── */}
      <section>
        <SectionLabel>Clinical AI Overlays</SectionLabel>
        <div className="card" style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Grad-CAM toggle */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: 12, color: '#cbd5e1', fontWeight: 500 }}>Pathology Overlay (Grad-CAM)</span>
            <button
              onClick={() => setShowGradCam(prev => !prev)}
              style={{
                width: 38, height: 22, borderRadius: 11, padding: 2, border: 'none',
                background: showGradCam ? '#0ea5e9' : '#334155',
                display: 'flex', alignItems: 'center',
                justifyContent: showGradCam ? 'flex-end' : 'flex-start',
                cursor: 'pointer', transition: 'background .2s, justify-content .2s',
              }}
            >
              <div style={{ width: 18, height: 18, borderRadius: '50%', background: '#fff', transition: 'all .2s' }} />
            </button>
          </div>

          <hr className="divider" />

          {/* Heatmap opacity — default 0.50 */}
          <SliderRow
            label="Blended Opacity"
            value={heatmapOpacity}
            display={`${Math.round(heatmapOpacity * 100)}%`}
            min={0} max={1} step={0.05}
            onChange={setHeatmapOpacity}
            disabled={!showGradCam || !canEdit}
          />

          {/* Localization blur — default 0 */}
          <SliderRow
            label="Localization Softness (Blur)"
            value={heatmapBlur}
            display={`${heatmapBlur}px`}
            min={0} max={8} step={1}
            onChange={v => setHeatmapBlur(Math.round(v))}
            disabled={!showGradCam || !canEdit}
          />
        </div>
      </section>

      {/* ── NEW: Image Contrast / DICOM Windowing ── */}
      <section>
        <SectionLabel>DICOM Windowing</SectionLabel>
        <div className="card" style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
            <div style={{
              width: 30, height: 30, borderRadius: 8,
              background: 'rgba(14,165,233,.1)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Ic d={IC.contrast} size={16} stroke="#38bdf8" />
            </div>
            <div>
              <div style={{ fontSize: 12, fontWeight: 500, color: '#cbd5e1' }}>Image Contrast</div>
              <div style={{ fontSize: 10.5, color: '#4b5563' }}>Simulates radiologist window levelling</div>
            </div>
          </div>

          <SliderRow
            label="Contrast Level"
            value={windowContrast}
            display={`${windowContrast.toFixed(2)}×`}
            min={0.5} max={3.0} step={0.05}
            onChange={setWindowContrast}
            disabled={state === 'idle' || state === 'processing'}
          />

          <button
            onClick={() => { setWindowContrast(1.0); toast('Contrast reset to default', 'info', 1600); }}
            style={{
              fontSize: 11, fontWeight: 500, color: '#64748b', background: 'rgba(255,255,255,.04)',
              border: '1px solid rgba(255,255,255,.07)', borderRadius: 7,
              padding: '5px 10px', cursor: 'pointer', textAlign: 'center',
              transition: 'background .15s, color .15s', fontFamily: "'DM Sans',sans-serif",
            }}
            onMouseEnter={e => { e.target.style.background = 'rgba(255,255,255,.08)'; e.target.style.color = '#94a3b8'; }}
            onMouseLeave={e => { e.target.style.background = 'rgba(255,255,255,.04)'; e.target.style.color = '#64748b'; }}
          >
            Reset to Default (1.0×)
          </button>
        </div>
      </section>

      {/* ── Model Info ── */}
      <section>
        <SectionLabel>Model Information</SectionLabel>
        <div className="card-sm" style={{ padding: 14 }}>
          {[
            { k: 'Architecture', v: 'ResNet-34'     },
            { k: 'Parameters',   v: '21.8M'         },
            { k: 'Input Size',   v: '224 × 224'     },
            { k: 'Dataset',      v: 'NIH ChestX-14' },
            { k: 'XAI Method',   v: 'Grad-CAM'      },
            { k: 'Val. Acc.',    v: '95.3%'         },
          ].map(({ k, v }) => (
            <div key={k} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 7 }}>
              <span style={{ fontSize: 11, color: '#374151' }}>{k}</span>
              <span className="mono" style={{ fontSize: 11, color: '#38bdf8' }}>{v}</span>
            </div>
          ))}
        </div>
      </section>
    </aside>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   MAIN COMPONENT
───────────────────────────────────────────────────────────────────────────── */
export default function ChestXRayAnalyzer() {

  /* ── Core app state ─────────────────────────────────────────────────────── */
  const [state,      setState]      = useState('idle');   // idle|processing|thinking|result
  const [file,       setFile]       = useState(null);
  const [result,     setResult]     = useState(null);
  const [progress,   setProgress]   = useState(0);
  const [statusIdx,  setStatusIdx]  = useState(0);
  const [dragging,   setDragging]   = useState(false);

  /* ── Session stat ───────────────────────────────────────────────────────── */
  const [studiesCount, setStudiesCount] = useState(14);
  const [studiesFlash, setStudiesFlash] = useState(false);

  /* ── Viewer / overlay controls ──────────────────────────────────────────── */
  const [activeTool,     setActiveTool]     = useState('pan');
  const [isInverted,     setIsInverted]     = useState(false);
  const [showGradCam,    setShowGradCam]    = useState(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.50);  // req: default 0.50
  const [heatmapBlur,    setHeatmapBlur]    = useState(0);     // req: default 0
  const [windowContrast, setWindowContrast] = useState(1.0);   // DICOM windowing

  /* ── Segmentation status (polled from /health on mount) ─────────────────── */
  const [segmenterReady, setSegmenterReady] = useState(null);  // null=unknown, true/false

  /* ── Crosshair state ────────────────────────────────────────────────────── */
  const [crosshair, setCrosshair] = useState(null); // { x, y } in canvas-relative px | null

  /* ── Refs ───────────────────────────────────────────────────────────────── */
  const canvasRef      = useRef(null);
  const imageRef       = useRef(null);
  const offscreenRef   = useRef(null);
  const heatmapRef     = useRef(null);
  const animFrameRef   = useRef(null);
  const startTimeRef   = useRef(null);
  const viewportRef    = useRef(null);

  /*
   * FIX: stale-closure bug in rAF loop.
   * drawCanvas is recreated whenever its dependencies change (showGradCam,
   * heatmapOpacity, heatmapBlur, windowContrast, isInverted).
   * The rAF loop closes over drawCanvasRef.current so it always calls
   * the LATEST version — no stale closure, no missing props.
   */
  const drawCanvasRef = useRef(null);

  /* ────────────────────────────────────────────────────────────────────────
     HEALTH CHECK — poll /health once on mount to learn whether the
     lung-segmentation U-Net loaded successfully on the backend.
     Result is displayed in the canvas HUD badge so the radiologist knows
     whether shortcut-learning bias-reduction is active.
  ──────────────────────────────────────────────────────────────────────── */
  useEffect(() => {
    fetch('http://127.0.0.1:8000/health')
      .then(r => r.json())
      .then(d => setSegmenterReady(!!d.segmenter_ready))
      .catch(() => setSegmenterReady(false));
  }, []);

  /* ────────────────────────────────────────────────────────────────────────
     HEATMAP PIXEL WRITER  (pure — no state deps, stable ref)
  ──────────────────────────────────────────────────────────────────────── */
  const renderHeatmap = useCallback((offCtx, heatmap, threshold, opacity) => {
    if (!heatmap) return;
    const id = offCtx.createImageData(224, 224);
    const d  = id.data;
    for (let y = 0; y < 224; y++) {
      for (let x = 0; x < 224; x++) {
        const v = heatmap[y][x];
        const i = (y * 224 + x) * 4;
        if (v >= threshold) {
          const [r, g, b] = colorMap(v);
          d[i] = r; d[i + 1] = g; d[i + 2] = b; d[i + 3] = Math.floor(opacity * 255);
        } else {
          d[i] = d[i + 1] = d[i + 2] = d[i + 3] = 0;
        }
      }
    }
    offCtx.putImageData(id, 0, 0);
  }, []);

  /* ────────────────────────────────────────────────────────────────────────
     DRAW CANVAS
     Depends on: showGradCam, heatmapOpacity, heatmapBlur, windowContrast.
     Kept as a regular function (not useCallback) so it re-creates on every
     render — drawCanvasRef.current always points to the freshest version.
  ──────────────────────────────────────────────────────────────────────── */
  const drawCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas || !imageRef.current) return;
    const ctx = canvas.getContext('2d');
    canvas.width  = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // 1. Base X-ray with DICOM windowing via canvas filter
    ctx.save();
    ctx.filter = `contrast(${windowContrast})`;
    ctx.drawImage(imageRef.current, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.restore();

    // 2. Grad-CAM heatmap composite (only when enabled and heatmap exists)
    if (offscreenRef.current && showGradCam) {
      ctx.save();
      ctx.globalAlpha             = heatmapOpacity;
      ctx.globalCompositeOperation = 'screen';
      if (heatmapBlur > 0) ctx.filter = `blur(${heatmapBlur}px)`;
      ctx.drawImage(offscreenRef.current, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
      ctx.restore();
    }
  };

  // Keep ref in sync every render — this is the stale-closure fix
  drawCanvasRef.current = drawCanvas;

  /* ────────────────────────────────────────────────────────────────────────
     FILE INGESTION
  ──────────────────────────────────────────────────────────────────────── */
  const ingestFile = useCallback((f) => {
    if (!f || !f.type.startsWith('image/')) {
      toast('Please upload a valid image file (PNG, JPG)', 'error');
      return;
    }
    setFile(f);
    setResult(null);
    setProgress(0);
    setStatusIdx(0);
    setIsInverted(false);
    setActiveTool('pan');
    setCrosshair(null);

    const reader = new FileReader();
    reader.onload = (ev) => {
      const img = new Image();
      img.onload = () => {
        imageRef.current     = img;
        offscreenRef.current = null;
        heatmapRef.current   = null;
        analyzeXray(f);
      };
      img.src = ev.target.result;
    };
    reader.readAsDataURL(f);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleFile      = (e) => { ingestFile(e.target.files[0]); e.target.value = ''; };
  const handleDrop      = (e) => { e.preventDefault(); setDragging(false); ingestFile(e.dataTransfer.files[0]); };
  const handleDragOver  = (e) => { e.preventDefault(); setDragging(true); };
  const handleDragLeave = ()  => setDragging(false);

  /* ────────────────────────────────────────────────────────────────────────
     API CALL  (original robust version with proper error toast)
  ──────────────────────────────────────────────────────────────────────── */
  const analyzeXray = async (f) => {
    setState('processing');
    const fd = new FormData();
    fd.append('file', f); // key matches FastAPI: file: UploadFile = File(...)

    try {
      const res = await fetch('http://127.0.0.1:8000/predict', { method: 'POST', body: fd });
      if (!res.ok) {
        const txt = await res.text().catch(() => res.statusText);
        throw new Error(`Server ${res.status}: ${txt}`);
      }
      const data = await res.json();

      setResult(data);
      heatmapRef.current = data.heatmap;

      // Req: auto-disable Grad-CAM if backend returns null heatmap
      if (!data.heatmap || data.heatmap.length === 0) {
        setShowGradCam(false);
        toast('No activation map returned — Grad-CAM disabled', 'info');
      } else {
        setShowGradCam(true);
      }

      setState('thinking');
    } catch (err) {
      console.error('[PneumoScan]', err);
      toast(`API error — ${err.message}`, 'error', 4500);
      setState('idle');
    }
  };

  /* ────────────────────────────────────────────────────────────────────────
     4-SECOND THINKING / ANIMATION PHASE
     FIX: rAF loop calls drawCanvasRef.current() — always the fresh closure.
  ──────────────────────────────────────────────────────────────────────── */
  const startThinking = useCallback(() => {
    startTimeRef.current = performance.now();

    // Step indicator timers
    const stepTimers = STATUS_STEPS.map(({ pct }, i) =>
      setTimeout(() => setStatusIdx(i), (pct / 100) * 3800)
    );

    // Completion at 4 000 ms
    const doneTimer = setTimeout(() => {
      setState('result');
      setProgress(100);
      cancelAnimationFrame(animFrameRef.current);
      setStudiesCount(n => n + 1);
      setStudiesFlash(true);
      setTimeout(() => setStudiesFlash(false), 600);
      toast('Analysis complete', 'success');
    }, 4000);

    // 224×224 offscreen canvas for heatmap
    const off = document.createElement('canvas');
    off.width = 224; off.height = 224;
    offscreenRef.current = off;
    const offCtx = off.getContext('2d');

    // rAF loop — threshold shrinks 0.9→0, opacity rises 0.15→0.85
    const animate = (now) => {
      const t = Math.min((now - startTimeRef.current) / 4000, 1);
      setProgress(t * 100);
      renderHeatmap(offCtx, heatmapRef.current, 0.9 - t * 0.9, 0.15 + t * 0.7);
      drawCanvasRef.current(); // ← STALE CLOSURE FIX: always the latest drawCanvas
      animFrameRef.current = requestAnimationFrame(animate);
    };
    animFrameRef.current = requestAnimationFrame(animate);

    return () => {
      stepTimers.forEach(clearTimeout);
      clearTimeout(doneTimer);
      cancelAnimationFrame(animFrameRef.current);
    };
  }, [renderHeatmap]); // renderHeatmap is stable; drawCanvas via ref — no dep needed

  useEffect(() => {
    if (state === 'thinking') return startThinking();
  }, [state, startThinking]);

  // Redraw when overlay controls change (result state)
  useEffect(() => {
    if (state === 'result' || state === 'thinking') drawCanvasRef.current();
  }, [state, showGradCam, heatmapOpacity, heatmapBlur, windowContrast]);

  /* ────────────────────────────────────────────────────────────────────────
     TOOLBAR  (original: toast for Pan/Zoom/Measure, real invert filter)
  ──────────────────────────────────────────────────────────────────────── */
  const handleToolClick = (toolId) => {
    if (toolId === 'invert') {
      const next = !isInverted;
      setIsInverted(next);
      setActiveTool('invert');
      toast(next ? 'Image colours inverted' : 'Colours restored', 'info', 1800);
    } else {
      setActiveTool(toolId);
      const labels = { pan: 'Pan', zoom: 'Zoom', measure: 'Measure' };
      toast(`${labels[toolId]} — feature coming soon`, 'info', 2200);
    }
  };

  /* ────────────────────────────────────────────────────────────────────────
     CROSSHAIR  — NEW "absolute cinema" feature
     Mouse position is tracked relative to the canvas element and stored
     as {x, y} in CSS pixel space (0–CANVAS_SIZE). An SVG overlay draws
     a subtle clinical crosshair at that position.
  ──────────────────────────────────────────────────────────────────────── */
  const handleCanvasMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const scaleX = CANVAS_SIZE / rect.width;
    const scaleY = CANVAS_SIZE / rect.height;
    setCrosshair({
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top)  * scaleY,
    });
  };

  const handleCanvasMouseLeave = () => setCrosshair(null);

  /* ────────────────────────────────────────────────────────────────────────
     RESET
  ──────────────────────────────────────────────────────────────────────── */
  const reset = () => {
    cancelAnimationFrame(animFrameRef.current);
    setState('idle');
    setResult(null);
    setFile(null);
    setProgress(0);
    setStatusIdx(0);
    setIsInverted(false);
    setActiveTool('pan');
    setCrosshair(null);
    setWindowContrast(1.0);
    imageRef.current     = null;
    offscreenRef.current = null;
    heatmapRef.current   = null;
  };

  /* ────────────────────────────────────────────────────────────────────────
     STAT CARDS  — "Studies Today" is live
  ──────────────────────────────────────────────────────────────────────── */
  const stats = [
    { id: 'studies', label: 'Studies Today',    val: String(studiesCount), sub: '↑ 3 from yesterday', icon: 'xray',     color: '#38bdf8', flash: studiesFlash },
    { id: 'pending', label: 'Pending Review',   val: '3',                  sub: '2 high priority',    icon: 'clock',    color: '#f59e0b', flash: false },
    { id: 'acc',     label: 'Model Accuracy',   val: '95.3%',              sub: 'NIH validation set', icon: 'chart',    color: '#22c55e', flash: false },
    { id: 'time',    label: 'Avg. Infer. Time', val: '1.2s',               sub: 'GPU inference',      icon: 'activity', color: '#a78bfa', flash: false },
  ];

  const tools = [
    { id: 'pan',     label: 'Pan',     icon: 'pan'     },
    { id: 'zoom',    label: 'Zoom',    icon: 'zoom'    },
    { id: 'measure', label: 'Measure', icon: 'measure' },
    { id: 'invert',  label: 'Invert',  icon: 'invert'  },
  ];

  /* ────────────────────────────────────────────────────────────────────────
     RENDER
  ──────────────────────────────────────────────────────────────────────── */
  return (
    <>
      <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: '#0f1117' }}>
        <Sidebar />

        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          <TopHeader />

          <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

            {/* ════════════════ CENTER STAGE ════════════════ */}
            <main style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 22, gap: 16, overflow: 'auto', minWidth: 0 }}>

              {/* Page header */}
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexShrink: 0 }}>
                <div>
                  <h1 style={{ fontSize: 21, fontWeight: 700, color: '#f1f5f9', lineHeight: 1, letterSpacing: '-.01em' }}>
                    AI Diagnostic Workstation
                  </h1>
                  <p style={{ fontSize: 13, color: '#4b5563', marginTop: 5 }}>
                    ResNet-34 · Pneumonia Detection · Grad-CAM XAI
                  </p>
                </div>
                {file && (
                  <button className="tool-btn" onClick={reset}
                    style={{ border: '1px solid rgba(255,255,255,.08)', padding: '8px 16px', borderRadius: 9, gap: 7, fontSize: 13 }}>
                    <Ic d={IC.reset} size={14} />
                    New Study
                  </button>
                )}
              </div>

              {/* ── Viewer card ── */}
              <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minHeight: 380 }}>

                {/* Toolbar */}
                <div style={{
                  borderBottom: '1px solid rgba(255,255,255,.055)',
                  padding: '7px 12px',
                  display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0,
                }}>
                  <span style={{ fontSize: 11, fontWeight: 700, color: '#2d3748', letterSpacing: '.06em', textTransform: 'uppercase', marginRight: 6 }}>
                    Viewer
                  </span>
                  <div style={{ width: 1, height: 18, background: 'rgba(255,255,255,.07)', marginRight: 4 }} />

                  {tools.map(t => (
                    <button key={t.id}
                      className={`tool-btn ${activeTool === t.id ? 'active' : ''} ${t.id === 'invert' && isInverted ? 'active' : ''}`}
                      onClick={() => handleToolClick(t.id)}
                      disabled={state !== 'thinking' && state !== 'result' && t.id !== 'pan'}
                      style={{ opacity: (state !== 'thinking' && state !== 'result' && t.id !== 'pan') ? .35 : 1 }}
                    >
                      <Ic d={IC[t.icon]} size={14} />
                      {t.label}
                    </button>
                  ))}

                  <div style={{ flex: 1 }} />

                  {/* File info pill */}
                  {file && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 12, color: '#4b5563' }}>
                      <Ic d={IC.xray} size={13} stroke="#374151" />
                      <span className="mono" style={{ fontSize: 11 }}>{file.name}</span>
                      <span className="tag" style={{ background: 'rgba(56,189,248,.09)', color: '#38bdf8' }}>
                        {(file.size / 1024).toFixed(0)} KB
                      </span>
                    </div>
                  )}
                </div>

                {/* ── Viewport ── */}
                <div
                  ref={viewportRef}
                  style={{
                    flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    background: '#0d1220', position: 'relative', overflow: 'hidden',
                  }}
                >
                  {/* DROP ZONE — idle */}
                  {state === 'idle' && (
                    <div
                      className={`upload-zone ${dragging ? 'drag-over' : ''}`}
                      style={{ width: 400, padding: '52px 36px', textAlign: 'center', userSelect: 'none' }}
                      onDrop={handleDrop}
                      onDragOver={handleDragOver}
                      onDragLeave={handleDragLeave}
                      onClick={() => document.getElementById('xray-file-input').click()}
                    >
                      <input id="xray-file-input" type="file" accept="image/*"
                        onChange={handleFile} style={{ display: 'none' }} />
                      <div style={{
                        width: 60, height: 60, borderRadius: 16, margin: '0 auto 20px',
                        background: 'rgba(14,165,233,.1)', border: '1px solid rgba(14,165,233,.18)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                      }}>
                        <Ic d={IC.upload} size={28} stroke="#38bdf8" sw={1.5} />
                      </div>
                      <div style={{ fontSize: 16, fontWeight: 600, color: '#e2e8f0', marginBottom: 8, lineHeight: 1.3 }}>
                        Drop chest X-ray image here
                      </div>
                      <div style={{ fontSize: 13, color: '#374151', marginBottom: 24, lineHeight: 1.5 }}>
                        Supports PNG, JPG and DICOM-exported images.<br />
                        Files are processed locally and not stored.
                      </div>
                      <div style={{
                        display: 'inline-flex', alignItems: 'center', gap: 8,
                        background: '#0ea5e9', color: '#fff', padding: '9px 22px',
                        borderRadius: 9, fontSize: 13, fontWeight: 600,
                        boxShadow: '0 2px 14px rgba(14,165,233,.28)',
                      }}>
                        <Ic d={IC.upload} size={14} stroke="#fff" sw={2.5} />
                        Browse Files
                      </div>
                    </div>
                  )}

                  {/* API SPINNER — processing */}
                  {state === 'processing' && (
                    <div style={{ textAlign: 'center' }}>
                      <div style={{
                        width: 48, height: 48, borderRadius: '50%',
                        border: '2px solid rgba(56,189,248,.1)',
                        borderTop: '2px solid #38bdf8',
                        animation: 'spin .75s linear infinite',
                        margin: '0 auto 14px',
                      }} />
                      <div style={{ fontSize: 14, color: '#64748b', fontWeight: 500 }}>
                        Sending to inference engine…
                      </div>
                      <div className="mono" style={{ fontSize: 11, color: '#2d3748', marginTop: 5 }}>
                        POST localhost:8000/predict
                      </div>
                    </div>
                  )}

                  {/* CANVAS — thinking & result */}
                  {(state === 'thinking' || state === 'result') && (
                    <div
                      className="canvas-viewport"
                      style={{ position: 'relative', flexShrink: 0 }}
                      onMouseMove={handleCanvasMouseMove}
                      onMouseLeave={handleCanvasMouseLeave}
                    >
                      {/* Sweep scanline (thinking only) */}
                      {state === 'thinking' && <div className="sweep-line" />}

                      {/* X-ray canvas */}
                      <canvas
                        ref={canvasRef}
                        style={{
                          width: CANVAS_SIZE, height: CANVAS_SIZE,
                          display: 'block', borderRadius: 6,
                          // Invert tool is a pure CSS filter — no canvas redraw needed
                          filter: isInverted ? 'invert(1)' : 'none',
                          transition: 'filter .2s ease',
                        }}
                      />

                      {/* ── NEW: Radiology Crosshair SVG overlay ── */}
                      {crosshair && (
                        <svg
                          width={CANVAS_SIZE}
                          height={CANVAS_SIZE}
                          viewBox={`0 0 ${CANVAS_SIZE} ${CANVAS_SIZE}`}
                          style={{
                            position: 'absolute', top: 0, left: 0,
                            pointerEvents: 'none', borderRadius: 6,
                          }}
                        >
                          {/* Horizontal line */}
                          <line
                            x1={0} y1={crosshair.y} x2={CANVAS_SIZE} y2={crosshair.y}
                            stroke="rgba(56,189,248,0.55)" strokeWidth={0.75}
                            strokeDasharray="4 4"
                          />
                          {/* Vertical line */}
                          <line
                            x1={crosshair.x} y1={0} x2={crosshair.x} y2={CANVAS_SIZE}
                            stroke="rgba(56,189,248,0.55)" strokeWidth={0.75}
                            strokeDasharray="4 4"
                          />
                          {/* Centre dot */}
                          <circle
                            cx={crosshair.x} cy={crosshair.y} r={3}
                            fill="none" stroke="rgba(56,189,248,0.9)" strokeWidth={1.2}
                          />
                          {/* Coordinate readout */}
                          <text
                            x={crosshair.x + 8}
                            y={crosshair.y - 6}
                            fill="rgba(56,189,248,0.8)"
                            fontSize={10}
                            fontFamily="'JetBrains Mono', monospace"
                          >
                            {Math.round(crosshair.x)},{Math.round(crosshair.y)}
                          </text>
                        </svg>
                      )}

                      {/* HUD badge — top-left */}
                      <div className="mono" style={{
                        position: 'absolute', top: 10, left: 10,
                        background: 'rgba(13,18,32,.75)',
                        border: '1px solid rgba(56,189,248,.15)',
                        backdropFilter: 'blur(6px)',
                        padding: '5px 9px', borderRadius: 7,
                        fontSize: 10.5, color: 'rgba(56,189,248,.75)', lineHeight: 1.6,
                        pointerEvents: 'none',
                      }}>
                        CHEST PA · {CANVAS_SIZE}×{CANVAS_SIZE}<br />
                        {state === 'thinking'
                          ? 'COMPUTING…'
                          : `GRAD-CAM ${showGradCam ? 'ACTIVE' : 'OFF'} · W:${windowContrast.toFixed(2)}×`}
                        <br />
                        {segmenterReady === null
                          ? 'SEG: CHECKING…'
                          : segmenterReady
                            ? <span style={{ color: 'rgba(74,222,128,.85)' }}>SEG: LUNG-ISO ✓</span>
                            : <span style={{ color: 'rgba(251,191,36,.75)' }}>SEG: PASSTHROUGH</span>
                        }
                      </div>

                      {/* Diagnosis stamp — bottom-right */}
                      {state === 'result' && result && (
                        <div className="fade-in" style={{
                          position: 'absolute', bottom: 10, right: 10,
                          background: result.diagnosis === 'Pneumonia' ? 'rgba(239,68,68,.9)' : 'rgba(34,197,94,.85)',
                          backdropFilter: 'blur(4px)',
                          color: '#fff', padding: '5px 12px', borderRadius: 7,
                          fontSize: 12, fontWeight: 700,
                          boxShadow: '0 2px 12px rgba(0,0,0,.4)',
                          pointerEvents: 'none',
                        }}>
                          {result.diagnosis.toUpperCase()}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* ── Stats row ── */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12, flexShrink: 0 }}>
                {stats.map(({ id, label, val, sub, icon, color, flash }) => (
                  <div key={id} className={`stat-card${flash ? ' flash' : ''}`}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
                      <span style={{ fontSize: 12, color: '#4b5563', fontWeight: 500 }}>{label}</span>
                      <div style={{
                        width: 30, height: 30, borderRadius: 8,
                        background: `${color}18`,
                        display: 'flex', alignItems: 'center', justifyContent: 'center', color,
                      }}>
                        <Ic d={IC[icon]} size={15} />
                      </div>
                    </div>
                    <div key={val} className={flash ? 'count-flash' : ''}
                      style={{ fontSize: 24, fontWeight: 700, color: '#f1f5f9', lineHeight: 1, marginBottom: 5 }}>
                      {val}
                    </div>
                    <div style={{ fontSize: 11, color: '#374151' }}>{sub}</div>
                  </div>
                ))}
              </div>

              {/*
               * ── OOD / Disclaimer footer ──
               * Req: remove the intrusive red overlay; replace with subtle static text below stats.
               * Muted slate-500 colour — professional disclaimer, non-obstructive.
               */}
              <div style={{
                flexShrink: 0, display: 'flex', alignItems: 'center', gap: 8,
                padding: '10px 14px', borderRadius: 9,
                background: 'rgba(255,255,255,.02)',
                border: '1px solid rgba(255,255,255,.045)',
              }}>
                <Ic d={IC.info} size={14} stroke="#475569" />
                <p style={{ fontSize: 11, color: '#475569', lineHeight: 1.55, margin: 0 }}>
                  <strong style={{ color: '#64748b', fontWeight: 600 }}>Distribution Warning:</strong>
                  {' '}This model was trained on the NIH ChestX-14 dataset. Images acquired with
                  different equipment, patient positioning, or exposure settings may fall
                  outside the training distribution and yield unreliable predictions.
                  Always confirm findings with a qualified radiologist.
                </p>
              </div>
            </main>

            {/* ════ RIGHT PANEL ════ */}
            <RightPanel
              state={state}
              result={result}
              progress={progress}
              statusIdx={statusIdx}
              showGradCam={showGradCam}       setShowGradCam={setShowGradCam}
              heatmapOpacity={heatmapOpacity} setHeatmapOpacity={setHeatmapOpacity}
              heatmapBlur={heatmapBlur}       setHeatmapBlur={setHeatmapBlur}
              windowContrast={windowContrast} setWindowContrast={setWindowContrast}
            />
          </div>
        </div>
      </div>

      {/* Toast portal — above everything */}
      <ToastProvider />
    </>
  );
}