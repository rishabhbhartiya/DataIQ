"""drift_template.py — MLRadar Drift Report Template

Standalone HTML report for train vs test / reference vs production drift.

Placeholders
------------
  __DATASET_NAME__   — dataset display name
  __GENERATED_AT__   — timestamp string
  __DRIFT_DATA__     — JSON from DriftAnalyzer.analyze()

Sections rendered
-----------------
  Summary card       — overall PSI level, verdict, counts
  Schema diff        — appeared / disappeared / dtype-changed columns
  Missing-rate drift — columns where null rate shifted > 3pp
  Per-column table   — PSI, level, KS/chi², mean shift for every column
  Per-column detail  — expandable accordion per drifted column
"""


def get_drift_template() -> str:
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>MLRadar Drift — __DATASET_NAME__</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#05070f; --bg2:#080c18; --bg3:#0d1225;
  --glass:rgba(15,20,45,0.6); --glass2:rgba(20,28,60,0.5);
  --border:rgba(79,142,247,0.12); --border2:rgba(255,255,255,0.06);
  --accent:#4f8ef7; --accent2:#7c3aed; --accent3:#06b6d4;
  --green:#22c55e; --yellow:#f59e0b; --red:#ef4444; --orange:#f97316;
  --text:#e2e8f0; --text2:#94a3b8; --text3:#475569;
  --mono:'JetBrains Mono',monospace; --sans:'Space Grotesk',sans-serif;
  --radius:14px; --radius-sm:8px;
  --glow:0 0 30px rgba(79,142,247,0.08);
}
*{box-sizing:border-box;margin:0;padding:0}
html{font-size:13px;scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh}
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:
    radial-gradient(ellipse 60% 40% at 20% 10%,rgba(79,142,247,0.06) 0%,transparent 60%),
    radial-gradient(ellipse 50% 50% at 80% 80%,rgba(124,58,237,0.05) 0%,transparent 60%);
}

/* ── Layout ── */
.layout{display:flex;min-height:100vh;position:relative;z-index:1}
.sidebar{
  width:240px;flex-shrink:0;background:rgba(8,12,24,0.9);
  backdrop-filter:blur(20px);border-right:1px solid var(--border);
  position:sticky;top:0;height:100vh;overflow-y:auto;
  display:flex;flex-direction:column;
}
.main{flex:1;min-width:0;padding:28px 36px}

/* ── Sidebar ── */
.logo-area{padding:22px 18px 16px;border-bottom:1px solid var(--border)}
.logo-name{
  font-size:18px;font-weight:700;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.logo-sub{font-size:10px;font-family:var(--mono);color:var(--text3);margin-top:2px}
.ds-info{padding:12px 18px;border-bottom:1px solid var(--border)}
.ds-name{font-weight:600;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ds-meta{font-size:10px;font-family:var(--mono);color:var(--text3);margin-top:3px}
.nav-group{padding:10px 0}
.nav-header{padding:4px 18px;font-size:9px;font-weight:600;color:var(--text3);letter-spacing:1.5px;text-transform:uppercase;font-family:var(--mono)}
.nav-item{
  display:flex;align-items:center;gap:8px;padding:8px 18px;
  cursor:pointer;color:var(--text2);font-size:12px;font-weight:500;
  transition:all .15s;border-left:2px solid transparent;
}
.nav-item:hover{background:rgba(79,142,247,0.06);color:var(--text)}
.nav-item.active{background:linear-gradient(90deg,rgba(79,142,247,0.12),transparent);color:var(--accent);border-left-color:var(--accent)}
.nav-item .ni{font-size:13px;width:16px;text-align:center;flex-shrink:0}
.nav-badge{
  margin-left:auto;background:var(--bg3);border:1px solid var(--border2);
  border-radius:999px;padding:0 6px;font-size:10px;font-family:var(--mono);color:var(--text3);
}
.nav-badge.bad{background:rgba(239,68,68,.15);border-color:rgba(239,68,68,.3);color:var(--red)}
.nav-badge.warn{background:rgba(245,158,11,.15);border-color:rgba(245,158,11,.3);color:var(--yellow)}
.nav-badge.good{background:rgba(34,197,94,.15);border-color:rgba(34,197,94,.3);color:var(--green)}
.nav-footer{padding:12px 14px;border-top:1px solid var(--border);margin-top:auto;font-size:10px;font-family:var(--mono);color:var(--text3)}

/* ── Page header ── */
.page-header{margin-bottom:28px}
.page-title{font-size:26px;font-weight:700;letter-spacing:-0.5px}
.page-title span{background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.page-meta{color:var(--text3);font-size:11px;margin-top:5px;font-family:var(--mono)}

/* ── Sections ── */
.section{display:none;animation:fadeUp .2s ease}
.section.active{display:block}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.sec-header{margin-bottom:18px}
.sec-title{font-size:19px;font-weight:700;letter-spacing:-.3px}
.sec-desc{color:var(--text2);font-size:12px;margin-top:3px}

/* ── Cards ── */
.card{
  background:var(--glass);backdrop-filter:blur(16px);
  border:1px solid var(--border);border-radius:var(--radius);
  padding:20px 22px;margin-bottom:16px;box-shadow:var(--glow);
  position:relative;overflow:hidden;
}
.card::before{content:'';position:absolute;inset:0;border-radius:var(--radius);pointer-events:none;
  background:linear-gradient(135deg,rgba(79,142,247,0.03) 0%,transparent 60%);}
.card-sm{padding:14px 18px}
.card h2{font-size:14px;font-weight:600;margin-bottom:14px;color:var(--text)}

/* ── Stats strip ── */
.stats-strip{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:12px;margin-bottom:18px}
.stat-box{
  background:var(--glass2);backdrop-filter:blur(12px);
  border:1px solid var(--border2);border-radius:var(--radius-sm);
  padding:13px 15px;transition:border-color .2s;
}
.stat-box:hover{border-color:var(--border)}
.sb-label{font-size:9px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:1px;font-family:var(--mono)}
.sb-value{font-size:22px;font-weight:700;margin-top:3px;line-height:1;font-family:var(--mono)}
.sb-sub{font-size:10px;color:var(--text3);margin-top:3px;font-family:var(--mono)}
.sb-value.accent{color:var(--accent)} .sb-value.green{color:var(--green)}
.sb-value.yellow{color:var(--yellow)} .sb-value.red{color:var(--red)}
.sb-value.orange{color:var(--orange)} .sb-value.cyan{color:var(--accent3)}

/* ── PSI level pill ── */
.psi-pill{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:999px;font-size:12px;font-weight:600;font-family:var(--mono)}

/* ── Badges ── */
.badge{display:inline-flex;align-items:center;padding:2px 7px;border-radius:999px;font-size:10px;font-family:var(--mono);font-weight:600;border:1px solid}
.badge-major   {background:rgba(239,68,68,.15);color:var(--red);border-color:rgba(239,68,68,.3)}
.badge-moderate{background:rgba(245,158,11,.15);color:var(--yellow);border-color:rgba(245,158,11,.3)}
.badge-minor   {background:rgba(132,204,22,.15);color:#84cc16;border-color:rgba(132,204,22,.3)}
.badge-negligible{background:rgba(34,197,94,.12);color:var(--green);border-color:rgba(34,197,94,.25)}
.badge-ok      {background:rgba(34,197,94,.12);color:var(--green);border-color:rgba(34,197,94,.25)}
.badge-warn    {background:rgba(245,158,11,.12);color:var(--yellow);border-color:rgba(245,158,11,.25)}
.badge-bad     {background:rgba(239,68,68,.12);color:var(--red);border-color:rgba(239,68,68,.25)}

/* ── Table ── */
.tbl{width:100%;border-collapse:collapse;font-size:12px}
.tbl th{text-align:left;padding:8px 10px;font-size:10px;font-weight:700;color:var(--text3);text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid var(--border2);font-family:var(--mono)}
.tbl td{padding:9px 10px;border-bottom:1px solid rgba(255,255,255,0.03);color:var(--text2);vertical-align:middle}
.tbl tr:last-child td{border-bottom:none}
.tbl tr:hover td{background:rgba(79,142,247,0.03)}
.mono{font-family:var(--mono)!important;font-size:11px!important}

/* ── Progress bar ── */
.pbar-wrap{display:flex;align-items:center;gap:8px;min-width:80px}
.pbar-track{flex:1;height:4px;background:var(--bg3);border-radius:2px;overflow:hidden}
.pbar-fill{height:100%;border-radius:2px;transition:width .4s}
.pbar-label{font-size:10px;font-family:var(--mono);color:var(--text3);white-space:nowrap}

/* ── Column accordion ── */
.col-acc{margin-bottom:8px}
.col-acc-hdr{
  display:flex;align-items:center;gap:10px;padding:10px 16px;
  background:var(--glass2);border:1px solid var(--border2);border-radius:var(--radius-sm);
  cursor:pointer;transition:all .15s;
}
.col-acc-hdr:hover{border-color:var(--border);background:rgba(79,142,247,.04)}
.col-acc-hdr.open{border-bottom-left-radius:0;border-bottom-right-radius:0;border-color:var(--border)}
.col-acc-body{
  background:rgba(8,12,24,.5);border:1px solid var(--border2);
  border-top:none;border-radius:0 0 var(--radius-sm) var(--radius-sm);
  padding:16px;display:none;
}
.col-acc-body.open{display:block}
.chevron{margin-left:auto;font-size:10px;color:var(--text3);transition:transform .2s}
.col-acc-hdr.open .chevron{transform:rotate(180deg)}
.chart-wrap{border-radius:8px;overflow:hidden;background:rgba(5,7,15,.4);margin-top:10px}

/* ── Col chip ── */
.col-chip{background:var(--bg3);border:1px solid var(--border2);border-radius:5px;padding:1px 7px;font-size:10px;font-family:var(--mono);color:var(--text2)}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

/* ── Responsive ── */
@media(max-width:900px){.sidebar{display:none}.main{padding:16px}}
</style>
</head>
<body>
<div class="layout">

<!-- ── SIDEBAR ── -->
<nav class="sidebar">
  <div class="logo-area">
    <div class="logo-name">MLRadar Drift</div>
    <div class="logo-sub">Train vs Production Monitor</div>
  </div>
  <div class="ds-info">
    <div class="ds-name">__DATASET_NAME__</div>
    <div class="ds-meta" id="sb-meta">Loading...</div>
  </div>
  <div class="nav-group">
    <div class="nav-header">Report</div>
    <div class="nav-item active" onclick="nav('summary',this)"><span class="ni">◈</span>Summary</div>
    <div class="nav-item" onclick="nav('columns',this)"><span class="ni">⊞</span>Column Drift<span class="nav-badge" id="nb-cols">—</span></div>
    <div class="nav-item" onclick="nav('drifted',this)"><span class="ni">⚠</span>Drifted Columns<span class="nav-badge" id="nb-drift">—</span></div>
  </div>
  <div class="nav-group">
    <div class="nav-header">Issues</div>
    <div class="nav-item" onclick="nav('missing',this)"><span class="ni">◌</span>Missing Rate Drift<span class="nav-badge" id="nb-miss">—</span></div>
    <div class="nav-item" onclick="nav('schema',this)"><span class="ni">⊗</span>Schema Changes<span class="nav-badge" id="nb-schema">—</span></div>
  </div>
  <div class="nav-footer">Generated __GENERATED_AT__</div>
</nav>

<!-- ── MAIN ── -->
<main class="main">
  <div class="page-header">
    <div class="page-title">__DATASET_NAME__ <span>— Drift Report</span></div>
    <div class="page-meta" id="page-meta">Loading...</div>
  </div>

  <!-- Summary -->
  <div class="section active" id="sec-summary">
    <div class="sec-header"><div class="sec-title">Drift Summary</div><div class="sec-desc">Population Stability Index (PSI) across all features. PSI ≥ 0.25 = Major drift.</div></div>
    <div id="summary-content"></div>
  </div>

  <!-- Column Drift Table -->
  <div class="section" id="sec-columns">
    <div class="sec-header"><div class="sec-title">All Columns</div><div class="sec-desc">PSI, KS test, and distribution shift per feature.</div></div>
    <div id="columns-content"></div>
  </div>

  <!-- Drifted Only -->
  <div class="section" id="sec-drifted">
    <div class="sec-header"><div class="sec-title">Drifted Columns</div><div class="sec-desc">Only columns with Minor drift or worse (PSI ≥ 0.10), with expandable detail.</div></div>
    <div id="drifted-content"></div>
  </div>

  <!-- Missing Rate Drift -->
  <div class="section" id="sec-missing">
    <div class="sec-header"><div class="sec-title">Missing Rate Drift</div><div class="sec-desc">Columns where the null rate changed by more than 3 percentage points.</div></div>
    <div id="missing-content"></div>
  </div>

  <!-- Schema Changes -->
  <div class="section" id="sec-schema">
    <div class="sec-header"><div class="sec-title">Schema Changes</div><div class="sec-desc">Columns that appeared, disappeared, or changed dtype between reference and new data.</div></div>
    <div id="schema-content"></div>
  </div>
</main>
</div>

<script>
// ═══════════════════════════════════════════════
//  DATA
// ═══════════════════════════════════════════════
const D = __DRIFT_DATA__;

// ═══════════════════════════════════════════════
//  BOOT
// ═══════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('sb-meta').textContent =
    `ref: ${(D.n_ref||0).toLocaleString()} rows  ·  new: ${(D.n_new||0).toLocaleString()} rows`;
  document.getElementById('page-meta').textContent =
    `Reference: ${(D.n_ref||0).toLocaleString()} rows  ·  New: ${(D.n_new||0).toLocaleString()} rows  ·  ${D.n_cols_analyzed||0} columns analyzed  ·  Generated __GENERATED_AT__`;

  updateNavBadges();
  renderSummary();
  renderColumns();
  renderDrifted();
  renderMissingDrift();
  renderSchema();
});

// ═══════════════════════════════════════════════
//  NAVIGATION
// ═══════════════════════════════════════════════
function nav(id, el) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('sec-' + id).classList.add('active');
  el.classList.add('active');
}

function updateNavBadges() {
  const cols = D.columns || [];
  document.getElementById('nb-cols').textContent = cols.length;
  const drifted = cols.filter(c => c.drift_level && c.drift_level !== 'Negligible');
  const nb = document.getElementById('nb-drift');
  nb.textContent = drifted.length;
  nb.className = 'nav-badge ' + (drifted.length > 3 ? 'bad' : drifted.length > 0 ? 'warn' : 'good');
  const miss = (D.missing_drift || []).length;
  const nbm  = document.getElementById('nb-miss');
  nbm.textContent = miss;
  nbm.className = 'nav-badge ' + (miss > 3 ? 'bad' : miss > 0 ? 'warn' : 'good');
  const sc = D.schema || {};
  const schemaCount = (sc.appeared||[]).length + (sc.disappeared||[]).length + (sc.dtype_changed||[]).length;
  const nbs = document.getElementById('nb-schema');
  nbs.textContent = schemaCount;
  nbs.className = 'nav-badge ' + (schemaCount > 0 ? 'warn' : 'good');
}

// ═══════════════════════════════════════════════
//  PSI helpers
// ═══════════════════════════════════════════════
function psiColor(level) {
  const m = {Major:'var(--red)',Moderate:'var(--yellow)',Minor:'#84cc16',Negligible:'var(--green)'};
  return m[level] || 'var(--text2)';
}

function psiBar(psi, level) {
  const max = 0.5;
  const w   = Math.min((psi / max) * 100, 100);
  const c   = psiColor(level);
  return `<div class="pbar-wrap" style="min-width:100px">
    <div class="pbar-track"><div class="pbar-fill" style="width:${w}%;background:${c}"></div></div>
    <div class="pbar-label" style="color:${c}">${psi}</div>
  </div>`;
}

function levelBadge(level) {
  if (!level) return '—';
  const cls = level === 'Major' ? 'major' : level === 'Moderate' ? 'moderate' :
              level === 'Minor' ? 'minor' : 'negligible';
  return `<span class="badge badge-${cls}">${level}</span>`;
}

// ═══════════════════════════════════════════════
//  SUMMARY
// ═══════════════════════════════════════════════
function renderSummary() {
  const el    = document.getElementById('summary-content');
  const level = D.drift_level || 'Unknown';
  const color = D.drift_color || 'var(--text2)';

  const psiScale = [
    {label:'Negligible', range:'PSI < 0.10', color:'var(--green)',  desc:'No action needed'},
    {label:'Minor',      range:'PSI < 0.20', color:'#84cc16',       desc:'Monitor periodically'},
    {label:'Moderate',   range:'PSI < 0.25', color:'var(--yellow)', desc:'Investigate, plan retraining'},
    {label:'Major',      range:'PSI ≥ 0.25', color:'var(--red)',    desc:'Retrain immediately'},
  ];

  el.innerHTML = `
    <!-- Verdict Banner -->
    <div class="card" style="border-left:4px solid ${color};margin-bottom:20px">
      <div style="display:flex;align-items:center;gap:18px">
        <div style="font-size:42px">${level==='Stable'||level==='Negligible'?'✅':'⚠️'}</div>
        <div style="flex:1">
          <div style="font-size:20px;font-weight:700;color:${color}">${level}</div>
          <div style="font-size:13px;color:var(--text2);margin-top:4px">${D.verdict||''}</div>
        </div>
        <div style="text-align:right;font-family:var(--mono)">
          <div style="font-size:10px;color:var(--text3)">Avg PSI</div>
          <div style="font-size:28px;font-weight:700;color:${color}">${D.avg_psi||0}</div>
        </div>
      </div>
    </div>

    <!-- Stats -->
    <div class="stats-strip">
      <div class="stat-box"><div class="sb-label">Ref Rows</div><div class="sb-value accent">${(D.n_ref||0).toLocaleString()}</div><div class="sb-sub">reference dataset</div></div>
      <div class="stat-box"><div class="sb-label">New Rows</div><div class="sb-value accent">${(D.n_new||0).toLocaleString()}</div><div class="sb-sub">new dataset</div></div>
      <div class="stat-box"><div class="sb-label">Cols Analyzed</div><div class="sb-value cyan">${D.n_cols_analyzed||0}</div></div>
      <div class="stat-box"><div class="sb-label">Major</div><div class="sb-value red">${D.n_major||0}</div><div class="sb-sub">PSI ≥ 0.25</div></div>
      <div class="stat-box"><div class="sb-label">Moderate</div><div class="sb-value yellow">${D.n_moderate||0}</div><div class="sb-sub">PSI 0.20–0.25</div></div>
      <div class="stat-box"><div class="sb-label">Minor</div><div class="sb-value" style="color:#84cc16">${D.n_minor||0}</div><div class="sb-sub">PSI 0.10–0.20</div></div>
      <div class="stat-box"><div class="sb-label">Negligible</div><div class="sb-value green">${D.n_negligible||0}</div><div class="sb-sub">PSI < 0.10</div></div>
    </div>

    <!-- PSI Scale Reference -->
    <div class="card card-sm">
      <h2>PSI Severity Scale</h2>
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px">
        ${psiScale.map(s=>`
          <div style="padding:10px 14px;background:var(--bg3);border-radius:8px;border:1px solid ${s.color}33">
            <div style="font-size:12px;font-weight:700;color:${s.color}">${s.label}</div>
            <div style="font-size:10px;font-family:var(--mono);color:var(--text3);margin-top:2px">${s.range}</div>
            <div style="font-size:11px;color:var(--text2);margin-top:4px">${s.desc}</div>
          </div>`).join('')}
      </div>
    </div>

    <!-- Top drifted columns -->
    ${(() => {
      const drifted = (D.columns||[]).filter(c => c.drift_level && c.drift_level !== 'Negligible').slice(0,5);
      if (!drifted.length) return '<div class="card card-sm" style="text-align:center;color:var(--green)">✅ All columns stable</div>';
      return `<div class="card"><h2>Most Drifted Columns</h2>
        <table class="tbl"><thead><tr><th>Column</th><th>Type</th><th>PSI</th><th>Level</th></tr></thead>
        <tbody>${drifted.map(c=>`<tr>
          <td class="mono">${c.column}</td>
          <td style="color:var(--text3);font-size:11px">${c.type}</td>
          <td>${psiBar(c.psi||0, c.drift_level)}</td>
          <td>${levelBadge(c.drift_level)}</td>
        </tr>`).join('')}</tbody></table></div>`;
    })()}`;
}

// ═══════════════════════════════════════════════
//  ALL COLUMNS TABLE
// ═══════════════════════════════════════════════
function renderColumns() {
  const el   = document.getElementById('columns-content');
  const cols = D.columns || [];
  if (!cols.length) { el.innerHTML='<div class="card">No columns analyzed.</div>'; return; }

  const rows = cols.map(c => {
    const isNum = c.type === 'numeric';
    const extra = isNum
      ? `${c.mean_shift_pct != null ? `<span style="font-size:10px;color:var(--text3);font-family:var(--mono)">mean shift ${c.mean_shift_pct}%</span>` : ''}`
      : `${c.n_appeared ? `<span class="col-chip" style="border-color:rgba(34,197,94,.3);color:var(--green)">+${c.n_appeared} cats</span>` : ''}
         ${c.n_disappeared ? `<span class="col-chip" style="border-color:rgba(239,68,68,.3);color:var(--red)">-${c.n_disappeared} cats</span>` : ''}`;

    const sig = isNum ? (c.ks_significant ? '<span class="badge badge-warn">KS sig</span>' : '')
                      : (c.chi_significant ? '<span class="badge badge-warn">χ² sig</span>' : '');

    return `<tr>
      <td class="mono">${c.column}</td>
      <td style="font-size:11px;color:var(--text3)">${c.type}</td>
      <td>${c.psi != null ? psiBar(c.psi, c.drift_level) : '<span style="color:var(--text3)">—</span>'}</td>
      <td>${levelBadge(c.drift_level)}</td>
      <td>${sig}</td>
      <td style="font-size:11px">${extra}</td>
    </tr>`;
  }).join('');

  el.innerHTML = `<div class="card"><h2>All Columns (${cols.length})</h2>
    <table class="tbl"><thead><tr>
      <th>Column</th><th>Type</th><th>PSI</th><th>Drift Level</th><th>Stat Test</th><th>Details</th>
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

// ═══════════════════════════════════════════════
//  DRIFTED COLUMNS — ACCORDION
// ═══════════════════════════════════════════════
function renderDrifted() {
  const el      = document.getElementById('drifted-content');
  const drifted = (D.columns||[]).filter(c => c.drift_level && c.drift_level !== 'Negligible');
  document.getElementById('nb-drift').textContent = drifted.length;

  if (!drifted.length) {
    el.innerHTML=`<div class="card" style="text-align:center;padding:48px">
      <div style="font-size:48px;margin-bottom:12px">✅</div>
      <div style="font-size:18px;font-weight:600;color:var(--green)">All features stable!</div>
      <div style="color:var(--text2);margin-top:4px">No columns with PSI ≥ 0.10.</div>
    </div>`;
    return;
  }

  el.innerHTML = drifted.map((c, i) => {
    const color  = psiColor(c.drift_level);
    const isNum  = c.type === 'numeric';

    // Detail body
    let detailHtml = '';
    if (isNum) {
      const stats = [
        ['Ref Mean', c.ref_mean], ['New Mean', c.new_mean], ['Mean Shift', c.mean_shift_pct + '%'],
        ['Ref Std',  c.ref_std],  ['New Std',  c.new_std],  ['Std Shift',  c.std_shift_pct + '%'],
        ['Ref Median', c.ref_median], ['New Median', c.new_median],
        ['KS Stat', c.ks_stat],   ['KS p-value', c.ks_pvalue],
      ];
      detailHtml = `
        <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px;margin-bottom:14px">
          ${stats.filter(([,v])=>v!=null).map(([l,v])=>`
            <div style="background:var(--bg3);border:1px solid var(--border2);border-radius:6px;padding:8px 10px">
              <div style="font-size:9px;color:var(--text3);font-family:var(--mono);text-transform:uppercase">${l}</div>
              <div style="font-size:13px;font-weight:600;font-family:var(--mono);color:var(--text);margin-top:2px">${v}</div>
            </div>`).join('')}
        </div>`;

      // Percentile comparison table
      if (c.ref_percentiles && c.new_percentiles) {
        const pctKeys = Object.keys(c.ref_percentiles);
        detailHtml += `<h2 style="font-size:13px;margin-bottom:8px">Percentile Comparison</h2>
          <table class="tbl"><thead><tr>
            <th>Percentile</th><th>Reference</th><th>New</th><th>Δ</th>
          </tr></thead><tbody>
          ${pctKeys.map(p => {
            const rv = c.ref_percentiles[p], nv = c.new_percentiles[p];
            const delta = rv != null && nv != null ? (nv - rv).toFixed(4) : '—';
            const dc    = parseFloat(delta) > 0 ? 'var(--red)' : parseFloat(delta) < 0 ? 'var(--green)' : 'var(--text3)';
            return `<tr>
              <td class="mono">${p}</td>
              <td class="mono">${rv??'—'}</td>
              <td class="mono">${nv??'—'}</td>
              <td class="mono" style="color:${dc}">${delta!=='—'?(parseFloat(delta)>0?'+':'')+delta:'—'}</td>
            </tr>`;
          }).join('')}</tbody></table>`;
      }
    } else {
      // Categorical
      const appeared    = c.appeared   || [];
      const disappeared = c.disappeared || [];
      detailHtml = `
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:14px">
          <div>
            <div style="font-size:11px;font-weight:600;color:var(--text2);margin-bottom:8px">Reference Distribution (top 10)</div>
            ${(c.ref_freq_table||[]).map(r=>`<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">
              <span class="col-chip" style="width:100px;overflow:hidden;text-overflow:ellipsis">${r.label}</span>
              <div class="pbar-wrap" style="flex:1"><div class="pbar-track"><div class="pbar-fill" style="width:${r.pct}%;background:var(--accent)"></div></div>
              <div class="pbar-label">${r.pct}%</div></div>
            </div>`).join('')}
          </div>
          <div>
            <div style="font-size:11px;font-weight:600;color:var(--text2);margin-bottom:8px">New Distribution (top 10)</div>
            ${(c.new_freq_table||[]).map(r=>`<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">
              <span class="col-chip" style="width:100px;overflow:hidden;text-overflow:ellipsis">${r.label}</span>
              <div class="pbar-wrap" style="flex:1"><div class="pbar-track"><div class="pbar-fill" style="width:${r.pct}%;background:var(--accent2)"></div></div>
              <div class="pbar-label">${r.pct}%</div></div>
            </div>`).join('')}
          </div>
        </div>
        ${appeared.length ? `<div style="margin-bottom:8px"><span style="font-size:11px;color:var(--green)">✚ New categories in production: </span>
          ${appeared.map(a=>`<span class="col-chip" style="border-color:rgba(34,197,94,.3);color:var(--green)">${a}</span>`).join(' ')}</div>` : ''}
        ${disappeared.length ? `<div><span style="font-size:11px;color:var(--red)">✖ Categories disappeared: </span>
          ${disappeared.map(a=>`<span class="col-chip" style="border-color:rgba(239,68,68,.3);color:var(--red)">${a}</span>`).join(' ')}</div>` : ''}`;
    }

    return `<div class="col-acc" id="da-${i}">
      <div class="col-acc-hdr" onclick="toggleDA(${i})">
        <span style="font-family:var(--mono);font-size:12.5px;font-weight:600;color:var(--text)">${c.column}</span>
        ${levelBadge(c.drift_level)}
        <span style="font-size:11px;color:var(--text3);font-family:var(--mono)">${c.type}</span>
        <div style="margin-left:auto;display:flex;align-items:center;gap:14px">
          ${c.psi != null ? psiBar(c.psi, c.drift_level) : ''}
          <span class="chevron">▼</span>
        </div>
      </div>
      <div class="col-acc-body" id="da-body-${i}">${detailHtml}</div>
    </div>`;
  }).join('');
}

function toggleDA(i) {
  const h = document.querySelector(`#da-${i} .col-acc-hdr`);
  const b = document.getElementById(`da-body-${i}`);
  h.classList.toggle('open'); b.classList.toggle('open');
}

// ═══════════════════════════════════════════════
//  MISSING RATE DRIFT
// ═══════════════════════════════════════════════
function renderMissingDrift() {
  const el   = document.getElementById('missing-content');
  const rows = D.missing_drift || [];
  document.getElementById('nb-miss').textContent = rows.length;

  if (!rows.length) {
    el.innerHTML=`<div class="card" style="text-align:center;padding:40px">
      <div style="font-size:48px;margin-bottom:12px">✅</div>
      <div style="font-size:16px;font-weight:600;color:var(--green)">No significant missing rate changes</div>
    </div>`;
    return;
  }

  const tableRows = rows.map(r => {
    const dc    = r.delta > 0 ? 'var(--red)' : 'var(--green)';
    const arrow = r.delta > 0 ? '↑' : '↓';
    return `<tr>
      <td class="mono">${r.column}</td>
      <td class="mono">${r.ref_miss}%</td>
      <td class="mono">${r.new_miss}%</td>
      <td class="mono" style="color:${dc};font-weight:600">${arrow} ${Math.abs(r.delta)}pp</td>
      <td><span class="badge badge-${r.severity==='high'?'bad':r.severity==='medium'?'warn':'ok'}">${r.severity.toUpperCase()}</span></td>
    </tr>`;
  }).join('');

  el.innerHTML = `<div class="card"><h2>Missing Rate Changes (${rows.length} columns)</h2>
    <p style="font-size:12px;color:var(--text2);margin-bottom:14px">Columns where null rate changed by more than 3 percentage points. A sudden spike may indicate upstream data pipeline issues.</p>
    <table class="tbl"><thead><tr>
      <th>Column</th><th>Ref Missing</th><th>New Missing</th><th>Change</th><th>Severity</th>
    </tr></thead><tbody>${tableRows}</tbody></table></div>`;
}

// ═══════════════════════════════════════════════
//  SCHEMA DIFF
// ═══════════════════════════════════════════════
function renderSchema() {
  const el     = document.getElementById('schema-content');
  const schema = D.schema || {};
  const appeared     = schema.appeared     || [];
  const disappeared  = schema.disappeared  || [];
  const dtypeChanged = schema.dtype_changed || [];

  if (!schema.has_changes) {
    el.innerHTML=`<div class="card" style="text-align:center;padding:40px">
      <div style="font-size:48px;margin-bottom:12px">✅</div>
      <div style="font-size:16px;font-weight:600;color:var(--green)">No schema changes detected</div>
      <div style="color:var(--text2);margin-top:4px">All column names and dtypes match.</div>
    </div>`;
    return;
  }

  el.innerHTML = `
    ${appeared.length ? `<div class="card"><h2>✚ New Columns (in production, not in reference)</h2>
      <div style="display:flex;flex-wrap:wrap;gap:6px">
        ${appeared.map(c=>`<span class="col-chip" style="border-color:rgba(34,197,94,.3);color:var(--green)">${c}</span>`).join('')}
      </div>
      <div style="margin-top:10px;font-size:11px;color:var(--text3)">These columns are present in new data but were not in the training set. Your model has never seen them.</div>
    </div>` : ''}

    ${disappeared.length ? `<div class="card"><h2>✖ Missing Columns (in reference, not in production)</h2>
      <div style="display:flex;flex-wrap:wrap;gap:6px">
        ${disappeared.map(c=>`<span class="col-chip" style="border-color:rgba(239,68,68,.3);color:var(--red)">${c}</span>`).join('')}
      </div>
      <div style="margin-top:10px;font-size:11px;color:var(--text3)">These columns were used in training but are missing from production data. Your model will fail or produce incorrect results.</div>
    </div>` : ''}

    ${dtypeChanged.length ? `<div class="card"><h2>⚙ Dtype Changes</h2>
      <table class="tbl"><thead><tr><th>Column</th><th>Reference dtype</th><th>New dtype</th></tr></thead>
      <tbody>${dtypeChanged.map(d=>`<tr>
        <td class="mono">${d.column}</td>
        <td class="mono" style="color:var(--red)">${d.ref_dtype}</td>
        <td class="mono" style="color:var(--green)">${d.new_dtype}</td>
      </tr>`).join('')}</tbody></table>
      <div style="margin-top:10px;font-size:11px;color:var(--text3)">Dtype mismatches can cause silent casting errors or exceptions at prediction time.</div>
    </div>` : ''}`;
}
</script>
</body>
</html>"""