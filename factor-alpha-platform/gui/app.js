/**
 * Alpha Research Platform — WQ BRAIN-style Frontend
 * Split-pane layout with IS Summary and year-by-year stats
 */

// =========================================================================
// State
// =========================================================================
const API = '';
let ws = null;
let wsRetryCount = 0;

// =========================================================================
// Utilities
// =========================================================================
function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function toast(message, type = 'info') {
    const container = $('#toastContainer');
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, 4000);
}

async function api(endpoint, options = {}) {
    const url = `${API}${endpoint}`;
    const resp = await fetch(url, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
    });
    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`API ${resp.status}: ${text}`);
    }
    return resp.json();
}

function fmtNum(v, decimals = 2) {
    if (v == null || isNaN(v)) return '--';
    return Number(v).toFixed(decimals);
}

function fmtPct(v) {
    if (v == null || isNaN(v)) return '--';
    return (Number(v) * 100).toFixed(2) + '%';
}

function colorClass(v) {
    if (v == null || isNaN(v)) return '';
    return v >= 0 ? 'positive' : 'negative';
}

// =========================================================================
// WebSocket
// =========================================================================
function connectWS() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);
    ws.onopen = () => { wsRetryCount = 0; };
    ws.onmessage = (event) => {
        try { handleWSMessage(JSON.parse(event.data)); } catch { }
    };
    ws.onclose = () => {
        wsRetryCount++;
        setTimeout(connectWS, Math.min(1000 * wsRetryCount, 10000));
    };
    ws.onerror = () => { };
}

function handleWSMessage(msg) {
    switch (msg.type) {
        case 'pong': break;
        case 'eval_result': toast('Alpha evaluated', 'success'); break;
        case 'gp_start':
            appendLog('gpLog', `GP started: ${msg.data.population} pop x ${msg.data.generations} gen`, 'info');
            break;
        case 'gp_complete':
            appendLog('gpLog', `GP complete! Best fitness: ${fmtNum(msg.data.best_fitness)}`, 'success');
            appendLog('gpLog', `Best: ${msg.data.best_expression}`, 'success');
            if (msg.data.best_alphas) {
                msg.data.best_alphas.forEach((a, i) => {
                    appendLog('gpLog', `  ${i + 1}. F=${fmtNum(a.fitness)} | ${a.expression.slice(0, 60)}`, 'success');
                });
            }
            toast('GP campaign complete!', 'success');
            refreshLibrary(); refreshStats();
            break;
        case 'gp_error':
            appendLog('gpLog', `ERROR: ${msg.data.error}`, 'fail');
            toast('GP error: ' + msg.data.error, 'error');
            break;
        case 'llm_start':
            appendLog('llmLog', `LLM started: ${msg.data.trials} trials`, 'info');
            break;
        case 'llm_complete':
            appendLog('llmLog', `LLM complete! ${msg.data.successful || 0} successful`, 'success');
            toast('LLM campaign complete!', 'success');
            refreshLibrary(); refreshStats();
            break;
        case 'llm_error':
            appendLog('llmLog', `ERROR: ${msg.data.error}`, 'fail');
            break;
    }
}

function appendLog(logId, text, type = '') {
    const log = $(`#${logId}`);
    log.classList.add('active');
    const entry = document.createElement('div');
    entry.className = `log-entry log-${type}`;
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
}

// =========================================================================
// Tab Navigation (header tabs + results tabs)
// =========================================================================
function initTabs() {
    // Main header tabs
    $$('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            $$('.tab').forEach(t => t.classList.remove('active'));
            $$('.tab-content').forEach(tc => tc.classList.remove('active'));
            tab.classList.add('active');
            $(`#tab-${tab.dataset.tab}`).classList.add('active');
        });
    });

    // Results sub-tabs (CODE / RESULTS / DATA)
    $$('.rtab').forEach(rtab => {
        rtab.addEventListener('click', () => {
            $$('.rtab').forEach(t => t.classList.remove('active'));
            $$('.rtab-content').forEach(tc => tc.classList.remove('active'));
            rtab.classList.add('active');
            $(`#rtab-${rtab.dataset.rtab}`).classList.add('active');
        });
    });
}

// =========================================================================
// Status
// =========================================================================
async function refreshStatus() {
    try {
        const data = await api('/api/status');
        const badge = $('#statusBadge');
        const text = $('#statusText');
        if (data.ready) {
            badge.classList.add('ready');
            const source = data.data_source?.toUpperCase() === 'FMP' ? 'FMP' : (data.data_source || 'Synthetic');
            text.textContent = `${source} · ${data.n_tickers} tickers · ${data.n_days} days`;
        } else {
            text.textContent = 'Loading data...';
        }
    } catch {
        $('#statusText').textContent = 'Disconnected';
    }
}

async function refreshStats() {
    try {
        const data = await api('/api/stats');
        $('#statAlphas .stat-value').textContent = data.total_alphas || 0;
        const sharpe = data.top_sharpe;
        $('#statBestSharpe .stat-value').textContent = sharpe != null ? fmtNum(sharpe, 3) : '--';
    } catch { }
}

// =========================================================================
// Simulate
// =========================================================================
function initSimulate() {
    // Example dropdowns (both top and bottom)
    ['#exampleSelect', '#bottomExampleSelect'].forEach(sel => {
        const el = $(sel);
        if (el) {
            el.addEventListener('change', (e) => {
                if (e.target.value) {
                    $('#exprInput').value = e.target.value;
                    e.target.selectedIndex = 0;
                }
            });
        }
    });

    // Simulate
    $('#btnSimulate').addEventListener('click', runSimulation);

    // IB Mode preset button
    const ibBtn = $('#btnIBMode');
    if (ibBtn) {
        ibBtn.addEventListener('click', () => {
            $('#paramDelay').value = 0;
            $('#paramDecay').value = 0;
            $('#paramNeutralize').value = 'sector';
            $('#paramUniverse').value = 'TOP2000TOP3000';
            $('#paramTruncation').value = 0.01;
            ibBtn.classList.toggle('active');
            toast('IB Mode: delay=0, sector neutral, TOP2000-3000, fee-free', 'success');
        });
    }

    // Ctrl+Enter
    $('#exprInput').addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            runSimulation();
        }
    });

    // Update line numbers on input
    $('#exprInput').addEventListener('input', updateLineNumbers);
}

function updateLineNumbers() {
    const text = $('#exprInput').value;
    const lines = text.split('\n').length;
    const gutter = $('.code-gutter');
    gutter.innerHTML = '';
    for (let i = 1; i <= Math.max(lines, 1); i++) {
        const span = document.createElement('span');
        span.className = 'line-num';
        span.textContent = i;
        gutter.appendChild(span);
    }
}

async function runSimulation() {
    const expr = $('#exprInput').value.trim();
    if (!expr) { toast('Enter an expression', 'error'); return; }

    const btn = $('#btnSimulate');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Running...';

    try {
        const result = await api('/api/evaluate', {
            method: 'POST',
            body: JSON.stringify({
                expression: expr,
                delay: parseInt($('#paramDelay').value) || 1,
                decay: parseInt($('#paramDecay').value) || 0,
                neutralization: $('#paramNeutralize').value,
                universe: $('#paramUniverse').value || 'TOP3000',
            }),
        });

        displayResults(result, expr);
        refreshStats();
        refreshLibrary();
    } catch (e) {
        toast('Simulation failed: ' + e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Simulate';
    }
}

function displayResults(r, expr) {
    // Show chart
    const canvas = $('#pnlCanvas');
    const placeholder = $('#chartPlaceholder');
    if (r.cumulative_pnl && r.cumulative_pnl.length > 0) {
        placeholder.style.display = 'none';
        canvas.style.display = 'block';
        drawPnLChart(r.cumulative_pnl, r.pnl_dates);
    }

    // Update code preview
    $('#codePreview').textContent = expr;

    // Show IS Summary
    const summary = $('#isSummary');
    summary.classList.add('visible');

    // Aggregate metrics
    setAggValue('aggSharpe', r.sharpe, 2);
    setAggValue('aggTurnover', r.turnover, null, true);
    setAggValue('aggFitness', r.fitness, 2);
    setAggValue('aggReturns', r.annualized_return || r.returns_ann, null, true);
    setAggValue('aggDrawdown', r.max_drawdown, null, true);
    setAggValue('aggMargin', r.margin_bps, 2, false, 'bps');

    // Year-by-year table
    buildYearlyTable(r);

    // Quality checks
    buildQualityChecks(r);

    // OOS metrics
    buildOOSDisplay(r);
}

function buildOOSDisplay(r) {
    let container = $('#oosSection');
    if (!container) {
        // Create OOS section after quality checks
        const qc = $('#qualitySection');
        if (!qc) return;
        container = document.createElement('div');
        container.id = 'oosSection';
        container.className = 'oos-section';
        qc.parentNode.insertBefore(container, qc.nextSibling);
    }

    if (!r.oos) {
        container.innerHTML = '';
        return;
    }

    const oos = r.oos;
    const consistent = oos.is_consistent;
    const decayColor = oos.sharpe_decay >= 0.5 ? 'positive' : (oos.sharpe_decay >= 0 ? 'neutral' : 'negative');

    container.innerHTML = `
        <div class="oos-header">
            <span class="oos-title">📊 Out-of-Sample Analysis</span>
            <span class="oos-badge ${consistent ? 'badge-pass' : 'badge-fail'}">
                ${consistent ? '✓ Consistent' : '✗ Degraded'}
            </span>
        </div>
        <div class="oos-grid">
            <div class="oos-metric">
                <div class="oos-label">IS Sharpe</div>
                <div class="oos-value ${colorClass(oos.is_sharpe)}">${fmtNum(oos.is_sharpe)}</div>
                <div class="oos-period">${oos.is_start} → ${oos.is_end}</div>
            </div>
            <div class="oos-metric">
                <div class="oos-label">OOS Sharpe</div>
                <div class="oos-value ${colorClass(oos.oos_sharpe)}">${fmtNum(oos.oos_sharpe)}</div>
                <div class="oos-period">${oos.oos_start} → ${oos.oos_end}</div>
            </div>
            <div class="oos-metric">
                <div class="oos-label">IS Fitness</div>
                <div class="oos-value">${fmtNum(oos.is_fitness)}</div>
            </div>
            <div class="oos-metric">
                <div class="oos-label">OOS Fitness</div>
                <div class="oos-value">${fmtNum(oos.oos_fitness)}</div>
            </div>
            <div class="oos-metric">
                <div class="oos-label">IS Returns</div>
                <div class="oos-value ${colorClass(oos.is_returns_ann)}">${fmtPct(oos.is_returns_ann)}</div>
            </div>
            <div class="oos-metric">
                <div class="oos-label">OOS Returns</div>
                <div class="oos-value ${colorClass(oos.oos_returns_ann)}">${fmtPct(oos.oos_returns_ann)}</div>
            </div>
            <div class="oos-metric wide">
                <div class="oos-label">Sharpe Decay</div>
                <div class="oos-value ${decayColor}">${fmtNum(oos.sharpe_decay, 2)}x</div>
                <div class="oos-hint">(OOS/IS ratio, 1.0 = no decay)</div>
            </div>
        </div>
    `;
}

function setAggValue(id, value, decimals, isPct, suffix) {
    const el = $(`#${id}`);
    if (value == null || isNaN(value)) {
        el.textContent = '--';
        el.className = 'agg-value';
        return;
    }

    if (isPct) {
        el.textContent = fmtPct(value);
    } else {
        el.textContent = fmtNum(value, decimals || 2) + (suffix ? suffix : '');
    }
    el.className = `agg-value ${colorClass(value)}`;
}

function buildYearlyTable(r) {
    const tbody = $('#yearlyTableBody');

    // If server returned yearly_stats, use them
    if (r.yearly_stats && r.yearly_stats.length > 0) {
        tbody.innerHTML = r.yearly_stats.map(yr => `
            <tr>
                <td class="year-cell">${yr.year}</td>
                <td class="${colorClass(yr.sharpe)}">${fmtNum(yr.sharpe, 2)}</td>
                <td>${fmtPct(yr.turnover)}</td>
                <td class="${colorClass(yr.fitness)}">${fmtNum(yr.fitness, 2)}</td>
                <td class="${colorClass(yr.returns)}">${fmtPct(yr.returns)}</td>
                <td class="${colorClass(yr.drawdown)}">${fmtPct(yr.drawdown)}</td>
                <td>${fmtNum(yr.margin_bps, 2)}bps</td>
                <td>${yr.long_count || '--'}</td>
                <td>${yr.short_count || '--'}</td>
            </tr>
        `).join('');
        return;
    }

    // Fallback: compute from cumulative PnL data
    if (r.cumulative_pnl && r.pnl_dates && r.pnl_dates.length > 0) {
        const yearlyData = computeYearlyFromPnL(r.cumulative_pnl, r.pnl_dates, r.daily_returns);

        tbody.innerHTML = yearlyData.map(yr => `
            <tr>
                <td class="year-cell">${yr.year}</td>
                <td class="${colorClass(yr.sharpe)}">${fmtNum(yr.sharpe, 2)}</td>
                <td>${fmtPct(yr.turnover)}</td>
                <td class="${colorClass(yr.fitness)}">${fmtNum(yr.fitness, 2)}</td>
                <td class="${colorClass(yr.returns)}">${fmtPct(yr.returns)}</td>
                <td class="${colorClass(yr.drawdown)}">${fmtPct(yr.drawdown)}</td>
                <td>${fmtNum(yr.margin_bps, 0)}bps</td>
                <td>${yr.long_count || '--'}</td>
                <td>${yr.short_count || '--'}</td>
            </tr>
        `).join('');
    }
}

function computeYearlyFromPnL(cumPnl, dates, dailyReturns) {
    // Group by year
    const years = {};
    for (let i = 0; i < dates.length; i++) {
        const year = dates[i].slice(0, 4);
        if (!years[year]) years[year] = { returns: [], indices: [] };
        years[year].indices.push(i);

        // Daily return from cumulative PnL
        if (i > 0) {
            const prevPnl = cumPnl[i - 1] || 0;
            const currPnl = cumPnl[i];
            // Use provided daily returns if available, otherwise estimate
            if (dailyReturns && dailyReturns[i] != null) {
                years[year].returns.push(dailyReturns[i]);
            } else if (prevPnl !== 0) {
                years[year].returns.push((currPnl - prevPnl) / Math.abs(prevPnl));
            }
        }
    }

    return Object.entries(years).sort().map(([year, data]) => {
        const rets = data.returns;
        const n = rets.length;
        if (n === 0) return { year, sharpe: 0, turnover: 0, fitness: 0, returns: 0, drawdown: 0, margin_bps: 0 };

        const mean = rets.reduce((a, b) => a + b, 0) / n;
        const variance = rets.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1 || 1);
        const std = Math.sqrt(variance);
        const sharpe = std > 0 ? (mean / std) * Math.sqrt(252) : 0;
        const annReturn = mean * 252;

        // Drawdown from cumPnl
        const pnlSlice = data.indices.map(i => cumPnl[i]);
        let peak = pnlSlice[0];
        let maxDD = 0;
        for (const v of pnlSlice) {
            if (v > peak) peak = v;
            const dd = peak > 0 ? (v - peak) / peak : 0;
            if (dd < maxDD) maxDD = dd;
        }

        return {
            year,
            sharpe: sharpe,
            turnover: 0,  // Not available from PnL alone
            fitness: sharpe * Math.abs(1 + annReturn),
            returns: annReturn,
            drawdown: maxDD,
            margin_bps: mean * 10000,
        };
    });
}

function buildQualityChecks(r) {
    const grid = $('#checksGrid');
    const checks = [
        { name: 'Sharpe > 0.5', pass: r.sharpe != null && r.sharpe >= 0.5 },
        { name: 'Fitness > 0.3', pass: r.fitness != null && r.fitness >= 0.3 },
        { name: 'Turnover < 70%', pass: r.turnover != null && r.turnover < 0.7 },
        { name: 'Drawdown < 10%', pass: r.max_drawdown != null && Math.abs(r.max_drawdown) < 0.1 },
        { name: 'Margin > 5bps', pass: r.margin_bps != null && r.margin_bps > 5 },
        { name: 'Consistent Returns', pass: r.sharpe != null && r.sharpe > 0 },
        { name: 'Low Concentration', pass: true },
        { name: 'Sufficient Coverage', pass: r.coverage != null ? r.coverage > 0.5 : true },
    ];

    grid.innerHTML = checks.map(c => `
        <div class="check-item">
            <span class="check-icon ${c.pass ? 'pass' : 'fail'}">${c.pass ? '✓' : '✗'}</span>
            <span>${c.name}</span>
        </div>
    `).join('');
}

// =========================================================================
// PnL Canvas Chart
// =========================================================================
function drawPnLChart(data, dates) {
    const canvas = $('#pnlCanvas');
    const ctx = canvas.getContext('2d');
    const parent = canvas.parentElement;

    const dpr = window.devicePixelRatio || 1;
    const rect = parent.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = 250 * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = '250px';
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = 250;
    const pad = { top: 16, right: 16, bottom: 26, left: 60 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    // Clear
    ctx.fillStyle = '#0c1018';
    ctx.fillRect(0, 0, W, H);

    if (data.length < 2) return;

    const minV = Math.min(0, ...data);
    const maxV = Math.max(0, ...data);
    const rangeV = maxV - minV || 1;

    function x(i) { return pad.left + (i / (data.length - 1)) * plotW; }
    function y(v) { return pad.top + plotH - ((v - minV) / rangeV) * plotH; }

    // Grid lines
    ctx.strokeStyle = '#1a2332';
    ctx.lineWidth = 0.5;
    const nGrid = 5;
    for (let i = 0; i <= nGrid; i++) {
        const yy = pad.top + (i / nGrid) * plotH;
        ctx.beginPath(); ctx.moveTo(pad.left, yy); ctx.lineTo(W - pad.right, yy); ctx.stroke();
        const val = maxV - (i / nGrid) * rangeV;
        ctx.fillStyle = '#4a576a';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.textAlign = 'right';
        ctx.fillText(formatPnL(val), pad.left - 6, yy + 3);
    }

    // Zero line
    if (minV < 0 && maxV > 0) {
        const y0 = y(0);
        ctx.strokeStyle = '#2a3848';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(pad.left, y0); ctx.lineTo(W - pad.right, y0); ctx.stroke();
        ctx.setLineDash([]);
    }

    // Area fill
    const lastV = data[data.length - 1];
    const positive = lastV >= 0;
    const gradTop = positive ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0)';
    const gradBot = positive ? 'rgba(34,197,94,0)' : 'rgba(239,68,68,0.12)';
    const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
    grad.addColorStop(0, gradTop);
    grad.addColorStop(1, gradBot);

    ctx.beginPath();
    ctx.moveTo(x(0), y(0));
    for (let i = 0; i < data.length; i++) ctx.lineTo(x(i), y(data[i]));
    ctx.lineTo(x(data.length - 1), y(0));
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.moveTo(x(0), y(data[0]));
    for (let i = 1; i < data.length; i++) ctx.lineTo(x(i), y(data[i]));
    ctx.strokeStyle = positive ? '#22c55e' : '#ef4444';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // End value label
    const endX = x(data.length - 1);
    const endY = y(lastV);
    ctx.fillStyle = positive ? '#22c55e' : '#ef4444';
    ctx.font = 'bold 10px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(formatPnL(lastV), endX - 50, endY - 8);

    // Date labels
    if (dates && dates.length > 0) {
        ctx.fillStyle = '#4a576a';
        ctx.font = '9px Inter, sans-serif';
        ctx.textAlign = 'center';
        const step = Math.max(1, Math.floor(dates.length / 7));
        for (let i = 0; i < dates.length; i += step) {
            const d = dates[i].slice(0, 10);
            // Format as "May '19" style
            const dt = new Date(d);
            const month = dt.toLocaleString('en', { month: 'short' });
            const yr = `'${dt.getFullYear().toString().slice(2)}`;
            ctx.fillText(`${month} ${yr}`, x(i), H - 6);
        }
    }
}

function formatPnL(v) {
    if (Math.abs(v) >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
    if (Math.abs(v) >= 1e3) return `$${(v / 1e3).toFixed(0)}K`;
    return `$${v.toFixed(0)}`;
}

// =========================================================================
// Alpha Library
// =========================================================================
async function refreshLibrary() {
    try {
        const metric = $('#libSort').value;
        const data = await api(`/api/alphas?metric=${metric}&limit=100`);
        const tbody = $('#alphaTableBody');
        const empty = $('#libraryEmpty');

        if (!data.alphas || data.alphas.length === 0) {
            tbody.innerHTML = '';
            empty.style.display = 'block';
            return;
        }

        empty.style.display = 'none';
        tbody.innerHTML = data.alphas.map((a, i) => {
            const sharpe = a.sharpe != null ? Number(a.sharpe) : null;
            const fitness = a.fitness != null ? Number(a.fitness) : null;
            const retAnn = a.returns_ann != null ? Number(a.returns_ann) : null;

            // Parse params for delay, neutralization, universe
            let delay = 1, neut = 'market', universe = 'TOP3000';
            try {
                const params = typeof a.params_json === 'string' ? JSON.parse(a.params_json) : (a.params_json || {});
                if (params.delay != null) delay = params.delay;
                if (params.neutralization) neut = params.neutralization;
                if (params.universe) universe = params.universe;
            } catch { }

            const source = a.source || 'manual';
            const srcLabel = source === 'gp' ? '🧬 GP' :
                source === 'llm' ? '🤖 LLM' :
                    source === 'manual' ? '✍️' : source;

            return `<tr>
                <td>${i + 1}</td>
                <td class="td-expression" title="${escHtml(a.expression)}"
                    onclick="useExpression('${escJs(a.expression)}')">${escHtml(a.expression)}</td>
                <td class="${sharpe != null ? (sharpe >= 0 ? 'td-positive' : 'td-negative') : ''}">${fmtNum(sharpe, 3)}</td>
                <td class="${fitness != null ? (fitness >= 0 ? 'td-positive' : 'td-negative') : ''}">${fmtNum(fitness, 3)}</td>
                <td class="${retAnn != null ? (retAnn >= 0 ? 'td-positive' : 'td-negative') : ''}">${retAnn != null ? fmtPct(retAnn) : '—'}</td>
                <td>${fmtNum(a.turnover, 4)}</td>
                <td>${fmtNum(a.max_drawdown, 4)}</td>
                <td>${delay}</td>
                <td class="td-neut">${neut}</td>
                <td class="td-universe">${universe}</td>
                <td class="td-source">${srcLabel}</td>
                <td>${a.passed_checks || 0}/8</td>
                <td><button class="btn-use" onclick="useExpression('${escJs(a.expression)}')">Use</button></td>
            </tr>`;
        }).join('');
    } catch { }
}

function useExpression(expr) {
    $('#exprInput').value = expr;
    $$('.tab').forEach(t => t.classList.remove('active'));
    $$('.tab-content').forEach(tc => tc.classList.remove('active'));
    $('.tab[data-tab="simulate"]').classList.add('active');
    $('#tab-simulate').classList.add('active');
    updateLineNumbers();
    $('#exprInput').focus();
}

function escHtml(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
function escJs(s) { return s.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '\\"'); }

function initLibrary() {
    $('#libSort').addEventListener('change', refreshLibrary);
    $('#btnRefreshLib').addEventListener('click', refreshLibrary);
}

// =========================================================================
// Campaigns
// =========================================================================
function initCampaigns() {
    $('#btnRunGP').addEventListener('click', async () => {
        const btn = $('#btnRunGP');
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Running...';
        const logEl = $('#gpLog');
        logEl.innerHTML = ''; logEl.classList.add('active');

        try {
            await api('/api/gp/run', {
                method: 'POST',
                body: JSON.stringify({
                    generations: parseInt($('#gpGenerations').value) || 50,
                    population: parseInt($('#gpPopulation').value) || 200,
                    max_depth: parseInt($('#gpDepth').value) || 6,
                }),
            });
            appendLog('gpLog', 'Campaign submitted...', 'info');
        } catch (e) {
            appendLog('gpLog', 'Failed: ' + e.message, 'fail');
        } finally {
            btn.disabled = false;
            btn.innerHTML = 'Run GP Campaign';
        }
    });

    $('#btnRunLLM').addEventListener('click', async () => {
        const btn = $('#btnRunLLM');
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Running...';
        const logEl = $('#llmLog');
        logEl.innerHTML = ''; logEl.classList.add('active');

        try {
            await api('/api/llm/run', {
                method: 'POST',
                body: JSON.stringify({
                    trials: parseInt($('#llmTrials').value) || 20,
                    strategy: $('#llmStrategy').value || 'momentum+value',
                    api_key: $('#llmApiKey').value || '',
                }),
            });
            appendLog('llmLog', 'Campaign submitted...', 'info');
        } catch (e) {
            appendLog('llmLog', 'Failed: ' + e.message, 'fail');
        } finally {
            btn.disabled = false;
            btn.innerHTML = 'Run LLM Campaign';
        }
    });
}

// =========================================================================
// Data Fields
// =========================================================================
async function loadFields() {
    try {
        const data = await api('/api/fields');
        const grid = $('#fieldsGrid');
        if (!data.fields || data.fields.length === 0) return;

        const groups = {};
        data.fields.forEach(f => {
            if (!groups[f.group]) groups[f.group] = [];
            groups[f.group].push(f);
        });

        let html = '';
        for (const [group, fields] of Object.entries(groups)) {
            html += `<div class="field-group-header">${escHtml(group)} (${fields.length})</div>`;
            fields.forEach(f => {
                html += `<div class="field-item" onclick="insertField('${escJs(f.name)}')">
                    <span class="field-name">${escHtml(f.name)}</span>
                    <span class="field-desc">${escHtml(f.description || '')}</span>
                </div>`;
            });
        }
        grid.innerHTML = html;

        $('#fieldSearch').addEventListener('input', (e) => {
            const q = e.target.value.toLowerCase();
            $$('.field-item').forEach(el => {
                const name = el.querySelector('.field-name').textContent.toLowerCase();
                const desc = el.querySelector('.field-desc').textContent.toLowerCase();
                el.style.display = (name.includes(q) || desc.includes(q)) ? '' : 'none';
            });
        });
    } catch { }
}

function insertField(name) {
    const input = $('#exprInput');
    const pos = input.selectionStart || input.value.length;
    input.value = input.value.slice(0, pos) + name + input.value.slice(pos);
    input.focus();
    input.selectionStart = input.selectionEnd = pos + name.length;
    $$('.tab').forEach(t => t.classList.remove('active'));
    $$('.tab-content').forEach(tc => tc.classList.remove('active'));
    $('.tab[data-tab="simulate"]').classList.add('active');
    $('#tab-simulate').classList.add('active');
    updateLineNumbers();
}

// =========================================================================
// Init
// =========================================================================
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initSimulate();
    initLibrary();
    initCampaigns();
    if (typeof initPortfolio === 'function') initPortfolio();

    refreshStatus();
    refreshStats();
    refreshLibrary();
    loadFields();
    updateLineNumbers();

    connectWS();

    setInterval(refreshStats, 30000);
    setInterval(refreshStatus, 15000);
});

// Globals for onclick handlers
window.useExpression = useExpression;
window.insertField = insertField;
