// =========================================================================
// Portfolio Optimization
// =========================================================================

const PALETTE = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'];

let portAlphas = [];
let portResult = null;
let portAllResults = null;

function initPortfolio() {
    $('#portRefreshAlphas').addEventListener('click', loadPortfolioAlphas);
    $('#portSelectAll').addEventListener('click', function () { toggleAllPortAlphas(true); });
    $('#portSelectNone').addEventListener('click', function () { toggleAllPortAlphas(false); });
    $('#btnOptimize').addEventListener('click', runPortfolioOptimization);
}

async function loadPortfolioAlphas() {
    try {
        const data = await api('/api/alphas?limit=50&metric=sharpe');
        portAlphas = data.alphas || [];
        renderPortfolioAlphaList();
    } catch (e) {
        toast('Failed to load alphas: ' + e.message, 'error');
    }
}

function renderPortfolioAlphaList() {
    const container = $('#portAlphaList');
    if (!portAlphas.length) {
        container.innerHTML = '<div class="portfolio-empty">No alphas found. Run simulations first.</div>';
        updatePortCount();
        return;
    }
    container.innerHTML = portAlphas.map(function (a, i) {
        const expr = a.expression || a.alpha_expression || '??';
        const sharpe = (a.sharpe || 0).toFixed(2);
        const cls = +sharpe < 0 ? 'negative' : '';
        return '<div class="portfolio-alpha-item" onclick="togglePortAlpha(' + i + ')">' +
            '<input type="checkbox" id="portCheck' + i + '" checked>' +
            '<span class="portfolio-alpha-expr" title="' + escHtml(expr) + '">' + escHtml(expr) + '</span>' +
            '<span class="portfolio-alpha-sharpe ' + cls + '">' + sharpe + '</span>' +
            '</div>';
    }).join('');
    updatePortCount();
}

function togglePortAlpha(i) {
    var cb = document.getElementById('portCheck' + i);
    cb.checked = !cb.checked;
    updatePortCount();
}

function toggleAllPortAlphas(state) {
    portAlphas.forEach(function (_, i) {
        var cb = document.getElementById('portCheck' + i);
        if (cb) cb.checked = state;
    });
    updatePortCount();
}

function updatePortCount() {
    var count = 0;
    portAlphas.forEach(function (_, i) {
        var cb = document.getElementById('portCheck' + i);
        if (cb && cb.checked) count++;
    });
    $('#portSelectedCount').textContent = count;
}

function getSelectedAlphaIds() {
    var ids = [];
    portAlphas.forEach(function (a, i) {
        var cb = document.getElementById('portCheck' + i);
        if (cb && cb.checked) ids.push(a.alpha_id || a.id);
    });
    return ids;
}

async function runPortfolioOptimization() {
    var ids = getSelectedAlphaIds();
    if (ids.length < 2) {
        toast('Select at least 2 alphas for portfolio optimization', 'warning');
        return;
    }

    var btn = $('#btnOptimize');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Optimizing...';

    var compareAll = $('#portCompareAll').checked;

    try {
        var data = await api('/api/portfolio/optimize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                alpha_ids: ids,
                method: $('#portMethod').value,
                booksize: +$('#portBooksize').value,
                compare_all: compareAll,
            }),
        });

        if (data.error) {
            toast('Optimization failed: ' + data.error, 'error');
        } else if (data.compare_all) {
            portAllResults = data;
            var firstMethod = $('#portMethod').value;
            portResult = data.methods[firstMethod] || Object.values(data.methods)[0];
            renderPortfolioResults(data);
        } else {
            portResult = data;
            portAllResults = null;
            renderSinglePortfolioResult(data);
        }
    } catch (e) {
        toast('Portfolio optimization error: ' + e.message, 'error');
    }

    btn.disabled = false;
    btn.innerHTML = '\ud83d\ude80 Optimize Portfolio';
}

function renderPortfolioResults(data) {
    var container = $('#portResults');
    var html = '';

    html += '<h3>\ud83d\udcca Method Comparison</h3>';
    html += '<div class="port-method-cards">';
    var methods = data.methods;
    for (var method in methods) {
        if (!methods.hasOwnProperty(method)) continue;
        var result = methods[method];
        var isActive = (result === portResult);
        var sharpe = (result.sharpe || 0).toFixed(2);
        var color = +sharpe >= 0 ? 'var(--positive)' : 'var(--negative)';
        var dd = ((result.max_drawdown || 0) * 100).toFixed(1);
        var ret = ((result.returns_ann || 0) * 100).toFixed(1);
        var label = method.replace(/_/g, ' ').replace(/\b\w/g, function (c) { return c.toUpperCase(); });
        html += '<div class="port-method-card ' + (isActive ? 'active' : '') + '" onclick="selectPortMethod(\'' + method + '\')">';
        html += '<div class="port-method-name">' + label + '</div>';
        html += '<div class="port-method-sharpe" style="color:' + color + '">' + sharpe + '</div>';
        html += '<div class="port-method-label">Sharpe Ratio</div>';
        html += '<div class="port-method-metrics"><span>Ret: ' + ret + '%</span><span>DD: ' + dd + '%</span></div>';
        html += '</div>';
    }
    html += '</div>';

    html += buildPortfolioDetail(portResult, data.alpha_info);
    container.innerHTML = html;

    if (portResult.cumulative_pnl && portResult.pnl_dates) {
        setTimeout(function () { drawPortPnLChart(portResult.cumulative_pnl, portResult.pnl_dates); }, 100);
    }
}

function renderSinglePortfolioResult(data) {
    var container = $('#portResults');
    container.innerHTML = buildPortfolioDetail(data, data.alpha_info);
    if (data.cumulative_pnl && data.pnl_dates) {
        setTimeout(function () { drawPortPnLChart(data.cumulative_pnl, data.pnl_dates); }, 100);
    }
}

function selectPortMethod(method) {
    if (!portAllResults) return;
    portResult = portAllResults.methods[method];
    renderPortfolioResults(portAllResults);
}

function buildPortfolioDetail(result, alphaInfo) {
    var html = '';

    // PnL chart
    html += '<div class="port-chart-section">';
    html += '<h3>\ud83d\udcc8 Combined Portfolio PnL</h3>';
    html += '<div class="port-chart-area"><canvas id="portPnlCanvas"></canvas></div>';
    html += '</div>';

    html += '<div class="port-detail-grid">';

    // Weights
    html += '<div class="port-detail-card"><h4>\u2696 Alpha Weights</h4>';
    if (result.weights) {
        var wsorted = Object.entries(result.weights).sort(function (a, b) { return b[1] - a[1]; });
        wsorted.forEach(function (entry, i) {
            var name = entry[0], w = entry[1];
            var pct = (w * 100).toFixed(1);
            var ci = i % PALETTE.length;
            var expr = alphaInfo ? (alphaInfo.find(function (a) { return a.name === name; }) || {}).expression || name : name;
            html += '<div class="port-weight-bar" title="' + escHtml(expr) + '">';
            html += '<span class="port-weight-name">' + name + '</span>';
            html += '<div class="port-weight-track"><div class="port-weight-fill" style="width:' + pct + '%;background:' + PALETTE[ci] + '"></div></div>';
            html += '<span class="port-weight-pct">' + pct + '%</span>';
            html += '</div>';
        });
    }
    html += '</div>';

    // PnL Contribution
    html += '<div class="port-detail-card"><h4>\ud83d\udcb0 PnL Contribution</h4>';
    if (result.alpha_contributions) {
        var csorted = Object.entries(result.alpha_contributions).sort(function (a, b) { return b[1] - a[1]; });
        csorted.forEach(function (entry, i) {
            var name = entry[0], pct = entry[1];
            var ci = i % PALETTE.length;
            var absPct = Math.abs(pct);
            var clr = pct >= 0 ? 'var(--positive)' : 'var(--negative)';
            html += '<div class="port-weight-bar">';
            html += '<span class="port-weight-name">' + name + '</span>';
            html += '<div class="port-weight-track"><div class="port-weight-fill" style="width:' + Math.min(absPct, 100) + '%;background:' + PALETTE[ci] + '"></div></div>';
            html += '<span class="port-weight-pct" style="color:' + clr + '">' + pct.toFixed(1) + '%</span>';
            html += '</div>';
        });
    }
    html += '</div>';
    html += '</div>'; // end detail-grid

    // OOS section
    if (result.oos_sharpe != null) {
        var oosColor = result.oos_sharpe >= 0 ? 'var(--positive)' : 'var(--negative)';
        var isColor = result.sharpe >= 0 ? 'var(--positive)' : 'var(--negative)';
        html += '<div class="port-oos-section">';
        html += '<h4>\ud83d\udcca Out-of-Sample Analysis</h4>';
        html += '<div class="port-oos-grid">';
        html += '<div class="port-oos-metric"><div class="metric-label">IS Sharpe</div><div class="metric-value" style="color:' + isColor + '">' + (result.sharpe || 0).toFixed(2) + '</div></div>';
        html += '<div class="port-oos-metric"><div class="metric-label">OOS Sharpe</div><div class="metric-value" style="color:' + oosColor + '">' + (result.oos_sharpe || 0).toFixed(2) + '</div></div>';
        html += '<div class="port-oos-metric"><div class="metric-label">IS Returns</div><div class="metric-value">' + ((result.returns_ann || 0) * 100).toFixed(1) + '%</div></div>';
        html += '<div class="port-oos-metric"><div class="metric-label">OOS Returns</div><div class="metric-value">' + ((result.oos_returns_ann || 0) * 100).toFixed(1) + '%</div></div>';
        html += '</div></div>';
    }

    // Correlation
    if (result.correlation) {
        var corr = result.correlation;
        html += '<div class="port-detail-card" style="margin-bottom:24px">';
        html += '<h4>\ud83d\udd17 Alpha Correlations</h4>';
        html += '<div style="font-size:12px;color:var(--text-secondary);margin-bottom:8px">';
        html += 'Avg pairwise: <strong>' + (corr.avg_pairwise_corr || 0).toFixed(3) + '</strong> | ';
        html += 'Max pairwise: <strong>' + (corr.max_pairwise_corr || 0).toFixed(3) + '</strong>';
        html += '</div>';
        if (corr.highly_correlated && corr.highly_correlated.length > 0) {
            html += '<div style="font-size:11px;color:var(--text-muted)">';
            corr.highly_correlated.forEach(function (p) {
                html += '<div>\u26a0 ' + p.alpha_a + ' \u2194 ' + p.alpha_b + ': ' + p.corr.toFixed(3) + '</div>';
            });
            html += '</div>';
        } else {
            html += '<div style="font-size:11px;color:var(--positive)">\u2713 No highly correlated pairs (threshold 0.5)</div>';
        }
        html += '</div>';
    }

    // Summary bar
    var totalPnl = formatPnL(result.total_pnl || 0);
    var maxDD = ((result.max_drawdown || 0) * 100).toFixed(1);
    var fitness = (result.fitness || 0).toFixed(2);
    var pnlColor = (result.total_pnl || 0) >= 0 ? 'var(--positive)' : 'var(--negative)';
    html += '<div class="port-detail-card">';
    html += '<h4>\ud83d\udccb Portfolio Summary</h4>';
    html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;text-align:center">';
    html += '<div><div style="font-size:10px;color:var(--text-muted)">TOTAL PNL</div><div style="font-size:18px;font-weight:700;color:' + pnlColor + '">' + totalPnl + '</div></div>';
    html += '<div><div style="font-size:10px;color:var(--text-muted)">MAX DRAWDOWN</div><div style="font-size:18px;font-weight:700;color:var(--negative)">' + maxDD + '%</div></div>';
    html += '<div><div style="font-size:10px;color:var(--text-muted)">FITNESS</div><div style="font-size:18px;font-weight:700">' + fitness + '</div></div>';
    html += '<div><div style="font-size:10px;color:var(--text-muted)">ALPHAS</div><div style="font-size:18px;font-weight:700">' + Object.keys(result.weights || {}).length + '</div></div>';
    html += '</div></div>';

    return html;
}

function drawPortPnLChart(data, dates) {
    var canvas = document.getElementById('portPnlCanvas');
    if (!canvas) return;

    var rect = canvas.parentElement.getBoundingClientRect();
    var dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';

    var ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    var W = rect.width, H = rect.height;
    var pad = { top: 20, right: 20, bottom: 30, left: 70 };
    var pw = W - pad.left - pad.right;
    var ph = H - pad.top - pad.bottom;

    ctx.clearRect(0, 0, W, H);
    if (!data || data.length < 2) return;

    var min = Math.min.apply(null, data);
    var max = Math.max.apply(null, data);
    var range = max - min || 1;

    function xPos(i) { return pad.left + (i / (data.length - 1)) * pw; }
    function yPos(v) { return pad.top + (1 - (v - min) / range) * ph; }

    // Zero line
    if (min < 0 && max > 0) {
        ctx.strokeStyle = 'rgba(255,255,255,0.1)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(pad.left, yPos(0));
        ctx.lineTo(W - pad.right, yPos(0));
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Gradient fill
    var grad = ctx.createLinearGradient(0, pad.top, 0, H - pad.bottom);
    var lastVal = data[data.length - 1];
    if (lastVal >= 0) {
        grad.addColorStop(0, 'rgba(0, 200, 150, 0.25)');
        grad.addColorStop(1, 'rgba(0, 200, 150, 0.02)');
    } else {
        grad.addColorStop(0, 'rgba(255, 60, 80, 0.02)');
        grad.addColorStop(1, 'rgba(255, 60, 80, 0.25)');
    }

    ctx.beginPath();
    ctx.moveTo(xPos(0), yPos(data[0]));
    for (var i = 1; i < data.length; i++) ctx.lineTo(xPos(i), yPos(data[i]));
    ctx.lineTo(xPos(data.length - 1), H - pad.bottom);
    ctx.lineTo(xPos(0), H - pad.bottom);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.moveTo(xPos(0), yPos(data[0]));
    for (var j = 1; j < data.length; j++) ctx.lineTo(xPos(j), yPos(data[j]));
    ctx.strokeStyle = lastVal >= 0 ? '#00c896' : '#ff3c50';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Y-axis labels
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '10px Inter';
    ctx.textAlign = 'right';
    for (var k = 0; k <= 4; k++) {
        var v = min + (range * k / 4);
        ctx.fillText(formatPnL(v), pad.left - 8, yPos(v) + 3);
    }

    // X-axis labels
    ctx.textAlign = 'center';
    if (dates) {
        var step = Math.max(1, Math.floor(dates.length / 6));
        for (var m = 0; m < dates.length; m += step) {
            ctx.fillText(dates[m], xPos(m), H - 8);
        }
    }
}

window.selectPortMethod = selectPortMethod;
window.togglePortAlpha = togglePortAlpha;
