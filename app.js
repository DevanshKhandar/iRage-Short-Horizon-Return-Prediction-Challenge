/**
 * ReturnPredictor — Quantitative Analytics Dashboard
 * Professional chart rendering and data binding
 */

// Chart.js defaults
Chart.defaults.color = '#555570';
Chart.defaults.borderColor = 'rgba(255,255,255,0.03)';
Chart.defaults.font.family = "'IBM Plex Sans', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 8;
Chart.defaults.plugins.legend.labels.padding = 14;

const C = {
    green: '#00d47e', greenA: 'rgba(0,212,126,0.2)',
    blue: '#4d8dff', blueA: 'rgba(77,141,255,0.2)',
    amber: '#ffb830', amberA: 'rgba(255,184,48,0.2)',
    red: '#ff4757', redA: 'rgba(255,71,87,0.2)',
    cyan: '#00cec9', cyanA: 'rgba(0,206,201,0.2)',
    purple: '#8b5cf6', purpleA: 'rgba(139,92,246,0.2)',
};

const TOOLTIP = {
    backgroundColor: 'rgba(16,16,24,0.96)',
    borderColor: 'rgba(30,30,46,0.8)',
    borderWidth: 1,
    padding: 10,
    titleFont: { weight: '600', size: 11 },
    bodyFont: { size: 11 },
    cornerRadius: 4,
};

// ─── Data Loading ────────────────────────────────────────────
async function loadData() {
    const base = '../output/';
    try {
        const [cv, fi, pa, cfg] = await Promise.all([
            fetch(base + 'cv_results.json').then(r => r.json()),
            fetch(base + 'feature_importance.json').then(r => r.json()),
            fetch(base + 'predictions_analysis.json').then(r => r.json()),
            fetch(base + 'model_config.json').then(r => r.json()),
        ]);
        return { cv, fi, pa, cfg };
    } catch (e) {
        console.error('Load failed:', e);
        return null;
    }
}

// ─── KPI Rendering ───────────────────────────────────────────
function renderKPIs(cv, cfg) {
    const map = {
        r2: { val: cv.overall_r2, dec: 6 },
        rmse: { val: cv.overall_rmse, dec: 6 },
        mae: { val: cv.overall_mae, dec: 6 },
        features: { val: cv.n_features, dec: 0 },
        models: { val: cfg.total_models || 45, dec: 0 },
        train: { val: cv.n_train, dec: 0 },
    };

    document.querySelectorAll('.kpi-value').forEach(el => {
        const key = el.dataset.target;
        if (map[key]) {
            const { val, dec } = map[key];
            animateCounter(el, val, dec);
        }
    });
}

function animateCounter(el, target, decimals, duration = 1200) {
    const start = performance.now();
    const format = v => {
        if (decimals === 0 && target >= 1000) return Math.round(v).toLocaleString();
        return v.toFixed(decimals);
    };
    function tick(now) {
        const p = Math.min((now - start) / duration, 1);
        const ease = 1 - Math.pow(1 - p, 3);
        el.textContent = format(target * ease);
        if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

// ─── Fold Chart ──────────────────────────────────────────────
function renderFoldChart(cv) {
    const ctx = document.getElementById('foldChart').getContext('2d');
    const folds = cv.folds;
    const meanR2 = cv.mean_r2;

    document.getElementById('mean-r2-badge').textContent =
        `Mean: ${meanR2.toFixed(6)} \u00B1 ${cv.std_r2.toFixed(6)}`;

    const colors = [C.blue, C.cyan, C.green, C.purple, C.amber];

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: folds.map(f => `F${f.fold}`),
            datasets: [
                {
                    label: 'R\u00B2',
                    data: folds.map(f => f.r2),
                    backgroundColor: folds.map((_, i) => colors[i % colors.length] + '88'),
                    borderColor: folds.map((_, i) => colors[i % colors.length]),
                    borderWidth: 1,
                    borderRadius: 3,
                    barPercentage: 0.55,
                },
                {
                    label: 'Mean',
                    data: folds.map(() => meanR2),
                    type: 'line',
                    borderColor: C.red,
                    borderWidth: 1.5,
                    borderDash: [4, 3],
                    pointRadius: 0,
                    fill: false,
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: true, position: 'top' }, tooltip: TOOLTIP },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { callback: v => v.toFixed(4), font: { family: "'IBM Plex Mono'" } } },
                x: { grid: { display: false }, ticks: { font: { family: "'IBM Plex Mono'", size: 10 } } }
            }
        }
    });
}

// ─── Fold Table ──────────────────────────────────────────────
function renderFoldTable(cv) {
    const el = document.getElementById('fold-table');
    let html = `<div class="dt-row dt-head"><div>FOLD</div><div>R\u00B2</div><div>RMSE</div><div>MAE</div></div>`;
    cv.folds.forEach(f => {
        html += `<div class="dt-row">
            <div class="dt-fold">#${f.fold}</div>
            <div class="dt-r2">${f.r2.toFixed(6)}</div>
            <div class="dt-rmse">${f.rmse.toFixed(6)}</div>
            <div class="dt-mae">${f.mae.toFixed(6)}</div>
        </div>`;
    });
    html += `<div class="dt-row dt-summary">
        <div class="dt-fold">OOF</div>
        <div class="dt-r2">${cv.overall_r2.toFixed(6)}</div>
        <div class="dt-rmse">${cv.overall_rmse.toFixed(6)}</div>
        <div class="dt-mae">${cv.overall_mae.toFixed(6)}</div>
    </div>`;
    el.innerHTML = html;
}

// ─── Feature Chart ───────────────────────────────────────────
let featureChart = null;
function renderFeatureChart(fi, type = 'gain') {
    const ctx = document.getElementById('featureChart').getContext('2d');
    const n = 30;
    const features = fi.features.slice(0, n).reverse();
    const values = fi[type].slice(0, n).reverse();
    const maxV = Math.max(...values);

    if (featureChart) featureChart.destroy();

    featureChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features,
            datasets: [{
                data: values,
                backgroundColor: values.map(v => {
                    const r = v / maxV;
                    if (r > 0.6) return C.green + 'cc';
                    if (r > 0.3) return C.cyan + '99';
                    return C.blue + '66';
                }),
                borderRadius: 2,
                barPercentage: 0.65,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { ...TOOLTIP, callbacks: { label: c => `${type}: ${c.parsed.x.toLocaleString()}` } }
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { font: { family: "'IBM Plex Mono'", size: 10 } } },
                y: { grid: { display: false }, ticks: { font: { family: "'IBM Plex Mono'", size: 9 } } }
            }
        }
    });
}

// ─── Distribution Chart ──────────────────────────────────────
function renderDistChart(pa) {
    const ctx = document.getElementById('distChart').getContext('2d');
    const a = pa.actual_hist, p = pa.pred_hist;
    const labels = a.bin_edges.slice(0, -1).map((b, i) => ((b + a.bin_edges[i+1]) / 2).toFixed(3));

    new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                { label: 'Actual', data: a.counts, borderColor: C.blue, backgroundColor: C.blueA, fill: true, tension: 0.4, pointRadius: 0, borderWidth: 1.5 },
                { label: 'Predicted', data: p.counts, borderColor: C.green, backgroundColor: C.greenA, fill: true, tension: 0.4, pointRadius: 0, borderWidth: 1.5 },
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { position: 'top' }, tooltip: TOOLTIP },
            scales: {
                x: { grid: { display: false }, ticks: { maxTicksLimit: 8, callback: function(v) { return parseFloat(this.getLabelForValue(v)).toFixed(2); } } },
                y: { grid: { color: 'rgba(255,255,255,0.02)' } }
            }
        }
    });
}

// ─── Scatter Chart ───────────────────────────────────────────
function renderScatterChart(pa) {
    const ctx = document.getElementById('scatterChart').getContext('2d');
    const { actual, predicted } = pa.scatter;
    const pts = actual.map((a, i) => ({ x: a, y: predicted[i] }));
    const all = [...actual, ...predicted];
    const mn = Math.min(...all), mx = Math.max(...all);

    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                { label: 'Predictions', data: pts, backgroundColor: C.blue + '44', pointRadius: 1.5 },
                { label: 'y=x', data: [{x:mn,y:mn},{x:mx,y:mx}], type: 'line', borderColor: C.red, borderWidth: 1.5, borderDash: [4,3], pointRadius: 0, fill: false },
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' },
                tooltip: { ...TOOLTIP, callbacks: { label: c => `Act: ${c.parsed.x.toFixed(4)}, Pred: ${c.parsed.y.toFixed(4)}` } }
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.02)' }, title: { display: true, text: 'Actual', color: '#3a3a54', font: { size: 10 } } },
                y: { grid: { color: 'rgba(255,255,255,0.02)' }, title: { display: true, text: 'Predicted', color: '#3a3a54', font: { size: 10 } } }
            }
        }
    });
}

// ─── Residual Chart ──────────────────────────────────────────
function renderResidualChart(pa) {
    const ctx = document.getElementById('residualChart').getContext('2d');
    const r = pa.residual_hist;
    const labels = r.bin_edges.slice(0,-1).map((b,i) => ((b+r.bin_edges[i+1])/2).toFixed(4));

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                data: r.counts,
                backgroundColor: r.counts.map((_, i) => parseFloat(labels[i]) < 0 ? C.redA : C.greenA),
                borderColor: r.counts.map((_, i) => parseFloat(labels[i]) < 0 ? C.red + '66' : C.green + '66'),
                borderWidth: 1,
                borderRadius: 1,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: TOOLTIP },
            scales: {
                x: { grid: { display: false }, ticks: { maxTicksLimit: 8, font: { size: 9 } } },
                y: { grid: { color: 'rgba(255,255,255,0.02)' } }
            }
        }
    });
}

// ─── Test Prediction Chart ───────────────────────────────────
function renderTestPredChart(pa) {
    const ctx = document.getElementById('testPredChart').getContext('2d');
    const h = pa.test_pred_hist;
    const labels = h.bin_edges.slice(0,-1).map((b,i) => ((b+h.bin_edges[i+1])/2).toFixed(4));

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                data: h.counts,
                backgroundColor: C.cyanA,
                borderColor: C.cyan + '66',
                borderWidth: 1,
                borderRadius: 1,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: TOOLTIP },
            scales: {
                x: { grid: { display: false }, ticks: { maxTicksLimit: 8, font: { size: 9 } } },
                y: { grid: { color: 'rgba(255,255,255,0.02)' } }
            }
        }
    });
}

// ─── Stats Tables ────────────────────────────────────────────
function renderStats(pa) {
    function fill(id, stats) {
        document.getElementById(id).innerHTML = Object.entries(stats)
            .map(([k,v]) => `<div class="stat-row"><span class="stat-label">${k}</span><span class="stat-value">${typeof v === 'number' ? v.toFixed(6) : v}</span></div>`)
            .join('');
    }
    fill('actual-stats', pa.actual_stats);
    fill('pred-stats', pa.pred_stats);
}

// ─── Config Grid ─────────────────────────────────────────────
function renderConfig(cfg) {
    const el = document.getElementById('config-grid');
    const items = {
        'Model Type': cfg.model_type || 'LightGBM Ensemble',
        'Configs': cfg.n_configs || 3,
        'Loss Functions': (cfg.loss_functions || []).join(', '),
        'CV Strategy': cfg.cv_strategy || 'GroupKFold',
        'N Folds': cfg.n_folds || 5,
        'Features': cfg.total_features,
        'Feature Selection': cfg.feature_selection || '--',
        'Target Processing': cfg.target_preprocessing || '--',
        'Multi-Seed': cfg.multi_seed ? 'Yes' : 'No',
        'Seeds/Config': cfg.seeds_per_config || 3,
        'Total Models': cfg.total_models || '--',
        'Optimal Weights': (cfg.optimal_weights || []).map(w => w.toFixed(2)).join(', '),
    };
    el.innerHTML = Object.entries(items)
        .map(([k,v]) => `<div class="config-item"><div class="config-key">${k}</div><div class="config-val">${v}</div></div>`)
        .join('');
}

// ─── Nav Scroll ──────────────────────────────────────────────
function setupNav() {
    const links = document.querySelectorAll('.tn-link');
    const panels = document.querySelectorAll('.panel');

    window.addEventListener('scroll', () => {
        let current = 'overview';
        panels.forEach(p => {
            if (p.getBoundingClientRect().top <= 100) current = p.id;
        });
        links.forEach(l => {
            l.classList.toggle('active', l.dataset.section === current);
        });
    });
}

// ─── Feature Toggle ──────────────────────────────────────────
function setupToggle(fi) {
    document.getElementById('btn-gain').addEventListener('click', () => {
        document.getElementById('btn-gain').classList.add('active');
        document.getElementById('btn-split').classList.remove('active');
        renderFeatureChart(fi, 'gain');
    });
    document.getElementById('btn-split').addEventListener('click', () => {
        document.getElementById('btn-split').classList.add('active');
        document.getElementById('btn-gain').classList.remove('active');
        renderFeatureChart(fi, 'split');
    });
}

// ─── Main ────────────────────────────────────────────────────
async function main() {
    setupNav();
    const data = await loadData();
    if (!data) return;

    const { cv, fi, pa, cfg } = data;
    renderKPIs(cv, cfg);
    renderFoldChart(cv);
    renderFoldTable(cv);
    renderFeatureChart(fi, 'gain');
    renderDistChart(pa);
    renderScatterChart(pa);
    renderResidualChart(pa);
    renderTestPredChart(pa);
    renderStats(pa);
    renderConfig(cfg);
    setupToggle(fi);
}

main();
