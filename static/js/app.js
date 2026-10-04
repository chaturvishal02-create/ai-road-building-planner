/* ==========================================================================
   app.js — the page's only controller
   --------------------------------------------------------------------------
   Everything on screen comes from the Flask JSON API: /meta, /metrics,
   /dataset, /dashboard and /predict. No number is written in this file, so a
   retrained model changes what the page shows without touching a line of
   JavaScript. The dashboard half of the page lives in dashboard.js; the KNN
   canvas lives in knn-demo.js.

   Money is formatted by inr() below, which is the exact twin of
   data_layer/schema.format_inr on the server. Keep the two in step.
   ========================================================================== */

(function () {
  'use strict';

  /* ------------------------------------------------------------------ *
   * Helpers
   * ------------------------------------------------------------------ */

  const $ = (selector, root) => (root || document).querySelector(selector);
  const $$ = (selector, root) => Array.prototype.slice.call((root || document).querySelectorAll(selector));

  function readJsonFromScript(id, fallback) {
    const node = document.getElementById(id);
    if (!node) return fallback;
    try {
      return JSON.parse(node.textContent || '{}');
    } catch (err) {
      return fallback;
    }
  }

  const TEAM = readJsonFromScript('team-data', {});
  const SETTINGS = readJsonFromScript('settings-data', {});

  /**
   * Twin of schema.format_inr: below 1 crore show lakh, at or above show crore,
   * and drop the decimals once the number is in four-digit crores.
   * Amounts travel through this file as rupee LAKH.
   */
  function inr(lakh) {
    let value = Number(lakh);
    if (!Number.isFinite(value)) return '—';
    const sign = value < 0 ? '-' : '';
    value = Math.abs(value);
    if (value < 100) return sign + '₹ ' + value.toFixed(1) + ' lakh';
    if (value < 100000) return sign + '₹ ' + (value / 100).toFixed(2) + ' crore';
    return sign + '₹ ' + Math.round(value / 100) + ' crore';
  }

  function pct(value, decimals) {
    const d = decimals === undefined ? 1 : decimals;
    return Number(value).toFixed(d) + '%';
  }

  function signedPct(value, decimals) {
    const d = decimals === undefined ? 2 : decimals;
    return (value > 0 ? '+' : '') + Number(value).toFixed(d) + '%';
  }

  function rate(value, decimals) {
    return Number(value).toFixed(decimals === undefined ? 4 : decimals);
  }

  function esc(value) {
    return String(value === null || value === undefined ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function setText(selector, value) {
    const node = $(selector);
    if (node) node.textContent = value === null || value === undefined ? '—' : String(value);
  }

  function show(selector, visible) {
    const node = $(selector);
    if (node) node.hidden = !visible;
  }

  function classList(selector, add, remove) {
    const node = $(selector);
    if (!node) return;
    (add || []).forEach((name) => node.classList.add(name));
    (remove || []).forEach((name) => node.classList.remove(name));
  }

  /* ---------------------------- toasts ---------------------------------- */

  function toast(message, kind) {
    const stack = $('#toast-stack');
    if (!stack) return;
    const box = document.createElement('div');
    box.className = 'toast ' + (kind === 'error' ? 'toast-error' : kind === 'ok' ? 'toast-ok' : 'badge-neutral');
    box.textContent = message;
    stack.appendChild(box);
    window.setTimeout(() => box.remove(), kind === 'error' ? 6500 : 4200);
  }

  function banner(message) {
    const node = $('#api-banner');
    if (!node) return;
    if (!message) {
      node.classList.add('hidden');
      node.textContent = '';
      return;
    }
    node.className = 'notice notice-saffron';
    node.setAttribute('role', 'status');
    node.textContent = message;
  }

  /* ---------------------------- fetching -------------------------------- */

  class RequestError extends Error {
    constructor(message, fields, status) {
      super(message);
      this.fields = fields || null;
      this.status = status;
    }
  }

  async function readResponse(response) {
    let body = null;
    try {
      body = await response.json();
    } catch (err) {
      body = null;
    }
    if (!response.ok) {
      const message = (body && body.error) ? body.error : 'Request failed (' + response.status + ')';
      throw new RequestError(message, body && body.fields, response.status);
    }
    return body;
  }

  async function getJson(url) {
    const response = await fetch(url, { headers: { Accept: 'application/json' } });
    return readResponse(response);
  }

  async function postJson(url, payload) {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(payload || {}),
    });
    return readResponse(response);
  }

  async function postMultipart(url, formData) {
    const response = await fetch(url, { method: 'POST', body: formData });
    return readResponse(response);
  }

  /* ------------------------------------------------------------------ *
   * State
   * ------------------------------------------------------------------ */

  const state = {
    meta: null,
    metrics: null,
    dashboard: null,
    page: 1,
    size: 25,
    sort: 'project_name',
    dir: 'asc',
    search: '',
    filters: { scheme_name: '', state: '', contractor_type: '', over_budget: '' },
    last: null,
    metricModel: 'rf',
  };

  /* ------------------------------------------------------------------ *
   * Theme, nav, reveal, team
   * ------------------------------------------------------------------ */

  function isDark() {
    return document.documentElement.classList.contains('dark');
  }

  function syncThemeButton() {
    const label = $('#theme-toggle [data-theme-label]');
    if (label) label.textContent = isDark() ? 'Light' : 'Dark';
    const button = $('#theme-toggle');
    if (button) button.setAttribute('aria-pressed', isDark() ? 'true' : 'false');
  }

  function initTheme() {
    syncThemeButton();
    const button = $('#theme-toggle');
    if (!button) return;
    button.addEventListener('click', () => {
      const dark = !isDark();
      document.documentElement.classList.toggle('dark', dark);
      try {
        window.localStorage.setItem('budget-theme', dark ? 'dark' : 'light');
      } catch (err) { /* private mode: the toggle still works for this page load */ }
      syncThemeButton();
      if (window.Charts) Charts.refreshTheme();
      if (window.KnnDemo) KnnDemo.refreshTheme();
    });
  }

  function initNav() {
    const links = $$('.nav-link');
    const sections = links
      // Only in-page anchors are scroll-spied; a link to another page (/building) is
      // not a selector, and querySelector would throw on it.
      .filter((link) => (link.getAttribute('href') || '').startsWith('#'))
      .map((link) => $(link.getAttribute('href')))
      .filter(Boolean);
    if (!('IntersectionObserver' in window) || !sections.length) return;
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        links.forEach((link) => {
          link.classList.toggle('active', link.getAttribute('href') === '#' + entry.target.id);
        });
      });
    }, { rootMargin: '-45% 0px -50% 0px', threshold: 0 });
    sections.forEach((section) => observer.observe(section));
  }

  function initReveal() {
    const nodes = $$('.reveal');
    if (!('IntersectionObserver' in window)) {
      nodes.forEach((node) => node.classList.add('is-visible'));
      return;
    }
    // Threshold 0, not a percentage: the ratio is of the *section's own area*, and the route
    // planner section is 6 700 px tall on a narrow window -- 8 % of it can never fit on any
    // screen, so the section stayed at opacity 0 and the map looked missing.
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0, rootMargin: '0px 0px -12% 0px' });
    nodes.forEach((node) => observer.observe(node));
  }

  function initTeam() {
    const host = $('#team-cards');
    if (!host) return;
    const members = [
      { role: 'Developed by', names: TEAM.developers || [], note: 'Dataset engineering, model training and the Flask API.' },
      { role: 'Under the guidance of', names: TEAM.guide ? [TEAM.guide] : [], note: 'Methodology review and validation of the reported metrics.' },
      { role: 'Institution', names: TEAM.institution ? [TEAM.institution] : [], note: TEAM.programme || '' },
    ];
    host.innerHTML = members.map((member) => (
      '<article class="card">' +
      '<p class="eyebrow">' + esc(member.role) + '</p>' +
      member.names.map((name) => '<h3 class="card-title mt-1">' + esc(name) + '</h3>').join('') +
      '<p class="mt-2 text-sm leading-relaxed text-[var(--muted)]">' + esc(member.note) + '</p>' +
      '</article>'
    )).join('');
  }

  /* ------------------------------------------------------------------ *
   * Animated counters
   * ------------------------------------------------------------------ */

  function animateNumber(node, target, format) {
    if (!node) return;
    const start = performance.now();
    const duration = 750;
    const shown = format || ((value) => String(Math.round(value)));
    function step(now) {
      const progress = Math.min(1, (now - start) / duration);
      // ease-out so the last digits settle rather than slam into place
      const eased = 1 - Math.pow(1 - progress, 3);
      node.textContent = shown(target * eased);
      if (progress < 1) requestAnimationFrame(step);
      else node.textContent = shown(target);
    }
    requestAnimationFrame(step);
  }

  function renderCounters() {
    const totals = state.dashboard.totals;
    animateNumber($('#counter-projects'), totals.projects);
    animateNumber($('#counter-states'), totals.states);
    animateNumber($('#counter-districts'), totals.districts);
    animateNumber($('#counter-schemes'), totals.schemes);
    animateNumber($('#counter-sanctioned'), totals.sanctioned_lakh, (value) => inr(value));
    animateNumber($('#counter-overrun'), totals.over_budget_share_pct, (value) => pct(value, 2));
  }

  /* ------------------------------------------------------------------ *
   * /meta -> selects, sliders, source panel
   * ------------------------------------------------------------------ */

  function fillSelect(node, values, allLabel) {
    if (!node) return;
    const current = node.value;
    node.innerHTML = '';
    const all = document.createElement('option');
    all.value = '';
    all.textContent = allLabel || 'All';
    node.appendChild(all);
    values.forEach((value) => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = value;
      node.appendChild(option);
    });
    if (values.indexOf(current) >= 0) node.value = current;
  }

  function fillRequired(node, values) {
    if (!node) return;
    const current = node.value;
    node.innerHTML = '';
    values.forEach((value) => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = value;
      node.appendChild(option);
    });
    if (values.indexOf(current) >= 0) node.value = current;
  }

  function renderMeta() {
    const vocab = state.meta.vocabularies;

    fillRequired($('#in-scheme'), vocab.scheme_name);
    fillRequired($('#in-state'), vocab.state);
    fillRequired($('#in-contractor'), vocab.contractor_type);
    fillSelect($('#filter-scheme'), vocab.scheme_name, 'All schemes');
    fillSelect($('#filter-state'), vocab.state, 'All states');
    fillSelect($('#filter-contractor'), vocab.contractor_type, 'All');
    fillSelect($('#filter-target'), vocab.target_values, 'All');

    const kSlider = $('#in-k');
    if (kSlider) {
      kSlider.min = String(state.meta.k_range[0]);
      kSlider.max = String(state.meta.k_range[1]);
      kSlider.value = String(state.meta.default_k);
    }
    setText('#k-value', state.meta.default_k);

    const budget = $('#in-budget');
    if (budget) {
      budget.min = String(vocab.budget_min_lakh);
      budget.max = String(vocab.budget_max_lakh);
      budget.placeholder = String(Math.round((vocab.budget_min_lakh + vocab.budget_max_lakh) / 2 / 10) * 10);
      budget.title = 'Training range: ' + inr(vocab.budget_min_lakh) + ' to ' + inr(vocab.budget_max_lakh);
    }
    const duration = $('#in-duration');
    if (duration) {
      duration.min = String(vocab.duration_min);
      duration.max = String(vocab.duration_max);
      duration.placeholder = String(Math.round((vocab.duration_min + vocab.duration_max) / 2));
      duration.title = 'Training range: ' + vocab.duration_min + '-' + vocab.duration_max + ' months';
    }

    const modelSelect = $('#in-model');
    if (modelSelect) {
      const auto = modelSelect.querySelector('option[value="auto"]');
      if (auto) auto.textContent = 'Best model (auto — ' + state.meta.comparison.winner_label + ')';
      modelSelect.value = 'auto';
    }

    renderSourcePanel();
  }

  function renderSourcePanel() {
    const sources = state.meta.sources || [];
    const active = sources.filter((entry) => entry.active)[0] || sources[0] || {};
    const line = $('#source-line');
    if (line) {
      line.textContent = 'Serving ' + state.meta.dataset.rows + ' rows through the '
        + active.name + ' adapter. ' + (state.dashboard ? state.dashboard.source.ref : '');
    }
    const select = $('#source-store');
    if (select) select.value = active.name || 'local';
    const ref = $('#source-ref');
    if (ref) {
      ref.value = active.detail || '';
      ref.placeholder = active.name === 'http' ? 'https://example.gov/dataset.csv'
        : active.name === 's3' ? 'bucket/key or key' : 'data/india_projects_2025_26.csv';
    }

    const notes = sources.map((entry) => {
      const mark = entry.configured ? 'ready' : 'not configured';
      const why = entry.configured ? (entry.writable ? ', writes enabled' : ', read-only') : ' — ' + (entry.reason || 'needs configuration');
      return '<strong>' + esc(entry.name) + '</strong> ' + esc(mark) + esc(why);
    }).join('<br>');
    setText2('#source-availability', notes);
  }

  function setText2(selector, html) {
    const node = $(selector);
    if (node) node.innerHTML = html;
  }

  /* ------------------------------------------------------------------ *
   * /dataset -> table, filters, paging
   * ------------------------------------------------------------------ */

  const SORT_LABELS = {
    project_name: 'Project', scheme_name: 'Scheme', district: 'District',
    contractor_type: 'Execution', sanctioned_budget_lakh: 'Sanctioned',
    duration_months: 'Months', actual_cost_lakh: 'Actual', overrun_pct: 'Overrun',
    over_budget: 'Outcome',
  };

  function datasetQuery() {
    const params = new URLSearchParams();
    params.set('page', state.page);
    params.set('size', state.size);
    params.set('sort', state.sort);
    params.set('dir', state.dir);
    if (state.search) params.set('search', state.search);
    Object.keys(state.filters).forEach((key) => {
      if (state.filters[key]) params.set(key, state.filters[key]);
    });
    return '/dataset?' + params.toString();
  }

  async function loadDataset() {
    const body = $('#dataset-body');
    try {
      const data = await getJson(datasetQuery());
      state.dataset = data;
      renderDatasetRows(data);
      renderPageInfo(data);
      markSortHeaders();
    } catch (err) {
      if (body) body.innerHTML = '<tr><td colspan="9" class="empty">' + esc(err.message) + '</td></tr>';
      toast('Dataset table: ' + err.message, 'error');
    }
  }

  function renderDatasetRows(data) {
    const body = $('#dataset-body');
    if (!body) return;
    if (!data.rows.length) {
      body.innerHTML = '<tr><td colspan="9" class="empty">No project matches those filters.</td></tr>';
      return;
    }
    body.innerHTML = data.rows.map((row) => (
      '<tr>' +
      nameCell(row.project_name) +
      '<td>' + esc(row.scheme_name) + '</td>' +
      '<td>' + esc(row.district) + '<span class="block text-xs text-[var(--muted)]">' + esc(row.state) + '</span></td>' +
      '<td class="hidden md:table-cell">' + esc(row.contractor_type) + '</td>' +
      '<td class="num">' + esc(row.sanctioned_inr) + '</td>' +
      '<td class="num">' + row.duration_months + '</td>' +
      '<td class="num">' + esc(row.actual_inr) + '</td>' +
      '<td class="num">' + signedPct(row.overrun_pct) + '</td>' +
      '<td>' + badge(row.over_budget) + '</td>' +
      '</tr>'
    )).join('');
  }

  /** The informative half of a project name is its work code, e.g. MH-JJM-2526-012. */
  function nameParts(name) {
    const text = String(name);
    const match = /\[Work ([^\]]+)\]\s*$/.exec(text);
    const code = match ? match[1] : '';
    let title = match ? text.slice(0, match.index).trim() : text;
    title = title.replace(/[,.\s]+$/, '');
    if (title.length > 52) title = title.slice(0, 49).replace(/[,.\s]+$/, '') + '…';
    return { title: title, code: code };
  }

  function nameCell(name) {
    const parts = nameParts(name);
    return '<td class="truncate-cell" title="' + esc(name) + '">' + esc(parts.title)
      + (parts.code ? '<span class="block text-xs text-[var(--muted)]">' + esc(parts.code) + '</span>' : '')
      + '</td>';
  }

  function badge(outcome) {
    return '<span class="badge ' + (outcome === 'Yes' ? 'badge-yes' : 'badge-no') + '">'
      + (outcome === 'Yes' ? 'Over sanction' : 'Within sanction') + '</span>';
  }

  function renderPageInfo(data) {
    const from = data.total === 0 ? 0 : (data.page - 1) * data.size + 1;
    const to = Math.min(data.total, data.page * data.size);
    setText('#page-info', 'Showing ' + from + '-' + to + ' of ' + data.total
      + ' projects · page ' + data.page + ' of ' + data.pages);
    const prev = $('#page-prev');
    const next = $('#page-next');
    if (prev) prev.disabled = data.page <= 1;
    if (next) next.disabled = data.page >= data.pages;
  }

  function markSortHeaders() {
    $$('#dataset th[data-sort]').forEach((header) => {
      const field = header.getAttribute('data-sort');
      const base = SORT_LABELS[field] || field;
      if (field !== state.sort) {
        header.textContent = base;
        header.removeAttribute('aria-sort');
        return;
      }
      header.textContent = base + (state.dir === 'asc' ? ' ▲' : ' ▼');
      header.setAttribute('aria-sort', state.dir === 'asc' ? 'ascending' : 'descending');
    });
  }

  function initDatasetControls() {
    const search = $('#dataset-search');
    if (search) {
      let timer = null;
      search.addEventListener('input', () => {
        window.clearTimeout(timer);
        timer = window.setTimeout(() => {
          state.search = search.value.trim();
          state.page = 1;
          loadDataset();
        }, 260);
      });
    }

    const map = {
      '#filter-scheme': 'scheme_name',
      '#filter-state': 'state',
      '#filter-contractor': 'contractor_type',
      '#filter-target': 'over_budget',
    };
    Object.keys(map).forEach((selector) => {
      const node = $(selector);
      if (!node) return;
      node.addEventListener('change', () => {
        state.filters[map[selector]] = node.value;
        state.page = 1;
        loadDataset();
      });
    });

    const clear = $('#dataset-clear');
    if (clear) {
      clear.addEventListener('click', () => {
        state.search = '';
        state.filters = { scheme_name: '', state: '', contractor_type: '', over_budget: '' };
        state.page = 1;
        state.sort = 'project_name';
        state.dir = 'asc';
        if (search) search.value = '';
        Object.keys(map).forEach((selector) => { $(selector).value = ''; });
        loadDataset();
      });
    }

    $$('#dataset th[data-sort]').forEach((header) => {
      header.addEventListener('click', () => {
        const field = header.getAttribute('data-sort');
        if (state.sort === field) state.dir = state.dir === 'asc' ? 'desc' : 'asc';
        else { state.sort = field; state.dir = 'asc'; }
        state.page = 1;
        loadDataset();
      });
    });

    const prev = $('#page-prev');
    if (prev) prev.addEventListener('click', () => { state.page -= 1; loadDataset(); });
    const next = $('#page-next');
    if (next) next.addEventListener('click', () => { state.page += 1; loadDataset(); });
  }

  /* ------------------------------------------------------------------ *
   * Dataset charts (from /dashboard aggregates)
   * ------------------------------------------------------------------ */

  function renderDatasetCharts() {
    const d = state.dashboard;
    if (!window.Charts) return;
    Charts.render('scheme', 'chart-scheme', d.by_scheme.map((row) => ({
      label: row.scheme_name, value: row.projects,
    })));
    Charts.render('overrun', 'chart-overrun', d.by_scheme.map((row) => ({
      label: row.scheme_name, value: row.over_budget_share_pct,
    })));
    Charts.render('state', 'chart-state', d.by_state.map((row) => ({
      label: row.state, within: row.within_budget, over: row.over_budget,
    })));
    Charts.render('contractor', 'chart-contractor', d.by_contractor.map((row) => ({
      label: row.contractor_type, over: row.over_budget, within: row.projects - row.over_budget,
    })));
  }

  /* ------------------------------------------------------------------ *
   * /metrics -> the evaluation section
   * ------------------------------------------------------------------ */

  function renderPerformance() {
    const metrics = state.metrics;
    state.metricModel = metrics.comparison.winner;

    const compare = $('#compare-body');
    if (compare) {
      compare.innerHTML = metrics.comparison.rows.map((row) => (
        '<tr' + (row.key === metrics.comparison.winner ? ' class="font-semibold"' : '') + '>' +
        '<td>' + esc(row.label) + (row.key === metrics.comparison.winner
          ? ' <span class="chip chip-green">selected</span>' : '') + '</td>' +
        '<td class="num">' + rate(row.accuracy) + '</td>' +
        '<td class="num">' + rate(row.precision) + '</td>' +
        '<td class="num">' + rate(row.recall) + '</td>' +
        '<td class="num">' + rate(row.f1) + '</td>' +
        '<td class="num">' + rate(row.roc_auc) + '</td>' +
        '</tr>'
      )).join('');
    }

    setText('#selection-rule', 'Selection rule: ' + metrics.comparison.rule
      + '. Winner: ' + metrics.comparison.winner_label + '. Trained at '
      + metrics.trained_at + ' on ' + metrics.dataset.train_size + ' projects, scored on '
      + metrics.dataset.test_size + ' held-out projects; the encoder expands the six fields to '
      + metrics.model.encoded_width + ' columns.');

    $$('#performance .tab[data-metric-model]').forEach((tab) => {
      tab.classList.toggle('is-active', tab.getAttribute('data-metric-model') === state.metricModel);
      tab.addEventListener('click', () => {
        state.metricModel = tab.getAttribute('data-metric-model');
        $$('#performance .tab[data-metric-model]').forEach((other) => {
          other.classList.toggle('is-active', other === tab);
        });
        renderConfusion();
        renderMetricPanel();
        renderReport();
      });
    });

    renderConfusion();
    renderMetricPanel();
    renderReport();

    if (window.Charts) {
      Charts.render('kcurve', 'chart-kcurve', metrics.k_curve);
      Charts.render('importance', 'chart-importance-rf', metrics.importances.random_forest);
    }
    renderAblation();
  }

  function renderConfusion() {
    const host = $('#confusion-grid');
    if (!host) return;
    const model = state.metrics.models[state.metricModel];
    const labels = model.labels;
    const matrix = model.confusion_matrix;
    const maximum = Math.max.apply(null, matrix.map((row) => Math.max.apply(null, row)));

    const cell = (value, kind, caption) => (
      '<div class="heat-cell ' + kind + '">' +
      '<span class="heat-value">' + value + '</span>' +
      '<span class="heat-label">' + caption + '</span>' +
      '</div>'
    );

    const shade = (value, kind) => {
      const alpha = maximum ? 0.14 + 0.6 * (value / maximum) : 0.14;
      const rgb = kind === 'miss' ? '185, 28, 28' : '19, 136, 8';
      return 'rgba(' + rgb + ', ' + alpha.toFixed(3) + ')';
    };

    let html = '<div class="heat-grid">';
    html += '<div class="heat-axis"></div>';
    html += '<div class="heat-axis">Said ' + esc(labels[1]) + '</div>';
    html += '<div class="heat-axis">Said ' + esc(labels[0]) + '</div>';
    // Column order is the prediction axis; rows are the truth.
    [[matrix[0][1], matrix[0][0], 0],
      [matrix[1][1], matrix[1][0], 1]].forEach((rowInfo) => {
      const truth = rowInfo[2];
      html += '<div class="heat-axis">Truth ' + esc(labels[truth]) + '</div>';
      html += '<div class="heat-cell" style="background:' + shade(rowInfo[0], truth === 1 ? 'ok' : 'miss') + '">'
        + '<span class="heat-value">' + rowInfo[0] + '</span>'
        + '<span class="heat-label">' + (truth === 1 ? 'caught the overrun' : 'false alarm') + '</span></div>';
      html += '<div class="heat-cell" style="background:' + shade(rowInfo[1], truth === 1 ? 'miss' : 'ok') + '">'
        + '<span class="heat-value">' + rowInfo[1] + '</span>'
        + '<span class="heat-label">' + (truth === 1 ? 'missed overrun' : 'correctly within') + '</span></div>';
    });
    html += '</div>';
    host.innerHTML = html;
  }

  function renderMetricPanel() {
    const model = state.metrics.models[state.metricModel];
    setText('#m-model-name', model.model + ' · ' + state.metricModel.toUpperCase());
    setText('#m-accuracy', rate(model.accuracy) + ' (' + model.correct + '/' + model.total + ')');
    setText('#m-precision', rate(model.precision));
    setText('#m-recall', rate(model.recall));
    setText('#m-f1', rate(model.f1));
    setText('#m-auc', model.roc_auc === null ? 'n/a' : rate(model.roc_auc));
    setText('#m-correct', model.correct + ' of ' + model.total + ' test projects');
    setText('#m-baseline', rate(model.baseline_always_no_accuracy));
  }

  function renderReport() {
    const model = state.metrics.models[state.metricModel];
    const body = $('#report-body');
    if (!body) return;
    const keys = model.labels.concat(['macro avg', 'weighted avg']);
    body.innerHTML = keys.map((key) => {
      const row = model.report[key];
      if (!row) return '';
      const strong = key === 'Yes' ? ' class="font-semibold"' : '';
      return '<tr' + strong + '><td>' + esc(key === 'Yes' ? 'Yes — over sanction'
        : key === 'No' ? 'No — within sanction' : key) + '</td>'
        + '<td class="num">' + rate(row.precision) + '</td>'
        + '<td class="num">' + rate(row.recall) + '</td>'
        + '<td class="num">' + rate(row.f1) + '</td>'
        + '<td class="num">' + row.support + '</td></tr>';
    }).join('');
  }

  function renderAblation() {
    const host = $('#knn-ablation');
    if (!host) return;
    const ablation = state.metrics.importances.knn;
    host.innerHTML = '<li class="text-xs text-[var(--muted)]">' + esc(ablation.method)
      + '. Baseline accuracy ' + rate(ablation.baseline_accuracy) + '.</li>'
      + ablation.features.map((row) => (
        '<li class="metric"><dt>' + esc(row.feature) + '</dt>'
        + '<dd>' + signedPct(row.accuracy_drop * 100, 2)
        + ' <span class="text-xs text-[var(--muted)]">(blanked: ' + rate(row.accuracy_with_blank) + ')</span></dd></li>'
      )).join('');
  }

  /* ------------------------------------------------------------------ *
   * Predictor
   * ------------------------------------------------------------------ */

  function districtsFor(selectedState) {
    const vocab = state.meta.vocabularies;
    if (!selectedState) return vocab.district;
    return vocab.district.filter((district) => {
      const states = vocab.district_states[district] || [];
      return states.indexOf(selectedState) >= 0;
    });
  }

  function cascadeDistricts() {
    const selectedState = $('#in-state') ? $('#in-state').value : '';
    const district = $('#in-district');
    if (!district) return;
    const options = districtsFor(selectedState);
    const previous = district.value;
    district.innerHTML = '';
    options.forEach((value) => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = value;
      district.appendChild(option);
    });
    district.value = options.indexOf(previous) >= 0 ? previous : options[0];
  }

  function updateBudgetHint() {
    const node = $('#in-budget');
    const hint = $('#budget-live-hint');
    if (!node || !hint) return;
    const value = parseFloat(node.value);
    if (!Number.isFinite(value) || node.value === '') {
      hint.textContent = '';
      return;
    }
    const vocab = state.meta.vocabularies;
    let text = inr(value);
    if (value < vocab.budget_min_lakh || value > vocab.budget_max_lakh) {
      text += ' — outside the training range (' + inr(vocab.budget_min_lakh)
        + ' to ' + inr(vocab.budget_max_lakh) + '); the model extrapolates here.';
    } else {
      text += ' sanctioned';
    }
    hint.textContent = text;
  }

  function collectInputs() {
    return {
      scheme_name: $('#in-scheme').value,
      state: $('#in-state').value,
      district: $('#in-district').value,
      contractor_type: $('#in-contractor').value,
      sanctioned_budget_lakh: $('#in-budget').value,
      duration_months: $('#in-duration').value,
      k: $('#in-k').value,
      model: $('#in-model').value,
    };
  }

  function clearErrors() {
    $$('.field-error').forEach((node) => { node.textContent = ''; });
    $$('#predict-form .field-input').forEach((node) => node.removeAttribute('aria-invalid'));
    const list = $('#form-errors');
    if (list) { list.hidden = true; list.innerHTML = ''; }
  }

  function showErrors(message, fields) {
    const known = {
      scheme_name: '#in-scheme', state: '#in-state', district: '#in-district',
      contractor_type: '#in-contractor', sanctioned_budget_lakh: '#in-budget',
      duration_months: '#in-duration', model: '#in-model', k: '#in-k',
    };
    const list = $('#form-errors');
    if (fields && Object.keys(fields).length) {
      const items = Object.keys(fields).map((name) => {
        const input = $(known[name]);
        const slot = $('[data-error-for="' + name + '"]');
        if (slot) slot.textContent = fields[name];
        if (input) input.setAttribute('aria-invalid', 'true');
        return '<li>' + esc(fields[name]) + '</li>';
      });
      if (list) { list.innerHTML = items.join(''); list.hidden = false; }
      return;
    }
    if (list) { list.innerHTML = '<li>' + esc(message) + '</li>'; list.hidden = false; }
  }

  function setLoading(loading) {
    show('#predict-spinner', loading);
    const button = $('#predict-submit');
    if (button) {
      button.disabled = loading;
      button.classList.toggle('opacity-60', loading);
    }
    if (loading) {
      show('#predict-result', false);
      show('#predict-placeholder', false);
    }
  }

  function gaugeRisk(riskPercent) {
    const fill = $('#risk-gauge-fill');
    const circumference = 326.7;
    if (fill) {
      const fraction = Math.max(0, Math.min(100, riskPercent)) / 100;
      fill.setAttribute('stroke-dashoffset', String((circumference * (1 - fraction)).toFixed(1)));
      const styles = getComputedStyle(document.documentElement);
      const read = (name, fallback) => (styles.getPropertyValue(name) || '').trim() || fallback;
      fill.style.stroke = riskPercent >= 66 ? read('--danger', '#b91c1c')
        : riskPercent >= 34 ? read('--saffron', '#ff9933')
          : read('--india-green', '#138808');
    }
    setText('#result-risk', pct(riskPercent, 1));
  }

  function renderResult(result) {
    state.last = result;

    const card = $('#result-card');
    if (card) {
      card.classList.toggle('is-over', result.prediction === 'Yes');
      card.classList.toggle('is-within', result.prediction !== 'Yes');
    }
    setText('#result-verdict', result.verdict);
    const byModel = result.risk_pct_by_model;
    setText('#result-model-line', result.model_label + ' · K=' + result.k
      + ' · over-budget class risk: KNN ' + pct(byModel.knn, 1) + ', Random Forest ' + pct(byModel.rf, 1)
      + ' · compared against ' + result.train_size + ' training projects');

    gaugeRisk(result.risk_pct);

    setText('#cost-sanctioned', result.cost.sanctioned_inr);
    setText('#cost-expected', result.cost.expected_actual_inr);
    setText('#cost-overrun', result.cost.expected_overrun_inr + ' (' + signedPct(result.cost.expected_overrun_pct, 1) + ')');
    setText('#vote-line', result.votes.yes + ' over · ' + result.votes.no + ' within of ' + result.votes.total);

    setText('#result-explanation', result.explanation);
    setText('#specific-advisory', result.specific_advisory);

    const body = $('#neighbour-body');
    if (body) {
      body.innerHTML = result.neighbours.map((row) => (
        '<tr>' +
        '<td class="num">' + row.rank + '</td>' +
        nameCell(row.project_name) +
        '<td>' + esc(row.state) + ' / ' + esc(row.district) + '</td>' +
        '<td class="num">' + esc(row.sanctioned_inr) + '</td>' +
        '<td class="num">' + esc(row.actual_inr) + '</td>' +
        '<td>' + badge(row.over_budget) + '</td>' +
        '<td class="num">' + row.distance.toFixed(3) + '</td>' +
        '</tr>'
      )).join('');
    }

    show('#predict-spinner', false);
    show('#predict-placeholder', false);
    show('#predict-result', true);

    if (window.Dashboard) Dashboard.noteLocalPrediction(result);
  }

  async function runPrediction(payload) {
    if (!state.meta) return null;
    const input = payload || collectInputs();
    clearErrors();
    setLoading(true);
    try {
      const result = await postJson('/predict', input);
      renderResult(result);
      return result;
    } catch (err) {
      show('#predict-placeholder', true);
      if (err.fields) {
        showErrors(err.message, err.fields);
        toast(err.message, 'error');
      } else {
        showErrors(err.message, null);
        toast('Prediction failed: ' + err.message, 'error');
      }
      return null;
    } finally {
      // renderResult() never re-enables the button itself, so a successful prediction
      // would otherwise leave "Predict overrun risk" dead for the rest of the visit.
      setLoading(false);
    }
  }

  function resetPredictor() {
    clearErrors();
    $('#in-model').value = 'auto';
    $('#in-k').value = String(state.meta.default_k);
    setText('#k-value', state.meta.default_k);
    $('#in-budget').value = '';
    $('#in-duration').value = '';
    cascadeDistricts();
    updateBudgetHint();
    show('#predict-result', false);
    show('#predict-spinner', false);
    show('#predict-placeholder', true);
    state.last = null;
  }

  function loadExample() {
    const vocab = state.meta.vocabularies;
    const pick = (list, wanted, fallbackIndex) => (list.indexOf(wanted) >= 0 ? wanted : list[fallbackIndex]);
    $('#in-scheme').value = pick(vocab.scheme_name, 'PMGSY (Rural Roads)', 0);
    $('#in-state').value = pick(vocab.state, 'Maharashtra', 0);
    cascadeDistricts();
    const options = districtsFor($('#in-state').value);
    $('#in-district').value = options.indexOf('Pune') >= 0 ? 'Pune' : options[0];
    $('#in-contractor').value = pick(vocab.contractor_type, 'Govt', 0);
    $('#in-budget').value = '250';
    $('#in-duration').value = '18';
    $('#in-model').value = 'auto';
    $('#in-k').value = String(state.meta.default_k);
    setText('#k-value', state.meta.default_k);
    updateBudgetHint();
  }

  function bindPredictor() {
    const form = $('#predict-form');
    if (!form) return;

    const stateSelect = $('#in-state');
    if (stateSelect) stateSelect.addEventListener('change', cascadeDistricts);
    cascadeDistricts();

    const budget = $('#in-budget');
    if (budget) budget.addEventListener('input', updateBudgetHint);

    const slider = $('#in-k');
    if (slider) {
      slider.addEventListener('input', () => setText('#k-value', slider.value));
    }

    form.addEventListener('submit', (event) => {
      event.preventDefault();
      runPrediction();
    });

    const reset = $('#predict-reset');
    if (reset) reset.addEventListener('click', resetPredictor);
    const example = $('#predict-example');
    if (example) {
      example.addEventListener('click', () => {
        loadExample();
        runPrediction();
      });
    }
    initExport();
  }

  /* ------------------------------------------------------------------ *
   * Exports
   * ------------------------------------------------------------------ */

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1500);
  }

  function csvCell(value) {
    const text = String(value === null || value === undefined ? '' : value);
    return /[",\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  }

  function csvRow(cells) {
    return cells.map(csvCell).join(',');
  }

  function initExport() {
    const csv = $('#export-csv');
    if (csv) csv.addEventListener('click', exportCsv);
    const pdf = $('#export-pdf');
    if (pdf) pdf.addEventListener('click', exportPdf);
  }

  function exportCsv() {
    if (!state.last) { toast('Make a prediction first.', 'error'); return; }
    const result = state.last;
    const lines = [];
    lines.push(csvRow(['Government Construction Project Budget Prediction — India FY 2025-26']));
    lines.push(csvRow([TEAM.credit_line || '']));
    lines.push(csvRow(['Dataset', result.train_size + ' training projects (simulated)', 'Model', result.model_label, 'K', result.k]));
    lines.push(csvRow(['Generated', result.generated_at]));
    lines.push('');
    lines.push(csvRow(['Input field', 'Value']));
    Object.keys(result.inputs).forEach((key) => {
      if (key === 'k' || key === 'model') return;
      let value = result.inputs[key];
      if (key === 'sanctioned_budget_lakh') value = inr(value);
      lines.push(csvRow([key, value]));
    });
    lines.push('');
    lines.push(csvRow(['Prediction', result.prediction]));
    lines.push(csvRow(['Verdict', result.verdict]));
    lines.push(csvRow(['Risk percent', result.risk_pct]));
    lines.push(csvRow(['Risk percent KNN', result.risk_pct_by_model.knn]));
    lines.push(csvRow(['Risk percent Random Forest', result.risk_pct_by_model.rf]));
    lines.push(csvRow(['Sanctioned', result.cost.sanctioned_inr]));
    lines.push(csvRow(['Expected actual', result.cost.expected_actual_inr]));
    lines.push(csvRow(['Expected overrun', result.cost.expected_overrun_inr + ' (' + result.cost.expected_overrun_pct + '%)']));
    lines.push(csvRow(['Neighbour votes', result.votes.yes + ' over / ' + result.votes.no + ' within of ' + result.votes.total]));
    lines.push('');
    lines.push(csvRow(['Explanation']));
    lines.push(csvRow([result.explanation]));
    lines.push('');
    lines.push(csvRow(['Specific advisory']));
    lines.push(csvRow([result.specific_advisory]));
    lines.push('');
    lines.push(csvRow(['Rank', 'Project', 'Scheme', 'State', 'District', 'Execution', 'Sanctioned', 'Actual', 'Months', 'Outcome', 'Distance']));
    result.neighbours.forEach((row) => {
      lines.push(csvRow([row.rank, row.project_name, row.scheme_name, row.state, row.district,
        row.contractor_type, row.sanctioned_inr, row.actual_inr, row.duration_months,
        row.over_budget, row.distance]));
    });
    lines.push('');
    lines.push(csvRow([TEAM.data_note || '']));

    const blob = new Blob(['\ufeff' + lines.join('\r\n')], { type: 'text/csv;charset=utf-8' });
    downloadBlob(blob, 'prediction-india-' + Date.now() + '.csv');
    toast('CSV downloaded.', 'ok');
  }

  /**
   * jsPDF's built-in Helvetica has no rupee glyph, so PDF copy writes the amount as
   * "INR 250.0 lakh" -- same number, readable text. The screen and CSV keep the sign.
   */
  function forPdf(text) {
    return String(text).replace(/₹/g, 'INR');
  }

  function exportPdf() {
    if (!state.last) { toast('Make a prediction first.', 'error'); return; }
    if (!window.jspdf || !window.jspdf.jsPDF) { toast('PDF library did not load.', 'error'); return; }
    const result = state.last;
    const doc = new window.jspdf.jsPDF({ unit: 'pt', format: 'a4' });
    const width = doc.internal.pageSize.getWidth();
    const margin = 44;
    let y = margin;

    function line(text, size, style, gap) {
      doc.setFont('helvetica', style || 'normal');
      doc.setFontSize(size || 10);
      doc.setTextColor(30, 41, 59);
      const wrapped = doc.splitTextToSize(forPdf(String(text)), width - margin * 2);
      wrapped.forEach((part) => {
        if (y > doc.internal.pageSize.getHeight() - margin) {
          doc.addPage();
          y = margin;
        }
        doc.text(part, margin, y);
        y += (size || 10) * 1.35;
      });
      y += gap === undefined ? 4 : gap;
    }

    // letterhead: saffron / white / green bands
    doc.setFillColor(255, 153, 51); doc.rect(0, 0, width, 8, 'F');
    doc.setFillColor(19, 136, 8); doc.rect(0, width > 0 ? doc.internal.pageSize.getHeight() - 8 : 0, width, 8, 'F');
    y = 40;

    line(TEAM.title || 'Budget prediction', 16, 'bold', 6);
    line(TEAM.subtitle || '', 11, 'normal', 2);
    line(TEAM.credit_line || '', 9, 'normal', 8);
    doc.setDrawColor(214, 222, 235);
    doc.line(margin, y, width - margin, y);
    y += 14;

    line('Prediction — ' + result.generated_at, 12, 'bold', 4);
    line('Verdict: ' + result.verdict + ' (' + result.prediction + ')', 12, 'bold', 2);
    line('Model: ' + result.model_label + ', K = ' + result.k
      + ' · KNN risk ' + result.risk_pct_by_model.knn + '%, Random Forest risk '
      + result.risk_pct_by_model.rf + '%', 10, 'normal', 2);
    line('Cost-overrun risk: ' + result.risk_pct + '%', 12, 'bold', 4);

    line('Sanctioned: ' + result.cost.sanctioned_inr
      + '  |  Expected actual: ' + result.cost.expected_actual_inr
      + '  |  Expected overrun: ' + result.cost.expected_overrun_inr
      + ' (' + result.cost.expected_overrun_pct + '%)', 10, 'normal', 2);
    line('Neighbour votes: ' + result.votes.yes + ' over sanction, ' + result.votes.no
      + ' within, of ' + result.votes.total + ' training projects', 10, 'normal', 8);

    line('Inputs', 11, 'bold', 2);
    Object.keys(result.inputs).forEach((key) => {
      if (key === 'k' || key === 'model') return;
      const value = key === 'sanctioned_budget_lakh' ? inr(result.inputs[key]) : result.inputs[key];
      line('· ' + key + ': ' + value, 10, 'normal', 0);
    });
    y += 6;

    line('Explanation', 11, 'bold', 2);
    line(result.explanation, 10, 'normal', 4);
    line('Procurement note', 11, 'bold', 2);
    line(result.specific_advisory, 10, 'normal', 4);

    line('Comparable training projects', 11, 'bold', 2);
    result.neighbours.forEach((row) => {
      line(row.rank + '. ' + row.project_name + ' — ' + row.state + '/' + row.district
        + ', ' + row.contractor_type + ', sanctioned ' + row.sanctioned_inr
        + ', actual ' + row.actual_inr + ', ' + row.over_budget + ', distance ' + row.distance.toFixed(3),
      9, 'normal', 0);
    });

    y += 10;
    line(TEAM.data_note || '', 8, 'italic', 0);
    line('Heuristic: expected spend = sanctioned x (1 + risk x mean overrun of over-budget projects in this scheme). '
      + 'Not a trained cost-regression model.', 8, 'italic', 0);

    doc.save('budget-prediction-india.pdf');
    toast('PDF report downloaded.', 'ok');
  }

  /* ------------------------------------------------------------------ *
   * Source switching, pipeline rebuild, retraining
   * ------------------------------------------------------------------ */

  function initSourceControls() {
    const store = $('#source-store');
    const ref = $('#source-ref');
    const load = $('#source-load');
    if (store && ref) {
      store.addEventListener('change', () => {
        const presets = { local: 'data/india_projects_2025_26.csv', http: '', s3: 'india_projects_2025_26.csv' };
        ref.value = presets[store.value] || '';
        ref.placeholder = store.value === 'http' ? 'https://example.gov/dataset.csv' : 'key or path';
      });
    }
    if (load) {
      load.addEventListener('click', async () => {
        load.disabled = true;
        try {
          const data = await postJson('/sources', { store: store.value, ref: ref.value || null });
          state.meta.sources = data.meta.sources;
          state.meta.vocabularies = data.meta.vocabularies;
          state.meta.dataset = data.meta.dataset;
          state.meta.comparison = data.meta.comparison;
          toast('Loaded ' + data.source.rows + ' rows from ' + data.source.store + '.', 'ok');
          await refreshAfterDataChange();
        } catch (err) {
          toast(err.message, 'error');
        } finally {
          load.disabled = false;
        }
      });
    }

    const rebuild = $('#pipeline-rebuild');
    if (rebuild) {
      rebuild.addEventListener('click', async () => {
        rebuild.disabled = true;
        try {
          const data = await postJson('/pipeline/rebuild', {});
          if (window.Dashboard) Dashboard.renderPipeline(data.manifest);
          toast('Feature store rebuilt: ' + data.manifest.row_count + ' rows in '
            + data.manifest.file_count + ' files.', 'ok');
        } catch (err) {
          toast('Pipeline rebuild failed: ' + err.message, 'error');
        } finally {
          rebuild.disabled = false;
        }
      });
    }

    const run = $('#retrain-run');
    const file = $('#retrain-file');
    if (run && file) {
      run.addEventListener('click', async () => {
        if (!file.files || !file.files.length) { toast('Choose a CSV file first.', 'error'); return; }
        const data = new FormData();
        data.append('file', file.files[0]);
        run.disabled = true;
        setText('#retrain-status', 'Retraining KNN and Random Forest…');
        try {
          const result = await postMultipart('/retrain', data);
          const winner = result.comparison.winner;
          setText('#retrain-status', 'Retrained on ' + result.rows + ' rows. Winner: '
            + result.comparison.winner_label + ' with Yes-F1 '
            + rate(result.metrics[winner].f1) + ', accuracy '
            + rate(result.metrics[winner].accuracy) + '.');
          toast('Retrain complete.', 'ok');
          await refreshAfterDataChange();
        } catch (err) {
          setText('#retrain-status', err.message);
          toast('Retrain failed: ' + err.message, 'error');
        } finally {
          run.disabled = false;
        }
      });
    }

    const resetButton = $('#retrain-reset');
    if (resetButton) {
      resetButton.addEventListener('click', async () => {
        resetButton.disabled = true;
        try {
          const result = await postJson('/retrain', { reset: true });
          setText('#retrain-status', 'Back on the ' + result.rows + ' simulated rows.');
          toast('Restored the simulated dataset.', 'ok');
          await refreshAfterDataChange();
        } catch (err) {
          toast(err.message, 'error');
        } finally {
          resetButton.disabled = false;
        }
      });
    }
  }

  /** Refetch everything that depends on rows or weights, then repaint. */
  async function refreshAfterDataChange() {
    state.meta = await getJson('/meta');
    state.metrics = await getJson('/metrics');
    state.dashboard = await getJson('/dashboard');
    renderMeta();
    renderCounters();
    renderDatasetCharts();
    renderPerformance();
    if (window.Dashboard) Dashboard.renderAggregates(state.dashboard);
    await loadDataset();
    const pipelineData = await getJson('/pipeline');
    if (window.Dashboard) Dashboard.renderPipeline(pipelineData.manifest);
  }

  /* ------------------------------------------------------------------ *
   * Boot
   * ------------------------------------------------------------------ */

  async function boot() {
    initTheme();
    initNav();
    initReveal();
    initTeam();
    initDatasetControls();
    initSourceControls();

    try {
      state.meta = await getJson('/meta');
      renderMeta();
      state.metrics = await getJson('/metrics');
      state.dashboard = await getJson('/dashboard');
      bindPredictor();
      renderPerformance();
      renderCounters();
      renderDatasetCharts();
      await loadDataset();

      if (window.Dashboard) {
        Dashboard.renderAggregates(state.dashboard);
        Dashboard.startStream();
        const pipelineData = await getJson('/pipeline');
        Dashboard.renderPipeline(pipelineData.manifest);
      }

      // One example prediction on load, so the result card, the neighbour table and the
      // live feed all have real content without the visitor having to hunt for it.
      loadExample();
      await runPrediction();

      if (window.Routes) {
        // The map and its charts live in routes.js; it only needs the catalogue, which
        // is static reference data, so a route can be priced even if /meta failed.
        Routes.init(await getJson('/route/cities'));
      }
    } catch (err) {
      banner('The page loaded but the API call failed: ' + err.message
        + ' — restart the server with  py app.py  and reload.');
      toast('Startup failed: ' + err.message, 'error');
    }
  }

  window.App = {
    state: state,
    inr: inr,
    esc: esc,
    pct: pct,
    team: TEAM,
    toast: toast,
    getJson: getJson,
    postJson: postJson,
    forPdf: forPdf,
    csvRow: csvRow,
    downloadBlob: downloadBlob,
    runPrediction: runPrediction,
    lastPrediction: () => state.last,
    refreshAfterDataChange: refreshAfterDataChange,
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
