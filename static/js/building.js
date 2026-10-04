/* ==========================================================================
   building.js — the controller for /building, and nothing else
   --------------------------------------------------------------------------
   A separate page for a separate estimator, so it is a separate file with its
   own small helpers. app.js is NOT loaded here: its boot() reaches for the
   predictor, the dataset table and the map, none of which exist on this
   template, and a missing node would throw before the form was usable.

   Every rupee and every month on this page comes from POST /predict-building.
   The rates below are only ever printed, never multiplied: the factor tables
   arrive from model/building.py through the template so the explanation cannot
   drift from the arithmetic.
   ========================================================================== */

window.Building = (function () {
  'use strict';

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.prototype.slice.call(document.querySelectorAll(selector));

  function readJson(id, fallback) {
    const node = document.getElementById(id);
    if (!node) return fallback;
    try {
      return JSON.parse(node.textContent || '{}');
    } catch (err) {
      return fallback;
    }
  }

  const TEAM = readJson('team-data', {});
  const CATALOGUE = readJson('building-data', {});

  const state = {
    hierarchy: [],        // [{name, districts: [{name, talukas: [{name, villages}]}]}]
    counts: {},
    last: null,
    requestToken: 0,
    ready: false,
  };

  /* Twin of schema.format_inr / route_engine.format_inr: below 1 crore show
     lakh, above it show crore. Amounts travel as rupee LAKH. */
  function inr(lakh) {
    let value = Number(lakh);
    if (!Number.isFinite(value)) return '—';
    const sign = value < 0 ? '-' : '';
    value = Math.abs(value);
    if (value < 100) return sign + '₹ ' + value.toFixed(1) + ' lakh';
    if (value < 100000) return sign + '₹ ' + (value / 100).toFixed(2) + ' crore';
    return sign + '₹ ' + Math.round(value / 100) + ' crore';
  }

  function esc(value) {
    return String(value === null || value === undefined ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function num(value) {
    return Number(value).toLocaleString('en-IN');
  }

  function toast(message, kind) {
    const stack = $('#toast-stack');
    if (!stack) return;
    const box = document.createElement('div');
    box.className = 'toast ' + (kind === 'error' ? 'toast-error' : 'badge-neutral');
    box.textContent = message;
    stack.appendChild(box);
    window.setTimeout(() => box.remove(), kind === 'error' ? 6500 : 4200);
  }

  function banner(message) {
    const node = $('#api-banner');
    if (!node) return;
    if (!message) {
      node.className = 'hidden';
      node.textContent = '';
      return;
    }
    node.className = 'notice notice-saffron';
    node.setAttribute('role', 'status');
    node.textContent = message;
    node.removeAttribute('hidden');
  }

  /* ------------------------------------------------------------------ *
   * Fetching
   * ------------------------------------------------------------------ */

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
      const message = (body && body.error) ? body.error
        : 'Request failed (' + response.status + ')';
      throw new RequestError(message, body && body.fields, response.status);
    }
    return body;
  }

  async function getJson(url) {
    return readResponse(await fetch(url, { headers: { Accept: 'application/json' } }));
  }

  async function postJson(url, payload) {
    return readResponse(await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(payload || {}),
    }));
  }

  /* ------------------------------------------------------------------ *
   * Theme, fade-in, team
   * ------------------------------------------------------------------ */

  function isDark() {
    return document.documentElement.classList.contains('dark');
  }

  function initTheme() {
    const label = () => {
      const node = $('#theme-toggle [data-theme-label]');
      if (node) node.textContent = isDark() ? 'Light' : 'Dark';
      const button = $('#theme-toggle');
      if (button) button.setAttribute('aria-pressed', isDark() ? 'true' : 'false');
    };
    label();
    const button = $('#theme-toggle');
    if (!button) return;
    button.addEventListener('click', () => {
      const dark = !isDark();
      document.documentElement.classList.toggle('dark', dark);
      try {
        window.localStorage.setItem('budget-theme', dark ? 'dark' : 'light');
      } catch (err) { /* private mode: the toggle still works for this page load */ }
      label();
      if (window.Charts) window.Charts.refreshTheme();
    });
  }

  function initReveal() {
    const nodes = $$('.reveal');
    if (!('IntersectionObserver' in window)) {
      nodes.forEach((node) => node.classList.add('is-visible'));
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08 });
    nodes.forEach((node) => observer.observe(node));
  }

  function initTeam() {
    const host = $('#bd-team');
    if (!host) return;
    const members = [
      { role: 'Developed by', names: TEAM.developers || [],
        note: 'Dataset engineering, model training, and this building estimator.' },
      { role: 'Under the guidance of', names: TEAM.guide ? [TEAM.guide] : [],
        note: 'Methodology review and validation of the reported metrics.' },
      { role: 'Institution', names: TEAM.institution ? [TEAM.institution] : [],
        note: TEAM.programme || '' },
    ];
    host.innerHTML = members.map((member) => (
      '<article class="card">'
      + '<p class="eyebrow">' + esc(member.role) + '</p>'
      + member.names.map((name) => '<h3 class="card-title mt-1">' + esc(name) + '</h3>').join('')
      + '<p class="mt-2 text-sm leading-relaxed text-[var(--muted)]">' + esc(member.note) + '</p>'
      + '</article>')).join('');
  }

  /* ------------------------------------------------------------------ *
   * Rate tables — printed from the catalogue, never recomputed here
   * ------------------------------------------------------------------ */

  function fillTable(selector, rows) {
    const node = $(selector);
    if (node) node.innerHTML = rows.join('');
  }

  function renderRateTables() {
    fillTable('#rate-locations', (CATALOGUE.location_types || []).map((row) => (
      '<tr><td>' + esc(row.key) + '</td><td class="num">₹ ' + num(row.base_cost_per_sqft)
      + '</td><td class="num">×' + row.factor.toFixed(2) + '</td></tr>')));
    fillTable('#rate-floors', (CATALOGUE.floor_factors || []).map((row) => (
      '<tr><td>' + row.floors + (row.floors === 1 ? ' floor' : ' floors')
      + '</td><td class="num">×' + row.factor.toFixed(2) + '</td></tr>')));
    fillTable('#rate-qualities', (CATALOGUE.qualities || []).map((row) => (
      '<tr><td>' + esc(row.label) + '</td><td class="num">×' + row.factor.toFixed(2)
      + '</td><td class="num">' + num(row.sqft_per_day) + '</td></tr>')));
    fillTable('#rate-types', (CATALOGUE.building_types || []).map((row) => (
      '<tr><td>' + esc(row.label) + '</td><td class="num">×' + row.factor.toFixed(2)
      + '</td></tr>')));
    fillTable('#rate-split', (CATALOGUE.work_split || []).map((row) => (
      '<tr><td>' + esc(row.label) + '</td><td class="num">' + row.percent.toFixed(0)
      + ' %</td></tr>')));
  }

  /* ------------------------------------------------------------------ *
   * Selects
   * ------------------------------------------------------------------ */

  function options(items, valueKey, labelFn, placeholder) {
    const rows = placeholder ? [{ value: '', label: placeholder }] : [];
    items.forEach((item) => rows.push({ value: item[valueKey], label: labelFn(item) }));
    return rows.map((row) => '<option value="' + esc(row.value) + '">' + esc(row.label)
      + '</option>').join('');
  }

  function byId(selector) {
    return $(selector) ? $(selector).value : '';
  }

  function find(key, list) {
    return (list || []).filter((row) => row.key === key)[0];
  }

  function buildFormSelects() {
    const types = CATALOGUE.building_types || [];
    $('#bd-type').innerHTML = options(types, 'key', (row) => row.label);
    $('#bd-floors').innerHTML = options(CATALOGUE.floor_factors || [], 'floors',
      (row) => row.floors + (row.floors === 1 ? ' floor' : ' floors'));
    $('#bd-quality').innerHTML = options(CATALOGUE.qualities || [], 'key',
      (row) => row.label);
    $('#bd-location-type').innerHTML = options(CATALOGUE.location_types || [], 'key',
      (row) => row.key + ' — ₹ ' + num(row.base_cost_per_sqft) + ' / sq ft');
    if (types.length) $('#bd-type').value = types[0].key;
    $('#bd-floors').value = '1';
    $('#bd-quality').value = 'medium';
    $('#bd-location-type').value = 'rural';
  }

  function syncHints() {
    const type = find(byId('#bd-type'), CATALOGUE.building_types);
    $('#bd-type-hint').textContent = type
      ? '×' + type.factor.toFixed(2) + ' — ' + type.note : '';

    const floors = (CATALOGUE.floor_factors || [])
      .filter((row) => String(row.floors) === byId('#bd-floors'))[0];
    $('#bd-floors-hint').textContent = floors
      ? '×' + floors.factor.toFixed(2) + ' on the footprint — '
        + num(Number($('#bd-area').value || 0) * floors.factor) + ' sq ft raised'
      : '';

    const quality = find(byId('#bd-quality'), CATALOGUE.qualities);
    $('#bd-quality-hint').textContent = quality
      ? '×' + quality.factor.toFixed(2) + ' · ' + num(quality.sqft_per_day)
        + ' sq ft per day per site crew'
      : '';

    const location = find(byId('#bd-location-type'), CATALOGUE.location_types);
    $('#bd-location-type-hint').textContent = location
      ? '₹ ' + num(location.base_cost_per_sqft) + ' per sq ft base · ×'
        + location.factor.toFixed(2) + ' location factor'
      : '';
  }

  /* ------------------------------------------------------------------ *
   * Location cascade
   * ------------------------------------------------------------------ */

  function stateNode(name) {
    return state.hierarchy.filter((node) => node.name === name)[0] || null;
  }

  function districtNode(stateName, districtName) {
    const node = stateNode(stateName);
    if (!node) return null;
    return (node.districts || []).filter((row) => row.name === districtName)[0] || null;
  }

  function talukaNode(stateName, districtName, talukaName) {
    const node = districtNode(stateName, districtName);
    if (!node) return null;
    return (node.talukas || []).filter((row) => row.name === talukaName)[0] || null;
  }

  /* Rebuild every dropdown strictly below the one that changed, and clear the
     ones below that again: a stale taluka of the old district would silently
     price a building on a plot that is not in the state now chosen. */
  function rebuild(level) {
    if (level === 'state') {
      const node = stateNode($('#bd-state').value);
      $('#bd-district').innerHTML = options(node ? node.districts : [], 'name',
        (row) => row.name, 'Choose a district');
      $('#bd-district').value = '';
    }
    if (level === 'state' || level === 'district') {
      const district = districtNode($('#bd-state').value, $('#bd-district').value);
      $('#bd-taluka').innerHTML = options(district ? district.talukas : [], 'name',
        (row) => row.name, district ? 'Optional — all talukas' : 'Choose a district first');
      $('#bd-taluka').value = '';
      $('#bd-village').innerHTML = options([], 'village', (row) => row.village,
        'Choose a taluka first');
      $('#bd-village').value = '';
      return;
    }
    if (level === 'taluka') {
      const taluka = talukaNode($('#bd-state').value, $('#bd-district').value,
        $('#bd-taluka').value);
      $('#bd-village').innerHTML = options(taluka ? taluka.villages : [], 'village',
        (row) => row.village,
        taluka ? 'Optional — district headquarters' : 'Choose a taluka first');
      $('#bd-village').value = '';
    }
  }

  function seedSite() {
    const names = state.hierarchy.map((node) => node.name);
    if (!names.length) return;
    $('#bd-state').value = names.indexOf('Maharashtra') >= 0 ? 'Maharashtra' : names[0];
    rebuild('state');
    const preferred = districtNode($('#bd-state').value, 'Pune') ? 'Pune' : '';
    if (preferred) {
      $('#bd-district').value = preferred;
      rebuild('district');
    }
    syncSiteLine();
  }

  function syncSiteLine() {
    const node = $('#bd-site-line');
    const given = [$('#bd-village').value, $('#bd-taluka').value,
      $('#bd-district').value, $('#bd-state').value].filter(Boolean);
    if (!given.length) {
      node.textContent = 'No site chosen — the cost does not depend on it.';
      return;
    }
    const taluka = talukaNode($('#bd-state').value, $('#bd-district').value,
      $('#bd-taluka').value);
    const count = taluka ? taluka.villages.length : 0;
    node.innerHTML = 'Site <strong>' + esc(given.slice().reverse().join(' › '))
      + '</strong>' + (count ? ' · ' + count + ' villages priced under this taluka' : '')
      + '. Districts and coordinates are real; taluka and village names are generated '
        + 'placeholders.';
  }

  async function loadHierarchy() {
    const payload = await getJson('/locations');
    state.hierarchy = (payload.catalogue || {}).states || [];
    state.counts = (payload.catalogue || {}).counts || {};
    $('#bd-state').innerHTML = options(state.hierarchy, 'name', (row) => row.name,
      'Choose a state');
    seedSite();
  }

  /* ------------------------------------------------------------------ *
   * Errors
   * ------------------------------------------------------------------ */

  function clearErrors() {
    const list = $('#building-errors');
    list.hidden = true;
    list.innerHTML = '';
    $$('#building-form [data-error-for]').forEach((node) => { node.textContent = ''; });
    $$('#building-form .field-input').forEach((node) => node.removeAttribute('aria-invalid'));
  }

  function showErrors(message, fields) {
    const keys = Object.keys(fields || {});
    const list = $('#building-errors');
    list.innerHTML = '<li>' + esc(message) + '</li>'
      + keys.map((key) => '<li>' + esc(fields[key]) + '</li>').join('');
    list.hidden = false;
    keys.forEach((key) => {
      const slot = document.querySelector('#building-form [data-error-for="' + key + '"]');
      if (slot) slot.textContent = fields[key];
      const input = document.querySelector('#building-form [name="' + key + '"]');
      if (input) input.setAttribute('aria-invalid', 'true');
    });
  }

  /* ------------------------------------------------------------------ *
   * Estimate
   * ------------------------------------------------------------------ */

  function payload() {
    return {
      building_type: $('#bd-type').value,
      area_sqft: $('#bd-area').value,
      floors: $('#bd-floors').value,
      quality: $('#bd-quality').value,
      location_type: $('#bd-location-type').value,
      state: $('#bd-state').value,
      district: $('#bd-district').value,
      taluka: $('#bd-taluka').value,
      village: $('#bd-village').value,
    };
  }

  function setLoading(loading) {
    $('#building-loading').hidden = !loading;
    $('#bd-submit').disabled = loading;
    if (loading) {
      $('#building-result').hidden = true;
      $('#building-placeholder').hidden = true;
    }
  }

  async function submit() {
    if (!state.ready) return null;
    clearErrors();
    setLoading(true);
    const token = ++state.requestToken;
    try {
      const result = await postJson('/predict-building', payload());
      if (token !== state.requestToken) return null;
      state.last = result;
      render(result);
      return result;
    } catch (err) {
      if (token !== state.requestToken) return null;
      $('#building-result').hidden = true;
      $('#building-placeholder').hidden = false;
      showErrors(err.message || 'The building could not be costed.', err.fields || {});
      return null;
    } finally {
      if (token === state.requestToken) setLoading(false);
    }
  }

  function render(result) {
    $('#building-placeholder').hidden = true;
    $('#building-result').hidden = false;

    $('#bd-out-cost').textContent = '₹ ' + result.cost_lakh.toFixed(2) + ' lakh';
    $('#bd-out-cost').title = result.cost_formatted;
    $('#bd-out-crore').textContent = '₹ ' + result.cost_crore.toFixed(2) + ' crore';
    $('#bd-out-time').textContent = result.time_months + ' months';
    $('#bd-out-rate').textContent = '₹ ' + num(result.effective_cost_per_sqft);
    $('#bd-out-built').textContent = num(result.built_up_sqft) + ' sq ft';
    $('#bd-out-base').textContent = '₹ ' + num(result.base_cost_per_sqft) + ' / sq ft';

    $('#bd-factors').innerHTML = [
      ['Type', result.building_type_label + ' ×' + result.building_type_factor.toFixed(2)],
      ['Storeys', result.floors + ' ×' + result.floor_factor.toFixed(2)],
      ['Location', result.location_type + ' ×' + result.location_factor.toFixed(2)],
      ['Quality', result.quality_label + ' ×' + result.quality_factor.toFixed(2)],
      ['Area', num(result.area_sqft) + ' sq ft'],
    ].map((pair) => '<span>' + esc(pair[0]) + ' <strong>' + esc(pair[1]) + '</strong></span>')
      .join('');

    const location = result.location || {};
    const given = [location.village, location.taluka, location.district, location.state]
      .filter(Boolean);
    $('#bd-site-result').innerHTML = given.length
      ? 'Site <strong>' + esc(given.slice().reverse().join(' › ')) + '</strong> — the site '
        + 'identifies the work; the rate comes from the location type.'
      : 'No site named — the cost is the same for any plot in this location type.';

    $('#bd-basis').textContent = result.basis;

    // Flask sorts JSON keys, so the seven lines would arrive alphabetically. The
    // catalogue carries them in cost order, and the table and the pie both read
    // through it -- structure first, site development last.
    const keys = (CATALOGUE.work_split || []).map((row) => row.key)
      .filter((key) => result.breakdown[key]);
    Object.keys(result.breakdown).forEach((key) => {
      if (keys.indexOf(key) < 0) keys.push(key);
    });
    const rows = keys.map((key) => {
      const row = result.breakdown[key];
      return '<tr><td>' + esc(row.label) + '</td><td class="num">' + row.percent.toFixed(0)
        + ' %</td><td class="num">' + row.lakh.toFixed(2) + '</td><td class="num">'
        + num(row.rupees) + '</td></tr>';
    });
    // Summed from the rows the table just printed, so a reader can check the total
    // against the card above it -- and see immediately if the two ever disagree.
    const totalLakh = keys.reduce((sum, key) => sum + result.breakdown[key].lakh, 0);
    const totalRupees = keys.reduce((sum, key) => sum + result.breakdown[key].rupees, 0);
    rows.push('<tr class="font-semibold"><td>Total (all ' + keys.length
      + ' sections)</td><td class="num">100 %</td><td class="num">'
      + totalLakh.toFixed(2) + '</td><td class="num">' + num(totalRupees)
      + '</td></tr>');
    $('#bd-split-body').innerHTML = rows.join('');

    const time = result.construction_time;
    $('#bd-time-basis').textContent = time.basis + ' — ' + num(time.built_up_sqft)
      + ' sq ft of built-up area at ' + num(time.sqft_per_day) + ' sq ft per day, plus '
      + num(time.mobilisation_days) + ' fixed days of site set-up.';

    $('#bd-note').textContent = result.note;

    if (window.Charts) {
      window.Charts.render('costSplit', 'chart-building-split', keys.map((key) => ({
        label: result.breakdown[key].label,
        value: result.breakdown[key].lakh,
      })));
    }
  }

  /* ------------------------------------------------------------------ *
   * Presets — one per building type, so the page opens with real content
   * ------------------------------------------------------------------ */

  function buildExamples() {
    const presets = [
      { label: 'Gram Panchayat · 1,200 sq ft · rural', building_type: 'gram_panchayat',
        area_sqft: 1200, floors: 1, quality: 'medium', location_type: 'rural' },
      { label: 'School block · 4,000 sq ft · urban', building_type: 'school',
        area_sqft: 4000, floors: 1, quality: 'medium', location_type: 'urban' },
      { label: 'Urban health centre · 6,000 sq ft · 2 floors', building_type: 'hospital',
        area_sqft: 6000, floors: 2, quality: 'high', location_type: 'urban' },
    ];
    const host = $('#bd-examples');
    host.innerHTML = presets.map((preset, index) => (
      '<button type="button" class="chip chip-route" data-index="' + index + '">'
      + esc(preset.label) + '</button>')).join('');
    host.querySelectorAll('button').forEach((button) => {
      button.addEventListener('click', () => {
        const preset = presets[Number(button.dataset.index)];
        $('#bd-type').value = preset.building_type;
        $('#bd-area').value = preset.area_sqft;
        $('#bd-floors').value = String(preset.floors);
        $('#bd-quality').value = preset.quality;
        $('#bd-location-type').value = preset.location_type;
        syncHints();
        submit();
      });
    });
  }

  /* ------------------------------------------------------------------ *
   * Wiring
   * ------------------------------------------------------------------ */

  function bindForm() {
    $('#building-form').addEventListener('submit', (event) => {
      event.preventDefault();
      submit();
    });
    ['#bd-type', '#bd-floors', '#bd-quality', '#bd-location-type'].forEach((sel) => {
      $(sel).addEventListener('change', () => { syncHints(); submit(); });
    });
    $('#bd-area').addEventListener('change', () => { syncHints(); submit(); });
    $('#bd-area').addEventListener('input', syncHints);
    ['#bd-state', '#bd-district', '#bd-taluka', '#bd-village'].forEach((sel) => {
      const node = $(sel);
      node.addEventListener('change', () => {
        rebuild(node.dataset.level);
        syncSiteLine();
        submit();
      });
    });
    $('#bd-reset').addEventListener('click', () => {
      $('#building-form').reset();
      buildFormSelects();
      clearErrors();
      syncHints();
      submit();
    });
  }

  async function boot() {
    initTheme();
    initReveal();
    initTeam();
    renderRateTables();
    buildFormSelects();
    buildExamples();
    bindForm();
    syncHints();
    try {
      await loadHierarchy();
    } catch (err) {
      banner('The location list could not load: ' + err.message
        + ' — the estimator still works; the site dropdowns stay empty.');
    }
    state.ready = true;
    submit();
  }

  return {
    boot: boot,
    submit: submit,
    state: state,
    inr: inr,
    last: () => state.last,
  };
})();

/* app.js auto-boots the dashboard on DOMContentLoaded; this page boots its own
   controller the same way, and stops if the API is unreachable. */
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', window.Building.boot);
} else {
  window.Building.boot();
}
