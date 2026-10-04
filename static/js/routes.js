/* ==========================================================================
   routes.js — the interactive India map and the corridor cost planner
   --------------------------------------------------------------------------
   Leaflet draws OpenStreetMap tiles and the city markers; every rupee comes
   from the Flask API (/predict-route, /route/samples, /locations). This file
   never does cost arithmetic: distance, quantities and components are computed
   server-side so the map, the table, the chart and a pytest agree on the same
   number.

   A corridor is named either by two district cities or by two points of the
   State -> District -> Taluka -> Village hierarchy; both paths post to
   /predict-route and both render through the same paint functions.

   The map is the only part that may be missing on an offline machine, so the
   whole map layer is wrapped in a capability check and reports itself through
   #map-status / #map-fallback instead of throwing.
   ========================================================================== */

window.Routes = (function () {
  'use strict';

  const state = {
    catalogue: null,
    cities: {},          // name -> city record
    hierarchy: null,     // /locations catalogue, loaded after the first estimate
    hierarchyReady: false,
    mode: 'cities',      // 'cities' | 'hierarchy'
    map: null,
    markers: {},
    pointMarkers: [],
    line: null,
    last: null,
    series: [],
    requestToken: 0,
    ready: false,
  };

  const $ = (sel) => document.querySelector(sel);
  const inr = (value) => window.App.inr(value);
  const esc = (value) => window.App.esc(value);
  // A sub-15 km village connector really does round to one month, and "1 months" is
  // the kind of thing a viva reader notices.
  const months = (count) => count + (count === 1 ? ' month' : ' months');
  const LEVELS = ['state', 'district', 'taluka', 'village'];

  /* ------------------------------------------------------------------ *
   * Start-up
   * ------------------------------------------------------------------ */

  function init(payload) {
    state.catalogue = payload.catalogue;
    state.catalogue.cities.forEach((city) => { state.cities[city.name] = city; });

    fillSelect('#rt-start', state.catalogue.cities);
    fillSelect('#rt-end', state.catalogue.cities);
    fillSelect('#rt-type', state.catalogue.road_types, 'key', 'label');
    // 'key'/'label' matter here: withAuto builds {key, label} records, and fillSelect
    // defaults to 'name' -- without the explicit keys every option lands with an empty
    // value and an empty caption, and the two dropdowns look blank and cannot be used.
    fillSelect('#rt-terrain', withAuto(state.catalogue.terrains, 'Auto-detect from cities'),
      'key', 'label');
    fillSelect('#rt-soil', withAuto(state.catalogue.soils, 'Auto-detect from cities'),
      'key', 'label');
    fillSelect('#rt-series-type', state.catalogue.road_types, 'key', 'label');

    $('#rt-start').value = 'Nashik';
    $('#rt-end').value = 'Pune';
    $('#rt-series-type').value = 'national_highway_4';
    $('#rt-perkm').placeholder = 'blank = itemised (default ₹ '
      + state.catalogue.default_cost_per_km_crore.toFixed(2) + ' crore/km)';

    buildMap();
    buildExamples();
    bindForm();
    syncHints();
    loadSeries();

    state.ready = true;
    // Price the corridor the page opens on, so the section is never an empty box.
    submit();
    // The hierarchy tree is a second, larger payload; loading it after the first
    // estimate keeps the map and the city path instant.
    loadHierarchy();
  }

  function withAuto(items, label) {
    return [{ key: '', label: label }].concat(items);
  }

  function fillSelect(selector, items, valueKey, labelKey) {
    const node = $(selector);
    if (!node) return;
    const value = valueKey || 'name';
    const label = labelKey || 'name';
    node.innerHTML = items.map((item) => '<option value="' + esc(item[value]) + '">'
      + esc(item[label]) + '</option>').join('');
  }

  function buildExamples() {
    const host = $('#rt-examples');
    host.innerHTML = (state.catalogue.featured_routes || []).map((route) => (
      '<button type="button" class="chip chip-route" data-start="' + esc(route.start)
      + '" data-end="' + esc(route.end) + '">' + esc(route.start) + ' → ' + esc(route.end)
      + '</button>')).join('');
    host.querySelectorAll('button').forEach((button) => {
      button.addEventListener('click', () => {
        // The chips name cities, so they put the form back into city mode first.
        $('#rt-mode').value = 'cities';
        setMode('cities');
        $('#rt-start').value = button.dataset.start;
        $('#rt-end').value = button.dataset.end;
        syncHints();
        submit();
      });
    });
  }

  function bindForm() {
    $('#route-form').addEventListener('submit', (event) => {
      event.preventDefault();
      submit();
    });
    ['#rt-start', '#rt-end'].forEach((sel) => {
      $(sel).addEventListener('change', () => { syncHints(); drawCorridor(); submit(); });
    });
    ['#rt-type', '#rt-terrain', '#rt-soil', '#rt-width', '#rt-perkm', '#rt-distance',
      '#rt-contractor', '#rt-model'].forEach((sel) => {
      const node = $(sel);
      node.addEventListener('change', () => { syncHints(); submit(); });
      if (node.type === 'number') node.addEventListener('input', () => syncHints());
    });
    $('#rt-mode').addEventListener('change', () => {
      setMode($('#rt-mode').value);
      drawCorridor();
      submit();
    });
    bindCascade();
    bindSearch();
    $('#rt-swap').addEventListener('click', swapEnds);
    $('#rt-reset').addEventListener('click', () => {
      $('#route-form').reset();
      setMode($('#rt-mode').value);
      $('#rt-start').value = 'Nashik';
      $('#rt-end').value = 'Pune';
      clearErrors();
      syncHints(); drawCorridor(); submit();
    });
    $('#rt-series-type').addEventListener('change', loadSeries);
    $('#rt-export-csv').addEventListener('click', exportCsv);
    $('#rt-export-pdf').addEventListener('click', exportPdf);
  }

  function setMode(mode) {
    state.mode = mode === 'hierarchy' ? 'hierarchy' : 'cities';
    const hierarchical = state.mode === 'hierarchy';
    $('#rt-cities').hidden = hierarchical;
    $('#rt-hierarchy').hidden = !hierarchical;
    // The city selects keep `required` markup; in location mode they are hidden, and a
    // hidden required field would still be read by a browser validating the form.
    $('#rt-start').required = !hierarchical;
    $('#rt-end').required = !hierarchical;
    if (hierarchical) readout();
  }

  function swapEnds() {
    if (state.mode === 'hierarchy') {
      const a = selectedNames('start');
      const b = selectedNames('end');
      applyPath('start', b);
      applyPath('end', a);
      syncHints(); drawCorridor(); submit();
      return;
    }
    const a = $('#rt-start').value;
    $('#rt-start').value = $('#rt-end').value;
    $('#rt-end').value = a;
    syncHints(); drawCorridor(); submit();
  }

  /* ------------------------------------------------------------------ *
   * Location hierarchy: State -> District -> Taluka -> Village
   * ------------------------------------------------------------------ */

  async function loadHierarchy() {
    // Its own node, because readout() rewrites #rt-loc-readout every time a level
    // changes -- the disclosure has to survive being next to a priced corridor.
    const source = $('#rt-loc-source');
    try {
      const data = await window.App.getJson('/locations');
      state.hierarchy = data.catalogue;
      state.hierarchyReady = true;
      fillOptions(sideNode('start', 'state'), state.hierarchy.states.map((s) => s.name),
                  'Choose a state');
      fillOptions(sideNode('end', 'state'), state.hierarchy.states.map((s) => s.name),
                  'Choose a state');
      cascade('start', 'state');
      cascade('end', 'state');
      // Open on a corridor the visitor can find without reading the dropdowns: two
      // Pune talukas a few kilometres apart, which is what a panchayat plan prices.
      const maharashtra = state.hierarchy.states
        .filter((s) => s.name === 'Maharashtra')[0];
      const pune = maharashtra && maharashtra.districts
        .filter((d) => d.name === 'Pune')[0];
      const nasik = maharashtra && maharashtra.districts
        .filter((d) => d.name === 'Nashik')[0];
      if (pune && nasik) {
        applyPath('start', { state: 'Maharashtra', district: 'Pune' });
        applyPath('end', { state: 'Maharashtra', district: 'Nashik' });
      }
      source.textContent = state.hierarchy.counts.villages + ' locations in '
        + state.hierarchy.counts.states + ' states, ' + state.hierarchy.counts.districts
        + ' districts and ' + state.hierarchy.counts.talukas + ' talukas. Taluka and village '
        + 'names are generated placeholders around real district coordinates — the '
        + 'coordinates, terrain and soil are the catalogue the models use.';
    } catch (err) {
      source.textContent = 'The location hierarchy could not load: ' + err.message
        + '. City corridors still price normally.';
    }
  }

  function sideNode(side, level) {
    return document.getElementById('rt-' + (side === 'start' ? 's' : 'e') + '-' + level);
  }

  function fillOptions(node, names, placeholder) {
    if (!node) return;
    const rows = names.map((name) => ({ value: name, label: name }));
    if (placeholder) rows.unshift({ value: '', label: placeholder });
    node.innerHTML = rows.map((row) => '<option value="' + esc(row.value) + '">'
      + esc(row.label) + '</option>').join('');
  }

  /** Refill every level below `level` for one side and clear the choices it has
   *  invalidated, so the four dropdowns can only ever name a path that is really in the
   *  dataset. */
  function cascade(side, level) {
    if (!state.hierarchyReady) return;
    const chosen = selectedNames(side);
    LEVELS.slice(LEVELS.indexOf(level) + 1).forEach((name) => { chosen[name] = ''; });

    const stateNode = findState(chosen.state);
    fillOptions(sideNode(side, 'district'),
                stateNode ? stateNode.districts.map((d) => d.name) : [],
                stateNode ? 'Choose a district' : 'Choose a state first');
    const districtNode = stateNode && stateNode.districts
      .filter((d) => d.name === chosen.district)[0];
    fillOptions(sideNode(side, 'taluka'),
                districtNode ? districtNode.talukas.map((t) => t.name) : [],
                districtNode ? 'Whole district (headquarters)' : 'Choose a district first');
    const talukaNode = districtNode && districtNode.talukas
      .filter((t) => t.name === chosen.taluka)[0];
    fillOptions(sideNode(side, 'village'),
                talukaNode ? talukaNode.villages.map((v) => v.village) : [],
                talukaNode ? 'Choose a village' : 'Choose a taluka first');
    LEVELS.forEach((name) => { sideNode(side, name).value = chosen[name] || ''; });
    readout();
  }

  function findState(name) {
    if (!state.hierarchyReady) return null;
    return state.hierarchy.states.filter((s) => s.name === name)[0] || null;
  }

  function selectedNames(side) {
    const out = {};
    LEVELS.forEach((level) => {
      const node = sideNode(side, level);
      out[level] = node ? node.value : '';
    });
    return out;
  }

  /** Set one side from a {state, district, taluka, village} object, refilling the
   *  dependent dropdowns as it goes. Used by the search box and by swap. */
  function applyPath(side, path) {
    if (!state.hierarchyReady || !path.state) return false;
    sideNode(side, 'state').value = path.state;
    cascade(side, 'state');
    sideNode(side, 'district').value = path.district || '';
    cascade(side, 'district');
    if (path.taluka) {
      sideNode(side, 'taluka').value = path.taluka;
      cascade(side, 'taluka');
    }
    if (path.village) {
      sideNode(side, 'village').value = path.village;
      readout();
    }
    return true;
  }

  function bindCascade() {
    if (!document.getElementById('rt-s-state')) return;
    ['start', 'end'].forEach((side) => {
      ['state', 'district', 'taluka', 'village'].forEach((level) => {
        sideNode(side, level).addEventListener('change', () => {
          // Changing a level invalidates everything below it, so refill first and only
          // then ask the server: a stale village would price a different point.
          cascade(side, level);
          if (selectedNames(side).village || selectedNames(side).district) {
            syncHints(); drawCorridor(); submit();
          }
        });
      });
    });
  }

  /** The one line that says what the four dropdowns actually mean, including the level
   *  the server will resolve to. */
  function readout() {
    const node = $('#rt-loc-readout');
    if (!node || !state.hierarchyReady) return;
    const parts = ['start', 'end'].map((side) => {
      const chosen = selectedNames(side);
      if (!chosen.district) return side + ': not chosen';
      if (!chosen.village) {
        return side + ': ' + (chosen.taluka ? chosen.taluka + ', ' : '') + chosen.district
          + ' — no village picked, so the corridor starts at the district headquarters';
      }
      const village = villageRecord(chosen);
      if (!village) return side + ': ' + chosen.village + ', ' + chosen.district;
      return side + ': ' + chosen.village + ' (' + chosen.taluka + ', ' + chosen.district
          + ') — connector ' + village.distance_km.toFixed(1) + ' km, ₹ '
          + (village.budget / 100).toFixed(2) + ' crore, ' + months(village.time_months)
          + ', ' + village.risk_band + ' risk';
    });
    node.textContent = parts.join(' · ');
  }

  function villageRecord(chosen) {
    const stateNode = findState(chosen.state);
    const districtNode = stateNode && stateNode.districts
      .filter((d) => d.name === chosen.district)[0];
    const talukaNode = districtNode && districtNode.talukas
      .filter((t) => t.name === chosen.taluka)[0];
    return talukaNode && talukaNode.villages
      .filter((v) => v.village === chosen.village)[0] || null;
  }

  /* ------------------------------------------------------------------ *
   * Search-as-you-type over the hierarchy
   * ------------------------------------------------------------------ */

  function bindSearch() {
    const input = $('#rt-search');
    const results = $('#rt-search-results');
    let timer = null;
    input.addEventListener('input', () => {
      window.clearTimeout(timer);
      const query = input.value.trim();
      if (query.length < 2) {
        results.hidden = true;
        results.innerHTML = '';
        return;
      }
      timer = window.setTimeout(async () => {
        try {
          const data = await window.App.getJson('/locations?q='
            + encodeURIComponent(query));
          paintResults(data.results || []);
        } catch (err) {
          results.hidden = false;
          results.innerHTML = '<li class="search-empty">Search failed: '
            + esc(err.message) + '</li>';
        }
      }, 240);
    });
  }

  function paintResults(rows) {
    const results = $('#rt-search-results');
    if (!rows.length) {
      results.hidden = false;
      results.innerHTML = '<li class="search-empty">Nothing in the planning dataset '
        + 'matches that. It searches village, taluka, district and state names.</li>';
      return;
    }
    results.hidden = false;
    results.innerHTML = rows.map((row, index) => (
      '<li><button type="button" class="search-hit" data-index="' + index + '">'
      + '<strong>' + esc(row.village) + '</strong>'
      + '<span>' + esc(row.taluka) + ' · ' + esc(row.district) + ' · ' + esc(row.state)
      + '</span><em>' + row.distance_km.toFixed(1) + ' km · ₹ '
      + (row.budget / 100).toFixed(2) + ' crore · ' + months(row.time_months) + ' · '
      + esc(row.risk_band) + ' risk</em></button></li>')).join('');
    // The rows are the API's own; keep them so a click can name the exact path without
    // re-reading the DOM.
    results.dataset.hits = JSON.stringify(rows);
    results.querySelectorAll('button').forEach((button) => {
      button.addEventListener('click', () => {
        const hits = JSON.parse(results.dataset.hits || '[]');
        const row = hits[Number(button.dataset.index)];
        if (!row) return;
        applyPath($('#rt-search-side').value, row);
        results.hidden = true;
        syncHints(); drawCorridor(); submit();
      });
    });
  }


  /* ------------------------------------------------------------------ *
   * Map
   * ------------------------------------------------------------------ */

  function buildMap() {
    const status = $('#map-status');
    if (!window.L || typeof window.L.map !== 'function') {
      status.textContent = 'Leaflet not loaded — map disabled';
      $('#map-fallback').hidden = false;
      $('#india-map').hidden = true;
      return;
    }
    try {
      // Maharashtra sits near the centre of the peninsular road network, which is why
      // the brief asks for it: zoom 5 keeps every marker in view without a huge canvas.
      const map = window.L.map('india-map', {
        center: [20.4, 76.5], zoom: 5, scrollWheelZoom: false, zoomControl: true,
      });
      let failures = 0;
      const tiles = window.L.tileLayer(
        'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          maxZoom: 18,
          attribution: '© OpenStreetMap contributors',
        });
      tiles.on('tileerror', () => {
        failures += 1;
        if (failures === 1) {
          status.textContent = 'Tiles unavailable — corridor still drawn';
          $('#map-fallback').hidden = false;
        }
      });
      tiles.on('load', () => { status.textContent = 'OpenStreetMap tiles live'; });
      tiles.addTo(map);

      Object.keys(state.cities).forEach((name) => {
        const city = state.cities[name];
        const marker = window.L.circleMarker([city.lat, city.lon], {
          radius: 6, color: '#0b2545', weight: 2, fillColor: '#ff9933', fillOpacity: 0.9,
        }).addTo(map);
        marker.bindTooltip(city.name + ' · ' + city.state, { direction: 'top', offset: [0, -6] });
        state.markers[name] = marker;
      });

      state.map = map;
      status.textContent = 'Interactive — ' + Object.keys(state.markers).length + ' cities';
      drawCorridor();
      // Leaflet measures its own container: a map created while hidden renders with
      // the wrong size, so the reveal observer nags it once the section scrolls in.
      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => { if (entry.isIntersecting) map.invalidateSize(); });
      }, { threshold: 0.05 });
      observer.observe(document.getElementById('routes'));
    } catch (err) {
      status.textContent = 'Map could not start: ' + err.message;
      $('#map-fallback').hidden = false;
    }
  }

  function straightSampled(a, b) {
    // A provisional line for the frame between changing a dropdown and the response
    // arriving; the real path is the API's geodesic.
    const out = [];
    for (let i = 0; i <= 24; i++) {
      const f = i / 24;
      out.push([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]);
    }
    return out;
  }

  function selectionKey() {
    if (state.mode === 'hierarchy') {
      return 'H' + JSON.stringify(compact(selectedNames('start')))
        + JSON.stringify(compact(selectedNames('end')));
    }
    return 'C' + $('#rt-start').value + '>' + $('#rt-end').value;
  }

  /** The two ends of the corridor as the map needs them. In location mode the
   *  coordinates only ever come from an API response: this file does not compute a
   *  position, a distance or a rupee. */
  function currentEnds() {
    if (state.mode === 'hierarchy') {
      const result = state.last;
      if (!result || state.lastKey !== selectionKey() || !result.start_point
          || !result.route_path) return null;
      return { points: result.route_path, start: result.start, end: result.end,
               result: result };
    }
    const start = $('#rt-start').value;
    const end = $('#rt-end').value;
    const a = state.cities[start];
    const b = state.cities[end];
    if (!a || !b || start === end) return null;
    const fresh = state.last && state.lastKey === selectionKey() && state.last.route_path;
    return { points: fresh ? state.last.route_path : straightSampled([a.lat, a.lon],
                                                                     [b.lat, b.lon]),
             start: start, end: end, result: fresh ? state.last : null };
  }

  function clearPointMarkers() {
    if (!state.map) return;
    state.pointMarkers.forEach((marker) => state.map.removeLayer(marker));
    state.pointMarkers = [];
  }

  function paintEndpointMarkers(result) {
    // The popup the planning brief asks for: the place, what its connector costs, and
    // what the corridor costs. Both figures are the server's.
    [['start_point', 'Start'], ['end_point', 'End']].forEach(([key, role]) => {
      const point = result[key];
      if (!point || typeof point.lat !== 'number') return;
      const connector = point.level === 'village'
        ? '<br>' + esc(point.village) + ' → ' + esc(point.district) + ' HQ: '
          + point.connector_km.toFixed(1) + ' km<br>' + esc(inr(point.connector_budget_lakh))
          + ' · ' + months(point.connector_time_months) + ' · '
          + esc(point.connector_risk_band) + ' risk<br>'
        : '<br>Resolved at the ' + esc(point.level) + ' level: ' + point.count
          + ' villages sit under it, so no single one was priced.<br>';
      const marker = window.L.circleMarker([point.lat, point.lng], {
        radius: 8, color: '#138808', weight: 2, fillColor: '#138808', fillOpacity: 0.5,
      }).addTo(state.map);
      marker.bindPopup('<strong>' + role + ': ' + esc(point.label) + '</strong>'
        + connector + '<br><em>Selected corridor:</em> ' + result.distance_km.toFixed(1)
        + ' km<br>' + esc(result.total_inr) + ' · ' + months(result.time_months) + ' · '
        + esc(result.risk_percent.band) + ' risk');
      state.pointMarkers.push(marker);
    });
  }

  function drawCorridor() {
    if (!state.map || !window.L) return;
    if (state.line) { state.map.removeLayer(state.line); state.line = null; }
    clearPointMarkers();
    const ends = currentEnds();
    if (!ends) return;
    state.line = window.L.polyline(ends.points, {
      color: '#0b2545', weight: 4, opacity: 0.85, dashArray: '6 6',
    }).addTo(state.map);
    window.L.marker(ends.points[0], { opacity: 0 }).addTo(state.map)
      .bindTooltip('Start: ' + ends.start);
    window.L.marker(ends.points[ends.points.length - 1], { opacity: 0 }).addTo(state.map)
      .bindTooltip('End: ' + ends.end);
    if (ends.result) paintEndpointMarkers(ends.result);
    state.map.fitBounds(state.line.getBounds(), { padding: [32, 32], maxZoom: 9 });
  }

  /* ------------------------------------------------------------------ *
   * Hints and errors
   * ------------------------------------------------------------------ */

  function syncHints() {
    const type = state.catalogue.road_types
      .filter((row) => row.key === $('#rt-type').value)[0];
    if (type) {
      $('#rt-type-hint').textContent = type.lanes + ' lane · carriageway '
        + type.width_m + ' m · coefficient applies on top';
      if (!$('#rt-width').value) $('#rt-width').placeholder = 'default ' + type.width_m + ' m';
    }
    const terrain = state.catalogue.terrains
      .filter((row) => row.key === $('#rt-terrain').value)[0];
    $('#rt-terrain-hint').textContent = terrain
      ? '×' + terrain.factor.toFixed(2) + ' — ' + terrain.note
      : 'Auto: worst of the two ends (' + terrainOf('start') + ' / ' + terrainOf('end') + ')';
    const soil = state.catalogue.soils
      .filter((row) => row.key === $('#rt-soil').value)[0];
    $('#rt-soil-hint').textContent = soil
      ? '×' + soil.factor.toFixed(2) + ' — ' + soil.note
      : 'Auto: costlier of the two end soils';
  }

  function cityTerrain(name) {
    const city = state.cities[name];
    return city ? city.terrain : '?';
  }

  /** The terrain the corridor would auto-detect for one side, whichever naming mode is
   *  in use. Terrain is inherited from the district throughout the planning dataset, so
   *  a district-level choice has one answer and any row under it carries it. */
  function terrainOf(side) {
    if (state.mode !== 'hierarchy' || !state.hierarchyReady) {
      return cityTerrain(side === 'start' ? $('#rt-start').value : $('#rt-end').value);
    }
    const chosen = selectedNames(side);
    const village = chosen.village && villageRecord(chosen);
    if (village) return village.terrain;
    const stateNode = findState(chosen.state);
    const districtNode = stateNode && stateNode.districts
      .filter((d) => d.name === chosen.district)[0];
    const first = districtNode && districtNode.talukas[0]
      && districtNode.talukas[0].villages[0];
    return first ? first.terrain : '?';
  }

  function clearErrors() {
    $('#route-errors').hidden = true;
    $('#route-errors').innerHTML = '';
    document.querySelectorAll('#route-form [data-error-for]').forEach((node) => {
      node.textContent = '';
    });
    document.querySelectorAll('#route-form .field-input').forEach((node) => {
      node.removeAttribute('aria-invalid');
    });
  }

  function showErrors(message, fields) {
    const list = $('#route-errors');
    list.innerHTML = '<li>' + esc(message) + '</li>'
      + Object.keys(fields || {}).map((key) => '<li>' + esc(fields[key]) + '</li>').join('');
    list.hidden = false;
    Object.keys(fields || {}).forEach((key) => {
      const slot = document.querySelector('#route-form [data-error-for="' + key + '"]');
      if (slot) slot.textContent = fields[key];
      const input = document.querySelector('#route-form [name="' + key + '"]');
      if (input) input.setAttribute('aria-invalid', 'true');
    });
  }

  /* ------------------------------------------------------------------ *
   * Estimate
   * ------------------------------------------------------------------ */

  /** A {state, district, taluka, village} object with the unchosen levels left out, so
   *  the server resolves exactly what the dropdowns name -- a district with no village
   *  means the district headquarters, not an arbitrary village inside it. */
  function compact(names) {
    const out = {};
    LEVELS.forEach((level) => { if (names[level]) out[level] = names[level]; });
    return out;
  }

  function payload() {
    const shared = {
      road_type: $('#rt-type').value,
      terrain: $('#rt-terrain').value,
      soil: $('#rt-soil').value,
      width_m: $('#rt-width').value,
      cost_per_km_crore: $('#rt-perkm').value,
      distance_km: $('#rt-distance').value,
      contractor_type: $('#rt-contractor').value,
      model: $('#rt-model').value,
    };
    if (state.mode === 'hierarchy') {
      return Object.assign({ start: compact(selectedNames('start')),
                             end: compact(selectedNames('end')) }, shared);
    }
    return Object.assign({ start: $('#rt-start').value,
                           end: $('#rt-end').value }, shared);
  }

  function setLoading(loading) {
    $('#route-loading').hidden = !loading;
    $('#rt-submit').disabled = loading;
    if (loading) $('#route-result').hidden = true;
  }

  async function submit() {
    if (!state.ready) return null;
    clearErrors();
    setLoading(true);
    drawCorridor();
    // Two dropdown changes in quick start fire two requests; whichever finishes
    // last must not be the older one, so each call carries a token and a stale
    // response is dropped instead of overwriting the current result.
    const token = ++state.requestToken;
    const key = selectionKey();
    try {
      const result = await window.App.postJson('/predict-route', payload());
      if (token !== state.requestToken) return null;
      state.last = result;
      state.lastKey = key;
      // Redraw with the geodesic the API actually measured: on the longest
      // corridors it bows up to 35 km away from the straight line drawn above.
      drawCorridor();
      render(result);
      return result;
    } catch (err) {
      if (token !== state.requestToken) return null;
      $('#route-result').hidden = true;
      $('#route-placeholder').hidden = false;
      const fields = err.fields || (err.payload && err.payload.fields);
      if (fields && Object.keys(fields).length) {
        showErrors(err.message, fields);
      } else {
        showErrors(err.message || 'The route could not be estimated.', {});
      }
      return null;
    } finally {
      if (token === state.requestToken) setLoading(false);
    }
  }

  function render(result) {
    $('#route-placeholder').hidden = true;
    $('#route-result').hidden = false;

    $('#rt-out-distance').textContent = result.distance_km.toFixed(1) + ' km';
    $('#rt-out-total').textContent = result.total_inr;
    $('#rt-out-perkm').textContent = '₹ ' + result.derived_cost_per_km_crore.toFixed(2)
      + ' crore/km';
    const time = result.construction_time;
    $('#rt-out-time').textContent = months(result.time_months) + ' ('
      + time.min_months + '–' + time.max_months + ')';
    $('#rt-out-time').title = time.basis;
    $('#rt-out-terrain').textContent = '+ ' + window.App.inr(result.terrain_impact_lakh);
    $('#rt-out-soil').textContent = '+ ' + inr(result.soil_impact_lakh);
    const risk = result.risk_percent;
    $('#rt-out-risk').textContent = risk.percent.toFixed(1) + '% ' + risk.band;

    $('#rt-verdict').textContent = risk.band + ' risk · ' + result.terrain_label
      + ' ×' + result.terrain_factor.toFixed(2)
      + ' · ' + result.soil_label + ' ×' + result.soil_factor.toFixed(2);
    $('#rt-verdict').className = 'route-verdict '
      + (risk.band === 'High' ? 'is-over' : (risk.band === 'Medium' ? 'is-mixed' : 'is-within'));

    renderComponents(result);
    renderQuantities(result);
    renderPlanning(result);
    renderModelCheck(result);
    drawCharts(result);

    $('#rt-locations-line').textContent = 'Selected: ' + result.start + '  →  ' + result.end
      + (result.location_levels ? ' (resolved at the ' + result.location_levels.join(' and ')
        + ' level)' : '') + '. ' + result.distance_basis + '.';

    const scaling = result.scaling_note
      ? ' ' + result.scaling_note
      : '';
    $('#rt-note').textContent =
      'Works ₹ ' + (result.works_lakh / 100).toFixed(2) + ' crore + '
      + result.overheads_pct + '% overheads = ₹ ' + result.total_crore.toFixed(2)
      + ' crore for ' + result.distance_km.toFixed(1) + ' km. Distance basis: '
      + result.distance_basis + '. Time basis: ' + result.construction_time.basis + '. The '
      + result.planned_duration_months + '-month contract programme is a different, longer '
      + 'figure: it carries land, utilities and monsoon idle time, and it is what the '
      + 'trained model is shown.' + scaling;
  }

  function renderComponents(result) {
    const total = result.components.reduce((sum, row) => sum + row.lakh, 0) || 1;
    $('#rt-component-body').innerHTML = result.components.map((row) => (
      '<tr><td>' + esc(row.label) + '</td><td class="text-xs text-[var(--muted)]">'
      + esc(row.basis) + '</td><td class="num">' + esc(quantityFor(row.label, result))
      + '</td><td class="num">' + esc(inr(row.lakh)) + '</td><td class="num">'
      + (100 * row.lakh / total).toFixed(1) + '%</td></tr>')).join('');
  }

  function quantityFor(label, result) {
    const q = result.quantities;
    return {
      Cement: q.cement_bags.toLocaleString('en-IN') + ' bags',
      Steel: q.steel_tonnes.toLocaleString('en-IN') + ' t',
      Bitumen: q.bitumen_tonnes.toLocaleString('en-IN') + ' t',
      Aggregate: q.aggregate_tonnes.toLocaleString('en-IN') + ' t',
      Sand: q.sand_tonnes.toLocaleString('en-IN') + ' t',
      'Sub-base (GSB)': q.gsb_m3.toLocaleString('en-IN') + ' m³',
      Earthwork: q.earthwork_m3.toLocaleString('en-IN') + ' m³',
      Labour: q.labour_person_days.toLocaleString('en-IN') + ' person-days',
      Machinery: q.machinery_hours.toLocaleString('en-IN') + ' machine-h',
      'Shoulders & furniture': 'km × ' + result.width_m + ' m carriageway',
      'Cross drainage': q.drain_lines + ' line(s) × km',
    }[label] || '—';
  }

  function renderQuantities(result) {
    const q = result.quantities;
    $('#rt-quantities').innerHTML = [
      ['Carriageway', q.carriageway_m2.toLocaleString('en-IN') + ' m²'],
      ['Pavement', q.pavement_m3.toLocaleString('en-IN') + ' m³'],
      ['Earthwork', q.earthwork_m3.toLocaleString('en-IN') + ' m³'],
      ['Bitumen', q.bitumen_tonnes.toLocaleString('en-IN') + ' t'],
      ['Cement', q.cement_bags.toLocaleString('en-IN') + ' bags'],
      ['Steel', q.steel_tonnes.toLocaleString('en-IN') + ' t'],
      ['Labour', q.labour_person_days.toLocaleString('en-IN') + ' person-days'],
      ['Plant', q.machinery_hours.toLocaleString('en-IN') + ' machine-h'],
      ['Construction runtime', months(result.time_months) + ' ('
        + result.construction_time.min_months + '–' + result.construction_time.max_months + ')'],
      ['Contract programme', months(result.planned_duration_months)],
      ['Calibrated distance', (result.calibrated_km || result.distance_km).toFixed(1) + ' km'],
    ].map(([key, value]) => '<li><span class="text-[var(--muted)]">' + esc(key)
      + '</span> <strong class="text-[var(--text)]">' + esc(value) + '</strong></li>').join('');
  }

  /** The planning formula's view of the same corridor. Every figure here is the
   *  server's arithmetic; this function only lays it out. */
  function renderPlanning(result) {
    const plan = result.planning;
    if (!plan) return;
    $('#pf-base').textContent = '₹ ' + plan.base_cost_per_km_crore.toFixed(2)
      + ' / km';
    $('#pf-perkm').textContent = '₹ ' + plan.cost_per_km_crore.toFixed(2) + ' / km';
    $('#pf-total').textContent = '₹ ' + plan.total_crore.toFixed(2) + ' crore';
    $('#pf-risk').textContent = plan.risk.band + ' (₹ '
      + plan.risk.thresholds.low_max.toFixed(0) + '-'
      + plan.risk.thresholds.medium_max.toFixed(0) + ' Cr band)';
    $('#pf-verdict').textContent = plan.rate_class + ' base × '
      + plan.width_factor.toFixed(3) + ' width (' + plan.width_m + ' m) × '
      + plan.terrain_factor.toFixed(2) + ' terrain × ' + plan.soil_factor.toFixed(2)
      + ' soil';
    $('#pf-basis').textContent = plan.basis;
    const rows = plan.materials;
    $('#pf-material-body').innerHTML = rows.map((row) => (
      '<tr><td>' + esc(row.label) + '</td><td class="num">' + row.percent.toFixed(0)
      + ' %</td><td class="num">' + esc(inr(row.lakh)) + '</td></tr>')).join('')
      + '<tr><td><strong>Total</strong></td><td class="num"><strong>100 %</strong></td>'
      + '<td class="num"><strong>' + esc(inr(plan.total_lakh)) + '</strong></td></tr>';
    $('#pf-time').textContent = plan.time.basis + (plan.time.under_one_month
      ? ' Rounds up to ' + plan.time_months + ' month for the card and the CSV; the '
        + 'unrounded figure is beside it.' : '');
  }

  function renderModelCheck(result) {
    const check = result.model_check || {};
    const gauge = $('#ml-gauge-fill');
    if (!check.available) {
      $('#ml-model-line').textContent = 'Cross-check unavailable';
      $('#ml-risk').textContent = '—';
      $('#ml-expected').textContent = '—';
      $('#ml-gauge-value').textContent = '—';
      $('#ml-weight-line').textContent = result.risk_percent.note;
      $('#ml-note').textContent = check.reason || 'The trained models are not loaded.';
      gauge.setAttribute('stroke-dashoffset', '326.7');
      return;
    }
    $('#ml-model-line').textContent = check.model + ' · ' + check.verdict
      + ' · ' + check.neighbours + ' comparable projects';
    $('#ml-risk').textContent = check.risk_pct.toFixed(1) + '%';
    $('#ml-expected').textContent = check.expected_actual_inr;
    $('#ml-gauge-value').textContent = check.risk_pct.toFixed(1) + '%';
    gauge.setAttribute('stroke-dashoffset', (326.7 * (1 - check.risk_pct / 100)).toFixed(1));
    $('#ml-weight-line').textContent = 'Formula grade: ' + result.risk_percent.percent.toFixed(1)
      + '% (' + result.risk_percent.band + '). Weights: base '
      + result.risk_percent.weights.base + ' + terrain '
      + result.risk_percent.weights.terrain + ' + soil '
      + result.risk_percent.weights.soil_excess + ' + complexity '
      + result.risk_percent.weights.complexity + ' + length '
      + result.risk_percent.weights.length + '.';
    const note = $('#ml-note');
    note.textContent = check.note;
    note.className = 'mt-3 rounded-lg border-l-4 px-3 py-2 text-sm advisory '
      + (check.within_training_range ? '' : 'is-caution');
  }

  function drawCharts(result) {
    if (!window.Charts) return;
    window.Charts.render('routeSplit', 'chart-route-split', result.components
      .map((row) => ({ label: row.label, value: row.lakh })));
    if (result.planning) {
      window.Charts.render('costSplit', 'chart-planning-split',
        result.planning.materials.map((row) => ({
          label: row.label, value: row.lakh,
        })));
    }
    paintSeries();
  }

  /* ------------------------------------------------------------------ *
   * Comparison series: budget, distance and time
   * ------------------------------------------------------------------ */

  async function loadSeries() {
    const type = $('#rt-series-type').value || 'rural_two_lane';
    try {
      const data = await window.App.getJson('/route/samples?road_type=' + encodeURIComponent(type));
      state.series = data.rows;
      paintSeries();
    } catch (err) {
      $('#rt-note').textContent = 'Comparison series unavailable: ' + err.message;
    }
  }

  function paintSeries() {
    if (!window.Charts || !state.series.length) return;
    const rows = state.series.map((row) => ({
      label: row.label, distance: row.distance_km, cost: row.total_crore,
      published: row.published_km, perKm: row.cost_per_km_crore,
      months: row.time_months, min: row.min_time_months, max: row.max_time_months,
      speed: row.km_per_day,
    }));
    window.Charts.render('routeBudget', 'chart-route-budget', rows);
    window.Charts.render('routeTime', 'chart-route-time', rows);
  }

  /* ------------------------------------------------------------------ *
   * Exports
   * ------------------------------------------------------------------ */

  function exportCsv() {
    const result = state.last;
    if (!result) { window.App.toast('Estimate a route first.', 'error'); return; }
    const rows = [
      window.App.csvRow(['Route estimate — ' + result.start + ' to ' + result.end]),
      window.App.csvRow(['Road type', result.road_type_label]),
      window.App.csvRow(['Distance (km)', result.distance_km]),
      window.App.csvRow(['Calibrated distance (km)', result.calibrated_km || '']),
      window.App.csvRow(['Terrain', result.terrain_label + ' x' + result.terrain_factor]),
      window.App.csvRow(['Soil', result.soil_label + ' x' + result.soil_factor]),
      window.App.csvRow(['Width (m)', result.width_m]),
      window.App.csvRow(['Construction time (months)', result.time_months + ' ('
        + result.construction_time.min_months + '-' + result.construction_time.max_months
        + ' at ' + result.construction_time.km_per_day.join('-') + ' km/day)']),
      window.App.csvRow(['Contract programme (months)', result.planned_duration_months]),
      window.App.csvRow(['Risk grade', result.risk_percent.percent + '% '
        + result.risk_percent.band]),
      [],
      window.App.csvRow(['Component', 'Rate basis', '₹ lakh']),
    ];
    result.components.forEach((row) => {
      rows.push(window.App.csvRow([row.label, row.basis, row.lakh]));
    });
    rows.push(window.App.csvRow(['Works total', '', result.works_lakh]));
    rows.push(window.App.csvRow(['Overheads ' + result.overheads_pct + '%', '',
      result.overheads_lakh]));
    rows.push(window.App.csvRow(['TOTAL', result.distance_basis, result.total_lakh]));
    rows.push(window.App.csvRow(['Total in figures', '',
      '₹ ' + result.total_crore.toFixed(2) + ' crore']));
    if (result.location_levels) {
      rows.push(window.App.csvRow(['Resolved location levels',
        result.start + ' | ' + result.end, result.location_levels.join(' / ')]));
    }
    if (result.planning) {
      const plan = result.planning;
      rows.push([]);
      rows.push(window.App.csvRow(['PLANNING FORMULA VIEW — a different measurement, '
        + 'not a check on the take-off']));
      rows.push(window.App.csvRow(['Base rate (' + plan.rate_class + ')',
        '₹ per km', '₹ ' + plan.base_cost_per_km_crore.toFixed(2) + ' crore']));
      rows.push(window.App.csvRow(['Width factor', plan.width_m + ' m against '
        + plan.standard_width_m + ' m standard', plan.width_factor]));
      rows.push(window.App.csvRow(['Terrain x soil', plan.terrain + ' / ' + plan.soil,
        plan.terrain_factor + ' x ' + plan.soil_factor]));
      plan.materials.forEach((row) => {
        rows.push(window.App.csvRow([row.label,
          row.percent + ' % of the formula budget', row.lakh]));
      });
      rows.push(window.App.csvRow(['Formula total', plan.basis, plan.total_lakh]));
      rows.push(window.App.csvRow(['Formula time', plan.time.basis,
        months(plan.time_months)]));
      rows.push(window.App.csvRow(['Risk by sanction size', plan.risk.rule,
        plan.risk.band]));
    }
    if (result.model_check && result.model_check.available) {
      rows.push([]);
      rows.push(window.App.csvRow(['Trained-model cross-check',
        result.model_check.model, result.model_check.risk_pct + '%']));
      rows.push(window.App.csvRow(['Expected actual cost', '',
        result.model_check.expected_actual_inr]));
    }
    rows.push([]);
    rows.push(window.App.csvRow([window.App.team.data_note || '']));

    const blob = new Blob(['\ufeff' + rows.map((r) => r.join('\r\n')).join('\r\n')
      .replace(/\r\n$/, '')], { type: 'text/csv;charset=utf-8' });
    window.App.downloadBlob(blob, 'route-' + result.start + '-' + result.end + '.csv'.replace(/\s/g, '_'));
    window.App.toast('Route breakdown downloaded.', 'ok');
  }

  function exportPdf() {
    const result = state.last;
    if (!result) { window.App.toast('Estimate a route first.', 'error'); return; }
    if (!window.jspdf || !window.jspdf.jsPDF) {
      window.App.toast('PDF library did not load.', 'error');
      return;
    }
    const doc = new window.jspdf.jsPDF({ unit: 'pt', format: 'a4' });
    const width = doc.internal.pageSize.getWidth();
    const margin = 44;
    let y = margin;

    function line(text, size, style, gap) {
      doc.setFont('helvetica', style || 'normal');
      doc.setFontSize(size || 10);
      doc.setTextColor(30, 41, 59);
      // Helvetica has no rupee glyph, so ₹ becomes INR in the PDF only.
      const wrapped = doc.splitTextToSize(window.App.forPdf(String(text)), width - margin * 2);
      doc.text(wrapped, margin, y);
      y += wrapped.length * (size || 10) * 1.3 + (gap === undefined ? 4 : gap);
    }

    line('Route cost estimate — ' + result.start + ' to ' + result.end, 17, 'bold', 8);
    line(window.App.team.credit_line, 9, 'normal', 10);
    line('Cross-section: ' + result.road_type_label + ', ' + result.width_m + ' m carriageway', 10);
    line('Distance: ' + result.distance_km.toFixed(1) + ' km (' + result.distance_basis + ')', 10);
    line('Ground: ' + result.terrain_label + ' ×' + result.terrain_factor.toFixed(2)
      + ', ' + result.soil_label + ' ×' + result.soil_factor.toFixed(2), 10);
    line('Construction time: ' + months(result.time_months) + ' ('
      + result.construction_time.min_months + '–' + result.construction_time.max_months
      + ' at ' + result.construction_time.km_per_day.join('–') + ' km/day)', 10);
    line('Contract programme: ' + months(result.planned_duration_months) + ' — includes land, '
      + 'utilities and monsoon idle time', 10, 'normal', 10);
    line('TOTAL: ' + result.total_inr + '  ('
      + result.derived_cost_per_km_crore.toFixed(2) + ' crore per km)', 13, 'bold', 6);
    line('Overrun risk: ' + result.risk_percent.percent.toFixed(1) + '% — '
      + result.risk_percent.band, 12, 'bold', 10);

    line('Cost components', 12, 'bold', 4);
    result.components.forEach((row) => {
      line(row.label + '  —  ' + row.basis + '  —  ' + row.lakh.toLocaleString('en-IN')
        + ' lakh', 10);
    });
    line('Works ' + result.works_lakh.toLocaleString('en-IN') + ' lakh + overheads '
      + result.overheads_pct + '% = ' + result.total_lakh.toLocaleString('en-IN') + ' lakh',
    10, 'bold', 10);

    if (result.model_check && result.model_check.available) {
      line('Trained-model cross-check', 12, 'bold', 4);
      line(result.model_check.model + ': ' + result.model_check.risk_pct.toFixed(1)
        + '% probability of overrun, ' + result.model_check.verdict + '.', 10);
      line('Expected actual cost ' + result.model_check.expected_actual_inr + '.', 10);
      line(result.model_check.note, 9, 'italic', 10);
    } else {
      line('The corridor sanction sits outside the range the trained model was fitted on, '
        + 'so no model probability is quoted.', 9, 'italic', 10);
    }
    line(result.basis, 9, 'italic', 6);
    line(result.risk_percent.note, 9, 'italic');

    if (result.planning) {
      const plan = result.planning;
      line('Planning formula view — distance x base ₹/km x width x terrain x soil',
      12, 'bold', 4);
      line(plan.basis, 10);
      line(plan.materials.map((row) => row.label + ' '
        + row.percent.toFixed(0) + '% = ₹ '
        + row.lakh.toLocaleString('en-IN') + ' lakh').join(', '), 10);
      line('Formula time: ' + plan.time.basis, 10);
      line('Risk by sanction size: ' + plan.risk.band + ' — ' + plan.risk.rule, 9,
        'italic', 10);
    }

    doc.save('route-estimate.pdf');
    window.App.toast('Route PDF downloaded.', 'ok');
  }

  return {
    init: init,
    submit: submit,
    state: state,
    isReady: () => state.ready,
  };
})();
