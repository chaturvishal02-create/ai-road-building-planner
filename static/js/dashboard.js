/* ==========================================================================
   dashboard.js — the live half of the page
   --------------------------------------------------------------------------
   Three jobs:
     1. draw the dashboard charts and KPIs from /dashboard aggregates;
     2. show what the chunked pipeline produced, read from its manifest;
     3. keep /stream (Server-Sent Events) open and append each prediction as
        it arrives, with no page refresh.

   app.js calls renderAggregates(), renderPipeline() and startStream(); nothing
   here fetches on its own except the SSE subscription.
   ========================================================================== */

window.Dashboard = (function () {
  'use strict';

  const MAX_FEED_ROWS = 12;
  const seen = new Set();
  let stream = null;
  let totals = null;

  const $ = (selector) => document.querySelector(selector);

  function setText(selector, value) {
    const node = $(selector);
    if (node) node.textContent = (value === null || value === undefined) ? '—' : String(value);
  }

  function esc(value) {
    return String(value === null || value === undefined ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  /** Same rule as schema.format_inr / app.js inr(), kept local so the file stands alone. */
  function inr(lakh) {
    let value = Number(lakh);
    if (!Number.isFinite(value)) return '—';
    const sign = value < 0 ? '-' : '';
    value = Math.abs(value);
    if (value < 100) return sign + '₹ ' + value.toFixed(1) + ' lakh';
    if (value < 100000) return sign + '₹ ' + (value / 100).toFixed(2) + ' crore';
    return sign + '₹ ' + Math.round(value / 100) + ' crore';
  }

  /* ------------------------------------------------------------------ *
   * 1. aggregates
   * ------------------------------------------------------------------ */

  function renderAggregates(data) {
    totals = data.totals;
    setText('#kpi-sanctioned', inr(totals.sanctioned_lakh));
    setText('#kpi-actual', inr(totals.actual_lakh));
    setText('#kpi-overrun', inr(totals.overrun_lakh));
    setText('#kpi-share', totals.over_budget_share_pct.toFixed(2) + '% of projects');
    setText('#kpi-predictions', data.predictions_served);

    if (window.Charts) {
      Charts.render('budgetActual', 'chart-budget-actual', data.by_scheme);
      Charts.render('stateDist', 'chart-state-dist', data.by_state);
      Charts.render('contractorRisk', 'chart-contractor-risk', data.by_contractor);
    }

    const body = $('#top-overruns');
    if (body) {
      body.innerHTML = data.top_overruns.map((row) => (
        '<tr>' +
        '<td class="truncate-cell" title="' + esc(row.project_name) + '">'
        + esc(row.project_name.replace(/\s*\[Work\s+[^\]]+\]\s*$/, '').slice(0, 46))
        + '<span class="block text-xs text-[var(--muted)]">'
        + esc((/\[Work\s+([^\]]+)\]/.exec(row.project_name) || ['', ''])[1]) + '</span></td>' +
        '<td>' + esc(row.state) + '<span class="block text-xs text-[var(--muted)]">'
        + esc(row.scheme_name) + '</span></td>' +
        '<td class="num">' + esc(row.sanctioned_inr) + '</td>' +
        '<td class="num">' + esc(row.actual_inr) + '</td>' +
        '<td class="num">' + esc(row.overrun_inr)
        + '<span class="block text-xs text-[var(--saffron-text)]">+' + row.overrun_pct.toFixed(2) + '%</span></td>' +
        '</tr>'
      )).join('');
    }
  }

  /* ------------------------------------------------------------------ *
   * 2. pipeline manifest
   * ------------------------------------------------------------------ */

  function renderPipeline(manifest) {
    const host = $('#pipeline-status');
    if (!host || !manifest) return;
    const partitions = Object.keys(manifest.partitions || {});
    const rows = [
      ['Engine', manifest.engine],
      ['Chunk size', manifest.chunk_size + ' rows per pass'],
      ['Partition column', manifest.partition_column],
      ['Rows written', manifest.row_count],
      ['Files', manifest.file_count + ' (' + partitions.length + ' partitions)'],
      ['Format', manifest.format + ' — ' + manifest.format_note],
      ['Elapsed', manifest.elapsed_seconds + ' s'],
      ['Source checksum', String(manifest.source_sha256 || '').slice(0, 16) + '…'],
      ['Generated', manifest.generated_at],
    ];
    host.innerHTML = rows.map((pair) => (
      '<div class="metric"><dt>' + esc(pair[0]) + '</dt><dd>' + esc(pair[1]) + '</dd></div>'
    )).join('') + '<div class="metric"><dt>Partitions</dt><dd>'
      + partitions.map((key) => {
        const entry = manifest.partitions[key];
        return esc(entry.partition_value) + ' (' + entry.rows + ' rows, ' + entry.files + ' file'
          + (entry.files === 1 ? '' : 's') + ')';
      }).join('<br>') + '</dd></div>';
  }

  /* ------------------------------------------------------------------ *
   * 3. live stream
   * ------------------------------------------------------------------ */

  function status(text, kind) {
    const node = $('#stream-status');
    if (!node) return;
    node.textContent = text;
    node.className = 'text-sm ' + (kind === 'down' ? 'text-[var(--danger)]' : 'text-[var(--green-text)]');
  }

  function clockOf(iso) {
    const match = /T(\d{2}:\d{2}:\d{2})/.exec(String(iso || ''));
    return match ? match[1] : new Date().toTimeString().slice(0, 8);
  }

  function appendPrediction(payload) {
    if (!payload || payload.type !== 'prediction') return;
    const row = payload.payload;
    if (seen.has(row.sequence)) return;
    seen.add(row.sequence);

    const body = $('#live-feed');
    if (!body) return;
    const placeholder = body.querySelector('.empty');
    if (placeholder) placeholder.closest('tr').remove();

    const tr = document.createElement('tr');
    tr.dataset.sequence = row.sequence;
    tr.innerHTML = '<td class="num">' + esc(clockOf(row.generated_at)) + '</td>'
      + '<td>' + esc(row.scheme_name) + '</td>'
      + '<td>' + esc(row.district) + '</td>'
      + '<td class="num">' + Number(row.risk_pct).toFixed(1) + '%</td>'
      + '<td><span class="badge ' + (row.prediction === 'Yes' ? 'badge-yes' : 'badge-no') + '">'
      + esc(row.verdict) + '</span></td>';
    body.insertBefore(tr, body.firstChild);
    while (body.children.length > MAX_FEED_ROWS) body.removeChild(body.lastChild);

    setText('#kpi-predictions', row.sequence);
  }

  function startStream() {
    if (!('EventSource' in window)) {
      status('not supported by this browser', 'down');
      return;
    }
    if (stream) stream.close();
    stream = new EventSource('/stream');

    stream.addEventListener('hello', (event) => {
      const data = JSON.parse(event.data);
      status('connected · ' + data.subscribers + ' viewer' + (data.subscribers === 1 ? '' : 's')
        + ' · ' + data.predictions_served + ' served', 'up');
    });

    stream.addEventListener('prediction', (event) => {
      appendPrediction(JSON.parse(event.data));
    });

    stream.addEventListener('heartbeat', (event) => {
      const data = JSON.parse(event.data);
      if (data.totals && data.totals.projects) {
        totals = data.totals;
        setText('#kpi-sanctioned', inr(totals.sanctioned_lakh));
        setText('#kpi-actual', inr(totals.actual_lakh));
        setText('#kpi-overrun', inr(totals.overrun_lakh));
        setText('#kpi-share', totals.over_budget_share_pct.toFixed(2) + '% of projects');
      }
      setText('#kpi-predictions', data.predictions_served);
      status('connected · heartbeat ' + data.at, 'up');
    });

    stream.onerror = () => {
      // EventSource reconnects by itself; say so rather than claim it is dead.
      status('reconnecting…', 'down');
    };
  }

  function stopStream() {
    if (stream) {
      stream.close();
      stream = null;
    }
  }

  /**
   * The SSE round trip can lose a race with the very first prediction made right after
   * the page opens, because the connection is still being established. app.js hands the
   * result here too; appendPrediction de-duplicates on the sequence number.
   */
  function noteLocalPrediction(result) {
    appendPrediction({
      type: 'prediction',
      payload: {
        sequence: result.sequence,
        verdict: result.verdict,
        prediction: result.prediction,
        risk_pct: result.risk_pct,
        scheme_name: result.inputs.scheme_name,
        district: result.inputs.district,
        generated_at: result.generated_at,
      },
    });
  }

  return {
    renderAggregates: renderAggregates,
    renderPipeline: renderPipeline,
    startStream: startStream,
    stopStream: stopStream,
    noteLocalPrediction: noteLocalPrediction,
    feedRows: () => document.querySelectorAll('#live-feed tr').length,
    seenSequences: seen,
  };
})();
