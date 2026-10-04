/* ==========================================================================
   knn-demo.js — the interactive "How KNN Works" canvas.
   --------------------------------------------------------------------------
   A small, self-contained 2-D illustration of the same idea the real model
   uses across 81 encoded columns:
     1. measure the Euclidean distance from the test project to every sample;
     2. keep the K closest;
     3. take a majority vote of their labels.
   The test project can be dragged with a mouse, a pen or a finger.
   Horizontal axis: scaled sanctioned cost. Vertical axis: how many category
   fields differ, which is exactly what the one-hot block contributes to the
   real distance calculation.
   Public API: window.KnnDemo.init(), window.KnnDemo.refreshTheme()
   ========================================================================== */

(function () {
  'use strict';

  const canvas = document.getElementById('knn-canvas');
  if (!canvas) return; // section not present — nothing to do

  const ctx = canvas.getContext('2d');
  const kSlider = document.getElementById('knn-k');
  const kValue = document.getElementById('knn-k-value');
  const readoutEl = document.getElementById('knn-readout');
  const resetBtn = document.getElementById('knn-reset');
  const randomBtn = document.getElementById('knn-random');

  const SAMPLE_COUNT = 60;
  const YES_SHARE = 0.42; // matches the 42% over-sanction rate in the real dataset
  // Start the test project on the boundary between the two clusters: at K=5 the
  // vote is close, so moving K visibly changes the answer.
  const DEFAULT_TEST = { x: 0.58, y: 0.5 };
  let points = [];
  let test = { x: DEFAULT_TEST.x, y: DEFAULT_TEST.y }; // normalised 0..1 data space
  let dragging = false;
  let seed = 20220213;

  /** Deterministic pseudo-random generator, so the demo looks the same every load. */
  function mulberry32(seed) {
    return function () {
      seed |= 0;
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /**
   * Build two overlapping clusters: most projects stay within sanction (green),
   * a smaller group goes over (red) — mirroring the real 58/42 imbalance.
   */
  function buildPoints() {
    const rand = mulberry32(seed);
    const gaussian = () => {
      // Box–Muller: turn two uniform numbers into a normal one.
      const u = Math.max(rand(), 1e-9);
      const v = rand();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };

    const made = [];
    const yesCount = Math.round(SAMPLE_COUNT * YES_SHARE);
    for (let i = 0; i < SAMPLE_COUNT; i++) {
      const isYes = i < yesCount;
      // Cluster centres differ a little, so the vote changes as you drag.
      const cx = isYes ? 0.66 : 0.40;
      const cy = isYes ? 0.62 : 0.40;
      made.push({
        x: clamp01(cx + gaussian() * 0.13),
        y: clamp01(cy + gaussian() * 0.13),
        label: isYes ? 'Yes' : 'No'
      });
    }
    // Shuffle so the Yes points are spread through the array.
    for (let i = made.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const tmp = made[i];
      made[i] = made[j];
      made[j] = tmp;
    }
    return made;
  }

  function clamp01(value) {
    return Math.min(0.97, Math.max(0.03, value));
  }

  /** Read the current theme colours from CSS so the canvas follows dark mode. */
  function colors() {
    if (window.Charts && typeof window.Charts.palette === 'function') {
      const p = window.Charts.palette();
      return {
        text: p.text,
        muted: p.muted,
        grid: p.grid,
        danger: p.danger,
        success: p.greenText,
        amber: p.saffron,
        surface: p.surface,
      };
    }
    const styles = getComputedStyle(document.documentElement);
    const read = (name, fallback) =>
      (styles.getPropertyValue(name) || '').trim() || fallback;
    return {
      text: read('--text', '#0f172a'),
      muted: read('--muted', '#55637a'),
      grid: read('--grid-line', '#e6ecf5'),
      danger: read('--danger', '#b91c1c'),
      success: read('--success', '#15803d'),
      amber: read('--amber-strong', '#f59e0b'),
      surface: read('--surface-2', '#f6f8fc')
    };
  }

  /** Keep the backing store sharp on high-DPI screens. */
  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { width, height };
  }

  /** Map normalised data coordinates to canvas pixels (with padding). */
  function toPixel(size, p) {
    const pad = 26;
    return {
      x: pad + p.x * (size.width - pad * 2),
      y: size.height - pad - p.y * (size.height - pad * 2)
    };
  }

  /** Map a pointer position back to normalised data coordinates. */
  function toData(size, clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const pad = 26;
    const x = (clientX - rect.left - pad) / (size.width - pad * 2);
    const y = (size.height - pad - (clientY - rect.top)) / (size.height - pad * 2);
    return { x: clamp01(x), y: clamp01(y) };
  }

  /** The heart of the demo: distances, K nearest, majority vote. */
  function computeNeighbours(k) {
    const withDistance = points.map((point) => ({
      point,
      // Euclidean distance in the 2-D projection shown on screen.
      distance: Math.sqrt(
        Math.pow(point.x - test.x, 2) + Math.pow(point.y - test.y, 2)
      )
    }));
    withDistance.sort((a, b) => a.distance - b.distance);
    const nearest = withDistance.slice(0, k);
    let yes = 0;
    let no = 0;
    nearest.forEach((entry) => {
      if (entry.point.label === 'Yes') yes += 1;
      else no += 1;
    });
    return { nearest, yes, no, closest: withDistance[0] };
  }

  function draw() {
    const size = resizeCanvas();
    const theme = colors();
    const k = currentK();
    const result = computeNeighbours(k);
    const neighbourSet = new Set(result.nearest.map((entry) => entry.point));

    ctx.clearRect(0, 0, size.width, size.height);

    // Plot background + grid.
    ctx.fillStyle = theme.surface;
    ctx.fillRect(0, 0, size.width, size.height);
    ctx.strokeStyle = theme.grid;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const gx = 26 + (i / 10) * (size.width - 52);
      const gy = size.height - 26 - (i / 10) * (size.height - 52);
      ctx.beginPath();
      ctx.moveTo(gx, 26);
      ctx.lineTo(gx, size.height - 26);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(26, gy);
      ctx.lineTo(size.width - 26, gy);
      ctx.stroke();
    }

    // Axis labels.
    ctx.fillStyle = theme.muted;
    ctx.font = '600 10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Sanctioned cost (scaled) →', size.width / 2, size.height - 8);
    ctx.save();
    ctx.translate(11, size.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Category difference →', 0, 0);
    ctx.restore();

    const testPixel = toPixel(size, test);

    // Distance lines to the K nearest neighbours.
    ctx.lineWidth = 1.4;
    ctx.setLineDash([4, 4]);
    result.nearest.forEach((entry) => {
      const pixel = toPixel(size, entry.point);
      ctx.strokeStyle = entry.point.label === 'Yes' ? theme.danger : theme.success;
      ctx.globalAlpha = 0.55;
      ctx.beginPath();
      ctx.moveTo(testPixel.x, testPixel.y);
      ctx.lineTo(pixel.x, pixel.y);
      ctx.stroke();
    });
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    // Sample points. Neighbours get an amber ring so they stand out.
    points.forEach((point) => {
      const pixel = toPixel(size, point);
      const isNeighbour = neighbourSet.has(point);
      ctx.beginPath();
      ctx.arc(pixel.x, pixel.y, isNeighbour ? 7 : 5, 0, Math.PI * 2);
      ctx.fillStyle = point.label === 'Yes' ? theme.danger : theme.success;
      ctx.globalAlpha = isNeighbour ? 1 : 0.5;
      ctx.fill();
      ctx.globalAlpha = 1;
      if (isNeighbour) {
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = theme.amber;
        ctx.stroke();
      }
    });

    // The draggable test point.
    ctx.beginPath();
    ctx.arc(testPixel.x, testPixel.y, 11, 0, Math.PI * 2);
    ctx.fillStyle = theme.amber;
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(testPixel.x, testPixel.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#1a1205';
    ctx.fill();

    updateReadouts(result, k, theme);
  }

  function currentK() {
    if (!kSlider) return 5;
    const value = parseInt(kSlider.value, 10);
    const max = parseInt(kSlider.max || '25', 10);
    return Number.isFinite(value) ? Math.min(max, Math.max(1, value)) : 5;
  }

  /** Write the vote, the counts and the nearest distance into the readout line. */
  function updateReadouts(result, k, theme) {
    if (kValue) kValue.textContent = String(k);
    if (kSlider) {
      const max = parseInt(kSlider.max || '25', 10);
      kSlider.style.setProperty('--fill', (((k - 1) / Math.max(1, max - 1)) * 100).toFixed(1) + '%');
    }
    if (!readoutEl) return;

    const winner = result.yes === result.no ? 'Tie' : result.yes > result.no ? 'Yes' : 'No';
    const nearest = result.closest ? result.closest.distance.toFixed(3) : '0';
    if (winner === 'Yes') {
      readoutEl.textContent = 'Yes — OVER SANCTION. ' + result.yes + ' of ' + k
        + ' neighbours went over. Closest neighbour distance ' + nearest + '.';
      readoutEl.style.color = theme.danger;
    } else if (winner === 'No') {
      readoutEl.textContent = 'No — WITHIN SANCTION. ' + result.no + ' of ' + k
        + ' neighbours stayed within. Closest neighbour distance ' + nearest + '.';
      readoutEl.style.color = theme.success;
    } else {
      readoutEl.textContent = 'Tie at K=' + k + ': ' + result.yes
        + ' voted over and ' + result.no + ' voted within — sklearn breaks the tie by class order, '
        + 'which is exactly why the predictor shows the vote counts.';
      readoutEl.style.color = theme.amber;
    }
  }

  /* ---------------------------- interaction ---------------------------- */

  function pointerPosition(event) {
    const size = {
      width: canvas.getBoundingClientRect().width,
      height: canvas.getBoundingClientRect().height
    };
    return toData(size, event.clientX, event.clientY);
  }

  function onDown(event) {
    dragging = true;
    canvas.classList.add('dragging');
    if (typeof canvas.setPointerCapture === 'function' && event.pointerId !== undefined) {
      try { canvas.setPointerCapture(event.pointerId); } catch (err) { /* ignore */ }
    }
    test = pointerPosition(event);
    draw();
    event.preventDefault();
  }

  function onMove(event) {
    if (!dragging) return;
    test = pointerPosition(event);
    draw();
    event.preventDefault();
  }

  function onUp(event) {
    if (!dragging) return;
    dragging = false;
    canvas.classList.remove('dragging');
    if (typeof canvas.releasePointerCapture === 'function' && event && event.pointerId !== undefined) {
      try { canvas.releasePointerCapture(event.pointerId); } catch (err) { /* ignore */ }
    }
  }

  function bind() {
    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);

    if (kSlider) {
      kSlider.addEventListener('input', draw);
    }
    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        test = { x: DEFAULT_TEST.x, y: DEFAULT_TEST.y };
        draw();
      });
    }
    if (randomBtn) {
      // A new deterministic sample set, so two viewers can compare "same K, different data".
      randomBtn.addEventListener('click', () => {
        seed = (seed * 1103515245 + 12345) % 2147483647;
        points = buildPoints();
        draw();
      });
    }

    // Redraw when the section is resized (window resize, sidebar collapse,
    // orientation change). ResizeObserver is not available in very old browsers.
    if (typeof ResizeObserver === 'function') {
      new ResizeObserver(() => draw()).observe(canvas);
    } else {
      window.addEventListener('resize', draw);
    }
  }

  function init() {
    points = buildPoints();
    bind();
    draw();
  }

  window.KnnDemo = {
    init,
    draw,
    refreshTheme: draw,
    /** Used by the smoke test to confirm the demo reacts to a drag. */
    moveTestPointTo: (x, y) => {
      test = { x: clamp01(x), y: clamp01(y) };
      draw();
      return { vote: readoutEl ? readoutEl.textContent : '', k: currentK() };
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
