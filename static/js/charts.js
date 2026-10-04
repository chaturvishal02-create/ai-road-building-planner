/* ==========================================================================
   charts.js -- every Chart.js instance in the app, in one place
   --------------------------------------------------------------------------
   app.js calls Charts.render(key, canvasId, data). Keeping all chart configs
   here means the colour palette, the axis formatting and the dark-mode
   behaviour are decided once instead of in nine places.

   Chart.js reads CSS variables poorly, so palette() samples the custom
   properties from the document -- that is what makes refreshTheme() work after
   the toggle is pressed.
   ========================================================================== */

window.Charts = (function () {
  'use strict';

  const instances = Object.create(null);

  function cssVar(name, fallback) {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  }

  function palette() {
    return {
      saffron: cssVar('--saffron', '#ff9933'),
      saffronText: cssVar('--saffron-text', '#b25a00'),
      green: cssVar('--india-green', '#138808'),
      greenText: cssVar('--green-text', '#0b6b0b'),
      navy: cssVar('--navy', '#0b2545'),
      navy300: cssVar('--navy-300', '#8aa5c8'),
      danger: cssVar('--danger', '#b91c1c'),
      text: cssVar('--text', '#0f172a'),
      muted: cssVar('--muted', '#55637a'),
      grid: cssVar('--grid-line', '#e6ecf5'),
      surface: cssVar('--surface', '#ffffff'),
    };
  }

  // Series colours cycle through the tricolour family rather than Chart.js defaults.
  function series(p, index) {
    const cycle = [p.saffron, p.green, p.navy, p.saffronText, p.navy300, p.danger];
    return cycle[index % cycle.length];
  }

  function withAlpha(hex, alpha) {
    const value = String(hex).replace('#', '');
    if (value.length !== 6) return hex;
    const n = parseInt(value, 16);
    return 'rgba(' + ((n >> 16) & 255) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + alpha + ')';
  }

  function baseOptions(p) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 450 },
      plugins: {
        legend: {
          labels: { color: p.text, boxWidth: 12, font: { size: 11, family: 'Inter' } },
        },
        tooltip: {
          backgroundColor: p.navy,
          titleColor: '#ffffff',
          bodyColor: '#ffffff',
          padding: 9,
          displayColors: false,
        },
      },
      scales: {
        x: { ticks: { color: p.muted, font: { size: 10 } }, grid: { color: p.grid } },
        y: { ticks: { color: p.muted, font: { size: 10 } }, grid: { color: p.grid } },
      },
    };
  }

  /* ---------------------------------------------------------------------- */
  /* builders: each takes (canvas, data, p) and returns a Chart config      */
  /* ---------------------------------------------------------------------- */

  const builders = {
    // Scheme distribution: doughnut of project counts.
    scheme(canvas, data, p) {
      return {
        type: 'doughnut',
        data: {
          labels: data.map((d) => d.label),
          datasets: [{
            data: data.map((d) => d.value),
            backgroundColor: data.map((d, i) => series(p, i)),
            borderColor: p.surface,
            borderWidth: 2,
          }],
        },
        options: Object.assign(baseOptions(p), {
          cutout: '58%',
          scales: { x: { display: false }, y: { display: false } },
          plugins: Object.assign(baseOptions(p).plugins, {
            legend: { position: 'right', labels: { color: p.text, boxWidth: 12, font: { size: 11 } } },
          }),
        }),
      };
    },

    // Over-budget rate by scheme: bar, percentage axis.
    overrun(canvas, data, p) {
      return {
        type: 'bar',
        data: {
          labels: data.map((d) => d.label),
          datasets: [{
            label: 'Over sanction (%)',
            data: data.map((d) => d.value),
            backgroundColor: data.map((d) => (d.value >= 45 ? withAlpha(p.danger, 0.85) : withAlpha(p.saffron, 0.85))),
            borderRadius: 5,
          }],
        },
        options: Object.assign(baseOptions(p), {
          indexAxis: 'y',
          scales: {
            x: { beginAtZero: true, ticks: { color: p.muted, callback: (v) => v + '%' }, grid: { color: p.grid } },
            y: { ticks: { color: p.muted, font: { size: 10 } }, grid: { display: false } },
          },
          plugins: Object.assign(baseOptions(p).plugins, { legend: { display: false } }),
        }),
      };
    },

    // State distribution: stacked bar within-budget / over-budget.
    state(canvas, data, p) {
      return {
        type: 'bar',
        data: {
          labels: data.map((d) => d.label),
          datasets: [
            {
              label: 'Within sanction',
              data: data.map((d) => d.within),
              backgroundColor: withAlpha(p.green, 0.8),
              stack: 's',
              borderRadius: 3,
            },
            {
              label: 'Over sanction',
              data: data.map((d) => d.over),
              backgroundColor: withAlpha(p.saffron, 0.92),
              stack: 's',
              borderRadius: 3,
            },
          ],
        },
        options: Object.assign(baseOptions(p), {
          scales: {
            x: { stacked: true, ticks: { color: p.muted, font: { size: 9 }, maxRotation: 60 }, grid: { display: false } },
            y: { stacked: true, beginAtZero: true, ticks: { color: p.muted }, grid: { color: p.grid } },
          },
        }),
      };
    },

    // Contractor split: grouped bar of counts by class.
    contractor(canvas, data, p) {
      return {
        type: 'bar',
        data: {
          labels: data.map((d) => d.label),
          datasets: [
            {
              label: 'Over sanction',
              data: data.map((d) => d.over),
              backgroundColor: withAlpha(p.saffron, 0.92),
              borderRadius: 4,
            },
            {
              label: 'Within sanction',
              data: data.map((d) => d.within),
              backgroundColor: withAlpha(p.green, 0.8),
              borderRadius: 4,
            },
          ],
        },
        options: baseOptions(p),
      };
    },

    // K versus accuracy line, with the shipped default and the best K marked.
    kcurve(canvas, data, p) {
      const best = data.points.reduce((a, b) => (b.accuracy > a.accuracy ? b : a), data.points[0]);
      return {
        type: 'line',
        data: {
          labels: data.ks,
          datasets: [{
            label: 'Test accuracy',
            data: data.accuracies,
            borderColor: p.saffronText,
            backgroundColor: withAlpha(p.saffron, 0.18),
            pointRadius: data.ks.map((k) => (k === best.k || k === data.default_k ? 5 : 2)),
            pointBackgroundColor: data.ks.map((k) => (k === best.k ? p.green : p.saffronText)),
            fill: true,
            tension: 0.25,
          }],
        },
        options: Object.assign(baseOptions(p), {
          scales: {
            // 31 K values in a half-width card: without a tick limit the labels collide
            // and get cut in half, so show every fourth K and let the tooltip fill in.
            x: {
              title: { display: true, text: 'K neighbours', color: p.muted },
              ticks: { color: p.muted, maxTicksLimit: 8 },
              grid: { color: p.grid },
            },
            y: { suggestedMin: 0.4, suggestedMax: 1, ticks: { color: p.muted }, grid: { color: p.grid } },
          },
          plugins: Object.assign(baseOptions(p).plugins, {
            legend: { display: false },
            tooltip: Object.assign(baseOptions(p).plugins.tooltip, {
              callbacks: {
                label(item) {
                  const row = data.points[item.dataIndex];
                  return 'accuracy ' + row.accuracy.toFixed(4) + ' on ' + data.test_size + ' test projects';
                },
              },
            }),
          }),
        }),
      };
    },

    // Horizontal importance bars (Random Forest).
    importance(canvas, data, p) {
      return {
        type: 'bar',
        data: {
          labels: data.map((d) => d.feature),
          datasets: [{
            label: 'Importance (%)',
            data: data.map((d) => d.importance_pct),
            backgroundColor: withAlpha(p.navy, 0.75),
            borderRadius: 4,
          }],
        },
        options: Object.assign(baseOptions(p), {
          indexAxis: 'y',
          scales: {
            x: { beginAtZero: true, ticks: { color: p.muted, callback: (v) => v + '%' }, grid: { color: p.grid } },
            y: { ticks: { color: p.muted, font: { size: 10 } }, grid: { display: false } },
          },
          plugins: Object.assign(baseOptions(p).plugins, { legend: { display: false } }),
        }),
      };
    },

    // Dashboard: sanctioned against actual, in crore.
    budgetActual(canvas, data, p) {
      const lakhPerCrore = 100;
      return {
        type: 'bar',
        data: {
          labels: data.map((d) => d.scheme_name),
          datasets: [
            {
              label: 'Sanctioned (crore)',
              data: data.map((d) => +(d.sanctioned_lakh / lakhPerCrore).toFixed(2)),
              backgroundColor: withAlpha(p.navy, 0.72),
              borderRadius: 4,
            },
            {
              label: 'Actual (crore)',
              data: data.map((d) => +(d.actual_lakh / lakhPerCrore).toFixed(2)),
              backgroundColor: withAlpha(p.saffron, 0.92),
              borderRadius: 4,
            },
          ],
        },
        options: Object.assign(baseOptions(p), {
          scales: {
            x: { ticks: { color: p.muted, font: { size: 9 }, maxRotation: 45 }, grid: { display: false } },
            y: {
              beginAtZero: true,
              title: { display: true, text: '₹ crore', color: p.muted },
              ticks: { color: p.muted },
              grid: { color: p.grid },
            },
          },
          plugins: Object.assign(baseOptions(p).plugins, {
            tooltip: Object.assign(baseOptions(p).plugins.tooltip, {
              callbacks: {
                afterBody(items) {
                  const row = data[items[0].dataIndex];
                  return 'gap ' + row.overrun_inr + ' (' + row.overrun_pct + '%)';
                },
              },
            }),
          }),
        }),
      };
    },

    // Dashboard: state-wise distribution, stacked by outcome.
    stateDist(canvas, data, p) {
      return builders.state(canvas, data.map((d) => ({
        label: d.state, within: d.within_budget, over: d.over_budget,
      })), p);
    },

    // Route planner: doughnut of the cost breakdown in ₹ lakh.
    routeSplit(canvas, data, p) {
      const total = data.reduce((sum, d) => sum + (d.value || 0), 0) || 1;
      return {
        type: 'doughnut',
        data: {
          labels: data.map((d) => d.label),
          datasets: [{
            data: data.map((d) => d.value),
            backgroundColor: data.map((d, i) => series(p, i)),
            borderColor: p.surface,
            borderWidth: 2,
          }],
        },
        options: Object.assign(baseOptions(p), {
          cutout: '58%',
          scales: { x: { display: false }, y: { display: false } },
          plugins: Object.assign(baseOptions(p).plugins, {
            legend: { position: 'right', labels: { color: p.text, boxWidth: 12, font: { size: 10 } } },
            tooltip: Object.assign(baseOptions(p).plugins.tooltip, {
              callbacks: {
                label(item) {
                  const value = item.parsed;
                  return item.label + ': ₹ ' + value.toFixed(1) + ' lakh (' +
                    ((value / total) * 100).toFixed(1) + '%)';
                },
              },
            }),
          }),
        }),
      };
    },

    // Planning formula + building estimator: one pie of a stated percentage split.
    // Shared by both pages because both show "these are the lines the total is made
    // of"; the numbers always come from the API, never from this file.
    costSplit(canvas, data, p) {
      const total = data.reduce((sum, d) => sum + (d.value || 0), 0) || 1;
      return {
        type: 'pie',
        data: {
          labels: data.map((d) => d.label),
          datasets: [{
            data: data.map((d) => d.value),
            backgroundColor: data.map((d, i) => series(p, i)),
            borderColor: p.surface,
            borderWidth: 2,
          }],
        },
        options: Object.assign(baseOptions(p), {
          scales: { x: { display: false }, y: { display: false } },
          plugins: Object.assign(baseOptions(p).plugins, {
            legend: { position: 'right', labels: { color: p.text, boxWidth: 12, font: { size: 10 } } },
            tooltip: Object.assign(baseOptions(p).plugins.tooltip, {
              callbacks: {
                label(item) {
                  const value = item.parsed;
                  return item.label + ': ₹ ' + value.toLocaleString('en-IN')
                    + ' lakh (' + ((value / total) * 100).toFixed(1) + '%)';
                },
              },
            }),
          }),
        }),
      };
    },

    // Route planner: total ₹ crore per corridor against its length.
    routeBudget(canvas, data, p) {
      return {
        type: 'bar',
        data: {
          labels: data.map((d) => d.label),
          datasets: [
            {
              label: 'Estimated cost (₹ crore)',
              data: data.map((d) => d.cost),
              backgroundColor: withAlpha(p.navy, 0.72),
              borderRadius: 4,
              yAxisID: 'y',
            },
            {
              label: 'Distance (km)',
              data: data.map((d) => d.distance),
              type: 'line',
              borderColor: p.saffronText,
              backgroundColor: withAlpha(p.saffron, 0.16),
              pointRadius: 3,
              pointBackgroundColor: p.saffron,
              borderWidth: 2,
              fill: false,
              yAxisID: 'y1',
            },
          ],
        },
        options: Object.assign(baseOptions(p), {
          scales: {
            x: { ticks: { color: p.muted, font: { size: 9 }, maxRotation: 55 }, grid: { display: false } },
            y: {
              beginAtZero: true,
              position: 'left',
              title: { display: true, text: '₹ crore', color: p.muted },
              ticks: { color: p.muted, maxTicksLimit: 8 },
              grid: { color: p.grid },
            },
            y1: {
              beginAtZero: true,
              position: 'right',
              title: { display: true, text: 'km', color: p.muted },
              ticks: { color: p.muted, maxTicksLimit: 8 },
              grid: { display: false },
            },
          },
          plugins: Object.assign(baseOptions(p).plugins, {
            tooltip: Object.assign(baseOptions(p).plugins.tooltip, {
              callbacks: {
                afterBody(items) {
                  const row = data[items[0].dataIndex];
                  const lines = [];
                  if (row.published) lines.push('published road distance ' + row.published + ' km');
                  lines.push('₹ ' + row.perKm + ' crore per km');
                  return lines;
                },
              },
            }),
          }),
        }),
      };
    },

    // Route planner: completion months per corridor, with its budget on the far axis.
    routeTime(canvas, data, p) {
      return {
        type: 'bar',
        data: {
          labels: data.map((d) => d.label),
          datasets: [
            {
              label: 'Construction time (months)',
              data: data.map((d) => d.months),
              backgroundColor: withAlpha(p.green, 0.78),
              borderRadius: 4,
              yAxisID: 'y',
            },
            {
              label: 'Budget (₹ crore)',
              data: data.map((d) => d.cost),
              type: 'line',
              borderColor: p.saffronText,
              backgroundColor: withAlpha(p.saffron, 0.16),
              pointRadius: 3,
              pointBackgroundColor: p.saffron,
              borderWidth: 2,
              fill: false,
              yAxisID: 'y1',
            },
          ],
        },
        options: Object.assign(baseOptions(p), {
          scales: {
            x: { ticks: { color: p.muted, font: { size: 9 }, maxRotation: 55 }, grid: { display: false } },
            y: {
              beginAtZero: true,
              position: 'left',
              title: { display: true, text: 'months', color: p.muted },
              ticks: { color: p.muted, maxTicksLimit: 8 },
              grid: { color: p.grid },
            },
            y1: {
              beginAtZero: true,
              position: 'right',
              title: { display: true, text: '₹ crore', color: p.muted },
              ticks: { color: p.muted, maxTicksLimit: 8 },
              grid: { display: false },
            },
          },
          plugins: Object.assign(baseOptions(p).plugins, {
            tooltip: Object.assign(baseOptions(p).plugins.tooltip, {
              callbacks: {
                afterBody(items) {
                  const row = data[items[0].dataIndex];
                  return row.min + '-' + row.max + ' months at '
                    + row.speed.join('-') + ' km/day';
                },
              },
            }),
          }),
        }),
      };
    },

    // Dashboard: overrun share and severity by execution type.
    contractorRisk(canvas, data, p) {
      return {
        type: 'bar',
        data: {
          labels: data.map((d) => d.contractor_type),
          datasets: [
            {
              label: 'Share over sanction (%)',
              data: data.map((d) => d.over_budget_share_pct),
              backgroundColor: withAlpha(p.saffron, 0.92),
              borderRadius: 4,
            },
            {
              label: 'Mean overrun of those works (%)',
              data: data.map((d) => d.mean_overrun_pct),
              backgroundColor: withAlpha(p.green, 0.8),
              borderRadius: 4,
            },
          ],
        },
        options: Object.assign(baseOptions(p), {
          scales: {
            x: { ticks: { color: p.muted }, grid: { display: false } },
            y: { beginAtZero: true, ticks: { color: p.muted, callback: (v) => v + '%' }, grid: { color: p.grid } },
          },
        }),
      };
    },
  };

  function destroy(key) {
    if (instances[key]) {
      instances[key].destroy();
      delete instances[key];
    }
  }

  /**
   * Build or rebuild a chart. Always destroys the previous instance first: Chart.js
   * leaks a canvas listener otherwise, and the theme toggle re-renders everything.
   */
  function render(key, canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    destroy(key);
    const p = palette();
    const builder = builders[key];
    if (!builder) throw new Error('Charts.render: unknown chart key "' + key + '"');
    const config = builder(canvas, data || [], p);
    config.options = config.options || {};
    config.options.plugins = config.options.plugins || {};
    instances[key] = new Chart(canvas, config);
    return instances[key];
  }

  function refreshTheme() {
    Object.keys(instances).forEach((key) => {
      const chart = instances[key];
      const p = palette();
      const datasets = chart.data.datasets || [];
      // Rather than rebuild every config, recolour what is visible and let Chart.js redraw.
      datasets.forEach((set, index) => {
        if (chart.config.type === 'doughnut') {
          set.backgroundColor = (set.data || []).map((_, i) => series(p, i));
          set.borderColor = p.surface;
        } else if (!set.label || !/Within/i.test(set.label)) {
          set.borderColor = set.borderColor ? series(p, index) : set.borderColor;
        }
      });
      if (chart.options.scales) {
        Object.values(chart.options.scales).forEach((scale) => {
          if (scale.ticks) scale.ticks.color = p.muted;
          if (scale.grid) scale.grid.color = p.grid;
          if (scale.title) scale.title.color = p.muted;
        });
      }
      if (chart.options.plugins.legend && chart.options.plugins.legend.labels) {
        chart.options.plugins.legend.labels.color = p.text;
      }
      chart.update('none');
    });
  }

  return {
    render: render,
    destroy: destroy,
    refreshTheme: refreshTheme,
    palette: palette,
    instanceCount: () => Object.keys(instances).length,
  };
})();
