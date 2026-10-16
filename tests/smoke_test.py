"""Headless browser smoke test (Playwright) for the India FY 2025-26 build.

Run it directly -- it starts the Flask server itself, so nothing else needs to
be running::

    py tests/smoke_test.py

It checks, in a real Chromium window:
  * every section exists and is visible, and the app booted without an error banner
  * the team credits (both developers, the guide, DBATU) appear in the hero, the Team
    section and the footer, and the simulated data is labelled as simulated
  * the page is India-only: rupee amounts everywhere, no "$", no leftover paper/Thai text
  * all thirteen Chart.js charts on the main page are registered with real data, and all
    fourteen canvases, including the hand-drawn KNN demo, have painted pixels
  * the hero counters, the KPIs and the feature-store manifest show the 700 simulated rows
  * the dataset explorer searches, filters, sorts, paginates and links to the sample CSV
  * the predictor validates, shows verdict + risk %, rupee cost estimates, the neighbour
    table at K, the model switch, and the advisory
  * the route planner lists Indian cities, draws the corridor, prices it in ₹ crore,
    prints the completion months from the km/day speed band, honours a forced ₹/km rate,
    rejects a bad length, and labels the trained-model cross-check when the sanction is
    larger than anything it was trained on
  * the same panel switched to location mode walks State -> District -> Taluka -> Village
    on both ends, refills the dependent dropdowns, resolves a village searched from the
    auto-suggest list, and fills the planning-formula panel and its material-split chart
    with the server's own arithmetic
  * /building is a second page on the same project: its four cascades, the cost card in
    ₹ lakh, the seven work-section lines that add back to the card, the pie, a preset
    chip, and a rejected area that names the field
  * a fresh page in a 500x513 window reveals every section and leaves the map visible -- a
    scroll-reveal threshold expressed as a share of the section's own area can never be met
    by a section taller than ~12 screens, which is how the route map came to look missing
  * the CSV and PDF exports download and carry the credits
  * the live SSE feed connects and grows with each prediction, without duplicates
  * the KNN demo reacts to a real mouse drag, to the K slider and to its buttons
  * the retrain guardrail rejects the 20-row sample instead of poisoning the models
  * dark mode changes the *rendered* colours
  * there is no horizontal overflow at 1280 px and at 375 px, and the predictor still
    works at phone width
  * there are no console errors, no uncaught page errors and no failed requests

Screenshots of every section are written to ``screenshots/``.
Exits non-zero if any check fails.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# The report prints rupee signs and middle dots; the Windows console default (cp1252)
# cannot encode them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCREENSHOT_DIR = os.path.join(PROJECT_ROOT, "screenshots")
SAMPLE_CSV = os.path.join(PROJECT_ROOT, "data", "sample_20_projects.csv")

SECTIONS = ["home", "dataset", "predictor", "performance", "dashboard", "routes",
            "knn", "about", "team", "footer"]

# The nine original Chart.js charts, the four the route planner adds (three series
# charts plus the planning formula's material split), plus the hand-drawn KNN demo
# canvas. The building page has its own pie and is checked on its own page.
CHARTJS_CANVASES = [
    "chart-scheme", "chart-overrun", "chart-state", "chart-contractor",
    "chart-kcurve", "chart-importance-rf",
    "chart-budget-actual", "chart-state-dist", "chart-contractor-risk",
    "chart-route-split", "chart-planning-split",
    "chart-route-budget", "chart-route-time",
]
PAINT_CANVASES = CHARTJS_CANVASES + ["knn-canvas"]

BUILDING_CANVAS = "chart-building-split"

# Leaflet, its CSS and the OpenStreetMap raster tiles all come off public hosts.
# A machine with no outbound network (or one OSM edge being briefly grumpy) must
# not fail the whole smoke run: the app shows an on-page fallback for exactly this
# case, so those requests are tolerated here and the fallback is asserted instead.
EXTERNAL_HOSTS = ("unpkg.com", "tile.openstreetmap.org", "cdnjs.cloudflare.com",
                 "cdn.jsdelivr.net")


def from_external(url: str) -> bool:
    return any(host in url for host in EXTERNAL_HOSTS)


SCHEMES = ["Jal Jeevan Mission", "PMAY (Housing)", "PMGSY (Rural Roads)", "Smart Cities Mission"]
CORE_STATES = ["Maharashtra", "Karnataka", "Delhi", "Uttar Pradesh"]
CORE_DISTRICTS = ["Pune", "Mumbai", "Nashik", "Bengaluru"]

DEVELOPERS = ["Vishal Raghunath Chatur", "Mohsin Jabiulla Balgyar"]
GUIDE = "Tapase Miss"
RUPEE = "₹"

# The rewrite removed the Thai paper basis entirely; none of these may reappear.
LEFTOVER_TERMS = ["kusonkhum", "thailand", "bangkok", "chiang", "us$", "$",
                  "reported in paper", "procurement method"]

# Requests that are *supposed* to answer 4xx: the deliberate bad-input probes on each
# of the two pages.
EXPECTED_CLIENT_PATHS = ("/predict", "/retrain", "/predict-route", "/predict-building")

# Collected results: (name, passed, detail)
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(passed), detail))
    print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    return bool(passed)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_server(port: int, log_path: str) -> subprocess.Popen:
    env = dict(os.environ, PORT=str(port), PYTHONUNBUFFERED="1")
    log = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def wait_for_server(url: str, timeout: float = 180.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return True
        except (URLError, OSError):
            time.sleep(0.5)
    return False


# --------------------------------------------------------------------------- #
# Browser-side probes
# --------------------------------------------------------------------------- #

JS_CANVAS_REPORT = """
(ids) => ids.map((id) => {
  const canvas = document.getElementById(id);
  if (!canvas) return { id, exists: false };
  const rect = canvas.getBoundingClientRect();
  let painted = 0;
  let error = null;
  try {
    const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
    for (let i = 3; i < data.length; i += 400) { if (data[i] !== 0) painted++; }
  } catch (err) { error = String(err); }
  return {
    id, exists: true, error,
    bufferWidth: canvas.width, bufferHeight: canvas.height,
    cssWidth: Math.round(rect.width), cssHeight: Math.round(rect.height),
    painted
  };
})
"""

# Chart.js keeps every live chart in Chart.instances; read the label sets back out so
# a chart that was registered with empty data cannot pass as "rendered".
JS_CHART_REPORT = """
() => {
  const live = (window.Chart && Chart.instances) ? Object.values(Chart.instances) : [];
  return live.map((chart) => ({
    id: chart.canvas ? chart.canvas.id : '',
    type: chart.config.type,
    labels: chart.data.labels ? chart.data.labels.slice(0, 30) : [],
    points: chart.data.datasets.reduce((sum, d) => sum + (d.data ? d.data.length : 0), 0)
  }));
}
"""

JS_OVERFLOW = """
() => {
  const de = document.documentElement;
  const body = document.body;
  const viewport = de.clientWidth;
  const offenders = [];
  document.querySelectorAll('body *').forEach((el) => {
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) return;
    // Only right-hand overflow can widen the scrollable area. Elements parked
    // off-screen to the left (the skip link at left:-999px) cannot.
    if (rect.right <= viewport + 1) return;
    // Ignore anything inside a container that is allowed to scroll or clip.
    let node = el.parentElement;
    let clipped = false;
    while (node && node !== body) {
      const style = getComputedStyle(node);
      if (/(auto|scroll|hidden)/.test(style.overflowX)) { clipped = true; break; }
      node = node.parentElement;
    }
    if (clipped) return;
    offenders.push({
      tag: el.tagName.toLowerCase(),
      id: el.id || '',
      cls: String(el.className || '').slice(0, 70),
      left: Math.round(rect.left),
      right: Math.round(rect.right)
    });
  });
  return {
    viewport,
    htmlScrollWidth: de.scrollWidth,
    htmlClientWidth: de.clientWidth,
    bodyScrollWidth: body.scrollWidth,
    bodyClientWidth: body.clientWidth,
    innerWidth: window.innerWidth,
    offenders: offenders.slice(0, 12),
    offenderCount: offenders.length
  };
}
"""

JS_RENDERED = """
(selector) => {
  const el = document.querySelector(selector);
  if (!el) return null;
  const style = getComputedStyle(el);
  const rect = el.getBoundingClientRect();
  return {
    display: style.display,
    visibility: style.visibility,
    opacity: style.opacity,
    width: Math.round(rect.width),
    height: Math.round(rect.height),
    text: (el.textContent || '').trim().slice(0, 200)
  };
}
"""

JS_BODY_BG = "() => getComputedStyle(document.body).backgroundColor"

JS_SCROLL_TO = "(id) => { const el = document.getElementById(id); if (el) el.scrollIntoView({ block: 'start' }); }"

# The stylesheet asks for a smooth scroll, so a fixed wait can measure a position the
# page has not reached yet. 'instant' puts the canvas exactly where the drag needs it:
# fully inside the viewport, because Playwright moves the real mouse in viewport pixels.
JS_CENTER_INSTANTLY = """(id) => {
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ block: 'center', behavior: 'instant' });
}"""

JS_SET_RANGE = """
([id, value]) => {
  const node = document.getElementById(id);
  node.value = String(value);
  node.dispatchEvent(new Event('input', { bubbles: true }));
  node.dispatchEvent(new Event('change', { bubbles: true }));
}
"""

JS_FIELD = "(id) => { const el = document.getElementById(id); return el ? el.value : null; }"


def rendered(page, selector):
    return page.evaluate(JS_RENDERED, selector)


def text_of(page, selector) -> str:
    return page.inner_text(selector).strip()


def is_visible(state) -> bool:
    return bool(
        state
        and state["display"] != "none"
        and state["visibility"] != "hidden"
        and float(state["opacity"] or 0) > 0.01
        and state["width"] > 0
        and state["height"] > 0
    )


def set_range(page, element_id: str, value) -> None:
    page.evaluate(JS_SET_RANGE, [element_id, value])
    page.wait_for_timeout(200)


def submit_predict(page, wait_ms: int = 1600) -> None:
    page.click("#predict-submit")
    page.wait_for_timeout(wait_ms)


# ---------------------------------------------------------------------------
# Helpers for the longer assertions
# ---------------------------------------------------------------------------

def check_sections(page) -> None:
    for section_id in SECTIONS:
        state = rendered(page, f"#{section_id}")
        check(f"section #{section_id} present and visible", is_visible(state),
              "" if not state else f"display={state['display']} h={state['height']}")
    # Brief B asked for a Building link added without removing an existing one, so both
    # halves are pinned here: every section anchor still resolves, and exactly one new
    # link points at the second page.
    nav = page.eval_on_selector_all("#navbar .nav-link",
                                    "els => els.map(e => e.getAttribute('href'))")
    check("navbar keeps a link to every section of the main page",
          all("#" + section_id in nav for section_id in SECTIONS[:-1]), str(nav))
    building_links = [href for href in nav if (href or "").endswith("/building")]
    check("navbar adds one link to the building page", len(building_links) == 1,
          "%d of %d links: %s" % (len(building_links), len(nav), building_links))
    check("desktop sidebar is fixed and content clears it", page.evaluate("""() => {
      const sidebar = document.getElementById('navbar');
      return getComputedStyle(sidebar).position === 'fixed' && !sidebar.inert
        && document.querySelector('.dashboard-workspace').getBoundingClientRect().left
          >= sidebar.getBoundingClientRect().right;
    }"""))
    page.click('#navbar a[href="#predictor"]')
    page.wait_for_timeout(900)
    check("sidebar navigation updates active item and topbar", page.evaluate("""() =>
      document.getElementById('workspace-title').textContent === 'Budget predictor'
      && document.querySelector('#navbar a[href="#predictor"]').getAttribute('aria-current') === 'location'
      && document.querySelectorAll('#navbar .nav-link.active').length === 1"""))
    page.click('#navbar a[href="#home"]')
    page.wait_for_timeout(900)


def check_credits(page, body_text: str) -> None:
    hero = page.inner_text("#home")
    cards = page.inner_text("#team-cards")
    team_section = page.inner_text("#team")
    footer = page.inner_text("#footer")
    for name in DEVELOPERS:
        check(f"hero mentions {name}", name in hero)
        check(f"team cards mention {name}", name in cards)
        check(f"footer mentions {name}", name in footer)
    check("hero mentions the guide", GUIDE in hero)
    check("team section mentions the guide", GUIDE in team_section)
    check("footer mentions the guide", GUIDE in footer)
    check("hero mentions DBATU", "DBATU" in hero)
    check("team section mentions DBATU", "DBATU" in team_section)
    check("footer mentions DBATU", "DBATU" in footer)
    check("hero labels the project rows as simulated", "simulated" in hero.lower())
    check("hero names the financial year", "2025-26" in hero)
    check("footer carries the data note",
          "Simulated Indian scheme dataset" in footer, footer.replace("\n", " ")[:120])
    lowered = body_text.lower()
    found = [term for term in LEFTOVER_TERMS if term in lowered]
    check("no leftover Thai-paper or dollar wording anywhere", not found,
          f"found {found}")


def check_currency(page, body_text: str) -> None:
    check(f"rupee sign appears throughout the page ({RUPEE})", body_text.count(RUPEE) >= 30,
          f"{body_text.count(RUPEE)} occurrences")
    check("no dollar sign in the visible page", "$" not in body_text,
          "; ".join(line.strip()[:50] for line in body_text.splitlines() if "$" in line)[:180])
    sanctioned = text_of(page, "#counter-sanctioned")
    check("hero sanctioned counter is a rupee amount",
          sanctioned.startswith(RUPEE) and ("crore" in sanctioned or "lakh" in sanctioned),
          sanctioned)


def check_counters(page) -> None:
    expectations = {
        "counter-projects": "700",
        "counter-states": "14",
        "counter-schemes": "4",
    }
    for element_id, expected in expectations.items():
        check(f"#{element_id} settles at {expected}",
              text_of(page, f"#{element_id}") == expected, text_of(page, f"#{element_id}"))
    districts = text_of(page, "#counter-districts")
    check("#counter-districts is a plausible district count",
          districts.isdigit() and 20 <= int(districts) <= 60, districts)
    overrun = text_of(page, "#counter-overrun")
    check("#counter-overrun is a percentage", bool(re.match(r"^\d{2}\.\d{2}%$", overrun)), overrun)


def check_charts(page) -> None:
    canvas_report = page.evaluate(JS_CANVAS_REPORT, PAINT_CANVASES)
    for entry in canvas_report:
        canvas_id = entry["id"]
        if not entry.get("exists"):
            check(f"canvas #{canvas_id} exists", False, "element not found")
            continue
        painted_ok = (
            entry["bufferWidth"] > 0
            and entry["bufferHeight"] > 0
            and entry["cssWidth"] > 0
            and entry["cssHeight"] > 0
            and entry["painted"] > 0
            and not entry.get("error")
        )
        check(f"canvas #{canvas_id} rendered pixels", painted_ok,
              f"buffer={entry['bufferWidth']}x{entry['bufferHeight']} "
              f"css={entry['cssWidth']}x{entry['cssHeight']} painted={entry['painted']}"
              + (f" error={entry['error']}" if entry.get("error") else ""))

    charts = {entry["id"]: entry for entry in page.evaluate(JS_CHART_REPORT)}
    for canvas_id in CHARTJS_CANVASES:
        entry = charts.get(canvas_id)
        check(f"chart #{canvas_id} is registered with data",
              bool(entry) and entry["points"] > 0
              and entry["type"] in ("bar", "line", "doughnut", "pie"),
              "" if not entry else f"type={entry['type']} labels={len(entry['labels'])} "
                                   f"points={entry['points']}")

    budget = charts.get("chart-budget-actual") or {}
    state_dist = charts.get("chart-state-dist") or {}
    check("Budget vs actual chart covers the four central schemes",
          sorted(budget.get("labels", [])) == SCHEMES, str(budget.get("labels")))
    check("state-wise distribution chart lists the lead states",
          all(name in state_dist.get("labels", []) for name in CORE_STATES),
          str(state_dist.get("labels")))


def check_vocabularies(page) -> None:
    options = {
        "scheme": page.eval_on_selector_all("#in-scheme option", "els => els.map(e => e.value)"),
        "state": page.eval_on_selector_all("#in-state option", "els => els.map(e => e.value)"),
        "district": page.eval_on_selector_all("#in-district option", "els => els.map(e => e.value)"),
        "contractor": page.eval_on_selector_all(
            "#in-contractor option", "els => els.map(e => e.value)"),
    }
    check("predictor offers the four Indian schemes",
          sorted(options["scheme"]) == SCHEMES, str(options["scheme"]))
    check("predictor offers the four lead states",
          all(name in options["state"] for name in CORE_STATES), str(options["state"]))
    page.select_option("#in-state", "Maharashtra")
    page.wait_for_timeout(400)
    maharashtra = page.eval_on_selector_all(
        "#in-district option", "els => els.map(e => e.value)")
    check("district list cascades to the chosen state",
          "Pune" in maharashtra and "Bengaluru Urban" not in maharashtra, str(maharashtra))
    page.select_option("#in-state", "Karnataka")
    page.wait_for_timeout(400)
    karnataka = page.eval_on_selector_all("#in-district option", "els => els.map(e => e.value)")
    check("cascading districts change with the state",
          "Bengaluru Urban" in karnataka and "Pune" not in karnataka, str(karnataka))
    page.select_option("#in-state", "Maharashtra")
    page.wait_for_timeout(300)
    check("execution types are the Indian Govt / Private pair",
          sorted(options["contractor"]) == ["Govt", "Private"], str(options["contractor"]))
    all_districts = page.evaluate("() => window.App.state.meta.vocabularies.district")
    missing = [want for want in CORE_DISTRICTS
               if not any(district.startswith(want) for district in all_districts)]
    check("dataset vocabulary covers the Indian districts asked for",
          not missing, f"{len(all_districts)} districts; missing {missing}")


def check_performance(page) -> None:
    model_line = text_of(page, "#compare-body")
    check("comparison table shows both models",
          "K-Nearest Neighbours" in model_line and "Random Forest" in model_line)
    check("comparison marks the selected winner", "selected" in model_line)

    cells = page.eval_on_selector_all("#compare-body td", "els => els.map(e => e.textContent.trim())")
    numbers = [c for c in cells if re.match(r"^0\.\d{4}$", c)]
    check("every published metric is a real 4-decimal number",
          len(numbers) == 10, f"{len(numbers)} of {len(cells)} cells")

    accuracy = text_of(page, "#m-accuracy")
    check("selected model accuracy reads as a fraction with its counts",
          bool(re.match(r"^0\.\d{4} \(\d+/\d+\)$", accuracy)), accuracy)
    check("metric panel names the selected model",
          "Random Forest" in text_of(page, "#m-model-name"), text_of(page, "#m-model-name"))
    for element_id in ("m-precision", "m-recall", "m-f1", "m-auc", "m-baseline"):
        value = text_of(page, f"#{element_id}")
        check(f"#{element_id} carries a measured value",
              bool(re.match(r"^(0\.\d{4}|n/a)$", value)), value)
    check("#m-correct states the test split",
          text_of(page, "#m-correct").endswith("test projects"), text_of(page, "#m-correct"))

    rule = text_of(page, "#selection-rule")
    check("selection rule explains the F1 choice and the split it was measured on",
          all(word in rule for word in ("Selection rule", "Winner", "F1", "560", "140")), rule[:200])

    heat_cells = page.evaluate("() => document.querySelectorAll('#confusion-grid .heat-cell').length")
    check("confusion matrix has 4 cells", heat_cells == 4, f"cells={heat_cells}")
    heat_values = page.evaluate(
        """() => Array.from(document.querySelectorAll('#confusion-grid .heat-value'))
                        .map((n) => parseInt(n.textContent, 10))""")
    check("confusion matrix totals the held-out split", sum(heat_values) == 140,
          f"values={heat_values} sum={sum(heat_values)}")

    report_rows = page.evaluate("() => document.querySelectorAll('#report-body tr').length")
    check("classification report has both classes and both averages", report_rows == 4,
          f"rows={report_rows}")

    ablation = page.evaluate("() => document.querySelectorAll('#knn-ablation li').length")
    check("KNN permutation ablation lists the six inputs", ablation >= 7, f"items={ablation}")

    page.click('#performance .tab[data-metric-model="knn"]')
    page.wait_for_timeout(600)
    check("model tab switches the metric panel to KNN",
          "K-Nearest Neighbours · KNN" in text_of(page, "#m-model-name"),
          text_of(page, "#m-model-name"))
    knn_accuracy = text_of(page, "#m-accuracy")
    page.click('#performance .tab[data-metric-model="rf"]')
    page.wait_for_timeout(600)
    check("model tab switches back to Random Forest",
          "Random Forest · RF" in text_of(page, "#m-model-name"), text_of(page, "#m-model-name"))
    check("the two models score differently on the same split",
          knn_accuracy != text_of(page, "#m-accuracy"),
          f"knn {knn_accuracy} vs rf {text_of(page, '#m-accuracy')}")


def check_dataset_explorer(page) -> None:
    rows = page.evaluate("() => document.querySelectorAll('#dataset-body tr').length")
    check("dataset table shows a full page of rows", rows == 25, f"rows={rows}")
    check("page info counts the whole dataset",
          text_of(page, "#page-info") == "Showing 1-25 of 700 projects · page 1 of 28",
          text_of(page, "#page-info"))
    first_row = page.inner_text("#dataset-body tr:first-child")
    check("dataset rows carry Indian scheme geography",
          RUPEE in first_row and any(state in first_row for state in CORE_STATES),
          first_row.replace("\n", " | ")[:160])

    page.fill("#dataset-search", "Pune")
    page.wait_for_timeout(1200)
    narrowed = text_of(page, "#page-info")
    total = page.evaluate("() => window.App.state.dataset.total")
    check("search by district narrows the dataset", 0 < total < 700, narrowed)
    page.fill("#dataset-search", "")
    page.wait_for_timeout(1200)

    page.select_option("#filter-target", "Yes")
    page.wait_for_timeout(1200)
    outcomes = page.evaluate("() => window.App.state.dataset.rows.map((r) => r.over_budget)")
    check("outcome filter keeps only over-sanction rows",
          outcomes and all(o == "Yes" for o in outcomes), str(set(outcomes)))
    check("outcome filter is reflected in the badges",
          "Within sanction" not in page.inner_text("#dataset-body"))
    page.click("#dataset-clear")
    page.wait_for_timeout(1200)
    check("clear restores all 700 projects",
          page.evaluate("() => window.App.state.dataset.total") == 700,
          text_of(page, "#page-info"))

    page.click("#page-next")
    page.wait_for_timeout(1200)
    check("pagination advances", "page 2 of" in text_of(page, "#page-info"),
          text_of(page, "#page-info"))
    check("previous button re-enabled on page 2",
          page.evaluate("() => !document.getElementById('page-prev').disabled"))
    page.click("#page-prev")
    page.wait_for_timeout(1200)
    check("pagination returns to page 1", "page 1 of" in text_of(page, "#page-info"),
          text_of(page, "#page-info"))

    page.click('th[data-sort="sanctioned_budget_lakh"]')
    page.wait_for_timeout(1300)
    ascending = page.evaluate(
        "() => window.App.state.dataset.rows.map((r) => r.sanctioned_budget_lakh)")
    check("sorting by sanctioned cost is ascending",
          page.get_attribute('th[data-sort="sanctioned_budget_lakh"]', "aria-sort") == "ascending"
          and ascending == sorted(ascending), f"{ascending[:4]} ...")
    page.click('th[data-sort="sanctioned_budget_lakh"]')
    page.wait_for_timeout(1300)
    descending = page.evaluate(
        "() => window.App.state.dataset.rows.map((r) => r.sanctioned_budget_lakh)")
    check("a second click reverses the sort",
          page.get_attribute('th[data-sort="sanctioned_budget_lakh"]', "aria-sort") == "descending"
          and descending == sorted(descending, reverse=True), f"{descending[:4]} ...")

    check("sample CSV download points at /sample.csv",
          page.get_attribute("#download-sample", "href").endswith("/sample.csv"),
          page.get_attribute("#download-sample", "href") or "")


def check_predictor(page) -> None:
    page.evaluate(JS_SCROLL_TO, "predictor")
    page.wait_for_timeout(400)

    check("boot made an example prediction", page.evaluate("() => !!window.App.lastPrediction()"))

    page.click("#predict-reset")
    page.wait_for_timeout(500)
    check("reset shows the placeholder again", is_visible(rendered(page, "#predict-placeholder")))
    check("reset hides the result card", not is_visible(rendered(page, "#predict-result")))

    page.click("#predict-example")
    page.wait_for_selector("#predict-result:not([hidden])", timeout=20000)
    page.wait_for_timeout(900)
    check("Use example fills the form from the vocabularies",
          page.evaluate(JS_FIELD, "in-scheme") == "PMGSY (Rural Roads)"
          and page.evaluate(JS_FIELD, "in-state") == "Maharashtra"
          and page.evaluate(JS_FIELD, "in-district") == "Pune"
          and page.evaluate(JS_FIELD, "in-budget") == "250",
          " / ".join(str(page.evaluate(JS_FIELD, name)) for name in
                     ("in-scheme", "in-state", "in-district", "in-budget")))
    check("budget hint restates the amount in rupees",
          text_of(page, "#budget-live-hint").startswith(RUPEE), text_of(page, "#budget-live-hint"))

    card = rendered(page, "#result-card")
    check("result card is visible after submit", is_visible(card))
    check("submit button is re-enabled after a successful prediction",
          page.evaluate("() => !document.getElementById('predict-submit').disabled"),
          page.evaluate("() => document.getElementById('predict-submit').className"))
    verdict = text_of(page, "#result-verdict")
    check("result card shows a verdict", verdict in ("OVER BUDGET", "WITHIN BUDGET"), verdict)
    over = verdict == "OVER BUDGET"
    card_classes = page.get_attribute("#result-card", "class") or ""
    check("result card is coloured for its verdict",
          ("is-over" in card_classes) == over and ("is-within" in card_classes) != over,
          card_classes[:120])

    risk = text_of(page, "#result-risk")
    check("cost-overrun risk % is shown", bool(re.match(r"^\d{1,3}(\.\d)?%$", risk)), risk)
    offset = float(page.get_attribute("#risk-gauge-fill", "stroke-dashoffset") or "0")
    check("risk gauge is drawn to the risk fraction", 0 <= offset <= 326.7, f"dashoffset={offset}")
    expected_offset = 326.7 * (1 - float(risk.rstrip("%")) / 100)
    check("gauge fill matches the printed risk", abs(offset - expected_offset) < 1.0,
          f"{offset} vs {round(expected_offset, 1)}")

    model_line = text_of(page, "#result-model-line")
    check("result line compares both models and names the training size",
          "KNN" in model_line and "Random Forest" in model_line and "560 training projects" in model_line,
          model_line[:200])

    for element_id in ("cost-sanctioned", "cost-expected"):
        value = text_of(page, f"#{element_id}")
        check(f"#{element_id} is a rupee amount", value.startswith(RUPEE), value)
    overrun_value = text_of(page, "#cost-overrun")
    check("#cost-overrun is a rupee amount with a percentage",
          overrun_value.startswith(RUPEE) and "%" in overrun_value, overrun_value)

    explanation = text_of(page, "#result-explanation")
    check("plain-English explanation is present", len(explanation) > 120, f"{len(explanation)} chars")
    check("explanation speaks in rupees, not dollars",
          RUPEE in explanation and "$" not in explanation, explanation[:140])
    advisory = text_of(page, "#specific-advisory")
    check("advisory text is rendered for the chosen execution type", len(advisory) > 40,
          advisory[:120])

    default_k = int(page.evaluate("() => window.App.state.meta.default_k"))
    check("vote line matches the default K",
          text_of(page, "#vote-line").endswith(f"of {default_k}"), text_of(page, "#vote-line"))
    votes = page.evaluate("() => window.App.lastPrediction().votes")
    check("neighbour votes add up to K", votes["yes"] + votes["no"] == votes["total"] == default_k,
          str(votes))

    set_range(page, "in-k", 12)
    check("K slider label follows the slider", text_of(page, "#k-value") == "12",
          text_of(page, "#k-value"))
    submit_predict(page)
    check("changing K re-runs the prediction",
          text_of(page, "#vote-line") == page.evaluate(
              "() => { const v = window.App.lastPrediction().votes; "
              "return v.yes + ' over · ' + v.no + ' within of ' + v.total; }"),
          text_of(page, "#vote-line"))
    vote_line = text_of(page, "#vote-line")
    check("vote line reads as a 12-neighbour split",
          bool(re.match(r"^\d+ over · \d+ within of 12$", vote_line)), vote_line)
    neighbours = page.evaluate("() => document.querySelectorAll('#neighbour-body tr').length")
    check("neighbour table lists K comparable projects", neighbours == 12, f"rows={neighbours}")
    neighbour_row = page.inner_text("#neighbour-body tr:first-child")
    check("comparable projects are Indian works with rupee figures",
          RUPEE in neighbour_row and any(state in neighbour_row for state in CORE_STATES),
          neighbour_row.replace("\n", " | ")[:160])
    distances = page.evaluate(
        "() => window.App.lastPrediction().neighbours.map((n) => n.distance)")
    check("neighbours are ordered by distance", distances == sorted(distances),
          str(distances[:4]))

    page.select_option("#in-model", "knn")
    submit_predict(page)
    knn_line = text_of(page, "#result-model-line")
    check("model switch predicts with K-Nearest Neighbours",
          "K-Nearest Neighbours" in knn_line, knn_line[:120])
    check("KNN explanation describes distance voting",
          "distance" in text_of(page, "#result-explanation").lower(),
          text_of(page, "#result-explanation")[:140])
    page.select_option("#in-model", "auto")
    submit_predict(page)

    # Validation: an empty budget and an impossible duration must be rejected, in the
    # browser, with the server's field-level messages.
    page.evaluate("() => { document.getElementById('in-budget').value = ''; "
                  "document.getElementById('in-duration').value = '400'; }")
    submit_predict(page)
    errors = rendered(page, "#form-errors")
    check("invalid numbers are rejected in the form", is_visible(errors),
          "" if not errors else errors["text"][:120])
    check("invalid fields are marked aria-invalid",
          page.get_attribute("#in-budget", "aria-invalid") == "true"
          and page.get_attribute("#in-duration", "aria-invalid") == "true")
    check("the placeholder is shown when a prediction fails",
          is_visible(rendered(page, "#predict-placeholder")))
    page.click("#predict-example")
    page.wait_for_selector("#predict-result:not([hidden])", timeout=20000)
    page.wait_for_timeout(800)
    check("validation clears once the form is valid again",
          not is_visible(rendered(page, "#form-errors")))
    check("the result card returns after a rejected submission",
          is_visible(rendered(page, "#predict-result")))


def wait_for_corridor(page, start, end=None, timeout=25000):
    """Wait for the planner to finish the corridor the form currently names, and if it
    never arrives, report the state it got stuck on instead of a bare timeout."""
    condition = ("() => { const r = window.Routes.state.last; if (!r) return false;"
                 " return r.start === '%s'%s; }"
                 % (start, "" if end is None else " && r.end === '%s'" % end))
    try:
        page.wait_for_function(condition, timeout=timeout)
    except Exception:
        detail = page.evaluate("""() => ({
            last: window.Routes.state.last
                    ? [window.Routes.state.last.start, window.Routes.state.last.end,
                       window.Routes.state.last.distance_km] : null,
            token: window.Routes.state.requestToken,
            errors: document.getElementById('route-errors').textContent.trim(),
            fields: [...document.querySelectorAll('#route-form .field-error')]
                      .map(n => n.textContent).filter(t => t),
            form: [document.getElementById('rt-start').value,
                   document.getElementById('rt-end').value],
        })""")
        raise AssertionError("planner stuck: %s (wanted %s -> %s)" % (detail, start, end))


def check_routes(page) -> None:
    page.evaluate(JS_SCROLL_TO, "routes")
    page.wait_for_timeout(900)

    check("route planner booted", page.evaluate("() => !!(window.Routes && window.Routes.isReady())"))

    cities = page.eval_on_selector_all(
        "#rt-start option", "els => els.map(e => e.value).filter(v => v)")
    check("start dropdown lists the project cities", len(cities) >= 50, f"{len(cities)} cities")
    for city in ("Nashik", "Pune", "Mumbai City", "New Delhi", "Jaipur"):
        check(f"{city} is selectable", city in cities)
    # The rewrite is India-only, and the dropdowns are drawn from the trained
    # vocabulary, so nothing foreign and no non-district name may appear.
    check("dropdowns are Indian cities only",
          not any(c.lower() in ("bangkok", "london", "kusonkhum") for c in cities),
          ", ".join(cities[:6]))

    terrains = page.eval_on_selector_all("#rt-terrain option", "els => els.map(e => e.value)")
    check("terrain dropdown offers auto plus every class the cost model knows",
          terrains[0] == "" and {"plain", "urban", "ghat", "hilly", "rocky_or_hilly",
                                 "coastal"} <= set(terrains), str(terrains))

    map_state = page.evaluate("""() => ({
        leaflet: typeof window.L !== 'undefined',
        markers: document.querySelectorAll('#india-map path.leaflet-interactive').length,
        tiles: [...document.querySelectorAll('#india-map img.leaflet-tile')]
                 .filter(i => i.complete && i.naturalWidth > 0).length,
        fallbackShown: !document.getElementById('map-fallback').hidden,
        status: document.getElementById('map-status').textContent.trim(),
    })""")
    # Tiles come off public CDN and OSM hosts: an offline machine must show the
    # stated fallback rather than a blank box, and either outcome is acceptable.
    check("map has city markers or says the tiles are unavailable",
          (map_state["leaflet"] and map_state["markers"] >= len(cities)) or map_state["fallbackShown"],
          f"markers={map_state['markers']} tiles={map_state['tiles']} "
          f"fallback={map_state['fallbackShown']} status={map_state['status'][:60]}")

    # Mumbai -> Pune: a known ~149 km corridor that climbs the Bhor ghat.
    page.select_option("#rt-start", "Mumbai City")
    page.select_option("#rt-end", "Pune")
    wait_for_corridor(page, "Mumbai City", "Pune")
    page.wait_for_timeout(900)

    distance = text_of(page, "#rt-out-distance")
    km = float(re.match(r"^([\d.]+)", distance).group(1))
    check("distance is printed in km", bool(re.match(r"^\d+(\.\d)? km$", distance)), distance)
    check("Mumbai-Pune distance is near the published 149 km corridor",
          120 <= km <= 175, f"{km} km (published 149 km, calibrated estimate)")

    total = text_of(page, "#rt-out-total")
    check("budget card is a rupee crore amount",
          total.startswith(RUPEE) and "crore" in total, total)
    perkm = text_of(page, "#rt-out-perkm")
    check("per-km card is a rupee crore rate", perkm.startswith(RUPEE) and "/km" in perkm, perkm)
    time_card = text_of(page, "#rt-out-time")
    check("completion time card shows months and the band it came from",
          bool(re.match(r"^\d+ months? \(\d+(\.\d)?–\d+(\.\d)?\)$", time_card)), time_card)
    risk = text_of(page, "#rt-out-risk")
    check("risk card shows a percentage and a band",
          bool(re.match(r"^\d{1,3}(\.\d)?% (Low|Medium|High)$", risk)), risk)
    terrain_out = text_of(page, "#rt-out-terrain")
    check("terrain impact is a rupee amount", terrain_out.startswith("+ " + RUPEE), terrain_out)

    payload = page.evaluate("() => window.Routes.state.last")
    check("the printed months are the band the API returned",
          time_card.startswith("%s month" % payload["time_months"])
          and payload["construction_time"]["min_months"] <= payload["time_months"]
          <= payload["construction_time"]["max_months"],
          f"{time_card} | band {payload['construction_time']['km_per_day']} km/day")
    check("construction runtime is shorter than the contract programme",
          payload["time_months"] < payload["planned_duration_months"],
          f"{payload['time_months']} months laid vs "
          f"{payload['planned_duration_months']} month programme")
    check("corridor polyline is drawn on the map", page.evaluate(
        """() => [...document.querySelectorAll('#india-map .leaflet-overlay-pane path')]
                 .some(p => (p.getAttribute('stroke-dasharray') || '').length > 0)"""))
    check("the geodesic route path came back from the API",
          payload and len(payload.get("route_path") or []) >= 20,
          f"{len((payload or {}).get('route_path') or [])} points")

    component_sum = round(sum(row["lakh"] for row in payload["components"]), 1)
    check("cost components add up to the works total",
          abs(component_sum - payload["works_lakh"]) <= 1.0,
          f"components {component_sum} vs works {payload['works_lakh']}")
    check("works plus overheads equals the printed budget",
          abs((payload["works_lakh"] + payload["overheads_lakh"]) - payload["total_lakh"]) <= 1.0,
          f"{payload['works_lakh']} + {payload['overheads_lakh']} vs {payload['total_lakh']}")
    check("ghat terrain raises the cost",
          payload["terrain_factor"] >= 1.2 and payload["terrain_impact_lakh"] > 0,
          f"x{payload['terrain_factor']} = {terrain_out}")
    rows = page.evaluate("() => document.querySelectorAll('#rt-component-body tr').length")
    check("component table lists the material, labour and machinery lines", rows >= 8, f"{rows} rows")
    table_text = page.inner_text("#rt-component-body")
    for item in ("Cement", "Steel", "Bitumen", "Labour", "Machinery"):
        check(f"component table includes {item}", item in table_text)
    check("no dollar sign in the route panel",
          "$" not in page.inner_text("#routes") and "us$" not in page.inner_text("#routes").lower())

    # A plain corridor (Delhi -> Jaipur) must show no terrain uplift at all.
    page.select_option("#rt-start", "New Delhi")
    page.select_option("#rt-end", "Jaipur")
    wait_for_corridor(page, "New Delhi", "Jaipur")
    plain = page.evaluate("() => window.Routes.state.last")
    check("plain corridor gets terrain factor 1.00 and zero terrain impact",
          plain["terrain_factor"] == 1.0 and plain["terrain_impact_lakh"] == 0.0,
          f"x{plain['terrain_factor']} impact={plain['terrain_impact_lakh']}")
    check("plain corridor distance is near the published 274 km",
          230 <= plain["distance_km"] <= 310, f"{plain['distance_km']} km")
    plain_months = plain["construction_time"]["likely_months"]
    check("a rural cross-section on the plain runs at 1-2 km/day",
          plain["construction_time"]["band"] == "rural"
          and plain["construction_time"]["km_per_day"] == [1.0, 2.0]
          and plain["time_months"] < plain["planned_duration_months"],
          f"{plain['time_months']} months of running inside a "
          f"{plain['planned_duration_months']}-month programme")

    # The brief's urban class: +15 % on the bill and the same drag on the clock.
    page.select_option("#rt-terrain", "urban")
    page.wait_for_function("() => { const r = window.Routes.state.last;"
                           " return !!r && r.terrain === 'urban'; }", timeout=25000)
    urban = page.evaluate("() => window.Routes.state.last")
    check("choosing urban raises the terrain factor to 1.15",
          urban["terrain_factor"] == 1.15 and urban["terrain_impact_lakh"] > 0,
          f"x{urban['terrain_factor']} impact={urban['terrain_impact_lakh']} lakh")
    check("urban working lengthens the same corridor by about 15 %",
          urban["construction_time"]["terrain_drag"] == 1.15
          and abs(urban["construction_time"]["likely_months"] - plain_months * 1.15) <= 0.1,
          f"{plain_months} -> {urban['construction_time']['likely_months']} months")
    page.select_option("#rt-terrain", "")
    page.wait_for_function("() => { const r = window.Routes.state.last;"
                           " return !!r && r.terrain === 'plain'; }", timeout=25000)

    # The brief's simple formula has to be honoured when the user forces a rate.
    page.select_option("#rt-start", "Nashik")
    page.select_option("#rt-end", "Pune")
    wait_for_corridor(page, "Nashik", "Pune")
    # fill() types into a number box but only blurs on the next focus change, and the
    # form re-estimates on change -- so an explicit submit is what a user would press.
    page.fill("#rt-perkm", "2")
    page.click("#rt-submit")
    page.wait_for_function("() => { const r = window.Routes.state.last;"
                           " return !!r && r.assumed_cost_per_km_crore === 2; }", timeout=25000)
    forced = page.evaluate("() => window.Routes.state.last")
    check("forcing ₹2 crore/km makes the budget distance x rate",
          abs(forced["total_crore"] - forced["distance_km"] * 2.0) < 0.05,
          f"{forced['total_crore']} crore vs {round(forced['distance_km'] * 2.0, 2)}")
    check("the unforced take-off is still reported beside it",
          forced["takeoff_total_crore"] > forced["total_crore"]
          and "takeoff_cost_per_km_crore" in forced,
          f"take-off {forced['takeoff_total_crore']} crore at "
          f"{forced['takeoff_cost_per_km_crore']} crore/km")
    check("the rescaling is explained on screen",
          "rescaled" in text_of(page, "#rt-note").lower(), text_of(page, "#rt-note")[:160])
    page.fill("#rt-perkm", "")
    page.click("#rt-submit")
    page.wait_for_function("() => { const r = window.Routes.state.last;"
                           " return !!r && !r.assumed_cost_per_km_crore; }", timeout=25000)

    # Trained-model cross-check: honest about extrapolating past the data.
    gauge = page.evaluate("""() => ({
        risk: document.getElementById('ml-risk').textContent.trim(),
        offset: document.getElementById('ml-gauge-fill').getAttribute('stroke-dashoffset'),
        note: document.getElementById('ml-note').textContent.trim(),
        expected: document.getElementById('ml-expected').textContent.trim(),
    })""")
    check("model cross-check shows a probability", bool(re.match(r"^\d{1,3}(\.\d)?%$", gauge["risk"])),
          gauge["risk"])
    check("model gauge is filled to its own percentage",
          abs(float(gauge["offset"]) - 326.7 * (1 - float(gauge["risk"].rstrip("%")) / 100)) < 1.5,
          f"dashoffset={gauge['offset']}")
    check("cross-check expected cost is in rupees", gauge["expected"].startswith(RUPEE),
          gauge["expected"])
    check("a sanction above every training project is flagged as extrapolation",
          "extrapolat" in gauge["note"].lower()
          and "above every project" in page.evaluate(
              "() => window.Routes.state.last.model_check.note"),
          gauge["note"][:170])

    # Deliberate bad input: a DPR length outside the accepted range.
    page.evaluate("() => { document.querySelector('details.route-more').open = true; }")
    page.fill("#rt-distance", "9000")
    page.click("#rt-submit")
    page.wait_for_timeout(1500)
    errors_visible = is_visible(rendered(page, "#route-errors"))
    error_text = page.inner_text("#route-errors") if errors_visible else ""
    check("impossible route length is rejected with a field message",
          errors_visible and "length" in error_text.lower(), error_text.replace("\n", " | ")[:150])
    check("the rejected submission hides the stale result card",
          not is_visible(rendered(page, "#route-result")))
    page.fill("#rt-distance", "")
    page.click("#rt-submit")
    page.wait_for_function("""() => { const r = window.Routes.state.last;
        return !!r && !r.assumed_cost_per_km_crore
            && !document.getElementById('route-result').hidden; }""", timeout=25000)
    check("a valid corridor estimates after the rejected one",
          is_visible(rendered(page, "#route-result")), text_of(page, "#rt-out-distance"))
    check("the field error goes away once the length is valid",
          not is_visible(rendered(page, "#route-errors")))
    # Loading animation: park the request, look at the page, then let it go.
    # Sleeping inside the handler would not work -- the sync API runs handlers on
    # the same dispatch loop as wait_for_timeout, so the sleep delays the checks
    # instead of delaying the response and the animation is over before we look.
    parked = []
    page.route("**/predict-route", lambda route: parked.append(route))
    page.click("#rt-submit")
    deadline = time.time() + 25.0
    while time.time() < deadline and not parked:
        page.wait_for_timeout(100)
    if not parked:
        raise AssertionError("planner request was never intercepted")
    check("a skeleton and spinner cover the in-flight estimate",
          is_visible(rendered(page, "#route-loading"))
          and is_visible(rendered(page, "#route-loading .spinner")),
          "skeleton tiles=" + str(page.evaluate(
              "() => document.querySelectorAll('#route-loading .skeleton').length")))
    check("the submit button is disabled while the estimate runs",
          page.evaluate("() => document.getElementById('rt-submit').disabled"))
    check("the stale cards are hidden while the estimate runs",
          not is_visible(rendered(page, "#route-result")))
    parked.pop().continue_()
    page.unroute("**/predict-route")
    page.wait_for_function("() => document.getElementById('route-loading').hidden", timeout=25000)
    check("the skeleton clears and the cards return when the estimate lands",
          is_visible(rendered(page, "#route-result"))
          and not is_visible(rendered(page, "#route-loading")),
          text_of(page, "#rt-out-total"))

    check("swap button reverses the corridor", page.evaluate("""() => {
        const before = [document.getElementById('rt-start').value, document.getElementById('rt-end').value];
        document.getElementById('rt-swap').click();
        const after = [document.getElementById('rt-start').value, document.getElementById('rt-end').value];
        return before[0] === after[1] && before[1] === after[0];
    }"""))
    page.wait_for_timeout(1600)


def option_values(page, selector):
    return page.eval_on_selector_all(selector + " option", "els => els.map(e => e.value)")


def wait_for_point(page, side, timeout=25000, **wanted):
    """Wait until the priced corridor names `wanted` (level, state, district, taluka,
    village) on one end. If it never does, report what the planner is holding instead of
    a bare timeout."""
    tests = " && ".join("p['%s'] === '%s'" % (key, value)
                        for key, value in wanted.items())
    condition = ("() => { const r = window.Routes.state.last;"
                 " if (!r || !r['%(side)s_point']) return false;"
                 " const p = r['%(side)s_point']; return %(tests)s; }"
                 % {"side": side, "tests": tests})
    try:
        page.wait_for_function(condition, timeout=timeout)
    except Exception:
        detail = page.evaluate("""() => ({
            last: window.Routes.state.last || null,
            token: window.Routes.state.requestToken,
            errors: document.getElementById('route-errors').textContent.trim(),
            chosen: ['s', 'e'].map((p) => ['state', 'district', 'taluka', 'village']
                       .map((l) => document.getElementById('rt-' + p + '-' + l).value)),
        })""")
        raise AssertionError("hierarchy planner stuck: %s (wanted %s on the %s end)"
                             % (detail, wanted, side))


def check_planning(page) -> None:
    """State -> District -> Taluka -> Village on both ends, and the planning-formula
    panel that prices whichever corridor the four dropdowns name."""
    page.evaluate(JS_SCROLL_TO, "routes")
    page.wait_for_timeout(700)

    page.select_option("#rt-mode", "hierarchy")
    wait_for_point(page, "start", level="district")
    page.wait_for_timeout(700)

    check("location mode shows the cascades and hides the city boxes",
          is_visible(rendered(page, "#rt-hierarchy"))
          and not is_visible(rendered(page, "#rt-cities")))
    check("the hidden city selects are not left required",
          not page.evaluate("() => document.getElementById('rt-start').required"))

    states = option_values(page, "#rt-s-state")
    check("state dropdown lists the catalogue states", len(states) >= 5, str(states[:6]))
    seeded = page.evaluate("() => [window.Routes.state.last.start_point,"
                           " window.Routes.state.last.end_point]")
    check("the corridor opens on two real Maharashtra districts",
          [seeded[0]["district"], seeded[1]["district"]] == ["Pune", "Nashik"]
          and seeded[0]["level"] == "district" == seeded[1]["level"],
          "%s -> %s" % (seeded[0]["label"], seeded[1]["label"]))
    readout = text_of(page, "#rt-loc-readout")
    check("the readout says a district alone means its headquarters",
          "headquarters" in readout, readout[:170])
    # Survives the mode switch, which rewrites the line above it: this is the only place
    # the generated village names are labelled as generated on this panel.
    source = text_of(page, "#rt-loc-source")
    check("the hierarchy discloses which names are generated",
          "placeholder" in source and "504" in source, source[:200])

    # The brief's fourth level: a village inside a taluka inside a district.
    talukas = [v for v in option_values(page, "#rt-s-taluka") if v]
    check("choosing a district fills its talukas", len(talukas) >= 3, str(talukas[:4]))
    page.select_option("#rt-s-taluka", talukas[0])
    villages = [v for v in option_values(page, "#rt-s-village") if v]
    check("choosing a taluka fills its villages", len(villages) >= 3, str(villages[:4]))
    page.select_option("#rt-s-village", villages[0])
    wait_for_point(page, "start", level="village", taluka=talukas[0],
                   village=villages[0])
    page.wait_for_timeout(600)
    village_point = page.evaluate("() => window.Routes.state.last.start_point")
    check("a village-to-district corridor is priced from the two coordinates",
          village_point["connector_km"] > 0
          and village_point["connector_budget_lakh"] > 0,
          "%s: connector %.1f km" % (village_point["village"],
                                     village_point["connector_km"]))
    readout = text_of(page, "#rt-loc-readout")
    check("the readout names the village, its taluka and its connector cost",
          villages[0] in readout and talukas[0] in readout and RUPEE in readout
          and "connector" in readout.lower(), readout[:220])
    # The brief asks the map popup to carry the place, its cost and its time.
    check("the endpoint popup carries the place, its connector and the corridor",
          page.evaluate("""() => {
              const markers = window.Routes.state.pointMarkers || [];
              if (!markers.length) return false;
              return markers.every((m) => {
                  const html = String((m.getPopup() || {}).getContent
                                       ? m.getPopup().getContent() : '');
                  return html.indexOf('Selected corridor') >= 0
                      && (html.indexOf('₹') >= 0 || html.indexOf('level') >= 0);
              });
          }"""),
          "%d endpoint markers" % page.evaluate(
              "() => (window.Routes.state.pointMarkers || []).length"))

    # Changing a level must invalidate everything below it, or a stale village name
    # would price a point the reader did not choose.
    page.select_option("#rt-s-state", "Karnataka")
    page.wait_for_timeout(500)
    karnataka_districts = option_values(page, "#rt-s-district")
    maharashtra_districts = page.evaluate(
        "() => window.Routes.state.hierarchy.states"
              ".filter((s) => s.name === 'Maharashtra')[0].districts"
              ".map((d) => d.name)")
    check("changing the state refills the districts and clears the lower levels",
          "Bengaluru Urban" in karnataka_districts
          and not set(karnataka_districts) & set(maharashtra_districts)
          and page.input_value("#rt-s-taluka") == ""
          and page.input_value("#rt-s-village") == "",
          "%d options, taluka=%r" % (len(karnataka_districts),
                                     page.input_value("#rt-s-taluka")))
    page.select_option("#rt-s-state", "Maharashtra")
    page.wait_for_timeout(400)
    page.select_option("#rt-s-district", "Pune")
    wait_for_point(page, "start", level="district", district="Pune")

    # Search-as-you-type resolves to the same four dropdowns, not to a parallel path.
    page.fill("#rt-search", "Pimpri")
    page.wait_for_selector("#rt-search-results .search-hit", timeout=15000)
    hits = page.eval_on_selector_all("#rt-search-results .search-hit",
                                     "els => els.map(e => e.innerText)")
    check("the search bar suggests villages with their distance and cost",
          len(hits) >= 1 and RUPEE in hits[0] and "km" in hits[0], hits[0][:120])
    page.select_option("#rt-search-side", "end")
    page.click("#rt-search-results .search-hit")
    page.wait_for_timeout(700)
    check("a search hit fills the end cascade it was aimed at",
          page.input_value("#rt-e-village") != ""
          and page.input_value("#rt-e-state") != "",
          "%s > %s > %s" % (page.input_value("#rt-e-state"),
                            page.input_value("#rt-e-district"),
                            page.input_value("#rt-e-village")))
    wait_for_point(page, "end", level="village",
                   village=page.input_value("#rt-e-village"))
    hit_point = page.evaluate("() => window.Routes.state.last.end_point")
    check("the searched village is the one the server resolved",
          hit_point["village"] == page.input_value("#rt-e-village")
          and hit_point["taluka"] == page.input_value("#rt-e-taluka"),
          "%s" % hit_point["label"])
    check("each end is priced at the level its dropdowns reached",
          page.evaluate("() => window.Routes.state.last.location_levels")
          == ["district", "village"],
          str(page.evaluate("() => window.Routes.state.last.location_levels")))

    # The planning-formula panel: the server's arithmetic, laid out and added up.
    plan = page.evaluate("() => window.Routes.state.last.planning")
    check("the planning panel carries the formula view of the same corridor",
          plan and plan["distance_km"] > 0, "" if not plan else
          "class=%s base=%.2f/km" % (plan["rate_class"], plan["base_cost_per_km_crore"]))
    check("base rate card names the class rate",
          text_of(page, "#pf-base").startswith(RUPEE)
          and ("%.2f" % plan["base_cost_per_km_crore"]) in text_of(page, "#pf-base"),
          text_of(page, "#pf-base"))
    check("effective per-km card includes the terrain and soil premia",
          plan["cost_per_km_crore"] >= plan["base_cost_per_km_crore"]
          and text_of(page, "#pf-perkm").startswith(RUPEE)
          and ("%.2f" % plan["cost_per_km_crore"]) in text_of(page, "#pf-perkm"),
          "%s vs base %.4f" % (text_of(page, "#pf-perkm"),
                               plan["base_cost_per_km_crore"]))
    check("formula budget card is a rupee crore amount",
          text_of(page, "#pf-total").startswith(RUPEE) and "crore" in text_of(page, "#pf-total")
          and ("%.2f" % plan["total_crore"]) in text_of(page, "#pf-total"),
          text_of(page, "#pf-total"))
    check("risk card is the size band the brief sets",
          plan["risk"]["band"] in text_of(page, "#pf-risk"), text_of(page, "#pf-risk"))
    verdict = text_of(page, "#pf-verdict")
    check("the panel prints the four multipliers it used",
          all(token in verdict for token in ("base", "width", "terrain", "soil")), verdict)
    basis = text_of(page, "#pf-basis")
    check("the basis line prints the whole multiplication, factors included",
          basis.startswith("%.1f km" % plan["distance_km"])
          and " x " in basis and RUPEE in basis
          and ("%.2f terrain" % plan["terrain_factor"]) in basis
          and ("%.2f soil" % plan["soil_factor"]) in basis, basis[:200])
    rows = page.eval_on_selector_all("#pf-material-body tr", "els => els.map(e => e.innerText)")
    check("material table lists the five lines and their total", len(rows) == 6,
          "%d rows" % len(rows))
    labels = [row.split("\n")[0] for row in rows]
    check("materials stay in the order the brief states them",
          [row["key"] for row in plan["materials"]] == ["cement", "steel", "bitumen",
                                                        "labor", "machinery"],
          str([row["key"] for row in plan["materials"]]))
    check("the five shares are 25 / 20 / 15 / 20 / 20",
          [row["percent"] for row in plan["materials"]] == [25.0, 20.0, 15.0, 20.0, 20.0],
          str([row["percent"] for row in plan["materials"]]))
    split = round(sum(row["lakh"] for row in plan["materials"]), 2)
    check("the five material lines add back to the formula budget",
          abs(split - plan["total_lakh"]) <= 0.02,
          "%.2f lakh vs %.2f" % (split, plan["total_lakh"]))
    check("the last row totals the five lines the table just printed",
          "Total" in rows[-1] and "100 %" in rows[-1] and rows[-1].count(RUPEE) == 1,
          rows[-1].replace("\n", " | ")[:110])
    check("the planning time line states the km/day speed band",
          "km" in text_of(page, "#pf-time") and "month" in text_of(page, "#pf-time").lower(),
          text_of(page, "#pf-time")[:170])
    takeoff = page.evaluate("() => window.Routes.state.last.total_crore")
    check("the two views of one road stay visibly different",
          "different measurement" in page.inner_text("#routes").lower()
          and abs(takeoff - plan["total_crore"]) > 0.01,
          "take-off %.2f vs formula %.2f crore" % (takeoff, plan["total_crore"]))

    charts = {entry["id"]: entry for entry in page.evaluate(JS_CHART_REPORT)}
    pie = charts.get("chart-planning-split") or {}
    check("planning material pie is drawn from the five server lines",
          pie.get("type") in ("doughnut", "pie") and len(pie.get("labels") or []) == 5
          and pie.get("points") == 5,
          "type=%s labels=%s" % (pie.get("type"), pie.get("labels")))

    # Same point at both ends: the engine refuses it and the form says why. The copy
    # walks down the four levels with real change events, because a cascade that never
    # fired would leave the lower dropdowns holding another state's names.
    page.evaluate("""() => {
        ['state', 'district', 'taluka', 'village'].forEach((level) => {
          const from = document.getElementById('rt-s-' + level);
          const to = document.getElementById('rt-e-' + level);
          to.value = from.value;
          to.dispatchEvent(new Event('change', { bubbles: true }));
        });
    }""")
    page.click("#rt-submit")
    page.wait_for_timeout(1600)
    error_text = page.inner_text("#route-errors") if is_visible(
        rendered(page, "#route-errors")) else ""
    check("a corridor that starts and ends at the same village is refused",
          "same" in error_text.lower(), error_text.replace("\n", " | ")[:150])
    check("the rejection clears the result card",
          not is_visible(rendered(page, "#route-result")))

    page.select_option("#rt-e-district", "Nashik")
    page.wait_for_timeout(400)
    page.click("#rt-submit")
    page.wait_for_function("""() => { const r = window.Routes.state.last;
        return !!r && r.end_point && r.end_point.district === 'Nashik'
            && !document.getElementById('route-result').hidden; }""", timeout=25000)
    check("a valid corridor prices again after the refusal",
          is_visible(rendered(page, "#route-result")), text_of(page, "#rt-out-distance"))

    # And back to cities, so the panel the rest of the run assumes is really restored.
    page.select_option("#rt-mode", "cities")
    page.wait_for_timeout(2200)
    check("switching back restores the city dropdowns and their corridor",
          is_visible(rendered(page, "#rt-cities"))
          and not is_visible(rendered(page, "#rt-hierarchy"))
          and not page.evaluate("() => window.Routes.state.last.start_point"))
    check("the planning panel still prices a city corridor",
          page.evaluate("() => !!(window.Routes.state.last.planning"
                        " && window.Routes.state.last.planning.materials.length === 5)"))


BUILDING_SECTIONS = ["home", "building", "rates", "limits", "team", "footer"]


def wait_for_building(page, condition, wanted, timeout=25000) -> None:
    """Wait for the building card to show the estimate the form just asked for, and if it
    never arrives, say what the page is holding."""
    try:
        page.wait_for_function("() => { const r = window.Building.last();"
                               " return !!r && (%s); }" % condition, timeout=timeout)
    except Exception:
        detail = page.evaluate("""() => ({
            last: window.Building.last(),
            errors: document.getElementById('building-errors').textContent.trim(),
            fields: [...document.querySelectorAll('#building-form .field-error')]
                      .map((n) => n.textContent).filter((t) => t),
            form: ['bd-type', 'bd-area', 'bd-floors', 'bd-quality', 'bd-location-type']
                    .map((id) => document.getElementById(id).value),
        })""")
        raise AssertionError("building estimate stuck: %s (wanted %s)" % (detail, wanted))


def number_in(text: str) -> float:
    """The first Indian-grouped number in a string, as a float: '₹ 1,350' -> 1350.0.
    The cards print with `toLocaleString('en-IN')`, so a plain float() never matches."""
    match = re.search(r"[\d,]+(?:\.\d+)?", text.replace(" ", " "))
    return float(match.group(0).replace(",", "")) if match else float("nan")


def check_building_page(base_url, context, attach) -> None:
    """/building -- the second page, the second engine, the same project and rules."""
    page = context.new_page()
    attach(page)
    page.goto(base_url + "/building", wait_until="domcontentloaded", timeout=90000)
    page.wait_for_function("""() => typeof Chart !== 'undefined'
                                   && !!window.Building && window.Building.state.ready
                                   && !!window.Building.last()""", timeout=90000)
    page.wait_for_timeout(1200)
    for section_id in BUILDING_SECTIONS:
        page.evaluate(JS_SCROLL_TO, section_id)
        page.wait_for_timeout(300)
    page.evaluate("() => window.scrollTo(0, 0)")
    page.wait_for_timeout(1200)

    check("the building page booted without an error banner",
          not is_visible(rendered(page, "#api-banner")))
    for section_id in BUILDING_SECTIONS:
        state = rendered(page, "#%s" % section_id)
        check("building section #%s present and visible" % section_id, is_visible(state),
              "" if not state else "display=%s h=%s" % (state["display"], state["height"]))
    # Brief B asked for a separate system in the same project: it must not pull in the
    # dashboard controller, and it must still hand the reader back to the corridor.
    check("the building page runs its own controller, not the dashboard's",
          page.evaluate("() => typeof window.Building !== 'undefined'"
                        " && typeof window.App === 'undefined'"))
    nav = page.eval_on_selector_all("#navbar .nav-link, nav .nav-link",
                                    "els => els.map(e => e.getAttribute('href'))")
    check("the building page nav keeps the main page links",
          any((href or "").endswith("#routes") for href in nav)
          and any((href or "").endswith("/building") for href in nav), str(nav))
    hero = page.inner_text("#home")
    for name in DEVELOPERS:
        check("building hero credits %s" % name, name in hero)
    check("building footer carries the guide and DBATU",
          GUIDE in page.inner_text("#footer") and "DBATU" in page.inner_text("#footer"))
    body_text = page.evaluate("() => document.body.innerText")
    check("rupees throughout the building page", body_text.count(RUPEE) >= 12,
          "%d occurrences" % body_text.count(RUPEE))
    check("no dollar sign on the building page", "$" not in body_text,
          "; ".join(line.strip()[:60] for line in body_text.splitlines()
                    if "$" in line)[:160])

    result = page.evaluate("() => window.Building.last()")
    card = text_of(page, "#bd-out-cost")
    check("estimated cost card is a rupee lakh amount",
          bool(re.match(r"^₹ \d[\d.]* lakh$", card))
          and abs(number_in(card) - result["cost_lakh"]) < 0.005, card)
    check("time card is months", bool(re.match(r"^\d+ months$", text_of(page, "#bd-out-time"))),
          text_of(page, "#bd-out-time"))
    check("effective rate card is rupees per square foot",
          text_of(page, "#bd-out-rate").startswith(RUPEE)
          and abs(number_in(text_of(page, "#bd-out-rate"))
                  - result["effective_cost_per_sqft"]) < 0.01,
          text_of(page, "#bd-out-rate"))
    check("built-up area card follows the storeys, not the footprint",
          "sq ft" in text_of(page, "#bd-out-built")
          and abs(number_in(text_of(page, "#bd-out-built"))
                  - result["built_up_sqft"]) < 0.6,
          "%s (footprint %.0f)" % (text_of(page, "#bd-out-built"), result["area_sqft"]))
    check("crore card is the same money as the lakh card",
          abs(float(re.search(r"([\d.]+)", text_of(page, "#bd-out-crore")).group(1))
              - result["cost_lakh"] / 100.0) < 0.01,
          "%s vs %s" % (text_of(page, "#bd-out-crore"), card))

    basis = text_of(page, "#bd-basis")
    check("the basis line prints every factor the card multiplied",
          "sq ft" in basis and RUPEE in basis
          and ("%.2f floors" % result["floor_factor"]) in basis
          and ("%.2f quality" % result["quality_factor"]) in basis
          and ("%.2f building type" % result["building_type_factor"]) in basis, basis[:220])
    chips = page.eval_on_selector_all("#bd-factors span", "els => els.map(e => e.innerText)")
    check("one chip per factor, so nothing is applied silently",
          len(chips) == 5 and any("×%.2f" % result["location_factor"] in chip
                                 for chip in chips), " | ".join(chips)[:200])

    rows = page.eval_on_selector_all(
        "#bd-split-body tr", "els => els.map(e => [...e.children].map((td) => td.innerText))")
    check("seven work sections and a total line", len(rows) == 8, "%d rows" % len(rows))
    percent = sum(float(row[1].replace("%", "").strip()) for row in rows[:-1])
    check("the work split adds to 100 %", abs(percent - 100.0) < 0.01, "%.2f %%" % percent)
    lakh = [float(row[2]) for row in rows[:-1]]
    card_lakh = float(re.match(r"^₹ ([\d,.]+) lakh$", card).group(1).replace(",", ""))
    check("the seven printed lines add up to the cost card",
          abs(sum(lakh) - card_lakh) <= 0.005, "%.2f lakh vs %s" % (sum(lakh), card))
    check("the total row agrees with the card above it",
          abs(float(rows[-1][2]) - card_lakh) <= 0.005
          and "100 %" in rows[-1][1] and rows[-1][0].startswith("Total"),
          "%s vs %s" % (rows[-1][2], card))
    rupees = sum(float(row[3].replace(",", "")) for row in rows[:-1])
    check("the rupee column is the same money as the lakh column",
          abs(rupees - result["cost_rupees"]) <= 7, "%.0f vs %d" % (rupees,
                                                                    result["cost_rupees"]))
    order = page.evaluate("""() => JSON.parse(
        document.getElementById('building-data').textContent)
        .work_split.map((row) => row.label)""")
    check("sections print in cost order, not alphabetically",
          order == [row[0] for row in rows[:-1]], str(order[:3]))

    time_line = text_of(page, "#bd-time-basis")
    check("the time line states the sq ft per day rate and the fixed site days",
          "sq ft per day" in time_line and "set-up" in time_line
          and str(int(result["construction_time"]["sqft_per_day"])) in time_line,
          time_line[:200])
    note = text_of(page, "#bd-note")
    check("the page says this is an estimate, not a trained model",
          "not a trained-model output" in note and "no land" in note.lower(), note[:180])

    canvas = page.evaluate(JS_CANVAS_REPORT, [BUILDING_CANVAS])[0]
    check("work-split canvas has painted pixels",
          canvas.get("exists") and canvas["painted"] > 0
          and canvas["bufferWidth"] > 0,
          "buffer=%sx%s painted=%s" % (canvas.get("bufferWidth"),
                                       canvas.get("bufferHeight"), canvas.get("painted")))
    pie = {entry["id"]: entry
           for entry in page.evaluate(JS_CHART_REPORT)}.get(BUILDING_CANVAS) or {}
    check("work-split pie is registered with the seven sections",
          pie.get("type") in ("doughnut", "pie") and len(pie.get("labels") or []) == 7,
          "type=%s labels=%s" % (pie.get("type"), pie.get("labels")))

    rates = {sel: page.evaluate("() => document.querySelectorAll('#%s tr').length" % sel)
             for sel in ("rate-locations", "rate-floors", "rate-qualities", "rate-types",
                         "rate-split")}
    check("the assumption tables print the same tables the server uses",
          rates["rate-locations"] == 3 and rates["rate-floors"] == 3
          and rates["rate-qualities"] == 3 and rates["rate-types"] == 3
          and rates["rate-split"] == 7, str(rates))

    # The four-level cascade, reusing the road project's own location catalogue.
    check("the site cascade opens on a real district",
          page.input_value("#bd-state") == "Maharashtra"
          and page.input_value("#bd-district") == "Pune",
          "%s > %s" % (page.input_value("#bd-state"), page.input_value("#bd-district")))
    site_line = text_of(page, "#bd-site-line")
    check("the site line discloses the generated village names",
          "placeholder" in site_line and "real" in site_line, site_line[:200])
    talukas = [v for v in option_values(page, "#bd-taluka") if v]
    page.select_option("#bd-taluka", talukas[0])
    wait_for_building(page, "true", "a corridor after a taluka change")
    villages = [v for v in option_values(page, "#bd-village") if v]
    check("choosing a taluka fills its villages", len(villages) >= 3, str(villages[:4]))
    page.select_option("#bd-village", villages[0])
    wait_for_building(page, "true", "a corridor after a village change")
    check("the site reads down to the village and back in the result card",
          villages[0] in text_of(page, "#bd-site-line")
          and villages[0] in text_of(page, "#bd-site-result"),
          text_of(page, "#bd-site-result")[:160])
    check("naming a site does not change the price",
          abs(page.evaluate("() => window.Building.last().cost_lakh")
              - result["cost_lakh"]) < 0.005,
          "%.2f vs %.2f lakh" % (page.evaluate("() => window.Building.last().cost_lakh"),
                                 result["cost_lakh"]))

    # A preset and a factor change, both through the form.
    page.click("#bd-examples button:nth-child(2)")
    wait_for_building(page, "r.building_type === 'school' && r.area_sqft === 4000",
                      "the school preset")
    school = page.evaluate("() => window.Building.last()")
    check("a preset fills the form and reprices it",
          page.input_value("#bd-area") == "4000"
          and page.input_value("#bd-location-type") == "urban"
          and school["base_cost_per_sqft"] == 2500.0,
          "%s at %s / sq ft" % (text_of(page, "#bd-out-cost"),
                                text_of(page, "#bd-out-base")))
    page.select_option("#bd-type", "hospital")
    wait_for_building(page, "r.building_type === 'hospital'", "the hospital type")
    hospital = page.evaluate("() => window.Building.last()")
    check("a health centre costs more per square foot than a classroom block",
          hospital["cost_lakh"] > school["cost_lakh"]
          and hospital["building_type_factor"] == 1.35,
          "%.2f vs %.2f lakh" % (hospital["cost_lakh"], school["cost_lakh"]))
    page.select_option("#bd-floors", "3")
    wait_for_building(page, "r.floors === 3", "three storeys")
    three = page.evaluate("() => window.Building.last()")
    check("three storeys cost 2.65 times one, not three times",
          three["floor_factor"] == 2.65
          and abs(three["built_up_sqft"] - 4000 * 2.65) < 1.0
          and abs(number_in(text_of(page, "#bd-out-built")) - 4000 * 2.65) < 1.0,
          "x%.2f -> %s sq ft" % (three["floor_factor"], text_of(page, "#bd-out-built")))

    # Deliberate bad input: a footprint below the range the engine accepts.
    page.fill("#bd-area", "5")
    page.click("#bd-submit")
    page.wait_for_timeout(1500)
    error_text = page.inner_text("#building-errors") if is_visible(
        rendered(page, "#building-errors")) else ""
    slot = page.evaluate("() => document.querySelector('#building-form"
                         " [data-error-for=\"area_sqft\"]').textContent.trim()")
    check("an impossible footprint is rejected and names the field",
          "area" in error_text.lower() and "sq ft" in slot,
          "%s | %s" % (error_text.replace("\n", " ")[:110], slot[:90]))
    check("the rejected estimate hides the stale card",
          not is_visible(rendered(page, "#building-result")))
    page.fill("#bd-area", "1200")
    page.click("#bd-submit")
    wait_for_building(page, "r.area_sqft === 1200", "a valid footprint again")
    check("a valid estimate returns after the rejection",
          is_visible(rendered(page, "#building-result")), text_of(page, "#bd-out-cost"))

    # Loading state, parked the same way the corridor test parks its request.
    parked = []
    page.route("**/predict-building", lambda route: parked.append(route))
    page.select_option("#bd-quality", "low")
    deadline = time.time() + 25.0
    while time.time() < deadline and not parked:
        page.wait_for_timeout(100)
    if not parked:
        raise AssertionError("building request was never intercepted")
    check("a skeleton and spinner cover the in-flight estimate",
          is_visible(rendered(page, "#building-loading"))
          and is_visible(rendered(page, "#building-loading .spinner")),
          "skeleton tiles=" + str(page.evaluate(
              "() => document.querySelectorAll('#building-loading .skeleton').length")))
    check("the estimate button is disabled while it runs",
          page.evaluate("() => document.getElementById('bd-submit').disabled"))
    parked.pop().continue_()
    page.unroute("**/predict-building")
    wait_for_building(page, "r.quality === 'low'", "the low-quality estimate")
    check("the card returns when the estimate lands",
          is_visible(rendered(page, "#building-result")), text_of(page, "#bd-out-cost"))

    report = page.evaluate(JS_OVERFLOW)
    check("no horizontal overflow on the building page at 1280px",
          report["htmlScrollWidth"] <= report["htmlClientWidth"] + 1
          and report["bodyScrollWidth"] <= report["bodyClientWidth"] + 1,
          "html %s/%s offenders=%s" % (report["htmlScrollWidth"],
                                       report["htmlClientWidth"], report["offenderCount"]))
    page.evaluate(JS_CLEAR_TOASTS)
    page.screenshot(path=os.path.join(SCREENSHOT_DIR, "30-building-desktop.png"),
                    full_page=True)
    page.set_viewport_size({"width": 375, "height": 812})
    page.wait_for_timeout(700)
    mobile = page.evaluate(JS_OVERFLOW)
    check("no horizontal overflow on the building page at 375px",
          mobile["htmlScrollWidth"] <= mobile["htmlClientWidth"] + 1,
          "html %s/%s offenders=%s" % (mobile["htmlScrollWidth"],
                                       mobile["htmlClientWidth"],
                                       mobile["offenderCount"]))
    page.evaluate(JS_CLEAR_TOASTS)
    page.screenshot(path=os.path.join(SCREENSHOT_DIR, "31-building-mobile.png"),
                    full_page=True)
    page.close()


def check_exports(page, tmp_dir: str) -> None:
    try:
        with page.expect_download(timeout=25000) as download_info:
            page.click("#export-pdf")
        download = download_info.value
        pdf_path = os.path.join(tmp_dir, download.suggested_filename)
        download.save_as(pdf_path)
        size = os.path.getsize(pdf_path)
        check("PDF export downloads", download.suggested_filename.endswith(".pdf") and size > 2000,
              f"{download.suggested_filename} {size} bytes")
        with open(pdf_path, "rb") as handle:
            content = handle.read().decode("latin-1", errors="ignore")
        check("PDF starts with the %PDF header", content.startswith("%PDF"))
        check("PDF contains the team credits",
              DEVELOPERS[0] in content and DEVELOPERS[1] in content and GUIDE in content)
        check("PDF writes rupees as INR text (Helvetica has no rupee glyph)",
              "INR" in content and RUPEE not in content)
        check("PDF is about India FY 2025-26", "2025-26" in content or "India" in content)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the run
        check("PDF export downloads", False, f"{type(exc).__name__}: {exc}")

    try:
        with page.expect_download(timeout=25000) as download_info:
            page.click("#export-csv")
        download = download_info.value
        csv_path = os.path.join(tmp_dir, download.suggested_filename)
        download.save_as(csv_path)
        with open(csv_path, encoding="utf-8-sig") as handle:
            content = handle.read()
        check("CSV export downloads", download.suggested_filename.endswith(".csv")
              and len(content) > 500, f"{len(content)} chars")
        check("CSV contains the team credits",
              DEVELOPERS[0] in content and DEVELOPERS[1] in content and GUIDE in content)
        check("CSV contains the verdict and the risk percent",
              "Verdict" in content and "Risk percent" in content)
        check("CSV keeps the rupee sign", RUPEE in content and "$" not in content)
        check("CSV lists the comparable projects",
              content.count("PMGSY") + content.count("PMAY") + content.count("Jal Jeevan")
              + content.count("Smart Cities") > 0)
    except Exception as exc:  # noqa: BLE001
        check("CSV export downloads", False, f"{type(exc).__name__}: {exc}")


def check_dashboard_stream(page) -> None:
    page.evaluate(JS_SCROLL_TO, "dashboard")
    page.wait_for_timeout(600)

    for element_id in ("kpi-sanctioned", "kpi-actual", "kpi-overrun"):
        value = text_of(page, f"#{element_id}")
        check(f"#{element_id} is a rupee total", value.startswith(RUPEE), value)
    share = text_of(page, "#kpi-share")
    check("#kpi-share is a percentage of projects",
          share.endswith("% of projects") and "%" in share, share)
    served = text_of(page, "#kpi-predictions")
    check("#kpi-predictions counts served predictions", served.isdigit() and int(served) >= 1,
          served)

    worst_rows = page.evaluate("() => document.querySelectorAll('#top-overruns tr').length")
    check("dashboard shows the ten worst overruns", worst_rows == 10, f"rows={worst_rows}")
    worst = page.inner_text("#top-overruns tr:first-child")
    check("worst-overrun rows carry work codes and rupees",
          RUPEE in worst and "+" in worst, worst.replace("\n", " | ")[:150])

    pipeline = page.inner_text("#pipeline-status")
    check("feature-store manifest is shown",
          "scheme_name" in pipeline and "700" in pipeline and "partitions" in pipeline.lower(),
          pipeline.replace("\n", " | ")[:200])
    check("manifest reports partitions per scheme",
          all(scheme in pipeline for scheme in ["PMGSY", "PMAY", "Jal Jeevan", "Smart Cities"]),
          pipeline.replace("\n", " | ")[-180:])

    source_line = text_of(page, "#source-line")
    check("active data source is reported",
          "700" in source_line and "local" in source_line, source_line)
    check("store availability explains http and s3",
          "http" in page.inner_text("#source-availability").lower()
          and "s3" in page.inner_text("#source-availability").lower(),
          page.inner_text("#source-availability").replace("\n", " | ")[:180])

    status = text_of(page, "#stream-status")
    check("SSE stream is connected", "connected" in status.lower(), status)

    def feed_sequences():
        return page.eval_on_selector_all(
            "#live-feed tr", "els => els.map(e => Number(e.dataset.sequence || 0))")

    rows_before = page.evaluate("() => window.Dashboard.feedRows()")
    sequences_before = page.evaluate("() => window.Dashboard.seenSequences.size")
    newest_before = (feed_sequences() or [0])[0]
    for budget in ("90", "4500", "1200"):
        page.evaluate("(value) => { document.getElementById('in-budget').value = value; }", budget)
        submit_predict(page, 1400)
    page.wait_for_timeout(1500)
    rows_after = page.evaluate("() => window.Dashboard.feedRows()")
    sequences_after = page.evaluate("() => window.Dashboard.seenSequences.size")
    newest_after = (feed_sequences() or [0])[0]
    # The feed is a rolling window of the last twelve events, so by this point in the
    # run it is already full: growth is proven by the sequence numbers, not by the row
    # count. (Route-planner cross-checks are real predictions and push rows too.)
    check("live feed advances with each new prediction",
          sequences_after >= sequences_before + 3 and newest_after > newest_before
          and rows_after <= 12 and rows_after >= 3,
          f"rows {rows_before} -> {rows_after}, sequences "
          f"{sequences_before} -> {sequences_after}, newest {newest_before} -> {newest_after}")
    check("live feed renders every event exactly once",
          len(feed_sequences()) == len(set(feed_sequences())),
          f"{len(feed_sequences())} rows")
    feed_sample = page.inner_text("#live-feed tr:first-child")
    check("live feed rows show scheme, district and risk %",
          "%" in feed_sample and any(scheme.split(" ")[0] in feed_sample
                                     for scheme in SCHEMES), feed_sample.replace("\n", " | ")[:140])
    check("served-prediction counter followed the stream",
          int(text_of(page, "#kpi-predictions")) >= sequences_after,
          text_of(page, "#kpi-predictions"))

    page.evaluate("() => { document.getElementById('in-budget').value = '250'; }")
    submit_predict(page, 1200)


def check_knn_demo(page) -> None:
    page.evaluate(JS_CENTER_INSTANTLY, "knn-canvas")
    page.wait_for_timeout(600)

    readout_before = text_of(page, "#knn-readout")
    check("KNN demo shows a vote", "of 5 neighbours" in readout_before or "K=5" in readout_before,
          readout_before[:150])

    box = page.locator("#knn-canvas").bounding_box()
    if box and 0 <= box["y"] and box["y"] + box["height"] <= 900:
        page.mouse.move(box["x"] + box["width"] * 0.30, box["y"] + box["height"] * 0.25)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] * 0.78, box["y"] + box["height"] * 0.80, steps=12)
        page.mouse.up()
        page.wait_for_timeout(500)
        readout_after = text_of(page, "#knn-readout")
        check("dragging the test point changes its vote", readout_after != readout_before,
              f"'{readout_before[:60]}' -> '{readout_after[:60]}'")
        check("drag cursor class clears after release",
              "dragging" not in (page.get_attribute("#knn-canvas", "class") or ""))
    else:
        check("KNN canvas is on screen for the drag test", False, f"bounding_box()={box}")

    set_range(page, "knn-k", 9)
    check("demo K slider updates its label", text_of(page, "#knn-k-value") == "9",
          text_of(page, "#knn-k-value"))
    readout_k9 = text_of(page, "#knn-readout")
    check("demo vote reflects the new K", "of 9 neighbours" in readout_k9, readout_k9[:150])

    page.click("#knn-random")
    page.wait_for_timeout(500)
    check("new sample redraws the demo",
          "neighbours" in text_of(page, "#knn-readout") or "Tie" in text_of(page, "#knn-readout"),
          text_of(page, "#knn-readout")[:150])
    canvas = page.evaluate(JS_CANVAS_REPORT, ["knn-canvas"])[0]
    check("redrawn demo canvas still has pixels", canvas.get("painted", 0) > 0,
          f"painted={canvas.get('painted')}")

    page.click("#knn-reset")
    page.wait_for_timeout(500)
    check("demo reset restores a readable readout",
          text_of(page, "#knn-readout") not in ("", "—"), text_of(page, "#knn-readout")[:120])


def check_retrain_guardrail(page) -> None:
    """The 20-row sample is below the retrain minimum; the app must refuse it and keep
    serving the 700 simulated rows rather than half-retraining."""
    page.evaluate(JS_SCROLL_TO, "dashboard")
    page.wait_for_timeout(300)
    if not os.path.exists(SAMPLE_CSV):
        check("sample CSV exists for the retrain guardrail", False, SAMPLE_CSV)
        return
    page.set_input_files("#retrain-file", SAMPLE_CSV)
    page.click("#retrain-run")
    # The button is disabled while the upload is in flight; its release is the signal
    # that the server has answered, whatever the status line currently says.
    page.wait_for_function("() => !document.getElementById('retrain-run').disabled", timeout=60000)
    page.wait_for_timeout(400)
    status = text_of(page, "#retrain-status")
    check("a too-small upload is refused", "30" in status and "row" in status.lower(), status[:160])
    check("models are unchanged after a refused retrain",
          page.evaluate("() => window.App.state.metrics.dataset.rows") == 700
          or page.evaluate("() => window.App.state.metrics.dataset.train_size + "
                           "window.App.state.metrics.dataset.test_size") == 700,
          str(page.evaluate("() => window.App.state.metrics.dataset")))
    page.click("#predict-submit")
    page.wait_for_timeout(1500)
    check("the predictor still works after a refused retrain",
          text_of(page, "#result-verdict") in ("OVER BUDGET", "WITHIN BUDGET"),
          text_of(page, "#result-verdict"))


def check_theme(page) -> None:
    light_bg = page.evaluate(JS_BODY_BG)
    html_class_before = page.evaluate("() => document.documentElement.className")
    page.click("#theme-toggle")
    page.wait_for_timeout(900)
    dark_bg = page.evaluate(JS_BODY_BG)
    html_class_after = page.evaluate("() => document.documentElement.className")
    check("dark mode adds the dark class",
          "dark" not in html_class_before and "dark" in html_class_after,
          f"{html_class_before!r} -> {html_class_after!r}")
    check("dark mode changes the rendered body colour", light_bg != dark_bg,
          f"{light_bg} -> {dark_bg}")
    check("theme toggle reports aria-pressed",
          page.get_attribute("#theme-toggle", "aria-pressed") == "true")
    dark_text = page.evaluate("() => document.body.innerText")
    check("dark mode keeps the rupee and drops no copy",
          RUPEE in dark_text and "$" not in dark_text and len(dark_text) > 2000,
          f"{dark_text.count(RUPEE)} rupee signs, {len(dark_text)} chars")
    check("charts are re-painted for the dark theme",
          page.evaluate(JS_CANVAS_REPORT, ["chart-state"])[0].get("painted", 0) > 0)

    page.screenshot(path=os.path.join(SCREENSHOT_DIR, "10-dark-mode.png"), full_page=False)

    page.click("#theme-toggle")
    page.wait_for_timeout(900)
    back_bg = page.evaluate(JS_BODY_BG)
    check("toggling back restores the light colours", back_bg == light_bg, f"{dark_bg} -> {back_bg}")
    check("theme toggle aria-pressed resets",
          page.get_attribute("#theme-toggle", "aria-pressed") == "false")


def check_small_screen(page) -> None:
    page.set_viewport_size({"width": 375, "height": 812})
    page.wait_for_timeout(1200)
    report = page.evaluate(JS_OVERFLOW)
    check("no horizontal overflow at 375px",
          report["htmlScrollWidth"] <= report["htmlClientWidth"] + 1
          and report["bodyScrollWidth"] <= report["bodyClientWidth"] + 1,
          f"html {report['htmlScrollWidth']}/{report['htmlClientWidth']} "
          f"body {report['bodyScrollWidth']}/{report['bodyClientWidth']} "
          f"offenders={report['offenderCount']}")
    if report["offenderCount"]:
        print("      offenders @375:", report["offenders"][:6])

    page.evaluate(JS_SCROLL_TO, "predictor")
    page.wait_for_timeout(600)
    page.click("#predict-example")
    page.wait_for_selector("#predict-result:not([hidden])", timeout=20000)
    page.wait_for_timeout(900)
    check("predictor works at 375px",
          text_of(page, "#result-verdict") in ("OVER BUDGET", "WITHIN BUDGET"),
          text_of(page, "#result-verdict"))
    nav_links = page.evaluate(
        "() => document.querySelectorAll('#navbar nav a').length")
    check("navbar links survive the mobile layout", nav_links >= 6, f"links={nav_links}")
    check("mobile sidebar starts hidden and inert", page.evaluate("""() => {
      const sidebar = document.getElementById('navbar');
      return sidebar.inert && getComputedStyle(sidebar).visibility === 'hidden'
        && document.getElementById('sidebar-toggle').getAttribute('aria-expanded') === 'false';
    }"""))
    page.click("#sidebar-toggle")
    check("mobile menu opens with keyboard focus inside", page.evaluate("""() => {
      const sidebar = document.getElementById('navbar');
      return !sidebar.inert && getComputedStyle(sidebar).visibility === 'visible'
        && document.activeElement.id === 'sidebar-close'
        && document.getElementById('sidebar-toggle').getAttribute('aria-expanded') === 'true';
    }"""))
    page.locator('#navbar a[href="#team"]').focus()
    page.keyboard.press("Tab")
    check("mobile drawer traps forward keyboard focus", page.evaluate("""() =>
      document.activeElement === document.querySelector('#navbar .brand-link')"""))
    page.keyboard.press("Shift+Tab")
    check("mobile drawer traps backward keyboard focus", page.evaluate("""() =>
      document.activeElement === document.querySelector('#navbar a[href="#team"]')"""))
    page.keyboard.press("Escape")
    check("Escape closes the drawer and restores focus", page.evaluate("""() =>
      document.getElementById('navbar').inert
      && document.activeElement.id === 'sidebar-toggle'
      && !document.body.classList.contains('sidebar-open')"""))
    page.click("#sidebar-toggle")
    page.click('#navbar a[href="#predictor"]')
    page.wait_for_timeout(900)
    check("mobile navigation closes menu and focuses destination", page.evaluate("""() =>
      document.getElementById('navbar').inert && document.activeElement.id === 'predictor'
      && document.getElementById('workspace-title').textContent === 'Budget predictor'"""))
    page.click("#sidebar-toggle")
    page.mouse.click(350, 100)
    check("backdrop closes the drawer", page.evaluate("""() =>
      document.getElementById('navbar').inert && document.activeElement.id === 'sidebar-toggle'"""))
    page.click("#sidebar-toggle")
    page.set_viewport_size({"width": 1280, "height": 900})
    check("desktop resize restores sidebar and releases scroll lock", page.evaluate("""() =>
      !document.getElementById('navbar').inert
      && getComputedStyle(document.getElementById('navbar')).visibility === 'visible'
      && !document.body.classList.contains('sidebar-open')
      && document.getElementById('sidebar-backdrop').hidden"""))
    page.set_viewport_size({"width": 375, "height": 812})
    page.wait_for_timeout(300)


JS_WALK_REVEAL = """
async () => {
  // Scroll to each section the way a reader does and wait out the 0.55s CSS fade, so a
  // section that never gets .is-visible is reported rather than a mid-transition one.
  const stuck = [];
  for (const el of [...document.querySelectorAll('section.reveal')]) {
    el.scrollIntoView({block: 'center'});
    await new Promise((r) => setTimeout(r, 900));
    const opacity = Number(getComputedStyle(el).opacity);
    if (!el.classList.contains('is-visible') || opacity < 0.95) {
      stuck.push(el.id + '=opacity ' + opacity.toFixed(2)
        + ' of a ' + Math.round(el.getBoundingClientRect().height) + 'px section');
    }
  }
  return stuck;
}
"""


def check_reveal_short_window(base_url, browser, attach) -> None:
    """A fresh page in a deliberately short window.

    The reveal gate used to be `threshold: 0.08`, which is a fraction of the *section's own
    area*. The route planner is ~6 700 px tall at 500 px wide, so 8 % of it fits on no screen
    at any scroll position: the section stayed at opacity 0, and the map inside it -- with its
    tiles, markers and corridor line all painted correctly -- simply was not there. A 500x513
    viewport is that arithmetic made visible, and only a fresh load catches it, because the
    1280x900 pass earlier in this run already flipped every section on.
    """
    page = browser.new_page(viewport={"width": 500, "height": 513})
    attach(page)
    try:
        page.goto(base_url.rstrip("/") + "/", wait_until="load")
        page.wait_for_function("() => window.Routes && window.Routes.state.map", timeout=30000)
        stuck = page.evaluate(JS_WALK_REVEAL)
        check("every section reveals in a 500x513 window", not stuck,
              "; ".join(stuck)[:260])
        page.evaluate("() => document.getElementById('india-map').scrollIntoView({block:'center'})")
        page.wait_for_timeout(700)
        shown = page.evaluate("""() => {
          const map = document.getElementById('india-map');
          const rect = map.getBoundingClientRect();
          const tiles = [...map.querySelectorAll('img.leaflet-tile')];
          return {
            opacity: Number(getComputedStyle(document.getElementById('routes')).opacity),
            width: Math.round(rect.width), height: Math.round(rect.height),
            tiles: tiles.length,
            painted: tiles.filter((t) => t.complete && t.naturalWidth > 0).length,
            markers: map.querySelectorAll('.leaflet-marker-icon, path.leaflet-interactive').length,
          };
        }""")
        check("the route planner is actually visible at that size",
              shown["opacity"] > 0.95, f"section opacity {shown['opacity']}")
        check("the map keeps its 420px canvas at 500px wide",
              shown["height"] >= 380 and shown["width"] >= 300,
              f"{shown['width']}x{shown['height']}")
        # Tiles only count if the CDN answered; the page's own fallback is the accepted
        # alternative, exactly as in the 1280x900 planner checks.
        check("the map shows tiles or its documented offline fallback",
              shown["painted"] > 0 or shown["markers"] > 0,
              f"tiles={shown['tiles']} painted={shown['painted']} markers={shown['markers']}")
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "32-routes-500px.png"))
    finally:
        page.close()


# --------------------------------------------------------------------------- #
# The test itself
# --------------------------------------------------------------------------- #

def run_browser_checks(base_url: str) -> None:
    from playwright.sync_api import sync_playwright

    console_errors: list[str] = []
    page_errors: list[str] = []
    server_errors: list[str] = []
    client_errors: list[str] = []

    def on_console(msg) -> None:
        if msg.type != "error":
            return
        text = msg.text
        # A map tile or the Leaflet CDN being unreachable is an environment fact, not
        # an app bug; the page shows its own fallback for it.
        location = (msg.location or {}).get("url", "") if isinstance(msg.location, dict) else ""
        if from_external(location) or from_external(text):
            return
        # Chromium logs a 4xx response as a console error; the deliberate bad-input
        # probes below are expected to produce exactly those, and nothing else.
        if "Failed to load resource" in text and re.search(r"status of 4\d\d", text):
            return
        console_errors.append(f"{msg.type}: {text}")

    def on_response(response) -> None:
        if from_external(response.url):
            return
        if response.status >= 500:
            server_errors.append(f"{response.status} {response.url}")
        elif response.status >= 400:
            client_errors.append(f"{response.status} {response.url}")

    def attach(p) -> None:
        """Wire the four collectors onto a page. /building is a second page in the same
        context, so it needs the same listeners or its console errors and 4xx responses
        would go uncounted."""
        p.on("console", on_console)
        p.on("pageerror", lambda exc: page_errors.append(str(exc)))
        p.on("response", on_response)
        p.on("requestfailed", lambda request: None if from_external(request.url)
             else server_errors.append(f"FAILED {request.url} ({request.failure})"))

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            accept_downloads=True,
            device_scale_factor=1,
        )
        page = context.new_page()

        attach(page)

        page.goto(base_url + "/", wait_until="domcontentloaded", timeout=90000)

        # Wait for the CDN libraries and for the app to have fetched its data and made
        # its opening prediction.
        page.wait_for_function(
            """() => typeof Chart !== 'undefined'
                     && window.App && window.App.state
                     && window.App.state.meta
                     && window.App.state.metrics
                     && window.App.state.dashboard
                     && !!window.App.lastPrediction()
                     && window.Routes && window.Routes.isReady()""",
            timeout=90000,
        )
        page.wait_for_timeout(2000)

        # Scroll the whole page once so every IntersectionObserver reveal has fired.
        for section_id in SECTIONS:
            page.evaluate(JS_SCROLL_TO, section_id)
            page.wait_for_timeout(350)
        page.evaluate("() => window.scrollTo(0, 0)")
        page.wait_for_timeout(1500)

        banner = rendered(page, "#api-banner")
        check("no startup error banner", not is_visible(banner),
              "" if not banner else banner["text"][:120])

        body_text = page.evaluate("() => document.body.innerText")
        # One phase crashing (a hung button, a missing id) must not hide the result of
        # the others, so each runs behind a guard that reports the crash as a check.
        phases = [
            ("sections", check_sections, (page,)),
            ("credits", check_credits, (page, body_text)),
            ("currency", check_currency, (page, body_text)),
            ("hero counters", check_counters, (page,)),
            ("charts", check_charts, (page,)),
            ("vocabularies", check_vocabularies, (page,)),
            ("performance panel", check_performance, (page,)),
            ("dataset explorer", check_dataset_explorer, (page,)),
            ("predictor", check_predictor, (page,)),
            ("route planner", check_routes, (page,)),
            ("planning hierarchy", check_planning, (page,)),
            ("dashboard stream", check_dashboard_stream, (page,)),
            ("KNN demo", check_knn_demo, (page,)),
            ("retrain guardrail", check_retrain_guardrail, (page,)),
            ("theme toggle", check_theme, (page,)),
        ]
        for label, phase, phase_args in phases:
            try:
                phase(*phase_args)
            except Exception as exc:  # noqa: BLE001 - report and carry on
                check(f"{label} phase completed", False, f"{type(exc).__name__}: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            try:
                check_exports(page, tmp)
            except Exception as exc:  # noqa: BLE001
                check("exports phase completed", False, f"{type(exc).__name__}: {exc}")

        page.evaluate(JS_CLEAR_TOASTS)
        page.set_viewport_size({"width": 1280, "height": 900})
        page.wait_for_timeout(700)
        capture_sections(page, "desktop")

        report = page.evaluate(JS_OVERFLOW)
        check("no horizontal overflow at 1280px",
              report["htmlScrollWidth"] <= report["htmlClientWidth"] + 1
              and report["bodyScrollWidth"] <= report["bodyClientWidth"] + 1,
              f"html {report['htmlScrollWidth']}/{report['htmlClientWidth']} "
              f"body {report['bodyScrollWidth']}/{report['bodyClientWidth']} "
              f"offenders={report['offenderCount']}")
        if report["offenderCount"]:
            print("      offenders @1280:", report["offenders"][:6])

        try:
            check_small_screen(page)
        except Exception as exc:  # noqa: BLE001
            check("mobile phase completed", False, f"{type(exc).__name__}: {exc}")

        capture_sections(page, "mobile")
        page.evaluate(JS_CLEAR_TOASTS)
        page.set_viewport_size({"width": 1280, "height": 900})
        page.wait_for_timeout(500)
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "00-full-page-desktop.png"),
                        full_page=True)
        page.evaluate(JS_CLEAR_TOASTS)
        page.set_viewport_size({"width": 375, "height": 812})
        page.wait_for_timeout(500)
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "00-full-page-mobile.png"),
                        full_page=True)

        # The second page in its own tab: a fresh context page keeps the 1280x900 the
        # context was created with, even though the first page is now at phone width.
        try:
            check_building_page(base_url, context, attach)
        except Exception as exc:  # noqa: BLE001
            check("building page phase completed", False, f"{type(exc).__name__}: {exc}")

        # A brand-new page at a deliberately awkward window size, so a reveal gate that only
        # fires at desktop proportions cannot pass on the state the 1280x900 run left behind.
        try:
            check_reveal_short_window(base_url, browser, attach)
        except Exception as exc:  # noqa: BLE001
            check("short-window reveal phase completed", False, f"{type(exc).__name__}: {exc}")

        browser.close()

    # ---------------- console / network ---------------- #
    check("no console errors", not console_errors,
          f"{len(console_errors)} error(s): " + " || ".join(console_errors[:5]))
    check("no uncaught page errors", not page_errors,
          f"{len(page_errors)}: " + " || ".join(page_errors[:5]))
    check("no failed or 5xx requests", not server_errors,
          f"{len(server_errors)}: " + " || ".join(server_errors[:5]))
    unexpected = [entry for entry in client_errors
                  if not any(entry.endswith(path) for path in EXPECTED_CLIENT_PATHS)]
    check("every 4xx is one of the deliberate validation probes", not unexpected,
          f"{len(client_errors)} 4xx total; unexpected: " + " || ".join(unexpected[:5]))


JS_CLEAR_TOASTS = """() => {
    const stack = document.getElementById('toast-stack');
    if (stack) stack.innerHTML = '';
}"""

# The sticky navbar and the fixed toast stack are painted wherever they happen
# to sit, so an element screenshot of a tall section showed the navbar in the
# middle of it. Hide them while capturing, then put them back.
JS_TOGGLE_CHROME = """(hide) => {
    ['navbar', 'toast-stack', 'api-banner'].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.style.display = hide ? 'none' : '';
    });
}"""


def capture_sections(page, tag: str) -> None:
    """Screenshot each section individually for the visual review pass."""
    page.evaluate(JS_TOGGLE_CHROME, True)
    page.evaluate(JS_CLEAR_TOASTS)
    try:
        for index, section_id in enumerate(SECTIONS, start=1):
            handle = page.query_selector(f"#{section_id}")
            if not handle:
                continue
            try:
                handle.scroll_into_view_if_needed(timeout=8000)
            except Exception:  # noqa: BLE001
                pass
            page.wait_for_timeout(450)
            path = os.path.join(SCREENSHOT_DIR, f"{index:02d}-{section_id}-{tag}.png")
            try:
                handle.screenshot(path=path, timeout=30000)
            except Exception as exc:  # noqa: BLE001
                print(f"      could not screenshot #{section_id} ({tag}): {exc}")
    finally:
        page.evaluate(JS_TOGGLE_CHROME, False)


def main() -> int:
    shutil.rmtree(SCREENSHOT_DIR, ignore_errors=True)
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = os.path.join(tempfile.gettempdir(), "india_smoke_server.log")

    print(f"Starting Flask server on {base_url} (log: {log_path})")
    server = start_server(port, log_path)
    try:
        if not wait_for_server(base_url + "/meta"):
            with open(log_path, encoding="utf-8", errors="replace") as handle:
                print("Server did not start. Log:\n" + handle.read()[-4000:])
            check("server starts", False, "no response from /meta")
            return 1
        check("server starts", True, base_url)
        run_browser_checks(base_url)
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            log_tail = handle.read()
        if "Traceback" in log_tail:
            check("server logged no tracebacks", False, log_tail[-1500:])
        else:
            check("server logged no tracebacks", True)

    failed = [name for name, passed, _ in RESULTS if not passed]
    print("\n" + "=" * 72)
    print(f"SMOKE TEST: {len(RESULTS) - len(failed)} passed, {len(failed)} failed")
    if failed:
        print("Failed checks:")
        for name in failed:
            print("  -", name)
    print(f"Screenshots: {SCREENSHOT_DIR}")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
