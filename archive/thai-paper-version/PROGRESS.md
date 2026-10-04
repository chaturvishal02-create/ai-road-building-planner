# PROGRESS

Project: **Government Construction Project Budget Prediction Using Machine Learning (KNN)**
Team: Vishal Raghunath Chatur & Mohsin Jabiulla Balgyar · Guide: Tapase Miss · DBATU (B.Tech AI & DS) · Mini Project
Spec: see `SPEC.md`

---

## Environment setup

- [x] Python available (`py` launcher, 3.14.3)
- [x] Dependencies installed (Flask, scikit-learn, pandas, numpy, joblib, pytest)
- [x] **Blocker found & resolved:** Windows Smart App Control blocked `sklearn/utils/_weight_vector.cp314-win_amd64.pyd` in scikit-learn 1.9.1, which cascaded into `sklearn.neighbors` → `KNeighborsClassifier` unusable. Fixed by pinning `scikit-learn==1.8.0` (verified loadable). Smart App Control left enabled.
- [x] Playwright + Chromium installed

---

## Phase 1 — SPEC

- [x] `SPEC.md` written (requirements, file structure, API routes, data flow, testing strategy, acceptance criteria, assumptions)
- [x] `PROGRESS.md` created with every acceptance criterion as an unchecked box

## Phase 2 — BUILD

- [x] `model/generate_dataset.py` → `data/projects.csv` (692 rows)
- [x] `model/train.py` → `model/knn_model.joblib` + `model/model_meta.json`
- [x] `app.py` Flask API: `/meta`, `/predict`, `/metrics`, `/dataset`, `/k-curve`, `/retrain`
- [x] `templates/index.html` + `static/` — sections 1–8 + footer
- [x] PDF / CSV prediction export
- [x] `README.md`

## Phase 3 — TEST

- [x] `pip install -r requirements.txt`, train model, start server
- [x] `pytest` green — 113 passed (`tests/test_model.py`, `tests/test_api.py`)
- [x] `tests/smoke_test.py` green — 81 passed, screenshots written to `screenshots/`

## Phase 4 — CRITIQUE

- [x] Issues list written below

## Phase 5 — FIX & RE-TEST

- [x] Round 1 — 14 issues found, all fixed; re-test 113 pytest + 81 smoke passed
- [x] Round 2 — 1 issue found (neighbour table clipped its OUTCOME column), fixed; re-test 113 + 81 passed
- [x] Round 3 — critique found **zero new issues** → loop terminated
- [ ] Round 4 (not needed)
- [ ] Round 5 (not needed)
- [ ] Round 6 (limit, not reached)

## Phase 6 — POLISH

- [x] Screenshot review (spacing, alignment, contrast) + fixes
- [x] Final full test re-run (113 pytest + 81 smoke, 0 failures)
- [x] Clean-venv check of the single run command (`pip install -r requirements.txt && python app.py`)
      → first run failed: a fresh venv resolved scipy 1.18.1, whose
      `scipy/linalg/_solve_toeplitz*.pyd` is blocked by Smart App Control (same
      "newest wheel has no reputation" pattern that forced the `scikit-learn==1.8.0` pin).
      Pinned `scipy==1.17.1`, reinstalled → imports OK, `app.py` served `/`, `/meta`,
      `/dataset`, `/metrics`, `/k-curve`, `/predict` with 200s (400 on a bad K), no traceback.
      Scratch venv deleted afterwards.

---

## ACCEPTANCE CRITERIA

- [x] App starts with no errors via the single run command
- [x] `projects.csv` has 692 rows, ~20.66 % over budget, correct category counts
- [x] `OneHotEncoder` + `KNeighborsClassifier` Pipeline, 80/20 stratified split, fixed seed
- [x] Predictor works for all dropdown combinations and validates input
- [x] Changing K updates the prediction and the vote breakdown
- [x] Confusion matrix, classification report and K-vs-accuracy chart render correctly
- [x] "Reported in paper" vs "Live" comparison is shown
- [x] No console errors; responsive at 375 px and 1280 px; dark/light mode works
- [x] Team names, guide and DBATU details appear in hero, Team section, footer, README and PDF export
- [x] pytest and the Playwright smoke test both pass
- [x] README covers setup, dataset note, model explanation, assumptions, screenshots
- [x] Code is commented and beginner-readable (explainable in viva)

---

## ISSUES LOG

### Round 1 — critique (after the first full test pass)

Environment / model

1. **[blocker, fixed]** Windows Smart App Control blocked `sklearn/utils/_weight_vector.cp314-win_amd64.pyd`
   in scikit-learn 1.9.1; the failure cascaded `_sgd_fast` → `sklearn.linear_model` → `sklearn.neighbors`,
   making `KNeighborsClassifier` unusable. Resolved by pinning `scikit-learn==1.8.0` (verified loadable by
   extracting each wheel's `.pyd` and loading it with `ctypes.WinDLL`). Smart App Control left enabled.
2. **[bug, fixed]** `ModelService.predict` called `knn.kneighbors()` with the raw 4-column query,
   raising `ValueError: X has 4 features, but KNeighborsClassifier is expecting 126`. The query is now
   encoded through the pipeline's `OneHotEncoder` before the neighbour lookup.
3. **[bug, fixed]** `classification_report(..., output_dict=True)` keys the F1 column `f1-score`, not
   `f1_score` → `KeyError`. Added `_normalize_report()` so live and reported figures share one shape.
4. **[fake-number risk, fixed]** The plain-English explanation hard-coded "553 training projects"; after
   `/retrain` that would have been false. The real training-set size is now passed in.
5. **[fake-number risk, fixed]** The K-curve tooltip hard-coded 139 test rows. `compute_k_curve` now stores
   `test_size` / `train_size` and the chart reads them.

Layout / visual

6. **[layout, fixed]** A long `<option>` ("Office of the National Economic and Social Development Board")
   stretched `#predict-form` to 817 px at a 375 px viewport — 45 overflow offenders, `html.scrollWidth`
   833 vs 375. A `<select>` sizes itself to its longest option and grid items default to `min-width:auto`.
   Fixed with `.grid > * { min-width: 0 }` and `.field { min-width: 0 }`.
7. **[visual, fixed]** Confusion-matrix axis labels rendered as "PREDICTEDNO": `.heat-axis` was a row
   flexbox, which collapses `<br>`. Now two stacked children in a column flexbox.
8. **[contrast, fixed]** The "Live model" compare-panel header used white text on amber (~2:1 contrast).
   Now near-black on amber; the navy "Reported in paper" header keeps white.
9. **[visual, fixed]** The predictor form stretched to the full grid row height, leaving a large empty
   panel under its buttons. Added `self-start`.
10. **[visual, fixed]** The neighbour table was clipped horizontally inside the result column and made the
    result card ~1500 px tall. Dropped the Location column from the on-screen table (it stays in the PDF
    and CSV exports), added a `compact` min-width and a `neighbour-scroll` max-height with internal scroll.

Test harness

11. **[test, fixed]** Element screenshots painted the sticky navbar in the middle of tall sections and a
    toast over the content. `capture_sections` now hides `#navbar` / `#toast-stack` / `#api-banner` while
    capturing and clears toasts before the full-page shots.
12. **[test, fixed]** The drag test failed because `#knn-canvas` was off-screen, so Playwright clicked the
    wrong coordinates. The canvas is scrolled to the centre of the viewport first.
13. **[test, fixed]** The overflow probe flagged the skip link parked at `left:-999px`, which cannot create
    scrollable overflow. Only right-hand overflow is now reported.

Polish

14. **[polish, fixed]** The KNN demo's default test point sat 0.008 from a sample point, giving a boring
    0/5 vote. Moved to (0.58, 0.50): at K=5 the vote is 3 Yes / 2 No and at K=12 it flips to 4 Yes / 8 No,
    which demonstrates the K sensitivity the section is meant to teach.

### Round 2 — critique (after the round-1 re-test)

15. **[visual, fixed]** The neighbour table still clipped its OUTCOME column at 1280 px: the
    department cell's 16 rem truncate cap plus default cell padding made the table's natural
    width (~780 px) exceed the result column's inner width (~647 px). Tightened
    `.data-table.compact` to `min-width: 30rem`, `0.5rem/0.6rem` padding and a 10 rem truncate
    cap, so all six columns now fit without horizontal scrolling on desktop while mobile keeps
    its scroll.

Re-test after the fix: **113 pytest + 81 smoke checks, 0 failures.**

### Round 3 — critique (after the round-2 re-test)

Reviewed all regenerated screenshots (home, dataset, predictor, performance, KNN demo, about,
team, footer, dark mode, both viewports). **Zero new issues found**, so the test-and-fix loop
terminated here per the workflow.

Observations deliberately left as-is, with reasons:

* The doughnut charts leave horizontal whitespace inside their cards — Chart.js keeps a square
  aspect for pie/doughnut types; forcing a wider box would distort them.
* The "Methodology" card in About is slightly taller than its content because it shares a grid
  row with the citation card; equal-height cards read as intentional alignment.
* The dark-mode screenshot shows the "PDF downloaded / CSV downloaded" toasts, which is correct
  behaviour (they were triggered seconds earlier by the export checks) and demonstrates them.
