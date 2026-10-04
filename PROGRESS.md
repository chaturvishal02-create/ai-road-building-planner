# PROGRESS — v2 India rewrite + v2.1 route planner + v2.2 completion time + v3 hierarchy & building estimator + v3.2 deploy

Run log for the India 2025–26 build, the route planner added on top of it, the completion-time /
urban-terrain pass that followed, the v3 pair of briefs (the four-level road-planning hierarchy with
the brief's top-down formula, and the additive government-building estimator), and the v3.2 pass that
makes the same code deployable with `gunicorn app:app`.
Spec: `SPEC_INDIA.md` (§12 route planner, §13 hierarchy + planning formula + building module,
F20 + README §15 deploy).
v1 was copied to `archive/thai-paper-version/` before git was initialised (the repo's single history
commit contains it), so "archived" there means preserved in the tree, not deleted.

**Status: complete.** `py -m pytest` → 445 passed in ~15 s. `py tests/smoke_test.py` → 332 checks
passed, 0 failed, 26 screenshots regenerated, no console errors. Dev server: `py -X utf8 app.py` →
http://127.0.0.1:5000 (and `/building`), bound to `0.0.0.0` so a phone on the same Wi-Fi can open it
too.

---

## Phase 1 — SPEC

- [x] `SPEC_INDIA.md` written (schema, models, data layer, currency rules, acceptance criteria)
- [x] v1 artifacts moved to `archive/thai-paper-version/` (not deleted)
- [x] PROGRESS reset with every criterion unchecked, then worked down
- [x] §12 + F14/F15 added when the route-planner brief arrived; §12.3 + F16 when the follow-up
      asked for completion time and urban working

## Phase 2 — BUILD

- [x] 2a Dataset generator → `data/india_projects_2025_26.csv` (700 rows) + `data/sample_20_projects.csv`
- [x] 2b Data layer → `data_layer/storage.py` (local/HTTP/S3), `data_layer/pipeline.py`, feature store + manifest
- [x] 2c Model → `model/train.py`, KNN + RandomForest comparison, `model_meta.json`
- [x] 2d Flask API → `/meta /predict /metrics /dataset /dashboard /sources /retrain /stream /pipeline /health`
- [x] 2e Frontend → 9 sections, ₹ formatting, tricolour accents, dashboard charts, SSE feed
- [x] 2f Exports (PDF/CSV) + README
- [x] 2g Route engine → `model/route_engine.py` (56 cities, haversine + corridor calibration,
      quantity take-off, terrain/soil factors, risk weights, geodesic polyline)
- [x] 2h Route API → `/route/cities`, `/route/samples`, `/predict-route` (with the brief's
      `{distance, budget, risk}` contract kept beside the full breakdown)
- [x] 2i Route UI → Leaflet 1.9.4 + OpenStreetMap, city dropdowns, swap, corridor redraw from the
      measured geodesic, ₹ cards, component doughnut, Budget-vs-Distance chart, model cross-check
      gauge, skeleton + spinner loading state, offline tile fallback, dashboard styling
- [x] 2j v2.2 engine → `construction_time()`: the brief's three km/day bands (highway 3–5, rural
      1–2, ghat/hilly 0.5–1) with hilly terrain overriding the cross-section, `DAYS_PER_MONTH`
      calendar conversion, min/likely/max months plus a printed `basis`; urban added as a terrain
      class in **four** places (cost 1.15, time drag 1.15, risk weight 14, auto-detect ranking) so
      it cannot be half-implemented; runtime kept distinct from `planned_duration_months`
- [x] 2k v2.2 API + UI → `/predict-route` brief now `{distance, budget, time_months, risk}`;
      `/route/cities` ships the speed bands; `/route/samples` returns months per corridor; a seventh
      "Completion time" card with its band in the tooltip, the "Time basis:" line in the note, and a
      Budget-versus-Time chart (month bars, ₹ crore line)
- [x] 2l v3 planning catalogue → `model/planning_data.py`: seeded State→District→Taluka→Village
      generation (14 × 4 × 3 × 3 = 504 villages), real district centres with placeholder leaves
      seated 4–55 km out, Haversine connectors × 1.1510, per-column `PROVENANCE`, `hierarchy()` /
      `lookup()` / `suggest()` / `point()`, and `data/india_road_project_planning_dataset_2025_26.csv`
      written with the brief's exact 17 columns
- [x] 2m v3 planning engine → `model/planning.py`: the brief's `distance × ₹/km × width × terrain ×
      soil` (rural 1 / highway 2 / expressway 5 crore per km, ghat 1.30, urban 1.15, rocky 1.20,
      clay 1.10, sandy 1.05), fixed 25/20/15/20/20 split as an **ordered list** (Flask sorts dict
      keys), 4/2/1 km-per-day time, and High/Medium/Low by sanction size at ₹500 / ₹200 crore
- [x] 2n v3 API → `GET /locations` (whole tree + `?q=` auto-suggestions across four levels),
      `/predict-route` accepts the brief's nested `{start:{state,district,taluka,village}, end:{…}}`
      body and answers `start_point`/`end_point` + a `planning` sub-object, leaving the original
      five top-level keys byte-identical
- [x] 2o v3 hierarchy UI → `static/js/routes.js`: mode switch (cities ↔ location), four cascading
      selects per end with lower levels cleared on every change, village naming filling its own
      taluka/district, search box resolving to either end, readout line naming what each end resolved
      to, map popups carrying "location name + budget + time", and the planning-formula panel with
      its own material doughnut
- [x] 2p v3 building module (additive only) → `model/building.py` (area × ₹/sq ft × floor × location
      × quality, rural 1 500 / urban 2 500 / remote 1 800, floors 1.00/1.90/2.65, quality 0.85/1.00/1.25,
      7-work split, time = area ÷ sq-ft-per-day + 120 mobilisation days), `POST /predict-building`,
      `templates/building.html` + `static/js/building.js` with their own `/locations` cascade, three
      preset chips and five rate tables — no existing route, file or response touched

## Phase 3 — TEST

- [x] `py -m pytest` green (325: 36 data layer + 37 model + 169 API + 83 routes)
- [x] `tests/test_routes.py` written: calibration band, take-off arithmetic, forced-rate
      reconciliation, risk weight sum, 12 engine + 13 API 400 cases, extrapolation flag; extended in
      v2.2 with the speed bands, the months-vs-printed-speed identity, the hilly override, the brief's
      Nashik→Pune 6–10 month example, the urban 1.15 clock, forced ₹/km leaving time alone, and
      runtime < programme
- [x] Smoke test extended: `routes` added to the section list (10), three new canvases (12 charts /
      13 painted), ~50 planner checks incl. the completion card and the terrain dropdown's option
      values, OSM/CDN hosts excused from the network assertions
- [x] Playwright smoke test green, 23 screenshots regenerated
- [x] v3 pytest → **430** (36 data layer + 37 model + 169 API + 83 routes + 63 planning + 42 building)
- [x] `tests/test_planning.py` (63) written: catalogue regeneration stable from the seed, 504 rows ×
      the 17 briefed columns, every parent link valid, connectors inside the 4–55 km window and equal
      to Haversine × coefficient, `/locations` search at all four levels, ambiguous leaf name refused
      with the taluka named, provenance split per column, the formula's rate/terrain/soil tables equal
      to the brief's numbers, materials summing to the total, thresholds flipping at ₹200/₹500 crore,
      and every printed `basis` string re-deriving its own number
- [x] `tests/test_building.py` (42) written: five-factor product = `cost`, bounds by field name, the
      7-work split summing to 100 % and back to the printed ₹ lakh total, time = area ÷ rate + 120
      days, hospital > school > panchayat per sq ft, and `/predict-building` answering only `{cost,
      time_months, breakdown}`-shaped bodies
- [x] Smoke test extended → **328** checks / **25** screenshots: a `planning hierarchy` phase (mode
      switch, both four-level cascades, refills and clears, village→taluka→district back-fill, search
      resolve, popup content, the whole planning panel and its doughnut, same-point refusal, recovery,
      switch back to cities) and a second-page `/building` phase (cascades, cost card in ₹ lakh, the
      seven lines adding to the card, preset chips, floors×2.65, rejected area naming its field,
      skeleton with the request parked, overflow at 1280/375, two new screenshots). Console/response
      collectors refactored into `attach(page)` so the new tab is covered by the same rules
- [x] Main-page canvases recounted: 13 Chart.js charts / 14 painted canvases (the planning split is
      the thirteenth chart)
- [x] v3.2 `tests/test_deploy.py` (15) → pytest **445** (36 data layer + 37 model + 169 API + 83
      routes + 63 planning + 42 building + 15 deploy). It holds a real `/stream` connection open on a
      threaded HTTP server and measures four page loads plus a prediction against it; that is the
      check that would fail the day someone sets `worker_class = "sync"`
- [x] v3.2 re-run after the start-up and requirements change: pytest **445**, smoke **332/332**, 26 screenshots, zero
      console errors — the browser behaviour is identical because only the import-time bootstrap and
      the bind address changed

## Phase 4 — CRITIQUE / Phase 5 — FIX & RE-TEST

Every entry below is something a test or a screenshot caught, with what changed.

| # | Found by | Issue | Fix |
|---|---|---|---|
| 1 | smoke test | `runPrediction()` re-enabled the submit button only in its error branch — one prediction per page load | button state moved to a `finally` |
| 2 | code review against the brief | `cost_per_km_crore` was cosmetic: the card kept showing the itemised total, so the hint "type a rate to force it" lied | forced rate is now authoritative; take-off preserved as `takeoff_*`, components rescaled |
| 3 | screenshot review | rescaled component rows still printed "₹ 350 / bag" as if the rate had been used | a rescaled row is tagged "· share rescaled" and the note explains the split of assumption vs engineering |
| 4 | probe | typing a DPR length left `distance_basis` claiming the calibrated corridor | basis now appends "the length typed in the request replaced it" |
| 5 | code review | the map drew a straight lat/lon line and never used the geodesic the API measured (up to 35 km of lateral bow on the longest catalogue pairs, measured) | corridor redrawn from `route_path` after the response lands |
| 6 | code review | two quick dropdown changes raced; the older reply could paint over the newer one | monotonic `requestToken`, stale responses dropped |
| 7 | pytest | terrain/soil impact split was arithmetic guesswork, not the engine's closed form | correct forms asserted: `total × (tf−1)/(tf×sf)` and `total × (sf−1)/sf` |
| 8 | pytest | distance was rounded in one place and not another, so 189.2 × 2 ≠ the printed total on a calculator | rounded once, right after validation |
| 9 | smoke test | `page.fill()` on a number input does **not** fire `change`, so the forced-rate check read a stale state | the test clicks submit explicitly; behaviour verified in the browser |
| 10 | smoke test | route cross-checks push real events into the live feed; the feed caps at 12 rows and two events can look identical, so the old count-based assertions broke | rows carry `data-sequence`; checks assert sequence growth and uniqueness |
| 11 | smoke test | OSM tile and CDN failures were being reported as our bugs | those four hosts are whitelisted with attribution; the page ships its own offline fallback card |
| 12 | smoke test (self-inflicted) | `time.sleep()` inside a Playwright route handler blocks the sync dispatch loop, so the loading-animation checks ran *after* the response | the request is parked (handler only records the route) and released explicitly |
| 13 | screenshot review | components table header said "₹ lakh" over values formatted in crore | header → "Cost"; values stay ₹ crore/lakh per `formatINR` |
| 14 | docs audit | README/SPEC still quoted the pre-planner counts (242 pytest, 181 smoke, 9 sections, 21 screenshots, 14 routes) | counts corrected to what actually runs: 314 / 231 / 10 / 23 / 17 |
| 15 | screenshot review, app-wide | `.card` and `.btn` had no padding in `styles.css` and the markup puts no spacing utilities on them, so all 44 cards rendered their content flush against the border and every button was a 26 px text-pill — this predates the route planner and had survived the earlier review | `.card` gets `1.05rem 1.15rem` (and `1.25rem 1.4rem` at `lg`), `.btn` gets `0.5rem 1.05rem`; smoke test re-run green, no overflow at 375 px or 1280 px |
| 16 | the terrain-dropdown check added with v2.2 | `fillSelect` defaults to reading a `name` key, but the terrain and soil catalogue rows are `{key, label}` — so both route-planner dropdowns had rendered blank, unusable options in every browser since the planner shipped, and no test looked at their values | `withAuto(...)` calls now pass `'key', 'label'`; the smoke test asserts the option *values* as well as the selection, so the two lists can never silently empty again |
| 17 | docs audit after v2.2 | README/SPEC/PROGRESS still carried the pre-time counts and the two-figure `{distance, budget, risk}` contract | counts corrected to what actually runs now: 325 pytest / 240 smoke / 10 sections / 23 screenshots / 12 Chart.js charts (13 canvases) / 17 routes, contract `{distance, budget, time_months, risk}`, §5.4-§5.6 and SPEC §12.3-§12.5 renumbered, viva Q12 added |
| 18 | the new hierarchy checks, first run | the synthetic-location disclosure was written into `#rt-loc-readout`, which `readout()` rewrites on **every** level change — so as soon as a taluka or village was picked, the page silently stopped saying the leaf names are generated placeholders | disclosure moved to its own node (`#rt-loc-source`) in `index.html` + `routes.js`; the smoke test now asserts the word "placeholder" survives a full four-level selection, so the sentence can never again be priced out of existence |
| 19 | the new `/building` phase, first run | the second page shipped with a formula card, rate tables and a footer credit line but **no team credits in its hero** — Brief B asked for a new page, not a smaller one, and the credits rule is project-wide | hero byline added to `templates/building.html` naming both developers, the guide and DBATU; the check asserts them on that page specifically |
| 20 | writing the cascade test | asserted `"Bengaluru" in districts`, which failed: the catalogue calls the district **Bengaluru Urban**, and `fillOptions` prepends an empty placeholder so 5 real districts render as 6 options | the check names `Bengaluru Urban` and asserts Maharashtra's and Karnataka's district sets are **disjoint** and that taluka/village are cleared — a stronger assertion than the substring it replaced |
| 21 | writing the ₹ assertions | `toLocaleString('en-IN')` groups as `1,350` / `10,600`, so `"%s" % value` substring comparisons could never match the rendered cards | added `number_in()` to parse rendered en-IN numerals; card checks now compare numbers, not strings |
| 22 | writing the same-point refusal test | copying `#rt-s-*` values into `#rt-e-*` by hand left the destination district select holding the *previous* state's option list, so setting it to a valid name silently no-opped and the test chased a phantom bug in the engine | the copy dispatches a `change` event per level so each cascade refills exactly as a user's click would |
| 23 | manual re-check after the engine edits | a stale `debug=False` server from the previous phase kept answering a valid `{"start":"Pune","end":"Nashik"}` with 400 "A route needs two cities" — the code was right, the process was old | server killed and restarted before any verification; every "the app is broken" report in this project has since been checked against a fresh process first |
| 24 | **the user's own screen**: "route map not show" | the map was fine — tiles loaded, 63 markers painted, corridor drawn — but the whole `#routes` section sat at `opacity: 0`. `initReveal` used `IntersectionObserver` with `threshold: 0.08`, and that ratio is of the **section's own area**: at 500 px wide the planner section is 6 693 px tall, so 8 % of it is 535 px — more than the entire viewport, at any scroll position. The class could never be added. Every desktop-sized window clears it (1280×900 → 0.24, 1366×768 → 0.21, 960×720 → 0.13, 375×667 → 0.09) which is exactly why 328 green checks never saw it: the suite only ever loads the page at one tall-enough ratio, and the mobile pass *resizes* a page that had already revealed | reveal gate is now `threshold: 0, rootMargin: '0px 0px -12% 0px'` — fires on entering the viewport, so section height cannot out-run it. New phase `check_reveal_short_window` loads a **fresh** page at 500×513, walks all nine sections with the 0.55 s fade waited out, and asserts the map is visible with a 420 px canvas and painted tiles (screenshot `32-routes-500px.png`). Smoke test now 332 checks / 26 screenshots |
| 25 | the same investigation | `.map-canvas .leaflet-container { background: transparent }` never matched anything — Leaflet puts `.leaflet-container` on the **same** element that carries `.map-canvas`, so unpkg's own `.leaflet-container { background: #ddd }` won on order and every map showed a flat grey frame before tiles arrived, which reads as "broken map" even when it is only slow | selector corrected to `.map-canvas.leaflet-container` (specificity 0,2,0) with `var(--surface-2)`, so the panel's own surface shows through in both themes |
| 26 | v3.2, before any deploy file existed | `gunicorn app:app` **imports** `app.py`; `bootstrap()` lived only inside `main()`, which a WSGI server never calls. The process would have started cleanly, `/health` would report `ok: false`, and the first page would 500 on an empty `ModelService` — a deploy that looks like a code bug | `bootstrap()` moved to module level with the two start paths written out in the comment, `main()` no longer calls it, and `tests/test_deploy.py::test_importing_the_module_without_app_run_loads_both_models` now fails loudly if the call ever moves back |
| 27 | v3.2, reading gunicorn's defaults | two more silent deploy-killers: its default bind is `127.0.0.1:8000` while a platform routes to the port it injects in `$PORT` (Render's documented default is 10000), and its default `sync` worker handles one request at a time — so the first visitor to open the dashboard's `/stream` would hold the only worker and freeze the site for everyone else | `gunicorn.conf.py` (auto-loaded from the working directory, so the Start Command stays exactly `gunicorn app:app`) binds `0.0.0.0:$PORT`, sets `worker_class = "gthread"` with 8 threads, and keeps `workers = 1` because `EventBroker` is in-process. Both choices are measured, not asserted: the config is parsed at 10000 and 20500, and the held-stream test proves page loads stay under a second |
| 28 | v3.2, auditing what the repo would ship | `requirements.txt` installed pytest **and** Playwright on a 512 MB build box (Playwright then wants a browser download), and with no `.gitignore` in a repo that had none until then, the deploy copy carried 21 `.pyc` files, a literal `%TEMP%/` folder, `dev_server.log` and `.pytest_cache/` | runtime pins moved to `requirements.txt`, test tools to `requirements-dev.txt`, and a commented `.gitignore` added; the tracked junk needs one `git rm -r --cached` (see README §15.3) because ignore files do not un-track anything |
| 29 | v3.2, **a live Render deploy log** | `bash: line 1: gunicorn: command not found`, then `Exited with status 127`. 127 is the shell reporting a missing program: Render installs from the **pushed** `requirements.txt`, and that commit still listed only the training libraries — every deploy fix so far lived in the working tree. `git show HEAD:requirements.txt \| grep -ci gunicorn` returned 0, and HEAD's `app.py` still called `bootstrap()` from `main()` | gunicorn 26.2.0 pinned in the file the build reads, Werkzeug named too because `app.py` imports `HTTPException` from it directly (Flask only guarantees `>=3.1.0`), and `tests/test_deploy.py` now parses every shipped module with `ast` and fails if any third-party import is undeclared, so "the build installs what the code imports" is a test rather than a hope. README §11 gained the exact 127 row and §15.2/§15.3 the `git show HEAD:` pre-flight. Nothing here changes a route or a number |

## Phase 6 — POLISH

- [x] Screenshot review (desktop + mobile for all 10 sections, plus dark mode)
- [x] Final full test re-run: pytest 325, smoke 240/240
- [x] v3 final re-run: pytest **430**, smoke **332/332**, 26 screenshots, zero console errors, and
      every 4xx traceable to a deliberate bad-input probe
- [x] v3.1 hotfix after the user reported "route map not show" — the reveal gate above (issue 24) and
      the dead map-background selector (issue 25); suites re-run green afterwards
- [x] v3 UI pass: singular/plural months ("1 month", not "1 months") in every corridor, connector and
      building card — a sub-15 km village connector rounds to one month and a viva reader notices
- [x] README: §5 route planner with the calibration table, rate table, worked Nashik→Pune
      example, §5.4 completion time (bands, calendar days, runtime vs programme) and §5.5/§5.6
      renumbered; §9 API list; §10 testing; §12 limitations; §13 viva Q11 + Q12; §14 layout
- [x] SPEC §12 route-planner specification with §12.3 time model, F14/F15/F16, acceptance 15
- [x] v3 docs: README §5.7 hierarchy + provenance table, §5.8 planning formula with a worked
      Ghoti→Nashik example and the two-totals explanation, §5.9 `/building`, §1/§9/§10/§12 counts and
      assumptions 16–19, viva Q13/Q14/Q15, two-member split covering the three new engines and the
      second page, §14 layout re-measured file by file (16,842 lines of code, 18,472 with docs);
      SPEC retitled v3 with F17/F18/F19, §5 route table, §6 section list, §9 counts, §10 acceptance
      16–18, §11 assumptions 14–18 and a new §13 in three parts
- [x] Honest labelling checked end-to-end: synthetic dataset disclosed in three places, route
      take-off labelled as assumptions, model cross-check labelled as extrapolation

## Phase 7 — DEPLOY (v3.2, Render free tier)

- [x] `app.py` start-up split documented and fixed: `bootstrap()` at import so a WSGI server that
      never calls `main()` still serves; `host = os.environ.get("FLASK_HOST", "0.0.0.0")` and the
      banner printing the bind, so a container or a phone on the same Wi-Fi can reach it; port still
      from `$PORT` via `config.PORT`
- [x] `requirements.txt` reduced to runtime pins (Flask, **gunicorn 26.2.0**, pandas, numpy,
      scikit-learn 1.8.0, scipy, joblib, boto3 still commented out) and the reason the sklearn pin is
      *load-time* rather than taste written next to it: `model/*.joblib` are pickles of 1.8.0's
      classes. Test tools moved to `requirements-dev.txt`
- [x] `gunicorn.conf.py` added — auto-loaded from the working directory (verified in gunicorn's own
      `get_default_config_file()`), binds `0.0.0.0:$PORT`, `gthread` × 8 threads, 1 worker, 120 s
      timeout, stdout logs, `preload_app = False`
- [x] `Procfile` (`web: gunicorn app:app`), `.python-version` (`3.14`, the file Render reads),
      `runtime.txt` (`python-3.14.3`, what this was built on; Render ignores it), `render.yaml`
      blueprint with the two commands + `healthCheckPath: /health`, and `.gitignore`
- [x] Wheel/interpreter check rather than guesswork: every pinned dependency publishes a
      `manylinux_2_28` wheel for cp313 **and** cp314, gunicorn is `py3-none-any`, and Render's default
      for instances created after 2026-02-11 is Python 3.14.3 — the same interpreter as here
- [x] `tests/test_deploy.py` (15) written and green: WSGI import readiness, pages/static resolving
      after `chdir`, requirements contents plus an `ast` audit of every third-party import in the shipped
      code, sklearn pin matching the artifacts on disk, config parsed
      at two `$PORT` values, Procfile/blueprint agreeing with the documented commands, and the
      held-`/stream` concurrency measurement (4 page loads and a prediction, each well under 2 s,
      with the prediction broadcast to the held connection)
- [x] Memory read honestly: the running server with both pipelines loaded measures **172 MB working
      set** on this Windows machine against the free tier's 512 MB — one worker, and the reason is
      recorded next to the setting
- [x] Verified / not verified stated plainly in README §15.4: gunicorn cannot start on Windows
      (fork-based), so the Start Command is proven by its parsed config and the same WSGI callable
      over a threaded HTTP server; the first Render build log closes that last gap
- [x] Docs: README new §15 (deploy files, the four fixes, push commands and the `git show HEAD:` pre-flight,
      free-tier behaviour), §1
      Python floor from numpy's own metadata (3.12) and the dev-requirements line, §7 five new
      environment rows, §10 the 445 count and the deploy-test bullet, §11 a deploy-side
      troubleshooting table with the live 127 symptom, §13 viva Q16, §14 layout re-measured;
      SPEC retitled v3.2 with F20,
      §2 deploy files, §8 the gunicorn row marked done, §9 the deploy contract and 445, acceptance 19

---

## ACCEPTANCE CRITERIA (from SPEC_INDIA.md §10)

- [x] 1. App starts with no errors via the single run command
- [x] 2. Dataset has 700 rows, 42 % ±3 % over budget, all 10 columns
- [x] 3. `sample_20_projects.csv` exists with 20 Indian rows, loads via the adapter
- [x] 4. `ColumnTransformer` (one-hot + scaler); KNN and RF on the same 80/20 stratified split, fixed seed
- [x] 5. KNN-vs-RF comparison table names a winner by Yes-class F1
- [x] 6. Predictor takes all 7 inputs, validates bad ones, returns verdict + risk % + predicted ₹
- [x] 7. Dashboard Budget-vs-Actual + state-wise distribution from real aggregates
- [x] 8. Live SSE feed updates without reload
- [x] 9. All amounts ₹; `$` appears nowhere in UI, API or exports
- [x] 10. Subtle tricolour accents, accessible contrast, dark/light works
- [x] 11. Local adapter default; S3 path env-var only; no credentials committed
- [x] 12. Team / guide / DBATU in hero, Team, footer, README, PDF export
- [x] 13. pytest + smoke test pass; screenshots regenerated
- [x] 14. README covers setup, provenance, schema, comparison, architecture, migration, route-planner method, assumptions
- [x] 15. Route planner estimates Nashik→Pune, distance within 16 % of published, components sum to the printed total, completion months equal the printed distance ÷ the printed km/day speed, no paid API
- [x] 16. Location mode: four cascading levels per end from `GET /locations`, search resolves a village, ambiguous leaf names refused, placeholder provenance visible after a full selection, city mode unchanged
- [x] 17. Planning panel: the brief's top-down total printed with its own multipliers, five material lines summing to it, and the gap to the take-off stated instead of averaged
- [x] 18. `/building`: separate page + endpoint, `{cost, time_months, breakdown}` in ₹ lakh, seven work lines adding to the card, no risk percentage, `window.App` absent, and no pre-existing route or response modified
- [~] 19. Deployable (v3.2): `gunicorn app:app` imports and serves every route with `$PORT` bound and `/stream` non-blocking — proven locally (445 pytest incl. 15 deploy checks). Marked partial, not done, until the first Render build log shows `Listening at: http://0.0.0.0:$PORT`, because gunicorn cannot run on Windows and that log is the only real confirmation

## Known open items (deliberate, not blockers)

1. Corridor geometry is a calibrated great-circle, not a routed alignment. Swapping in OSRM or an
   OSM road-graph would need a routing service — free, but a second process to run for a viva.
2. The take-off prices one uniform cross-section per corridor; no structures, land or utility cost.
3. Route cross-checks beyond ₹ 227.80 crore are extrapolations by construction — the roads this
   engine costs are bigger than anything in the 700-row training file, which is exactly why the
   warning is on screen rather than in the README only.
4. Leaflet and Tailwind still come from CDNs. Offline the map degrades to a documented fallback and
   every number still arrives from Flask; vendoring the two files is a one-line change per tag.
5. Completion time is a flat km/day band over calendar days: no monsoon shutdown, no acquisition
   phasing, no bridge/tunnel critical path, and no plant-mix effect between the bands. A
   critical-path schedule would be a different piece of software, so the card prints the band it
   assumed and the range that band allows instead of a single confident number.
6. v3: taluka and village **names** are seeded generated placeholders, not the survey of record —
   no free, redistributable source gives coordinates for India's ~6 lakh villages. The 56 district
   centres are real and are what the distance model reads; the disclosure is on the page, in the CSV
   and in `PROVENANCE`. Swapping in a real village gazetteer is a data change, not a code change.
7. v3: `route_engine` and `planning` carry deliberately different soil/terrain tables (engineering
   factors vs the brief's stated uplifts), so one corridor prices two ways and both are printed.
   Merging them would hide which brief each number answers.
8. v3: `/building` is a rate table with no learned component and no probability, because the
   classifier's 10-column contract has no area/floors/quality column. Adding a regression target
   there means publishing a real building-cost dataset first, not inventing one.
9. v3.2: the Render start command is verified by config parsing and a threaded WSGI server, not by
   gunicorn itself — it is fork-based and cannot start on Windows. The first platform build log
   closes that gap, and README §15.4 says so rather than implying a live deploy already happened.
   That log has now been seen once, and it failed for a reason local tests could not see: the build
   installs the **pushed** `requirements.txt`, which did not yet contain gunicorn (`Exited with
   status 127`). The file is fixed and the import audit covers the whole class, but item 9 still
   stands until a build log ends `Listening at: http://0.0.0.0:$PORT` — and it needs the commit and
   the push, not just the working tree.
10. v3.2: `.gitignore` exists only now, so 21 `.pyc` files, a literal `%TEMP%/` scratch folder,
    `dev_server.log` and `.pytest_cache/` are still **tracked** from the repo's first commit. An
    ignore file does not un-track anything; README §15.3 lists the `git rm -r --cached` that does.
