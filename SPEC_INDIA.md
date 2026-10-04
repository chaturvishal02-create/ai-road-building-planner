# SPEC v3.2 — Government Construction Project Budget Prediction (India, 2025–26)

**Project:** Government Construction Project Budget Prediction Using Machine Learning
**Developed by:** Vishal Raghunath Chatur and Mohsin Jabiulla Balgyar
**Guide:** Tapase Miss
**Programme:** B.Tech in Artificial Intelligence & Data Science Engineering, DBATU (Mini Project)
**Version:** 3.2 — v3.0's road-planning hierarchy and building estimator, plus the deploy contract
(F20: `gunicorn app:app`, `$PORT`, `gthread`, runtime/dev requirements split). Built on v2's Indian
dataset rewrite (v2.1 route planner, v2.2 completion time). Supersedes v1 (Thai e-GP reproduction of
a published paper).

---

## 0. What changed from v1, and the consequences

| Area | v1 (Thai / paper) | v2 (India) |
|---|---|---|
| Data | 692 Thai e-GP projects, 4 categorical inputs | Indian schemes, mixed categorical + numeric inputs |
| Reference | Reproduced Kusonkhum et al. (2022) exactly | **No reference paper.** Our own design choices, stated as such |
| "Reported vs Live" panel | Core feature | **Removed** — nothing external to compare against |
| Model | KNN only (one-hot, 4 inputs) | KNN **and** RandomForest, trained and scored side by side |
| Currency | Thai baht / US$ | **₹ only**, in lakh and crore |
| Target rate | 20.66 % over budget (paper-mandated) | ~42 % over budget (our design choice, see §3.4) |

Two decisions the user made explicitly, recorded here because they change the grade-facing story:

1. **Replace fully — India only.** The project no longer cites a research paper. Every number is now *our* result, so the README and viva must present it as an original modelling exercise on simulated Indian data, not as a reproduction.
2. **Cloud-shaped, not cloud-running.** No PySpark (this machine has no JVM), no S3/Azure account, no credentials. The scalable design is real but locally runnable: chunked partitioned batch processing, a pluggable storage adapter with a working `s3://` code path, and Server-Sent Events for live updates. §8 states exactly which line changes when a real cluster or bucket appears.

---

## 1. Requirements

### 1.1 Functional

- **F1 — Indian geography.** States include Maharashtra, Karnataka, Delhi, Uttar Pradesh (plus 10 more). Districts include Pune, Mumbai, Nashik, Bengaluru. Every district is valid *within* its state, enforced by the generator and by API validation.
- **F2 — Indian schemes.** PMGSY (rural roads), Smart Cities Mission, PMAY (housing), Jal Jeevan Mission (water supply), each mapped to a project type.
- **F3 — Schema.** `project_name`, `state`, `district`, `sanctioned_budget_lakh`, `duration_months`, `contractor_type` (Govt/Private), `scheme_name`, plus `actual_cost_lakh` and the `over_budget` target.
- **F4 — Currency.** Every displayed amount uses ₹, formatted as lakh below ₹1 crore and crore at or above it. The character `$` must not appear in any rendered page, export or API response.
- **F5 — Sample dataset.** `data/sample_20_projects.csv` — 20 human-readable Indian rows, used by the test suite and downloadable from the UI.
- **F6 — Predictor.** All seven inputs, model selector (KNN / RandomForest / auto-best), verdict card, **cost-overrun risk %**, predicted actual cost in ₹, and the nearest comparable projects.
- **F7 — Model comparison.** KNN vs RandomForest: accuracy, precision, recall, F1, AUC, per-class report, confusion matrix, feature importances.
- **F8 — Dashboard.** Budget-vs-Actual per scheme, state-wise project distribution (with overrun share), risk-rate by contractor type, top overruns, and a live SSE feed of incoming predictions.
- **F9 — Data layer.** Load from local disk, an HTTP URL, or an S3 bucket through one interface; retrain from any of them.
- **F10 — How it works.** KNN canvas demo retained, plus a RandomForest explainer (Gini reduction, why importances differ).
- **F11 — Credits.** Team, guide, DBATU in hero, Team section, footer, README and PDF export.
- **F12 — Export.** PDF and CSV of a prediction, letterhead with title/team/guide, ₹ values.
- **F13 — Theme.** Dark/light toggle; subtle tricolour accents (§7.3).
- **F14 — Route planner (v2.1).** Pick two Indian cities on a Leaflet/OpenStreetMap map and get a corridor: distance measured and calibrated, a bottom-up ₹ cost take-off (materials, labour, machinery) × terrain × soil + overheads, the months that corridor takes to build at a stated km/day speed, a risk band whose weights are printed, and the trained model's own probability beside it with an extrapolation warning. `POST /predict-route {"start":"Nashik","end":"Pune"}` answers `{"distance","budget","time_months","risk"}` plus the full breakdown. No paid map or routing API, no key (§12).
- **F15 — Two cost views, reconciled on screen.** The brief's `Budget = Distance × ₹ per km` (default ₹ 2 crore/km, adjustable) and the itemised take-off cannot both be the headline, so a typed rate is authoritative and the unforced take-off stays visible as `takeoff_*` with the rescale explained (§12.4).
- **F16 — Completion time and urban working (v2.2).** `Time = Distance ÷ construction speed` over the brief's three bands (highway 3–5, rural 1–2, ghat/hilly 0.5–1 km per day), reported as likely months plus the band's own min–max, with the arithmetic in a printed `basis`; urban stretches cost 1.15 × money **and** 1.15 × time. The construction runtime is kept distinct from the contract programme the classifier reads, and both are shown labelled (§12.3).
- **F17 — Four-level location hierarchy (v3).** Both corridor ends can be picked as State → District → Taluka → Village instead of a city, from `data/india_road_project_planning_dataset_2025_26.csv` (504 rows · 17 columns · 14 states · 56 districts · 168 talukas · 504 villages). `GET /locations` serves the whole tree and answers `?q=` searches across all four levels for the auto-suggestion box; each level refills the ones below it and clears them, a village inherits its taluka and district, and a village name repeated inside the chosen district is refused rather than guessed. District centres carry real coordinates/terrain/soil; taluka and village names are generated placeholders seated 4–55 km out, and the page says so in a node that survives every readout rewrite (§13.1).
- **F18 — The brief's planning formula as a second, labelled view (v3).** `Total = Distance × base cost per km × road width × terrain factor × soil factor` with the brief's own numbers (rural ₹1 / highway ₹2 / expressway ₹5 crore per km, ghat +30 %, urban +15 %, rocky +20 % / clay +10 % / sandy +5 %) and the fixed 25/20/15/20/20 material split, plus size-threshold risk (>₹500 crore High, ₹200–500 Medium, <₹200 Low). It is attached to `/predict-route` as its own `planning` key so the original five-key response contract is untouched, and printed in a panel that shows the multipliers it used and the difference from the bottom-up take-off instead of averaging the two (§13.2).
- **F19 — Government building estimator, additive only (v3).** New page `/building`, new endpoint `POST /predict-building`, new engine `model/building.py`, new controller — no existing route, file or response modified, and a browser check asserts the main front-end bundle is absent on that page. `Total = Area × cost per sq ft × floor factor × location factor × quality factor` (rural ₹1,500 / urban ₹2,500 / remote ₹1,800 per sq ft), `Time = Area ÷ sq-ft-per-day + 120 mobilisation days`, answers `{cost, time_months, breakdown}` with a 7-work split. No probability, because the classifier's 10-column contract has no area/floors/quality column (§13.3).
- **F20 — Deployable on a hosted platform (v3.2).** The same code that runs with `py app.py` runs with `gunicorn app:app`: the WSGI import bootstraps the models, `gunicorn.conf.py` binds the platform's injected `$PORT`, `gthread` keeps the SSE `/stream` from starving the site, one worker keeps the live feed shared, `requirements.txt` carries runtime pins only and `requirements-dev.txt` carries the test tools. Health check `/health`. `requirements.txt` was the cause of the one real deploy failure this project has had — a Render log ending `bash: line 1: gunicorn: command not found` / `Exited with status 127`, because the pushed copy installed only the training libraries — so gunicorn is now pinned there, Werkzeug is named because `app.py` imports it directly, and a test parses every shipped module with `ast` to prove the build installs what the code imports. Nothing about this changes a route, a response or a number (README §15).

### 1.2 Non-functional

- **N1 — Never fake a metric.** All performance figures come from the trained artifacts. The dataset is synthetic and every page that shows it says so.
- **N2 — Simple, explainable code.** B.Tech level: no cluster manager, no ORM, no build step, no framework beyond Flask + Tailwind CDN + Chart.js.
- **N3 — Single run command.** `pip install -r requirements.txt && python app.py`.
- **N4 — No secrets in code.** Cloud credentials are read from environment variables only; nothing is uploaded anywhere without explicit user action.
- **N5 — Responsive** at 375 px and 1280 px, no console errors, accessible contrast.
- **N6 — Tests.** pytest + Playwright smoke test, both green.

---

## 2. File structure

```
app.py                        Flask routes + validation + explanation builder
config.py                     env-var settings (data source, port, cloud keys)
data_layer/
  __init__.py
  storage.py                  StorageAdapter: LocalStore | HttpStore | S3Store
  pipeline.py                 chunked map_partitions() batch job -> partitioned store
model/
  generate_dataset.py         India 2025-26 generator (seeded)
  train.py                    KNN + RandomForest, comparison, artifacts
  route_engine.py             corridor distance + engineering take-off + km/day time (nothing learned)
  planning_data.py            State→District→Taluka→Village catalogue, connectors, /locations search
  planning.py                 the brief's top-down formula + 25-20-15-20-20 split + threshold risk
  building.py                 area × ₹/sq ft × floor × location × quality estimator (nothing learned)
  india_knn.joblib            trained pipelines
  india_rf.joblib
  model_meta.json             metrics, importances, vocabularies, provenance
data/
  india_projects_2025_26.csv  full simulated dataset (~700 rows)
  sample_20_projects.csv      20-row test sample (F5)
  india_road_project_planning_dataset_2025_26.csv   504 locations · 17 columns (F17)
  feature_store/              partitioned chunk output + _manifest.json
templates/index.html          single page, 10 sections
templates/building.html       standalone /building page, own nav + 5 rate tables
static/css/styles.css         theme tokens, tricolour accents, map + loading skeletons
static/js/  app.js charts.js knn-demo.js dashboard.js routes.js building.js
tests/      conftest.py test_model.py test_api.py test_data_layer.py test_routes.py
            test_planning.py test_building.py test_deploy.py smoke_test.py
requirements.txt            runtime pins only — every import the code makes, gunicorn included
requirements-dev.txt        -r requirements.txt + pytest + Playwright
gunicorn.conf.py            bind 0.0.0.0:$PORT, gthread, 1 worker, stdout logs
Procfile · runtime.txt · .python-version · render.yaml · .gitignore · pytest.ini
screenshots/                  generated by smoke_test.py
README.md  SPEC_INDIA.md  PROGRESS.md
archive/thai-paper-version/   v1 dataset, model, docs, screenshots (kept for history)
```

---

## 3. Dataset specification

### 3.1 Columns

| Column | Type | Role | Notes |
|---|---|---|---|
| `project_name` | text | display only | e.g. "Pune–Nashik Segment Road Widening, Phase II" |
| `scheme_name` | 4 categories | **model feature** | PMGSY, Smart Cities Mission, PMAY (Housing), Jal Jeevan Mission |
| `project_type` | 4 categories | display only | Road Construction / Smart City Infrastructure / Housing / Water Supply |
| `state` | 14 categories | **model feature** | Maharashtra, Karnataka, Delhi, Uttar Pradesh, Gujarat, Rajasthan, Madhya Pradesh, Tamil Nadu, Telangana, Andhra Pradesh, West Bengal, Bihar, Punjab, Kerala |
| `district` | ~46 categories | **model feature** | nested under its state; Pune, Mumbai City, Nashik, Nagpur, Thane, Chhatrapati Sambhajinagar, Amravati; Bengaluru Urban, Mysuru, Belagavi, Dakshina Kannada; New Delhi, Central Delhi, East Delhi; Lucknow, Kanpur Nagar, Gautam Buddh Nagar, Agra, Prayagraj, Varanasi, Ghaziabad; etc. |
| `contractor_type` | 2 | **model feature** | `Govt` (department-executed / central PSU) or `Private` |
| `sanctioned_budget_lakh` | float | **model feature** | ₹ lakh, 1 decimal |
| `duration_months` | int | **model feature** | 6–72 |
| `actual_cost_lakh` | float | label source | `budget × (1 + overrun_ratio)` |
| `over_budget` | Yes/No | **target** | `actual_cost_lakh > sanctioned_budget_lakh` |

**Deliberately excluded from the feature set:** `project_type`. It is 1-to-1 with `scheme_name`, so including both would double-count the same information and distort Euclidean distance in KNN. It stays as a display column. This is a real modelling decision worth stating in the viva.

### 3.2 Volume and split
700 rows for FY-2025-26 modelling; 80/20 stratified split with `random_state=42` → 560 train / 140 test.

### 3.3 Scheme mix (our choice, documented as such)

| Scheme | Rows | Share | Rationale |
|---|---|---|---|
| PMGSY | 250 | 35.7 % | highest count of small rural road works |
| PMAY (Housing) | 210 | 30.0 % | large housing component counts |
| Jal Jeevan Mission | 145 | 20.7 % | piped-water schemes, state-heavy |
| Smart Cities Mission | 95 | 13.6 % | few but large comprehensive plans |

### 3.4 Overrun model and target rate

Overrun ratio is drawn per scheme from a shifted distribution, then modulated by contractor type, project size and duration — the patterns a reviewer will expect:

- **Base overrun mean:** Jal Jeevan Mission +18 %, PMGSY +14 %, Smart City +9 %, PMAY +11 %.
- **Contractor:** `Govt`-executed works get a +5 pp shift (department works carry land/utility-clearance delays).
- **Size:** smaller sanctioned budgets overrun by a larger *percentage* (fixed mobilisation cost over a small base).
- **Duration:** each extra year of scheduled time adds +1.5 pp.
- **Underruns exist:** a minority of projects finish below sanction, so the target is not a mechanical yes.

Result: **over_budget ≈ 42 % Yes**, deliberately more balanced than v1's 20.66 %. With no paper to match, a closer-to-balanced target makes the Yes class learnable and the precision/recall trade-off discussable instead of hopeless. Any row failing `actual > sanctioned ⟺ Yes` is a generator bug; the test suite asserts it.

### 3.5 Provenance statement (required on screen)
The dataset is **simulated**. Real project-level sanction/actual pairs for 2025-26 are published only in scattered CAG audit reports and state e-procurement portals, and are not available offline in one clean file. The generator reproduces the *structure and known patterns* of Indian scheme data, not actual projects. This appears in the footer, on the Dataset and Dashboard banners, in README §Dataset, and in the PDF export footer — because passing simulated data off as government records in a submission would be misrepresentation.

---

## 4. Model specification

### 4.1 Preprocessing
`ColumnTransformer`:
- `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` on `scheme_name`, `state`, `district`, `contractor_type`
- `StandardScaler()` on `sanctioned_budget_lakh`, `duration_months`

Numeric features must be scaled: budgets span ₹30 lakh–₹90 crore, which would otherwise swamp the one-hot dimensions in any distance calculation. This is the single most important v1→v2 pipeline change and the answer to "why did you add a scaler".

### 4.2 Two models, same split, same preprocessing

| Model | Parameters | Why |
|---|---|---|
| KNN | `n_neighbors=5`, `weights='uniform'`, `metric='euclidean'`, K swept 1–25 | keeps the project's KNN identity and the interactive demo |
| RandomForest | `n_estimators=300`, `max_depth=None`, `min_samples_leaf=5`, `class_weight='balanced'`, `random_state=42` | handles mixed types natively, gives feature importances, `class_weight` addresses imbalance |

**XGBoost was not used.** It ships large compiled binaries; on this machine Windows Smart App Control has already blocked unsigned `.pyd` files from scikit-learn 1.9.1 and scipy 1.18.1. RandomForest is stdlib-of-ML, needs no extra dependency, and answers the same "add a tree ensemble" requirement. Documented as an assumption.

### 4.3 Model selection
Winner = higher **Yes-class F1**, not accuracy (accuracy is inflated by the majority class). The UI shows both columns; the predictor defaults to the winner but either can be selected.

### 4.4 Cost-overrun risk % and predicted actual cost
- **Risk %** = `P(over_budget = Yes)` from the model. RandomForest gives this from the fraction of its trees voting Yes; KNN from the neighbour vote share. Both are honest, explainable probabilities — not calibrated scores.
- **Predicted actual cost** = `sanctioned × (1 + expected_overrun_pct)`, where `expected_overrun_pct = risk% × mean_overrun_pct_of_trainingYes_rows_in_same_scheme`. Labeled on screen as a *risk-weighted heuristic*, never as a second trained model. This is what makes the ₹ output meaningful rather than decorative.

---

## 5. API routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | renders the page, injects `TEAM` + config |
| `/meta` | GET | vocabularies, scheme→district map, model list, currency settings, data-source status |
| `/predict` | POST | one prediction: verdict, risk %, predicted ₹, neighbours, explanation |
| `/metrics` | GET | per-model metrics, comparison table, confusion matrix, report, importances, K-curve |
| `/dataset` | GET | paginated, searchable, ₹-formatted rows |
| `/dashboard` | GET | state distribution, budget-vs-actual per scheme, contractor risk, top overruns |
| `/sources` | GET/POST | list storage sources, load a different one (local/HTTP/S3), reload the model service |
| `/retrain` | POST | uploaded CSV or a chosen source → retrain in place, restore on failure |
| `/stream` | GET | Server-Sent Events: each new prediction + periodic aggregate heartbeat |
| `/pipeline` | GET | the feature-store manifest as written by the chunked pipeline |
| `/pipeline/rebuild` | POST | re-run the chunked pipeline over the active source and return the new manifest |
| `/sample.csv` | GET | the 20-row Indian sample file, for the "retrain on your own CSV" demo |
| `/route/cities` | GET | the map catalogue: 56 cities with coordinates/terrain/soil, road types, terrain and soil factors (incl. urban), unit rates, corridor coefficient, km/day construction-speed bands, calibration routes |
| `/route/samples` | GET | the 17 calibration corridors, priced and timed live by the engine at the requested road type |
| `/predict-route` | POST | corridor cost and duration: distance, ₹ crore total, component take-off, terrain/soil impact, completion months, risk band, the brief's `{distance, budget, time_months, risk}`, a trained-model `model_check`, and — for a nested State/District/Taluka/Village body — the resolved `start_point`/`end_point` and the top-down `planning` view (§13.1, §13.2) |
| `/locations` | GET | the planning catalogue: full State→District→Taluka→Village tree with counts and per-column provenance, or a flat `?q=` auto-suggestion list across all four levels (§13.1) |
| `/building` | GET | renders the second estimator page with `BUILDING` catalogue injected |
| `/predict-building` | POST | `area × ₹/sq ft × floor × location × quality` → `{cost, time_months, breakdown}` plus the 7-work split, factor basis and training-free note (§13.3) |
| `/health` | GET | liveness plus the active source, row count and loaded models |

Validation: any unknown category, a sanctioned budget outside 0.1–1 000 000 ₹ lakh, a duration outside **1–240 months**, or K outside **1–31** returns **400** with a per-field message, same shape as v1. The two new endpoints validate the same way: an unresolved or ambiguous location and an out-of-range area/floors value both return 400 naming the field (`start_state`, `village`, `area_sqft`, …).

Two limits were relaxed while building, and the spec follows the code rather than the other way round:

- **Duration 1–240, not 1–120.** Jal Jeevan and PMGSY packages in the generated data run to 60 months, but a student demoing "what if this is a metro depot" should not be told the model has never seen 10 years. Values outside the *training* range still get the on-screen "the model extrapolates here" hint from `#budget-live-hint`'s sibling logic, so the honesty is preserved without a hard rejection.
- **K 1–31, not 1–25.** The best K measured on the test split is 25; a slider that stops at the answer invites the question "what does the curve do after it?". 31 keeps the odd numbers (no two-way ties) and shows the curve flattening.

---

## 6. Frontend sections

1. **Home** — hero with team/guide/DBATU, India-specific problem statement (CAG-observed cost overruns on central and state works), animated counters (projects, states, districts, schemes, ₹ crore sanctioned, overrun share), pipeline diagram.
2. **Dataset Explorer** — synthetic-data banner, searchable paginated table with ₹ columns, 4 charts (scheme mix, state mix, overrun rate by scheme, Govt vs Private split), class-balance note.
3. **Predictor** — 7 inputs, model selector, K slider, risk gauge, verdict card (red/green), predicted actual cost and overrun ₹, comparable projects table, PMGSY/JJM caution note, validation, spinner.
4. **Model Comparison** — KNN vs RF scorecard, confusion matrix heatmap for the selected model, per-class report, feature importances bar, K-vs-accuracy curve.
5. **Dashboard** — Budget vs Actual grouped bars per scheme + parity scatter, state-wise stacked distribution, contractor risk rate, top-10 overruns by ₹, **live feed** (SSE).
6. **Route Planner** — Leaflet/OSM map of India centred on Maharashtra with a marker per project
   city, a **mode switch** between two corridor pickers — city dropdowns (Indian cities only) or
   four cascading State → District → Taluka → Village selects per end with a location-search box
   feeding them — swap, road type / terrain / soil / carriageway width / forced ₹ per km inputs,
   seven result cards (distance, total budget, ₹ per km, completion time with its min–max band,
   terrain impact, soil impact, risk level), component table with a share doughnut, terrain and
   soil impact, Budget-vs-Distance and Budget-vs-Time charts, trained-model cross-check gauge with
   an extrapolation warning, skeleton + spinner while the estimate is in flight, and an on-page
   fallback when the tile CDN is unreachable. Below the cards a **planning-formula panel** (§13.2)
   prices the same corridor by the brief's top-down rule: base ₹/km and width cards, a verdict line
   showing the multipliers, the measured × coefficient basis, five material rows, the km/day time
   line, the gap to the take-off stated in words, and its own doughnut.
7. **How the ML Works** — interactive KNN canvas (draggable test point, K highlighted, vote) + RandomForest explainer with a Gini-split example.
8. **About** — problem, methodology, findings, limitations, future scope (real CAG/state data, XGBoost/gradient boosting, calibrated probabilities, genuine Spark/DB migration, cost-regression target, real road geometry instead of the calibrated corridor).
9. **Team** — cards for both developers + guide.
10. **Footer** — exact credit line + synthetic-data disclosure.

`/building` is a **second page**, not a section: its own template, nav (links back to every section
above), hero credits, form, five rate tables, cost/time/rate cards, seven-row work split with its
own doughnut, and footer — see §13.3. The main page's nav carries exactly one link to it.

### 6.1 Currency formatting rule (single source of truth)
`formatINR(lakh)`: `< 100` → `₹ 45.6 lakh`; `≥ 100 and < 100000` → `₹ 4.56 crore`; `≥ 100000` → `₹ 456 crore`. Implemented once in `app.js` and once in `train.py`/`app.py` for exports, with a test asserting equivalence and that no output contains `$`.

### 6.2 Tricolour accent, kept accessible
Navy `#0A1F44` stays the base (the Ashoka chakra is itself navy). Saffron `#FF9933` is an **accent only** — borders, chart series, focus rings, headings above 1.25 rem. Green `#138808` marks the "within budget" verdict. Text-bearing saffron/green are darkened (`#B25A00`, `#0B6B0B`) so body text keeps ≥4.5:1 contrast; the raw flag hues fail on white and are never used for small text.

---

## 7. Data layer and "scalable pipeline" design

### 7.1 Storage adapter (`data_layer/storage.py`)
One interface, three implementations, chosen by env var:

```
StorageAdapter.load(ref) -> pandas.DataFrame      StorageAdapter.put(df, ref)
  LocalStore   reads/writes under data/                      (default)
  HttpStore    GETs a public CSV URL (api-based data loading)
  S3Store      boto3, lazy-imported; reads AWS_ACCESS_KEY_ID /
               AWS_SECRET_ACCESS_KEY / AWS_REGION / S3_BUCKET from env
```

`S3Store` is complete code that is simply never constructed unless `DATA_STORE=s3`. No keys exist in the repo, so nothing can be pushed accidentally; `put()` is refused unless the operator also sets `ALLOW_WRITES=1`. This is the honest version of "added S3": the integration exists, the credentials don't.

**Where a retrain lives.** A CSV uploaded through `/retrain` is written to `data/retrain_active.csv` through the same `LocalStore`, and `bootstrap()` loads that file instead of the shipped dataset when it exists. Without this, restarting the server would show 700 shipped rows on a page whose metrics were computed from a 40-row upload — two datasets on one screen. `#retrain-reset` deletes the file, which puts the shipped CSV back in charge; the shipped dataset is never overwritten, and a retrain smaller than 30 rows or missing one class is refused with 422 before any artifact is touched.

### 7.2 Chunked pipeline (`data_layer/pipeline.py`)
Spark-shaped without Spark:

- `read_chunks(source, chunksize=250)` — streaming iterator, constant memory.
- `map_partitions(fn, chunks)` — applies a transform per partition and concatenates; the same shape as `df.mapInPandas`/`transform`, so porting to Spark means swapping this function, not the business logic.
- Writes a **partitioned feature store**: `data/feature_store/scheme_name=<sanitised scheme>/part-0000N.csv`, one file per 250-row chunk that falls in that partition, plus `_manifest.json` (rows and files per partition, the states each partition holds, column list, sha256 of the source, generated_at, chunk size, elapsed seconds). Partitioned-directory-plus-manifest is exactly how Spark lays out parquet on disk. Directory names are sanitised (`PMGSY (Rural Roads)` → `PMGSY_Rural_Roads`), so `load_feature_store()` restores the true labels from the manifest rather than trusting the folder name.
- The reason we do not write `.parquet` is dependency safety: pyarrow is not installed and its binaries carry the same Smart App Control risk that already bit us on scikit-learn and scipy. Switching `to_csv` → `to_parquet` is a one-line change once pyarrow loads on the target machine, and the code says so.

### 7.3 Real-time
`/stream` is a Server-Sent Events generator. Each `/predict` publishes its result to an in-process queue; the dashboard subscribes and appends live rows, and a heartbeat recomputes aggregate KPIs. Single-process, no message broker, works with Flask's dev server. Documented future step: replace the in-process queue with Kinesis/SNS when genuinely multi-instance.

---

## 8. Migration path (state it, don't fake it)

| Today (this app) | Production equivalent | Change needed |
|---|---|---|
| `LocalStore` | S3 / Azure Blob | `DATA_STORE=s3` + env credentials |
| `map_partitions` on CSV chunks | `spark.sql.DataFrame.mapInPandas` | one function body |
| Partitioned CSV + `_manifest.json` | Delta/Iceberg table on the bucket | add pyarrow, change writer |
| In-process SSE queue | Kinesis / SNS / WebSocket service | publish to stream instead of list |
| joblib artifact on disk | SageMaker Model Registry / MLflow | add registry client |
| Flask dev server, `threaded=True` (local) | Gunicorn behind the platform's proxy | **done** — `gunicorn app:app` + `gunicorn.conf.py` bind `$PORT`, `gthread`, one worker (F20) |

---

## 9. Testing strategy

**pytest** — 445 tests in `test_data_layer.py` (36), `test_model.py` (37), `test_api.py` (169),
`test_routes.py` (83), `test_planning.py` (63), `test_building.py` (42) and `test_deploy.py` (15),
run by `py -m pytest` in about 15 s.

- dataset: 700 rows, exactly the 10 columns, `over_budget ⟺ actual > sanctioned`, every district legal for its state, scheme shares within ±1 of §3.3, target rate 42 % ±3 pp
- pipeline: both models share preprocessing, split sizes 560/140, stratification preserved, scaler fitted on train only, unknown categories encoded rather than raising
- metrics endpoint equals the joblib model recomputed in-process; comparison winner really has the higher Yes-F1; confusion cells sum to the 140-row test split
- `/predict`: verdict, risk %, per-model risk, K neighbours with real distances, ₹ cost heuristic; unknown district ("Pune is not a district of Karnataka"), negative budget, K=0/K=32 → 400 with field names
- risk % moves with budget and duration; a training row predicted at K=1 returns its own label at distance ≈ 0
- `/dataset`, `/dashboard`, `/metrics`, `/sources`, `/pipeline`, `/sample.csv`, `/health`: shapes, clamps, sort whitelist, and every aggregate recomputed straight from the CSV
- SSE: `EventBroker` unit tests plus a real `/stream` read of the hello chunk, a broadcast reaching a subscriber, and no orphaned subscriber after close
- retrain guardrails: 400 for a non-CSV, 422 for too few rows or a single class, the uploaded dataset surviving a simulated restart, and a reset restoring metrics identical to the shipped ones (asserted field by field)
- `HttpStore` against a local file-backed stub; `S3Store` construction refused without env vars, and `put()` refused without `ALLOW_WRITES`
- currency: lakh/crore thresholds at 99.99 / 100 / 100000, and no `$` in any API payload
- route engine: the corridor coefficient really being that median, all 17 calibration corridors inside 16 %, `Σ components = works` and `works × 1.18 = total`, forced ₹/km overriding the take-off while `takeoff_*` survives, the three km/day bands with months equal to the speed their own `basis` prints, hilly terrain forcing the 0.5–1 band regardless of cross-section, urban charging 1.15 on both money and clock, construction runtime strictly below the contract programme, risk grade = the sum of the weights it prints, and the brief's `{distance, budget, time_months, risk}` contract
- planning catalogue: the CSV regenerating byte-stable from the seed, 504 rows × the 17 briefed columns, every taluka/village parent valid, connectors inside the 4–55 km offset window and equal to Haversine × the corridor coefficient, `?q=` matching at all four levels, an ambiguous leaf name refused with the taluka named, and the real/placeholder provenance split asserted per column
- planning engine: base rates and both factor tables equal to the brief's numbers, `Σ materials = total`, the 25/20/15/20/20 shares exact, the threshold band flipping at ₹200 and ₹500 crore, width factor clamped to its stated range, and every `planning` sub-key re-derived from the printed `basis` string
- building engine: five-factor product equal to `cost`, ₹/sq ft by location type, floor factors 1.00/1.90/2.65, the 7-work split summing to 100 % and to the printed total with the residual in the largest line, time = area ÷ rate + 120 mobilisation days, and area/floor bounds rejected by field name
- deploy contract (F20): importing `app` without `app.run()` leaves both pipelines loaded and `/health` green; templates and the three map/dashboard scripts resolve from `__file__` after `chdir` to an unrelated directory; `requirements.txt` names Flask, Werkzeug, gunicorn, pandas, numpy, scikit-learn, scipy, joblib and not pytest/Playwright, **and an `ast` audit of every shipped module proves there is no third-party import the build would not install** (`boto3` excepted as the documented lazy S3-only import), with `scikit-learn==1.8.0` matching the pickles on disk; `gunicorn.conf.py` parsed the way gunicorn parses it binds `0.0.0.0:$PORT` at both 10000 and 20500, runs `gthread`/1 worker/`preload_app=False`/stdout logs; `Procfile` and `render.yaml` carry the two documented commands and `/health`; and with an `/stream` connection held on a real threaded HTTP server, four page loads and a prediction each finish in under 2 s and that prediction is broadcast on the held stream

**Playwright smoke test** (`smoke_test.py`) — 332 checks, exits non-zero on any failure, writes 26 screenshots.

- all 10 sections present and visible after the scroll-reveal has fired; no startup error banner; the nav links cover every section and carry exactly one `/building` link
- team, guide and DBATU in hero, Team cards and footer **and on the `/building` hero**; the simulated-data and FY 2025-26 labels present
- the string `$` appears nowhere in `document.body.innerText`; ≥30 ₹ signs do (≥12 on `/building`); no leftover Thai-paper wording
- all 13 Chart.js charts on the main page registered with data (labels read back out of `Chart.instances`) and all 14 canvases holding painted pixels (the 13 charts plus the KNN demo canvas), plus `/building`'s own pie
- hero counters settle at 700 projects / 14 states / 4 schemes; feature-store manifest shows 700 rows in 4 partitions
- dataset explorer: search, outcome filter, clear, pagination, and both directions of a numeric sort checked against the API rows
- predictor: example fill, verdict, gauge geometry matching the printed risk %, ₹ cost lines, K=12 vote and neighbour table, model switch changing the explanation's mechanism, and field-level validation errors
- exports: PDF (`%PDF` header, credits, `INR` because Helvetica has no rupee glyph) and CSV (credits, verdict, ₹)
- dashboard: KPI totals, ten worst overruns, manifest, active source line, SSE connected, feed growing across three predictions with no duplicates
- KNN demo: real mouse drag changing the vote, K slider, new sample, reset
- retrain guardrail: the 20-row sample refused with "at least 30 rows", models untouched, predictor still working afterwards
- dark mode class + rendered body colour + `aria-pressed`; no horizontal overflow at 1280 px or 375 px; predictor usable at 375 px
- route planner: dropdowns Indian-only **and carrying readable option values** (a `fillSelect` key mismatch had silently blanked the terrain and soil lists), corridor distance against the published figure (Mumbai→Pune 138.3 km vs 149), component table adding to the ₹ card, ghat uplift inside 20–30 % and a plain corridor at exactly 1.00 with zero impact, forcing ₹ 2 crore/km making the budget distance × rate with the take-off still printed beside it, the completion card matching the months the API returned and staying under the contract programme, a rural corridor reporting the 1–2 km/day band, selecting urban charging 1.15 on cost and 1.15 on time, the cross-check gauge filled to the percentage it displays, a 9 000 km DPR length rejected with a field message and the stale card cleared, swap reversing the corridor, and — with the response parked — the skeleton/spinner showing and submit disabled
- zero console errors, zero uncaught page errors, zero 5xx or failed requests, and every 4xx traceable to the deliberate bad-input probes (OpenStreetMap/CDN tile failures are excused with attribution: a tile 404 is not our bug, and the page shows its offline fallback instead)

---

## 10. Acceptance criteria

1. App starts with no errors via the single run command
2. `india_projects_2025_26.csv` has 700 rows, 42 % ±3 % over budget, all 10 columns
3. `sample_20_projects.csv` exists with 20 readable Indian rows and loads through the storage adapter
4. Preprocessing is `ColumnTransformer` (one-hot + scaler); KNN and RandomForest both trained on the same 80/20 stratified split with a fixed seed
5. Model comparison shows KNN vs RF on accuracy, precision, recall, F1, AUC and names a winner by Yes-F1
6. Predictor accepts all seven Indian inputs, validates bad ones, and returns verdict + risk % + predicted ₹ cost
7. Dashboard shows Budget-vs-Actual and state-wise distribution, both driven by real CSV aggregates
8. Live SSE feed updates without a page reload
9. Every amount is ₹ (lakh or crore); `$` appears nowhere in the UI, API or exports
10. Tricolour accents are subtle and body text keeps accessible contrast; dark/light works
11. Storage adapter runs on local by default and the S3 path is reachable only via env vars, with no credentials committed
12. Team, guide and DBATU appear in hero, Team section, footer, README and PDF export
13. pytest and the Playwright smoke test both pass; screenshots regenerated
14. README documents setup, synthetic-data provenance, schema, model comparison, architecture, migration path, route-planner method and assumptions
15. Route planner: Nashik→Pune estimates without an error, the distance lands within 16 % of the published corridor, the components add to the printed total, the completion card is the printed distance divided by the printed km/day speed, and no paid API or key is involved
16. Location mode: State → District → Taluka → Village cascades on both ends from `GET /locations`, a searched village resolves its own parents, an ambiguous leaf name is refused with the taluka named, the placeholder-name provenance stays visible next to the dropdowns, and city mode still works unchanged
17. Planning panel: the brief's rate/km × terrain × soil total is shown with the multipliers it used, its five material rows sum to its own total, and the gap to the bottom-up take-off is printed rather than averaged away
18. `/building`: a separate page and endpoint that costs `area × ₹/sq ft × factors`, returns `{cost, time_months, breakdown}`, sums its seven work lines to the card, shows no risk percentage, and leaves every pre-existing route and response untouched
19. Deployable (F20): `pip install -r requirements.txt` then `gunicorn app:app` on a host that injects `$PORT` — proven locally by the WSGI import serving every route, by the parsed gunicorn config binding `0.0.0.0:$PORT`, by an `ast` audit that the build installs every import the code makes (the pushed-file gap behind Render's `Exited with status 127`), and by a held `/stream` not starving page loads; the platform's own build log is the remaining, honestly unverified step

---

## 11. Assumptions

1. Dataset is simulated (§3.5) — real 2025-26 project-level actuals are not publicly available as one clean file.
2. Scheme mix, overrun base rates and the ~42 % target are our design choices informed by general CAG audit findings, not measured statistics.
3. Lakh/crore: 1 crore = 100 lakh; all stored values in ₹ lakh to one decimal.
4. Districts are capped at 59 so district-level one-hot stays meaningful; real Indian project data covers far more.
5. `project_type` excluded from features as collinear with `scheme_name` (§3.1).
6. RandomForest in place of XGBoost, for binary-compatibility and dependency reasons (§4.2).
7. Probabilities are model-native, not calibrated (no isotonic/Platt step at this scope).
8. No Spark/JVM/pyarrow/boto3 installed here; the S3 and Spark paths are code-complete but unexercised, and the README says exactly that rather than implying a live cluster.
9. `config.py` reads env vars; `.env` is not introduced (no new dependency, students can set variables in the shell).
10. v1 remains in `archive/thai-paper-version/` for history and is not loaded by anything.
11. Route terrain/soil/rate figures (§12) are published FY 2025-26 market assumptions, not measured from a tender, and the response says so in `basis`.
12. One corridor is priced at one uniform cross-section: land cost, structures, utility shifting and state-wise rate variation are out of scope.
13. Completion time (§12.3) is a stated km/day band over calendar days at 30.44 days/month — no monsoon shutdown, no acquisition phasing, no critical path — and is a different quantity from the contract programme months the classifier is fed. Both are printed and labelled.
14. Taluka and village names in the planning catalogue are seeded generated placeholders, not the survey of record, because no free/redistributable source gives coordinates for India's ~6 lakh villages; the district level is real and the difference is disclosed on the page, in the CSV provenance block and in §13.1.
15. Two soil and terrain tables exist on purpose (§13.2): the take-off uses engineering factors, `planning` uses the brief's factors. They are never blended, and a corridor priced by both prints both numbers.
16. `/building` is a rate table, not a learned model — no building-cost training rows exist in the 10-column contract, so the page shows factors and arithmetic and returns no probability (§13.3).
17. Building time adds 120 fixed mobilisation days (set-up, dewatering, scaffolding, approvals) with a 3-month floor; land, boundary wall, furnishing, statutory contingencies and state-wise PST rates are out of scope.
18. The planning dataset is generated into `data/` by `model/planning_data.py` and read back; it is not hand-maintained, so a name change or a new district is a re-run rather than an edit.

---

## 12. Route planner specification (v2.1, extended in v2.2)

Added after the v2 build, in answer to a separate brief: cost a road corridor between two Indian
cities on a map, and show the money itemised like a civil engineer's take-off. v2.2 answered the
same brief's follow-up: the months that corridor takes to build, an urban terrain class, and a
budget-versus-time chart (§12.3, F16).

### 12.1 Distance

`CITIES` = 56 districts (every district the classifier knows except three), each with latitude,
longitude, default terrain and default soil. Haversine on a 6371 km sphere gives the straight line;
`CORRIDOR_FACTOR = 1.151` (the median of published ÷ great-circle over 17 named routes) turns it
into road distance. A typed DPR length overrides it, and `distance_basis` must then say so.
`geodesic_path()` returns 25 interpolated points for the polyline — the drawn line is the great
circle, **not** a surveyed alignment, which the map card states under the tiles.

Tolerance: every calibration corridor within 16 % of its published distance (worst measured 15.3 %,
Ahmedabad→Vadodara). No routing engine, no paid API.

### 12.2 Cost

```
quantity(cross-section, length) × unit rate  →  × terrain factor × soil factor  →  + 18 % overheads
```

Terrains: plain 1.00, urban 1.15, coastal 1.15, hilly / rocky-hilly 1.25, ghat 1.30 (the brief asked for
"ghat +20–30 %"; the top of the band is taken because Nashik and Pune are rock-cut ghats, and the
brief's separate "+15 % urban" is charged on city stretches).
Soils: alluvial 1.00, red 1.05, laterite 1.05, rocky 1.08, black-cotton 1.10, sandy 1.12, clay 1.15.
Road types: state highway 3.75 m, rural two-lane 7.0 m, NH 4-lane 14.0 m, expressway 24.0 m, each
with its own pavement/GSB depths, bitumen kg/m², cement bags/m², steel t/km, labour days/km,
machinery hours/km and drain lines. Rates: cement ₹ 350/bag, steel ₹ 60 000/t, bitumen ₹ 50 000/t,
aggregate ₹ 1 250/t, sand ₹ 900/t, GSB ₹ 640/m³, earthwork ₹ 210/m³, labour ₹ 650/day, machinery
₹ 1 450/h, shoulders ₹ 1.2 lakh/km·m, drainage ₹ 7.5 lakh/line-km.

### 12.3 Completion time

```
months = length ÷ band midpoint × urban drag ÷ 30.44 calendar days
```

`SPEED_KM_PER_DAY` = highway 3–5, rural 1–2, ghat 0.5–1 — the three bands the brief states. A
corridor takes its band from the cross-section (`ROAD_SPEED_BAND`) unless its terrain is ghat, hilly
or rocky, which overrides to the slowest band whatever the width says. `min_months` / `max_months`
are the same division at the band's own fastest and slowest speeds, so the printed range is exactly
the range the assumption allows; `basis` spells the arithmetic out and is used as the card's tooltip.
`URBAN_TIME_FACTOR = 1.15` multiplies the day count for city stretches (diversions, night working,
utility shifting), matching the 1.15 cost factor. `DAYS_PER_MONTH = 365.25 / 12` — calendar days,
no monsoon shutdown, no land-take phasing, no critical path (§11).

Runtime is **not** the contract programme. `planned_duration_months` (`MONTHS_PER_KM` 0.16–0.45 by
cross-section, ×1.35 ghat, ×1.20 hilly, clamped 6–240) is the field `model_input` hands the
classifier, because that is the column the 700 training rows carry. Nashik→Pune is 8 months of
laying inside a 46-month programme; a test asserts runtime < programme so the two cannot drift
together or be confused, and the UI prints both with their names.

### 12.4 Reconciling the two totals

Blank ₹/km → the itemised take-off is the answer. A typed ₹/km → that rate is authoritative
(`total = distance × rate`), the component rows are rescaled and tagged "share rescaled", and
`takeoff_total_crore` / `takeoff_cost_per_km_crore` are returned alongside so the reader can see the
engineer's number the assumption replaced. The forced total also feeds `model_input`, so the
cross-check prices the same corridor the card shows.

### 12.5 Risk, honesty and side effects

`risk_percent` is the printed sum of stated weights (base 12, terrain 6–22, soil excess
(factor − 1) × 60, complexity 3–10, length 0/4/8), clamped 5–75 %, banded Low/Medium/High — labelled
as a planning number, never as a probability. The trained model's own probability is a separate
`model_check`, and any corridor above ₹ 227.80 crore (the largest sanctioned project in the training
rows) is reported with `within_training_range: false` and an extrapolation note. A cross-check is a
genuine prediction, so it publishes an SSE event and advances `predictions_served`; a test asserts
that instead of hiding it.

---

## 13. Location hierarchy, planning formula and the building module (v3)

### 13.1 `model/planning_data.py` — the four-level catalogue (F17)

Generated, seeded and re-derivable: 14 states × 4 districts × 3 talukas × 3 villages = **504
rows**, written to `data/india_road_project_planning_dataset_2025_26.csv` with exactly the 17
columns the brief lists (`state … machinery_cost`). Column provenance is declared per field in
`PROVENANCE` and shipped in every `/locations` response:

| Field | Source |
|---|---|
| state, district, lat, lng, terrain, soil_type | real district-centre values — these are what the distance and factor models consume |
| taluka, village | generated placeholders seated 4.0–55.0 km (`MIN/MAX_OFFSET_KM`) from the district centre, seeded, name-plausible, **labelled as placeholders on screen** |
| distance_km, budget, *_cost, time_months | derived per row: Haversine from the district centre × 1.1510, then priced by `model/planning.py`'s top-down formula and its 25/20/15/20/20 split, in ₹ lakh |
| road_type | drawn per row from four cross-section classes by `ROAD_WEIGHTS` — rural two-lane 46 %, state highway 28 %, NH 4-lane 20 %, expressway 6 % |

Cascade rules, all browser-tested: state → districts of that state only (and lower levels cleared),
district → talukas, taluka → villages; naming a village fills its own taluka and district upward;
the state select is not `required` because a searched village supplies the whole chain. `/locations`
with `?q=` searches all four levels and returns flat suggestions carrying `level`, which is how the
single search box can resolve "Pimpri" to a district *and* a village. Ambiguity is refused, never
guessed — a leaf name occurring twice under the chosen district raises *"'X' is in more than one
taluka of Y — choose the taluka as well"* (`planning_data.py:270`).

`/predict-route` accepts the brief's nested body alongside the old flat one:

```json
{"start": {"state": "Maharashtra", "district": "Pune", "taluka": "Mulshi", "village": "Ghoti"},
 "end":   {"state": "Maharashtra", "district": "Nashik"},
 "road_type": "national_highway_4", "terrain": "hilly"}
```

The resolved endpoint is echoed as `start_point` / `end_point` (with `connector_km`,
`connector_budget_lakh`, `connector_time_months`, `connector_risk_band` at village level) so a map
popup can show *location name + budget + time* without a second request. Existing city-mode callers
are unaffected: the response still carries exactly the same five top-level keys.

### 13.2 `model/planning.py` — the brief's formula, kept visibly separate (F18)

```
total = distance_km × base_crore_per_km × width_factor × terrain_factor × soil_factor
```

`BASE_COST_PER_KM_CRORE` rural 1.0 / highway 2.0 / expressway 5.0; terrain plain 1.00, urban 1.15,
coastal 1.15, hilly 1.25, rocky-hilly 1.25, **ghat 1.30**; soil rocky 1.20, clay 1.10, black-cotton
1.10, sandy 1.05, laterite 1.05, red 1.05, alluvial 1.00 — the brief's uplifts (rocky +20 %,
clay +10 %, sandy +5 %) with the remaining soils carried at their engineering values, not the
take-off's table. Materials are the fixed 25 % cement / 20 % steel / 15 %
bitumen / 20 % labour / 20 % machinery, and risk is the size threshold (>₹500 crore High, ₹200–500
Medium, <₹200 Low). Speed comes from the same three bands (highway 4, rural 2, ghat 1 km/day).

Two decisions worth defending in a viva:

1. **It is attached, not merged.** The engine returns a `planning` object on `/predict-route`, so
   the pre-existing contract (`distance`, `budget`, `time_months`, `risk`, `breakdown`) is
   byte-identical and every older test still passes. The take-off remains the headline.
2. **`materials` is an ordered list, not a dict.** Flask sorts JSON keys, which would have printed
   `bitumen, cement, labor, machinery, steel` and made the brief's 25-20-15-20-20 order invisible;
   a list keeps the briefed sequence on the page. `planning.time_months` and the round-trip check
   that the printed `basis` string reproduces the printed total are both asserted in tests.

Because the soil tables differ between `route_engine` (measured engineering factors) and `planning`
(the brief's factors) on purpose, the panel prints both totals and the difference between them
rather than a single reconciled figure — see §11 assumption 17 and README §12.

### 13.3 `model/building.py` + `/building` — additive second estimator (F19)

```
cost  = area_sqft × base_per_sqft × floor_factor × location_factor × quality_factor
time  = area_sqft ÷ sqft_per_day(quality) + 120 mobilisation days, ÷ 30.44
```

`BASE_COST_PER_SQFT` rural 1 500 / urban 2 500 / remote 1 800; floors 1.00 / 1.90 / 2.65;
location rural 1.00 / urban 1.00 / remote 1.10; quality low 0.85 / medium 1.00 / high 1.25;
sq ft per day 55 / 45 / 35 by quality (better finishing crews are slower per m²); 120 fixed
mobilisation days with a 3-month floor. Types are Gram Panchayat, school and hospital, and the
seven work sections split 42 structure / 12 masonry / 15 finishes / 11 electrical-plumbing /
12 labour / 6 fixtures / 2 site development — rounding residual lands in the largest line so the
printed rows add to the printed ₹ lakh total exactly.

Isolation is enforced, not promised: `templates/building.html` and `static/js/building.js` are
separate files, the only shared surface is read-only `/locations`, `window.App` is asserted
undefined on that page, `POST /predict-building` is the only new mutating endpoint, and no existing
route or response changed. It answers the brief's `{cost, time_months, breakdown}` and deliberately
returns **no** risk percentage — the classifier's 10-column contract has no area, floors or quality
column, so a probability here would be invented (N1).
