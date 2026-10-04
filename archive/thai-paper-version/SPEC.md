# SPEC — Government Construction Project Budget Prediction (KNN)

B.Tech Mini Project · Dept. of Artificial Intelligence & Data Science Engineering, DBATU

| | |
|---|---|
| **Developed by** | Vishal Raghunath Chatur, Mohsin Jabiulla Balgyar |
| **Guide** | Tapase Miss |
| **Based on** | Kusonkhum et al., *"Government Construction Project Budget Prediction Using Machine Learning"*, Journal of Advances in Information Technology, Vol. 13, No. 1, 2022 |
| **Problem** | Thai government construction projects frequently overrun budget (~US$1.5M lost per year). Predict **OVER BUDGET (Yes/No)** from categorical project attributes using **k-Nearest Neighbours**. |

---

## 1. Requirements

### 1.1 Functional

| ID | Requirement |
|---|---|
| F1 | Single-page responsive website with sticky navbar, smooth scroll, 8 sections + footer. |
| F2 | Synthetic dataset of **692 rows** reproducing the paper's distributions, saved to `data/projects.csv`. |
| F3 | KNN model trained through an **`OneHotEncoder` + `KNeighborsClassifier` scikit-learn `Pipeline`**, **80/20 stratified split**, fixed seed → **139 test rows**. |
| F4 | Predictor: 4 dropdowns (Department, Site location, Project type, Procurement method) + K slider (1–25). Returns OVER BUDGET (red) / WITHIN BUDGET (green) card, neighbour vote ("7 of 9 neighbours voted No"), plain-English explanation, and a corruption-risk warning when *Specific* is selected. |
| F5 | Dataset Explorer: searchable + paginated table and 4 charts (over/under budget, procurement method, project type, top departments) plus a class-imbalance warning. |
| F6 | Model Performance: live accuracy / precision / recall / F1, confusion-matrix heatmap, classification report, K-vs-accuracy chart (K = 1..25), and a **"Reported in paper" vs "Live model"** side-by-side panel. |
| F7 | Interactive "How KNN works" canvas: draggable test point, K highlighted neighbours, live majority vote, Euclidean distance formula. |
| F8 | About/Paper section: citation, original authors, methodology, conclusion, limitations, future scope. |
| F9 | Team section: cards for both students and the guide with DBATU details. |
| F10 | CSV upload → `/retrain` retrains the model on the uploaded data and refreshes every metric. |
| F11 | Export a prediction as **PDF** and **CSV**, each headed with project title, team names, guide and DBATU. |
| F12 | Dark/light theme toggle, persisted in `localStorage`. |
| F13 | Animated counters and fade-in-on-scroll animations in the hero/sections. |

### 1.2 Non-functional

| ID | Requirement |
|---|---|
| N1 | Python 3 + Flask + scikit-learn + pandas + joblib. Frontend: HTML + Tailwind (CDN) + vanilla JS + Chart.js (CDN). **No build step.** |
| N2 | Starts with one command: `pip install -r requirements.txt && python app.py`. |
| N3 | **No fabricated metrics.** Every number rendered must come from the real trained model, except values explicitly labelled *"Reported in paper"* which are hard-coded constants from the publication. |
| N4 | Code commented and beginner-readable (must be explainable in a viva). |
| N5 | Accessible contrast, no horizontal overflow at 375 px and 1280 px, no console errors. |
| N6 | Design: modern dashboard, navy + amber accent, rounded cards, soft shadows, Inter font, generous whitespace. |

### 1.3 Environment constraint (discovered during setup)

This machine enforces **Windows Smart App Control**, which blocks `sklearn/utils/_weight_vector.cp314-win_amd64.pyd` shipped in **scikit-learn 1.9.1**. That single blocked binary cascades (`_sgd_fast` → `sklearn.linear_model` → `sklearn.neighbors` / `sklearn.compose`), making `KNeighborsClassifier` unusable.
**Resolution:** pin `scikit-learn==1.8.0` in `requirements.txt` (verified loadable by Smart App Control). Do **not** disable Smart App Control.

---

## 2. File structure

```
constrution Qoo/
├── app.py                      # Flask app: routes, validation, model serving
├── requirements.txt            # pinned deps (scikit-learn==1.8.0 — see §1.3)
├── README.md                   # setup, dataset note, model explanation, assumptions, screenshots
├── PROGRESS.md                 # acceptance-criteria checklist + test/fix rounds
├── SPEC.md                     # this file
├── data/
│   └── projects.csv            # 692-row synthetic dataset (generated)
├── model/
│   ├── generate_dataset.py     # synthetic data generator (paper distributions)
│   ├── train.py                # builds Pipeline, splits, evaluates, saves artifacts
│   ├── knn_model.joblib        # trained Pipeline        (generated)
│   └── model_meta.json         # metrics + categories + reported-in-paper constants (generated)
├── templates/
│   └── index.html              # single page, all 8 sections + footer
├── static/
│   ├── css/styles.css          # theme tokens, animations, component polish
│   └── js/
│       ├── app.js              # nav, theme, counters, reveal, section orchestration, exports
│       ├── charts.js           # Chart.js: dataset + performance charts
│       └── knn-demo.js         # interactive draggable KNN canvas
├── tests/
│   ├── conftest.py             # shared fixtures (app, client, trained artifacts)
│   ├── test_model.py           # dataset + pipeline + split + metrics correctness
│   ├── test_api.py             # route contracts, validation, K-sensitivity
│   └── smoke_test.py           # Playwright headless browser QA + screenshots
└── screenshots/                # generated by smoke_test.py
```

---

## 3. Dataset specification

`data/projects.csv` — 692 rows, 5 columns, UTF-8, comma separated, header row present.

| Column | Type | Values |
|---|---|---|
| `department` | categorical | 48 distinct Thai government departments (skewed, Zipf-like usage) |
| `site_location` | categorical | 72 distinct Thai provinces/cities |
| `project_type` | categorical | `Building`, `Road`, `Irrigation` |
| `procurement_method` | categorical | `Specific`, `Bidding`, `Chosen` |
| `over_budget` | target | `Yes` / `No` |

### 3.1 Target quotas (exact, from the paper)

| Category | Count | Share |
|---|---|---|
| `over_budget = Yes` | **143** | 20.66 % |
| `over_budget = No` | **549** | 79.34 % |
| `procurement_method = Specific` | **601** | 86.85 % |
| `procurement_method = Bidding` | **48** | 6.94 % |
| `procurement_method = Chosen` | **43** | 6.21 % |

### 3.2 Assumed quotas (paper gives no breakdown — see README assumptions)

| Category | Count | Share |
|---|---|---|
| `project_type = Building` | 415 | 60.0 % |
| `project_type = Road` | 208 | 30.1 % |
| `project_type = Irrigation` | 69 | 10.0 % |

### 3.3 Label-assignment rule (deterministic, seed = 42)

To guarantee the published counts *exactly* while keeping a realistic correlation structure,
the generator does not sample labels independently. It works in two stages:

1. **Per-procurement quotas.** The 143 `Yes` rows are split across procurement methods up
   front: Bidding 22, Chosen 15, Specific 106. This is what makes procurement method the
   dominant attribute — the paper's key finding.
2. **Within-group risk ranking.** Every row gets a risk score

   ```
   risk = 0.35 * w_project_type[type]      # Road +0.50, Irrigation +0.40, Building −0.10
        + 0.15 * w_department[dept]        # per-department tendency, N(0,1), seeded
        + 0.10 * w_location[loc]           # per-province tendency, N(0,1), seeded
        + N(0, 0.25)                       # seeded noise
   ```

   and, inside each procurement group, the highest-scoring rows are promoted to `Yes` until
   that group's quota is filled.

Consequences (intended, and asserted by tests):
* Procurement method carries the largest weight → it is the most influential attribute, matching the paper's conclusion.
* `Specific` (small projects under ~US$16,800 with direct contractor choice) has the **lowest** per-row over-budget rate yet supplies the **majority** of `Yes` cases because it is 86.85 % of the data — exactly the corruption-risk dynamic the paper describes.
* Realised conditional rates land near: Bidding ≈ 46 % Yes, Chosen ≈ 35 % Yes, Specific ≈ 18 % Yes.

The generator writes the CSV deterministically: re-running it produces a byte-identical file.

---

## 4. Model specification

```python
Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ("knn",     KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="euclidean")),
])
train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)   # → 553 train / 139 test
```

* All four features are categorical → a bare `OneHotEncoder` (no `ColumnTransformer`) matches the paper.
* `weights="uniform"` + `metric="euclidean"` = plain Euclidean-distance **majority vote**, as in the paper.
* `handle_unknown="ignore"` lets an unseen category be encoded as an all-zero row instead of raising.
* Classes are the strings `"No"` / `"Yes"`; sorted order makes confusion-matrix row 0 = `No`, row 1 = `Yes`, matching the paper's `[[107, 4], [25, 3]]` orientation.
* Artifacts: `model/knn_model.joblib` (the fitted Pipeline) and `model/model_meta.json` (live metrics, per-class report, confusion matrix, K-curve, category vocabularies, split sizes, dataset row hash, `trained_at`).
* `train.py` is idempotent and also runs on server start if artifacts are missing, so a fresh clone works with one command.

### 4.1 Reported-in-paper constants (never recomputed, always labelled)

```
accuracy 0.86 (text)      confusion matrix [[107, 4], [25, 3]]
No  : precision 0.86, recall 0.99, f1 0.92, support 117
Yes : precision 0.75, recall 0.14, f1 0.23, support 22
```

The paper's own numbers are internally inconsistent (that matrix implies 110/139 ≈ 79 % accuracy, and its row sums 111/28 do not match the stated supports 117/22). The UI presents these values **neutrally** as *"as published"*, with no criticism, next to our live figures, and explains that our model is trained on synthetic data so exact agreement is not expected.

---

## 5. API routes

All responses are JSON except `GET /`. Errors use `{"error": "<message>", "details": {...}}` with a proper HTTP status.

### `GET /` → `templates/index.html` (200)

### `GET /meta` → 200
Category vocabularies + dataset summary + reported constants, so the page needs one round-trip to populate every control.
```jsonc
{
  "departments": ["...48..."], "locations": ["...72..."],
  "project_types": ["Building","Road","Irrigation"],
  "procurement_methods": ["Specific","Bidding","Chosen"],
  "dataset": { "rows": 692, "yes": 143, "no": 549, "train_size": 553, "test_size": 139, "source": "synthetic" },
  "reported_in_paper": { "accuracy": 0.86, "confusion_matrix": [[107,4],[25,3]], "no": {...}, "yes": {...} },
  "trained_at": "ISO-8601"
}
```

### `POST /predict` → 200 | 400
Request:
```json
{ "department": "...", "site_location": "...", "project_type": "Road",
  "procurement_method": "Specific", "k": 9 }
```
Validation (→ **400** with a field-level message): missing/blank required field, non-string
value, `k` non-integer or outside 1–25 (booleans are rejected because `True` is an `int` in
Python), malformed JSON body. `k` may be omitted — it then defaults to 5 — and a whole-number
float such as `9.0` is accepted and converted.
Response:
```jsonc
{
  "prediction": "Yes",                       // model output, never hard-coded
  "label": "OVER BUDGET",                    // or "WITHIN BUDGET"
  "probability": { "Yes": 0.78, "No": 0.22 },// from predict_proba
  "k": 9,
  "vote": { "yes": 7, "no": 2, "k": 9, "summary": "7 of 9 neighbours voted Yes" },
  "neighbours": [ { "department": "...", "site_location": "...", "project_type": "...",
                    "procurement_method": "...", "over_budget": "Yes", "distance": 1.41 } ],
  "explanation": "Plain-English sentence built from the real vote and probabilities.",
  "warnings": ["Specific procurement: ... corruption-risk note ..."],
  "unknown_categories": []                   // non-empty only if handle_unknown="ignore" absorbed a new value
}
```

### `GET /metrics` → 200
```jsonc
{
  "accuracy": 0.8x, "best_k": 5, "default_k": 5,
  "test_size": 139, "train_size": 553,
  "confusion_matrix": [[tn_or_nn, fp],[fn,tp]], "labels": ["No","Yes"],
  "report": { "No": {"precision":..,"recall":..,"f1":..,"support":..}, "Yes": {...},
              "macro avg": {...}, "weighted avg": {...} },
  "reported_in_paper": { ... },              // same constants as /meta
  "model": { "pipeline_steps": ["encoder","knn"], "metric": "euclidean",
             "weights": "uniform", "encoder": "OneHotEncoder(handle_unknown='ignore')" }
}
```

### `GET /dataset?page=1&per_page=15&q=&sort=over_budget&order=asc` → 200
```jsonc
{ "rows": [ {...} ], "page": 1, "per_page": 15, "total": 692, "pages": 47, "query": "",
  "aggregates": {
    "over_budget": {"Yes":143,"No":549},
    "procurement_method": {"Specific":601,"Bidding":48,"Chosen":43},
    "project_type": {"Building":415,"Road":208,"Irrigation":69},
    "top_departments": [{"name":"...","count":42,"yes":9}],   // top 10 by count
    "over_budget_rate_by_procurement": {"Specific":0.18,"Bidding":0.46,"Chosen":0.35}
  } }
```
`aggregates` are computed from the **full** dataset (never the current page) and are identical on every page so charts stay stable.

### `GET /k-curve` → 200
```json
{ "ks": [1,...,25], "accuracies": [0.7x,...], "best_k": 5, "best_accuracy": 0.8x, "default_k": 5 }
```
Each accuracy is a real re-fit of the KNN step on the same 553-row training split, scored on the same 139-row test split.

### `POST /retrain` (multipart/form-data, field `file`) → 200 | 400
* **400** when: no file, empty file, non-`.csv`, missing/extra columns vs. the 5 required, fewer than 20 rows, or `over_budget` not confined to `Yes`/`No`.
* On success: validates → writes `data/projects.csv` (previous file backed up to `data/projects.backup.csv`) → retrains → recomputes metrics + K-curve → persists artifacts → returns the new `/metrics` payload plus `{"retrained": true, "rows": N, "source": "uploaded"}`.
* The previous model is restored automatically if training raises, so a bad upload can never leave the app without a working model.

---

## 6. Data flow

```
model/generate_dataset.py  ──writes──▶  data/projects.csv
                                              │
model/train.py  ◀──reads──────────────────────┘
   │  OneHotEncoder + KNeighborsClassifier Pipeline, 80/20 stratified, seed 42
   ├──writes──▶ model/knn_model.joblib
   └──writes──▶ model/model_meta.json  (live metrics, vocab, K-curve, reported constants)
                                              │
app.py (Flask) ◀──loads at startup────────────┘
   │   GET  /meta      → vocabularies + dataset summary + reported constants
   │   POST /predict   → Pipeline.predict / predict_proba / kneighbors
   │   GET  /metrics   → live metrics + reported constants
   │   GET  /dataset   → page + full-dataset aggregates
   │   GET  /k-curve   → refit KNN for K = 1..25 on the stored split
   │   POST /retrain   → validate CSV → rewrite dataset → retrain → refresh artifacts
   ▼
templates/index.html + static/js
   app.js     ── fetch on load ──▶ /meta, /metrics, /dataset, /k-curve
   charts.js  ── renders Chart.js doughnut/bar charts + confusion-matrix heatmap
   knn-demo.js── self-contained 2-D canvas demo (no server calls)
   app.js     ── POST /predict on form submit ──▶ result card, vote, explanation, warnings
   app.js     ── jsPDF / Blob ──▶ prediction PDF + CSV export
```

State is server-side only (joblib + JSON). The browser keeps just the theme preference in `localStorage`. `/retrain` is the sole mutating route.

---

## 7. Frontend sections

| # | `id` | Contents |
|---|---|---|
| 1 | `home` | Hero: title, problem statement (~US$1.5M/yr), team + guide, "Try the Predictor" CTA, 4 animated counters (692 projects · 48 departments · 72 locations · 20.66 % over budget), and an Inputs → OneHotEncoder → KNN → Yes/No flow diagram. |
| 2 | `dataset` | Search box, sortable paginated table (15/page), 4 charts, class-imbalance warning banner. |
| 3 | `predictor` | 4 dropdowns + K slider (1–25, live value bubble), Predict button with spinner, client-side validation, red/green result card, neighbour-vote bar, plain-English explanation, *Specific* corruption-risk warning, PDF + CSV export buttons. |
| 4 | `performance` | Live metric tiles, confusion-matrix heatmap (2×2, colour-scaled, counts inside), classification report table, K-vs-accuracy line chart, "Reported in paper" vs "Live model" side-by-side panel. |
| 5 | `knn-demo` | Canvas with 2 projected feature axes, ~60 seeded sample points, draggable test point, K slider, highlighted K neighbours with distance lines, live majority vote, Euclidean formula rendered. |
| 6 | `about` | Citation, original authors, methodology, conclusion, limitations (class imbalance, `Specific`-method corruption risk, no budget-amount feature), future scope (SMOTE, Random Forest / XGBoost, more data, budget amount). |
| 7 | `team` | Cards: Vishal Raghunath Chatur, Mohsin Jabiulla Balgyar, guide Tapase Miss — each with DBATU / B.Tech AI&DS / Mini Project details. |
| 8 | `footer` | "Mini Project by Vishal Raghunath Chatur & Mohsin Jabiulla Balgyar \| Guide: Tapase Miss \| DBATU \| Synthetic data simulating the paper's dataset". |

Cross-cutting: sticky navbar with active-link highlighting, smooth scroll, dark/light toggle, fade-in-on-scroll via `IntersectionObserver`, hover lifts, mobile hamburger menu, `[hidden]{display:none!important}` guard.

---

## 8. Testing strategy

### 8.1 `tests/test_model.py`
dataset has 692 rows and the exact procurement/target quotas; `Yes` share ≈ 20.66 %; 48 departments / 72 locations / 3 project types; generator determinism (same seed → identical frame); Pipeline has exactly `OneHotEncoder` then `KNeighborsClassifier`; encoder uses `handle_unknown="ignore"`; KNN uses Euclidean + uniform weights; split is 553/139 and stratified; `random_state=42`; retraining reproduces identical metrics; stored `model_meta.json` metrics match a freshly scored model.

### 8.2 `tests/test_api.py`
`GET /` 200 and contains the team names; `/meta` vocabularies complete; **`/predict` with 10 different valid inputs** (spanning every project type, every procurement method, and K ∈ {1, 5, 9, 15, 25}) each returning a consistent prediction/vote/probability; invalid input handling (missing field, blank string, `k=0`, `k=26`, `k="abc"`, wrong type, malformed JSON → 400 with a useful message); **changing K changes the vote breakdown**; `/metrics` values equal the model's own scores and confusion matrix sums to 139; `/dataset` pagination, search filtering and page-invariant aggregates; `/k-curve` returns 25 real accuracies in [0,1] with a valid `best_k`; `/retrain` rejects a bad CSV and accepts a valid one, leaving the app serving a working model afterwards.

### 8.3 `tests/smoke_test.py` (Playwright, headless Chromium)
Boots the real server as a subprocess, then asserts: all 8 section ids + footer present; **zero console errors** and zero page errors; every expected `<canvas>` exists and is non-empty (non-zero pixel data, not just a sized box); the predictor form submits and a result card becomes visible; the neighbour-vote text matches `\d+ of \d+ neighbours voted`; the dark-mode toggle actually flips `document.documentElement.classList` **and** the computed `background-color` of `<body>` (rendered state, not just a class); no horizontal overflow at **375 px** and **1280 px** (`scrollWidth <= clientWidth + 1` on `html` and `body`); team names, guide and DBATU appear in hero, Team section and footer text. Screenshots of each section → `screenshots/`. Exit code non-zero on any failure.

### 8.4 Loop
Phases 3 → 4 → 5 repeat until a full pass yields zero new issues (max 6 rounds), then Phase 6 polish + one final full re-run.

---

## 9. Acceptance criteria

- [ ] App starts with no errors via the single run command
- [ ] `projects.csv` has 692 rows, ~20.66 % over budget, correct category counts
- [ ] `OneHotEncoder` + `KNeighborsClassifier` Pipeline, 80/20 stratified split, fixed seed
- [ ] Predictor works for all dropdown combinations and validates input
- [ ] Changing K updates the prediction and the vote breakdown
- [ ] Confusion matrix, classification report and K-vs-accuracy chart render correctly
- [ ] "Reported in paper" vs "Live" comparison is shown
- [ ] No console errors; responsive at 375 px and 1280 px; dark/light mode works
- [ ] Team names, guide and DBATU details appear in hero, Team section, footer, README and PDF export
- [ ] pytest and the Playwright smoke test both pass
- [ ] README covers setup, dataset note, model explanation, assumptions, screenshots
- [ ] Code is commented and beginner-readable (explainable in viva)

---

## 10. Assumptions

1. The original 692-project dataset is not public, so `data/projects.csv` is **synthetic data simulating the paper's dataset**. It reproduces the published marginal distributions and the qualitative correlation structure, but individual rows are fictitious. Labelled in the footer, README and `/meta`.
2. The paper reports no project-type breakdown → assumed Building 60 % / Road 30 % / Irrigation 10 %.
3. The paper reports no department/location names → 48 plausible Thai government departments and 72 Thai provinces are used.
4. Default `K = 5`; the paper does not state the K used for its headline numbers, so K is user-adjustable (1–25) and the K-curve is measured rather than assumed.
5. `train_test_split(test_size=0.2)` on 692 rows yields 553/139, matching the paper's stated 139 test rows.
6. Live metrics come from the synthetic data and therefore differ from the published values; both are always shown side by side and clearly labelled.
7. Tailwind, Chart.js and jsPDF load from CDNs, so the browser needs internet access. The Flask API itself has no network dependency.
8. PDF/CSV export is generated client-side (jsPDF / Blob) to keep the Python dependency list minimal.
9. `/retrain` requires the same 5 column names; a different schema is rejected with a 400 rather than guessed at.
