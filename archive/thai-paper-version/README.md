# Government Construction Project Budget Prediction Using Machine Learning (KNN)

A single-page interactive website that reimplements the study

> Kusonkhum, P., et al. (2022). *Government Construction Project Budget Prediction Using
> Machine Learning.* Journal of Advances in Information Technology, Vol. 13, No. 1.

Thai government construction projects regularly spend more than they were allocated —
around **US$1.5 million is lost every year**. The paper predicts, from four categorical
project attributes, whether a project will go **OVER BUDGET** using **k-Nearest
Neighbours**. This project rebuilds that pipeline and lets you explore, predict and
evaluate it in the browser.

| | |
|---|---|
| **Developed by** | Vishal Raghunath Chatur, Mohsin Jabiulla Balgyar |
| **Guide** | Tapase Miss |
| **Programme** | B.Tech in Artificial Intelligence & Data Science Engineering |
| **Institution** | Dr. Babasaheb Ambedkar Technological University (DBATU) |
| **Type** | Mini Project |

---

## 1. Quick start

Requirements: Python 3.10+ (tested on 3.14) and an internet connection for the three
CDN libraries (Tailwind, Chart.js, jsPDF).

```bash
pip install -r requirements.txt
python app.py
```

then open <http://127.0.0.1:5000>.

On Windows, if `python` resolves to the Microsoft Store stub, use the launcher instead:

```bash
py app.py
```

The first start trains the model automatically if `model/knn_model.joblib` is missing,
so a fresh clone works with the single command above. To train explicitly:

```bash
python model/generate_dataset.py   # recreate data/projects.csv (deterministic)
python model/train.py              # fit the pipeline and write the metrics JSON
```

### Running the tests

```bash
pytest                      # 113 unit + API tests
python tests/smoke_test.py  # headless Chromium: 81 checks + screenshots/
```

The smoke test starts its own server on a free port, so nothing else needs to be running.
It writes per-section screenshots (desktop and mobile) into `screenshots/`.

---

## 2. The dataset — please read this first

**The original 692-project dataset is not public.** `data/projects.csv` is therefore
**synthetic data simulating the paper's dataset**. It reproduces the published marginal
distributions exactly, but every individual row is fictitious. The footer, the hero and
`/meta` all say so.

Reproduced exactly from the paper:

| Quantity | Value |
|---|---|
| Projects (2019, Comptroller General's Department, Thailand, e-GP) | 692 |
| Over budget | 143 (20.66 %) |
| Within budget | 549 (79.34 %) |
| Procurement = Specific | 601 (86.85 %) |
| Procurement = Bidding | 48 (6.94 %) |
| Procurement = Chosen | 43 (6.21 %) |
| Departments / site locations | 48 / 72 |

Assumed (the paper gives no breakdown — see §6):

| Quantity | Value |
|---|---|
| Project type Building / Road / Irrigation | 415 / 208 / 69 |

Labels are not sampled at random. Each row gets a risk score built from project type,
department, location and seeded noise; inside each procurement group the highest-scoring
rows are promoted to `Yes` until that group's quota is filled. This guarantees the exact
published counts **and** keeps the paper's headline relationship intact:

| Procurement method | Over-budget rate |
|---|---|
| Bidding | 45.8 % (22 of 48) |
| Chosen | 34.9 % (15 of 43) |
| Specific | 17.6 % (106 of 601) |

`Specific` has the *lowest* rate yet produces the *most* overruns, because it is 86.85 %
of all projects — exactly the governance flaw the paper describes (small projects under
roughly US$16,800 where the contractor is chosen directly instead of through open
bidding).

The generator is deterministic: `python model/generate_dataset.py` always writes a
byte-identical file.

---

## 3. How the model works (viva-friendly)

1. **Inputs.** Four categorical attributes: `department`, `site_location`,
   `project_type`, `procurement_method`. Target: `over_budget` ∈ {Yes, No}.
2. **One-hot encoding.** `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
   turns every category into its own 0/1 column — 48 + 72 + 3 + 3 = **126 columns**.
   `handle_unknown="ignore"` means an unseen category becomes an all-zero row instead of
   raising, which is what lets `/retrain` and `/predict` tolerate new values.
3. **KNN.** `KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="euclidean")`.
   Uniform weights + Euclidean distance = a plain **majority vote** of the k closest
   training projects, exactly as in the paper.
4. **Split.** `train_test_split(test_size=0.20, stratify=y, random_state=42)` →
   **553 train / 139 test**, matching the paper's stated 139 test rows. Stratification
   keeps the 20.66 % Yes share in both halves; the fixed seed makes every number
   reproducible.
5. **Why KNN fits here.** All features are categorical and one-hot encoded, so "distance"
   means "how many attributes differ": two projects that differ in one attribute are
   exactly √2 ≈ 1.41 apart. KNN needs no training phase and is easy to explain to a
   non-technical auditor — a useful property for a governance tool.

The two steps are wrapped in a scikit-learn `Pipeline`, so encoding and prediction can
never drift apart:

```python
Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ("knn",     KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="euclidean")),
])
```

### Live results (measured on the 139 held-out rows — never hard-coded)

| Metric | Value |
|---|---|
| Accuracy | **0.8489** (118 of 139) |
| Confusion matrix (rows = actual, cols = predicted; order No, Yes) | `[[105, 5], [16, 13]]` |
| No — precision / recall / F1 / support | 0.868 / 0.955 / 0.909 / 110 |
| Yes — precision / recall / F1 / support | 0.722 / 0.448 / 0.553 / 29 |
| Best K on the K-curve | 1 (0.8489); K = 5 ties it |

The K-vs-accuracy chart refits the pipeline for every K from 1 to 25 on the same training
split and scores the same test split, so each point is a real measurement.

### Reported in paper vs live

The website shows the published figures **and** ours side by side, clearly labelled. The
paper reports accuracy ≈ 0.86, confusion matrix `[[107, 4], [25, 3]]`, No-class
0.86 / 0.99 / 0.92 (support 117) and Yes-class 0.75 / 0.14 / 0.23 (support 22). Our live
numbers differ because they are measured on synthetic data; the comparison panel states
this neutrally.

---

## 4. Architecture

```
model/generate_dataset.py ──writes──▶ data/projects.csv
                                            │
model/train.py ◀──reads─────────────────────┘
   │  OneHotEncoder + KNeighborsClassifier Pipeline, 80/20 stratified, seed 42
   ├──writes──▶ model/knn_model.joblib      (the fitted pipeline)
   └──writes──▶ model/model_meta.json       (live metrics, vocabularies, K-curve,
                                             reported-in-paper constants)
                                            │
app.py (Flask) ◀──loads at startup──────────┘
   GET  /          single page (templates/index.html)
   GET  /meta      dropdown vocabularies + dataset summary + paper constants
   POST /predict   Pipeline.predict / predict_proba / kneighbors → verdict + vote
   GET  /metrics   live accuracy, report, confusion matrix
   GET  /dataset   searchable, paginated rows + whole-dataset chart aggregates
   GET  /k-curve   accuracy for K = 1..25
   POST /retrain   validate an uploaded CSV → rewrite dataset → retrain → refresh
        │
templates/index.html + static/
   js/app.js       fetches the API, renders sections, predictor, exports
   js/charts.js    Chart.js: dataset + K-curve charts (theme-aware)
   js/knn-demo.js  self-contained draggable 2-D KNN canvas
   css/styles.css  theme tokens, animations, components
```

State lives server-side (joblib + JSON). The browser keeps only the theme preference in
`localStorage`. `/retrain` is the only mutating route; it backs up the dataset and model
first and restores them if training fails, so a bad upload can never leave the site
without a working model. `POST /retrain` with `{"reset": true}` restores the synthetic
dataset.

### Notable implementation decisions

* **Thread-safe K slider.** One fitted pipeline is shared by all requests, but the K
  slider must change `n_neighbors` before predicting. That temporary mutation happens
  inside a `threading.Lock`, so concurrent requests never see each other's K.
* **Neighbours are named, not just counted.** `/predict` re-derives the exact training
  split (same seed, same row order) so it can show *which* past projects voted, with
  their distances.
* **No fabricated numbers.** Every figure on the page arrives from `/metrics`,
  `/dataset` or `/predict`. The only constants are the paper's, kept in
  `model/train.py::REPORTED_IN_PAPER` and always labelled "Reported in paper".
* **Client-side exports.** The PDF (jsPDF) and CSV are built in the browser from the
  `/predict` response, keeping the Python dependency list minimal. Both carry the
  project title, team, guide and institution in their header.

---

## 5. Features

1. **Home** — hero with the problem statement, team and guide, animated counters
   (692 projects · 48 departments · 72 locations · 20.66 % over budget) and an
   Inputs → OneHotEncoder → KNN → Yes/No flow diagram.
2. **Dataset Explorer** — searchable, sortable, paginated table (15 rows/page), four
   charts (target split, procurement method, project type, top-10 departments) and a
   class-imbalance warning.
3. **Predictor** — four dropdowns plus a K slider (1–25). Returns a red OVER BUDGET /
   green WITHIN BUDGET card, the neighbour vote ("8 of 15 neighbours voted Yes"), a
   plain-English explanation, the actual neighbour table, and a corruption-risk warning
   whenever *Specific* is selected. Changing K re-runs the vote.
4. **Model Performance** — live metric tiles, a colour-scaled confusion-matrix heatmap,
   the classification report, the K-vs-accuracy chart, and the "Reported in paper" vs
   "Live model" panel.
5. **How KNN Works** — a draggable test point on a canvas, the K nearest neighbours
   highlighted with distance lines, a live majority vote and the Euclidean formula.
6. **About / Paper** — citation, original authors, methodology, key findings,
   limitations and future scope (SMOTE, Random Forest / XGBoost, more data, a budget
   amount feature).
7. **Team** — cards for both developers and the guide with DBATU details.
8. **Footer** — full credits plus the synthetic-data note.
9. **Extras** — dark/light theme (persisted), PDF and CSV prediction exports, fade-in
   animations, hover lifts, mobile menu, keyboard-accessible controls.

Retraining: upload any CSV with the same five columns (minimum 20 rows, `over_budget`
restricted to Yes/No) and the whole site — dropdowns, charts, metrics, K-curve — is
rebuilt on it.

---

## 6. Assumptions

1. The dataset is synthetic (see §2). Individual rows are fictitious.
2. Project-type mix assumed at 60 / 30 / 10 % (Building / Road / Irrigation); the paper
   publishes no breakdown.
3. Department and location *names* are plausible Thai government departments and
   provinces; the paper publishes only the counts (48 and 72).
4. Default K = 5. The paper does not state which K produced its headline numbers, so K
   is user-adjustable and the K-curve is measured rather than assumed.
5. `test_size=0.2` on 692 rows yields 139 test rows, matching the paper.
6. Live metrics are expected to differ from the published ones; both are always shown.
7. Tailwind, Chart.js and jsPDF load from CDNs, so the browser needs internet. The Flask
   API itself has no network dependency.
8. PDF/CSV export happens client-side.
9. `/retrain` requires exactly the five documented column names; a different schema is
   rejected with HTTP 400 rather than guessed at.

---

## 7. Screenshots

Captured automatically by `tests/smoke_test.py` at 1280 px and 375 px.

| Section | Desktop | Mobile |
|---|---|---|
| Full page | `screenshots/00-full-page-desktop.png` | `screenshots/00-full-page-mobile.png` |
| Home | `screenshots/01-home-desktop.png` | `screenshots/01-home-mobile.png` |
| Dataset Explorer | `screenshots/02-dataset-desktop.png` | `screenshots/02-dataset-mobile.png` |
| Predictor | `screenshots/03-predictor-desktop.png` | `screenshots/03-predictor-mobile.png` |
| Model Performance | `screenshots/04-performance-desktop.png` | `screenshots/04-performance-mobile.png` |
| How KNN Works | `screenshots/05-knn-desktop.png` | `screenshots/05-knn-mobile.png` |
| About the Paper | `screenshots/06-about-desktop.png` | `screenshots/06-about-mobile.png` |
| Team | `screenshots/07-team-desktop.png` | `screenshots/07-team-mobile.png` |
| Footer | `screenshots/08-footer-desktop.png` | `screenshots/08-footer-mobile.png` |
| Dark mode | `screenshots/10-dark-mode.png` | |

---

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `python` prints "Python was not found" | Windows Store alias. Use `py app.py`, or disable the alias in *Settings → Apps → Advanced app settings → App execution aliases*. |
| `ImportError: DLL load failed ... An Application Control policy has blocked this file` | Windows Smart App Control is refusing a compiled NumPy/scipy/scikit-learn binary in the venv. This project pins `scikit-learn==1.8.0` and `scipy==1.17.1`, whose wheels load cleanly; a newer auto-resolved version may not. Reinstall with `pip install -r requirements.txt --force-reinstall`. Do **not** turn Smart App Control off for this — the pins are the fix. |
| Charts or PDF export missing | The CDN scripts could not load (offline). Reconnect and reload; the Flask API still works without them. |
| Port 5000 already in use | `set PORT=5050` (or `PORT=5050` on Linux/macOS) before `python app.py`. |
| A bad upload broke my data | `POST /retrain` with `{"reset": true}` from the browser's *Reset to synthetic dataset* button, or re-run `python model/generate_dataset.py && python model/train.py`. |

---

## 9. Project layout

```
app.py                     Flask app: routes, validation, model serving
requirements.txt           pinned dependencies (see the scikit-learn note)
README.md / SPEC.md / PROGRESS.md
data/projects.csv          692-row synthetic dataset (generated)
model/generate_dataset.py  deterministic synthetic data generator
model/train.py             pipeline, split, evaluation, artifact saving
model/knn_model.joblib     fitted Pipeline (generated)
model/model_meta.json      live metrics + vocabularies + K-curve (generated)
templates/index.html       the single page
static/css/styles.css      theme tokens, animations, components
static/js/app.js           API glue, predictor, exports
static/js/charts.js        Chart.js charts
static/js/knn-demo.js      interactive KNN canvas
tests/conftest.py          shared pytest fixtures
tests/test_model.py        dataset + pipeline + split + metric correctness
tests/test_api.py          route contracts, validation, K-sensitivity, retrain
tests/smoke_test.py        Playwright headless QA + screenshots
screenshots/               generated by the smoke test
```
