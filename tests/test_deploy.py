"""Deployment contract tests -- what has to be true before `gunicorn app:app` can work.

Render runs Linux and starts the app with `gunicorn app:app`, which imports this
project instead of executing `python app.py`. That single difference is what these
tests guard:

* a WSGI import must leave the models loaded (no `app.run()` in sight),
* templates and static files must resolve from the project directory, not the
  working directory the server happens to start in,
* the port the platform injects through `$PORT` must be the port gunicorn binds,
* `/stream` must not be able to starve every other request,
* `requirements.txt` must install what the deploy needs and nothing it does not.

gunicorn itself is not exercised here: it is fork-based and cannot start on Windows.
The start command is verified by reading the config it would load and by serving the
same WSGI callable through a threaded HTTP server.
"""

import json
import os
import socket
import threading
import time
import urllib.request

import pytest
from werkzeug.serving import make_server

import app as app_module
from app import app as flask_app

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_root_file(name):
    with open(os.path.join(PROJECT_ROOT, name), encoding="utf-8") as handle:
        return handle.read()


def requirements_names(text):
    """Requirement names from a pip file, comments and options dropped."""
    found = []
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            found.append(line.split("==")[0].strip().lower())
    return found


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# ---------------------------------------------------------------------------
# A real HTTP server, because that is the thing a platform talks to
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_server():
    port = free_port()
    server = make_server("127.0.0.1", port, flask_app, threaded=True)
    server.daemon_threads = True          # a held /stream must not keep pytest alive
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.base_url = "http://127.0.0.1:%d" % port
    yield server
    server.shutdown()


# ---------------------------------------------------------------------------
# 1. the WSGI import path
# ---------------------------------------------------------------------------


def test_importing_the_module_without_app_run_loads_both_models(client):
    """`gunicorn app:app` never calls main(), so bootstrap() runs at import instead.

    If that call were missing, SERVICE would still be empty and every page below would
    answer 500 -- which is exactly the failure this test exists to catch.
    """
    body = client.get("/health").get_json()
    assert body["ok"] is True
    assert body["rows"] == len(app_module.SERVICE.frame) > 0
    assert sorted(body["models_loaded"]) == ["knn", "rf"]
    assert body["last_error"] is None


@pytest.mark.parametrize("path", ["/", "/building"])
def test_pages_render_over_wsgi_without_the_dev_server(path, client):
    response = client.get(path)
    assert response.status_code == 200
    assert "Budget Prediction" in response.get_data(as_text=True)


def test_templates_and_static_follow_the_project_not_the_cwd(client, tmp_path):
    """Render starts gunicorn from the repository root; a laptop does not have to.

    Paths are built from ``__file__``, so changing the working directory must not move
    a template or a stylesheet. This is the ModuleNotFoundError / template-not-found
    class of deploy bug, checked rather than assumed.
    """
    previous = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        page = client.get("/")
        assert page.status_code == 200
        assert app_module.app.root_path == PROJECT_ROOT
        for asset in ("/static/css/styles.css", "/static/js/app.js", "/static/js/routes.js"):
            response = client.get(asset)
            assert response.status_code == 200, asset
            assert len(response.get_data()) > 1000, asset
    finally:
        os.chdir(previous)


# ---------------------------------------------------------------------------
# 2. requirements
# ---------------------------------------------------------------------------


def test_runtime_requirements_cover_the_deploy_and_nothing_more():
    names = requirements_names(read_root_file("requirements.txt"))
    for wanted in ("flask", "werkzeug", "pandas", "numpy", "gunicorn", "scikit-learn",
                   "scipy", "joblib"):
        assert wanted in names, "%s is missing from requirements.txt" % wanted
    # A build machine must not download a browser it will never open.
    assert "pytest" not in names and "playwright" not in names


def test_sklearn_stays_pinned_to_the_version_that_picked_the_models():
    """model/india_knn.joblib is a pickle of scikit-learn's classes.

    Load it on a server with a different scikit-learn and the import of the pipeline
    fails, so the pin is a deploy requirement rather than a style preference.
    """
    text = read_root_file("requirements.txt")
    assert "scikit-learn==1.8.0" in text
    import sklearn
    assert sklearn.__version__ == "1.8.0"


def test_dev_requirements_extend_the_runtime_ones():
    text = read_root_file("requirements-dev.txt")
    assert "-r requirements.txt" in text
    names = requirements_names(text)
    assert "pytest" in names and "playwright" in names


# Import name -> distribution name, for the ones that differ.
DISTRIBUTION_FOR_IMPORT = {"sklearn": "scikit-learn"}

# Imported by code the deploy never runs unless it is configured to:
# data_layer/storage.py imports boto3 lazily, after it has been told to use S3.
OPTIONAL_IMPORTS = {"boto3"}


def project_python_files():
    """Every .py file the shipped app can run, tests and caches excluded."""
    skip_dirs = {"tests", "archive", "__pycache__", ".venv", "venv", ".git",
                 ".pytest_cache", "node_modules", "%TEMP%"}
    for dirpath, dirnames, filenames in os.walk(PROJECT_ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in skip_dirs and not d.startswith(".scratch")]
        for name in filenames:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def third_party_imports():
    """Top-level import names in the app that are not stdlib and not this project."""
    import ast
    import sys

    local = {os.path.splitext(os.path.basename(p))[0] for p in
             os.listdir(PROJECT_ROOT) if p.endswith(".py")}
    local |= {d for d in os.listdir(PROJECT_ROOT)
              if os.path.isdir(os.path.join(PROJECT_ROOT, d)) and not d.startswith(".")}

    found = set()
    for path in project_python_files():
        tree = ast.parse(read_file(path), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]      # relative imports have module=None/level>0
                if node.level:
                    continue
            else:
                continue
            for top in names:
                root = top.split(".")[0]
                if not root or root in local or root in sys.stdlib_module_names:
                    continue
                found.add(root)
    # Anything imported from inside a function body still counts: pip does not know
    # where the import statement sits.
    assert found, "the audit found no third-party import at all -- it is broken"
    return found


def read_file(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def test_every_third_party_import_is_declared_in_requirements():
    """`ModuleNotFoundError` on Render is a missing line in requirements.txt.

    Rather than trusting the list, this parses every shipped module, takes the
    top-level import names, drops the standard library and this project's own
    packages, and requires the rest to be pinned for the build.
    """
    declared = requirements_names(read_root_file("requirements.txt"))
    missing = sorted(i for i in third_party_imports() - OPTIONAL_IMPORTS
                     if DISTRIBUTION_FOR_IMPORT.get(i, i).lower() not in declared)
    assert not missing, "imported but not installed by the build: %s" % missing


def test_werkzeug_is_pinned_because_app_py_imports_it_directly():
    """Flask only guarantees `werkzeug>=3.1.0`; app.py uses its exception classes."""
    assert "werkzeug" in third_party_imports()
    assert "werkzeug" in requirements_names(read_root_file("requirements.txt"))


# ---------------------------------------------------------------------------
# 3. gunicorn's own configuration
# ---------------------------------------------------------------------------


def load_gunicorn_conf(monkeypatch, port="10000"):
    """exec() the file the way gunicorn does, with a fresh environment."""
    monkeypatch.setenv("PORT", port)
    for name in ("GUNICORN_BIND", "WEB_CONCURRENCY", "GUNICORN_THREADS", "GUNICORN_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    namespace = {"__file__": os.path.join(PROJECT_ROOT, "gunicorn.conf.py")}
    source = read_root_file("gunicorn.conf.py")
    exec(compile(source, "gunicorn.conf.py", "exec"), namespace, namespace)
    return namespace


def test_gunicorn_binds_the_port_the_platform_injects(monkeypatch):
    conf = load_gunicorn_conf(monkeypatch, "10000")
    assert conf["bind"] == "0.0.0.0:10000"
    # Render documents its default as 10000; any other value must still be honoured.
    conf = load_gunicorn_conf(monkeypatch, "20500")
    assert conf["bind"] == "0.0.0.0:20500"


def test_gunicorn_config_avoids_the_two_ways_this_app_could_freeze(monkeypatch):
    conf = load_gunicorn_conf(monkeypatch)
    # (a) a sync worker serves one request at a time and /stream is a long-lived one,
    # (b) more than one process splits the in-process SSE broker.
    assert conf["worker_class"] == "gthread"
    assert conf["threads"] >= 4
    assert conf["workers"] == 1
    assert conf["preload_app"] is False
    assert conf["timeout"] >= 60, "a cold start trains nothing but does read 1.5 MB of pickles"
    assert conf["accesslog"] == "-" and conf["errorlog"] == "-", "logs go to stdout, not a file"


def test_procfile_and_blueprint_agree_with_the_documented_commands():
    assert read_root_file("Procfile").strip() == "web: gunicorn app:app"
    blueprint = read_root_file("render.yaml")
    assert 'buildCommand: "pip install -r requirements.txt"' in blueprint
    assert 'startCommand: "gunicorn app:app"' in blueprint
    assert 'healthCheckPath: "/health"' in blueprint


def test_deploy_files_are_present_and_the_junk_is_ignored():
    for name in ("Procfile", "gunicorn.conf.py", "runtime.txt", ".python-version",
                 "render.yaml", "requirements-dev.txt", ".gitignore"):
        assert os.path.exists(os.path.join(PROJECT_ROOT, name)), name
    ignored = read_root_file(".gitignore")
    for pattern in ("__pycache__/", "*.py[cod]", ".pytest_cache/", "dev_server.log",
                    "%TEMP%/", "data/retrain_active.csv"):
        assert pattern in ignored, "%s is not ignored" % pattern


# ---------------------------------------------------------------------------
# 4. the concurrency a dashboard needs
# ---------------------------------------------------------------------------


def test_held_event_stream_does_not_block_other_requests(live_server):
    """One open dashboard tab must not stop everyone else from using the site.

    This is what a `sync` gunicorn worker would do, and what gthread plus the SSE
    broker is meant to prevent, so it is measured instead of asserted in a comment.
    """
    events = []
    stop = threading.Event()

    def hold():
        try:
            with urllib.request.urlopen(live_server.base_url + "/stream", timeout=20) as resp:
                while not stop.is_set():
                    line = resp.readline()
                    if not line:
                        break
                    events.append(line.decode("utf-8", "replace").strip())
                    if len(events) > 25:
                        break
        except Exception:                       # noqa: BLE001 -- the tab "disconnects"
            events.append("closed")

    reader = threading.Thread(target=hold, daemon=True)
    reader.start()
    time.sleep(1.0)
    assert "event: hello" in events, events[:3]

    timings = []
    for _ in range(4):
        started = time.time()
        with urllib.request.urlopen(live_server.base_url + "/", timeout=10) as resp:
            page = resp.read()
        timings.append((time.time() - started, len(page)))
    assert all(size > 20000 for _, size in timings), timings
    assert max(t for t, _ in timings) < 2.0, "SSE starved the worker: %s" % timings

    payload = json.dumps({
        "scheme_name": app_module.SERVICE.meta["vocabularies"]["scheme_name"][0],
        "state": "Maharashtra", "district": "Pune", "contractor_type": "Govt",
        "sanctioned_budget_lakh": 250, "duration_months": 18}).encode("utf-8")
    request = urllib.request.Request(live_server.base_url + "/predict", data=payload,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
    started = time.time()
    with urllib.request.urlopen(request, timeout=15) as resp:
        assert json.loads(resp.read().decode("utf-8"))["prediction"] is not None
    assert time.time() - started < 2.0

    time.sleep(1.0)
    stop.set()
    assert "event: prediction" in events, "the live feed missed the prediction: %s" % events[:8]


def test_health_endpoint_answers_for_a_platform_probe(live_server):
    with urllib.request.urlopen(live_server.base_url + "/health", timeout=10) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    assert resp.status == 200 and body["ok"] is True
