"""Gunicorn configuration -- read automatically by a plain ``gunicorn app:app``.

Gunicorn looks for ``./gunicorn.conf.py`` in the directory it is started from, so the
Render Start Command stays exactly ``gunicorn app:app`` and this file supplies the three
things that command would otherwise get wrong for this particular app:

* **bind** -- gunicorn's default is ``127.0.0.1:8000``. Render routes the public URL to
  the port it hands out in ``$PORT`` (10000 by default), so binding the default leaves the
  service reachable only from itself and the deploy fails its health check.
* **worker_class** -- the default ``sync`` worker serves one request at a time. ``/stream``
  is Server-Sent Events: it stays open for the life of the tab that opened it, so a single
  dashboard viewer would occupy the only worker and every other request -- including the
  page that is waiting for it -- would queue behind it. ``gthread`` keeps those long-lived
  connections on their own threads.
* **workers = 1** -- deliberate. Each worker loads two scikit-learn pipelines and a 700-row
  frame, and the free instance has 512 MB. It also keeps ``EventBroker`` honest: the SSE
  broker is in-process, so with more than one worker a prediction handled by worker A would
  never reach a dashboard subscribed to worker B. One worker means the live feed shows every
  prediction the service makes, which is what the page claims it does.

See the "Deploying" section of README.md for what else changes once this runs off-laptop
(ephemeral disk, cold starts, no writes to the repo).
"""

import os

# --- bind -------------------------------------------------------------------
PORT = int(os.environ.get("PORT", "10000"))
bind = os.environ.get("GUNICORN_BIND", "0.0.0.0:%d" % PORT)

# --- workers ----------------------------------------------------------------
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
worker_class = "gthread"
# One dashboard tab holds /stream open indefinitely; the page also fires a couple of
# parallel JSON calls on load, so 8 threads leaves room on a 512 MB instance.
threads = int(os.environ.get("GUNICORN_THREADS", "8"))

# --- timeouts ---------------------------------------------------------------
# Cold start trains nothing but does load two joblib pipelines, so the first request
# after a spin-down is given room. A stuck request is killed after this many seconds;
# an SSE connection that is simply idle is not, because the stream writes a heartbeat
# of its own every 15 s (see /stream) and the browser's EventSource reconnects by
# itself if the server ever does drop it.
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
graceful_timeout = 30
keep_alive = 5

# --- logging ----------------------------------------------------------------
# Render shows stdout/stderr in its log view; "-" means exactly that.
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("GUNICORN_LOG_LEVEL", "info")

# --- process ----------------------------------------------------------------
proc_name = "budget-prediction"
# Left at the default (False) on purpose: preloading imports app.py before the fork, and
# numpy/scikit-learn threads surviving a fork is a class of bug this project does not need.
preload_app = False
