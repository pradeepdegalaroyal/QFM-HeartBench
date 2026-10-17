"""Crash-safe task queue on a shared filesystem (works across PBS jobs and nodes).

* Each task has a deterministic id; its result is ``runs/results/<suite>/<id>.json``
  (+ ``.npz``). A task whose result exists is never run again, so any job can
  be killed and resubmitted at any time without losing finished work.
* A worker claims a task by ``mkdir runs/locks/<id>`` (atomic on NFS) and
  refreshes a heartbeat file every 60 s. Locks whose heartbeat is older than
  ``STALE_SECONDS`` (killed job, crashed node) are reclaimed automatically.
* Failures are logged to ``runs/errors/<id>.txt``; a task is retried up to
  ``MAX_ATTEMPTS`` times.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS = os.environ.get("QFMB_RUNS", os.path.join(ROOT, "runs"))
# 48h, not 30 min: node11 on this cluster has been observed with a system
# clock ~30h behind the other nodes (confirmed via `ssh node11 date`), and
# claim() compares a lock's heartbeat mtime (stamped by the owning node's own
# clock) against _server_now() (stamped by whichever node calls claim()). A
# short threshold would let a correctly-clocked node "steal" a lock that a
# skewed node is still actively heartbeating, wasting CPU on duplicate work
# (harmless to correctness -- results are idempotent -- but wasteful). The
# cost of 48h is that a lock orphaned by a genuinely killed job waits longer
# before being retried; acceptable given the multi-day run.
STALE_SECONDS = 172800
MAX_ATTEMPTS = 3


def code_version() -> str:
    h = hashlib.sha1()
    src = os.path.join(ROOT, "src", "qfmb")
    for dirpath, _, files in sorted(os.walk(src)):
        for f in sorted(files):
            if f.endswith(".py"):
                with open(os.path.join(dirpath, f), "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()[:10]


def _p(*parts) -> str:
    return os.path.join(RUNS, *parts)


def result_json(task) -> str:
    return _p("results", task["suite"], task["id"] + ".json")


def is_done(task) -> bool:
    return os.path.exists(result_json(task))


def attempts(task_id: str) -> int:
    try:
        with open(_p("errors", task_id + ".count")) as f:
            return int(f.read().strip() or 0)
    except FileNotFoundError:
        return 0


def _atomic_write(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{socket.gethostname()}.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _server_now() -> float:
    """Current time as seen by the shared filesystem (compute-node clocks can be skewed)."""
    probe = _p("locks", f".now.{socket.gethostname()}.{os.getpid()}")
    with open(probe, "w") as f:
        f.write("x")
    t = os.path.getmtime(probe)
    os.remove(probe)
    return t


def claim(task_id: str) -> bool:
    lock = _p("locks", task_id)
    os.makedirs(_p("locks"), exist_ok=True)
    try:
        os.mkdir(lock)
    except FileExistsError:
        hb = os.path.join(lock, "hb")
        now = _server_now()
        try:
            age = now - os.path.getmtime(hb)
        except FileNotFoundError:
            age = now - os.path.getmtime(lock)
        if age < STALE_SECONDS:
            return False
        shutil.rmtree(lock, ignore_errors=True)
        try:
            os.mkdir(lock)
        except FileExistsError:
            return False
    _atomic_write(os.path.join(lock, "owner.json"), json.dumps(
        {"host": socket.gethostname(), "pid": os.getpid(), "start": time.time(),
         "job": os.environ.get("PBS_JOBID", "local")}))
    _atomic_write(os.path.join(lock, "hb"), str(time.time()))
    return True


def release(task_id: str):
    shutil.rmtree(_p("locks", task_id), ignore_errors=True)


class Heartbeat(threading.Thread):
    def __init__(self, task_id: str, every: float = 60.0):
        super().__init__(daemon=True)
        self.path = _p("locks", task_id, "hb")
        self.every = every
        self.stop_evt = threading.Event()

    def run(self):
        while not self.stop_evt.wait(self.every):
            try:
                with open(self.path, "w") as f:
                    f.write(str(time.time()))
            except OSError:
                pass


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (tuple, set)):
        return list(o)
    return str(o)


def save_result(task, summary: Dict[str, Any], arrays: Optional[Dict[str, np.ndarray]], meta: Dict):
    summary = {**summary, "meta": meta}
    if arrays:
        npz = result_json(task)[:-5] + ".npz"
        os.makedirs(os.path.dirname(npz), exist_ok=True)
        tmp = npz + f".tmp.{os.getpid()}.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, npz)
    # light summary for the dashboard
    light = {"id": task["id"], "suite": task["suite"], "label": task.get("label", task["id"]),
             "dataset": task.get("dataset"), "group": task.get("group"), "finished": time.time(),
             "seconds": meta.get("seconds"), "host": meta.get("host")}
    folds = summary.get("folds")
    if folds:
        keys = ("balanced_accuracy", "accuracy", "auc", "sensitivity", "specificity", "mcc", "f1")
        light["n_folds"] = len(folds)
        light["metrics"] = {k: [f["metrics"].get(k) for f in folds] for k in keys}
    _atomic_write(_p("summary", task["suite"], task["id"] + ".json"), json.dumps(light, default=_jsonable))
    _atomic_write(result_json(task), json.dumps(summary, default=_jsonable))


def execute(task: Dict[str, Any]):
    kind = task.get("kind", "cv")
    if kind == "cv":
        from .evaluation import run_task
        return run_task(task)
    if kind == "diag":
        from .diagnostics import run_diagnostics
        return run_diagnostics(task)
    if kind == "gradvar":
        from .trainability import run_gradient_variance
        return run_gradient_variance(task)
    if kind == "noise":
        from .noise import run_noise_task
        return run_noise_task(task)
    raise ValueError(kind)


def worker_loop(tasks: List[Dict[str, Any]], deadline: Optional[float] = None, log=print,
                on_done: Optional[Callable] = None) -> int:
    from threadpoolctl import threadpool_limits
    n_done = 0
    version = code_version()
    for task in tasks:
        if deadline and time.time() > deadline:
            log("deadline reached; not claiming new tasks")
            break
        if is_done(task) or attempts(task["id"]) >= MAX_ATTEMPTS:
            continue
        if not claim(task["id"]):
            continue
        if is_done(task):                      # finished by someone else meanwhile
            release(task["id"])
            continue
        hb = Heartbeat(task["id"])
        hb.start()
        t0 = time.time()
        log(f"START {task['id']}")
        try:
            with threadpool_limits(task.get("threads", 1)):
                summary, arrays = execute(task)
            meta = {"seconds": time.time() - t0, "host": socket.gethostname(), "code_version": version,
                    "job": os.environ.get("PBS_JOBID", "local"), "threads": task.get("threads", 1),
                    "finished": time.time()}
            save_result(task, summary, arrays, meta)
            n_done += 1
            log(f"DONE  {task['id']} in {meta['seconds']:.1f}s")
        except Exception:
            tb = traceback.format_exc()
            os.makedirs(_p("errors"), exist_ok=True)
            with open(_p("errors", task["id"] + ".txt"), "a") as f:
                f.write(f"--- {time.ctime()} on {socket.gethostname()}\n{tb}\n")
            _atomic_write(_p("errors", task["id"] + ".count"), str(attempts(task["id"]) + 1))
            log(f"FAIL  {task['id']}\n{tb}")
        finally:
            hb.stop_evt.set()
            release(task["id"])
        if on_done:
            try:
                on_done()
            except Exception:
                log("dashboard update failed:\n" + traceback.format_exc())
    return n_done
