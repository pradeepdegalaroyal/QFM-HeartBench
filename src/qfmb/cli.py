"""Command-line interface.

  qfmb work  --suites main,external --workers 32 [--select small|big|all] [--hours 500]
  qfmb list  --suites main               (count tasks / pending)
  qfmb run-one --id <task id>           (debug a single task in the foreground)
  qfmb dashboard                         (regenerate runs/dashboard.html now)
  qfmb pending --suites ...              (exit code 0 if work remains, 1 if all done)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time


def _select(tasks, select):
    if select == "big":
        return [t for t in tasks if t.get("big")]
    if select == "small":
        return [t for t in tasks if not t.get("big")]
    return tasks


def _active_suites(default):
    """Suites listed in runs/active_suites.txt (comma/newline separated), else ``default``."""
    from .runner import RUNS
    from .suites import SUITES
    try:
        with open(os.path.join(RUNS, "active_suites.txt")) as f:
            names = [s.strip() for s in f.read().replace("\n", ",").split(",") if s.strip() in SUITES]
        return names or default
    except OSError:
        return default


def _worker(wid, suites, select, deadline, offset, wait=False, use_file=False):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from . import dashboard
    from .runner import MAX_ATTEMPTS, attempts, is_done, worker_loop
    from .suites import all_tasks
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')} w{wid}] {m}", flush=True)
    total = 0
    while True:
        # the active suite list can be changed while jobs run: runs/active_suites.txt
        active = _active_suites(suites) if use_file else suites
        tasks = _select(all_tasks(active), select)
        # light stagger inside the same priority level so workers start on different tasks
        tasks = sorted(tasks, key=lambda t: (t["priority"], (hash(t["id"]) + offset) % 7, -t.get("cost", 1)))
        total += worker_loop(tasks, deadline, log, on_done=dashboard.update)
        if not wait or (deadline and time.time() > deadline):
            break
        left = [t for t in tasks if not is_done(t) and attempts(t["id"]) < MAX_ATTEMPTS]
        if not left:
            break
        time.sleep(300)    # others are running the rest; re-check for stale locks
    log(f"worker finished, {total} tasks done")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qfmb")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("work")
    w.add_argument("--suites", default="main")
    w.add_argument("--workers", type=int, default=1)
    w.add_argument("--select", default="all", choices=["all", "small", "big"])
    w.add_argument("--hours", type=float, default=None, help="stop claiming new tasks after this many hours")
    w.add_argument("--wait", action="store_true", help="keep polling until every task is done")
    for name in ("list", "pending"):
        p = sub.add_parser(name)
        p.add_argument("--suites", default="main")
        p.add_argument("--select", default="all", choices=["all", "small", "big"])
    r = sub.add_parser("run-one")
    r.add_argument("--id", required=True)
    r.add_argument("--suites", default="main,external,diag,gradvar,train,samplesize,ablation,sens,noise,supp")
    sub.add_parser("dashboard")
    a = ap.parse_args(argv)

    from .suites import SUITES, all_tasks
    if a.cmd in ("work", "list", "pending"):
        suites = [s for s in a.suites.split(",") if s]
        if suites == ["active"]:                 # live list from runs/active_suites.txt
            suites = _active_suites(list(SUITES))
        for s in suites:
            if s not in SUITES:
                sys.exit(f"unknown suite {s}")
    if a.cmd == "work":
        from . import dashboard
        from .runner import RUNS
        os.makedirs(RUNS, exist_ok=True)
        dashboard.write_expected(all_tasks(list(SUITES)))
        deadline = time.time() + a.hours * 3600 if a.hours else None
        if a.workers == 1:
            _worker(0, suites, a.select, deadline, 0, a.wait, a.suites == "active")
        else:
            ctx = mp.get_context("spawn")
            procs = [ctx.Process(target=_worker, args=(i, suites, a.select, deadline, i, a.wait,
                                                       a.suites == "active"))
                     for i in range(a.workers)]
            for p in procs:
                p.start()
                time.sleep(0.5)
            for p in procs:
                p.join()
        dashboard.update(force=True)
    elif a.cmd in ("list", "pending"):
        from .runner import attempts, is_done, MAX_ATTEMPTS
        tasks = _select(all_tasks(suites), a.select)
        done = sum(is_done(t) for t in tasks)
        failed = sum((not is_done(t)) and attempts(t["id"]) >= MAX_ATTEMPTS for t in tasks)
        pending = len(tasks) - done - failed
        print(f"tasks={len(tasks)} done={done} failed={failed} pending={pending}")
        if a.cmd == "pending":
            sys.exit(0 if pending > 0 else 1)
    elif a.cmd == "run-one":
        from .runner import execute
        tasks = {t["id"]: t for t in all_tasks(a.suites.split(","))}
        t0 = time.time()
        summary, arrays = execute(tasks[a.id])
        import json
        print(json.dumps([f["metrics"] for f in summary.get("folds", [])][:3], indent=1, default=float))
        print(f"{time.time() - t0:.1f}s")
    elif a.cmd == "dashboard":
        from . import dashboard
        dashboard.update(force=True)


if __name__ == "__main__":
    main()
