#!/usr/bin/env python3
"""Run a command, record wall time, CPU time and peak RSS as one JSON line.

Usage: timeit.py LABEL RESULTS.jsonl STEPLOG -- cmd arg ...
stdout+stderr of the command go to STEPLOG.  Exit status is passed through.
(/usr/bin/time is not installed in this container.)  Run with `python3 -I`.
"""
import json
import resource
import subprocess
import sys
import time


def main():
    label, results, steplog = sys.argv[1:4]
    cmd = sys.argv[5:]
    t0 = time.time()
    with open(steplog, "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    rec = {"label": label, "wall_s": round(wall, 2), "cpu_s": round(ru.ru_utime + ru.ru_stime, 1),
           "maxrss_gb": round(ru.ru_maxrss / 1024 / 1024, 2), "exit": rc, "cmd": " ".join(cmd)[:300]}
    with open(results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{label}: {rec['wall_s']} s wall, {rec['cpu_s']} CPU-s, {rec['maxrss_gb']} GB peak RSS, exit {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
