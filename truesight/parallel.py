"""Run independent jobs side by side on several GPUs, and strictly one after the other when one needs the result of another.

A job is a command plus the names of the jobs that must have finished first (``after``). ``run_jobs`` keeps up to ``len(gpus)`` jobs
running at once, each pinned to one GPU through ``CUDA_VISIBLE_DEVICES``, and starts the first job (in list order = priority) whose
dependencies are all finished as soon as a GPU is free. So a GPU never waits while some job is ready: the long job that nothing can
share (the lambda = 0.0130 base codec) runs on one GPU while the independent K=1 / ReLU ablation bases use the other, the diffusion
decoder starts the moment the lambda = 0.0018 codec exists, and so on. Nothing is trained differently: the same commands run, in an
order that respects what needs what.

Outcome, the same convention as the rest of the project:
    0    every job finished (or was already finished: ``skip_if`` exists)
    75   a job paused because the session time budget ran out (state saved) -> run again
    else a job crashed. The jobs still running are sent SIGTERM (to their whole process group) so that each saves its exact state.
No new job is started after a pause or a crash. A SIGTERM / Ctrl-C to this process is forwarded to the running jobs.
A job with ``log`` writes its output to that file instead of this process's output, so two trainers do not interleave their lines.
"""
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

EXIT_PAUSED = 75
POSIX = os.name == "posix"


@dataclass
class Job:
    name: str
    cmd: list
    after: list = field(default_factory=list)       # names of jobs that must be finished first
    skip_if: object = None                          # a path: when it exists the job counts as already finished
    env: dict = field(default_factory=dict)         # extra environment variables for this job
    log: object = None                              # a path: stdout / stderr of the job are appended there


def visible_gpus():
    """GPU ids this process may use: from CUDA_VISIBLE_DEVICES when set, else every device torch sees (0 when none)."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None and env.strip():
        ids = [i.strip() for i in env.split(",") if i.strip()]
        return ids or ["0"]
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    return [str(i) for i in range(max(1, n))]


def validate(jobs):
    names = [j.name for j in jobs]
    if len(set(names)) != len(names):
        raise ValueError("job names must be unique")
    for j in jobs:
        for a in j.after:
            if a not in names:
                raise ValueError(f"job {j.name!r} waits for unknown job {a!r}")


def _terminate(p):
    """SIGTERM the job and everything it started (a job may be a shell script that runs python)."""
    try:
        if POSIX:
            os.killpg(p.pid, signal.SIGTERM)
        else:
            p.terminate()
    except (OSError, ProcessLookupError, AttributeError):
        try:
            p.terminate()
        except OSError:
            pass


def run_jobs(jobs, gpus=None, poll=0.5, log=print, popen=subprocess.Popen):
    """Run ``jobs`` (list order = priority) on ``len(gpus)`` slots. Returns 0, 75 or the exit status of the first crash."""
    validate(jobs)
    gpus = list(gpus) if gpus else visible_gpus()
    done = {j.name for j in jobs if j.skip_if is not None and Path(j.skip_if).exists()}
    for n in sorted(done):
        log(f"[parallel] skip {n}: already finished")
    pending = [j for j in jobs if j.name not in done]
    running = {}                                    # slot -> (job, process, log file)
    state = {"stop": False, "crash": 0, "paused": False}

    def stop_all():
        state["stop"] = True
        for _, p, _ in running.values():
            _terminate(p)

    def forward(signum, frame):
        state["paused"] = True
        stop_all()

    old = {}
    for name in ("SIGTERM", "SIGINT"):
        try:
            old[name] = signal.signal(getattr(signal, name), forward)
        except (ValueError, AttributeError, OSError):
            pass
    try:
        while pending or running:
            free = [s for s in range(len(gpus)) if s not in running]
            if not state["stop"]:
                for j in list(pending):
                    if not free:
                        break
                    if all(a in done for a in j.after):
                        slot = free.pop(0)
                        env = dict(os.environ, **{k: str(v) for k, v in j.env.items()}, CUDA_VISIBLE_DEVICES=str(gpus[slot]))
                        kw, fh = {}, None
                        if POSIX:
                            kw["start_new_session"] = True      # its own process group, so a stop reaches the python children too
                        if j.log:
                            Path(j.log).parent.mkdir(parents=True, exist_ok=True)
                            fh = open(j.log, "ab")
                            kw.update(stdout=fh, stderr=subprocess.STDOUT)
                        log(f"[parallel] start {j.name} on GPU {gpus[slot]}" + (f" (log: {j.log})" if j.log else ""))
                        running[slot] = (j, popen(j.cmd, env=env, **kw), fh)
                        pending.remove(j)
            for slot in list(running):
                j, p, fh = running[slot]
                rc = p.poll()
                if rc is None:
                    continue
                del running[slot]
                if fh is not None:
                    fh.close()
                if rc == 0:
                    done.add(j.name)
                    log(f"[parallel] finished {j.name}")
                    continue
                log(f"[parallel] {j.name} ended with status {rc}" + (" (paused)" if rc == EXIT_PAUSED else ""))
                if rc == EXIT_PAUSED:
                    state["paused"] = True
                elif not state["crash"]:
                    state["crash"] = rc                 # a crash outranks a pause
                if not state["stop"]:
                    stop_all()                          # nothing new starts; the others save and stop themselves
            if not running and (state["stop"] or not any(all(a in done for a in j.after) for j in pending)):
                break
            time.sleep(poll)
    finally:
        for name, h in old.items():
            try:
                signal.signal(getattr(signal, name), h)
            except (ValueError, OSError):
                pass
    return state["crash"] or (EXIT_PAUSED if state["paused"] else 0)
