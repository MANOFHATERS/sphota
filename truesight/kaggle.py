"""One Kaggle session of the full run: restore, run until the time budget, save, stop.   (driver: scripts/kaggle_run.py)

Kaggle stops a session after 12 hours and wipes its disk. A session therefore does exactly this:

    1. pick up the latest code           git pull   (so a bug you fixed on GitHub is used)
    2. make sure the packages exist      pip install only what is missing
    3. restore the data                  attached Kaggle dataset, or a Hugging Face dataset repo, or download once
    4. restore the run state             pull experiments/ (checkpoints, logs, stage markers) from storage
    5. run the pipeline                  every trainer resumes from its last checkpoint; stops itself before the 12 h limit
    6. save the run state                final push, plus a push every 30 minutes while running

Outcomes (the exit status of the pipeline):
    DONE     0    every requested stage finished
    PAUSED   75   the time budget ran out (state saved): start the next session, it continues exactly there
    CRASHED  other  a bug or an out-of-memory error: the trainer already wrote its exact state before dying and this driver
                    uploaded it; fix the code (push the fix to GitHub), start the next session, it continues from that step
"""
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

from . import runtime
from .sync import Syncer, data_pull, data_push

DONE, PAUSED, CRASHED = "DONE", "PAUSED", "CRASHED"
DEFAULT_HOURS = 11.5                     # Kaggle stops at 12 h: the last 30 minutes are for the final upload
REQUIRED = {                              # import name -> pip name
    "compressai": "compressai", "torchac": "torchac", "lpips": "lpips", "diffusers": "diffusers", "easyocr": "easyocr",
    "facenet_pytorch": "facenet-pytorch", "sklearn": "scikit-learn", "xgboost": "xgboost", "mlxtend": "mlxtend",
    "cleanfid": "clean-fid", "editdistance": "editdistance", "imagehash": "imagehash", "pytorch_msssim": "pytorch-msssim",
    "huggingface_hub": "huggingface_hub", "pandas": "pandas", "matplotlib": "matplotlib", "scipy": "scipy", "yaml": "pyyaml",
    "cv2": "opencv-python", "ninja": "ninja", "joblib": "joblib", "requests": "requests", "pyarrow": "pyarrow",
}
NO_DEPS = {"facenet-pytorch"}             # its pins would replace the CUDA torch Kaggle ships with


def log(msg):
    print(f"[kaggle] {msg}", flush=True)


def load_kaggle_secrets(names=("HF_TOKEN",)):
    """Copy Kaggle 'Add-ons > Secrets' into environment variables (does nothing outside Kaggle)."""
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return []
    got = []
    for n in names:
        if os.environ.get(n):
            continue
        try:
            os.environ[n] = UserSecretsClient().get_secret(n); got.append(n)
        except Exception:
            pass
    return got


def missing_packages(required=REQUIRED):
    return [pip for mod, pip in required.items() if importlib.util.find_spec(mod) is None]


def ensure_requirements(required=REQUIRED):
    miss = missing_packages(required)
    if not miss:
        log("all packages present")
        return []
    log(f"installing missing packages: {', '.join(miss)}")
    normal = [m for m in miss if m not in NO_DEPS]
    if normal:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *normal], check=True)
    for m in (m for m in miss if m in NO_DEPS):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", m], check=True)
    return miss


def git_pull(root):
    """Fast-forward to the latest commit so a fix pushed to GitHub is used. Never fails the session."""
    if not (Path(root) / ".git").exists():
        return "not a git checkout"
    r = subprocess.run(["git", "pull", "--ff-only"], cwd=root, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    return out[-1] if out else "ok"


def link(src, dst):
    """dst -> src (symlink), replacing an empty dst folder; leaves a non-empty real dst alone."""
    src, dst = Path(src), Path(dst)
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        if any(dst.iterdir()):
            return False
        dst.rmdir()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src, target_is_directory=True)
    return True


def prepare_data(root, data_input=None, data_dir=None, data_repo=None):
    """Make data/raw and data/processed available. Returns a short description of what was done.

    data_input  folder of an attached Kaggle dataset that contains raw/ (and optionally processed/): linked, instant
    data_dir    a roomy scratch folder (Kaggle: /kaggle/temp/...) that becomes <root>/data, because /kaggle/working holds 20 GB
    data_repo   Hugging Face dataset repo uploaded earlier with ``python -m truesight.sync data-push``: downloaded if missing
    With none of them the pipeline's own ``data`` stage downloads the datasets from their original sources (slow, once)."""
    root, data = Path(root), Path(root) / "data"
    notes = []
    if data_dir:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        if link(data_dir, data):
            notes.append(f"data -> {data_dir}")
    if data_input and (Path(data_input) / "raw").exists():
        (data).mkdir(parents=True, exist_ok=True)
        if link(Path(data_input) / "raw", data / "raw"):
            notes.append(f"data/raw -> {data_input}/raw")
        if (Path(data_input) / "processed").exists() and not (data / "processed" / "train.txt").exists():
            import shutil
            shutil.copytree(Path(data_input) / "processed", data / "processed", dirs_exist_ok=True)
            notes.append("processed lists copied")
    elif data_repo and not (data / "processed" / "train.txt").exists():
        if data_pull(data_repo, data):
            notes.append(f"restored from dataset {data_repo}")
    return "; ".join(notes) or "no data source given: the data stage downloads from the original sources"


def session_env(spec, hours, start=None, data_repo=None, base=None):
    env = dict(base if base is not None else os.environ)
    env["TS_SESSION_START"] = str(start if start is not None else time.time())
    env["TS_TIME_LIMIT_HOURS"] = str(hours)
    env["TS_SYNC"] = spec
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if data_repo:
        env["TS_DATA_REPO"] = data_repo
    return env


def classify(rc):
    return DONE if rc == 0 else PAUSED if rc == runtime.EXIT_PAUSED else CRASHED


def run_session(root, spec, hours=DEFAULT_HOURS, cmd=None, exp_dir=None, data_input=None, data_dir=None, data_repo=None,
                sync_every=30.0, do_git_pull=True, install=True, env_extra=None):
    """Run one session. Returns (status, exit code). ``cmd`` defaults to the full pipeline."""
    root = Path(root)
    exp = Path(exp_dir) if exp_dir else root / "experiments"
    start = time.time()
    got = load_kaggle_secrets()
    log(f"session start; budget {hours} h; run state in {spec}" + (f"; secrets loaded: {got}" if got else ""))
    if do_git_pull:
        log(f"git pull: {git_pull(root)}")
    if install:
        ensure_requirements()
    log(f"data: {prepare_data(root, data_input, data_dir, data_repo)}")
    sync = Syncer(spec, exp)
    try:
        sync.pull()
    except Exception as e:                                         # a first session has nothing to pull
        log(f"nothing restored ({type(e).__name__}: {e})")
    done_before = sorted(p.name for p in (exp / ".done").glob("*")) if (exp / ".done").exists() else []
    log(f"stages already finished: {done_before or 'none'}")
    sync.start_background(sync_every)
    env = session_env(spec, hours, start, data_repo)
    env.update(env_extra or {})
    cmd = cmd or ["bash", str(root / "scripts" / "run_full_pipeline.sh")]
    log("running: " + " ".join(map(str, cmd)))
    rc = subprocess.run(cmd, cwd=root, env=env).returncode
    status = classify(rc)
    sync.stop_background()
    log(f"pipeline ended with status {status} (exit {rc}); saving the run state")
    for attempt in range(3):
        try:
            sync.push(squash=status != CRASHED)
            break
        except Exception as e:
            log(f"final push failed ({type(e).__name__}: {e}); retrying" if attempt < 2 else
                f"FINAL PUSH FAILED: {e}. The state is still in {exp}; if this is a Kaggle notebook, save the version so /kaggle/working is kept.")
            time.sleep(5 * (attempt + 1))
    if data_repo and (root / "data" / "processed" / "train.txt").exists() and not (exp / ".done" / "data_uploaded").exists() \
            and not (root / "data").is_symlink() and not (Path(data_input or "x") / "raw").exists():
        try:                                                       # first session: keep the 23 GB so later sessions restore in minutes
            data_push(data_repo, root / "data")
            (exp / ".done").mkdir(parents=True, exist_ok=True); (exp / ".done" / "data_uploaded").write_text("")
            sync.push()
        except Exception as e:
            log(f"data upload failed (not fatal, the next session retries): {e}")
    minutes = (time.time() - start) / 60
    log({DONE: "ALL STAGES FINISHED.",
         PAUSED: f"PAUSED after {minutes:.0f} min: the state is saved. Start the next session (run the same cell); it continues from here.",
         CRASHED: "CRASHED. The exact state was saved before the failure and has been uploaded. Read the error above, fix the code "
                  "(git push the fix), then start the next session: it continues from that step, not from the beginning."}[status])
    return status, rc
