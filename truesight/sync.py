"""Keep experiments/ (checkpoints, logs, stage markers) in persistent storage so a new Kaggle session can continue.

A Kaggle session has a 12-hour limit and its disk is wiped afterwards, so the run state lives outside it:

    hf:<user>/<repo>     a private Hugging Face model repo (needs the HF_TOKEN environment variable / Kaggle secret)
    dir:<path>           any directory, for example an attached Kaggle Dataset copy or a Google Drive mount

    python -m truesight.sync pull   --dir experiments --spec hf:user/sphota-run
    python -m truesight.sync push   --dir experiments --spec hf:user/sphota-run
    python -m truesight.sync data-push --data-repo user/sphota-data      # once: upload data/raw + split lists
    python -m truesight.sync data-pull --data-repo user/sphota-data      # every new session: restore them in minutes

Only run state is stored: *.pt, *.csv, *.json, *.png, *.log, *.txt, *.md and the .done/ stage markers (no *.tmp files, nothing
larger than ``MAX_FILE_GB``). ``Syncer.start_background`` pushes every few minutes while training, so even a hard kill loses
little. Pull only fetches files that are missing or older locally; push only sends files that changed.
"""
import argparse
import os
import shutil
import sys
import threading
import time
from pathlib import Path

KEEP_SUFFIXES = {".pt", ".csv", ".json", ".png", ".log", ".txt", ".md", ".yaml"}
MAX_FILE_GB = 4.0


def wanted(rel, size=None):
    """Should this path (relative to the run folder, with / separators) be persisted?"""
    p = Path(rel)
    if p.name.endswith(".tmp") or ".tmp" in p.suffixes:
        return False
    if any(part in ("__pycache__", ".cache") for part in p.parts):
        return False
    if size is not None and size > MAX_FILE_GB * 2 ** 30:
        return False
    if p.parts and p.parts[0] == ".done":
        return True
    return p.suffix.lower() in KEEP_SUFFIXES


def local_files(root):
    root = Path(root)
    out = {}
    if not root.exists():
        return out
    for f in root.rglob("*"):
        if f.is_file():
            rel = f.relative_to(root).as_posix()
            if wanted(rel, f.stat().st_size):
                out[rel] = f
    return out


def parse_spec(spec):
    kind, _, rest = spec.partition(":")
    if kind not in ("hf", "dir") or not rest:
        raise ValueError(f"sync spec must be 'hf:<user>/<repo>' or 'dir:<path>', got {spec!r}")
    return kind, rest


def _copy_atomic(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


class Syncer:
    def __init__(self, spec, local_dir, token=None):
        self.kind, self.target = parse_spec(spec)
        self.local = Path(local_dir)
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.last_error = None

    # -- directory backend ------------------------------------------------------------------------------------------
    def _push_dir(self):
        dst = Path(self.target)
        n = 0
        for rel, f in local_files(self.local).items():
            d = dst / rel
            if not d.exists() or d.stat().st_size != f.stat().st_size or d.stat().st_mtime < f.stat().st_mtime - 1:
                _copy_atomic(f, d); n += 1
        return n

    def _pull_dir(self):
        src = Path(self.target)
        n = 0
        for rel, f in local_files(src).items():
            d = self.local / rel
            if not d.exists() or f.stat().st_mtime > d.stat().st_mtime + 1:
                _copy_atomic(f, d); n += 1
        return n

    # -- Hugging Face backend ---------------------------------------------------------------------------------------
    def _api(self):
        from huggingface_hub import HfApi
        return HfApi(token=self.token)

    def _push_hf(self, squash):
        api = self._api()
        api.create_repo(self.target, repo_type="model", private=True, exist_ok=True)
        files = local_files(self.local)
        if not files:
            return 0
        patterns = [f"**/*{s}" for s in sorted(KEEP_SUFFIXES)] + [f"*{s}" for s in sorted(KEEP_SUFFIXES)] + [".done/*"]
        api.upload_folder(folder_path=str(self.local), repo_id=self.target, repo_type="model", allow_patterns=patterns,
                          ignore_patterns=["*.tmp", "**/*.tmp", "**/__pycache__/*"],
                          commit_message=f"Run state at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
        if squash:                                    # old checkpoint versions would otherwise pile up in the repo
            try:
                api.super_squash_history(self.target, repo_type="model")
            except Exception as e:                    # not fatal: only saves space
                print(f"[sync] could not squash history: {e}", flush=True)
        return len(files)

    def _pull_hf(self):
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError
        self.local.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(self.target, repo_type="model", local_dir=str(self.local), token=self.token)
        except RepositoryNotFoundError:
            print(f"[sync] repo {self.target} does not exist yet: starting from scratch", flush=True)
            return 0
        return len(local_files(self.local))

    # -- public -----------------------------------------------------------------------------------------------------
    def push(self, squash=False):
        with self._lock:
            n = self._push_dir() if self.kind == "dir" else self._push_hf(squash)
        print(f"[sync] pushed ({n} files) to {self.kind}:{self.target}", flush=True)
        return n

    def pull(self):
        with self._lock:
            n = self._pull_dir() if self.kind == "dir" else self._pull_hf()
        print(f"[sync] pulled ({n} files) from {self.kind}:{self.target}", flush=True)
        return n

    def start_background(self, interval_min=30.0):
        """Push every ``interval_min`` minutes on a daemon thread. A failed push is remembered, not fatal."""
        def loop():
            while not self._stop.wait(interval_min * 60):
                try:
                    self.push()
                    self.last_error = None
                except Exception as e:
                    self.last_error = e
                    print(f"[sync] background push failed (will retry): {e}", flush=True)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def stop_background(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


DATA_PATTERNS = ["raw/**", "processed/*.txt", "processed/*.csv"]


def _hf_token(token=None):
    return token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def data_push(repo, local_data, token=None):
    """Upload data/raw and the split lists once to a private Hugging Face *dataset* repo (resumable, any size up to the
    account's storage). A later session restores them with ``data_pull`` in minutes instead of re-downloading 23 GB from the
    original sources."""
    from huggingface_hub import HfApi
    api = HfApi(token=_hf_token(token))
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=str(local_data), allow_patterns=DATA_PATTERNS)
    print(f"[sync] data uploaded to dataset {repo}", flush=True)


def data_pull(repo, local_data, token=None):
    """Download the data uploaded by ``data_push`` into ``local_data`` (skips files that are already complete)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError
    Path(local_data).mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(repo, repo_type="dataset", local_dir=str(local_data), token=_hf_token(token), allow_patterns=DATA_PATTERNS)
    except RepositoryNotFoundError:
        print(f"[sync] dataset {repo} does not exist yet", flush=True)
        return False
    print(f"[sync] data restored from dataset {repo}", flush=True)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["pull", "push", "data-pull", "data-push"])
    ap.add_argument("--data-repo", default=os.environ.get("TS_DATA_REPO"), help="Hugging Face dataset repo <user>/<name> (data-pull / data-push)")
    ap.add_argument("--data-dir", default="data", help="local data folder (contains raw/ and processed/)")
    ap.add_argument("--spec", default=os.environ.get("TS_SYNC"), help="hf:<user>/<repo> or dir:<path> (default: $TS_SYNC)")
    ap.add_argument("--dir", default="experiments", help="local run folder")
    ap.add_argument("--squash", action="store_true", help="hf push: drop old versions of the files afterwards")
    a = ap.parse_args(argv)
    if a.action in ("data-pull", "data-push"):
        if not a.data_repo:
            sys.exit("no dataset repo given: pass --data-repo <user>/<name> or set TS_DATA_REPO")
        (data_pull if a.action == "data-pull" else data_push)(a.data_repo, a.data_dir)
        return
    if not a.spec:
        sys.exit("no storage given: pass --spec or set TS_SYNC (for example hf:<user>/<repo>)")
    s = Syncer(a.spec, a.dir)
    s.pull() if a.action == "pull" else s.push(squash=a.squash)


if __name__ == "__main__":
    main()
