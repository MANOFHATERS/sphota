"""Demo cache: remember the finished results of a photo and replay them the next time the same photo is uploaded.

Capture (automatic, while the real pipeline runs): the Compress result + its stage events, the first decode (Trust page), the
decode the Compare page asks for, the Compare result and the Explain result are written to ``demo_cache/<entry>/``.
Replay: when an uploaded photo matches an entry that has all four parts, no model runs. The same events, images and numbers are
served again, spread over ``TS_DEMO_WAIT_S`` seconds (default 30), so every page looks exactly as it did live.

    TS_DEMO_CACHE     1 (default) = capture and replay, 0 = always run the real models
    TS_DEMO_DIR       where the entries live (default <repo>/demo_cache)
    TS_DEMO_WAIT_S    how long a replayed Compress takes (default 30)
    TS_DEMO_RUN_WAIT_S how long a replayed re-decode (Compare page) takes (default 8)

A photo matches by the SHA-256 of its bytes, or (so a re-saved copy of the same picture still matches) by a 64-bit difference hash
within a Hamming distance of 6. Pure Python (Pillow only), no torch.
"""
import hashlib
import io
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
JID, RUN = "@JID@", "@RUN@"               # placeholders for the session id and the id of one decode run inside saved text / file names
_URL_FILE = re.compile(r"/files/([^\"/?]+)")
MAX_HAMMING = 6
REQUIRED = ("compress.json", "decode_root/events.json", "compare_default.json", "explain.json")


def dhash(data):
    """64-bit difference hash of an image (as an int); None when the bytes are not an image."""
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("L").resize((9, 8), Image.LANCZOS)
    except Exception:
        return None
    px = list(im.getdata())
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | (px[r * 9 + c] > px[r * 9 + c + 1])
    return bits


class DemoCache:
    def __init__(self, root=None, enabled=None, wait_s=None, run_wait_s=None):
        self.root = Path(root or os.environ.get("TS_DEMO_DIR") or ROOT / "demo_cache")
        self.enabled = (os.environ.get("TS_DEMO_CACHE", "1") != "0") if enabled is None else enabled
        self.wait_s = float(os.environ.get("TS_DEMO_WAIT_S", 30)) if wait_s is None else wait_s
        self.run_wait_s = float(os.environ.get("TS_DEMO_RUN_WAIT_S", 8)) if run_wait_s is None else run_wait_s
        self._lock = threading.Lock()

    # ------------------------------------------------------------------------------------------------ lookup
    def complete(self, entry):
        return all((entry / p).is_file() for p in REQUIRED)

    def find(self, data, mode=None, network=None):
        """The complete entry whose photo is the uploaded one (same bytes, or the same picture re-saved), or None."""
        if not self.enabled or not self.root.is_dir():
            return None
        sha, dh = hashlib.sha256(data).hexdigest(), dhash(data)
        best, best_rank = None, None
        for d in self.root.iterdir():
            mp = d / "meta.json"
            if not mp.is_file() or not self.complete(d):
                continue
            try:
                m = json.loads(mp.read_text())
            except Exception:
                continue
            exact = m.get("sha256") == sha
            near = dh is not None and m.get("dhash") is not None and bin(dh ^ int(m["dhash"])).count("1") <= MAX_HAMMING
            if not (exact or near):
                continue
            rank = (exact, m.get("mode") == mode and m.get("network") == network, m.get("mode") == mode)
            if best_rank is None or rank > best_rank:
                best, best_rank = d, rank
        return best

    # ------------------------------------------------------------------------------------------------ capture
    @staticmethod
    def _sub(text, job):
        """Replace the session / run ids of a job by the placeholders."""
        text = text.replace(job.id, RUN)               # a re-decode's id ("<session>-d1") contains the session id, so it goes first
        if job.id != job.root_id:                      # (the first decode runs under the session id itself: run == session)
            text = text.replace(job.root_id, JID)
        return text

    @staticmethod
    def _sub_root(text, job):
        return text.replace(job.root_id, JID)

    def _write(self, path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding="utf-8")

    def _grab(self, text, src_dir, dest_dir, job):
        """Copy every file a saved text points at (``/files/<name>``) next to it, names with the ids replaced."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        for name in set(_URL_FILE.findall(text)):
            f = Path(src_dir) / name
            if f.is_file():
                shutil.copy2(f, dest_dir / self._sub(name, job).replace(job.root_id, JID))

    def begin(self, job, data, mode, network, target_kb, result):
        """Start a fresh entry for a live Compress. Returns nothing; the job remembers its entry for the later phases."""
        if not self.enabled:
            return
        sha = hashlib.sha256(data).hexdigest()
        entry = self.root / f"{sha[:16]}_{mode}_{network}"
        with self._lock:
            shutil.rmtree(entry, ignore_errors=True)
            entry.mkdir(parents=True, exist_ok=True)
            dh = dhash(data)
            (entry / "meta.json").write_text(json.dumps({"sha256": sha, "dhash": None if dh is None else str(dh), "mode": mode,
                                                          "network": network, "target_kb": target_kb, "bytes": result.get("bytes"),
                                                          "created": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")
            text = self._sub_root(json.dumps(result), job)
            (entry / "compress.json").write_text(text, encoding="utf-8")
            files = entry / "files"
            files.mkdir(exist_ok=True)
            for n in ("original.png", "seed.sphota"):
                if job.path(n).is_file():
                    shutil.copy2(job.path(n), files / n)
            self._write(entry / "compress_events.json", [json.loads(self._sub_root(json.dumps(e), job)) for e in list(job.events)])
        job.meta["demo_capture"] = str(entry)

    def capture_decode(self, run, kind):
        """Save a finished decode run (kind: ``decode_root`` for the first one, ``decode_run`` for a re-decode of the Compare page)."""
        entry = run.meta.get("demo_capture") if run.meta else None
        if entry is None and run.parent is not None:
            entry = run.parent.meta.get("demo_capture")
        if not entry or not run.finished or run.events[-1]["stage"] != "done":
            return
        entry = Path(entry)
        events = [json.loads(self._sub(json.dumps(e), run)) for e in list(run.events)]
        text = json.dumps(events)
        root = run.parent if run.parent is not None else run
        with self._lock:
            if not entry.is_dir():
                return
            dest = entry / kind
            shutil.rmtree(dest, ignore_errors=True)
            self._grab(json.dumps(list(run.events)), root.dir, dest / "files", run)
            self._write(dest / "events.json", json.loads(text))

    def capture_json(self, root, name, result, key=None):
        entry = root.meta.get("demo_capture")
        if not entry or not Path(entry).is_dir():
            return
        entry = Path(entry)
        text = self._sub_root(json.dumps(result), root)
        with self._lock:
            self._grab(json.dumps(result), root.dir, entry / "files", root)      # the images the result points at
            if name == "compare":
                for n in ([f"compare_{key}.json"] if key else []) + ["compare_default.json"]:
                    (entry / n).write_text(text, encoding="utf-8")
            else:
                (entry / f"{name}.json").write_text(text, encoding="utf-8")

    # ------------------------------------------------------------------------------------------------ replay
    def _load(self, entry, name, root_id, run_id=None):
        text = (Path(entry) / name).read_text(encoding="utf-8")
        if run_id:
            text = text.replace(RUN, run_id)
        return json.loads(text.replace(JID, root_id))

    @staticmethod
    def _copy_files(src, dest_dir, root_id, run_id=None):
        if not src.is_dir():
            return
        for f in src.iterdir():
            name = f.name.replace(RUN, run_id or root_id).replace(JID, root_id)
            shutil.copy2(f, dest_dir / name)

    def replay_compress(self, job, entry):
        """Serve the saved Compress of a photo: returns the CompressResult now, plays the events over ``wait_s`` in the background."""
        entry = Path(entry)
        self._copy_files(entry / "files", job.dir, job.id)
        self._copy_files(entry / "decode_root" / "files", job.dir, job.id, job.id)
        job.meta["demo_entry"] = str(entry)
        result = self._load(entry, "compress.json", job.id)
        evs = self._load(entry, "compress_events.json", job.id) + self._load(entry, "decode_root/events.json", job.id, job.id)
        threading.Thread(target=self._play, args=(job, evs, self.wait_s), daemon=True, name=f"demo-{job.id}").start()
        time.sleep(min(1.0, self.wait_s))
        return result

    def replay_decode(self, run, root):
        """A re-decode asked for by the Compare page: the saved one, over ``run_wait_s``."""
        entry = Path(root.meta["demo_entry"])
        kind = "decode_run" if (entry / "decode_run" / "events.json").is_file() else "decode_root"
        self._copy_files(entry / kind / "files", root.dir, root.id, run.id)
        evs = self._load(entry, f"{kind}/events.json", root.id, run.id)
        threading.Thread(target=self._play, args=(run, evs, self.run_wait_s), daemon=True, name=f"demo-{run.id}").start()

    def get(self, root, name, key=None):
        """Saved Compare / Explain result of a replayed session (a Compare with other settings gets the saved default)."""
        entry = Path(root.meta["demo_entry"])
        n = f"{name}_{key}.json" if key and (entry / f"{name}_{key}.json").is_file() else f"{name}_default.json" if name == "compare" else f"{name}.json"
        self._copy_files(entry / "files", root.dir, root.id)
        return self._load(entry, n, root.id)

    @staticmethod
    def _play(job, events, total):
        """Emit the saved events spread over ``total`` seconds: the stages first, then the denoise previews, then ``done``."""
        stages = [e for e in events if e["stage"] not in ("denoise", "done", "error")]
        denoise = [e for e in events if e["stage"] == "denoise"]
        end = [e for e in events if e["stage"] in ("done", "error")]
        plan = []
        for i, e in enumerate(stages):
            plan.append((0.30 * total * (i + 1) / len(stages), e))
        for i, e in enumerate(denoise):
            plan.append((0.30 * total + 0.65 * total * (i + 1) / (len(denoise) + 1), e))
        plan += [(total, e) for e in end]
        t0 = time.time()
        for off, e in plan:
            time.sleep(max(0.0, t0 + off - time.time()))
            job.emit(e)
