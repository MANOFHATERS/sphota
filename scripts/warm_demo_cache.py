"""Fill the demo cache for photos by running them through the real API once (the same calls the website makes):
Compress -> first decode (Trust) -> the decode the Compare page asks for -> Compare -> Explain.

    python scripts/warm_demo_cache.py photo1.jpg [photo2.png ...] [--mode small] [--network 4G] [--api http://127.0.0.1:8000]

The API must be running (python -m truesight.app.api). Afterwards, uploading the same photo in the website replays these results in
about 30 seconds (TS_DEMO_WAIT_S). Uses only the standard library.
"""
import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def post_multipart(url, fields, name, path):
    b = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{name}"; filename="{path.name}"\r\nContent-Type: {ctype}\r\n\r\n'.encode()
                 + path.read_bytes() + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    req = urllib.request.Request(url, data=b"".join(parts), headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def get_json(url):
    return json.load(urllib.request.urlopen(url, timeout=3600))


def wait_done(api, jid):
    """Read the event stream of a job until its done / error event."""
    with urllib.request.urlopen(f"{api}/api/jobs/{jid}/events", timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data:"):
                e = json.loads(line[5:])
                if e["stage"] == "error":
                    raise RuntimeError(e["message"])
                if e["stage"] == "done":
                    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--mode", default="small")
    ap.add_argument("--network", default="4G")
    ap.add_argument("--target-kb", default="2")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    a = ap.parse_args()
    for p in map(Path, a.photos):
        t0 = time.time()
        print(f"{p.name}: compress ...", flush=True)
        r = post_multipart(f"{a.api}/api/compress", {"mode": a.mode, "network": a.network, "target_kb": a.target_kb}, "file", p)
        jid = r["job_id"]
        wait_done(a.api, jid)
        print("  first decode done; re-decode for the Compare page ...", flush=True)
        body = json.dumps({"job_id": jid, "cfg": 1.5, "steps": 25, "seed": 7, "safe_mode": False}).encode()
        run = json.load(urllib.request.urlopen(urllib.request.Request(f"{a.api}/api/decode", data=body, headers={"Content-Type": "application/json"}), timeout=60))
        wait_done(a.api, run["job_id"])
        print("  compare ...", flush=True)
        get_json(f"{a.api}/api/compare/{jid}?cfg=1.5&steps=25")
        print("  explain ...", flush=True)
        get_json(f"{a.api}/api/explain/{jid}")
        print(f"  saved in {time.time() - t0:.0f} s", flush=True)
    print("done. Entries:", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as e:
        sys.exit(f"{e.code}: {e.read().decode()[:300]}")
