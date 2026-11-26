"""Add thumbnails of the nearest training photos to the Explain results already saved in demo_cache/ (entries captured before the server
produced thumbnails). Usage: python scripts/add_neighbour_thumbnails.py   (from the repo root)."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from truesight.app.explain import neighbour_thumbnail  # noqa: E402


def main():
    ctx = pd.read_csv(ROOT / "experiments" / "step14_intelligence" / "context.csv")
    for e in sorted((ROOT / "demo_cache").glob("*/explain.json")):
        entry, ex = e.parent, json.loads(e.read_text(encoding="utf-8"))
        (entry / "files").mkdir(exist_ok=True)
        for n in ex["neighbours"]:
            i = int(n["name"])
            name = ctx.loc[ctx["image"] == i, "name"].iloc[0]
            ok = neighbour_thumbnail(ROOT / "data" / "raw", name, entry / "files" / f"nb_{i}.jpg")
            n["thumb_url"] = f"/api/jobs/@JID@/files/nb_{i}.jpg" if ok else ""
            print(entry.name, i, name, "ok" if ok else "photo not found")
        e.write_text(json.dumps(ex), encoding="utf-8")


if __name__ == "__main__":
    main()
