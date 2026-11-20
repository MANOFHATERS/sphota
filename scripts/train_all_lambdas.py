"""Step 7 training schedule: all six lambdas + the K = 1 ablation.

    python scripts/train_all_lambdas.py                       # guide: 150k its for 0.0130, then 50k fine-tunes
    python scripts/train_all_lambdas.py --base-iters 2000 --finetune-iters 500 --batch 4 --crop 128 --N 32 --M 48 \
                                        --workers 0           # tiny CPU smoke run
    python scripts/train_all_lambdas.py --from-scratch        # every lambda trained independently (needs more compute)

Order: (1) lambda 0.0130 with K=3 (full run), (2) the other five K=3 lambdas fine-tuned from it, (3) K=1 ablation at
two lambdas (default 0.0130 and 0.0035), the K=1 base trained like step (1), the other fine-tuned from it.
Every run resumes from its own last.pt and is skipped when its final.pt already exists.
Layout under --out: lmbda<L>/ (K=3) and K1_lmbda<L>/ (K=1), each with log.csv, last.pt, final.pt.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402  (torch first: avoids the Windows sklearn/lpips DLL clash)
from truesight.codec.train_codec import train_codec  # noqa: E402
from truesight.codec.vae_codec import LAMBDAS  # noqa: E402

BASE = 0.0130


def run_dir(out, lmbda, K):
    return out / (f"lmbda{lmbda:.4f}" if K == 3 else f"K{K}_lmbda{lmbda:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "step7_codec"))
    ap.add_argument("--base-iters", type=int, default=150_000)
    ap.add_argument("--finetune-iters", type=int, default=50_000)
    ap.add_argument("--lambdas", type=float, nargs="*", default=list(LAMBDAS))
    ap.add_argument("--ablation-lambdas", type=float, nargs="*", default=[0.0130, 0.0035])
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--from-scratch", action="store_true", help="train every lambda for --base-iters, no fine-tuning")
    for k, d in dict(N=128, M=192, batch=8, crop=256, workers=4, ckpt_every=5000).items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=int, default=d)
    ap.add_argument("--train-list", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    kw = dict(N=a.N, M=a.M, batch=a.batch, crop=a.crop, workers=a.workers, ckpt_every=a.ckpt_every, train_list=a.train_list)

    def run(lmbda, K, base_final):
        d = run_dir(out, lmbda, K)
        if (d / "final.pt").exists():
            print(f"skip {d.name}: final.pt exists", flush=True)
            return d / "final.pt"
        scratch = a.from_scratch or base_final is None
        train_codec(lmbda, d, K=K, iters=a.base_iters if scratch else a.finetune_iters,
                    init_from=None if scratch else str(base_final), **kw)
        return d / "final.pt"

    for K, lams in [(3, a.lambdas)] + ([] if a.no_ablation else [(1, a.ablation_lambdas)]):
        base = None
        order = sorted(lams, key=lambda l: abs(l - BASE))            # the base lambda (closest to 0.0130) goes first
        for lmbda in order:
            final = run(lmbda, K, base)
            if base is None:
                base = final
    print("done:", sorted(p.parent.name for p in out.glob("*/final.pt")))


if __name__ == "__main__":
    main()
