"""Run one Kaggle session of the full pipeline (see KAGGLE.md). Put this in the notebook's only cell:

    !python scripts/kaggle_run.py --sync hf:YOUR_USER/sphota-run --data-repo YOUR_USER/sphota-data

It restores the previous run state, runs every unfinished stage until 11.5 hours are used (Kaggle stops at 12), saves everything
and exits. Start the notebook again and it continues exactly where it stopped; after a crash the state at the moment of the crash
is kept, so after you fix the bug it continues from that step, not from zero.

Exit status: 0 all stages finished, 75 paused at the time limit (run again), 76 not enough weekly GPU quota left (not started),
anything else: crashed (fix and run again). Always pass your quota:  --weekly-quota 30   (or  --quota-hours <hours left on the Kaggle page>).
Needs the Hugging Face token as the environment variable HF_TOKEN or as a Kaggle secret named HF_TOKEN (Add-ons > Secrets).
For a quick local check use --sync dir:/some/folder, which needs no account.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from truesight import kaggle  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sync", required=True, help="where the run state lives: hf:<user>/<repo> (private model repo) or dir:<path>")
    ap.add_argument("--hours", type=float, default=kaggle.DEFAULT_HOURS, help="time budget of this session (Kaggle limit is 12)")
    ap.add_argument("--data-repo", default=None, help="Hugging Face dataset repo <user>/<name> that keeps data/raw between sessions")
    ap.add_argument("--data-input", default=None, help="folder of an attached Kaggle dataset containing raw/ (and processed/)")
    ap.add_argument("--data-dir", default=None, help="roomy scratch folder to use as data/ (Kaggle: /kaggle/temp/sphota_data)")
    ap.add_argument("--stages", default=None, help='only these stages, for example "diffusion diffusion_eval"')
    ap.add_argument("--smoke", action="store_true", help="tiny settings (same commands, minutes instead of days)")
    ap.add_argument("--sync-every", type=float, default=kaggle.DEFAULT_SYNC_MIN, help="minutes between uploads while running (checkpoints are written more often)")
    ap.add_argument("--quota-hours", type=float, default=None, help="hours of GPU quota the Kaggle page shows as left RIGHT NOW; the session budget becomes this minus 0.5 h")
    ap.add_argument("--weekly-quota", type=float, default=None, help="your weekly GPU quota (for example 30): the driver counts the hours of the last 7 days itself from a ledger in the run state")
    ap.add_argument("--no-git-pull", action="store_true")
    ap.add_argument("--no-install", action="store_true")
    a = ap.parse_args()
    env = {}
    if a.stages:
        env["STAGES"] = a.stages
    if a.smoke:
        env["SMOKE"] = "1"
    status, rc = kaggle.run_session(ROOT, a.sync, a.hours, data_input=a.data_input, data_dir=a.data_dir, data_repo=a.data_repo,
                                    sync_every=a.sync_every, quota_hours=a.quota_hours, weekly_quota=a.weekly_quota, do_git_pull=not a.no_git_pull, install=not a.no_install, env_extra=env)
    sys.exit(rc)


if __name__ == "__main__":
    main()
