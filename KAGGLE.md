# Training the full project on Kaggle (12-hour sessions, crash-proof)

Kaggle stops a notebook after 12 hours and wipes its disk. This project is set up so that **nothing is lost** and **every new
session continues exactly where the last one stopped**, instead of starting again:

| Situation | What happens |
|---|---|
| The 12 hours are nearly used (default budget 11.5 h) | The running trainer saves its exact state (weights, optimiser, EMA, iteration), the driver uploads everything and exits with status 75 ("paused"). Start the notebook again: it downloads the state and continues. |
| Your code has a bug and crashes after 6 hours | The trainer catches the crash, writes the exact state of the last completed step to `last.pt` *before* dying, and the driver uploads it. Fix the bug, push the fix, start the notebook: it continues from that step, not from zero. |
| Kaggle kills the machine without warning | A checkpoint is written every 20 minutes and uploaded every 30 minutes, so you lose at most about 30 minutes. |
| A stage already finished | It is marked done and skipped in later sessions, so no result is ever recomputed. |

Where the state lives: a **private Hugging Face repo** (checkpoints, logs, results, stage markers). Training data lives in a
second private Hugging Face *dataset* repo (or an attached Kaggle Dataset), so a new session does not re-download 23 GB from the
original sites.

## One-time setup

1. **Hugging Face**: create an account, then a token with *write* access (Settings > Access Tokens). Nothing else to create: the two
   private repos are made automatically on first use.
2. **Kaggle notebook**: New Notebook, Accelerator = GPU (T4 x2 or P100), Internet = On, Add-ons > Secrets > add `HF_TOKEN` = your token.
   The pipeline uses one GPU; the second T4 is not used.
3. Put your names in the cell below (`YOUR_USER`), and keep this as the notebook's only cell.

```python
import os, subprocess
if not os.path.exists("sphota"):
    subprocess.run(["git", "clone", "https://github.com/MANOFHATERS/sphota.git"], check=True)
!python sphota/scripts/kaggle_run.py --sync hf:YOUR_USER/sphota-run --data-repo YOUR_USER/sphota-data --data-dir /kaggle/temp/sphota_data
```

(If the GitHub repository is private, clone with a token the same way, from a Kaggle secret.)

`--data-dir /kaggle/temp/...` keeps the 23 GB of images off `/kaggle/working`, which is limited to about 20 GB. Check `!df -h` once;
if `/kaggle/temp` is smaller than you need, tell me and I will change the layout.

## Every session

Press **Save Version > Save & Run All** (runs unattended for up to 12 hours) or run the cell interactively. When the cell ends:

* `ALL STAGES FINISHED.`  you are done.
* `PAUSED ... the state is saved.`  start the next session (same cell, nothing to change).
* `CRASHED ...`  read the error above it, fix the code (edit, commit, `git push`), start the next session. The driver runs
  `git pull` first, so the fix is picked up.

Restart time: fetching the run state (about 1-2 GB) and the code takes about a minute; missing Python packages are installed once per
session (1-3 minutes); the data restore takes a few minutes from the Hugging Face dataset and is instant from an attached Kaggle
Dataset. Do not expect exactly 2 minutes, expect roughly 3-8.

The first session also does the one-time work: the 23 GB download from the original dataset sites and its upload to your data repo.
That first session may end with `PAUSED` before training starts; the data is then ready for all later sessions.

## What runs, in order (resume-able stages)

`setup data baselines ae codec codec_eval bitstream step9 diffusion diffusion_eval protect guidance honesty intelligence codec_ablation
ablation_eval train_ablations study_pairs report`. Run a subset with `--stages "diffusion diffusion_eval"`; try everything with tiny
settings first with `--smoke` (about 15 minutes of GPU). Each finished stage is uploaded immediately.

## Honest limits

* **Kaggle GPU quota.** Kaggle limits GPU hours per week (check the number shown on your account; it has been about 30 h). The guide's
  budget is 120-180 T4-hours, so expect several weeks of sessions. Training alone is the large part; only the long runs
  (`codec`, `diffusion`, `codec_ablation`, `train_ablations`) matter for the quota.
* **Resume is exact in state, not bit-identical in data order.** Weights, optimiser, EMA and the step counter come back exactly;
  the random crops after a restart are drawn with a seed that depends on the step, so the run is statistically the same as an
  uninterrupted one but not byte-for-byte identical.
* **A stage that is not a trainer restarts from its beginning if it is cut off**, for example an evaluation that needs 40 minutes
  and is interrupted at minute 39. The driver avoids this by not starting a stage when too little session time is left.
* **If the whole Kaggle kernel is killed** (not the Python process), the last upload can be up to 30 minutes old.
* **Kaggle limits change** (session length, quota, disk, scheduled runs). Re-check them before a long run.
* The user study needs real people and a machine they can reach (`python scripts/user_study_app.py`), so it is not part of the unattended run.
