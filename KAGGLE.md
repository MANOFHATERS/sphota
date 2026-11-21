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
   Choose T4 x2: the pipeline runs independent jobs on both GPUs (see below).
3. Put your names in the cell below (`YOUR_USER`), and keep this as the notebook's only cell.

```python
import os, subprocess
if not os.path.exists("sphota"):
    subprocess.run(["git", "clone", "https://github.com/MANOFHATERS/sphota.git"], check=True)
!python sphota/scripts/kaggle_run.py --sync hf:YOUR_USER/sphota-run --data-repo YOUR_USER/sphota-data --data-dir /kaggle/temp/sphota_data --weekly-quota 30
```

`--weekly-quota 30` (your weekly GPU hours) is what protects the run from the quota ending mid-session; see "Weekly GPU quota" below.

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

## Weekly GPU quota (the driver plans around it)

Kaggle ends a session without warning when the weekly GPU quota is used up, and no API tells a notebook how much is left. So the
driver plans the budget itself and never starts a session it cannot finish cleanly:

| Flag | Meaning |
|---|---|
| `--weekly-quota 30` | Keeps a ledger (`quota/<session>.json`, uploaded with the run state) and counts the hours of the last 7 days: budget = min(11.5, 30 - used - 0.5). A killed session counts as having used its whole reservation, so the count is never too low. |
| `--quota-hours R` | R = the hours the Kaggle page shows as left right now: budget = min(11.5, R - 0.5). Use it instead, or as a correction if the ledger is off. |
| none | Works, but prints a warning: the quota is invisible to the driver. |

Trainers pause and save when the budget is used up, so the quota is never hit mid-run; the 0.5 h margin is for the final upload.
With less than 15 minutes of budget the session does not start (`NO_QUOTA`, exit 76, nothing touched). If the quota is wrong anyway,
checkpoints are written every 9 minutes and uploaded every 15 (`--sync-every`), so at most about 15 minutes are lost.

## Faster, with exactly the same data, models and results

Three things shorten the run without touching what is trained (all are on by default; `scripts/profile_train.py` measures them on your GPU in
about 5 minutes, run it once before the long run):

1. **Decoded-image cache** (`scripts/build_cache.py`, run automatically by the pipeline). Every training sample used to decode a full 2K
   PNG just to keep one crop. The images are now decoded once into a raw uint8 file and the trainers read crops from it. The crops are
   the same: tests check that the tensors are identical to the PNG path for the same random state. It needs about 41 GB for the 4,909
   images on the data disk (`--data-dir /kaggle/temp/...`). If the disk is smaller, as many images as fit are cached and the rest are
   decoded from their PNG as before: still the same data, just less speed-up. `NO_CACHE=1` skips it, `TS_IMAGE_CACHE=off` ignores it.
   The cache is rebuilt in each new session (a few minutes); it is not uploaded.
2. **Speed settings** (`truesight/speed.py`): `cudnn.benchmark`, `channels_last` for the fp16 U-Net (off for the fp32 codec, where it
   has no Tensor-Core benefit), and one fused call for the EMA update. Float results are the same apart from rounding noise.
   `TS_CUDNN_BENCHMARK=0` / `TS_CHANNELS_LAST=none|unet|all` change them if the profile shows a setting does not help on your GPU.
3. **One job graph over both GPUs** (`scripts/train_phase.py`, used with 2+ GPUs). A GPU never waits while some job is ready:

| Job | Needs | Runs |
|---|---|---|
| lambda 0.0130 base (150k its) | nothing | GPU 0 at once, alone: nothing can share it |
| K=1 ablation base, ReLU ablation base | nothing | GPU 1 at once (used to idle), then their fine-tunes |
| lambda 0.0018 fine-tune, then **diffusion** | the 0.0130 base; then the 0.0018 codec only | diffusion starts the moment the 0.0018 codec exists, not after all six |
| other 4 fine-tunes, `train_ablations`, `ae`, `step9` | the base / the 0.0018 codec / nothing | on whichever GPU is free |
| `baselines` | nothing | on the CPU beside the GPUs |
| after training: `codec_eval -> bitstream -> ablation_eval -> diffusion_eval -> clic_crops -> protect` and `guidance -> honesty -> study_pairs -> intelligence` | as named | two chains side by side, then `report` |

With one GPU everything runs one after the other, in the same order. If one job fails, the others get SIGTERM and save their exact state, so
no work is lost. Each job writes only its own folders; its console output goes to `experiments/logs/job_<name>.out`. Whether a "T4 x2"
session costs one or two sessions of quota is something to check on your account after the first run: look at the quota counter.

## Two notebooks on the same run repo

Safe, as long as they do different work: give each its own `STAGES` (`--stages "diffusion"` in one, `--stages "codec_ablation"` in
the other). Every stage is claimed first (`.running/<stage>.json`, refreshed by each upload, ignored after 45 minutes without a
refresh), so a stage that is running elsewhere or finished is skipped. A push sends only the files that session changed since its
last pull, so a session can never overwrite a newer checkpoint from the other with a stale copy. A stage whose input is made by the
other notebook waits (exit 75) until it exists. Both notebooks count against the same account quota.

## What runs, in order (resume-able stages)

`setup data baselines ae codec codec_eval bitstream step9 diffusion diffusion_eval clic_crops protect guidance honesty intelligence
codec_ablation ablation_eval train_ablations study_pairs report`. Run a subset with `--stages "diffusion diffusion_eval"`; try everything with tiny
settings first with `--smoke` (about 15 minutes of GPU). Each finished stage is uploaded immediately.

## Honest limits

* **Kaggle GPU quota.** Kaggle limits GPU hours per week (check the number shown on your account; it has been about 30 h). The guide's
  budget is 120-180 T4-hours, so expect several weeks of sessions. Training alone is the large part; only the long runs
  (`codec`, `diffusion`, `codec_ablation`, `train_ablations`) matter for the quota.
* **Resume is exact in state, not bit-identical in data order.** Weights, optimiser, EMA and the step counter come back exactly;
  the random crops after a restart are drawn with a seed that depends on the step, so the run is statistically the same as an
  uninterrupted one but not byte-for-byte identical.
* **Check the evaluation time before the long run:** `python scripts/benchmark_decode.py` prints the projected hours of every evaluation
  stage on your GPU (see CLOUD_RUNBOOK.md, 4b). Evaluation images are capped at 768 x 512 pixels per image for this reason.
* **A stage that is not a trainer restarts from its beginning if it is cut off** (the CLIC crop stage is the exception: it keeps its batches), for example an evaluation that needs 40 minutes
  and is interrupted at minute 39. The driver avoids this by not starting a stage when too little session time is left.
* **If the whole Kaggle kernel is killed** (not the Python process), the last upload can be up to 30 minutes old.
* **Kaggle limits change** (session length, quota, disk, scheduled runs). Re-check them before a long run.
* The user study needs real people and a machine they can reach (`python scripts/user_study_app.py`), so it is not part of the unattended run.
