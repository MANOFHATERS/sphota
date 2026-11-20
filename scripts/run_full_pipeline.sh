#!/usr/bin/env bash
# Full-scale run of the whole project as written in the build guide, resumable across crashes and 12-hour sessions.
#
#   bash scripts/run_full_pipeline.sh                  # everything, skipping stages already finished
#   STAGES="data baselines" bash scripts/run_full_pipeline.sh
#   SMOKE=1 bash scripts/run_full_pipeline.sh          # same commands with tiny settings (~15 min): run this first
#   python scripts/kaggle_run.py --sync hf:USER/REPO   # on Kaggle: restores state, runs, saves state (see KAGGLE.md)
#
# Stages, in order:
#   setup data baselines ae codec codec_eval bitstream step9 diffusion diffusion_eval protect guidance honesty intelligence
#   codec_ablation ablation_eval train_ablations study_pairs report
# (codec = the six lambdas; codec_ablation = K=1 and ReLU runs for A1 / A2; train_ablations = A7 / A8; study_pairs builds the
#  user-study images; the study itself needs people: scripts/user_study_app.py; report = every table and figure.)
#
# Resume rules: every trainer continues from its own last.pt; a finished stage leaves $EXP/.done/<stage>; a stage that is cut off
# by the time budget (TS_TIME_LIMIT_HOURS) saves and exits with status 75 and is NOT marked done; a crash saves the exact state
# first and is not marked done either. Run the same command again to continue. Logs: $EXP/logs/<stage>.log.
# If TS_SYNC is set (hf:<user>/<repo> or dir:<path>) the run state is uploaded after every finished stage.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
DEV=${DEV:-cuda}
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
ALL="setup data baselines ae codec codec_eval bitstream step9 diffusion diffusion_eval protect guidance honesty intelligence codec_ablation ablation_eval train_ablations study_pairs report"
STAGES=${STAGES:-$ALL}

# ---- scale: guide values by default, tiny values with SMOKE=1 -------------------------------------------------------------
if [ "${SMOKE:-0}" = "1" ]; then
  EXP=experiments_smoke; FIGS=report_smoke/figures; RES=report_smoke/results; STUDY=study_smoke
  AE_IT=300;  AE_BATCH=8;  AE_CROP=128
  C_BASE=400; C_FT=150;    C_BATCH=4;  C_CROP=128; C_N=32; C_M=48
  D_IT=300;   D_BATCH=8;   D_CROP=128; STEPS=10; MAIN_STEPS=10; D_CH="16 32 32 32"
  U_IT=200;   U_BATCH=8;   U_CH="16 32 32 32"; ABL_IT=100; ABL_BATCH=4; ABL_CH="16 32 32 32"
  N_IMG=8;    N_TBL=40;    N_HALL=12;  TUNE_N=4; STUDY_IMG=4
else
  EXP=experiments; FIGS=report/figures; RES=report/results; STUDY=study
  AE_IT=30000; AE_BATCH=16; AE_CROP=256
  C_BASE=150000; C_FT=50000; C_BATCH=8; C_CROP=256; C_N=128; C_M=192
  D_IT=200000; D_BATCH=32;  D_CROP=128; STEPS=25; MAIN_STEPS=50; D_CH="64 128 256 256"
  U_IT=20000;  U_BATCH=64;  U_CH="64 128 128 256"; ABL_IT=20000; ABL_BATCH=32; ABL_CH="64 128 256 256"
  N_IMG=500;   N_TBL=1500;  N_HALL=100; TUNE_N=20; STUDY_IMG=40
fi
mkdir -p "$EXP/.done" "$EXP/logs" "$FIGS" "$RES"
CODEC=$EXP/step7_codec; DIFF=$EXP/step10_diffusion
LO=$CODEC/lmbda0.0018/final.pt; HI=$CODEC/lmbda0.0483/final.pt; UNET=$DIFF/final.pt; FROZEN=$EXP/step12_guidance/frozen.json

want() { case " $STAGES " in *" $1 "*) [ ! -e "$EXP/.done/$1" ];; *) return 1;; esac; }
run()  { local s=$1; shift; echo "=== [$s] $*"; "$@" 2>&1 | tee -a "$EXP/logs/$s.log"; }   # any failure stops the script (set -e + pipefail)
need() { $PY -m truesight.runtime need "$1" || exit 75; }                                # not enough session time left: pause here
done_() {
  touch "$EXP/.done/$1"; echo "=== [$1] finished $(date -u +%FT%TZ)"
  if [ -n "${TS_SYNC:-}" ]; then $PY -m truesight.sync push --dir "$EXP" --spec "$TS_SYNC" || echo "(upload failed, will retry later)"; fi
}

if want setup; then
  $PY - <<'EOF'
import torch; assert torch.cuda.is_available(), "no CUDA GPU visible"; print(torch.cuda.get_device_name(0), torch.version.cuda)
EOF
  command -v gcc >/dev/null && command -v ninja >/dev/null || { echo "need gcc + ninja for torchac (apt install build-essential ninja-build)"; exit 1; }
  run setup $PY -c "import torchac; print('torchac ok')"          # compiles the extension once
  run setup $PY -m pytest tests -q
  done_ setup
fi

if want data; then                                              # ~23 GB download (skipped when restored), then filter / dedupe / split
  need 30
  run data $PY scripts/prepare_data.py --stage all --workers 8
  wc -l data/processed/*.txt | tee -a "$EXP/logs/data.log"
  done_ data
fi

if want baselines; then                                         # Steps 3-5 (CPU work)
  need 20
  run baselines $PY scripts/eval_rd.py --sets kodak clic
  run baselines $PY scripts/eval_course_baselines.py
  run baselines $PY scripts/train_size_predictor.py
  done_ baselines
fi

if want ae; then                                                # Step 6: PlainAE, M = 8, 16, 32
  run ae $PY scripts/train_ae.py --iters $AE_IT --batch $AE_BATCH --crop $AE_CROP --workers 4 --out "$EXP/step6_ae" --figs "$FIGS"
  done_ ae
fi

if want codec; then                                             # Step 7: lambda 0.0130 full run, then five fine-tunes (no ablations yet)
  run codec $PY scripts/train_all_lambdas.py --out "$CODEC" --base-iters $C_BASE --finetune-iters $C_FT \
      --batch $C_BATCH --crop $C_CROP --N $C_N --M $C_M --workers 4 --no-ablation
  done_ codec
fi

if want codec_eval; then                                        # RD curves, BD-rate vs JPEG / WebP, LPIPS, MS-SSIM on Kodak
  need 20
  run codec_eval $PY scripts/eval_codec.py --exp "$CODEC" --figs "$FIGS"
  done_ codec_eval
fi

if want bitstream; then                                         # Step 8: real .sphota files, bit-exact decode, header counted
  need 20
  run bitstream $PY scripts/eval_bitstream.py --exp "$CODEC" --out "$EXP/step8_bitstream" --figs "$FIGS" --device cpu
  done_ bitstream
fi

if want step9; then                                             # Step 9 warm-up: unconditional 64x64 DDPM + forward-process figure
  run step9 $PY scripts/step9_ddpm.py --out "$EXP/step9_ddpm" --figs "$FIGS" --iters $U_IT --batch $U_BATCH --channels $U_CH --workers 4
  done_ step9
fi

if want diffusion; then                                         # Step 10: conditional decoder on the lowest-lambda codec
  run diffusion $PY -m truesight.diffusion.train_diffusion --codec "$LO" --out "$DIFF" \
      --iters $D_IT --batch $D_BATCH --crop $D_CROP --channels $D_CH --workers 4
  done_ diffusion
fi

if want diffusion_eval; then                                    # CFG sweep (A3), steps sweep (A6), LPIPS / PSNR / FID vs the codec decode
  need 45
  run diffusion_eval $PY scripts/eval_diffusion.py --codec "$LO" --diffusion "$UNET" --out "$DIFF/eval" --figs "$FIGS" \
      --steps $STEPS --main-steps $MAIN_STEPS --device $DEV --fid
  done_ diffusion_eval
fi

if want protect; then                                           # Step 11: ROI coding, TFR / FIR on the text and face sets
  for S in text face; do
    need 30
    run protect $PY scripts/eval_protect.py --set $S --limit $N_IMG --codec-lo "$LO" --codec-hi "$HI" --diffusion "$UNET" \
        --steps $STEPS --tfr --fir --device $DEV --figs "$FIGS" --out "$EXP/step11_protect/$S"
  done
  done_ protect
fi

if want guidance; then                                          # Step 12: tune on val once, freeze, evaluate once on the test sets (A4, A5)
  need 60
  run guidance $PY scripts/tune_guidance.py --codec-lo "$LO" --codec-hi "$HI" --diffusion "$UNET" --limit $TUNE_N \
      --steps $MAIN_STEPS --tfr --fir --device $DEV --out "$EXP/step12_guidance"
  for S in text face kodak clic; do
    need 30
    run guidance $PY scripts/eval_sphota.py --set $S --limit $N_IMG --codec-lo "$LO" --codec-hi "$HI" --diffusion "$UNET" \
        --frozen "$FROZEN" --steps $MAIN_STEPS --tfr --fir --device $DEV --figs "$FIGS" --out "$EXP/step12_guidance/eval_$S"
  done
  done_ guidance
fi

if want honesty; then                                           # Step 13: per-patch signals, 8 detectors, learning curves, safe mode
  need 60
  run honesty $PY scripts/eval_hallucination.py --codec-lo "$LO" --codec-hi "$HI" --diffusion "$UNET" --limit $N_HALL \
      --steps $STEPS --guided --frozen "$FROZEN" --with-roi --encoder-boxes --tfr \
      --device $DEV --figs "$FIGS" --out "$EXP/step13_honesty"
  done_ honesty
fi

if want intelligence; then                                      # Step 14: real byte counts, generative modes, OCR + face checks
  need 60
  run intelligence $PY scripts/build_mode_table.py --list data/processed/train.txt --limit $N_TBL --codecs-dir "$CODEC" \
      --real-bytes --diffusion "$UNET" --steps $STEPS --ocr --faces --device $DEV --out "$EXP/step14_intelligence"
  run intelligence $PY scripts/train_intelligence.py --data "$EXP/step14_intelligence" --figs "$FIGS"
  done_ intelligence
fi

if want codec_ablation; then                                    # A1 (K = 1) and A2 (ReLU instead of GDN) at lambda 0.0130 and 0.0035
  run codec_ablation $PY scripts/train_all_lambdas.py --out "$CODEC" --base-iters $C_BASE --finetune-iters $C_FT \
      --batch $C_BATCH --crop $C_CROP --N $C_N --M $C_M --workers 4 --only-ablation
  done_ codec_ablation
fi

if want ablation_eval; then                                     # the same evaluation again: now it also writes ablation_k.csv and ablation_act.csv
  need 20
  run ablation_eval $PY scripts/eval_codec.py --exp "$CODEC" --figs "$FIGS"
  done_ ablation_eval
fi

if want train_ablations; then                                   # A7 (linear vs cosine schedule) and A8 (Adam vs AdamW, constant vs cosine LR)
  run train_ablations $PY scripts/run_train_ablations.py --codec "$LO" --out "$EXP/step10_ablation" --iters $ABL_IT \
      --batch $ABL_BATCH --channels $ABL_CH --workers 4 --device $DEV --steps $STEPS
  done_ train_ablations
fi

if want study_pairs; then                                       # images for the user study (the answers need real people)
  need 30
  run study_pairs $PY scripts/make_study_pairs.py --codec-lo "$LO" --codec-hi "$HI" --diffusion "$UNET" --frozen "$FROZEN" \
      --images $STUDY_IMG --out "$STUDY" --device $DEV
  done_ study_pairs
fi

if want report; then                                            # every table and figure from the evaluation outputs
  run report $PY scripts/eval_faithfulness.py --exp "$EXP" --out "$RES" --figs "$FIGS"
  run report $PY scripts/make_figures.py --exp "$EXP" --out "$RES" --figs "$FIGS" --study "$STUDY"
  done_ report
fi

echo "ALL REQUESTED STAGES DONE. Results: $EXP/ (tables, checkpoints), $RES/ and $FIGS/"
