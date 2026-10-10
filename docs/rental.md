# Rental runbook: Experiment 1 stages 3.3 (pop 1024) and 3.4

Approved by Noah 2026-10-10: rent after the pop-256 cosine ladder on the 3080 (done). Branch `exp/value-credit` is on origin.
Box: 4× RTX 3090 24 GB (preferred) or 4× RTX 4070 12 GB, Linux, on-demand.

## 0. Rent

- Image: a PyTorch template with CUDA ≥ 12.4 and torch ≥ 2.5 preinstalled (vast.ai "PyTorch (cuDNN Runtime)"), so the box downloads nothing large.
- Disk ≥ 30 GB. Give the lead the `ssh -p PORT root@HOST` line.

## 1. Setup (on the box)

Lessons from the first box (2026-10-10, 4× 3090, host went offline ~10 min into the runs, nothing recovered):
the vast base image has Python 3.12 + `uv` but **no torch**; a `--single-branch` clone lacks `main`, so the 4 upstream
bitwise tests skip; background jobs must detach stdin (`< /dev/null`) or the ssh session hangs; pull results home as
each job finishes; cap CPU threads per job (3 jobs at torch's default 48 threads each may have caused the slow smoke run).

```bash
git clone -b exp/value-credit https://github.com/holdoutlabs-ai/dust.git && cd dust
git fetch -q origin main:main             # upstream-equality tests read main:dust.py
git log --oneline -1                      # record the commit
uv venv -q -p 3.12 /root/venv
VIRTUAL_ENV=/root/venv uv pip install torch==2.10.0 pytest numpy \
    --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple   # ~3 min
/root/venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.device_count())"   # 2.10.0+cu128 4
/root/venv/bin/python -m pytest -q        # ~2 min; expect 62 passed, 0 skipped
mkdir -p data runs/bp-1m runs/logs
export OMP_NUM_THREADS=8                  # per job; 4 jobs on a 48-core box
```

Launch long jobs as `nohup bash script.sh > out 2>&1 < /dev/null &`, each job appending `name exit=$?` to `runs/logs/done`.

From the home box (Git Bash), upload the data and the backprop checkpoint (≈135 MB) so 3.3 uses the same checkpoint as the pop-256 run:

```bash
scp -P PORT data/train_1M.pt data/val.pt data/test.pt data/manifest.json root@HOST:dust/data/
scp -P PORT runs/bp-1m/best.pt root@HOST:dust/runs/bp-1m/
ssh -p PORT root@HOST 'cd dust && sha256sum data/train_1M.pt runs/bp-1m/best.pt'
# expect f65a53b4…6a6d (train_1M.pt), b3b6c305…0a79 (best.pt)
```

Upload takes ~5 min (best.pt is 125 MB).

## 2. Gate for 3.4: time one run (report to Noah before step 4)

```bash
CUDA_VISIBLE_DEVICES=0 python dust.py --tokens 1M --population 256 --steps 2 --output runs/smoke-A
```

Extrapolate: 61 steps + 31 evals per run, 9 runs, 4 GPUs in parallel (3 rounds of up to 4 runs each).

## 3. Stage 3.3 at pop 1024 (one GPU per checkpoint; can run alongside step 2)

```bash
R=results/exp1-value-credit
CUDA_VISIBLE_DEVICES=1 python -m diagnostics.cosine --config paper --variants A,B,C --population 1024 \
    --seeds 0,1,2 --batches 4 --data data/train_1M.pt --device cuda --out $R/3.3-cos-init-pop1024.json > $R/3.3-cos-init-pop1024.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m diagnostics.cosine --config paper --checkpoint runs/bp-1m/best.pt --variants A,B,C --population 1024 \
    --seeds 0,1,2 --batches 4 --data data/train_1M.pt --device cuda --out $R/3.3-cos-bp1m-pop1024.json > $R/3.3-cos-bp1m-pop1024.log 2>&1 &
```

On a 12 GB card, add `--nrep 2` if nrep 8 OOMs (chunking only; record it). Also re-run the pop-256 step-time
measurement here (`results/exp1-value-credit/3080-smoke/step_time.py {A,B,C} 5` on an idle GPU) for the A-vs-B/C timing readout.

## 4. Stage 3.4: 9 runs (only after Noah approves the extrapolated total)

```bash
run() { CUDA_VISIBLE_DEVICES=$1 python dust.py --tokens 1M --population 256 --seed $2 $3 --output runs/3.4/$4-s$2 > runs/3.4/$4-s$2.log 2>&1; }
mkdir -p runs/3.4
# round 1
run 0 42 "" A & run 1 42 "--value-credit analytic" B & run 2 42 "--value-credit analytic --value-gamma 1.0" C & run 3 43 "" A & wait
# round 2
run 0 43 "--value-credit analytic" B & run 1 43 "--value-credit analytic --value-gamma 1.0" C & run 2 44 "" A & run 3 44 "--value-credit analytic" B & wait
# round 3
run 0 44 "--value-credit analytic --value-gamma 1.0" C & wait
```

Seeds 42, 43, 44 share model init and data order across variants (`make_model(seed)`, `randperm(seed + 1)`), so the per-seed
differences B−A, C−A, C−B are paired. Copy back `config.json`, `metrics.jsonl` and `result.json` of each run (not `best.pt`) into
`results/exp1-value-credit/3.4/`.

## 5. Bring results home, then destroy the instance

```bash
scp -P PORT -r root@HOST:dust/results/exp1-value-credit/ results/   # merge, then commit at home
```
