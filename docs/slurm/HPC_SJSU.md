# Running on the SJSU HPC (College of Engineering Cluster)

A standalone guide to run this project on San Jose State University's College of
Engineering HPC cluster (`coe-hpc.sjsu.edu`). Ready-to-run batch scripts are in
[`slurm/`](slurm/). For the general (laptop/Mac) workflow see
[`PROJECT_RUNBOOK.md`](PROJECT_RUNBOOK.md); for the machine comparison see
[`HARDWARE_NOTES.md`](HARDWARE_NOTES.md).

Why the HPC: it is NVIDIA/CUDA, so the shipped `qwen_vl.py` / `llava.py` adapters
and `bitsandbytes` quantization (E3.2) run **unmodified**, inference is far faster
than Apple-Silicon MPS, and the full 120 × 2 × 3 = **720-prediction** Experiment 1
fans out across multiple GPUs as a Slurm array job.

---

## 0. The cluster at a glance

- **Nodes/cores:** 44 nodes, ~1,232 cores, ~100 TFlop/s peak.
- **GPUs:** 18× A100, 5× H100, 1× A40, 3× V100, 17× P100. GPU nodes have 256 GB RAM.
  A single A100/H100 in fp16 holds a 7B VLM (~14–16 GB) comfortably.
- **Scheduler:** Slurm.
- **Modules:** Lmod (`module load ...`).
- **Login:** SSH to `coe-hpc.sjsu.edu` (fallback `coe-hpc1.sjsu.edu`), campus
  network or **VPN required**.
- **Confirm live specs** once you're in: `sinfo -s` (partitions), `sinfo -o "%n %G"`
  (GPUs per node), `scontrol show partition`.

---

## 1. Get an account

Only **College of Engineering faculty** can request accounts; **students need a
professor to sponsor** the request. Access durations: ~6 months (class projects),
~1 year (capstone/research), renewed each semester. Line up a faculty sponsor
first — this is the real lead-time item.

---

## 2. Log in

```bash
# Connect to campus VPN first if you are off-campus.
ssh <SJSU_ID>@coe-hpc.sjsu.edu          # or coe-hpc1.sjsu.edu if it times out
```

The login node is for editing, submitting jobs, and light work only — **never run
model inference on the login node.** Use an interactive GPU session (§5) or sbatch
(§6).

---

## 3. Stage the code and data

**Code** — clone your repo (or `rsync` it up from your laptop):

```bash
# On the cluster:
git clone https://github.com/<your-username>/grounded-multimodal-memory.git
cd grounded-multimodal-memory

# ...or push from your laptop:
rsync -avP --exclude .venv --exclude data/raw \
    ./GroundedMultimodalSceneMemory/ <SJSU_ID>@coe-hpc.sjsu.edu:~/grounded-multimodal-memory/
```

**Data** — keep large datasets on **scratch**, not home (home quotas are small).
Copy GQA up once (~20 GB):

```bash
# find your scratch path (varies; often $SCRATCH or /scratch/$USER)
echo "$SCRATCH"
rsync -avP /local/path/to/gqa/ <SJSU_ID>@coe-hpc.sjsu.edu:"$SCRATCH"/gqa/
```

Then point `configs/datasets.yaml → gqa.root` (or the `GQA_ROOT` env var in
`.env`) at `$SCRATCH/gqa`. The `.gitignore` already keeps `data/raw/**` out of git.

---

## 4. Build the environment (one time)

No root is available — install everything in userspace with conda + pip.

```bash
module load python3

# Miniconda in your home dir (skip if conda is already available):
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/bin/activate"

conda create -y -n gmm python=3.11
conda activate gmm

# Install the project. On a GPU node you can install the full stack:
pip install -e ".[all]"        # torch(+CUDA), transformers, faiss, fastapi, viz, dev, ...

cp .env.example .env           # then edit: GQA_ROOT, HF_TOKEN (if gated), GMM_DEVICE=cuda
```

Keep model weights off your home quota:

```bash
export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
```

(The provided sbatch scripts set these for you.)

Verify without touching a GPU:

```bash
pytest -q -m "not gpu"
```

---

## 5. Interactive smoke test (do this before batching)

Grab a GPU node interactively and confirm one real model loads and runs:

```bash
bash slurm/interactive.sh                 # 1 GPU, 2h  (or: bash slurm/interactive.sh a100 4:00:00)

# once you land on the compute node:
module load python3
source "$HOME/miniconda3/bin/activate" gmm
export GMM_DEVICE=cuda HF_HOME="$SCRATCH/hf_cache"
nvidia-smi
python scripts/run_scene_experiment.py --images-dir "$SCRATCH/gqa/images" --limit 5
```

If those 5 images produce prediction JSONL in `results/predictions/`, you're ready
to batch the full run.

---

## 6. Batch the experiments with Slurm

### Experiment 1 — array job (fans out over 6 GPUs)

`slurm/e1_array.sbatch` runs **one (model, condition) pair per array task** — 2
models × 3 conditions = 6 tasks — so the whole 720-prediction run happens in
parallel. Edit its **EDIT BEFORE RUNNING** block (conda env, `GQA_IMAGES` path,
partition/GPU, model names) first, then:

```bash
sbatch slurm/e1_array.sbatch
squeue -u "$USER"                      # watch the 6 tasks
tail -f slurm/logs/e1_*_*.out          # follow output
```

When **all** array tasks finish, compute metrics once (cheap; can run on the login
node or a short job):

```bash
python scripts/evaluate_predictions.py
```

### Experiment 2 — memory & retrieval

Author your temporal sequences first (see `PROJECT_RUNBOOK.md` §8 — this is data
you create), put them in `data/sequences/`, then:

```bash
sbatch slurm/e2_memory.sbatch
```

### Experiment 3 — efficiency & quantization

On CUDA this runs as written, **including bitsandbytes int8/int4** (E3.2) — the one
thing an Apple-Silicon Mac can't do. Use the efficiency notebook / latency
utilities, and set `quantization` in `configs/models.yaml` (`none` / `int8` /
`int4`). Pin a specific GPU type so timings are comparable across runs, e.g.
`--gres=gpu:a100:1`.

> Slurm cheat-sheet: `squeue -u $USER` (your jobs), `scancel <jobid>` (cancel),
> `scancel -u $USER` (cancel all), `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS`
> (post-mortem resource use), `sinfo -s` (partition/GPU availability).

---

## 7. Notebooks over an SSH tunnel (optional)

To run the companion notebooks interactively on a GPU node:

```bash
# On the compute node (inside an interactive session):
jupyter lab --no-browser --port=8888 --ip=0.0.0.0

# On your laptop (new terminal) — tunnel through the login node:
ssh -N -L 8888:<compute-node-hostname>:8888 <SJSU_ID>@coe-hpc.sjsu.edu
# then open http://localhost:8888 with the token Jupyter printed.
```

For the final figure notebook (`13_…`), it's simpler to pull results down and run
it on your laptop (§8) — it needs no GPU.

---

## 8. Pull results back and finish locally

```bash
# On your laptop:
rsync -avP <SJSU_ID>@coe-hpc.sjsu.edu:~/grounded-multimodal-memory/results/ ./results/
python scripts/generate_figures.py          # needs the 'viz' extra locally
```

Then freeze tables/figures, fill `results/reports/experiment_summary.md`, and write
the blogs (`BLOG_SERIES.md`).

---

## 9. Is the HPC "better"? Trade-offs

**Better on the HPC:** CUDA (adapters + bitsandbytes work unmodified), much faster
7B inference on A100/H100, parallel array jobs for the 720-prediction run,
reproducible datacenter hardware to cite in the paper.

**Better on the Mac Studio:** always-available interactive development, no queue
waits, no VPN/account renewals, no data staging.

**Recommended split (and it strengthens the portfolio):** the target Apple SIML·ISE
role emphasizes **resource-constrained / on-device ML**. Run the heavy E1/E2 work
and the bitsandbytes quantization sweep on the **A100/H100 HPC**, then add an
**on-device Apple-Silicon (MLX) latency/memory comparison** for E3. A
"datacenter-GPU vs. on-device" efficiency frontier is exactly the tradeoff story
the role is looking for — and uses each machine for what it does best.

---

## 10. Common gotchas

- **Never run inference on the login node** — always a GPU node (interactive or sbatch).
- **Home quota is small** — put datasets and the HF cache on `$SCRATCH`.
- **Confirm partition/GPU names** with `sinfo -s`; `--partition=gpu` and GPU-type
  strings can differ — the sbatch scripts note where to check.
- **Modules first** — `module load python3` before activating conda in a job.
- **Account renewal** — access lapses each semester; renew before a deadline crunch.
- **Off-campus** — connect to the SJSU VPN before SSH.

Sources: [SJSU CoE HPC](https://www.sjsu.edu/cmpe/resources/hpc.php),
[SJSU Research Computing](https://www.sjsu.edu/it/research/resources/compute.php).
