# Hardware Notes — Where to Run This Project

Captured from the project discussion. Two questions came up: **(1) can this run
on a Mac Studio with local models?** and **(2) how would it run on SJSU's HPC, and
would that be better?** Both answers are preserved below. A standalone, expanded
HPC walkthrough lives in [`HPC_SJSU.md`](HPC_SJSU.md).

---

## 1. Running on a Mac Studio (Apple Silicon) with local models

Short answer: **yes for almost everything — with one real exception
(quantization) and one "works but slower" caveat (VLM inference).** Breakdown for
a Mac Studio (Apple Silicon):

**Runs natively, no issues:** the entire non-model stack — scene schema, all
evaluation metrics, memory/retrieval (FAISS `faiss-cpu` has Apple Silicon
wheels), the FastAPI service, tracking, notebooks, tests, and the whole offline
`MockVLM` pipeline. Experiments 1 and 2's *logic* all run fine on CPU.

**VLM inference (Qwen2.5-VL, LLaVA-NeXT) — works, but read this.** Via Hugging
Face Transformers on the MPS backend it will run, but expect some ops to fall
back to CPU (set `PYTORCH_ENABLE_MPS_FALLBACK=1`) and speeds well below a CUDA
GPU. The upside is that Mac Studio's large *unified* memory is genuinely great
here — a 7B VLM in fp16 is ~14–16 GB, so you can hold one (or both) models
comfortably. The better Apple-native route is **MLX / `mlx-vlm`**, which supports
Qwen2.5-VL and LLaVA on Apple Silicon and is noticeably faster than the
Transformers+MPS path.

**Quantization (Experiment 3.2) — this is the blocker as written.** The scaffold
uses `bitsandbytes` for int8/int4, and bitsandbytes' Apple Silicon/MPS support is
still experimental/in-progress (open PRs and community forks, not stable official
support). So E3.2 *as coded* won't run on your Mac. The clean fix: do the
quantization comparison through **MLX**, which ships quantized VLM variants
natively — you'd actually get a cleaner E3.2 than bitsandbytes would give you.

Practically:

- **E1 (grounding) and E2 (memory/retrieval):** fully runnable on the Mac Studio.
  MLX for speed, or Transformers+MPS with the shipped adapters.
- **E3 (efficiency):** latency/memory profiling and the VLM-only-vs-retrieval
  comparison all run; just swap the **quantization** sub-experiment from
  bitsandbytes to MLX.

**Code note:** the `qwen_vl.py` / `llava.py` adapters are skeletons that assume a
CUDA-style `device_map`. For Mac, point them at `device="mps"` (plus the fallback
env var), or — cleaner — add a small **MLX adapter** behind the same
`VisionLanguageModel` interface. Everything else calls that interface, so nothing
downstream changes.

Sources: [bitsandbytes Apple Silicon PR](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1853),
[bitsandbytes docs](https://huggingface.co/docs/transformers/v4.46.0/quantization/bitsandbytes),
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm),
[Qwen MLX guide](https://qwen.readthedocs.io/en/latest/run_locally/mlx-lm.html),
[Local Vision LLMs on Apple Silicon: MLX vs llama.cpp (2026)](https://contracollective.com/blog/local-vision-llm-apple-silicon-mlx-qwen-vl-moondream-2026).

---

## 2. Running on SJSU's HPC — and is it better?

Yes — for the actual model experiments, SJSU's HPC is meaningfully better than the
Mac Studio, mainly because it's NVIDIA/CUDA. The one thing the Mac can't do
(bitsandbytes quantization for E3.2) runs unmodified there, inference is much
faster, and you can batch the whole 720-prediction run.

**The hardware.** SJSU College of Engineering cluster: 44 nodes, ~1,232 cores, and
a strong GPU mix — **18× A100, 5× H100, 1× A40, 3× V100, 17× P100**, with 256 GB
RAM on the GPU nodes. Scheduler is **Slurm**, software via **Lmod modules**, login
by SSH to `coe-hpc.sjsu.edu` (campus network or VPN required). An A100/H100 chews
through 7B-VLM inference far faster than M-series MPS, and CUDA means the shipped
`qwen_vl.py`/`llava.py` adapters and `bitsandbytes` int8/int4 work as written.

**Access first.** Only College of Engineering faculty can request accounts;
**students need a professor to sponsor** them (6 months for class projects, 1 year
for capstone/research). Step zero is lining up a faculty sponsor.

**How you'd run it, end to end:**

1. **Log in + get the code there.** `ssh SJSU_ID@coe-hpc.sjsu.edu` (over VPN), then
   `git clone` your repo (or `rsync` it up). Put **GQA images/scene graphs on
   scratch**, not home — home quotas are usually tight; `rsync -avP` the ~20 GB
   dataset up once.
2. **Environment.** `module load python3`, create a conda env in your home dir,
   then `pip install -e ".[all]"`. No root needed; `faiss-cpu` and torch+CUDA
   install fine.
3. **Interactive smoke test** on a GPU before batching:
   `srun -p gpu --gres=gpu:1 --pty bash`, then
   `python scripts/run_scene_experiment.py --images-dir <scratch>/gqa/images --limit 5`.
4. **Batch the full E1 run** with an sbatch script (see `slurm/e1_array.sbatch`),
   ideally as a Slurm **array job** — one array index per model×condition — so the
   two VLMs × three prompts run in parallel across nodes.
5. **E3.2 quantization now works** — `bitsandbytes` int8/int4 on A100/H100, exactly
   as `configs/models.yaml` is written (no MLX detour needed).
6. **Notebooks:** run Jupyter on a compute node and connect through an SSH tunnel,
   or run the `.py` scripts in batch and do the figure notebook (`13_…`) locally on
   frozen results.
7. **Pull results back:** `rsync` `results/` down and generate/inspect figures on
   your laptop.

**Trade-offs vs the Mac Studio.** The HPC is better for *throughput and CUDA-only
features*, but you trade away interactivity: batch-queue waits, no root, semester
account renewals, VPN, and data staging. The Mac Studio is better for *fast
iterative development* and always-available interactive work.

**One strategic point.** Since this project targets Apple's SIML·ISE role — which
emphasizes **resource-constrained / on-device ML** — the Mac Studio is actually a
*feature*, not just a fallback, for Experiment 3. The strongest efficiency story
is: run the heavy E1/E2 work and the bitsandbytes quantization sweep on the
**A100/H100 HPC**, then add an **on-device (Apple Silicon, MLX) latency/memory
comparison** for E3. That "datacenter GPU vs. on-device" contrast is exactly the
kind of tradeoff analysis the role is looking for, and it uses both machines for
what each does best.

Sources: [SJSU CoE HPC](https://www.sjsu.edu/cmpe/resources/hpc.php),
[SJSU Research Computing resources](https://www.sjsu.edu/it/research/resources/compute.php),
[bitsandbytes CUDA quantization docs](https://huggingface.co/docs/transformers/v4.46.0/quantization/bitsandbytes).
