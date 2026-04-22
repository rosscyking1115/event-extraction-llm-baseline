# Event Extraction LLM Baseline

Zero-shot LLM baseline for event extraction using **Qwen2.5-7B-Instruct** on the University of Sheffield Stanage HPC.

## Project Summary

We evaluate whether a large language model can perform event extraction with no task-specific training, using only prompt engineering. We test two prompting strategies (unconstrained and constrained-label zero-shot) across two datasets.

## Datasets

| Dataset | Task | Split used |
|---|---|---|
| MAVEN | Event detection (trigger + type) | train / valid |
| WikiEvents (NAACL 2021) | Event extraction (trigger + type + arguments) | dev |

## Model

- **Qwen/Qwen2.5-7B-Instruct** via HuggingFace Transformers
- Inference on NVIDIA A100-SXM4-80GB (Stanage HPC)
- float16, greedy decoding

## Results

| Experiment | Samples | Valid JSON | Trigger Acc | Type Acc | Both |
|---|---|---|---|---|---|
| MAVEN unconstrained | 50 | 1.000 | 0.160 | 0.000 | 0.000 |
| MAVEN constrained | 50 | 1.000 | 0.220 | 0.280 | 0.100 |
| WikiEvents unconstrained | 345 | 1.000 | 0.272 | 0.128 | 0.067 |
| WikiEvents constrained | — | — | — | — | — |

## Scripts

| Script | Purpose |
|---|---|
| `scripts/qwen_smoke_test.py` | Verify model loads and generates output |
| `scripts/check_maven.py` | Inspect MAVEN dataset structure |
| `scripts/maven_qwen_eval.py` | MAVEN unconstrained zero-shot eval |
| `scripts/maven_qwen_eval_constrained.py` | MAVEN constrained-label zero-shot eval |
| `scripts/check_wikievents.py` | Inspect WikiEvents dataset structure |
| `scripts/wikievents_qwen_eval.py` | WikiEvents unconstrained zero-shot eval |
| `scripts/wikievents_qwen_eval_constrained.py` | WikiEvents constrained-label zero-shot eval |
| `scripts/error_analysis.py` | Categorise prediction errors |
| `scripts/summarise_results.py` | Print comparison table across all experiments |

## HPC Session Setup

```bash
ssh acp25ck@stanage.shef.ac.uk
srun --partition=gpu --qos=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
module purge
module load Anaconda3/2024.02-1
source activate ee-qwen
export HF_HOME=/mnt/parscratch/users/$USER/team-rg1/cache/huggingface
export TRANSFORMERS_CACHE=/mnt/parscratch/users/$USER/team-rg1/cache/huggingface/transformers
export HF_DATASETS_CACHE=/mnt/parscratch/users/$USER/team-rg1/cache/huggingface/datasets
cd ~/team-rg1
```

## Data Location (HPC only)

```
/mnt/parscratch/users/acp25ck/team-rg1/data/
  MAVEN Event Detection/   (train.jsonl, valid.jsonl, test.jsonl)
  wikievents/              (train.jsonl, dev.jsonl, test.jsonl)
```
