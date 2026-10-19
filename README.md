# Event template filling with open LLMs (MUC-3/4)

[![CodeQL](https://github.com/rosscyking1115/event-extraction-llm-baseline/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/rosscyking1115/event-extraction-llm-baseline/actions/workflows/github-code-scanning/codeql)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=rosscyking1115_event-extraction-llm-baseline&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=rosscyking1115_event-extraction-llm-baseline)

Can an open, instruction-tuned language model read a news report and fill in a fixed form about the event it describes (what happened, where, when, who did it, who or what was hit), with no fine-tuning and nothing but a prompt?

The test set is MUC-3/4, a classic information-extraction benchmark from the early 1990s: news reports about terrorist incidents in Latin America, each paired with hand-filled answer forms ("templates") with 23 slots to fill. This was an MSc team project at the University of Sheffield. The code here is my part of it.

**The answer, in one line:** partly. Across the team's prompt designs for Qwen2.5-7B-Instruct, strict micro F1 (a 0 to 1 score of how many slots were filled exactly right, balancing wrong answers against missed ones) went from 0.40 for the first baseline prompt to 0.45 for the best one. That is above the average original MUC-4 system and below the best one, though the scorers differ, so that comparison is only a rough reference.

> **Status.** Concluded: submitted in May 2026 as the team's final report for the COM6911 Team Project. Not maintained. Every score in this README is quoted from that submitted report, which is not public. They are team results, produced by a teammate's shared evaluation script that is not in this repository, so they cannot be reproduced from this code alone.

## Team results

All seven runs in the report's main results table: strict slot-level scores on MUC-3/4 with Qwen2.5-7B-Instruct. Micro scores pool every slot decision before computing precision, recall and F1; macro F1 averages over slots.

| Run by | Prompt design (strategy) | Micro P | Micro R | Micro F1 | Macro F1 |
|---|---|---|---|---|---|
| Team member | Zero-shot prompt (independent) | 0.7511 | 0.1550 | 0.2570 | 0.2093 |
| Team member | Baseline, full data (independent) | 0.6868 | 0.2861 | 0.4039 | 0.3909 |
| Team member | Decomposed micro-tasks (sequential) | 0.6969 | 0.3161 | 0.4349 | 0.4135 |
| Team member | Grouped zero-shot prompt (sequential) | 0.4484 | 0.4495 | **0.4490** | 0.4233 |
| Team member | Full-context strategy (sequential, full) | 0.5559 | 0.2912 | 0.3822 | 0.3721 |
| Me | Hybrid prompt (independent) | 0.6400 | 0.2689 | 0.3787 | 0.3378 |
| Team member | Slot-restricted prompt (independent) | 0.4046 | 0.1305 | 0.1974 | 0.1033 |

The zero-shot prompt in the first row is the clearest trade-off: when it answered it was usually right (precision 0.7511), but it missed most of the gold answers (recall 0.1550). The best prompt gave up precision for much higher recall.

For reference, the report quotes the original MUC-4 results under the official MUC scorer: F1 0.3219 for the average system and 0.5715 for the best.

## Who did what

This was a team of five. Six of the seven runs above are teammates' work; the hybrid-prompt run is mine.

My part was prompt design, an evaluator implementing the four-layer scoring scheme our professor specified, and the MUC-3/4 and MUC-6 data work behind them: parsers for the corpora and their answer keys, and converters that turn the keys into gold labels. That code is in this repository, with one gap: the hybrid prompt behind my run in the table is not here. The MUC prompts in this repository are the zero-shot, few-shot and v2 prompts in `muc_model_eval.py`. Teammates' code, including the shared evaluator behind the scores above, is not here either. Every commit here is mine because this is my repository, not because I did all the work.

## What is in this repository

| Path | What it does |
|---|---|
| `parse_muc34.py`, `parse_muc6.py` | Parse the MUC-3/4 and MUC-6 corpora and answer keys into JSON |
| `build_gold_labels.py`, `convert_gold_from_json.py` | Turn MUC-3/4 answer keys into gold labels for the team's shared evaluator |
| `convert_to_eval_format.py` | Convert predictions into that evaluator's input format |
| `muc_model_eval.py` | Zero-shot (v1 and v2) and few-shot inference with a Hugging Face causal language model |
| `evaluate_muc.py` | My implementation of the professor's four-layer scoring scheme (JSON validity, schema validity, strict match, fuzzy match at normalised Levenshtein similarity 0.8), with micro P/R/F1, macro F1, empty and majority-class baselines (majority: MUC-4 only), and an error type for every wrong slot |
| `run_muc_pipeline.sh` | The Stanage HPC job script for the whole sequence (see the known issues) |
| `scripts/`, `results/` | Preliminary trigger-detection baselines on MAVEN and WikiEvents, and their outputs |

The code also handles MUC-6 (corporate succession: who joined or left which post, 9 slots), and `run_muc_pipeline.sh` runs Llama-3.1-8B-Instruct as well as Qwen2.5-7B-Instruct. The report's results are for MUC-3/4 with Qwen only.

Before the team settled on MUC, I ran sentence-level baselines on MAVEN and WikiEvents (find the word that signals an event and name its type). They were preliminary and are kept for the record; no figures from them are claimed here.

## Running it

You need the MUC data (see below) and, for inference, a GPU. The parsers and the evaluator use only the Python standard library (the MUC-3/4 parser also shells out to `zcat` and `tar`); inference needs `torch` and `transformers`. There is no pinned environment file. This code loads the model in float16 and decodes greedily; I ran it on Sheffield's Stanage HPC cluster.

```bash
python parse_muc34.py --data_dir /path/to/muc34 --output_dir data/
python muc_model_eval.py --dataset muc4 --data_file data/muc4_tst3.json --model Qwen/Qwen2.5-7B-Instruct --prompt_type zero_shot --output_dir outputs/muc/
python evaluate_muc.py --predictions outputs/muc/muc4_tst3_qwen_qwen2.5_7b_instruct_zero_shot.jsonl --gold data/muc4_tst3.json --dataset muc4 --output_csv outputs/muc/muc4_tst3_qwen_zero_shot_scores.csv
python evaluate_muc.py --gold data/muc4_tst3.json --dataset muc4 --baseline majority --output_csv outputs/muc/muc4_tst3_majority_scores.csv
```

`data/` and `outputs/` are git-ignored, so MUC-derived files stay out of commits. `run_muc_pipeline.sh` runs the same steps on Stanage, MUC-6 and both models included, but see the known issues below.

## Known issues in this code

A code review after submission found defects in my evaluator, parsers and scripts. The team scores above were computed by the shared evaluator, which isn't here and which the review didn't cover. This code may still have fed them: `build_gold_labels.py` and `convert_gold_from_json.py` were written to make gold labels for that evaluator, `convert_to_eval_format.py` converts predictions for it, and nothing here records which gold file the reported runs used. No score computed by `evaluate_muc.py` is quoted here. The main defects:

- `evaluate_muc.py` scores only the first gold template in each MUC-3/4 document, so documents that describe several incidents are only partly scored.
- It doesn't normalise answer-key notation (cross-references such as `CIVILIAN: "JESUIT PRIESTS"`, and `/` alternatives), so some correct answers can never match exactly.
- MUC-6 events are paired greedily in file order, so the score can depend on that order.
- Fuzzy matching at 0.8 accepts near-miss numbers and dates (`2 KILLED` against `3 KILLED`). This affects the fuzzy scores only.
- `build_gold_labels.py` skips an answer key's continuation lines, so a slot with several answers keeps only the first. If the shared evaluator's gold labels came from it, this could affect the team scores.
- `run_muc_pipeline.sh` stops at step 4 because steps 4 to 7 look for prediction files under the wrong names.
- Few-shot inference doesn't check its examples. Without `--few_shot_file` it takes them from the test file itself, and if the example file is empty it runs with none while still labelling the run few-shot. The development split the pipeline uses for examples may parse empty, because the parser accepts only `-MUC4-` document ids; that is unconfirmed without the data.
- The MAVEN "constrained" script offered the model only the event types that occur in its 50 scored samples.

## Data and licences

No MUC data is included, and none should be committed. MUC-3/4 is free from [NIST's MUC archive](https://www-nlpir.nist.gov/related_projects/muc/muc_data/muc_data_index.html). MUC-6 is licensed through the Linguistic Data Consortium ([LDC2003T13](https://catalog.ldc.upenn.edu/LDC2003T13)).

The files in `results/` contain sentences and gold labels from two public datasets, alongside model and rule-baseline outputs:

- **MAVEN** (Wang et al., EMNLP 2020), released by its authors through [THU-KEG/MAVEN-dataset](https://github.com/THU-KEG/MAVEN-dataset), whose repository is MIT-licensed.
- **WikiEvents** (Li, Ji and Han, NAACL 2021), released by its authors through [raspberryice/gen-arg](https://github.com/raspberryice/gen-arg), whose repository is MIT-licensed.

Copyright in the underlying text stays with its original sources. Cite the dataset papers if you use these files.
