#!/usr/bin/env bash
# =============================================================================
# run_muc_pipeline.sh — Full MUC Experiment Pipeline for Sheffield Stanage HPC
#
# Runs the complete experiment sequence:
#   1. Parse MUC-3/4 and MUC-6 datasets into JSON
#   2. Run empty baseline evaluation
#   3. Run majority class baseline evaluation
#   4. Run Qwen2.5-7B-Instruct (zero-shot)
#   5. Run Qwen2.5-7B-Instruct (few-shot)
#   6. Run Llama-3.1-8B-Instruct (zero-shot)
#   7. Run Llama-3.1-8B-Instruct (few-shot)
#   8. Summarise all results
#
# Usage:
#   chmod +x run_muc_pipeline.sh
#   ./run_muc_pipeline.sh              # Run all steps
#   ./run_muc_pipeline.sh --step 4    # Run from step 4 onwards
#   ./run_muc_pipeline.sh --only 4    # Run only step 4
#
# Run inside an interactive GPU session:
#   srun --partition=gpu --qos=gpu --gres=gpu:1 --mem=40G --time=08:00:00 --pty bash
#   conda activate ee-qwen
#   ./run_muc_pipeline.sh
# =============================================================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PARSCRATCH="/mnt/parscratch/users/${USER}/team-rg1"
CODE_DIR="${HOME}/team-rg1"

RAW_DATA_MUC4="${PARSCRATCH}/data/raw/muc34"
RAW_DATA_MUC6="${PARSCRATCH}/data/raw/muc_6"
PARSED_DATA="${PARSCRATCH}/data/parsed"
RESULTS_DIR="${PARSCRATCH}/results/muc"
LOGS_DIR="${PARSCRATCH}/logs"

QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"
LLAMA_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Team member name for CSV metadata
MEMBER="Ross"

# ── Parse arguments ───────────────────────────────────────────────────────────
START_STEP=1
ONLY_STEP=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --step) START_STEP=$2; shift 2;;
        --only) ONLY_STEP=$2; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
should_run() {
    local step=$1
    if [[ -n "$ONLY_STEP" ]]; then
        [[ "$step" == "$ONLY_STEP" ]]
    else
        [[ "$step" -ge "$START_STEP" ]]
    fi
}

log() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  STEP $1: $2"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════════"
}

run_eval() {
    # run_eval DATASET GOLD_FILE PRED_FILE MODEL PROMPT_ID PROMPT_TYPE
    local dataset=$1 gold=$2 pred=$3 model=$4 pid=$5 ptype=$6
    local split
    split=$(basename "$gold" .json)
    local model_slug
    model_slug=$(echo "$model" | tr '/' '_' | tr '-' '_' | tr '[:upper:]' '[:lower:]')
    local csv_out="${RESULTS_DIR}/${split}_${model_slug}_${ptype}_scores.csv"

    echo "  Evaluating: $model / $ptype on $dataset ($split)"
    python3 "${CODE_DIR}/evaluate_muc.py" \
        --gold "$gold" \
        --dataset "$dataset" \
        --predictions "$pred" \
        --model "$model" \
        --prompt_id "$pid" \
        --prompt_type "$ptype" \
        --member "$MEMBER" \
        --output_csv "$csv_out"
}

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$PARSED_DATA" "$RESULTS_DIR" "$LOGS_DIR"
echo "Pipeline starting. Code: $CODE_DIR  Results: $RESULTS_DIR"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"


# ── Step 1: Parse MUC-3/4 ────────────────────────────────────────────────────
if should_run 1; then
    log 1 "Parse MUC-3/4 dataset"
    python3 "${CODE_DIR}/parse_muc34.py" \
        --data_dir "$RAW_DATA_MUC4" \
        --output_dir "$PARSED_DATA" \
        --splits tst3 tst4 dev \
        --stats
    echo "  MUC-4 parsing done."
fi


# ── Step 2: Parse MUC-6 ──────────────────────────────────────────────────────
if should_run 2; then
    log 2 "Parse MUC-6 dataset"
    python3 "${CODE_DIR}/parse_muc6.py" \
        --data_dir "$RAW_DATA_MUC6" \
        --output_dir "$PARSED_DATA" \
        --stats
    echo "  MUC-6 parsing done."
fi


# ── Step 3: Baselines ─────────────────────────────────────────────────────────
if should_run 3; then
    log 3 "Run baselines (empty + majority)"

    for split in tst3 tst4; do
        gold="${PARSED_DATA}/muc4_${split}.json"

        # Empty baseline
        python3 "${CODE_DIR}/evaluate_muc.py" \
            --gold "$gold" --dataset muc4 --baseline empty \
            --member "$MEMBER" \
            --output_csv "${RESULTS_DIR}/muc4_${split}_empty_baseline_scores.csv"

        # Majority baseline
        python3 "${CODE_DIR}/evaluate_muc.py" \
            --gold "$gold" --dataset muc4 --baseline majority \
            --member "$MEMBER" \
            --output_csv "${RESULTS_DIR}/muc4_${split}_majority_baseline_scores.csv"
    done

    # MUC-6 empty baseline
    python3 "${CODE_DIR}/evaluate_muc.py" \
        --gold "${PARSED_DATA}/muc6_test.json" --dataset muc6 --baseline empty \
        --member "$MEMBER" \
        --output_csv "${RESULTS_DIR}/muc6_test_empty_baseline_scores.csv"

    echo "  Baselines done."
fi


# ── Step 4: Qwen zero-shot on MUC-4 ──────────────────────────────────────────
if should_run 4; then
    log 4 "Qwen2.5-7B zero-shot on MUC-4 (tst3 + tst4)"

    for split in tst3 tst4; do
        gold="${PARSED_DATA}/muc4_${split}.json"
        pred_file="${RESULTS_DIR}/muc4_${split}_qwen2_5_7b_instruct_zero_shot.jsonl"

        python3 "${CODE_DIR}/muc_model_eval.py" \
            --dataset muc4 \
            --data_file "$gold" \
            --model "$QWEN_MODEL" \
            --prompt_type zero_shot \
            --output_dir "$RESULTS_DIR" \
            --max_new_tokens 512

        run_eval muc4 "$gold" "$pred_file" "$QWEN_MODEL" P1 zero_shot
    done
fi


# ── Step 5: Qwen few-shot on MUC-4 ───────────────────────────────────────────
if should_run 5; then
    log 5 "Qwen2.5-7B few-shot on MUC-4 (tst3 + tst4)"
    FEW_SHOT_SRC="${PARSED_DATA}/muc4_dev.json"

    for split in tst3 tst4; do
        gold="${PARSED_DATA}/muc4_${split}.json"
        pred_file="${RESULTS_DIR}/muc4_${split}_qwen2_5_7b_instruct_few_shot.jsonl"

        python3 "${CODE_DIR}/muc_model_eval.py" \
            --dataset muc4 \
            --data_file "$gold" \
            --few_shot_file "$FEW_SHOT_SRC" \
            --n_few_shot 2 \
            --model "$QWEN_MODEL" \
            --prompt_type few_shot \
            --output_dir "$RESULTS_DIR" \
            --max_new_tokens 512

        run_eval muc4 "$gold" "$pred_file" "$QWEN_MODEL" P2 few_shot
    done
fi


# ── Step 6: Llama zero-shot on MUC-4 ─────────────────────────────────────────
if should_run 6; then
    log 6 "Llama-3.1-8B zero-shot on MUC-4 (tst3 + tst4)"

    for split in tst3 tst4; do
        gold="${PARSED_DATA}/muc4_${split}.json"
        pred_file="${RESULTS_DIR}/muc4_${split}_meta_llama_llama_3_1_8b_instruct_zero_shot.jsonl"

        python3 "${CODE_DIR}/muc_model_eval.py" \
            --dataset muc4 \
            --data_file "$gold" \
            --model "$LLAMA_MODEL" \
            --prompt_type zero_shot \
            --output_dir "$RESULTS_DIR" \
            --max_new_tokens 512

        run_eval muc4 "$gold" "$pred_file" "$LLAMA_MODEL" P1 zero_shot
    done
fi


# ── Step 7: Llama few-shot on MUC-4 ──────────────────────────────────────────
if should_run 7; then
    log 7 "Llama-3.1-8B few-shot on MUC-4 (tst3 + tst4)"
    FEW_SHOT_SRC="${PARSED_DATA}/muc4_dev.json"

    for split in tst3 tst4; do
        gold="${PARSED_DATA}/muc4_${split}.json"
        pred_file="${RESULTS_DIR}/muc4_${split}_meta_llama_llama_3_1_8b_instruct_few_shot.jsonl"

        python3 "${CODE_DIR}/muc_model_eval.py" \
            --dataset muc4 \
            --data_file "$gold" \
            --few_shot_file "$FEW_SHOT_SRC" \
            --n_few_shot 2 \
            --model "$LLAMA_MODEL" \
            --prompt_type few_shot \
            --output_dir "$RESULTS_DIR" \
            --max_new_tokens 512

        run_eval muc4 "$gold" "$pred_file" "$LLAMA_MODEL" P2 few_shot
    done
fi


# ── Step 8: Qwen + Llama zero-shot on MUC-6 ──────────────────────────────────
if should_run 8; then
    log 8 "Qwen + Llama zero-shot on MUC-6"
    gold="${PARSED_DATA}/muc6_test.json"

    for model in "$QWEN_MODEL" "$LLAMA_MODEL"; do
        model_slug=$(echo "$model" | tr '/' '_' | tr '-' '_' | tr '[:upper:]' '[:lower:]')
        pred_file="${RESULTS_DIR}/muc6_test_${model_slug}_zero_shot.jsonl"

        python3 "${CODE_DIR}/muc_model_eval.py" \
            --dataset muc6 \
            --data_file "$gold" \
            --model "$model" \
            --prompt_type zero_shot \
            --output_dir "$RESULTS_DIR" \
            --max_new_tokens 768

        run_eval muc6 "$gold" "$pred_file" "$model" P1 zero_shot
    done
fi


# ── Done ──────────────────────────────────────────────────────────────────────
log "✓" "Pipeline complete"
echo ""
echo "Results in: $RESULTS_DIR"
echo "CSV files:"
ls -lh "${RESULTS_DIR}"/*.csv 2>/dev/null || echo "  (none yet)"
