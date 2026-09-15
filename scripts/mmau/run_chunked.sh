#!/usr/bin/env bash
# Run audio chunking experiments for a given model on MMAU.
#
# Runs pairs of chunk experiments in parallel across 2 GPUs.
# N=1 (full audio) is assumed to already exist as full.jsonl.
#
# Usage:
#   bash scripts/mmau/run_chunked.sh <model_id> <results_dir> [extra_args...]
#
# Set INFERENCE_SCRIPT env var to use a different inference script
# (default: scripts/mmau/run_inference.py).
#
# Examples:
#   bash scripts/mmau/run_chunked.sh \
#     Qwen/Qwen2.5-Omni-7B \
#     results/mmau/qwen2.5_omni_7b \
#     --batch_size 1 --max_model_len 4096
#
#   INFERENCE_SCRIPT=scripts/mmau/run_flamingo.py bash scripts/mmau/run_chunked.sh \
#     nvidia/audio-flamingo-3-hf \
#     results/mmau/audio_flamingo_3

set -euo pipefail

INFERENCE_SCRIPT="${INFERENCE_SCRIPT:-scripts/mmau/run_inference.py}"

MODEL_ID="${1:?Usage: $0 <model_id> <results_dir> [extra_args...]}"
RESULTS_DIR="${2:?Usage: $0 <model_id> <results_dir> [extra_args...]}"
shift 2
EXTRA_ARGS=("$@")

echo "Script: ${INFERENCE_SCRIPT}"
echo "Model: ${MODEL_ID}"
echo "Results dir: ${RESULTS_DIR}"
echo "Extra args: ${EXTRA_ARGS[*]:-}"
echo "GPUs: 0, 1 (parallel execution)"
echo ""

# Build list of all (N, K) jobs
JOBS=()
for N in 2 3 4 5; do
    for K in $(seq 0 $((N - 1))); do
        JOBS+=("${N}:${K}")
    done
done

echo "Total jobs: ${#JOBS[@]}"
echo "Jobs: ${JOBS[*]}"
echo ""

# Run a single chunk job on a specific GPU
run_job() {
    local GPU=$1
    local N=$2
    local K=$3
    local OUTPUT="${RESULTS_DIR}/n${N}_chunk${K}.jsonl"
    echo "[GPU ${GPU}] Starting N=${N}, chunk=${K} -> ${OUTPUT}"
    CUDA_VISIBLE_DEVICES=${GPU} python "${INFERENCE_SCRIPT}" \
        --model_id "${MODEL_ID}" \
        --output_path "${OUTPUT}" \
        --audio_condition full \
        --use_mini \
        --num_chunks "${N}" \
        --chunk_index "${K}" \
        "${EXTRA_ARGS[@]}"
    echo "[GPU ${GPU}] Done: ${OUTPUT}"
}

i=0
while [ $i -lt ${#JOBS[@]} ]; do
    # Parse job 1 (GPU 0)
    IFS=':' read -r N1 K1 <<< "${JOBS[$i]}"

    if [ $((i + 1)) -lt ${#JOBS[@]} ]; then
        # Parse job 2 (GPU 1) — run both in parallel
        IFS=':' read -r N2 K2 <<< "${JOBS[$((i + 1))]}"
        echo "=== Parallel: GPU0=(N=${N1},K=${K1}) + GPU1=(N=${N2},K=${K2}) ==="
        run_job 0 "${N1}" "${K1}" &
        PID0=$!
        run_job 1 "${N2}" "${K2}" &
        PID1=$!
        wait $PID0 || echo "[WARN] GPU0 job (N=${N1},K=${K1}) failed"
        wait $PID1 || echo "[WARN] GPU1 job (N=${N2},K=${K2}) failed"
        i=$((i + 2))
    else
        # Odd job out — run on GPU 0 alone
        echo "=== Solo: GPU0=(N=${N1},K=${K1}) ==="
        run_job 0 "${N1}" "${K1}" || echo "[WARN] GPU0 job (N=${N1},K=${K1}) failed"
        i=$((i + 1))
    fi
    echo ""
done

echo "All chunking experiments complete for ${MODEL_ID}"
