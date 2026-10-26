#!/usr/bin/env bash
# Start vLLM's OpenAI-compatible server with the persona LoRA, using flags from configs/serve.yaml.
# Any value can be overridden via env (VLLM_MODEL, VLLM_LORA_PATH, VLLM_MAX_MODEL_LEN, ...).
#
# Usage: scripts/run_vllm.sh [extra vllm args]
set -euo pipefail

cfg() {  # read serve.vllm.<key> from YAML
  python3 -c "import yaml,sys; v=yaml.safe_load(open('configs/serve.yaml'))['vllm'].get('$1'); print('' if v is None else v)"
}

MODEL="${VLLM_MODEL:-$(cfg model)}"
QUANT="${VLLM_QUANTIZATION:-$(cfg quantization)}"
LORA_PATH="${VLLM_LORA_PATH:-$(cfg lora_path)}"
MAX_LORAS="${VLLM_MAX_LORAS:-$(cfg max_loras)}"
MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-$(cfg max_lora_rank)}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$(cfg max_model_len)}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-$(cfg gpu_memory_utilization)}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-$(cfg max_num_seqs)}"
PORT="${VLLM_PORT:-$(cfg port)}"

ARGS=(
  --model "$MODEL"
  --host 0.0.0.0 --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"            # caps KV-cache per sequence (main OOM lever)
  --gpu-memory-utilization "$GPU_UTIL"        # leave headroom for CUDA graphs / image service
  --max-num-seqs "$MAX_NUM_SEQS"              # concurrent sequences in a batch
  --enable-prefix-caching                     # persona system prompt + tool schemas are shared
  --enable-auto-tool-choice --tool-call-parser hermes
)
[ -n "$QUANT" ] && ARGS+=(--quantization "$QUANT")
if [ -f "$LORA_PATH/adapter_config.json" ]; then
  ARGS+=(--enable-lora --lora-modules "persona=$LORA_PATH" --max-loras "$MAX_LORAS" --max-lora-rank "$MAX_LORA_RANK")
else
  echo "[warn] no adapter at $LORA_PATH — serving the base model only (model name: $MODEL)" >&2
fi

echo "vllm serve ${ARGS[*]} $*"
exec vllm serve "${ARGS[@]}" "$@"
