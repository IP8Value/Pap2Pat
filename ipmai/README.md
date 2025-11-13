# ipmai Utilities

This package contains standalone scripts for running Ollama-based experiments on the Pap2Pat dataset without modifying the original repository code.

## Prerequisites
- Install project dependencies (preferably via `conda env create -f environment.yml`)
- Activate the environment: `conda activate pap2pat` (or source your venv)
- Ensure your Ollama service is running and accessible via HTTP (`--ollama_api_base`)
- (Optional) set `PYTHONPATH` if running from a different working directory:
  ```bash
  export PYTHONPATH="./Outline_Guided_Generation/src:${PYTHONPATH}"
  export PYTHONPATH="./Outline_Guided_Generation:${PYTHONPATH}"
  ```

All scripts read data from `Pap2Pat/data/` and write results under the repository root `outputs/runs/` folder.

## Scripts

### 1. `generate_ollama.py`
Chunk-based outline-guided patent generation using Ollama (multi-call, mirrors `generate.py`).
```
python ipmai/generate_ollama.py \
    --ollama_api_base "http://<host>:<port>" \
    --ollama_model "qwen2.5:72b-instruct" \
    --data.outline_suffix "long" \
    --data.max_total_length 8192 \
    --temperature 0.6 \
    --max_samples 3 \
    --run_dir "outputs/runs/ollama-chunk-test"
```
- Generates per-chunk prompts; outputs under `run_dir/predictions/<split>/<sample_id>/`
- Recorded prompts in `conversations/chunk-*-system.md` and `chunk-*-user.md`

### 2. `generate_ollama_single.py`
Single LLM-call baseline: feed the entire paper and outline to one Ollama call.
```
python ipmai/generate_ollama_single.py \
    --ollama_api_base "http://<host>:<port>" \
    --ollama_model "qwen2.5:72b-instruct" \
    --outline_suffix "long" \
    --temperature 0.6 \
    --max_tokens 15000 \
    --max_samples 3 \
    --run_dir "outputs/runs/ollama-single-test"
```
- Writes `generated.md`, `reference.md`, `outline.md`, `paper.md` for each sample
- Useful for replicating the “Single LLM-call” baseline in the paper
- `--max_tokens` 控制单次调用允许的最大输出长度（token 数）。
  建议根据“输入 outline+paper 的长度”预留总上下文（例如 Qwen2.5 72B 支持 32k token，可设置 12000~15000 作为输出上限）。

### 3. `evaluate_subset.py`
Evaluate only the predictions you generated (no need to cover the full split).
```
python ipmai/evaluate_subset.py \
    --run_dir "outputs/runs/ollama-single-test" \
    --split "val" \
    --scores "bleu,rouge,tokens,rr" \
    --sample_ids "W2048923264-US20050037009,W4230060112-US20050065064"
```
- Omit `--sample_ids` to evaluate all IDs present in `predictions/<split>/`.
- Results are saved per-sample (in each prediction folder) and aggregated under `<run_dir>/metrics_subset.json`.

### Outline Options
`--outline_suffix` can be `long`, `medium`, `short`, or `empty`, corresponding to different prompt granularity (`patent_outline_*.md`).

## Evaluating Results
Use the official evaluator to compute BLEU, ROUGE, RR, Tokens, etc. Example:
```
cd Outline_Guided_Generation/src
python evaluate.py \
    --run_dir "../outputs/runs/ollama-single-test" \
    --splits "val" \
    --scores "bleu,rouge,tokens,rr"
```
Outputs are written to `metrics.json` and per-sample `metrics.json` under each prediction folder.

