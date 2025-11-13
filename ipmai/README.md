# ipmai Utilities

This package contains standalone scripts for running Ollama-based experiments on the Pap2Pat dataset without modifying the original repository code.

## Prerequisites
- Install project dependencies (preferably via `conda env create -f environment.yml`)
- Activate the environment: `conda activate pap2pat` (or source your venv)
- Ensure your Ollama service is running and accessible via HTTP (`--ollama_api_base`)
- (Optional) set `PYTHONPATH` if running from a different working directory:
  ```bash
  export PYTHONPATH="./Outline_Guided_Generation/src:${PYTHONPATH}"
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
    --max_samples 3 \
    --run_dir "outputs/runs/ollama-single-test"
```
- Writes `generated.md`, `reference.md`, `outline.md`, `paper.md` for each sample
- Useful for replicating the “Single LLM-call” baseline in the paper

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

