# Metrics Scripts (Aligned with Paper)

Three categories of metrics, matching the Pap2Pat paper implementation:

## 1. Simple Metrics (No Models/Tools) - `simple_metrics.py`

**Same as paper:**
- **ROUGE-1, 2, 3, 4, L** (F1, Precision, Recall) - using `rouge_metric` library` with numba JIT optimization
- **BLEU** - using `sacrebleu` library
- Basic stats: chars, lines, whitespace tokens
- Jaccard similarity (whitespace tokens)
- Simple RR (whitespace tokens, sliding window)

**Note:** Paper also computes ROUGE by section clusters (Background/Summary/Detailed Description), but this script computes overall ROUGE for simplicity. You can add section clustering later if needed.

## 2. Tokenizer-based Metrics - `token_metrics.py`

**Same as paper:**
- **Tokens**: token counts using real tokenizer (not whitespace)
  - `generated`, `reference`, `fraction` (same as paper's `Tokens` class)
- **RR (Repetition Rate)**: using real tokenizer tokens
  - `generated`, `reference` RR scores (same as paper's `RR` class)
  - Uses geometric mean of n-gram repetition rates (n=1 to max_n-1)

**Configurable:**
- Tokenizer: HuggingFace (default: `meta-llama/Meta-Llama-3-8B-Instruct`) or tiktoken
- Max n-gram for RR: `--max_n` (default=4, same as paper)

## 3. Embedding-based Metrics

### 3a. BERTScore (Token-level) - `bertscore_metrics.py` ⭐ **Paper's BERTScore**

**Same as paper:**
- **BERTScore**: token-level alignment using Transformer encoder embeddings
- Each token gets an embedding, then greedy cosine matching
- Returns P, R, F1 (same as paper)

**Configurable:**
- Model: `--model_id` (default: `allenai/scibert_scivocab_uncased`, same as paper)
- Device: `--device` (default: cuda:0 if available)

### 3b. Document Embedding - `embedding_metrics.py` (Optional, NOT in paper)

**NOT in paper, but useful as additional metric:**
- Document-level embeddings (one vector per document)
- Cosine similarity between document embeddings
- Chunked cosine mean for long documents

**Use this if you want document-level similarity (different from BERTScore's token-level alignment).**

---

## Usage

### Pred-dir Mode (Recommended)

Point each script at a predictions folder, and it will:
- Auto-discover all sample folders (containing `generated.md`)
- Pair `generated.md` with `patent.md` (or `reference.md` for Ollama)
- Compute one row per patent id
- Write/update one CSV: `<model>-res.csv` under `outputs/single-llm-call/`
- Print progress after each sample

#### Example: DeepSeek V3

```bash
# 1. Simple metrics (ROUGE, BLEU, basic stats)
python ipmai/evaluate/simple_metrics.py \
  --pred_dir "/Users/kevin/project_python/Pap2Pat/outputs/single-llm-call/deepseek-v3/pred_test"

# 2. Tokenizer metrics (Tokens, RR)
python ipmai/evaluate/token_metrics.py \
  --pred_dir "/Users/kevin/project_python/Pap2Pat/outputs/single-llm-call/deepseek-v3/pred_test" \
  --tokenizer hf \
  --model meta-llama/Meta-Llama-3-8B-Instruct

# 3a. BERTScore (token-level, same as paper)
python ipmai/evaluate/bertscore_metrics.py \
  --pred_dir "/Users/kevin/project_python/Pap2Pat/outputs/single-llm-call/deepseek-v3/pred_test" \
  --model_id allenai/scibert_scivocab_uncased \
  --device cuda:0

# 3b. Document embedding (optional, NOT in paper)
python ipmai/evaluate/embedding_metrics.py \
  --pred_dir "/Users/kevin/project_python/Pap2Pat/outputs/single-llm-call/deepseek-v3/pred_test" \
  --backend api \
  --base_url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --api_key "$BAILIAN_API_KEY" \
  --model text-embedding-v4
```

All scripts will write/update:
`/Users/kevin/project_python/Pap2Pat/outputs/single-llm-call/deepseek-v3-res.csv`

#### Example: Ollama Qwen2.5

```bash
python ipmai/evaluate/simple_metrics.py \
  --pred_dir "/Users/kevin/project_python/Pap2Pat/outputs/single-llm-call/ollama-qwen2.5/predictions"

python ipmai/evaluate/token_metrics.py \
  --pred_dir "/Users/kevin/project_python/Pap2Pat/outputs/single-llm-call/ollama-qwen2.5/predictions" \
  --tokenizer hf --model meta-llama/Meta-Llama-3-8B-Instruct

python ipmai/evaluate/bertscore_metrics.py \
  --pred_dir "/Users/kevin/project_python/Pap2Pat/outputs/single-llm-call/ollama-qwen2.5/predictions" \
  --model_id allenai/scibert_scivocab_uncased --device cuda:0
```

This will pair `generated.md` with `reference.md` (Ollama format).

---

## Differences from Paper

**What's the same:**
- ✅ ROUGE (F/P/R for 1,2,3,4,L) - same library, same computation
- ✅ BLEU - same library (sacrebleu)
- ✅ Tokens - same algorithm (tokenizer-based counts)
- ✅ RR - same algorithm (geometric mean of n-gram repetition)
- ✅ BERTScore - same library, same token-level alignment

**What's different:**
- ❌ Section clustering: Paper computes ROUGE/BLEU by section clusters (Background/Summary/Detailed Description). Our scripts compute overall metrics for simplicity.
- ⚠️ Model flexibility: You can use different tokenizers/models/embeddings (configurable), but the algorithms are the same.

**What's extra:**
- ➕ Document embedding cosine (not in paper, but useful)
- ➕ Jaccard similarity (not in paper, but useful)

---

## Notes

- All scripts support `--pred_dir` mode for batch processing
- Each script appends columns to the same CSV (one row per patent id)
- Scripts are designed to be run multiple times (they update existing CSVs)
- You can use different models/tokenizers/embeddings, but the algorithms match the paper
