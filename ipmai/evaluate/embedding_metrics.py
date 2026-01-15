#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document-level embedding similarity (NOT in paper, but useful as additional metric).

This computes document-level embeddings (one vector per document), NOT token-level like BERTScore.
- BERTScore: each token gets an embedding, then token-level alignment (see bertscore_metrics.py)
- This script: entire document -> one embedding vector -> cosine similarity

Two backends:
A) OpenAI-compatible Embeddings API (recommended for Qwen/百炼):
   - Uses `openai` python SDK (>=1.0)
   - You set: --base_url, --api_key, --model (e.g., text-embedding-v4)
B) Local HF embedding model (if you prefer offline):
   - Uses transformers AutoModel + mean pooling

Scores:
- doc_cosine: cosine similarity between *document embeddings* (pooled, one vector per doc)
- chunk_cosine_mean: split long doc into chunks, embed each chunk, and average best-match similarity

Note: This is NOT the same as BERTScore (which is token-level alignment).
      For BERTScore (as in paper), use bertscore_metrics.py instead.

Usage (Bailian/Qwen via OpenAI-compatible):
  export BAILIAN_API_KEY="..."
  python embedding_metrics.py --gen generated.md --ref patent.md \
    --backend api --base_url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --api_key $BAILIAN_API_KEY --model text-embedding-v4

Usage (local HF):
  python embedding_metrics.py --gen generated.md --ref patent.md \
    --backend hf --hf_model sentence-transformers/all-MiniLM-L6-v2

Pred-dir mode:
  python embedding_metrics.py --pred_dir /path/to/<model>/pred_test --backend api --api_key "$BAILIAN_API_KEY"
"""

from __future__ import annotations
import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from pred_dir_utils import infer_csv_path, iter_pred_samples, load_or_init_csv, upsert_metrics, write_csv

_WS = re.compile(r"\s+")

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def clean_md(text: str) -> str:
    text = re.sub(r"[`*_>#]+", " ", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    text = _WS.sub(" ", text).strip()
    return text

def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def chunk_text(text: str, chunk_chars: int = 4000, overlap_chars: int = 400) -> List[str]:
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_chars)
        chunks.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap_chars)
    return chunks

# -------- API backend (OpenAI-compatible) --------

def embed_api(texts: List[str], base_url: str, api_key: str, model: str) -> List[List[float]]:
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)
    out = []
    # Batch in small groups to avoid payload limits
    bs = 16
    for i in range(0, len(texts), bs):
        resp = client.embeddings.create(model=model, input=texts[i:i+bs])
        out.extend([d.embedding for d in resp.data])
    return out

# -------- HF backend (local) --------

def embed_hf(texts: List[str], hf_model: str) -> List[List[float]]:
    import torch
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(hf_model, use_fast=True)
    mdl = AutoModel.from_pretrained(hf_model)
    mdl.eval()

    out = []
    with torch.no_grad():
        for t in texts:
            inputs = tok(t, return_tensors="pt", truncation=True, max_length=512)
            h = mdl(**inputs).last_hidden_state  # [1, L, H]
            mask = inputs["attention_mask"].unsqueeze(-1)  # [1, L, 1]
            h = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # mean pooling
            out.append(h.squeeze(0).cpu().tolist())
    return out

def pair_files(gen_dir: Path, ref_dir: Path):
    gen_files = [p for p in gen_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt"}]
    ref_map_name = {p.name: p for p in ref_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt"}}
    ref_map_stem = {p.stem: p for p in ref_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt"}}
    pairs = []
    for g in gen_files:
        r = ref_map_name.get(g.name) or ref_map_stem.get(g.stem)
        if r:
            pairs.append((g, r))
    return pairs

def compute_one(gen_path: Path, ref_path: Path, backend: str, api_conf: dict, hf_model: str,
                chunk_chars: int, overlap_chars: int) -> Dict[str, object]:
    gen = clean_md(read_text(gen_path))
    ref = clean_md(read_text(ref_path))

    # Document-level embedding: embed truncated representative text (first ~chunk_chars)
    gen_doc = gen[:chunk_chars]
    ref_doc = ref[:chunk_chars]

    # Chunking for long-doc robust similarity
    gen_chunks = chunk_text(gen, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    ref_chunks = chunk_text(ref, chunk_chars=chunk_chars, overlap_chars=overlap_chars)

    if backend == "api":
        gen_vec = embed_api([gen_doc], **api_conf)[0]
        ref_vec = embed_api([ref_doc], **api_conf)[0]
        # For chunks, embed all and do best-match averaging (IR-ish)
        gen_chunk_vecs = embed_api(gen_chunks, **api_conf)
        ref_chunk_vecs = embed_api(ref_chunks, **api_conf)
    else:
        gen_vec = embed_hf([gen_doc], hf_model=hf_model)[0]
        ref_vec = embed_hf([ref_doc], hf_model=hf_model)[0]
        gen_chunk_vecs = embed_hf(gen_chunks, hf_model=hf_model)
        ref_chunk_vecs = embed_hf(ref_chunks, hf_model=hf_model)

    doc_cos = cosine(gen_vec, ref_vec)

    # best-match similarity: for each gen chunk, find max cosine over ref chunks, then average
    bests = []
    for gv in gen_chunk_vecs:
        m = max(cosine(gv, rv) for rv in ref_chunk_vecs) if ref_chunk_vecs else 0.0
        bests.append(m)
    chunk_cos_mean = sum(bests)/len(bests) if bests else 0.0

    return {
        "embed_backend": backend,
        "embed_model": api_conf.get("model") if backend == "api" else hf_model,
        "embed_doc_cosine": doc_cos,
        "embed_chunk_cosine_mean": chunk_cos_mean,
        "embed_gen_chunks": len(gen_chunks),
        "embed_ref_chunks": len(ref_chunks),
        "embed_chunk_chars": chunk_chars,
        "embed_overlap_chars": overlap_chars,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=str, default=None)
    ap.add_argument("--ref", type=str, default=None)
    ap.add_argument("--gen_dir", type=str, default=None)
    ap.add_argument("--ref_dir", type=str, default=None)
    ap.add_argument("--out", type=str, default="metrics_embedding.csv")
    ap.add_argument("--pred_dir", type=str, default=None, help="Prediction root (contains <id>/generated.md and <id>/patent.md or reference.md)")
    ap.add_argument("--out_csv", type=str, default=None, help="Output merged CSV path (default: auto infer <model>-res.csv under outputs/single-llm-call/)")

    ap.add_argument("--backend", type=str, choices=["api", "hf"], default="api")
    ap.add_argument("--chunk_chars", type=int, default=4000)
    ap.add_argument("--overlap_chars", type=int, default=400)

    # API (OpenAI-compatible)
    ap.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    ap.add_argument("--api_key", type=str, default=None)
    ap.add_argument("--model", type=str, default="text-embedding-v4")

    # HF local
    ap.add_argument("--hf_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")

    args = ap.parse_args()

    rows = []
    api_conf = {"base_url": args.base_url, "api_key": args.api_key, "model": args.model}

    if args.backend == "api" and not args.api_key:
        raise SystemExit("For --backend api, you must pass --api_key.")

    # New preferred mode: --pred_dir (updates a single <model>-res.csv)
    if args.pred_dir:
        pred_root = Path(args.pred_dir)
        out_csv = Path(args.out_csv) if args.out_csv else infer_csv_path(pred_root)
        fieldnames, rows_by_id = load_or_init_csv(out_csv, id_col="id")

        n = 0
        for s in iter_pred_samples(pred_root):
            metrics = compute_one(
                s.gen_path,
                s.ref_path,
                args.backend,
                api_conf,
                args.hf_model,
                args.chunk_chars,
                args.overlap_chars,
            )
            fieldnames = upsert_metrics(fieldnames, rows_by_id, s.sample_id, metrics, id_col="id")
            write_csv(out_csv, fieldnames, rows_by_id, id_col="id")
            n += 1
            print(f"{s.sample_id}\t" + "\t".join(f"{k}={metrics[k]}" for k in sorted(metrics.keys())), flush=True)

        print(f"Wrote/updated {n} samples -> {out_csv}")
        return

    # Backwards compatible modes
    if args.gen and args.ref:
        rows.append(
            compute_one(
                Path(args.gen),
                Path(args.ref),
                args.backend,
                api_conf,
                args.hf_model,
                args.chunk_chars,
                args.overlap_chars,
            )
        )
    elif args.gen_dir and args.ref_dir:
        pairs = pair_files(Path(args.gen_dir), Path(args.ref_dir))
        if not pairs:
            raise SystemExit("No matched file pairs found. Ensure filenames (or stems) match.")
        for g, r in pairs:
            rows.append(
                compute_one(g, r, args.backend, api_conf, args.hf_model, args.chunk_chars, args.overlap_chars)
            )
    else:
        raise SystemExit("Provide either (--pred_dir) or (--gen --ref) or (--gen_dir --ref_dir).")

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"Wrote {len(rows)} rows to {out_p}")

if __name__ == "__main__":
    main()
