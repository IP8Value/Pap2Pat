#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenizer-based metrics (needs a tokenizer, but NOT a full model forward pass).

Backends:
- hf: HuggingFace AutoTokenizer (requires `pip install transformers`)
- tiktoken: OpenAI/GPT tokenizers (requires `pip install tiktoken`)

Features:
- Token count
- Token-level RR / RR>80 (n-gram repetition over token ids)
- Optionally dump first N decoded tokens for inspection

Usage (HF):
  python token_metrics.py --gen generated.md --ref patent.md --tokenizer hf --model meta-llama/Meta-Llama-3-8B-Instruct

Usage (tiktoken):
  python token_metrics.py --gen generated.md --ref patent.md --tokenizer tiktoken --encoding cl100k_base

Pred-dir mode:
  python token_metrics.py --pred_dir /path/to/<model>/pred_test
"""

from __future__ import annotations
import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

from pred_dir_utils import infer_csv_path, iter_pred_samples, load_or_init_csv, upsert_metrics, write_csv

_WS = re.compile(r"\s+")

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def clean_md(text: str) -> str:
    text = re.sub(r"[`*_>#]+", " ", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    text = _WS.sub(" ", text).strip()
    return text

def get_ngrams(tokens: List[int], n: int):
    """Get n-grams from token list (same as paper)"""
    i = 0
    while len(tokens) >= i + n:
        yield tuple(tokens[i : i + n])
        i += 1

def get_rr_score(tokens: List[int], max_n: int = 4) -> float:
    """
    Compute RR score (same as paper's RR class).
    RR = geometric mean of (1 - singleton_ratio) for n-grams 1 to max_n-1
    """
    if len(tokens) == 0:
        return 0.0
    rr = 1.0
    for n in range(1, max_n):
        n_gram_counts = Counter(get_ngrams(tokens, n))
        n_singleton = sum(count == 1 for count in n_gram_counts.values())
        n_total = len(n_gram_counts)
        if n_total == 0:
            rrn = 0.0
        else:
            rrn = (n_total - n_singleton) / n_total
        rr *= rrn
    return rr ** (1 / max_n) if max_n > 0 else 0.0

# -------- HF backend --------

def load_hf_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained("gpt2", use_fast=True)

def encode_hf(tok, text: str) -> List[int]:
    return tok.encode(text, add_special_tokens=False)

def decode_token_hf(tok, token_id: int) -> str:
    return tok.decode([token_id])

# -------- tiktoken backend --------

def load_tiktoken(encoding_name: str):
    import tiktoken
    return tiktoken.get_encoding(encoding_name)

def encode_tiktoken(enc, text: str) -> List[int]:
    return enc.encode(text)

def decode_token_tiktoken(enc, token_id: int) -> str:
    return enc.decode([token_id])

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

def compute_one(gen_path: Path, ref_path: Path, backend: str, tok: Any, max_n: int = 4, dump_n: int = 0) -> Dict[str, object]:
    gen = clean_md(read_text(gen_path))
    ref = clean_md(read_text(ref_path))

    if backend == "hf":
        gen_ids = encode_hf(tok, gen)
        ref_ids = encode_hf(tok, ref)
        decode_one = lambda i: decode_token_hf(tok, i)
    else:
        gen_ids = encode_tiktoken(tok, gen)
        ref_ids = encode_tiktoken(tok, ref)
        decode_one = lambda i: decode_token_tiktoken(tok, i)

    # Compute RR (same as paper: geometric mean of n-gram repetition rates)
    gen_rr = get_rr_score(gen_ids, max_n=max_n)
    ref_rr = get_rr_score(ref_ids, max_n=max_n)

    # Tokens (same as paper)
    gen_token_count = len(gen_ids)
    ref_token_count = len(ref_ids)
    token_fraction = gen_token_count / ref_token_count if ref_token_count > 0 else 0.0

    row: Dict[str, object] = {
        "token_backend": backend,
        "token_gen_count": gen_token_count,
        "token_ref_count": ref_token_count,
        "token_fraction": token_fraction,
        "token_rr_generated": gen_rr,
        "token_rr_reference": ref_rr,
    }

    if dump_n > 0:
        row["token_gen_first_tokens"] = " | ".join(decode_one(i) for i in gen_ids[:dump_n])
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=str, default=None)
    ap.add_argument("--ref", type=str, default=None)
    ap.add_argument("--gen_dir", type=str, default=None)
    ap.add_argument("--ref_dir", type=str, default=None)
    ap.add_argument("--out", type=str, default="metrics_token.csv")
    ap.add_argument("--pred_dir", type=str, default=None, help="Prediction root (contains <id>/generated.md and <id>/patent.md or reference.md)")
    ap.add_argument("--out_csv", type=str, default=None, help="Output merged CSV path (default: auto infer <model>-res.csv under outputs/single-llm-call/)")

    ap.add_argument("--tokenizer", type=str, choices=["hf", "tiktoken"], default="hf")
    ap.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                    help="HF tokenizer name (used when --tokenizer hf)")
    ap.add_argument("--encoding", type=str, default="cl100k_base",
                    help="tiktoken encoding name (used when --tokenizer tiktoken)")
    ap.add_argument("--max_n", type=int, default=4, help="Max n-gram for RR computation (same as paper, default=4)")
    ap.add_argument("--dump_tokens", type=int, default=0, help="Dump first N decoded tokens for inspection")
    args = ap.parse_args()

    if args.tokenizer == "hf":
        tok = load_hf_tokenizer(args.model)
    else:
        tok = load_tiktoken(args.encoding)

    # New preferred mode: --pred_dir (updates a single <model>-res.csv)
    if args.pred_dir:
        pred_root = Path(args.pred_dir)
        out_csv = Path(args.out_csv) if args.out_csv else infer_csv_path(pred_root)
        fieldnames, rows_by_id = load_or_init_csv(out_csv, id_col="id")

        n = 0
        for s in iter_pred_samples(pred_root):
            metrics = compute_one(s.gen_path, s.ref_path, args.tokenizer, tok, dump_n=args.dump_tokens)
            fieldnames = upsert_metrics(fieldnames, rows_by_id, s.sample_id, metrics, id_col="id")
            write_csv(out_csv, fieldnames, rows_by_id, id_col="id")
            n += 1
            print(f"{s.sample_id}\t" + "\t".join(f"{k}={metrics[k]}" for k in sorted(metrics.keys())), flush=True)

        print(f"Wrote/updated {n} samples -> {out_csv}")
        return

    # Backwards compatible modes
    rows = []
    if args.gen and args.ref:
        rows.append(compute_one(Path(args.gen), Path(args.ref), args.tokenizer, tok, max_n=args.max_n, dump_n=args.dump_tokens))
    elif args.gen_dir and args.ref_dir:
        pairs = pair_files(Path(args.gen_dir), Path(args.ref_dir))
        if not pairs:
            raise SystemExit("No matched file pairs found. Ensure filenames (or stems) match.")
        for g, r in pairs:
            rows.append(compute_one(g, r, args.tokenizer, tok, max_n=args.max_n, dump_n=args.dump_tokens))
    elif args.pred_dir:
        pred_root = Path(args.pred_dir)
        out_csv = Path(args.out_csv) if args.out_csv else infer_csv_path(pred_root)
        fieldnames, rows_by_id = load_or_init_csv(out_csv, id_col="id")

        n = 0
        for s in iter_pred_samples(pred_root):
            metrics = compute_one(s.gen_path, s.ref_path, args.tokenizer, tok, max_n=args.max_n, dump_n=args.dump_tokens)
            fieldnames = upsert_metrics(fieldnames, rows_by_id, s.sample_id, metrics, id_col="id")
            write_csv(out_csv, fieldnames, rows_by_id, id_col="id")
            n += 1
            print(f"{s.sample_id}\t" + "\t".join(f"{k}={metrics[k]}" for k in sorted(metrics.keys())), flush=True)

        print(f"Wrote/updated {n} samples -> {out_csv}")
        return
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
