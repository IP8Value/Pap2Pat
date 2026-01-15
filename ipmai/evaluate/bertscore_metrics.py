#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERTScore: token-level alignment using Transformer encoder embeddings.

Same as paper: uses bert_score library with token-level greedy matching.
Each token gets an embedding, then we do greedy cosine matching.

Usage:
  python bertscore_metrics.py --pred_dir /path/to/<model>/pred_test \
    --model_id allenai/scibert_scivocab_uncased --device cuda:0

Note: This is different from embedding_metrics.py which does document-level embedding.
BERTScore is token-level: each token has an embedding, then we align tokens.
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict

import bert_score
import torch

from pred_dir_utils import infer_csv_path, iter_pred_samples, load_or_init_csv, upsert_metrics, write_csv

_WS = re.compile(r"\s+")

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def clean_md(text: str) -> str:
    text = re.sub(r"[`*_>#]+", " ", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    text = _WS.sub(" ", text).strip()
    return text

def compute_one(gen_path: Path, ref_path: Path, scorer: bert_score.BERTScorer) -> Dict[str, float]:
    """
    Compute BERTScore (same as paper).
    Returns P, R, F1 for token-level alignment.
    """
    gen_text = clean_md(read_text(gen_path))
    ref_text = clean_md(read_text(ref_path))

    # BERTScore expects lists of strings
    scores = scorer.score([gen_text], [ref_text], batch_size=1)
    
    # scores is a tuple of (P, R, F1) tensors
    precision = scores[0].item()
    recall = scores[1].item()
    f1 = scores[2].item()

    return {
        "bertscore_p": precision,
        "bertscore_r": recall,
        "bertscore_f1": f1,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=str, default=None, help="Generated file path (.md/.txt)")
    ap.add_argument("--ref", type=str, default=None, help="Reference file path (.md/.txt)")
    ap.add_argument("--gen_dir", type=str, default=None, help="Directory of generated files")
    ap.add_argument("--ref_dir", type=str, default=None, help="Directory of reference files")
    ap.add_argument("--out", type=str, default="metrics_bertscore.csv", help="Output CSV path")
    ap.add_argument("--pred_dir", type=str, default=None, help="Prediction root (contains <id>/generated.md and <id>/patent.md or reference.md)")
    ap.add_argument("--out_csv", type=str, default=None, help="Output merged CSV path (default: auto infer <model>-res.csv under outputs/single-llm-call/)")
    
    ap.add_argument("--model_id", type=str, default="allenai/scibert_scivocab_uncased",
                    help="BERT model for embeddings (same as paper: SciBERT)")
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                    help="Device for BERT model")
    ap.add_argument("--batch_size", type=int, default=1, help="Batch size for BERTScore")
    args = ap.parse_args()

    # Initialize BERTScore scorer (same as paper)
    print(f"Loading BERTScore model: {args.model_id} on {args.device}", flush=True)
    scorer = bert_score.BERTScorer(
        model_type=args.model_id,
        use_fast_tokenizer=True,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # Paper uses SciBERT with max_length=512
    if "scibert" in args.model_id.lower():
        scorer._tokenizer.model_max_length = 512

    # New preferred mode: --pred_dir (updates a single <model>-res.csv)
    if args.pred_dir:
        pred_root = Path(args.pred_dir)
        out_csv = Path(args.out_csv) if args.out_csv else infer_csv_path(pred_root)
        fieldnames, rows_by_id = load_or_init_csv(out_csv, id_col="id")

        n = 0
        for s in iter_pred_samples(pred_root):
            metrics = compute_one(s.gen_path, s.ref_path, scorer)
            fieldnames = upsert_metrics(fieldnames, rows_by_id, s.sample_id, metrics, id_col="id")
            write_csv(out_csv, fieldnames, rows_by_id, id_col="id")
            n += 1
            print(f"{s.sample_id}\t" + "\t".join(f"{k}={metrics[k]:.4f}" for k in sorted(metrics.keys())), flush=True)

        print(f"Wrote/updated {n} samples -> {out_csv}")
        return

    # Backwards compatible modes
    import csv
    rows = []
    if args.gen and args.ref:
        metrics = compute_one(Path(args.gen), Path(args.ref), scorer)
        rows.append({**metrics, "gen_file": str(args.gen), "ref_file": str(args.ref)})
    elif args.gen_dir and args.ref_dir:
        from pathlib import Path as P
        gen_files = [p for p in P(args.gen_dir).rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt"}]
        ref_map = {p.name: p for p in P(args.ref_dir).rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt"}}
        for g in gen_files:
            r = ref_map.get(g.name)
            if r:
                metrics = compute_one(g, r, scorer)
                rows.append({**metrics, "gen_file": str(g), "ref_file": str(r)})
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
