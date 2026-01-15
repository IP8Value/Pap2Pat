#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple metrics for long documents using rouge_metric library (same as paper).

Supports:
- Tokens/words/chars/lines (whitespace tokens)
- ROUGE-1, 2, 3, 4, L (using rouge_metric library with numba JIT optimization)
- Repetition Rate (RR) and RR>80 (sliding window n-gram repetition)
- Jaccard overlap (token sets)

Usage:
  python simple_metrics.py --gen generated.md --ref patent.md
  python simple_metrics.py --gen_dir ./generated --ref_dir ./groundtruth --out metrics.csv
  python simple_metrics.py --pred_dir /path/to/<model>/pred_test
"""

from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import rouge_metric.py_rouge
from numba import jit
from rouge_metric.py_rouge import _lcs_elements
from rouge_metric import PyRouge

from pred_dir_utils import infer_csv_path, iter_pred_samples, load_or_init_csv, upsert_metrics, write_csv

_WS = re.compile(r"\s+")

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def ws_tokens(text: str) -> List[str]:
    # Keep it simple: split on whitespace; strip Markdown-ish noise lightly.
    text = re.sub(r"[`*_>#]+", " ", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)  # md links -> anchor text
    text = _WS.sub(" ", text).strip()
    return text.split(" ") if text else []

# Patch rouge_metric with JIT-compiled LCS (same as paper)
@jit(nopython=True)
def _lcs_table(a, b):
    m, n = len(a), len(b)
    table = np.zeros((m + 1, n + 1), dtype=np.int32)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table

def _lcs_union(hyps, ref):
    lcs_union = set()
    for hyp in hyps:
        if ref and hyp:  # numba cant handle empty lists ...
            lcs_elem = _lcs_elements(hyp, ref, _lcs_table(hyp, ref))  # type: ignore
        else:
            table = [[0.0 for _ in range(len(hyp) + 1)] for _ in range(len(hyp) + 1)]
            lcs_elem = _lcs_elements(hyp, ref, table)
        lcs_union = lcs_union.union(ref_idx for _, ref_idx in lcs_elem)
    return lcs_union

def patch_rouge():
    """Monkeypatch JIT-compiled LCS computation (same as paper)"""
    rouge_metric.py_rouge._lcs_table = _lcs_table
    rouge_metric.py_rouge._lcs_union = _lcs_union

# Initialize ROUGE scorer (same as paper: only ROUGE-L is reported in table)
# Note: Code computes ROUGE-1,2,3,4 internally, but paper only reports ROUGE-L F1
patch_rouge()
_rouge_scorer = PyRouge(rouge_n=(1, 2, 3, 4))  # Library needs this, but we only extract ROUGE-L

def repetition_rate(tokens: List[str], n: int = 4, window: int = 256, step: int = 128) -> Tuple[float, float]:
    """
    RR: average repetition fraction across windows:
        RR(win) = 1 - (#unique ngrams / #total ngrams)
    RR>80: fraction of windows where RR(win) >= 0.8
    """
    if len(tokens) < n:
        return 0.0, 0.0

    def win_rr(win_toks: List[str]) -> float:
        if len(win_toks) < n:
            return 0.0
        ngrams = [tuple(win_toks[i:i+n]) for i in range(0, len(win_toks)-n+1)]
        total = len(ngrams)
        uniq = len(set(ngrams))
        return 1.0 - (uniq / total) if total else 0.0

    rrs = []
    hits = 0
    starts = list(range(0, max(1, len(tokens) - window + 1), step))
    if not starts:
        starts = [0]
    for s in starts:
        win = tokens[s:s+window]
        rr = win_rr(win)
        rrs.append(rr)
        if rr >= 0.8:
            hits += 1

    avg_rr = sum(rrs) / len(rrs) if rrs else 0.0
    rr80 = hits / len(rrs) if rrs else 0.0
    return avg_rr, rr80

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union

def compute_one(gen_path: Path, ref_path: Path, max_tokens: int | None = None) -> Dict[str, float | int | str]:
    gen_text = read_text(gen_path)
    ref_text = read_text(ref_path)

    gen_toks = ws_tokens(gen_text)
    ref_toks = ws_tokens(ref_text)

    # Compute ROUGE-L F1 only (same as paper table - only ROUGE-L is reported)
    # Paper says "we also report ROUGE-L F1" - only F1, not P/R
    try:
        rouge_scores = _rouge_scorer.evaluate(
            hypotheses=[gen_text],
            multi_references=[[ref_text]],
        )
        # Paper table only shows ROUGE-L F1 (R-L column)
        rouge_l_f1 = rouge_scores.get("rouge-l", {}).get("f", 0.0)
    except Exception as e:
        # Fallback if rouge_metric fails
        print(f"Warning: ROUGE computation failed for {gen_path.name}: {e}", flush=True)
        rouge_l_f1 = 0.0

    # Other simple metrics (whitespace-based)
    rr, rr80 = repetition_rate(gen_toks, n=4, window=256, step=128)
    jac = jaccard(gen_toks, ref_toks)

    return {
        # Only metrics reported in paper table
        # ROUGE-L F1 (R-L column in paper table)
        "simple_rougeL_f1": rouge_l_f1,
        # RR and RR>80 (Repetitions columns in paper table)
        "simple_rr_ngram4": rr,
        "simple_rr80_ngram4": rr80,
    }

def pair_files(gen_dir: Path, ref_dir: Path) -> List[Tuple[Path, Path]]:
    # Match by filename (same name) first; fallback to stem match.
    gen_files = [p for p in gen_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt"}]
    ref_map_name = {p.name: p for p in ref_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt"}}
    ref_map_stem = {p.stem: p for p in ref_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt"}}

    pairs = []
    for g in gen_files:
        r = ref_map_name.get(g.name) or ref_map_stem.get(g.stem)
        if r:
            pairs.append((g, r))
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=str, default=None, help="Generated file path (.md/.txt)")
    ap.add_argument("--ref", type=str, default=None, help="Reference file path (.md/.txt)")
    ap.add_argument("--gen_dir", type=str, default=None, help="Directory of generated files")
    ap.add_argument("--ref_dir", type=str, default=None, help="Directory of reference files")
    ap.add_argument("--out", type=str, default="metrics_simple.csv", help="Output CSV path")
    ap.add_argument("--pred_dir", type=str, default=None, help="Prediction root (contains <id>/generated.md and <id>/patent.md or reference.md)")
    ap.add_argument("--out_csv", type=str, default=None, help="Output merged CSV path (default: auto infer <model>-res.csv under outputs/single-llm-call/)")
    args = ap.parse_args()

    # New preferred mode: --pred_dir (updates a single <model>-res.csv)
    if args.pred_dir:
        pred_root = Path(args.pred_dir)
        out_csv = Path(args.out_csv) if args.out_csv else infer_csv_path(pred_root)
        fieldnames, rows_by_id = load_or_init_csv(out_csv, id_col="id")

        n = 0
        for s in iter_pred_samples(pred_root):
            metrics = compute_one(s.gen_path, s.ref_path)
            fieldnames = upsert_metrics(fieldnames, rows_by_id, s.sample_id, metrics, id_col="id")
            write_csv(out_csv, fieldnames, rows_by_id, id_col="id")
            n += 1
            print(f"{s.sample_id}\t" + "\t".join(f"{k}={metrics[k]}" for k in sorted(metrics.keys())), flush=True)

        print(f"Wrote/updated {n} samples -> {out_csv}")
        return

    # Backwards compatible modes
    rows = []
    if args.gen and args.ref:
        rows.append(compute_one(Path(args.gen), Path(args.ref)))
    elif args.gen_dir and args.ref_dir:
        pairs = pair_files(Path(args.gen_dir), Path(args.ref_dir))
        if not pairs:
            raise SystemExit("No matched file pairs found. Ensure filenames (or stems) match.")
        for g, r in pairs:
            rows.append(compute_one(g, r))
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
