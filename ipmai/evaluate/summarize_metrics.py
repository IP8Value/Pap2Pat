#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize metrics from CSV files (compute median, mean, std, etc.)

Usage:
  python ipmai/evaluate/summarize_metrics.py \
    --csv /Users/kevin/project_python/Pap2Pat/outputs/single-llm-call/deepseek-v3-res.csv
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_csv(csv_path: Path) -> tuple[List[str], List[Dict[str, float]]]:
    """Load CSV and return fieldnames and rows (excluding id column)"""
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [k for k in reader.fieldnames if k != "id"]
        rows = []
        for row in reader:
            # Convert numeric columns to float
            numeric_row = {}
            for k in fieldnames:
                try:
                    numeric_row[k] = float(row[k]) if row[k] else None
                except ValueError:
                    numeric_row[k] = None
            rows.append(numeric_row)
    return fieldnames, rows


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute statistics for a list of values"""
    values = [v for v in values if v is not None]
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    
    arr = np.array(values)
    return {
        "count": len(values),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    ap.add_argument("--output", type=str, default=None, help="Output summary file (default: print to stdout)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")

    fieldnames, rows = load_csv(csv_path)
    
    # Compute statistics for each metric
    summary = {}
    for metric in fieldnames:
        values = [row[metric] for row in rows]
        summary[metric] = compute_stats(values)

    # Print summary
    output_lines = []
    output_lines.append(f"# Summary for {csv_path.name}")
    output_lines.append(f"Total samples: {len(rows)}")
    output_lines.append("")
    output_lines.append("## Statistics (per metric)")
    output_lines.append("")
    
    # Metrics that should be multiplied by 100 (same as paper table format)
    # Paper table shows ROUGE-L, BERTScore, RR, and RR>80 as percentages (0-100)
    metrics_to_scale_100 = {
        "simple_rougeL_f1",
        "bertscore_f1",
        "bertscore_p",
        "bertscore_r",
        "simple_rr_ngram4",  # Paper table shows RR as 14.4, 12.3, etc. (percentages)
        "simple_rr80_ngram4",  # Paper table shows RR>80 as 0.2, 0.1, etc. (percentages)
        "token_rr_generated",  # Same RR metric, should also be percentage
        "token_rr_reference",  # Same RR metric, should also be percentage
    }
    
    # Metric direction hints (from paper table)
    metric_hints = {
        "simple_rougeL_f1": "↑ (higher is better, 0-100, same as paper R-L column)",
        "token_rr_generated": "↓ (lower is better, 0-100)",
        "token_rr_reference": "↓ (lower is better, 0-100)",
        "simple_rr_ngram4": "↓ (lower is better, 0-100, same as paper RR column)",
        "simple_rr80_ngram4": "↓ (lower is better, 0-100, same as paper RR>80 column)",
        "bertscore_f1": "↑ (higher is better, 0-100, same as paper BS column)",
        "bertscore_p": "↑ (higher is better, 0-100)",
        "bertscore_r": "↑ (higher is better, 0-100)",
        "token_gen_count": "≈ (should match reference)",
        "token_ref_count": "≈ (reference)",
        "token_fraction": "≈ (should be close to 1.0)",
    }
    
    for metric in fieldnames:
        stats = summary[metric]
        hint = metric_hints.get(metric, "")
        
        # Scale by 100 if needed (same as paper table format)
        scale = 100.0 if metric in metrics_to_scale_100 else 1.0
        
        output_lines.append(f"### {metric} {hint}")
        if stats["count"] == 0:
            output_lines.append("  No valid values")
        else:
            output_lines.append(f"  Count: {stats['count']}")
            if scale == 100.0:
                # Show both raw (0-1) and scaled (0-100) values
                output_lines.append(f"  Mean:  {stats['mean'] * scale:.2f} (raw: {stats['mean']:.4f})")
                output_lines.append(f"  Median: {stats['median'] * scale:.2f} (raw: {stats['median']:.4f})")
                output_lines.append(f"  Std:   {stats['std'] * scale:.2f} (raw: {stats['std']:.4f})")
                output_lines.append(f"  Min:   {stats['min'] * scale:.2f} (raw: {stats['min']:.4f})")
                output_lines.append(f"  Max:   {stats['max'] * scale:.2f} (raw: {stats['max']:.4f})")
            else:
                output_lines.append(f"  Mean:  {stats['mean']:.4f}")
                output_lines.append(f"  Median: {stats['median']:.4f}")
                output_lines.append(f"  Std:   {stats['std']:.4f}")
                output_lines.append(f"  Min:   {stats['min']:.4f}")
                output_lines.append(f"  Max:   {stats['max']:.4f}")
        output_lines.append("")

    output_text = "\n".join(output_lines)
    
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
        print(f"Summary written to {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
