#!/usr/bin/env python3
"""Evaluate a subset of Pap2Pat predictions without requiring full split coverage."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import rich.console
import rich.syntax
from hydralette import Config, Field
from tqdm import tqdm

from pap2pat import evaluation as pap_eval

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Pap2Pat" / "data"

console = rich.console.Console()

ALL_METRICS_KEYS = (
    "bertscore",
    "bleu",
    "rouge",
    "rr",
    "tokens",
    "repetitions",
    "language",
    "factuality",
    "coherence",
    "chunk_stats",
)

ALL_METRICS = (
    "BERTScore",
    "bleu",
    "rouge",
    "rr",
    "tokens",
    "repetitions",
    "clustered_BERTScore",
    "clustered_bleu",
    "clustered_rouge",
    "clustered_tokens",
    "language",
    "factuality",
    "coherence",
    "chunk_stats",
)


cfg = Config(
    run_dir=Field(type=Path),
    split=Field(default="val"),
    scores=Field(default=ALL_METRICS_KEYS, convert=lambda s: s.split(",")),
    sample_ids=Field(default=None, convert=lambda s: s.split(",") if s else None),
    metrics_filename="metrics_subset.json",
)


def load_predictions(run_dir: Path, split: str) -> dict[str, str]:
    pred_root = run_dir / "predictions" / split
    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_root}")
    generated = {
        pred_dir.name: pred_dir.joinpath("generated.md").read_text()
        for pred_dir in tqdm(list(pred_root.glob("*")), desc="Loading pred files")
        if pred_dir.joinpath("generated.md").exists()
    }
    if not generated:
        raise ValueError(f"No predictions found in {pred_root}")
    return generated


def set_subset_metadata(split: str, sample_ids: list[str]) -> None:
    metadata_path = DATA_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    unknown = [sid for sid in sample_ids if sid not in metadata["splits"][split]]
    if unknown:
        raise ValueError(
            f"Sample IDs not present in original metadata for split '{split}': {unknown}"
        )

    metadata["splits"][split] = sample_ids
    pap_eval.pap2pat_metadata = metadata
    pap_eval.ground_truth_patents = {}
    pap_eval.papers = {}


def compute_metrics(predictions: pap_eval.Predictions, scores: list[str], run_dir: Path, split: str, metrics_filename: str) -> dict:
    if "tokens" in scores:
        predictions.compute_tokens()
    if "bleu" in scores:
        predictions.compute_bleu()
    if "rouge" in scores:
        predictions.compute_rouge()
    if "bertscore" in scores:
        predictions.compute_bertscore()
    if "rr" in scores:
        predictions.compute_rr()
    if "repetitions" in scores:
        predictions.compute_repetitions()
    if "language" in scores:
        predictions.compute_language()
    if "factuality" in scores:
        predictions.compute_factuality()
    if "coherence" in scores:
        predictions.compute_coherence()
    if "chunk_stats" in scores:
        predictions.compute_chunk_stats(run_dir)

    for sample_id in predictions.preds.keys():
        metrics = predictions.metrics_per_sample.get(sample_id, {})
        metrics_path = run_dir / "predictions" / split / sample_id / metrics_filename
        if metrics_path.exists():
            metrics_disk = json.loads(metrics_path.read_text())
        else:
            metrics_disk = {}
        metrics = {**metrics_disk, **metrics}
        metrics = {str(k): v for k, v in metrics.items() if k in ALL_METRICS}
        metrics_path.write_text(json.dumps(metrics, indent=4, sort_keys=True))
        predictions.metrics_per_sample[sample_id] = metrics

    return predictions.average_metrics()


def main(cfg: Config) -> None:
    console.print(rich.syntax.Syntax(cfg.to_yaml(), "yaml"))

    split: Literal["train", "val", "test", "non-contaminated-test"] = cfg.split  # type: ignore
    generated = load_predictions(cfg.run_dir, split)

    if cfg.sample_ids is None:
        sample_ids = sorted(generated.keys())
    else:
        sample_ids = [sid.strip() for sid in cfg.sample_ids if sid.strip()]
        missing = [sid for sid in sample_ids if sid not in generated]
        if missing:
            raise ValueError(f"Requested sample IDs not found in predictions: {missing}")
        generated = {sid: generated[sid] for sid in sample_ids}

    set_subset_metadata(split, sample_ids)

    preds = pap_eval.Predictions(generated, split, data_path=DATA_DIR)  # type: ignore[arg-type]
    aggregate = compute_metrics(preds, cfg.scores, cfg.run_dir, split, cfg.metrics_filename)

    metrics_path = cfg.run_dir / cfg.metrics_filename
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text())
    else:
        existing = {}
    existing[f"{split}_subset"] = aggregate
    metrics_path.write_text(json.dumps(existing, indent=4, sort_keys=True))

    console.print("\nSubset metrics (averaged):")
    console.print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    cfg.apply()
    main(cfg)
