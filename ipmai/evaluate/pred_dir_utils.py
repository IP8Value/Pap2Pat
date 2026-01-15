from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class PredSample:
    sample_id: str
    sample_dir: Path
    gen_path: Path
    ref_path: Path


def _is_split_dir_name(name: str) -> bool:
    # Common split folder names used in this repo / outputs.
    return name in {"train", "val", "test", "pred_test", "pred_val", "pred_train", "predictions"}


def iter_pred_samples(pred_root: Path) -> Iterable[PredSample]:
    """
    Given a prediction root directory, yield samples found underneath.

    Supports layouts like:
      A) .../<model>/pred_test/<sample_id>/{generated.md,patent.md}
      B) .../<model>/predictions/<split>/<sample_id>/{generated.md,reference.md}
      C) .../<model>/predictions/<sample_id>/... (less common, but we handle any depth)
    """
    pred_root = pred_root.expanduser().resolve()
    if not pred_root.exists():
        raise FileNotFoundError(f"pred_root not found: {pred_root}")

    # Find directories that contain generated.md; treat that directory as a sample dir.
    for gen_path in pred_root.rglob("generated.md"):
        sample_dir = gen_path.parent
        sample_id = sample_dir.name

        # Reference file differs for ollama runs (reference.md) vs dashscope runs (patent.md).
        ref_path = sample_dir / "patent.md"
        if not ref_path.exists():
            ref_path = sample_dir / "reference.md"
        if not ref_path.exists():
            # Can't score this sample without a reference.
            continue

        yield PredSample(
            sample_id=sample_id,
            sample_dir=sample_dir,
            gen_path=gen_path,
            ref_path=ref_path,
        )


def infer_model_name(pred_root: Path) -> str:
    """
    Infer model/run name from a prediction path.
    Expected paths usually include: outputs/single-llm-call/<model>/...
    """
    pred_root = pred_root.expanduser().resolve()
    parts = list(pred_root.parts)
    for i in range(len(parts) - 2):
        if parts[i] == "outputs" and parts[i + 1] == "single-llm-call":
            if i + 2 < len(parts):
                return parts[i + 2]
    # Fallback: parent folder name (e.g., deepseek-v3) for .../deepseek-v3/pred_test
    return pred_root.parent.name


def infer_csv_path(pred_root: Path) -> Path:
    """
    Default CSV location:
      .../outputs/single-llm-call/<model>-res.csv
    """
    pred_root = pred_root.expanduser().resolve()
    parts = list(pred_root.parts)
    for i in range(len(parts) - 1):
        if parts[i] == "outputs" and i + 1 < len(parts) and parts[i + 1] == "single-llm-call":
            base = Path(*parts[: i + 2])
            model = infer_model_name(pred_root)
            return base / f"{model}-res.csv"
    # Fallback: write next to the model folder
    model = infer_model_name(pred_root)
    return pred_root.parent / f"{model}-res.csv"


def load_or_init_csv(csv_path: Path, id_col: str = "id") -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """
    Returns (fieldnames, rows_by_id).
    Values are stored as strings (CSV-native).
    """
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.exists():
        return [id_col], {}

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            return [id_col], {}
        fieldnames = list(r.fieldnames)
        if id_col not in fieldnames:
            fieldnames = [id_col] + fieldnames
        rows_by_id: Dict[str, Dict[str, str]] = {}
        for row in r:
            sid = (row.get(id_col) or "").strip()
            if not sid:
                continue
            # normalize missing keys
            rows_by_id[sid] = {k: (row.get(k) or "") for k in fieldnames}
        return fieldnames, rows_by_id


def upsert_metrics(
    fieldnames: List[str],
    rows_by_id: Dict[str, Dict[str, str]],
    sample_id: str,
    metrics: Dict[str, object],
    id_col: str = "id",
) -> List[str]:
    """
    Upsert a single sample's metrics into the in-memory CSV table.
    Returns possibly-updated fieldnames.
    """
    if sample_id not in rows_by_id:
        rows_by_id[sample_id] = {k: "" for k in fieldnames}
        rows_by_id[sample_id][id_col] = sample_id

    # Ensure columns exist
    for k in metrics.keys():
        if k not in fieldnames:
            fieldnames.append(k)
            for sid, row in rows_by_id.items():
                row.setdefault(k, "")

    # Set values
    row = rows_by_id[sample_id]
    row[id_col] = sample_id
    for k, v in metrics.items():
        row[k] = "" if v is None else str(v)
    return fieldnames


def write_csv(csv_path: Path, fieldnames: List[str], rows_by_id: Dict[str, Dict[str, str]], id_col: str = "id") -> None:
    csv_path = csv_path.expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Stable ordering by sample_id (so diffs are clean)
    sample_ids = sorted(rows_by_id.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for sid in sample_ids:
            row = rows_by_id[sid]
            # Make sure all columns exist
            out = {k: row.get(k, "") for k in fieldnames}
            out[id_col] = sid
            w.writerow(out)

