#!/usr/bin/env python3
"""
Single LLM-call baseline for Pap2Pat.
Uses Ollama (OpenAI-compatible API) to generate an entire patent in one call
by providing the complete paper and outline as context.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import rich.console
import rich.syntax
from hydralette import Config, Field
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Pap2Pat" / "data"

# ensure src on path
SRC_DIR = PROJECT_ROOT / "Outline_Guided_Generation" / "src"
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.dataset.prompt import Prompt, outline_granularities  # type: ignore
from src.utils.general import get_logger, redirect_stdout_stderr, set_seed  # type: ignore

log = get_logger("ipmai_single")


def default_run_dir() -> Path:
    base = PROJECT_ROOT / "outputs" / "runs"
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / "ollama-single"
    if run_dir.exists():
        run_dir = base / f"ollama-single-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    return run_dir


cfg = Config(
    ollama_api_base=Field(default=None, help="Ollama API base URL, e.g. http://localhost:11434"),
    ollama_model=Field(default=None, help="Ollama model name, e.g. qwen2.5:72b-instruct"),
    tokenizer_model=Field(default="Qwen/Qwen2.5-7B-Instruct", help="Tokenizer used to format prompts"),
    outline_suffix=Field(default="long", help="Which outline variant to use (long/medium/short/empty)"),
    split=Field(default="val", help="Dataset split to process"),
    max_samples=Field(default=None, type=int, help="Limit number of samples (useful for quick tests)"),
    seed=Field(default=42),
    temperature=0.6,
    top_p=1.0,
    top_k=-1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    run_dir=Field(default_factory=default_run_dir),
)


def load_split_ids(split: str) -> list[str]:
    metadata_path = DATA_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if split not in metadata["splits"]:
        raise ValueError(f"Unknown split '{split}'. Available: {list(metadata['splits'].keys())}")
    return metadata["splits"][split]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_prompt(outline_text: str, paper_text: str, outline_suffix: str, tokenizer) -> list[dict[str, str]]:
    n_chars = outline_granularities.get(outline_suffix, outline_granularities["long"])
    n_paragraphs = max(1, n_chars // 590)
    n_words = max(1, n_chars // 7)

    system_prompt = Prompt.SYSTEM_PROMPT.format(n_paragraphs=n_paragraphs, n_words=n_words)

    length_info = Prompt.LENGTH_INFO.format(n_paragraphs=n_paragraphs, n_words=n_words)

    user_content = (
        f"{Prompt.OUTLINE_INTRO}\n\n"
        f"```md\n{outline_text.strip()}\n```\n\n"
        f"请根据以上大纲生成完整的专利描述，并确保风格和细节符合专利要求。\n\n"
        f"以下是研究论文正文，可作为参考：\n\n"
        f"```md\n{paper_text.strip()}\n```\n"
        f"{length_info}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return messages


def iter_samples(split: str, max_samples: int | None) -> Iterable[str]:
    sample_ids = load_split_ids(split)
    if max_samples is not None:
        sample_ids = sample_ids[:max_samples]
    return sample_ids


def run(cfg: Config) -> None:
    if cfg.ollama_api_base is None or cfg.ollama_model is None:
        raise ValueError(
            "请提供 --ollama_api_base 和 --ollama_model，例如："
            " python ipmai/generate_ollama_single.py --ollama_api_base http://localhost:11434 --ollama_model qwen2.5:72b-instruct"
        )

    set_seed(cfg.seed)

    console = rich.console.Console()
    console.print(rich.syntax.Syntax(cfg.to_yaml(), "yaml"))

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_model)
    api_base = cfg.ollama_api_base.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base = f"{api_base}/v1"
    client = OpenAI(base_url=api_base, api_key="ollama")

    (run_dir / "config.yaml").write_text(cfg.to_yaml())
    (run_dir / "overrides.txt").write_text(" ".join(sys.argv[1:]))

    with redirect_stdout_stderr((run_dir / "output.log").open("w", encoding="utf-8")):
        sample_ids = list(iter_samples(cfg.split, cfg.max_samples))
        for sample_id in tqdm(sample_ids, desc=f"Generating {cfg.split}"):
            sample_dir = DATA_DIR / sample_id
            paper_text = read_text(sample_dir / "paper.md")
            outline_text = read_text(sample_dir / f"patent_outline_{cfg.outline_suffix}.md")

            messages = build_prompt(outline_text, paper_text, cfg.outline_suffix, tokenizer)

            try:
                response = client.chat.completions.create(
                    model=cfg.ollama_model,
                    messages=messages,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    frequency_penalty=cfg.frequency_penalty,
                    presence_penalty=cfg.presence_penalty,
                )
                generated = response.choices[0].message.content
            except Exception as exc:  # pragma: no cover
                log.error(f"{sample_id}: generation failed: {exc}")
                generated = "[生成错误]"

            save_dir = run_dir / "predictions" / cfg.split / sample_id
            save_dir.mkdir(parents=True, exist_ok=True)

            reference_text = read_text(sample_dir / "patent.md")

            (save_dir / "generated.md").write_text(generated)
            (save_dir / "reference.md").write_text(reference_text)
            (save_dir / "outline.md").write_text(outline_text)
            (save_dir / "paper.md").write_text(paper_text)

    log.info("✅ Single LLM-call generation finished")


if __name__ == "__main__":
    cfg.apply()
    run(cfg)
