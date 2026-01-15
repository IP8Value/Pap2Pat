#!/usr/bin/env python3
"""
Single LLM-call baseline runner.
Reads Pap2Pat test split, builds a single prompt (system + outline + paper),
and generates full patent text per sample via an OpenAI-compatible API.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from client import get_profile  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Pap2Pat" / "data"

OUTLINE_GRANULARITIES = {
    "long": 800,
    "medium": 1000,
    "short": 2000,
    "empty": 0,
}

SYSTEM_PROMPT = """### ROLE

You are a highly skilled patent attorney with decades of experience in drafting high-quality patent applications.
You assist scientists in transforming their scientific discoveries into lucrative patents.

### TASK DESCRIPTION

Your task is to draft a patent application.

### INPUTS

As input, you will be provided a research paper and a patent outline, each serving a distinct purpose.

1. Research Paper:

The research paper describes a novel invention to be patented.
Your task is to extract the invention from the paper and write a patent application for it.

2. Patent Outline:

The patent outline summarizes the desired discourse structure of the patent document.
Use this outline as a rough guidance during drafting.

### GUIDELINES

- Copy the headings from the outline exactly. You must include only the headings provided in the outline.
- You must always write complete sentences and avoid keywords, bullet lists and enumerations.
- The patent must act as a standalone document, therefore do not refer to the research paper in the patent.
"""


def load_split_ids(split: str) -> list[str]:
    metadata = json.loads((DATA_DIR / "metadata.json").read_text())
    if split not in metadata["splits"]:
        raise ValueError(f"Unknown split '{split}'. Available: {list(metadata['splits'].keys())}")
    return metadata["splits"][split]


def iter_samples(split: str) -> Iterable[str]:
    return load_split_ids(split)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_user_prompt(outline_text: str, paper_text: str, outline_suffix: str) -> str:
    n_words = OUTLINE_GRANULARITIES.get(outline_suffix, OUTLINE_GRANULARITIES["long"])
    length_hint = f"Per bullet point, write roughly {n_words} words.\n"

    return (
        "Here is the outline of the desired patent application.\n"
        f"{length_hint}\n"
        "Example outline (bullet points are the lines starting with '- '):\n"
        "## DESCRIPTION OF THE INVENTION\n"
        "- describe discovery of ODAM protein in human epithelial cancers\n"
        "- describe method for aiding in diagnosis and management of cancer\n"
        "- describe specific embodiments of the invention\n"
        "- describe methods for determining presence of ODAM or anti-ODAM antibodies\n\n"
        "In the example above, each line beginning with '- ' is a bullet point.\n\n"
        f"```md\n{outline_text.strip()}\n```\n\n"
        "You need to draft a complete patent application that strictly follows the outline's section order "
        "and headings. Do not skip any bullet points. Use formal patent language. "
        "The generated patent must not be shorter than the research paper in word count.\n\n"
        "Here is the research paper that describes the invention:\n\n"
        f"```md\n{paper_text.strip()}\n```\n"
    )


def remove_outline_bullets(text: str) -> str:
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Single LLM-call baseline on Pap2Pat.")
    parser.add_argument("--model_profile", default="qwen3", choices=["qwen3", "deepseek-v3", "qwen3-max"], help="Model profile to use.")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable reasoning mode if supported (default: disabled).")
    parser.add_argument("--split", default="test")
    parser.add_argument("--outline_suffix", default="long", choices=list(OUTLINE_GRANULARITIES.keys()))
    parser.add_argument("--output_dir", required=True, help="Output directory for this run.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=32768)
    args = parser.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise ValueError("Missing DASHSCOPE_API_KEY env.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    client, model, extra_body = get_profile(
        args.model_profile,
        api_key=None,
        model=None,
        enable_thinking=args.enable_thinking,
    )

    sample_ids = list(iter_samples(args.split))
    random.shuffle(sample_ids)  # 随机打乱顺序，便于多进程并行处理
    print(f"[info] split={args.split} total_samples={len(sample_ids)}", flush=True)
    for sample_id in sample_ids:
        sample_dir = DATA_DIR / sample_id
        out_dir = output_dir / f"pred_{args.split}" / sample_id
        outline_marker = out_dir / "outline.md"
        
        # 检测 outline.md 是否存在（如果存在说明正在处理或已完成，直接跳过）
        if outline_marker.exists():
            print(f"[skip] {sample_id} -> {outline_marker}", flush=True)
            continue
        
        # 创建目录并立即创建 outline.md 作为标记，防止多进程重复处理
        out_dir.mkdir(parents=True, exist_ok=True)
        outline_text = read_text(sample_dir / f"patent_outline_{args.outline_suffix}.md")
        (out_dir / "outline.md").write_text(outline_text, encoding="utf-8")
        
        paper_text = read_text(sample_dir / "paper.md")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(outline_text, paper_text, args.outline_suffix)},
        ]

        generated_file = out_dir / "generated.md"
        try:
            print(f"\n[Generating for {sample_id}]", flush=True)
            generated = ""
            save_interval = 100  # 每100个字符保存一次
            last_save_length = 0

            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                extra_body=extra_body,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    # 实时打印每个字符
                    print(content, end="", flush=True)
                    generated += content

                    # 每100个字符保存一次
                    if len(generated) - last_save_length >= save_interval:
                        generated_file.write_text(generated, encoding="utf-8")
                        last_save_length = len(generated)

            # 最后保存完整内容
            generated_file.write_text(generated, encoding="utf-8")
            print(f"\n[Completed: {len(generated)} characters]", flush=True)

        except Exception as exc:
            error_msg = f"[generation failed] {exc}"
            print(f"\n{error_msg}", flush=True)
            generated = error_msg
            generated_file.write_text(generated, encoding="utf-8")
        # post-process: remove outline bullet points if any leaked into output
        cleaned = remove_outline_bullets(generated_file.read_text(encoding="utf-8"))
        generated_file.write_text(cleaned, encoding="utf-8")
        # 保存其他文件（outline.md 已在开始时创建）
        (out_dir / "paper.md").write_text(paper_text, encoding="utf-8")
        reference_text = read_text(sample_dir / "patent.md")
        (out_dir / "patent.md").write_text(reference_text, encoding="utf-8")
        (out_dir / "system_prompt.md").write_text(SYSTEM_PROMPT, encoding="utf-8")
        (out_dir / "user_prompt.md").write_text(messages[1]["content"], encoding="utf-8")



if __name__ == "__main__":
    main()
