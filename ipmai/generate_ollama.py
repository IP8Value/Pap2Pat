#!/usr/bin/env python3
"""
Ollama 专用生成脚本
位于 ipmai 包下，避免改动原有 src 目录。
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import rich.console
import rich.syntax
from hydralette import Config, Field
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "Outline_Guided_Generation" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.dataset.dataset import PatentDraftingDataset, PatentDraftingSample, data_cfg  # type: ignore
from src.dataset.prompt import (  # type: ignore
    format_outline_md,
    format_outline_ref,
    format_reference_patent,
    get_all_headings,
    iter_sections_with_next,
)
from src.utils.general import get_logger, redirect_stdout_stderr, set_seed  # type: ignore

log = get_logger(__name__)


def default_run_dir() -> Path:
    base = PROJECT_ROOT / "outputs" / "runs"
    base.mkdir(parents=True, exist_ok=True)
    candidate = base / "ollama"
    if candidate.exists():
        candidate = base / f"ollama-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    return candidate


def postprocess_ollama(chunk_responses: list[dict], description: dict) -> str:
    """合并 chunk 响应，容错处理 markdown/纯文本返回"""
    parts: list[str] = []
    for resp in chunk_responses:
        content = resp.get("content", "").strip()
        if not content:
            continue
        md_match = re.search(r"```md\s*(.*?)\s*```", content, re.DOTALL)
        if md_match:
            parts.append(md_match.group(1).strip())
            continue
        code_match = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
        if code_match:
            parts.append(code_match.group(1).strip())
            continue
        content = re.sub(r"^```md\s*", "", content)
        content = re.sub(r"```\s*$", "", content)
        parts.append(content.strip())

    combined = "\n\n".join(parts)

    headings = list(get_all_headings(description, return_hashes=True))
    cleaned_lines: list[str] = []
    for line in combined.split("\n"):
        is_heading = False
        if headings and (match := re.findall(r"^(#+\s+.*)\s*$", line)):
            heading = match[0]
            if heading == headings[0]:
                headings.pop(0)
            else:
                is_heading = True
        if not is_heading and line.strip() != "...":
            cleaned_lines.append(line)

    return re.sub("\n{2,}", "\n\n", "\n".join(cleaned_lines))


def format_ground_truth_patent_from_chunks(sample: PatentDraftingSample) -> str:
    sections: list[str] = []
    for chunk in sample.chunks:
        chunk_text = ""
        for (level, section), _ in iter_sections_with_next(chunk.content, level=chunk.level):
            heading = f'\n\n{"#" * (level + 1)} {section["title"]}\n\n'
            chunk_text += heading
            if section["paragraphs"]:
                chunk_text += "\n\n".join(section["paragraphs"])
        if chunk_text.strip():
            sections.append(chunk_text.strip())
    return "\n\n".join(sections)


def ensure_run_dir(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_client(api_base: str) -> OpenAI:
    api_base = api_base.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base = f"{api_base}/v1"
    log.info(f"连接 Ollama API: {api_base}")
    client = OpenAI(base_url=api_base, api_key="ollama")
    try:
        models = client.models.list()
        log.info(f"✅ 成功连接到 Ollama，可用模型: {[m.id for m in models.data]}")
    except Exception as exc:  # pragma: no cover
        log.warning(f"⚠️ 无法列出模型，但继续执行: {exc}")
    return client


def default_cfg() -> Config:
    return Config(
        ollama_api_base=Field(default=None, help="Ollama API 地址，例如: http://localhost:11434"),
        ollama_model=Field(default=None, help="Ollama 模型名称，例如: qwen2.5:72b-instruct"),
        model=Field(default="Qwen/Qwen2.5-7B-Instruct", help="用于 tokenizer 的模型路径"),
        debug=False,
        run_dir=Field(default_factory=default_run_dir),
        seed=Field(default=42),
        data=data_cfg,
        temperature=0.6,
        top_p=1.0,
        top_k=-1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_samples=Field(default=None, type=int, help="限制样本数量，用于测试"),
    )


cfg = default_cfg()


def main(cfg: Config) -> None:
    set_seed(cfg.seed)

    if cfg.ollama_api_base is None or cfg.ollama_model is None:
        raise ValueError(
            "请通过命令行参数提供 --ollama_api_base 和 --ollama_model，例如：\n"
            "  python ipmai/generate_ollama.py --ollama_api_base http://localhost:11434 --ollama_model qwen2.5:72b-instruct"
        )

    rich_console = rich.console.Console()  # type: ignore[attr-defined]
    rich_console.print(rich.syntax.Syntax(cfg.to_yaml(), "yaml"))  # type: ignore[attr-defined]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    log.info("Using Ollama, loading tokenizer from '%s' for text processing", cfg.model)

    ds = PatentDraftingDataset(
        data_dir=cfg.data.papers_dir,
        outline_suffix=cfg.data.outline_suffix,
        paper_extractor=cfg.data.paper_extractor,
        do_chunk=True,
        tokenizer=tokenizer,
        max_total_length=cfg.data.max_total_length,
        max_instruction_length=cfg.data.max_instruction_length,
        max_paper_length=cfg.data.max_paper_length,
        max_patent_length=cfg.data.max_patent_length,
        add_claim=cfg.data.add_claim,
    )

    run_dir = ensure_run_dir(Path(cfg.run_dir))
    (run_dir / "config.yaml").write_text(cfg.to_yaml())
    (run_dir / "overrides.txt").write_text(" ".join(sys.argv[1:]))

    with redirect_stdout_stderr(run_dir / "output.log"):
        client = build_client(cfg.ollama_api_base)
        total_processed = 0

        for split, samples in ds.splits.items():
            if split in ["test", "non-contaminated-test"]:
                continue

            pending = [s for s in samples if not run_dir.joinpath(f"predictions/{split}/{s.id}").exists()]
            if cfg.max_samples is not None:
                remaining = cfg.max_samples - total_processed
                if remaining <= 0:
                    log.info("已达到最大样本数限制，停止处理")
                    break
                pending = pending[:remaining]
                log.info(f"限制处理数量: {split} 分割处理 {len(pending)} 个样本")

            if not pending:
                log.info(f"跳过 {split} 分割（全部已生成或超限）")
                continue

            log.info(f"开始处理 {split} 分割: {len(pending)} 个样本")
            for sample in tqdm(pending, desc=f"Generating {split}"):
                total_processed += 1
                log.info(f"Current sample: {sample.id}")
                chunk_responses: list[dict] = []

                for idx, chunk in enumerate(sample.chunks):
                    log.info(f"  生成 chunk {idx + 1}/{len(sample.chunks)}")
                    messages = [
                        {"role": "system", "content": chunk.prompt.system},
                        {"role": "user", "content": chunk.prompt.user},
                    ]
                    try:
                        response = client.chat.completions.create(
                            model=cfg.ollama_model,
                            messages=messages,
                            temperature=cfg.temperature,
                            top_p=cfg.top_p,
                            max_tokens=cfg.data.max_total_length,
                            stop=["<s>", "</s>", "```"]
                        )
                        content = response.choices[0].message.content
                        chunk_responses.append({"role": "assistant", "content": content})
                        log.info(f"  ✅ Chunk {idx + 1} 生成完成，长度: {len(content)} 字符")
                    except Exception as exc:  # pragma: no cover
                        log.error(f"  ❌ Chunk {idx + 1} 生成失败: {exc}")
                        chunk_responses.append({"role": "assistant", "content": "```md\n\n[生成错误]\n```"})

                try:
                    generated = postprocess_ollama(chunk_responses, sample.patent_content["sections"][0])
                except Exception as exc:  # pragma: no cover
                    log.error(f"  ⚠️  Postprocess 失败: {exc}")
                    generated = "\n\n".join(resp.get("content", "") for resp in chunk_responses)

                pred_path = run_dir / f"predictions/{split}/{sample.id}"
                pred_path.mkdir(parents=True, exist_ok=True)

                description = sample.patent_content["sections"][0]
                reference_full = format_reference_patent([description]).strip()
                reference_from_chunks = format_ground_truth_patent_from_chunks(sample).strip()

                if reference_full != reference_from_chunks:
                    err_dir = pred_path / "errors"
                    err_dir.mkdir(parents=True, exist_ok=True)
                    (err_dir / "patent_from_chunks.md").write_text(reference_from_chunks)
                    (err_dir / "patent_from_whole.md").write_text(reference_full)
                    log.error(f"{sample.id}: Reference patent mismatch!")

                outline_full = format_outline_ref([description]).strip()
                outline_chunks = format_outline_md(sample.chunks).strip()
                if outline_full != outline_chunks:
                    err_dir = pred_path / "errors"
                    err_dir.mkdir(parents=True, exist_ok=True)
                    (err_dir / "outline_from_chunks.md").write_text(outline_chunks)
                    (err_dir / "outline_from_whole.md").write_text(outline_full)
                    log.error(f"{sample.id}: Reference outline mismatch!")

                (pred_path / "generated.md").write_text(generated)
                (pred_path / "reference.md").write_text(reference_full)

                conv_dir = pred_path / "conversations"
                conv_dir.mkdir(parents=True, exist_ok=True)
                for idx, (chunk, resp) in enumerate(zip(sample.chunks, chunk_responses)):
                    (conv_dir / f"chunk-{idx}-system.md").write_text(chunk.prompt.system)
                    (conv_dir / f"chunk-{idx}-user.md").write_text(chunk.prompt.user)
                    (conv_dir / f"chunk-{idx}-assistant.md").write_text(resp.get("content", ""))

                if cfg.max_samples is not None and total_processed >= cfg.max_samples:
                    log.info("已达到最大样本数限制，停止处理")
                    return

    log.info("✅ 生成完成！")


if __name__ == "__main__":
    cfg.apply()
    main(cfg)
