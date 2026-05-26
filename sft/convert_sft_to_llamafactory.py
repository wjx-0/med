#!/usr/bin/env python3
"""Convert CURE-Med SFT JSONL data to LLaMA-Factory OpenAI messages format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = """You are an expert multilingual medical doctor. When answering a medical question, follow these steps:
1. First, search your internal knowledge base thoroughly for relevant background information about the topic.
2. Understand and reason the question fully in English first.
3. Reason mainly in English, but code-switch naturally into the target language whenever useful for clarity or domain accuracy.
4. Consider multiple perspectives and potential answers before settling on your final response.
5. Evaluate the confidence in your answer based on the information available to you.
6. Provide the final answer clearly in the target language, making sure it's well-supported by your reasoning.
7. If there are significant uncertainties or gaps in your knowledge, acknowledge them transparently.

Your goal is to provide accurate, well-reasoned responses that demonstrate depth of understanding, not just surface-level answers.

You are an expert multilingual medical doctor. When answering a medical question, think and reason mainly in English with natural code-switching to the target language. Use multi-step reasoning wrapped in <step> tags inside <thinking>."""


def default_source_dir() -> Path:
    """Return the source SFT directory for the local sibling cure-med checkout."""
    workspace_root = Path(__file__).resolve().parents[2]
    return workspace_root / "cure-med" / "datasets" / "SFT_data"


def make_example(raw: dict) -> dict:
    language = raw["language"]
    question = raw["question"]
    reasoning = raw["reasoning"].strip()
    answer = raw["answer"].strip()
    user_prompt = (
        f"The question is in {language}. {question} "
        "Please think carefully with English-guided reasoning and code-switching, "
        "return your reasoning inside <thinking> </thinking> tags, and the final direct answer "
        f"inside <answer> </answer> tags. Final answer ONLY in {language}."
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": f"{reasoning}\n{answer}"},
        ],
        "language": language,
    }


def read_jsonl(path: Path) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            required = {"question", "reasoning", "answer", "language"}
            missing = required - raw.keys()
            if missing:
                raise ValueError(f"{path}:{line_no} missing fields: {sorted(missing)}")
            examples.append(make_example(raw))
    return examples


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def write_dataset_info(output_dir: Path) -> None:
    dataset_info = {
        "curemed_sft_spanish": {
            "file_name": "curemed_sft_spanish.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
        "curemed_sft_yoruba": {
            "file_name": "curemed_sft_yoruba.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
        "curemed_sft_spanish_yoruba": {
            "file_name": "curemed_sft_spanish_yoruba.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=default_source_dir(),
        help="Directory containing original CURE-Med SFT JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "llamafactory_data",
        help="Directory where converted LLaMA-Factory data will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    spanish = read_jsonl(source_dir / "Spanish.jsonl")
    yoruba = read_jsonl(source_dir / "Yoruba.jsonl")

    write_jsonl(output_dir / "curemed_sft_spanish.jsonl", spanish)
    write_jsonl(output_dir / "curemed_sft_yoruba.jsonl", yoruba)
    write_jsonl(output_dir / "curemed_sft_spanish_yoruba.jsonl", spanish + yoruba)
    write_dataset_info(output_dir)

    print(f"Wrote Spanish examples: {len(spanish)}")
    print(f"Wrote Yoruba examples: {len(yoruba)}")
    print(f"Wrote combined examples: {len(spanish) + len(yoruba)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
