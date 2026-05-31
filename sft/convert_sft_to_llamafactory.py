#!/usr/bin/env python3
"""Convert CURE-Med SFT JSONL data to LLaMA-Factory OpenAI messages format."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


SYSTEM_PROMPT = """You are an expert multilingual reasoning assistant with strong medical knowledge. Think carefully in English first, code-switching naturally into the target language when useful for clarity or domain accuracy. For medical questions, use appropriate clinical knowledge and acknowledge meaningful uncertainty. Provide accurate, well-supported responses with concise multi-step reasoning inside <thinking> and a final answer inside <answer>."""

ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)
THINKING_BLOCK_RE = re.compile(r"<thinking>(.*?)</thinking>", flags=re.IGNORECASE | re.DOTALL)
THINKING_TAG_RE = re.compile(r"</?thinking>", flags=re.IGNORECASE)
STEP_OPEN_RE = re.compile(r"<step\d+>", flags=re.IGNORECASE)
STEP_TAG_RE = re.compile(r"</?step\d+>", flags=re.IGNORECASE)


def default_source_dir() -> Path:
    """Return the source SFT directory for the local sibling cure-med checkout."""
    workspace_root = Path(__file__).resolve().parents[2]
    return workspace_root / "cure-med" / "datasets" / "SFT_data"


def strip_answer_blocks(text: str) -> str:
    return ANSWER_BLOCK_RE.sub("", text)


def extract_thinking_body(reasoning: str) -> str:
    text = strip_answer_blocks(reasoning.strip())
    thinking_blocks = [match.strip() for match in THINKING_BLOCK_RE.findall(text) if match.strip()]
    if thinking_blocks:
        return "\n\n".join(thinking_blocks)
    return THINKING_TAG_RE.sub("", text).strip()


def split_reasoning_steps(reasoning: str) -> list[str]:
    body = extract_thinking_body(reasoning)
    if not body:
        raise ValueError("reasoning is empty after removing tags")

    if STEP_OPEN_RE.search(body):
        chunks = STEP_OPEN_RE.split(body)
        steps = []
        for chunk in chunks:
            clean = STEP_TAG_RE.sub("", chunk).strip()
            if clean:
                steps.append(clean)
        if steps:
            return steps

    return [chunk.strip() for chunk in re.split(r"\n\s*\n+", STEP_TAG_RE.sub("", body)) if chunk.strip()]


def normalize_reasoning(reasoning: str) -> str:
    steps = split_reasoning_steps(reasoning)
    if not steps:
        raise ValueError("reasoning has no usable steps")

    lines = ["<thinking>"]
    for idx, step in enumerate(steps, start=1):
        lines.extend([f"<step{idx}>", step, f"</step{idx}>"])
    lines.append("</thinking>")
    return "\n".join(lines)


def normalize_answer(answer: str) -> str:
    text = answer.strip()
    matches = [match.strip() for match in ANSWER_BLOCK_RE.findall(text) if match.strip()]
    if matches:
        text = matches[-1]
    else:
        text = ANSWER_BLOCK_RE.sub("", text).strip()
    if not text:
        raise ValueError("answer is empty after removing tags")
    return f"<answer>{text}</answer>"


def make_user_prompt(question: str, language: str) -> str:
    return (
        f"The question is in {language}. {question}\n"
        "Think through the problem carefully in English, code-switching naturally into the target "
        "language when it helps. If the question is medical, use appropriate medical knowledge and "
        "note meaningful uncertainty. Return your reasoning inside <thinking> tags with ordered "
        f"<step1>, <step2>, ... steps. Return the final direct answer inside <answer> tags, written only in {language}."
    )


def make_example(raw: dict) -> dict:
    language = raw["language"]
    question = raw["question"]
    reasoning = normalize_reasoning(raw["reasoning"])
    answer = normalize_answer(raw["answer"])
    user_prompt = make_user_prompt(question, language)

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


def has_ordered_step_tags(text: str) -> bool:
    stack = []
    for match in re.finditer(r"</?step(\d+)>", text, flags=re.IGNORECASE):
        step_no = match.group(1)
        is_close = text[match.start() + 1] == "/"
        if is_close:
            if not stack or stack.pop() != step_no:
                return False
        else:
            stack.append(step_no)
    return not stack


def validate_examples(dataset_name: str, examples: list[dict]) -> None:
    duplicate_counter = Counter()
    errors = []

    for idx, example in enumerate(examples, start=1):
        messages = example.get("messages", [])
        roles = [message.get("role") for message in messages]
        if roles != ["system", "user", "assistant"]:
            errors.append(f"{dataset_name}:{idx} invalid roles: {roles}")
            continue

        user_content = messages[1].get("content", "")
        assistant_content = messages[2].get("content", "")
        duplicate_counter[user_content] += 1

        if assistant_content.count("<thinking>") != 1 or assistant_content.count("</thinking>") != 1:
            errors.append(f"{dataset_name}:{idx} invalid thinking tags")
        if assistant_content.count("<answer>") != 1 or assistant_content.count("</answer>") != 1:
            errors.append(f"{dataset_name}:{idx} invalid answer tags")
        if not has_ordered_step_tags(assistant_content):
            errors.append(f"{dataset_name}:{idx} invalid step tags")

    if errors:
        sample = "\n".join(errors[:10])
        raise ValueError(f"{dataset_name} validation failed with {len(errors)} errors:\n{sample}")

    duplicate_questions = sum(count - 1 for count in duplicate_counter.values() if count > 1)
    print(
        f"Validated {dataset_name}: {len(examples)} rows, "
        f"{duplicate_questions} duplicate user prompts kept."
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
    combined = spanish + yoruba

    validate_examples("curemed_sft_spanish", spanish)
    validate_examples("curemed_sft_yoruba", yoruba)
    validate_examples("curemed_sft_spanish_yoruba", combined)

    write_jsonl(output_dir / "curemed_sft_spanish.jsonl", spanish)
    write_jsonl(output_dir / "curemed_sft_yoruba.jsonl", yoruba)
    write_jsonl(output_dir / "curemed_sft_spanish_yoruba.jsonl", combined)
    write_dataset_info(output_dir)

    print(f"Wrote Spanish examples: {len(spanish)}")
    print(f"Wrote Yoruba examples: {len(yoruba)}")
    print(f"Wrote combined examples: {len(combined)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
