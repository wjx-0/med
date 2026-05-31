"""Batch inference for a Qwen2.5-1.5B LoRA SFT adapter."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/root/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/root/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "saves" / "qwen2.5_1.5B_qlora_spanish"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sft" / "qwen2.5_1.5B_qlora_spanish" / "generations"

languages = [
    "Amharic", "Bengali", "French", "Hausa", "Hindi", "Japanese",
    "Korean", "Spanish", "Swahili", "Thai", "Turkish", "Vietnamese", "Yoruba"
]

test_dir = PROJECT_ROOT / "data" / "test"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

system_prompt = """You are an expert multilingual reasoning assistant with strong medical knowledge. Think carefully in English first, code-switching naturally into the target language when useful for clarity or domain accuracy. For medical questions, use appropriate clinical knowledge and acknowledge meaningful uncertainty. Provide accurate, well-supported responses with concise multi-step reasoning inside <thinking> and a final answer inside <answer>."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter_dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    return parser.parse_args()


def load_model_and_tokenizer(adapter_dir: Path):
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_dir}. "
            "Train the LoRA adapter first or pass --adapter_dir."
        )

    print("Loading base model, tokenizer, and LoRA adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir, local_files_only=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Model, tokenizer, and adapter loaded successfully.")
    return model, tokenizer


def make_user_prompt(question: str, language: str) -> str:
    return (
        f"The question is in {language}. {question}\n"
        "Think through the problem carefully in English, code-switching naturally into the target "
        "language when it helps. If the question is medical, use appropriate medical knowledge and "
        "note meaningful uncertainty. Return your reasoning inside <thinking> tags with ordered "
        f"<step1>, <step2>, ... steps. Return the final direct answer inside <answer> tags, written only in {language}."
    )


def has_response(value) -> bool:
    return pd.notna(value) and bool(str(value).strip())


def restore_existing_responses(df: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    if not save_path.exists():
        return df

    existing_df = pd.read_csv(save_path)
    if "Question" not in existing_df.columns or "Test Response" not in existing_df.columns:
        print(f"Existing output {save_path.name} is missing Question/Test Response; starting without resume.")
        return df

    copied_rows = 0
    skipped_mismatches = 0
    compare_rows = min(len(df), len(existing_df))
    for idx in range(compare_rows):
        if str(existing_df.at[idx, "Question"]) != str(df.at[idx, "Question"]):
            skipped_mismatches += 1
            continue
        response = existing_df.at[idx, "Test Response"]
        if has_response(response):
            df.at[idx, "Test Response"] = response
            copied_rows += 1

    if copied_rows or skipped_mismatches or len(existing_df) != len(df):
        print(
            f"Resumed {copied_rows} completed rows from {save_path.name}; "
            f"skipped {skipped_mismatches} question mismatches."
        )
    if len(existing_df) != len(df):
        print(f"Existing output has {len(existing_df)} rows; current run has {len(df)} rows.")

    return df


def generate_responses(model, tokenizer, questions: list[str], language: str, max_new_tokens: int) -> list[str]:
    prompts = []
    for question in questions:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": make_user_prompt(question, language),
            },
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    device = next(model.parameters()).device
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    input_width = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    responses = []
    for output_ids in outputs:
        response = tokenizer.decode(output_ids[input_width:], skip_special_tokens=True)
        responses.append(response.strip())
    return responses


def main() -> None:
    args = parse_args()
    adapter_dir = args.adapter_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(adapter_dir)

    selected_language = input("Enter the language to process (or 'all' for all languages): ").strip().capitalize()
    num_samples = int(input("Enter the number of samples per language (-1 for all): "))
    batch_size = max(1, int(input("Enter batch size for inference (e.g., 16): ")))

    process_languages = languages if selected_language.lower() == "all" else [selected_language]

    for lang in process_languages:
        if lang not in languages:
            print(f"Invalid language: {lang}. Skipping.")
            continue

        csv_path = test_dir / f"{lang}.csv"
        if not csv_path.exists():
            print(f"File not found: {csv_path}. Skipping.")
            continue

        df = pd.read_csv(csv_path)
        total_rows = len(df)
        process_rows = total_rows if num_samples == -1 else min(num_samples, total_rows)
        df = df.iloc[:process_rows].copy()

        if "Test Response" not in df.columns:
            df["Test Response"] = ""

        save_path = output_dir / f"{lang}_qwen2.5_1.5B_instruct_inference_data.csv"
        df = restore_existing_responses(df, save_path)

        pending_indices = [idx for idx in range(len(df)) if not has_response(df.at[idx, "Test Response"])]
        if not pending_indices:
            print(f"No pending rows for {lang}; existing results are complete.")
            continue

        for start in range(0, len(pending_indices), batch_size):
            batch_indices = pending_indices[start:start + batch_size]
            questions = [df.at[idx, "Question"] for idx in batch_indices]
            print(
                f"Generating responses for rows "
                f"{batch_indices[0] + 1}-{batch_indices[-1] + 1}/{process_rows} in {lang}..."
            )

            batch_responses = generate_responses(
                model,
                tokenizer,
                questions,
                lang,
                args.max_new_tokens,
            )
            for idx, test_response in zip(batch_indices, batch_responses):
                df.at[idx, "Test Response"] = test_response

            df.to_csv(save_path, index=False)
            print(f"Saved progress through row {batch_indices[-1] + 1}.")

        print(f"Completed processing for {lang}. Full results saved to: {save_path}")

    print("All processing completed.")


if __name__ == "__main__":
    main()
