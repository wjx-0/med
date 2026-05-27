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

system_prompt = """You are an expert multilingual medical doctor. When answering a medical question, follow these steps:
1. First, search your internal knowledge base thoroughly for relevant background information about the topic.
2. Understand and reason the question fully in English first.
3. Reason mainly in English, but code-switch naturally into the target language whenever useful for clarity or domain accuracy.
4. Consider multiple perspectives and potential answers before settling on your final response.
5. Evaluate the confidence in your answer based on the information available to you.
6. Provide the final answer clearly in the target language, making sure it's well-supported by your reasoning.
7. If there are significant uncertainties or gaps in your knowledge, acknowledge them transparently.

Your goal is to provide accurate, well-reasoned responses that demonstrate depth of understanding, not just surface-level answers.

You are an expert multilingual medical doctor. When answering a medical question, think and reason mainly in English with natural code-switching to the target language. Use multi-step reasoning wrapped in <step> tags inside <thinking>."""


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


def generate_responses(model, tokenizer, questions: list[str], language: str, max_new_tokens: int) -> list[str]:
    prompts = []
    for question in questions:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"The question is in {language}. {question}\n"
                    "Please think carefully with English-guided reasoning and natural code-switching. "
                    "Return your reasoning inside <thinking> </thinking> tags, using numbered "
                    "<step1>, <step2>, ... tags when appropriate. "
                    f"Return the final answer inside <answer> </answer> tags. "
                    f"The final answer inside <answer> MUST be written only in {language}. "
                    "Do not use English or any other language inside <answer>."
                ),
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

        resume_from = 0
        if save_path.exists():
            existing_df = pd.read_csv(save_path)
            if "Test Response" in existing_df.columns:
                completed_rows = existing_df[
                    existing_df["Test Response"].notna()
                    & existing_df["Test Response"].astype(str).str.strip().astype(bool)
                ].shape[0]
                if 0 < completed_rows <= len(df):
                    print(f"Resuming from {completed_rows} completed rows in {save_path.name}")
                    df.iloc[:completed_rows] = existing_df.iloc[:completed_rows]
                    resume_from = completed_rows

        pending_indices = [
            idx for idx in range(resume_from, len(df))
            if not (pd.notna(df.at[idx, "Test Response"]) and str(df.at[idx, "Test Response"]).strip())
        ]

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
