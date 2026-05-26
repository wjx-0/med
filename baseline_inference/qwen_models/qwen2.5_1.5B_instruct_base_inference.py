# -------------------------------
# Qwen2.5-1.5B-Instruct Batch Inference Script for Test Datasets
# -------------------------------

"""
This script performs batch inference using the Qwen2.5-1.5B-Instruct model on multilingual medical reasoning test datasets.
It supports processing one or all languages, with options for resuming interrupted runs and progressive saving.
Key features:
- 4-bit quantization for memory efficiency.
- Chat template application for instruct tuning.
- Sampling-based generation for varied responses.
- Automatic handling of multiple CSV files per language.
"""


import os

os.environ.setdefault('HF_HOME', '/root/hf_cache')
os.environ.setdefault('TRANSFORMERS_CACHE', '/root/hf_cache')


import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from pathlib import Path


# -------------------------------
# Configuration Section
# -------------------------------
# List of supported languages for processing
languages = [
    "Amharic", "Bengali", "French", "Hausa", "Hindi", "Japanese",
    "Korean", "Spanish", "Swahili", "Thai", "Turkish", "Vietnamese", "Yoruba"
]


# Path to the test datasets directory
test_dir = Path("../../data/test")


# Model and tokenizer configuration
model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Update to match your intended model size


# Quantization config for 4-bit loading with optimized compute dtype to avoid warnings and improve speed
'''quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16  # Matches input dtype (float16) for faster inference
) '''


# System prompt for medical reasoning in the target language
system_prompt = (
    "You are an expert multilingual medical doctor. "
    "Please provide the correct answer to this medical query in the given language only after proper reasoning."
)


# -------------------------------
# Model and Tokenizer Loading Section
# -------------------------------


print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=quantization_config,
    torch_dtype=torch.float16,  # Ensures consistent dtype with compute_dtype for optimal performance
    device_map="auto",  # Automatically distributes across available GPUs
    trust_remote_code=True  # Required for some custom models like Qwen
)


tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)


# Add padding token if not present (common for some tokenizers)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

print("Model and tokenizer loaded successfully.")


# -------------------------------
# Response Generation Function
# -------------------------------
def generate_responses(questions: list[str], max_new_tokens: int = 1024) -> list[str]:
    """
    Generates responses for a batch of questions using the model's chat template.
    
    Args:
        questions (list[str]): The input questions/prompts.
        max_new_tokens (int): Maximum number of new tokens to generate.
    
    Returns:
        list[str]: The generated response texts.
    """
    prompts = []
    for question in questions:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    # Tokenize the prompts as a padded batch and move to device
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    input_width = inputs.input_ids.shape[1]
    
    # Generate responses with no_grad for inference efficiency
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,  # Enables sampling for more varied responses
            temperature=0.6,  # Controls randomness (lower = more deterministic)
            top_p=0.95,      # Nucleus sampling threshold
            top_k=20,        # Limits sampling to top-k tokens
            repetition_penalty=1.2,  # Discourages repetition
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens (excluding padded input prompt)
    responses = []
    for output_ids in outputs:
        response = tokenizer.decode(
            output_ids[input_width:],
            skip_special_tokens=True
        )
        responses.append(response.strip())

    return responses

# -------------------------------
# Main Processing Section
# -------------------------------
if __name__ == "__main__":
    # User inputs for flexibility
    selected_language = input("Enter the language to process (or 'all' for all languages): ").strip().capitalize()
    num_samples = int(input("Enter the number of samples per language (-1 for all): "))
    batch_size = max(1, int(input("Enter batch size for inference (e.g., 8): ")))
    
    # Determine languages to process
    if selected_language.lower() == 'all':
        process_languages = languages
    else:
        process_languages = [selected_language]
    
    # Create output directory if it doesn't exist
    output_dir = "../../outputs/baseline_inference/qwen2.5_1.5B/generations"  # Updated to match model size
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each selected language
    for lang in process_languages:
        # Validate language
        if lang not in languages:
            print(f"Invalid language: {lang}. Skipping.")
            continue
        
        # Load the CSV file for the language
        csv_path = test_dir / f"{lang}.csv"
        if not csv_path.exists():
            print(f"File not found: {csv_path}. Skipping.")
            continue
        
        df = pd.read_csv(csv_path)
        total_rows = len(df)
        process_rows = total_rows if num_samples == -1 else min(num_samples, total_rows)
        df = df.iloc[:process_rows].copy()
        
        # Initialize 'Test Response' column if needed
        if 'Test Response' not in df.columns:
            df['Test Response'] = ''
        
        # Define output filename and path
        new_filename = f"{lang}_qwen2.5_1.5B_instruct_inference_data.csv"  # Updated to match model size
        save_path = os.path.join(output_dir, new_filename)
        
        # Resume logic: Load existing data if file exists
        resume_from = 0
        if os.path.exists(save_path):
            existing_df = pd.read_csv(save_path)
            if 'Test Response' in existing_df.columns:
                completed_rows = existing_df[existing_df['Test Response'].notna()].shape[0]
                if completed_rows > 0 and completed_rows <= len(df):
                    print(f"Resuming from {completed_rows} completed rows in {new_filename}")
                    df.iloc[:completed_rows] = existing_df.iloc[:completed_rows]
                    resume_from = completed_rows
        
        # Generate responses in batches, skipping completed rows
        pending_indices = [
            idx for idx in range(resume_from, len(df))
            if not (
                pd.notna(df.at[idx, 'Test Response'])
                and str(df.at[idx, 'Test Response']).strip()
            )
        ]

        for start in range(0, len(pending_indices), batch_size):
            batch_indices = pending_indices[start:start + batch_size]
            questions = [df.at[idx, 'Question'] for idx in batch_indices]

            print(
                f"Generating responses for rows "
                f"{batch_indices[0] + 1}-{batch_indices[-1] + 1}/{process_rows} in {lang}..."
            )

            # Generate and assign responses
            batch_responses = generate_responses(questions)
            for idx, test_response in zip(batch_indices, batch_responses):
                df.at[idx, 'Test Response'] = test_response

            # Save progress after each batch to avoid data loss
            df.to_csv(save_path, index=False)
            print(f"Saved progress through row {batch_indices[-1] + 1}.")
        
        # Final confirmation
        print(f"Completed processing for {lang}. Full results saved to: {save_path}")
    
    print("All processing completed.")
