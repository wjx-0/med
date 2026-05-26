import openai
from dotenv import load_dotenv
import pandas as pd
import argparse
import time
import os
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")


def setup_deepseek():
    """
    Configure the DeepSeek API client using OpenAI-compatible API.
    """
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_deepseek_api_key_here":
        raise ValueError("No DeepSeek API key found. Set DEEPSEEK_API_KEY in med/.env.")
    client = openai.OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )
    return client


client = setup_deepseek()

parser = argparse.ArgumentParser(description="Evaluate language of responses using DeepSeek as a judge.")
parser.add_argument("--language", type=str, default=None, help="Language to evaluate, e.g. Spanish. Required if --all is not used.")
parser.add_argument("--all", action="store_true", help="Process all languages instead of a single one.")
parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to process per language. Use -1 to process all.")
parser.add_argument(
    "--run_dir",
    type=str,
    default=None,
    help=(
        "Directory containing generations/ and receiving language_evaluation/. "
        "Defaults to the Qwen2.5-1.5B baseline run."
    ),
)
args = parser.parse_args()

languages = [
    "Amharic", "Bengali", "French", "Hausa", "Hindi", "Japanese", "Korean",
    "Spanish", "Swahili", "Thai", "Turkish", "Vietnamese", "Yoruba"
]

if args.all:
    selected_languages = languages
else:
    if not args.language:
        raise ValueError("Must provide --language or use --all flag.")
    selected_languages = [args.language.strip().capitalize()]

if args.run_dir:
    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = (Path.cwd() / run_dir).resolve()
else:
    run_dir = PROJECT_ROOT / "outputs" / "baseline_inference" / "qwen2.5_1.5B"

dataset_dir = run_dir / "generations"
output_dir = run_dir / "language_evaluation"
output_dir.mkdir(parents=True, exist_ok=True)


def get_language_evaluation(question, test_response, language):
    """
    Evaluate whether the model-generated response is in the target language.
    Returns "True" if criteria are met, "False" otherwise.
    """
    user_prompt = f"""
<Question>
{question}
</Question>
<Model Response>
{test_response}
</Model Response>
You are provided with a question in {language} and a model-generated response (<Model Response>).
Determine if the model response is entirely in {language} (the language of the question)? Check for no mixing with other languages, no unnecessary repetitions, no drifts, and a proper conclusion. Your task is to simply output "True" if the criteria is met, and "False" otherwise.
"""
    for attempt in range(3):
        try:
            gen_response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": user_prompt}]
            )
            output = gen_response.choices[0].message.content.strip()
            if output.lower() == "true":
                return "True"
            if output.lower() == "false":
                return "False"
            raise ValueError(f"Invalid output: {output}")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for language evaluation. Error: {e}")
            time.sleep(2)
    print("Skipping language evaluation after failures.")
    return None


for lang in selected_languages:
    if lang not in languages:
        print(f"Invalid language: {lang}. Skipping.")
        continue

    input_csv_path = dataset_dir / f"{lang}_qwen2.5_1.5B_instruct_inference_data.csv"
    output_csv_path = output_dir / f"{lang}_qwen2.5_1.5B_language_evaluation.csv"

    print(f"Loading data for {lang} from {input_csv_path}...")
    if not input_csv_path.exists():
        print(f"Input CSV not found for {lang}: {input_csv_path}. Skipping.")
        continue

    df = pd.read_csv(input_csv_path)

    if "Language_Score" not in df.columns:
        df["Language_Score"] = None

    if output_csv_path.exists():
        existing_df = pd.read_csv(output_csv_path)
        min_len = min(len(df), len(existing_df))
        df.loc[:min_len - 1, "Language_Score"] = existing_df.loc[:min_len - 1, "Language_Score"]
        print(f"Resuming from existing output CSV for {lang}. Evaluated rows will be skipped.")
    else:
        print(f"Starting fresh for {lang}.")

    if args.num_samples != -1:
        df = df.head(args.num_samples)
        print(f"Limiting to first {args.num_samples} samples for {lang}.")

    unevaluated_mask = df["Language_Score"].isnull()
    unevaluated_indices = df[unevaluated_mask].index.tolist()

    if not unevaluated_indices:
        print(f"All rows already evaluated for {lang}. Skipping.")
        continue

    print(f"Evaluating {len(unevaluated_indices)} remaining responses in {lang}...")

    for idx in tqdm(unevaluated_indices):
        row = df.loc[idx]
        question = row.get("Question", "")
        test_response = row.get("Test Response", "")

        if not all(isinstance(x, str) and x.strip() for x in [question, test_response]):
            print(f"Skipping invalid row at index {idx} for {lang}")
            continue

        language_evaluation = get_language_evaluation(question, test_response, lang)
        if language_evaluation is not None:
            df.at[idx, "Language_Score"] = language_evaluation

        df.to_csv(output_csv_path, index=False)
        time.sleep(1)

    print(f"Completed language evaluation for {lang}. Results saved to: {output_csv_path}")
