import pandas as pd
from pathlib import Path
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
output_dir = PROJECT_ROOT / "outputs" / "baseline_inference" / "qwen2.5_1.5B" / "language_evaluation"

languages = [
    "Amharic", "Bengali", "French", "Hausa", "Hindi", "Japanese", "Korean",
    "Spanish", "Swahili", "Thai", "Turkish", "Vietnamese", "Yoruba"
]


def compute_language_stats():
    per_lang_results = []
    global_true = 0
    global_false = 0
    global_total = 0

    for lang in languages:
        csv_path = output_dir / f"{lang}_qwen2.5_1.5B_language_evaluation.csv"

        if not csv_path.exists():
            print(f"Skipping {lang}: CSV not found at {csv_path}.")
            continue

        df = pd.read_csv(csv_path)
        df["Language_Score"] = df["Language_Score"].astype(str).str.strip().str.title()
        df = df[df["Language_Score"].isin(["True", "False"])]

        total = len(df)
        if total == 0:
            print(f"Skipping {lang}: No evaluated samples.")
            continue

        true_count = (df["Language_Score"] == "True").sum()
        false_count = (df["Language_Score"] == "False").sum()
        accuracy = (true_count / total) * 100 if total > 0 else 0.0

        per_lang_results.append({
            "Language": lang,
            "Total Samples": total,
            "Correct (True)": true_count,
            "Incorrect (False)": false_count,
            "Accuracy (%)": f"{accuracy:.2f}"
        })

        global_true += true_count
        global_false += false_count
        global_total += total

    global_accuracy = (global_true / global_total) * 100 if global_total > 0 else 0.0
    global_stats = {
        "Global Total Samples": global_total,
        "Global Correct (True)": global_true,
        "Global Incorrect (False)": global_false,
        "Global Accuracy (%)": f"{global_accuracy:.2f}"
    }

    return per_lang_results, global_stats


language_per_lang, language_global = compute_language_stats()

print("\nLanguage Statistics Per Language:")
if language_per_lang:
    print(tabulate(language_per_lang, headers="keys", tablefmt="grid"))
else:
    print("No language data available.")

print("\nLanguage Global Statistics:")
print(tabulate([language_global], headers="keys", tablefmt="grid"))
