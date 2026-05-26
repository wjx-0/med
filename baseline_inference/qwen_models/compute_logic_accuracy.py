import pandas as pd
from pathlib import Path
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
output_dir = PROJECT_ROOT / "outputs" / "baseline_inference" / "qwen2.5_1.5B" / "logic_evaluation"

languages = [
    "Amharic", "Bengali", "French", "Hausa", "Hindi", "Japanese", "Korean",
    "Spanish", "Swahili", "Thai", "Turkish", "Vietnamese", "Yoruba"
]


def compute_logic_stats():
    per_lang_results = []
    global_true = 0
    global_false = 0
    global_total = 0

    for lang in languages:
        csv_path = output_dir / f"{lang}_qwen2.5_1.5B_logic_evaluation.csv"

        if not csv_path.exists():
            print(f"Skipping {lang}: CSV not found at {csv_path}.")
            continue

        df = pd.read_csv(csv_path)
        df["Logic_Score"] = df["Logic_Score"].astype(str).str.strip().str.title()
        df = df[df["Logic_Score"].isin(["True", "False"])]

        total = len(df)
        if total == 0:
            print(f"Skipping {lang}: No evaluated samples.")
            continue

        true_count = (df["Logic_Score"] == "True").sum()
        false_count = (df["Logic_Score"] == "False").sum()
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


logic_per_lang, logic_global = compute_logic_stats()

print("\nLogic Statistics Per Language:")
if logic_per_lang:
    print(tabulate(logic_per_lang, headers="keys", tablefmt="grid"))
else:
    print("No logic data available.")

print("\nLogic Global Statistics:")
print(tabulate([logic_global], headers="keys", tablefmt="grid"))
