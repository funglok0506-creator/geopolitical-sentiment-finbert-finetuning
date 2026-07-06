# Import libraries
import pandas as pd
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, AutoPeftModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

TEST_PATH = "data/geo_candidates_clean_fv.csv"
BASE_MODEL = "ProsusAI/finbert"

CONFIGS = {
    "lora_r4":  {'path': 'checkpoints/lora_r4/final',  'is_lora': True},
    "lora_r16": {"path": 'checkpoints/lora_r16/final', "is_lora": True},
    'full_ft':  {'path': 'checkpoints/full_ft/final',  'is_lora': False},
}

ID2LABEL = {0: 'Neg', 1: 'Neu', 2: 'Pos'}
LABELS   = ["Pos", "Neu", "Neg"]

# Load test set
df = pd.read_csv(TEST_PATH)
df["label"] = df["label"].str.strip()
df = df[df["label"].isin(LABELS)].reset_index(drop=True)
print(f"Test set: {len(df)} headlines")
print(df["label"].value_counts())

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def run_inference(model, headlines):
    model.eval()
    preds, confs = [], []
    with torch.no_grad():
        for headline in headlines:
            inputs = tokenizer(
                headline,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            )
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze()
            pred_id = torch.argmax(probs).item()
            preds.append(ID2LABEL[pred_id])
            confs.append(probs[pred_id].item())
    return preds, confs

all_results = {}

for config_name, config in CONFIGS.items():
    
    if config['is_lora']:
        # ensure PEFT adapter loading correctly
        model = AutoPeftModelForSequenceClassification.from_pretrained(
            config["path"],
            num_labels=3,
            ignore_mismatched_sizes=True
        )
        model = model.merge_and_unload()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            config["path"],
            num_labels=3
        )

    # Run inference
    preds, confs = run_inference(model, df["headline"].tolist())
    df[f"pred_{config_name}"]  = preds
    df[f"conf_{config_name}"]  = confs

    # Create Classification report
    report = classification_report(
        df["label"], preds,
        labels = LABELS,
        output_dict = True,
        zero_division=0
    )
    print(classification_report(
        df["label"], preds,
        labels=LABELS,
        zero_division=0
    ))

    all_results[config_name] = report

    # Create confusion matrix
    cm = confusion_matrix(df["label"], preds, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)
    disp.plot(ax=ax, colorbar=True)
    ax.set_title(f'{config_name} — Geo Test Set Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'results/{config_name}_geo_confusion.png', dpi = 150)
    plt.close()
    print(f'Saved: results/{config_name}_geo_confusion.png')

    # Tier breakdown - only if tier column exists
    if "tier" in df.columns:
        for tier in sorted(df["tier"].dropna().unique()):
            subset = df[df["tier"] == tier]
            t_preds = subset[f"pred_{config_name}"].tolist()
            t_true  = subset["label"].tolist()
            t_report = classification_report(
                t_true, t_preds,
                labels=LABELS,
                output_dict=True,
                zero_division=0
            )
            all_results[f'{config_name}_tier{int(tier)}'] = t_report
            print(f'\n-- Tier {int(tier)} (n={len(subset)}) --')
            print(classification_report(
                t_true, t_preds,
                labels=LABELS,
                zero_division=0
            ))

# Summary comparison plot
BASELINE_MACRO_F1 = 0.49  # from baseline.py result

config_names  = list(CONFIGS.keys())
macro_f1s     = [all_results[c]["macro avg"]["f1-score"] for c in config_names]

fig, ax = plt.subplots(figsize=(7, 5))
colors = ['#4878CF', '#FC8D59', '#2CA02C']
bars = ax.bar(config_names, macro_f1s, color=colors)
ax.axhline(BASELINE_MACRO_F1, color="gray", linestyle="--", label=f'Zero-shot Baseline ({BASELINE_MACRO_F1})')
ax.set_ylim(0, 1.0)
ax.set_ylabel('Macro F1 — Geopolitical Test Set')
ax.set_title('Domain-Shift Performance: All Configs vs Baseline')
ax.legend()
for bar, val in zip(bars, macro_f1s):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
            f"{val:.3f}", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig('results/geo_config_comparison_f1.png', dpi = 150)
plt.close()
print('\nSaved: results/geo_config_comparison_f1.png')

# Save all predictions
df.to_csv("results/geo_all_predictions.csv", index=False)
print("Saved: results/geo_all_predictions.csv")

# Save metrics JSON
with open("results/geo_metrics.json", "w") as f:
    json.dump(all_results, f, indent=2)
print('Saved: results/geo_metrics.json')
