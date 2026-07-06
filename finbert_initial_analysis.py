# Import libraries
import pandas as pd
import json
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score

import os; os.makedirs('results', exist_ok=True)

# Importing geopolitical news article dataset
news_data = pd.read_csv('geo_candidates_clean_fv.csv')

# Zero shot FinBERT
finbert_pipeline = pipeline('text-classification',
                            model = "ProsusAI/finbert",
                            tokenizer="ProsusAI/finbert",
                            device = -1)

# Run through an initial inference using original Finbert model
results = finbert_pipeline(news_data['headline'].tolist(), batch_size = 16)

# Mapping labeling outcome
label_mapping = {'positive': 'Pos', 'negative': 'Neg', 'neutral': 'Neu'}
news_data['predicted'] = [label_mapping[r['label']] for r in results]
news_data['confidence'] = [r['score'] for r in results]

# Save predictions from original FinBERT model
news_data.to_csv('results/baseline_predictions.csv', index = False)

# Evaluation on 3 Sentiment Classes: Positive, Neutral, Negative

print(classification_report(
    news_data['label'], news_data['predicted'],
    labels = ['Pos', 'Neu', 'Neg'],
    target_names=['Pos', 'Neu', 'Neg']
))


# Tier breakdown
for theme in news_data['theme'].unique():
    subset = news_data[news_data['theme'] == theme]
    print(f'\n--- Theme {theme} (n={len(subset)}) ---')
    print(classification_report(
        subset['label'], subset['predicted'],
        labels = ['Pos', 'Neu', 'Neg']
    ))

# Save metrics as a Json file for report
report_metric = classification_report(
    news_data['label'], news_data['predicted'],
    labels = ['Pos', 'Neu', 'Neg'],
    output_dict = True
)

with open('results/baseline_metrics.json', 'w') as f:
    json.dump(report_metric, f, indent = 4)

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

os.makedirs('results/plots', exist_ok=True)

#  Confusion Matrix
cm = confusion_matrix(news_data['label'], news_data['predicted'], labels = ['Pos', 'Neu', 'Neg'])
plt.figure(figsize = (6, 5))
sns.heatmap(cm, annot = True, fmt = 'd', xticklabels = ['Pos', 'Neu', 'Neg'],
            yticklabels = ['Pos', 'Neu', 'Neg'], cmap = 'Blues')
plt.title('FinBERT Baseline - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('results/plots/finbert_baseline_confusion_matrix.png')
plt.close()

# Per-class F1 Bar Chart
labels = ['Pos', 'Neu', 'Neg']
f1_scores = [report_metric[l]['f1-score'] for l in labels]
plt.figure(figsize = (6, 4))
plt.bar(labels, f1_scores, color=['green', 'blue', 'salmon'])
plt.title('FinBERT Baseline - F1 Score by Class')
plt.ylabel('F1 Score')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('results/plots/finbert_baseline_f1_by_class.png')
plt.close()

print("Plots saved to results/plots/")
    
