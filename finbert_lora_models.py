# Import Libraries
import pandas as pd
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os; os.makedirs('results/plots', exist_ok=True)

from datasets import load_dataset
# Models for sentiment/text classification
    # AutoModelForSequenceClassification: loads NN weights, adds decision heads in addition to category selection
from transformers import(AutoTokenizer, AutoModelForSequenceClassification,
                         TrainingArguments, Trainer)
 
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Step 2: load the Financial Phrasebank
finbank_data = load_dataset("takala/financial_phrasebank", "sentences_allagree")
# Labelling: 0 = negative, 1 = neutral, 2 = positive


# Splitting data to train/test
split = finbank_data['train'].train_test_split(test_size = 0.2, seed = 42)
train_data = split['train']
val_data = split['test']

# Step 3: Tokenization
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

def tokenize(batch):
    return tokenizer(batch['sentence'], truncation = True,
                     padding = 'max_length',
                     max_length = 128)

train_data = train_data.map(tokenize, batched = True)
val_data = val_data.map(tokenize, batched = True)
train_data = train_data.rename_column("label", "labels")
val_data = val_data.rename_column("label", "labels")
train_data.set_format('torch', columns = ['input_ids','attention_mask','labels'])
val_data.set_format('torch', columns = ['input_ids','attention_mask','labels'])

# Metrics Function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Get class with highest logit
    preds = np.argmax(logits, axis = -1)
    # Calculate metrics
    # Using 'macro' average is standard for balanced reporting across classes
    
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average = 'macro')
    recall = recall_score(labels, preds, average = 'macro')
    f1 = f1_score(labels, preds, average = 'macro')
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'macro_f1': f1
    }

# Training Configurations
configurations = {
    'lora_r4': {'lora': True, 'r': 4,  'alpha': 8},
    'lora_r16': {'lora': True, 'r': 16, 'alpha': 32},
    'full_ft':  {'lora': False}
}

loss_curves = {}

for configuration_name, configuration in configurations.items():
    print(f'\n{'='*50}\nTraining: {configuration_name}\n{'='*50}')
    # Import fresh model for each run
    model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert", num_labels=3)

# Use Lora
    if configuration['lora']:
        lora_config = LoraConfig(
            task_type = TaskType.SEQ_CLS,
            r = configuration['r'],
            lora_alpha = configuration['alpha'],
            lora_dropout = 0.1,
            target_modules = ['query', 'value']
            )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

# Training arguments
    training_args = TrainingArguments(
        output_dir = f'checkpoints/{configuration_name}',
        num_train_epochs = 5,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 32,
        learning_rate = 2e-5,
        eval_strategy = 'epoch',
        save_strategy = 'epoch',
        load_best_model_at_end = True,
        metric_for_best_model = 'macro_f1',
        logging_steps = 10,
        seed = 42,
        report_to = 'none'
    )

    trainer = Trainer(model = model, args = training_args,
                      train_dataset = train_data,
                      eval_dataset = val_data,
                      compute_metrics = compute_metrics,
                    )
    trainer.train()

# Save the loss curve
    loss_curves[configuration_name] = {
        'train_loss': [x['loss'] for x in trainer.state.log_history
                       if 'loss' in x
        ],
        'eval_loss': [x['eval_loss'] for x in trainer.state.log_history
                      if 'eval_loss' in x
        ],
        'eval_macro_f1': [x['eval_macro_f1'] for x in trainer.state.log_history
                          if 'eval_macro_f1' in x
        ]
    }

# Save model checkpoint
    model.save_pretrained(f'checkpoints/{configuration_name}/final')
    tokenizer.save_pretrained(f'checkpoints/{configuration_name}/final')
    print(f'Saved checkpoint: checkpoints/{configuration_name}/final')

# Save all loss curves
with open('results/loss_curves.json', 'w') as f:
    json.dump(loss_curves, f, indent = 2)

print('\nAll training complete. Loss curves saved.')

# Results visualizations

# Loss curves for each model
for configuration_name, curves in loss_curves.items():
    plt.plot(curves['train_loss'], label = 'Train Loss')
    plt.plot(curves['eval_loss'], label = 'Eval Loss')
    plt.title(f'{configuration_name} - Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'results/plots/{configuration_name}_loss_curve.png')
    plt.close()

# F1 learning curves across each epoch

for configuration_name, curves in loss_curves.items():
    plt.plot(curves['eval_macro_f1'], marker = 'o')
    plt.title(f'{configuration_name} - Macro F1 Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1')
    plt.savefig(f'results/plots/{configuration_name}_f1_curve.png')
    plt.close()


# F1 bar chart
# Run after all configs are done
configs = list(loss_curves.keys())
final_f1s = [max(loss_curves[c]['eval_macro_f1']) for c in configs]
plt.bar(configs, final_f1s, color=['steelblue', 'salmon', 'green'])
plt.axhline(y= 0.48, color = 'gray', linestyle = '--', label='Baseline F1')  # your baseline macro F1
plt.title('Best Macro F1 by Configuration')
plt.ylabel('Macro F1')
plt.legend()
plt.savefig('results/plots/config_comparison_f1.png')
plt.close()



# Conf matrix
preds_output = trainer.predict(val_data)
preds = np.argmax(preds_output.predictions, axis = -1)
label_names = ['Neg', 'Neu', 'Pos']
conf_matrix = confusion_matrix(preds_output.label_ids, preds)
sns.heatmap(conf_matrix, annot = True, fmt = 'd', xticklabels = label_names, yticklabels = label_names, cmap='Blues')
plt.title(f'{configuration_name} - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f'results/plots/{configuration_name}_confusion_matrix.png')
plt.close()

    
    
