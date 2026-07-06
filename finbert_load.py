
# Import libraries and packages
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
print(torch.__version__)
print(torch.cuda.is_available())


# Import Prototype FinBERY model

model_name = 'ProsusAI/finbert'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "The company reported strong earnings growth."

inputs = tokenizer(text, return_tensors="pt", truncation=True)
outputs = model(**inputs)

probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(probs)

# Use GPU acceleration instead of MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
inputs = {k:v.to(device) for k,v in inputs.items()}

