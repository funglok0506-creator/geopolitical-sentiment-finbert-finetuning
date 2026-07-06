# FinBERT Under Geopolitical Domain Shift
Project Title: FinBERT Under Geopolitical Domain Shift: Evaluating LoRA and Fine-Tuning for Cross-Domain Sentiment

Author: Fung Lok Lam

# Overview
The project assesses the extent that fine-tuning FinBERT on financial text improves or worsens sentiment classification on geopolitical headlines. Three model configurations are compared: LoRA r = 4, LoRA r = 16 and full fine-tuning. 
The key finding discovers that full fine-tuning achieves a validation macro F1 score of 0.955 on the Financial Phrasebank data but drops to 0.37 on geopolitical headlines, which is below the zero-shot baseline 0.49, suggesting negative transfer.

# Setup
'''bash
pip install -r requirements.txt
'''

# How to run 
Scripts are to be run in this order:
'''bash
python collect_datav4.py # serves to scrape geopolitical headlines with given filters
python finbert_initial_analysis # serves to train zero-shot baseline model and evaluate its in-domain performance
python finbert_lora_models.py # trains all three configurations and evaluates in domain performance
python evaluate.py # evaluates models on geopolitical headlines test set
'''

# Key results
| Model          | Val Macro F1| Geo Macro F1 |
|----------------|-------------|--------------|
| Zero-shot      | -           | 0.49         |
| LoRA r = 4     | 0.66        | 0.22*        |
| LoRA r = 16    | 0.89        | 0.27*        |
| Full fine-tune | 0.955       | 0.37         |

*LoRA geo results are possibly artefactual due to adapter loading issue.

# Repository structure
- 'finbert_initial_analysis.py' - zero-shot baseline model evaluation
- 'fibert_lora_models.py' - training pipeline for all 3 configurations
- 'evaluate.py' - domain-shift performance evaluation on geopolitical test set
- 'data/' - raw and labeled geopolitical headline dataset
- 'results/' - computed metrics, prediction performance and plots 
- 'checkpoints/' - saved model weights

# Known issue
LoRA adapter weights did not load correctly when undergoing geopolitical evaluation. Training results for LoRA configurations are valid but geopolitical F1 scores must have cautious interpretation.

# References
- Araci, D. (2019) FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv:1908.10063.
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
- Malo et al. (2014). Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts. Journal of the Association for Information Science and Technology.

# End of README




