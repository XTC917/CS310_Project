import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import load_ghostbuster_data
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm

# Define the path to your Ghostbuster English dataset
dataset_path_english = 'D:\\CSstudy\\NLP\\Project\\ghostbuster-data\\ghostbuster-data'

# Load the English dataset
print(f"Loading English dataset from {dataset_path_english}...")
df_english = load_ghostbuster_data(dataset_path_english)
print(f"Loaded {len(df_english)} English samples.")
print(df_english.head())

# Define topics for training and OOD testing
training_topics = ['essay', 'reuter']
ood_test_topics = ['wp']

# Split data into training and OOD test sets based on topics
df_train = df_english[df_english['topic'].isin(training_topics)].reset_index(drop=True)
df_ood_test = df_english[df_english['topic'].isin(ood_test_topics)].reset_index(drop=True)

print(f"Training samples (topics: {training_topics}): {len(df_train)}")
print(f"OOD Test samples (topics: {ood_test_topics}): {len(df_ood_test)}")

# Further split training data into train and validation sets for supervised learning
# We'll use stratified split to maintain the proportion of human/LLM samples
# Using a small validation set for simplicity, adjust test_size as needed
df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=42, stratify=df_train['label'])

print(f"Supervised Training samples: {len(df_train)}")
print(f"Supervised Validation samples: {len(df_val)}")

# We also need an in-domain test set from the training topics for complete evaluation
df_in_domain_test = df_train.sample(frac=0.2, random_state=42) # Take 20% of remaining train as in-domain test
df_train = df_train.drop(df_in_domain_test.index).reset_index(drop=True)
df_in_domain_test = df_in_domain_test.reset_index(drop=True)

print(f"Supervised Training samples (after splitting in-domain test): {len(df_train)}")
print(f"In-Domain Test samples: {len(df_in_domain_test)}")

# Combine in-domain and OOD test sets for overall evaluation
df_combined_test = pd.concat([df_in_domain_test, df_ood_test]).reset_index(drop=True)

print(f"Combined Test samples (In-Domain + OOD): {len(df_combined_test)}")

# Now we have:
# df_train: for supervised model training
# df_val: for supervised model validation during training
# df_in_domain_test: for evaluating supervised model on seen topics
# df_ood_test: for evaluating supervised model and zero-shot model on unseen topics
# df_combined_test: for evaluating both models on the full test set

# --- Supervised Learning Implementation ---

# 1. Tokenize and prepare data for the Transformer model.

# Choose a pre-trained tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True, cache_dir='D:\\CSstudy\\NLP\\Project\\models')

# Define a custom Dataset class
class TextDetectionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.texts = dataframe['text']
        self.labels = dataframe['label']
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create Dataset instances
train_dataset = TextDetectionDataset(df_train, tokenizer)
val_dataset = TextDetectionDataset(df_val, tokenizer)
in_domain_test_dataset = TextDetectionDataset(df_in_domain_test, tokenizer)
ood_test_dataset = TextDetectionDataset(df_ood_test, tokenizer)
combined_test_dataset = TextDetectionDataset(df_combined_test, tokenizer)

# Define DataLoaders
BATCH_SIZE = 32  # 使用GPU时可以增大batch size

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
in_domain_test_dataloader = DataLoader(in_domain_test_dataset, batch_size=BATCH_SIZE)
ood_test_dataloader = DataLoader(ood_test_dataset, batch_size=BATCH_SIZE)
combined_test_dataloader = DataLoader(combined_test_dataset, batch_size=BATCH_SIZE)

# 2. Define and train a Transformer-based binary classification model.

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

# Move model to device
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, local_files_only=True, cache_dir='D:\\CSstudy\\NLP\\Project\\models')
model = model.to(device)

# Set up optimizer and scheduler
EPOCHS = 4 # You can adjust the number of epochs
optimizer = AdamW(model.parameters(), lr=2e-5) # Learning rate
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Training loop
print("Starting supervised model training...")

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    model.train()
    total_loss = 0
    
    # Add progress bar
    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
    
    for step, batch in enumerate(progress_bar):
        # Move batch to GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        model.zero_grad()

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        # Update progress bar with GPU memory info
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'GPU Memory': f'{gpu_memory:.0f}MB'
            })
        else:
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    total_val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_val_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1).flatten()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"Average validation loss: {avg_val_loss:.4f}")

    # You can add validation accuracy/metrics here if needed

print("Supervised model training finished.")

# 3. Evaluate the supervised model on df_in_domain_test and df_ood_test.

def evaluate_model(model, data_loader, device):
    """
    Evaluates the model on the given data loader.
    Returns predicted labels and true labels.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
            )

            preds = torch.argmax(outputs.logits, dim=1).flatten()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds

def calculate_metrics(true_labels, predicted_labels):
    """
    Calculates evaluation metrics.
    """
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    # Note: roc_auc_score requires prediction probabilities, not just labels.
    # For simplicity with current setup, we'll skip AUROC for now or add it later
    # if we modify the evaluation to return probabilities.
    # auroc = roc_auc_score(true_labels, predicted_probabilities)

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        # 'AUROC': auroc
    }
    return metrics

# Evaluate on in-domain test set
print("Evaluating supervised model on in-domain test set...")
in_domain_true_labels, in_domain_preds = evaluate_model(model, in_domain_test_dataloader, device)
in_domain_metrics = calculate_metrics(in_domain_true_labels, in_domain_preds)

print("In-Domain Test Metrics:")
for metric, value in in_domain_metrics.items():
    print(f"{metric}: {value:.4f}")

# Evaluate on OOD test set
print("\nEvaluating supervised model on OOD test set...")
ood_true_labels, ood_preds = evaluate_model(model, ood_test_dataloader, device)
ood_metrics = calculate_metrics(ood_true_labels, ood_preds)

print("OOD Test Metrics (Supervised Model):")
for metric, value in ood_metrics.items():
    print(f"{metric}: {value:.4f}")

# --- Fast-DetectGPT Implementation ---

# 1. Integrate or adapt Fast-DetectGPT code.

def get_sampling_discrepancy_score(text, scoring_model, scoring_tokenizer, sampling_model, sampling_tokenizer, device, nsamples=10000):
    """
    Calculates the Fast-DetectGPT sampling discrepancy score for a single text.
    Adapted from Fast-DetectGPT repository.
    """
    # Tokenize text
    tokenized = scoring_tokenizer(text, return_tensors="pt", padding=True, truncation=True, return_token_type_ids=False).to(device)
    labels = tokenized.input_ids[:, 1:]

    if labels.shape[-1] == 0: # Handle empty or too short texts after truncation
        return 0.0 # Or NaN, depending on desired behavior for very short texts

    with torch.no_grad():
        # Get logits from scoring model
        logits_score = scoring_model(**tokenized).logits[:, :-1]

        # Get logits from sampling model
        if sampling_model is scoring_model and sampling_tokenizer is scoring_tokenizer:
            logits_ref = logits_score
        else:
            tokenized_sampling = sampling_tokenizer(text, return_tensors="pt", padding=True, truncation=True, return_token_type_ids=False).to(device)
            if torch.all(tokenized_sampling.input_ids[:, 1:] == labels):
                 logits_ref = sampling_model(**tokenized_sampling).logits[:, :-1]
            else:
                 # Handle tokenizer mismatch if necessary, or skip sample
                 print(f"Warning: Tokenizer mismatch for text: {text[:100]}...")
                 return 0.0 # Or NaN

        # Ensure vocabulary size matches if necessary
        if logits_ref.size(-1) != logits_score.size(-1):
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        # Get samples from reference model (sampling model)
        lprobs_ref = torch.log_softmax(logits_ref, dim=-1)
        distrib = torch.distributions.categorical.Categorical(logits=lprobs_ref)
        samples = distrib.sample([nsamples]).permute([1, 2, 0])

        # Get likelihood of sampled texts under scoring model
        lprobs_score = torch.log_softmax(logits_score, dim=-1)
        # Expand lprobs_score to match samples shape for broadcasting
        lprobs_score_expanded = lprobs_score.unsqueeze(-1).expand(-1, -1, -1, nsamples)
        log_likelihood_x_tilde = lprobs_score_expanded.gather(dim=-2, index=samples)
        log_likelihood_x_tilde = log_likelihood_x_tilde.mean(dim=1) # Average over tokens

        miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
        sigma_tilde = log_likelihood_x_tilde.std(dim=-1)

        # Get likelihood of original text under scoring model
        log_likelihood_x = lprobs_score.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1).mean(dim=1)

        # Calculate discrepancy score
        # Add a small epsilon to sigma_tilde to prevent division by zero
        discrepancy = (log_likelihood_x - miu_tilde) / (sigma_tilde + 1e-8)

    return discrepancy.item()

def apply_fast_detect_gpt(dataframe, device, scoring_model_name='gpt2', sampling_model_name='gpt2'):
    """
    Applies Fast-DetectGPT to each text in the DataFrame.
    Returns the DataFrame with an added 'fastgpt_score' column.
    """
    print(f"Loading scoring model {scoring_model_name}...")
    scoring_model = GPT2LMHeadModel.from_pretrained(scoring_model_name).to(device)
    scoring_tokenizer = GPT2Tokenizer.from_pretrained(scoring_model_name)
    scoring_tokenizer.pad_token = scoring_tokenizer.eos_token # GPT2 doesn't have a default pad token

    sampling_model = scoring_model
    sampling_tokenizer = scoring_tokenizer
    if sampling_model_name != scoring_model_name:
        print(f"Loading sampling model {sampling_model_name}...")
        sampling_model = GPT2LMHeadModel.from_pretrained(sampling_model_name).to(device)
        sampling_tokenizer = GPT2Tokenizer.from_pretrained(sampling_model_name)
        sampling_tokenizer.pad_token = sampling_tokenizer.eos_token

    scores = []
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Applying Fast-DetectGPT"):
        text = row['text']
        score = get_sampling_discrepancy_score(
            text,
            scoring_model,
            scoring_tokenizer,
            sampling_model,
            sampling_tokenizer,
            device
        )
        scores.append(score)

    dataframe['fastgpt_score'] = scores
    return dataframe

# 2. Apply Fast-DetectGPT to df_ood_test and df_in_domain_test.

print("\nApplying Fast-DetectGPT to in-domain test set...")
df_in_domain_test_with_scores = apply_fast_detect_gpt(df_in_domain_test.copy(), device)

print("\nApplying Fast-DetectGPT to OOD test set...")
df_ood_test_with_scores = apply_fast_detect_gpt(df_ood_test.copy(), device)

# 3. Evaluate the Fast-DetectGPT on df_in_domain_test and df_ood_test.

print("\nEvaluating Fast-DetectGPT on in-domain test set...")
in_domain_fastgpt_true_labels = df_in_domain_test_with_scores['label'].tolist()
in_domain_fastgpt_scores = df_in_domain_test_with_scores['fastgpt_score'].tolist()

# Calculate AUROC
in_domain_fastgpt_auroc = roc_auc_score(in_domain_fastgpt_true_labels, in_domain_fastgpt_scores)
print(f"In-Domain Test AUROC (Fast-DetectGPT): {in_domain_fastgpt_auroc:.4f}")

# Calculate other metrics using a threshold (e.g., 0)
threshold = 0
in_domain_fastgpt_preds = [1 if score > threshold else 0 for score in in_domain_fastgpt_scores]
in_domain_fastgpt_metrics = calculate_metrics(in_domain_fastgpt_true_labels, in_domain_fastgpt_preds)

print("In-Domain Test Metrics (Fast-DetectGPT, Threshold=0):")
for metric, value in in_domain_fastgpt_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nEvaluating Fast-DetectGPT on OOD test set...")
ood_fastgpt_true_labels = df_ood_test_with_scores['label'].tolist()
ood_fastgpt_scores = df_ood_test_with_scores['fastgpt_score'].tolist()

# Calculate AUROC
ood_fastgpt_auroc = roc_auc_score(ood_fastgpt_true_labels, ood_fastgpt_scores)
print(f"OOD Test AUROC (Fast-DetectGPT): {ood_fastgpt_auroc:.4f}")

# Calculate other metrics using a threshold (e.g., 0)
ood_fastgpt_preds = [1 if score > threshold else 0 for score in ood_fastgpt_scores]
ood_fastgpt_metrics = calculate_metrics(ood_fastgpt_true_labels, ood_fastgpt_preds)

print("OOD Test Metrics (Fast-DetectGPT, Threshold=0):")
for metric, value in ood_fastgpt_metrics.items():
    print(f"{metric}: {value:.4f}")

# --- Evaluation and Comparison (Final Steps) ---
# 1. Calculate all required metrics for both methods (done).
# 2. Present and discuss the results.

print("\n--- Results Summary ---")
print("Supervised Model (In-Domain Test):")
for metric, value in in_domain_metrics.items():
    print(f"{metric}: {value:.4f}")
print(f"AUROC: N/A (Requires probabilities)") # Note on AUROC for supervised model

print("\nSupervised Model (OOD Test):")
for metric, value in ood_metrics.items():
    print(f"{metric}: {value:.4f}")
print(f"AUROC: N/A (Requires probabilities)") # Note on AUROC for supervised model

print("\nFast-DetectGPT (In-Domain Test):")
for metric, value in in_domain_fastgpt_metrics.items():
    print(f"{metric}: {value:.4f}")
print(f"AUROC: {in_domain_fastgpt_auroc:.4f}")

print("\nFast-DetectGPT (OOD Test):")
for metric, value in ood_fastgpt_metrics.items():
    print(f"{metric}: {value:.4f}")
print(f"AUROC: {ood_fastgpt_auroc:.4f}")

print("\n--- Discussion ---")
print("Compare the performance of the supervised model and Fast-DetectGPT, especially on the OOD test set. Discuss the strengths and weaknesses of each method based on the metrics. For example, comment on how well the supervised model generalizes compared to the zero-shot method.")
print("Consider adding evaluation on the combined test set (df_combined_test) for a single overall performance comparison if needed.")
print("You might also want to experiment with different scoring and sampling models for Fast-DetectGPT and different supervised model architectures or training parameters to improve results.")
print("Finally, consider implementing the analysis on the Chinese dataset as an optional experiment.") 