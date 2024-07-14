
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import torch

# Load the dataset from the CSV file
dataset = load_dataset('csv', data_files='sentiment_dataset.csv')

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Label mapping to integers
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

# Function to encode the dataset
def encode(examples):
    # Tokenize the statements
    tokenized_inputs = tokenizer(examples['Statement'], padding='max_length', truncation=True, max_length=128)
    # Map string labels to integers
    tokenized_inputs['labels'] = [label_map[label] for label in examples['Label']]
    return tokenized_inputs

# Encode the dataset
dataset = dataset.map(encode, batched=True)

# Split the dataset into training and validation sets
dataset = dataset['train'].train_test_split(test_size=0.1)

# Load a pre-trained BERT model with a classification head
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map))

# Add label-to-id and id-to-label mappings to the model configuration
model.config.label2id = label_map
model.config.id2label = {id: label for label, id in label_map.items()}

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=1,
)

# Function to compute metrics
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(p):
    logits, labels = p
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# After training and evaluating, save the model and tokenizer
trainer.save_model("./my_finetuned_bert")
tokenizer.save_pretrained("./my_finetuned_bert")

# Additionally, explicitly save the model's state_dict
torch.save(model.state_dict(), "./my_finetuned_bert/pytorch_model.bin")
