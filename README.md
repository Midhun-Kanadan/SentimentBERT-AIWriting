
# SentimentBERT-AIWriting

This repository contains the code and resources for `SentimentBERT-AIWriting`, a fine-tuned version of `bert-base-uncased` for sentiment classification, tailored for AI-assisted argumentative writing. It classifies text into three categories: positive, negative, and neutral.

## Model Description

`SentimentBERT-AIWriting` extends the original BERT (Bidirectional Encoder Representations from Transformers) capabilities to the task of sentiment classification. It was trained on a diverse dataset of statements collected from various domains to ensure robustness and accuracy across different contexts.

## Purpose

The `SentimentBERT-AIWriting` model is designed to assist in understanding the sentiment of texts. This can be particularly useful for platforms requiring an understanding of user sentiment, such as customer feedback analysis, social media monitoring, and enhancing AI writing tools.

## How to Use the Model

You can use this model with the Hugging Face `transformers` library. Below is an example code snippet:

```python
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('MidhunKanadan/SentimentBERT-AIWriting')
model = BertForSequenceClassification.from_pretrained('MidhunKanadan/SentimentBERT-AIWriting')

text = "Your text goes here"

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
outputs = model(**inputs)

logits = outputs.logits
predictions = logits.argmax(-1)
labels = ['negative', 'neutral', 'positive']
predicted_label = labels[predictions.item()]

print(f"Text: {text}
predicted_label: {predicted_label}
")
```

## Examples

Here are some example statements and their corresponding sentiment predictions by the `SentimentBERT-AIWriting` model:

**Positive**

* Statement: "Despite initial skepticism, the new employee's contributions have been remarkable!"
* Predicted Label: `positive`

**Negative**

* Statement: "Nuclear energy can be a very efficient power source, but at the same time, it poses significant risks."
* Predicted Label: `negative`

**Neutral**

* Statement: "The documentary provides an overview of the event."
* Predicted Label: `neutral`

These examples demonstrate how `SentimentBERT-AIWriting` can effectively classify the sentiment of various statements.

## Limitations and Bias

While `SentimentBERT-AIWriting` is trained on a diverse dataset, no model is immune from bias. The model's predictions might still be influenced by inherent biases in the training data. It's important to consider this when interpreting the model's output, especially for sensitive applications.

## Contributions and Feedback

We welcome contributions to this model! You can suggest improvements or report issues by opening an issue on this repository.

If you find this model useful for your projects or research, feel free to cite it and provide feedback on its performance.

## Finetuning Code

The code used for fine-tuning the `SentimentBERT-AIWriting` model can be found in the `finetuning_script.py` file.
