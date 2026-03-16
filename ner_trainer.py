import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import evaluate

# -----------------------
# loading jsonl
# -----------------------
def load_jsonl(path):
    tokens = []
    labels = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            tokens.append(obj["tokens"])
            labels.append(obj["labels"])
    return Dataset.from_dict({"tokens": tokens, "labels": labels})

dataset = load_jsonl("data_clean.jsonl")

# -----------------------
# label mapping
# -----------------------
unique_labels = sorted({l for seq in dataset["labels"] for l in seq})
label2id = {l: i for i, l in enumerate(unique_labels)}
id2label = {i: l for l, i in label2id.items()}

def encode_labels(example):
    example["ner_tags"] = [label2id[l] for l in example["labels"]]
    return example

dataset = dataset.map(encode_labels)
dataset = dataset.train_test_split(test_size=0.1)

# -----------------------
# tokenizer
# -----------------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align(example):
    tokenized = tokenizer(
        example["tokens"],
        truncation=True,
        max_length=128, 
        padding="max_length",
        is_split_into_words=True
    )
    word_ids = tokenized.word_ids(0)
    labels = []
    previous = None
    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        elif word_id != previous:
            labels.append(example["ner_tags"][word_id])
        else:
            labels.append(example["ner_tags"][word_id])
        previous = word_id
    tokenized["labels"] = labels
    return tokenized

dataset = dataset.map(tokenize_and_align)

# -----------------------
# model
# -----------------------
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# -----------------------
# metrics
# -----------------------
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=2)
    true_predictions, true_labels = [], []
    for pred, lab in zip(predictions, labels):
        cur_preds, cur_labels = [], []
        for p, l in zip(pred, lab):
            if l != -100:
                cur_preds.append(id2label[p])
                cur_labels.append(id2label[l])
        true_predictions.append(cur_preds)
        true_labels.append(cur_labels)
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
    }

# -----------------------
# training args
# -----------------------
training_args = TrainingArguments(
    output_dir="ner_model",
    learning_rate=2e-5,
    per_device_train_batch_size=6, 
    per_device_eval_batch_size=6,
    num_train_epochs=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    dataloader_num_workers=0,
    dataloader_pin_memory=False
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# -----------------------
# train
# -----------------------
trainer.train()
trainer.save_model("product_ner_model_distilbert_6")