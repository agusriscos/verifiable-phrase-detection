from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

import torch
from torch.utils.data import Dataset
import transformers


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    df = pd.DataFrame(columns=["text", "label"])
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir / label_dir).iterdir():
            df = df.append({"text": text_file.read_text(), "label": 0 if label_dir == "neg" else 1}, ignore_index=True)
    return df


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


if __name__ == "__main__":
    # train_df = read_imdb_split('../../data/aclImdb/train')
    # test_df = read_imdb_split('../../data/aclImdb/test')
    # train_df.to_csv("../../data/aclImdb/imdb_train.csv", index=False)
    # test_df.to_csv("../../data/aclImdb/imdb_test.csv", index=False)

    train_df = pd.read_csv('../../data/aclImdb/imdb_train.csv')
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_df["text"].values.tolist(),
                                                                        train_df["label"].values.tolist(), test_size=.2)

    tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-cased")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)

    model = transformers.AutoModelForSequenceClassification.from_pretrained("distilbert-base-cased", num_labels=2)
    training_args = transformers.TrainingArguments(
        output_dir="../../data/imdb-bert-results",
        overwrite_output_dir=True,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_steps=100
    )
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
