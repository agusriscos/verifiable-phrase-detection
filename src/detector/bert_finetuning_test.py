from datasets import load_dataset
import transformers


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


if __name__ == "__main__":
    raw_datasets = load_dataset("imdb")
    print("Trace 1")
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    print("Trace 2")
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    print("Trace 3")
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))
    print("Trace 4")
    model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
    print("Trace 5")
    training_args = transformers.TrainingArguments("../../data/bert-test-trainer")
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset
    )
    trainer.train()
